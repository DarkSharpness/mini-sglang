#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct HicacheKernelParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const void* __restrict__ indices_dst;
  void* __restrict__ k_cache_src;
  void* __restrict__ v_cache_src;
  const void* __restrict__ indices_src;
  std::size_t length;
  std::size_t kv_cache_src_stride;
  std::size_t kv_cache_dst_stride;
};

template <
    std::integral T,
    std::size_t kElementSize,
    std::size_t kBlockQuota,
    std::size_t kNumThreads,
    std::size_t kMaxOccupancy,
    bool kUsePDL>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void hicache_transfer(
    const __grid_constant__ HicacheKernelParams params) {
  // each warp acts as a worker
  using namespace device;
  static_assert(kNumThreads % kWarpThreads == 0);
  constexpr auto kWarpsPerBlock = static_cast<uint32_t>(kNumThreads) / kWarpThreads;
  constexpr auto kWorkers = kWarpsPerBlock * kBlockQuota;

  const auto& [
    k_cache_dst, v_cache_dst, indices_dst, // dst
    k_cache_src, v_cache_src, indices_src, // src
    length, kv_cache_src_stride, kv_cache_dst_stride // metadata
  ] = params;
  const auto warp_id = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;

  // force to transfer 128 bytes per iteration
  // since the PCIe transaction size is 128 bytes aligned
  constexpr auto kGranularity = 256 / kWarpThreads;

  PDL::wait<kUsePDL>();

  for (auto i = warp_id; i < length; i += kWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
    warp::copy<kElementSize, kGranularity>(dst_k, src_k);
    const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
    warp::copy<kElementSize, kGranularity>(dst_v, src_v);
  }

  PDL::launch<kUsePDL>();
}

template <
    std::size_t kElementSize,
    std::size_t kBlockQuota,
    std::size_t kNumThreads,
    std::size_t kMaxOccupancy,
    bool kUsePDL>
struct HiCacheKernel {
  static void
  run(const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_cache_src,
      const tvm::ffi::TensorView v_cache_src,
      const tvm::ffi::TensorView indices_src) {
    using namespace host;

    auto D = SymbolicSize{"D"};  // cache dimension
    auto N = SymbolicSize{"N"};  // src kv stride
    auto M = SymbolicSize{"M"};  // dst kv stride
    auto L = SymbolicSize{"L"};  // indices length
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    TensorMatcher({-1, D})  //
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_src)
        .verify(v_cache_src);
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_dst)
        .verify(v_cache_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto element_bytes = D.unwrap() * dtype_bytes(cache_dtype.unwrap());
    RuntimeCheck(kElementSize == element_bytes, "HicacheKernel: cache dimension mismatch.");

    const auto k_cache_dst_ptr = k_cache_dst.data_ptr();
    const auto v_cache_dst_ptr = v_cache_dst.data_ptr();
    const auto k_cache_src_ptr = k_cache_src.data_ptr();
    const auto v_cache_src_ptr = v_cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto kv_cache_src_stride = static_cast<std::size_t>(N.unwrap());
    const auto kv_cache_dst_stride = static_cast<std::size_t>(M.unwrap());
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();

    constexpr auto kWarpsPerBlock = kNumThreads / device::kWarpThreads;
    const auto num_blocks = std::min(div_ceil(length, kWarpsPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .length = length,
        .kv_cache_src_stride = kv_cache_src_stride,
        .kv_cache_dst_stride = kv_cache_dst_stride,
    };
    const auto kernel = use_int32
                            ? hicache_transfer<int32_t, kElementSize, kBlockQuota, kNumThreads, kMaxOccupancy, kUsePDL>
                            : hicache_transfer<int64_t, kElementSize, kBlockQuota, kNumThreads, kMaxOccupancy, kUsePDL>;
    LaunchKernel(num_blocks, kNumThreads, device).with_attr(kUsePDL)(kernel, params);
  }
};

}  // namespace
