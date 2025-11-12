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

using std::size_t;
using std::uint64_t;

struct HicacheKernelParams {
  void *__restrict__ k_cache_dst;
  void *__restrict__ v_cache_dst;
  const void *__restrict__ indices_dst;
  void *__restrict__ k_cache_src;
  void *__restrict__ v_cache_src;
  const void *__restrict__ indices_src;
  std::size_t length;
  std::size_t kv_cache_src_stride;
  std::size_t kv_cache_dst_stride;
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kMaxBlocks, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    hicache_transfer(const __grid_constant__ HicacheKernelParams params) {
  // each warp as a worker
  constexpr auto kWarpPerBlock = static_cast<uint32_t>(kNumThreads) / 32;
  constexpr auto kWorkers = kWarpPerBlock * kMaxBlocks;
  static_assert(kNumThreads % 32 == 0);

  const auto &[k_cache_dst, v_cache_dst, indices_dst, k_cache_src, v_cache_src,
               indices_src, length, kv_cache_src_stride, kv_cache_dst_stride] =
      params;
  const auto warp_id = blockIdx.x * kWarpPerBlock + threadIdx.x / 32;
  cuda::PDL::wait<kUsePDL>();

  // 128 bytes per iteration
  constexpr auto kGranularity = 4;

  for (auto i = warp_id; i < length; i += kWorkers) {
    const auto pos_src = static_cast<const T *>(indices_src)[i];
    const auto pos_dst = static_cast<const T *>(indices_dst)[i];
    const auto src_k =
        cuda::pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_k =
        cuda::pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
    cuda::warp::copy<kElementSize, kGranularity>(dst_k, src_k);
    const auto src_v =
        cuda::pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_v =
        cuda::pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
    cuda::warp::copy<kElementSize, kGranularity>(dst_v, src_v);
  }

  cuda::PDL::launch<kUsePDL>();
}

template <std::size_t element_size,    // depends on data type and embedding dim
          std::size_t block_quota = 4, // how many blocks to use at most
          std::size_t num_threads = 128,   // number of threads per block
          std::size_t max_concurrency = 1, // max blocks per SM
          bool use_pdl = false>
struct HicacheKernel {
  static void run(const tvm::ffi::TensorView k_cache_dst,
                  const tvm::ffi::TensorView v_cache_dst,
                  const tvm::ffi::TensorView indices_dst,
                  const tvm::ffi::TensorView k_cache_src,
                  const tvm::ffi::TensorView v_cache_src,
                  const tvm::ffi::TensorView indices_src,
                  const std::size_t split_limit) {
    auto D = host::SymbolicSize{"D"};   // cache dimension
    auto ND = host::SymbolicSize{"ND"}; // src kv stride
    auto MD = host::SymbolicSize{"MD"}; // dst kv stride
    auto L = host::SymbolicSize{"L"};   // indices length
    auto cache_dtype = host::SymbolicDType{};
    auto indices_dtype = host::SymbolicDType{};
    auto indices_device = host::SymbolicDevice{};

    host::TensorMatcher({-1, D})
        .with_strides({ND, 1}) // last dim contiguous
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_src)
        .verify(v_cache_src);
    host::TensorMatcher({-1, D})
        .with_strides({MD, 1}) // last dim contiguous
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .with_dtype(cache_dtype)
        .verify(k_cache_dst)
        .verify(v_cache_dst);
    host::TensorMatcher({L}) // assume contiguous
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto entry_size = D.unwrap() * (cache_dtype.unwrap().bits / 8);
    host::RuntimeCheck(element_size == entry_size,
                       "HicacheKernel: cache dimension mismatch.");

    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto kv_cache_src_stride = static_cast<std::size_t>(ND.unwrap());
    const auto kv_cache_dst_stride = static_cast<std::size_t>(MD.unwrap());
    const auto use_int32 = indices_dtype.unwrap().bits == 32;

    constexpr auto kWarpsPerBlock = num_threads / 32;
    constexpr auto kMaxSplit = std::size_t{1} << 30;
    const auto step = std::min(split_limit, kMaxSplit);

    const auto k_cache_dst_ptr = k_cache_dst.data_ptr();
    const auto v_cache_dst_ptr = v_cache_dst.data_ptr();
    const auto k_cache_src_ptr = k_cache_src.data_ptr();
    const auto v_cache_src_ptr = v_cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto stream =
        host::LaunchKernel::resolve_device(indices_device.unwrap());

    bool can_continue = true;
    for (std::size_t i = 0; can_continue; i += step) {
      can_continue = (i + step * 2 < length);
      const auto current_length = can_continue ? step : (length - i);
      const auto num_blocks =
          std::min(math::div_ceil(current_length, kWarpsPerBlock), block_quota);
      const auto params = HicacheKernelParams{
          .k_cache_dst = k_cache_dst_ptr,
          .v_cache_dst = v_cache_dst_ptr,
          .indices_dst = host::pointer::offset(indices_dst_ptr, i),
          .k_cache_src = k_cache_src_ptr,
          .v_cache_src = v_cache_src_ptr,
          .indices_src = host::pointer::offset(indices_src_ptr, i),
          .length = current_length,
          .kv_cache_src_stride = kv_cache_src_stride,
          .kv_cache_dst_stride = kv_cache_dst_stride,
      };
      const auto kernel =
          use_int32 ? hicache_transfer<num_threads, max_concurrency, use_pdl,
                                       element_size, block_quota, std::int32_t>
                    : hicache_transfer<num_threads, max_concurrency, use_pdl,
                                       element_size, block_quota, std::int64_t>;
      host::LaunchKernel(num_blocks, num_threads, stream)
          .set_pdl(use_pdl)(kernel, params);
    }
  }
};

} // namespace
