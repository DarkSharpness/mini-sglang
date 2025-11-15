#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda/ptx>

namespace {

struct HiCacheTMAKernelParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const int32_t* __restrict__ indices_dst;
  void* __restrict__ k_cache_src;
  void* __restrict__ v_cache_src;
  const int32_t* __restrict__ indices_src;
  std::size_t length;
  std::size_t kv_cache_src_stride;
  std::size_t kv_cache_dst_stride;
};

namespace ptx = cuda::ptx;

[[maybe_unused]]
_CCCL_DEVICE void cp_async_bulk_gs  //
    (void* __dstMem, const void* __srcMem, const uint32_t& __size) {
  using ptx::__as_ptr_gmem, ptx::__as_ptr_smem;
  asm("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
}

[[maybe_unused]]
_CCCL_DEVICE inline void cp_async_bulk_sg  //
    (void* __dstMem, const void* __srcMem, const uint32_t& __size, uint64_t* __smem_bar) {
  using ptx::__as_ptr_gmem, ptx::__as_ptr_smem;
  // uint64_t policy;
  // asm("createpolicy.fractional.L2::evict_last.b64 cache_policy %0, 1.0;" : "=l"(policy));
  asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
      // ".L2::cache_hint"
      " [%0], [%1], %2, [%3]"
      // ", %4"
      ";"
      :
      : "r"(__as_ptr_smem(__dstMem)), "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "r"(__as_ptr_smem(__smem_bar))
      // , "l"(policy)
      : "memory");
}

[[maybe_unused]]
_CCCL_DEVICE inline auto load_persistent(const int32_t* src) -> int32_t {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  int32_t tmp;
  asm volatile("ld.global.L2::cache_hint.b32 %0, [%1], %2;" : "=r"(tmp) : "l"(src), "l"(policy) : "memory");
  return tmp;
}

constexpr auto kNumBuf = 4;
constexpr auto kPrefetch = 4;

template <
    std::size_t kElementSize,
    std::size_t kBlockQuota,
    std::size_t kNumThreads,
    std::size_t kMaxOccupancy,
    bool kUsePDL>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void hicache_transfer_tma(
    const __grid_constant__ HiCacheTMAKernelParams params) {
  // each warp acts as a worker
  using namespace device;
  static_assert(kNumThreads % kWarpThreads == 0);
  constexpr auto kWarpsPerBlock = static_cast<uint32_t>(kNumThreads) / kWarpThreads;
  constexpr auto kWorkers = kWarpsPerBlock * kBlockQuota;

  alignas(128) extern __shared__ char _s_buffer[][kNumBuf][2][kElementSize];
  __shared__ uint64_t _m_barrier[kWarpsPerBlock][kNumBuf];

  const auto& [
    k_cache_dst, v_cache_dst, indices_dst, // dst
    k_cache_src, v_cache_src, indices_src, // src
    length, kv_cache_src_stride, kv_cache_dst_stride // metadata
  ] = params;
  const auto local_warp_id = threadIdx.x / kWarpThreads;
  const auto warp_id = blockIdx.x * kWarpsPerBlock + local_warp_id;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto m_barrier = _m_barrier[local_warp_id];
  const auto s_buffer = _s_buffer[local_warp_id];
  if (lane_id < kNumBuf) {
    // a barrier that will never be used
    ptx::mbarrier_init(m_barrier + lane_id, 1024);
  }
  __syncwarp();

  PDL::wait<kUsePDL>();

  union {
    int4 indices_cache;
    int32_t indices_array[kPrefetch];
  };

  int32_t prev_dst[kNumBuf];
#pragma unroll kNumBuf
  for (auto i = 0; i < kNumBuf; ++i) {
    prev_dst[i] = -1;
  }

  static_assert(kPrefetch % kNumBuf == 0);
  if (const auto start = warp_id * kPrefetch; lane_id < 2 && start < length) {
    const auto mask = __activemask();
    const auto indices_ptr = lane_id == 0 ? indices_dst : indices_src;
    const auto dst_cache = lane_id == 0 ? k_cache_dst : v_cache_dst;
    const auto src_cache = lane_id == 0 ? k_cache_src : v_cache_src;
    auto i = start;
    do {
      indices_cache = *reinterpret_cast<const int4*>(&indices_ptr[i]);

#pragma unroll kPrefetch
      for (auto k = 0; k < kPrefetch; ++k) {
        const auto _pos = indices_array[k];
        const auto _id = k % kNumBuf;
        const auto pos_src = __shfl_sync(mask, _pos, 1);  // lane 1: src
        const auto bar = &m_barrier[_id];
        const auto buf = &s_buffer[_id][lane_id];
        if (const auto pos_dst = prev_dst[_id]; pos_dst >= 0) {
          const auto dst = pointer::offset(dst_cache, pos_dst * kv_cache_dst_stride);
          cp_async_bulk_gs(dst, buf, kElementSize);
        }
        const auto src = pointer::offset(src_cache, pos_src * kv_cache_src_stride);
        cp_async_bulk_sg(buf, src, kElementSize, bar);
        prev_dst[_id] = __shfl_sync(mask, _pos, 0);  // lane 0: dst;
      }
      i += kPrefetch * kWorkers;
    } while (i < length);

    // epilogue: flush remaining
#pragma unroll kNumBuf
    for (auto k = 0; k < kNumBuf; ++k) {
      if (const auto pos_dst = prev_dst[k]; pos_dst >= 0) {
        const auto dst = pointer::offset(dst_cache, pos_dst * kv_cache_dst_stride);
        cp_async_bulk_gs(dst, &s_buffer[k][lane_id], kElementSize);
      }
    }
  }

  PDL::launch<kUsePDL>();
}

template <
    std::size_t kElementSize,
    std::size_t kBlockQuota,
    std::size_t kNumThreads,
    std::size_t kMaxOccupancy,
    bool kUsePDL>
struct HiCacheTMAKernel {
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
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, "HiCacheTMAKernel: cache dimension mismatch.");

    const auto k_cache_dst_ptr = k_cache_dst.data_ptr();
    const auto v_cache_dst_ptr = v_cache_dst.data_ptr();
    const auto k_cache_src_ptr = k_cache_src.data_ptr();
    const auto v_cache_src_ptr = v_cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto kv_cache_src_stride = static_cast<std::size_t>(N.unwrap()) * dtype_size;
    const auto kv_cache_dst_stride = static_cast<std::size_t>(M.unwrap()) * dtype_size;
    const auto device = indices_device.unwrap();

    RuntimeCheck(length % kPrefetch == 0, "HiCacheTMAKernel: length must be multiple of ", kPrefetch);
    constexpr auto kWarpsPerBlock = kNumThreads / device::kWarpThreads;
    const auto num_blocks = std::min(div_ceil(length, (kWarpsPerBlock * kPrefetch)), kBlockQuota);
    const auto params = HiCacheTMAKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = static_cast<const int32_t*>(indices_dst_ptr),
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = static_cast<const int32_t*>(indices_src_ptr),
        .length = length,
        .kv_cache_src_stride = kv_cache_src_stride,
        .kv_cache_dst_stride = kv_cache_dst_stride,
    };
    constexpr auto kSmem = kNumBuf * 2 * kElementSize * kWarpsPerBlock + 128;
    constexpr auto kernel = hicache_transfer_tma<kElementSize, kBlockQuota, kNumThreads, kMaxOccupancy, kUsePDL>;
    set_smem_once<kernel>(kSmem);
    LaunchKernel(num_blocks, kNumThreads, device, kSmem).with_attr(kUsePDL)(kernel, params);
  }
};

}  // namespace
