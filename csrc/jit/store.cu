#include <algorithm>
#include <dlpack/dlpack.h>
#include <minisgl/utils.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

#include <minisgl/utils.cuh>
#include <minisgl/warp.cuh>

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

using std::size_t;
using std::uint64_t;

struct StoreKernelParams {
  void *__restrict__ k_cache;
  void *__restrict__ v_cache;
  const void *__restrict__ indices;
  const void *__restrict__ k;
  const void *__restrict__ v;
  std::size_t kv_cache_stride;
  std::size_t kv_input_stride;
  std::size_t length;
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    store_kv_cache(const __grid_constant__ StoreKernelParams params) {
  constexpr auto kSize = kElementSize;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);
  static_assert(kNumThreads % 32 == 0);

  const auto &[k_cache, v_cache, indices, k, v, kv_cache_stride,
               kv_input_stride, length] = params;
  const auto warp_id = (threadIdx.x / 32u) + blockIdx.x * kWarpPerBlock;
  cuda::PDL::wait<kUsePDL>();

  // each warp handles one element
  if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];
    const auto dst_k = cuda::pointer::offset(k_cache, pos * kv_cache_stride);
    const auto src_k = cuda::pointer::offset(k, warp_id * kv_input_stride);
    cuda::warp::copy<kSize>(dst_k, src_k);
    const auto dst_v = cuda::pointer::offset(v_cache, pos * kv_cache_stride);
    const auto src_v = cuda::pointer::offset(v, warp_id * kv_input_stride);
    cuda::warp::copy<kSize>(dst_v, src_v);
  }

  cuda::PDL::launch<kUsePDL>();
}

template <std::size_t element_size, // depends on data type and embedding dim
          std::size_t num_threads = 128,   // number of threads per block
          std::size_t max_concurrency = 1, // max blocks per SM
          bool use_pdl = false>
struct StoreKernel {
  static void run(const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView k, const tvm::ffi::TensorView v) {
    host::RuntimeCheck(k_cache.ndim() == 2 && k_cache.stride(1) == 1 &&
                       k_cache.device().device_type == kDLCUDA);
    host::RuntimeCheck(v_cache.ndim() == 2 && v_cache.stride(1) == 1 &&
                       v_cache.device().device_type == kDLCUDA);
    host::RuntimeCheck(indices.ndim() == 1 && indices.stride(0) == 1 &&
                       indices.device().device_type == kDLCUDA);
    host::RuntimeCheck(k.ndim() == 2 && k.stride(1) == 1 &&
                       k.device().device_type == kDLCUDA);
    host::RuntimeCheck(v.ndim() == 2 && v.stride(1) == 1 &&
                       v.device().device_type == kDLCUDA);
    host::RuntimeCheck(k_cache.stride(0) == v_cache.stride(0) &&
                       std::ranges::equal(k_cache.sizes(), v_cache.sizes()) &&
                       k_cache.dtype() == v_cache.dtype());
    host::RuntimeCheck(k.stride(0) == v.stride(0) &&
                       std::ranges::equal(k.sizes(), v.sizes()) &&
                       k.dtype() == v.dtype());
    host::RuntimeCheck(k.size(0) == indices.size(0) &&
                       k.size(1) == k_cache.size(1) &&
                       k.dtype() == k_cache.dtype());

    const auto indices_dtype = indices.dtype();
    const auto dtype_size = static_cast<std::size_t>(k.dtype().bits / 8);
    host::RuntimeCheck(indices_dtype.code == kDLInt &&
                       (indices_dtype.bits == 32 || indices_dtype.bits == 64));
    host::RuntimeCheck(element_size == dtype_size * k.size(1));

    const auto length = static_cast<std::size_t>(indices.size(0));
    const auto kv_cache_stride = k_cache.stride(0) * dtype_size;
    const auto kv_input_stride = k.stride(0) * dtype_size;

    constexpr auto kWarpPerBlock = num_threads / 32;
    const auto num_blocks = math::div_ceil(length, kWarpPerBlock);
    const auto params = StoreKernelParams{
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .kv_cache_stride = kv_cache_stride,
        .kv_input_stride = kv_input_stride,
        .length = length,
    };

    const auto device = k.device();
    const auto stream = static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
    const auto kernel =
        indices_dtype.bits == 32
            ? store_kv_cache<num_threads, max_concurrency, use_pdl,
                             element_size, std::int32_t>
            : store_kv_cache<num_threads, max_concurrency, use_pdl,
                             element_size, std::int64_t>;
    host::LaunchKernel(num_blocks, num_threads, stream)
        .set_pdl(use_pdl)(kernel, params);
  }
};

} // namespace
