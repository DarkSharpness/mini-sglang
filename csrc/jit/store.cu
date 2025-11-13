#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>

#include <tvm/ffi/container/tensor.h>

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
    cuda::warp::copy<kElementSize>(dst_k, src_k);
    const auto dst_v = cuda::pointer::offset(v_cache, pos * kv_cache_stride);
    const auto src_v = cuda::pointer::offset(v, warp_id * kv_input_stride);
    cuda::warp::copy<kElementSize>(dst_v, src_v);
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
    auto D = host::SymbolicSize{"D"}; // element size
    auto L = host::SymbolicSize{"L"}; // length
    auto X = host::SymbolicSize{"X"}; // stride kv cache
    auto Y = host::SymbolicSize{"Y"}; // stride kv input
    auto indices_dtype_ = host::SymbolicDType{};
    auto dtype_ = host::SymbolicDType{};
    auto device_ = host::SymbolicDevice{};

    host::TensorMatcher({-1, D})
        .with_strides({X, 1}) // last dim contiguous
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(k_cache)
        .verify(v_cache);

    host::TensorMatcher({L, D})
        .with_strides({Y, 1}) // last dim contiguous
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(k)
        .verify(v);

    host::TensorMatcher({L})
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);

    const auto dtype_size = static_cast<std::size_t>(dtype_.unwrap().bits) / 8;
    host::RuntimeCheck(element_size == dtype_size * D.unwrap());

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto kv_cache_stride = X.unwrap() * dtype_size;
    const auto kv_input_stride = Y.unwrap() * dtype_size;

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

    constexpr auto kWarpPerBlock = num_threads / 32;
    static_assert(num_threads % 32 == 0);
    const auto num_blocks = math::div_ceil(length, kWarpPerBlock);

    const auto kernel =
        use_int32 ? store_kv_cache<num_threads, max_concurrency, use_pdl,
                                   element_size, std::int32_t>
                  : store_kv_cache<num_threads, max_concurrency, use_pdl,
                                   element_size, std::int64_t>;
    host::LaunchKernel(num_blocks, num_threads, device)
        .set_pdl(use_pdl)(kernel, params);
  }
};

} // namespace
