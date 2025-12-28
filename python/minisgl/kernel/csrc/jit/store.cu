#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

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
  using namespace device;

  constexpr auto kWarpPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);

  const auto &[k_cache, v_cache, indices, k, v, kv_cache_stride,
               kv_input_stride, length] = params;
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  PDL::wait<kUsePDL>();

  // each warp handles one element
  if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];
    const auto dst_k = pointer::offset(k_cache, pos * kv_cache_stride);
    const auto src_k = pointer::offset(k, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_k, src_k);
    const auto dst_v = pointer::offset(v_cache, pos * kv_cache_stride);
    const auto src_v = pointer::offset(v, warp_id * kv_input_stride);
    warp::copy<kElementSize>(dst_v, src_v);
  }

  PDL::launch<kUsePDL>();
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
    using namespace host;
    auto D = SymbolicSize{"D"}; // element size
    auto L = SymbolicSize{"L"}; // length
    auto X = SymbolicSize{"X"}; // stride kv cache
    auto Y = SymbolicSize{"Y"}; // stride kv input
    auto indices_dtype_ = SymbolicDType{};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, D}) //
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({L, D}) //
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(k)
        .verify(v);
    TensorMatcher({L}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);

    const auto dtype_size = dtype_bytes(dtype_.unwrap());
    RuntimeCheck(element_size == dtype_size * D.unwrap());

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
    const auto num_blocks = div_ceil(length, kWarpPerBlock);
    const auto kernel = use_int32
                            ? store_kv_cache<num_threads, max_concurrency,
                                             use_pdl, element_size, int32_t>
                            : store_kv_cache<num_threads, max_concurrency,
                                             use_pdl, element_size, int64_t>;
    LaunchKernel(num_blocks, num_threads, device)
        .with_attr(use_pdl)(kernel, params);
  }
};

struct StoreMLAKernelParams {
  void *__restrict__ kv_buffer;
  const void *__restrict__ indices;
  const void *__restrict__ kv_c;
  const void *__restrict__ k_rope;
  std::size_t buffer_stride;
  std::size_t kv_c_input_stride;
  std::size_t k_rope_input_stride;
  std::size_t length;
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kKvCSize, std::size_t kKRopeSize, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    store_mla_cache_kernel(
        const __grid_constant__ StoreMLAKernelParams params) {
  using namespace device;

  constexpr auto kWarpPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);

  const auto &[kv_buffer, indices, kv_c, k_rope, buffer_stride,
               kv_c_input_stride, k_rope_input_stride, length] = params;

  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  PDL::wait<kUsePDL>();

  if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];

    // Store kv_c (Latent) at offset 0
    const auto dst_c = pointer::offset(kv_buffer, pos * buffer_stride);
    const auto src_c = pointer::offset(kv_c, warp_id * kv_c_input_stride);
    warp::copy<kKvCSize>(dst_c, src_c);

    // Store k_rope (RoPE) at offset kKvCSize
    const auto dst_r =
        pointer::offset(kv_buffer, pos * buffer_stride + kKvCSize);
    const auto src_r = pointer::offset(k_rope, warp_id * k_rope_input_stride);
    warp::copy<kKRopeSize>(dst_r, src_r);
  }

  PDL::launch<kUsePDL>();
}

template <std::size_t kv_c_size,   // Size in bytes of compressed latent vector
          std::size_t k_rope_size, // Size in bytes of rope part
          std::size_t num_threads = 128, std::size_t max_concurrency = 1,
          bool use_pdl = false>
struct StoreMLAKernel {
  static void run(const tvm::ffi::TensorView kv_buffer,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView kv_c,
                  const tvm::ffi::TensorView k_rope) {
    using namespace host;
    auto D_c = SymbolicSize{"Dc"};     // kv_c element size
    auto D_r = SymbolicSize{"Dr"};     // k_rope element size
    auto D_total = SymbolicSize{"Dt"}; // total element size
    auto L = SymbolicSize{"L"};        // length
    auto X = SymbolicSize{"X"};        // stride kv_buffer
    auto Y_c = SymbolicSize{"Yc"};     // stride kv_c input
    auto Y_r = SymbolicSize{"Yr"};     // stride k_rope input

    auto indices_dtype_ = SymbolicDType{};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    // Verify kv_buffer
    TensorMatcher({-1, D_total})
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(kv_buffer);

    // Verify inputs
    TensorMatcher({L, D_c})
        .with_strides({Y_c, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(kv_c);
    TensorMatcher({L, D_r})
        .with_strides({Y_r, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(k_rope);

    TensorMatcher({L})
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);

    const auto dtype_size = dtype_bytes(dtype_.unwrap());
    RuntimeCheck(kv_c_size == dtype_size * D_c.unwrap());
    RuntimeCheck(k_rope_size == dtype_size * D_r.unwrap());
    // Ensure the buffer is wide enough
    RuntimeCheck(D_total.unwrap() * dtype_size >= kv_c_size + k_rope_size);

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto length = static_cast<std::size_t>(L.unwrap());

    const auto params = StoreMLAKernelParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .indices = indices.data_ptr(),
        .kv_c = kv_c.data_ptr(),
        .k_rope = k_rope.data_ptr(),
        .buffer_stride = X.unwrap() * dtype_size,
        .kv_c_input_stride = Y_c.unwrap() * dtype_size,
        .k_rope_input_stride = Y_r.unwrap() * dtype_size,
        .length = length,
    };

    constexpr auto kWarpPerBlock = num_threads / 32;
    const auto num_blocks = div_ceil(length, kWarpPerBlock);
    const auto kernel =
        use_int32
            ? store_mla_cache_kernel<num_threads, max_concurrency, use_pdl,
                                     kv_c_size, k_rope_size, int32_t>
            : store_mla_cache_kernel<num_threads, max_concurrency, use_pdl,
                                     kv_c_size, k_rope_size, int64_t>;
    LaunchKernel(num_blocks, num_threads, device)
        .with_attr(use_pdl)(kernel, params);
  }
};

} // namespace
