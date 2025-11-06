#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>

#include <bit>
#include <concepts>
#include <cstddef>

namespace {

struct IndexKernelParams {
  void *__restrict__ output;
  const void *__restrict__ weight;
  const void *__restrict__ indice;
  std::size_t num_warps;
};

struct MaskedKernelParams {
  IndexKernelParams params;
  std::size_t start;
  std::size_t length;
};

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    index_kernel(const __grid_constant__ IndexKernelParams params) {
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  static_assert(kNumThreads % 32 == 0);
  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  const auto warp_id = (threadIdx.x / 32u) + blockIdx.x * kWarpPerBlock;
  cuda::PDL::wait<kUsePDL>();

  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits];
    const auto dst = cuda::pointer::offset(output, warp_id * kSizePerWarp);
    const auto src = cuda::pointer::offset(
        weight, pos * kSize, (warp_id % kNumSplits) * kSizePerWarp);
    cuda::warp::copy<kSizePerWarp>(dst, src);
  }

  cuda::PDL::launch<kUsePDL>();
}

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    masked_index_kernel(
        const __grid_constant__ MaskedKernelParams mask_params) {
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  static_assert(kNumThreads % 32 == 0);
  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[params, start, length] = mask_params;
  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  const auto warp_id = (threadIdx.x / 32u) + blockIdx.x * kWarpPerBlock;

  cuda::PDL::wait<kUsePDL>();

  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits] - start;
    const auto dst = cuda::pointer::offset(output, warp_id * kSizePerWarp);
    if (pos < length) {
      const auto src = cuda::pointer::offset(
          weight, pos * kSize, (warp_id % kNumSplits) * kSizePerWarp);
      cuda::warp::copy<kSizePerWarp>(dst, src);
    } else {
      // memset the warp to zero, using uint4 package
      cuda::warp::reset<kSizePerWarp>(dst);
    }
  }

  cuda::PDL::launch<kUsePDL>();
}

template <std::size_t element_size,   // depends on data type and embedding dim
          std::size_t num_splits = 1, // how many warps handles one element
          std::size_t num_threads = 128,   // number of threads per block
          std::size_t max_concurrency = 1, // max blocks per SM
          bool use_pdl = false>
struct IndexKernel {
  static void run(const tvm::ffi::TensorView weights, //
                  const tvm::ffi::TensorView indices, //
                  const tvm::ffi::TensorView output,  //
                  tvm::ffi::Optional<tvm::ffi::Tuple<int, int>> mask_opts) {
    const auto num_indices = indices.size(0);
    const auto embed_size = weights.size(1);
    const auto weights_dtype = weights.dtype();
    const auto indices_dtype = indices.dtype();
    const auto dtype_size = weights_dtype.bits / 8;

    // memory layout checks
    host::RuntimeCheck(weights.is_contiguous() && indices.is_contiguous() &&
                           output.is_contiguous(),
                       "All tensors must be contiguous.");
    host::RuntimeCheck(weights.device().device_type == kDLCUDA &&
                           indices.device().device_type == kDLCUDA &&
                           output.device().device_type == kDLCUDA,
                       "All tensors must be on CUDA device.");

    // dtype checks
    host::RuntimeCheck(weights.dtype() == output.dtype(),
                       "Weights and output must have the same dtype.");
    host::RuntimeCheck(indices_dtype.code == kDLInt,
                       "Indices must be of integer type.");
    host::RuntimeCheck(indices_dtype.bits == 32 || indices_dtype.bits == 64,
                       "Indices must be of 32 or 64 bits.");

    // shape checks
    host::RuntimeCheck(weights.ndim() == 2 && indices.ndim() == 1 &&
                           output.ndim() == 2,
                       "Weights must be 2-D, indices must be 1-D, "
                       "and output must be 2-D tensor.");
    host::RuntimeCheck(weights.size(1) == output.size(1),
                       "Weights and output must have the same embedding size.");
    host::RuntimeCheck(indices.size(0) == output.size(0),
                       "Indices and output must have the same number of rows.");
    host::RuntimeCheck(embed_size * dtype_size == element_size, //
                       "Element size ", element_size,
                       " does not match embedding size ", embed_size,
                       " * dtype size ", dtype_size, ".");

    constexpr auto kWarpPerBlock = num_threads / 32;
    const auto num_warps = num_splits * num_indices;
    const auto num_blocks = ::math::div_ceil(num_warps, kWarpPerBlock);
    const auto params = IndexKernelParams{
        .output = static_cast<char *>(output.data_ptr()),
        .weight = static_cast<const char *>(weights.data_ptr()),
        .indice = indices.data_ptr(),
        .num_warps = num_warps,
    };

    const auto device = weights.device();
    const auto stream = static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));

    if (mask_opts.has_value()) {
      const auto &obj = mask_opts.value();
      const auto start = obj.get<0>();
      const auto length = obj.get<1>();
      const auto m_params = MaskedKernelParams{
          .params = params,
          .start = static_cast<std::size_t>(start),
          .length = static_cast<std::size_t>(length),
      };
      const auto kernel =
          indices_dtype.bits == 32
              ? masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                    element_size, num_splits, std::int32_t>
              : masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                    element_size, num_splits, std::int64_t>;
      host::LaunchKernel(num_blocks, num_threads, stream)
          .set_pdl(use_pdl)(kernel, m_params);
    } else {
      const auto kernel =
          indices_dtype.bits == 32
              ? index_kernel<num_threads, max_concurrency, use_pdl,
                             element_size, num_splits, std::int32_t>
              : index_kernel<num_threads, max_concurrency, use_pdl,
                             element_size, num_splits, std::int64_t>;
      host::LaunchKernel(num_blocks, num_threads, stream)
          .set_pdl(use_pdl)(kernel, params);
    }
  }
};

} // namespace
