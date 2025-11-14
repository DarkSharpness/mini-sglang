#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

#include <cstdint>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr std::size_t kSmem = 24 * 1024 * sizeof(uint32_t);

struct FastTopKParams {
  const float* __restrict__ input;  // [B, input_size]
  int32_t* __restrict__ indices;    // [B, TopK]
  int32_t* __restrict__ lengths;    // [B]
  int64_t input_stride;
};

struct FastTopKTransformParams {
  FastTopKParams params;
  int32_t* __restrict__ dst_page_table;        // [B, TopK]
  const int32_t* __restrict__ src_page_table;  // [prefill_bs, src_size]
  int64_t src_stride;
  int32_t* __restrict__ cu_seqlens_q;  // [prefill_bs + 1]
  int64_t prefill_bs;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ void fast_topk_cuda(const float* __restrict__ input, int* __restrict__ index, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > TopK here, or it will crash
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = static_cast<int>(kSmem / (2 * sizeof(int)));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        [[likely]];
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        /// NOTE: (dark) fuse the histogram computation here
        if (pos < SMEM_INPUT_SIZE) {
          [[likely]];
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = static_cast<unsigned>(s_threshold_bin_id);
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < SMEM_INPUT_SIZE) {
              [[likely]];
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}
__device__ __forceinline__ void
fast_topk_dispatch(const float* __restrict__ input, int* __restrict__ index, int length) {
  if (length <= TopK) {
    naive_topk_cuda(input, index, length);
  } else {
    fast_topk_cuda(input, index, length);
  }
}

__device__ __forceinline__ void fast_topk_transform_dispatch(
    const float* __restrict__ input,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  if (length <= TopK) {
    naive_topk_transform(input, length, dst_page_table, src_page_table);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda(input, s_indices, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto tid = threadIdx.x;
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_table[idx_0] = src_page_table[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_table[idx_1] = src_page_table[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock, 2)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, indices, lengths, input_stride] = params;
  const auto bid = blockIdx.x;
  const auto length = lengths[bid];
  const auto indice = indices + bid * static_cast<int64_t>(TopK);
  const auto score = input + bid * input_stride;
  return fast_topk_dispatch(score, indice, length);
}

__global__ __launch_bounds__(kThreadsPerBlock, 2)  // decode
    void topk_transform_decode_kernel(const FastTopKTransformParams params) {
  const auto& [params_, dst_page_table, src_page_table, src_stride, cu_seqlens_q, prefill_bs] = params;
  const auto& [input, _, lengths, input_stride] = params_;
  const auto bid = blockIdx.x;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * static_cast<int64_t>(TopK);
  const auto score = input + bid * input_stride;
  return fast_topk_transform_dispatch(score, length, dst_page_entry, src_page_entry);
}

__global__ __launch_bounds__(kThreadsPerBlock, 2)  // prefill
    void topk_transform_prefill_kernel(const FastTopKTransformParams params) {
  const auto& [params_, dst_page_table, src_page_table, src_stride, cu_seqlens_q, prefill_bs] = params;
  const auto& [input, _, lengths, input_stride] = params_;
  const auto bid = blockIdx.x;
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto dst_page_entry = dst_page_table + bid * static_cast<int64_t>(TopK);
  const auto score = input + bid * input_stride;

  /// NOTE: We ensure that last cu_seqlens is equal to number of blocks launched
  __shared__ const int32_t* s_src_page_entry;
  if (prefill_bs <= kThreadsPerBlock) {
    if (tid < prefill_bs) {
      [[likely]];
      if (bid >= static_cast<uint32_t>(cu_seqlens_q[tid]) && bid < static_cast<uint32_t>(cu_seqlens_q[tid + 1])) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= static_cast<uint32_t>(cu_seqlens_q[i]) && bid < static_cast<uint32_t>(cu_seqlens_q[i + 1])) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  }
  __syncthreads();
  return fast_topk_transform_dispatch(score, length, dst_page_entry, s_src_page_entry);
}

auto fast_topk_interface(
    const tvm::ffi::TensorView score, const tvm::ffi::TensorView lengths, const tvm::ffi::TensorView indices) -> void {
  using namespace host;
  auto B = SymbolicSize{"B"};  // batch size
  auto S = SymbolicSize{"S"};  // score stride
  auto device_ = SymbolicDevice{};

  TensorMatcher({B, -1})  //
      .with_strides({S, 1})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(score);
  TensorMatcher({B})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(lengths);
  TensorMatcher({B, TopK})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(indices);

  const auto params = FastTopKParams{
      .input = static_cast<const float*>(score.data_ptr()),
      .indices = static_cast<int32_t*>(indices.data_ptr()),
      .lengths = static_cast<int32_t*>(lengths.data_ptr()),
      .input_stride = S.unwrap(),
  };
  set_smem_once<topk_kernel>(kSmem);
  LaunchKernel(B.unwrap(), kThreadsPerBlock, device_.unwrap(), kSmem)(topk_kernel, params);
}

auto fast_topk_transform_interface(
    const tvm::ffi::TensorView score,
    const tvm::ffi::TensorView lengths,
    const tvm::ffi::TensorView dst_page_table,
    const tvm::ffi::TensorView src_page_table,
    const tvm::ffi::TensorView cu_seqlens_q) -> void {
  using namespace host;
  auto B = SymbolicSize{"B"};  // batch size
  auto P = SymbolicSize{"P"};  // prefill batch size
  auto D = SymbolicSize{"D"};  // score size
  auto S = SymbolicSize{"S"};  // score stride
  auto T = SymbolicSize{"T"};  // src_page_table stride
  auto Q = SymbolicSize{"Q"};  // cu_seqlens_q length
  auto device_ = SymbolicDevice{};

  TensorMatcher({B, D})  //
      .with_strides({S, 1})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(score);
  TensorMatcher({B})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(lengths);
  TensorMatcher({B, TopK})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(dst_page_table);
  TensorMatcher({P, D})  //
      .with_strides({T, 1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src_page_table);
  TensorMatcher({Q})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(cu_seqlens_q);

  const auto num_tokens = B.unwrap();
  const auto prefill_bs = P.unwrap();
  RuntimeCheck(prefill_bs == Q.unwrap() - 1 && prefill_bs <= num_tokens);

  const auto device = device_.unwrap();
  const auto params = FastTopKTransformParams{
      .params =
          {
              .input = static_cast<const float*>(score.data_ptr()),
              .indices = nullptr,  // unused
              .lengths = static_cast<int32_t*>(lengths.data_ptr()),
              .input_stride = S.unwrap(),
          },
      .dst_page_table = static_cast<int32_t*>(dst_page_table.data_ptr()),
      .src_page_table = static_cast<const int32_t*>(src_page_table.data_ptr()),
      .src_stride = T.unwrap(),
      .cu_seqlens_q = static_cast<int32_t*>(cu_seqlens_q.data_ptr()),
      .prefill_bs = prefill_bs,
  };

  // dispatch to decode or prefill
  if (prefill_bs == num_tokens) {
    set_smem_once<topk_transform_decode_kernel>(kSmem);
    LaunchKernel(num_tokens, kThreadsPerBlock, device, kSmem)(topk_transform_decode_kernel, params);
  } else {
    set_smem_once<topk_transform_prefill_kernel>(kSmem);
    LaunchKernel(num_tokens, kThreadsPerBlock, device, kSmem)(topk_transform_prefill_kernel, params);
  }
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk, fast_topk_interface);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_transform, fast_topk_transform_interface);
