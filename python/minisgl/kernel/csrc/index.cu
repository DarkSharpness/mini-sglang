#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/stat.h>
#include <torch/python.h>
#include <utility>

namespace {

using std::int64_t;
using std::size_t;

struct index_range {
  int32_t start;
  int32_t length;
};

__global__ void __launch_bounds__(256) indexing_kernel_1_fold(
    uint4 *__restrict__ output, const uint4 *__restrict__ input,
    const int32_t *__restrict__ index_ptr,
    const size_t warp_count,      // how many warps for each item
    const size_t index_length,    // how many index to process
    const index_range index_range // valid index range
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto warp_id = idx / 32; // warp index in the grid
  const auto lane_id = idx % 32; // lane index in the warp
  if (warp_id >= index_length * warp_count)
    return;

  const auto o_ptr = output + idx;
  const auto pos = index_ptr[warp_id / warp_count];
  if (const auto index = pos - index_range.start;
      index >= 0 && index < index_range.length) {
    const auto which = index * warp_count + warp_id % warp_count;
    const auto i_ptr = input + which * 32 + lane_id;
    *o_ptr = *i_ptr;
  } else {
    *o_ptr = make_uint4(0, 0, 0, 0);
  }
}

auto get_warp(size_t bytes) -> size_t {
  constexpr auto kUnit = sizeof(uint4) * 32;
  TORCH_CHECK(bytes % kUnit == 0);
  return bytes / kUnit;
}

auto div_ceil(size_t a, size_t b) -> size_t { return (a + b - 1) / b; }

auto fused_indexing(at::Tensor output, at::Tensor input, at::Tensor index,
                    std::pair<int32_t, int32_t> vocab_range,
                    size_t block_size = 256, bool strict = true) -> void {
  TORCH_CHECK(output.is_cuda() && output.is_contiguous() && output.dim() == 2);
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 2);
  TORCH_CHECK(index.is_cuda() && index.is_contiguous() && index.dim() == 1);
  TORCH_CHECK(index.dtype() == at::kInt);
  TORCH_CHECK(output.dtype() == input.dtype());
  TORCH_CHECK(output.size(0) == index.size(0) &&
              output.size(1) == input.size(1));

  // launch configuration
  TORCH_CHECK(block_size % 32 == 0, "block size must be multiple of 32");
  TORCH_CHECK(block_size <= 256, "block size must be no larger than 256");
  const auto index_length = static_cast<size_t>(index.size(0));
  const auto warp_per_block = block_size / 32;
  const auto warp_count = get_warp(output.size(1) * output.element_size());
  const auto num_blocks = div_ceil(index_length * warp_count, warp_per_block);
  const auto stream = at::cuda::getCurrentCUDAStream();

  const auto [lower, upper] = vocab_range;
  const auto length = upper - lower;
  TORCH_CHECK(strict == false || length == input.size(0));
  return indexing_kernel_1_fold<<<num_blocks, block_size, 0, stream>>>(
      static_cast<uint4 *>(output.data_ptr()),
      static_cast<const uint4 *>(input.data_ptr()), index.data_ptr<int32_t>(),
      warp_count, index_length, index_range{lower, length});
}

} // namespace

PYBIND11_MODULE(indexing_kernel, m) {
  m.def("fused_indexing", &fused_indexing, "Fused Indexing (CUDA)");
}
