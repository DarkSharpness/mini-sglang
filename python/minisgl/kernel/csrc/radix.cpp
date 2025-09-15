#include <ATen/core/TensorBody.h>
#include <algorithm>
#include <c10/util/Exception.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace {

auto _is_1d_cpu_int_tensor(const at::Tensor &t) -> bool {
  return t.dim() == 1 && t.device().is_cpu() && t.dtype() == at::kInt;
}

auto fast_compare_key(at::Tensor a, at::Tensor b) -> size_t {
  TORCH_CHECK(_is_1d_cpu_int_tensor(a) && _is_1d_cpu_int_tensor(b),
              "Both tensors must be 1D CPU int tensors.");
  const auto a_ptr = a.data_ptr<int>();
  const auto b_ptr = b.data_ptr<int>();
  const auto common_len = std::min(a.size(0), b.size(0));
  // use memcmp to find the first different position
  const auto diff_pos = std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
  return static_cast<size_t>(diff_pos.first - a_ptr);
}

} // namespace

PYBIND11_MODULE(radix_tree, m) { m.def("fast_compare_key", &fast_compare_key); }
