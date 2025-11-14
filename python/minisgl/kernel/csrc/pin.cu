#include <ATen/core/TensorBody.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/TensorOptions.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <numa.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <torch/types.h>

namespace {

auto numa_max_node_fast() -> int {
  static const int kNumaMaxNode = [] {
    if (std::getenv("NUMA_MAX_NODE")) {
      return std::atoi(std::getenv("NUMA_MAX_NODE"));
    } else {
      return ::numa_max_node();
    }
  }();
  return kNumaMaxNode;
}

auto make_pin_tensor(const int64_t size, const torch::Dtype dtype,
                     bool write_combine, std::optional<int> numa_aff)
    -> at::Tensor {
  const auto size_bytes = size * at::elementSize(dtype);
  void *data_ptr;
  const auto options =
      at::TensorOptions(at::kCPU).dtype(dtype).pinned_memory(true);
  if (numa_aff.has_value()) {
    const auto kNumaMaxNode = numa_max_node_fast();
    const auto node = numa_aff.value();
    TORCH_CHECK(node >= 0 && node <= kNumaMaxNode, "Invalid NUMA node: ", node);
    TORCH_CHECK(!write_combine, "Write-combine is not supported with NUMA");
    // allocate on the specified NUMA node
    data_ptr = ::numa_alloc_onnode(size_bytes, node);
    TORCH_CHECK(data_ptr != nullptr,
                "Failed to allocate memory on NUMA node: ", node);
    const auto flags = cudaHostRegisterDefault;
    const auto result = ::cudaHostRegister(data_ptr, size_bytes, flags);
    if (result != ::cudaSuccess) {
      ::numa_free(data_ptr, size_bytes);
      TORCH_CHECK(false, "Failed to register pinned memory: ",
                  ::cudaGetErrorString(result));
    }
    return at::from_blob(
        data_ptr, {size},
        [size_bytes](void *data_ptr) {
          const auto result = ::cudaHostUnregister(data_ptr);
          ::numa_free(data_ptr, size_bytes);
          TORCH_CHECK(result == ::cudaSuccess,
                      "Failed to unregister pinned memory: ",
                      ::cudaGetErrorString(result));
        },
        options, at::kCPU);
  } else {
    const auto flags =
        write_combine ? cudaHostAllocWriteCombined : cudaHostAllocDefault;
    const auto result = ::cudaHostAlloc(&data_ptr, size_bytes, flags);
    TORCH_CHECK(result == ::cudaSuccess, "Failed to allocate pinned memory: ",
                ::cudaGetErrorString(result));
    return at::from_blob(
        data_ptr, {size},
        [](void *data_ptr) {
          const auto result = ::cudaFreeHost(data_ptr);
          TORCH_CHECK(result == ::cudaSuccess, "Failed to free pinned memory: ",
                      ::cudaGetErrorString(result));
        },
        options, at::kCPU);
  }
}

} // namespace

PYBIND11_MODULE(pin_memory_allocator, m) {
  m.def("make_pin_tensor", &make_pin_tensor);
  m.def("numa_count", [] { return numa_max_node_fast() + 1; });
}
