#include <cstdint>
#include <minisgl/utils.cuh>

#include <cstddef>
#include <cuda_runtime.h>
#include <numa.h>
#include <tvm/ffi/container/tensor.h>

namespace {

tvm::ffi::Tensor allocate_numa(size_t size, int node) {
  void *ptr = numa_alloc_onnode(size, node);
  host::CUDA_CHECK(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
  auto size_stride = new int64_t[2]{};
  size_stride[0] = size / sizeof(uint8_t);
  size_stride[1] = 1;
  const auto deleter = []([[maybe_unused]] DLManagedTensor *t) {
    return; // intentional skip the deallocation
    // auto ptr = t->dl_tensor.data;
    // cudaHostUnregister(ptr);
    // numa_free(ptr, t->dl_tensor.shape[0]);
    // delete[] t->dl_tensor.shape;
  };
  const auto dl_tensor = DLTensor{.data = ptr,
                                  .device = {kDLCPU, 0},
                                  .ndim = 1,
                                  .dtype = {kDLUInt, 8, 1},
                                  .shape = size_stride + 0,
                                  .strides = size_stride + 1,
                                  .byte_offset = 0};
  auto dl_managed_tensor = DLManagedTensor{
      .dl_tensor = dl_tensor,
      .manager_ctx = nullptr,
      .deleter = deleter,
  };
  return tvm::ffi::Tensor::FromDLPack(&dl_managed_tensor);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(allocate_numa, allocate_numa);

} // namespace
