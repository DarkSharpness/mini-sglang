#pragma once

#include "utils.h"
#include <concepts>
#include <cstddef>
#include <source_location>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace host {

inline auto RuntimeCudaCheck(
    ::cudaError_t error,
    std::source_location location = std::source_location::current()) -> void {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

inline auto RuntimeCudaCheck(
    std::source_location location = std::source_location::current()) -> void {
  return RuntimeCudaCheck(::cudaGetLastError(), location);
}

} // namespace host

namespace cuda {

namespace pointer {

template <std::same_as<void> T, std::integral... U>
__always_inline __device__ auto offset(T *ptr, U... offset) -> void * {
  return static_cast<char *>(ptr) + (... + offset);
}

template <std::same_as<void> T, std::integral... U>
__always_inline __device__ auto offset(const T *ptr, U... offset) -> const
    void * {
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer

namespace PDL {

template <bool kUsePDL> __always_inline __device__ void wait() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;");
  }
}

template <bool kUsePDL> __always_inline __device__ void launch() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
}

} // namespace PDL

} // namespace cuda

namespace host {

template <auto F> inline void set_smem_once(std::size_t smem_size) {
  static const auto last_smem_size = [&] {
    host::RuntimeCudaCheck(::cudaFuncSetAttribute(
        F, ::cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    return smem_size;
  }();
  host::RuntimeCheck(smem_size <= last_smem_size,
                     "Dynamic shared memory size exceeds the previously "
                     "set maximum size: ",
                     last_smem_size, " bytes");
}

struct LaunchKernel {
public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_attr(),
        m_config(s_make_config(grid_dim, block_dim, resolve_device(device),
                               dynamic_shared_mem_bytes)) {}

  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_attr(), m_config(s_make_config(grid_dim, block_dim, stream,
                                         dynamic_shared_mem_bytes)) {}

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  auto set_pdl(bool flag = true) -> LaunchKernel & {
    if (flag) {
      m_attr.id = ::cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attr.val.programmaticStreamSerializationAllowed = 1;
      m_config.attrs = &m_attr;
      m_config.numAttrs = 1;
    } else {
      m_config.numAttrs = 0;
    }
    return *this;
  }

  LaunchKernel(const LaunchKernel &) = delete;
  LaunchKernel &operator=(const LaunchKernel &) = delete;

  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
    host::RuntimeCudaCheck(
        ::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...));
  }

private:
  static auto s_make_config(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                            std::size_t dynamic_shared_mem_bytes)
      -> cudaLaunchConfig_t {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = dynamic_shared_mem_bytes;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }
  cudaLaunchAttribute m_attr;
  cudaLaunchConfig_t m_config;
};

} // namespace host
