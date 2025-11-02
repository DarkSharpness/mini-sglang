#pragma once
#include <cstddef>
#include <sys/cdefs.h>

namespace cuda::warp {

template <std::size_t kUnit> inline constexpr auto _get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint32_t{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4,
                  "Unsupported memory package size");
  }
}

inline constexpr auto _resolve_unit_size(std::size_t x, std::size_t y)
    -> std::size_t {
  if (y != 0)
    return y;
  if (x % (16 * 32) == 0)
    return 16;
  if (x % (8 * 32) == 0)
    return 8;
  if (x % (4 * 32) == 0)
    return 4;
  return 0; // trigger static assert in _get_mem_package
}

template <std::size_t kBytes, std::size_t kUnit>
using _mem_package_t =
    decltype(_get_mem_package<_resolve_unit_size(kBytes, kUnit)>());

template <std::size_t kBytes, std::size_t kUnit = 0>
__always_inline __device__ void copy(void *__restrict__ dst,
                                     const void *__restrict__ src) {
  using Package = _mem_package_t<kBytes, kUnit>;
  static_assert(kBytes % (sizeof(Package) * 32u) == 0,
                "warp_copy: kBytes must be multiple of 128 bytes");
  constexpr auto kLoopCount = kBytes / (sizeof(Package) * 32u);

  const auto dst_ = static_cast<Package *>(dst);
  const auto src_ = static_cast<const Package *>(src);
  const auto lane_id = threadIdx.x % 32u;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    dst_[i * 32u + lane_id] = src_[i * 32u + lane_id];
  }
}

template <std::size_t kBytes, std::size_t kUnit = 0>
__always_inline __device__ void reset(void *__restrict__ dst) {
  using Package = _mem_package_t<kBytes, kUnit>;
  static_assert(kBytes % (sizeof(Package) * 32u) == 0,
                "warp_copy: kBytes must be multiple of 128 bytes");
  constexpr auto kLoopCount = kBytes / (sizeof(Package) * 32u);

  const auto dst_ = static_cast<Package *>(dst);
  const auto lane_id = threadIdx.x % 32u;
  const auto zero_value = Package{};

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    dst_[i * 32u + lane_id] = zero_value;
  }
}

} // namespace cuda::warp
