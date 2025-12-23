#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

// SIMD intrinsics for x86_64
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>

namespace {

// Runtime CPU feature detection
inline bool has_avx2() {
  static const bool avx2_supported = []() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
      return (ebx & bit_AVX2) != 0;
    }
    return false;
  }();
  return avx2_supported;
}

// AVX2-optimized comparison for int32
// Compares 8 int32 elements at once (256 bits)
inline size_t compare_int32_avx2(const int32_t *a, const int32_t *b,
                                  size_t len) {
  size_t i = 0;
  const size_t simd_len = len & ~7; // Round down to multiple of 8

  // Process 8 elements at a time
  for (; i < simd_len; i += 8) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i cmp = _mm256_cmpeq_epi32(va, vb);

    // Convert comparison result to bitmask
    int mask = _mm256_movemask_epi8(cmp);

    // If not all equal (mask != 0xFFFFFFFF), find first difference
    if (mask != static_cast<int>(0xFFFFFFFF)) {
      // Each int32 uses 4 bytes, so we check in groups of 4 bits
      for (size_t j = 0; j < 8; ++j) {
        if (a[i + j] != b[i + j]) {
          return i + j;
        }
      }
    }
  }

  // Handle remaining elements with scalar comparison
  for (; i < len; ++i) {
    if (a[i] != b[i]) {
      return i;
    }
  }

  return len;
}

// AVX2-optimized comparison for int64
// Compares 4 int64 elements at once (256 bits)
inline size_t compare_int64_avx2(const int64_t *a, const int64_t *b,
                                  size_t len) {
  size_t i = 0;
  const size_t simd_len = len & ~3; // Round down to multiple of 4

  // Process 4 elements at a time
  for (; i < simd_len; i += 4) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    __m256i cmp = _mm256_cmpeq_epi64(va, vb);

    // Convert comparison result to bitmask
    int mask = _mm256_movemask_epi8(cmp);

    // If not all equal, find first difference
    if (mask != static_cast<int>(0xFFFFFFFF)) {
      for (size_t j = 0; j < 4; ++j) {
        if (a[i + j] != b[i + j]) {
          return i + j;
        }
      }
    }
  }

  // Handle remaining elements with scalar comparison
  for (; i < len; ++i) {
    if (a[i] != b[i]) {
      return i;
    }
  }

  return len;
}

} // namespace
#endif // __x86_64__ || _M_X64

namespace {

auto _is_1d_cpu_int_tensor(const tvm::ffi::TensorView tensor) -> bool {
  return tensor.ndim() == 1 && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         (tensor.dtype().code == kDLInt) &&
         (tensor.dtype().bits == 32 || tensor.dtype().bits == 64);
}

auto fast_compare_key(const tvm::ffi::TensorView a,
                      const tvm::ffi::TensorView b) -> size_t {
  host::RuntimeCheck(_is_1d_cpu_int_tensor(a) && _is_1d_cpu_int_tensor(b),
                     "Both tensors must be 1D CPU int tensors.");
  host::RuntimeCheck(a.dtype() == b.dtype());
  const auto a_ptr = a.data_ptr();
  const auto b_ptr = b.data_ptr();
  const auto common_len = std::min(a.size(0), b.size(0));

#if defined(__x86_64__) || defined(_M_X64)
  // Use SIMD-optimized version if AVX2 is available
  if (has_avx2()) {
    if (a.dtype().bits == 64) {
      const auto a_ptr_64 = static_cast<const int64_t *>(a_ptr);
      const auto b_ptr_64 = static_cast<const int64_t *>(b_ptr);
      return compare_int64_avx2(a_ptr_64, b_ptr_64, common_len);
    } else {
      const auto a_ptr_32 = static_cast<const int32_t *>(a_ptr);
      const auto b_ptr_32 = static_cast<const int32_t *>(b_ptr);
      return compare_int32_avx2(a_ptr_32, b_ptr_32, common_len);
    }
  }
#endif

  // Fallback to standard library implementation
  if (a.dtype().bits == 64) {
    const auto a_ptr_64 = static_cast<const int64_t *>(a_ptr);
    const auto b_ptr_64 = static_cast<const int64_t *>(b_ptr);
    const auto diff_pos =
        std::mismatch(a_ptr_64, a_ptr_64 + common_len, b_ptr_64);
    return static_cast<size_t>(diff_pos.first - a_ptr_64);
  } else {
    const auto a_ptr_32 = static_cast<const int32_t *>(a_ptr);
    const auto b_ptr_32 = static_cast<const int32_t *>(b_ptr);
    const auto diff_pos =
        std::mismatch(a_ptr_32, a_ptr_32 + common_len, b_ptr_32);
    return static_cast<size_t>(diff_pos.first - a_ptr_32);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_key, fast_compare_key);
