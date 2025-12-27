# SIMD Optimization Guide: Radix Key Comparison

## Overview

This document explains the AVX2 SIMD optimization applied to the `fast_compare_key` function in `python/minisgl/kernel/csrc/src/radix.cpp`.

## What Was Optimized?

**Function:** `fast_compare_key` - Compares two integer arrays and returns the index of the first mismatch

**Use Case:** This function is called frequently by the radix cache to find common prefixes between token sequences. Faster prefix matching = faster cache lookups = higher throughput.

## The Optimization

### Before (Scalar):
```cpp
std::mismatch(a_ptr_32, a_ptr_32 + common_len, b_ptr_32);
```
- Compares **1 element per iteration**
- ~4-8 cycles per element (depending on CPU)

### After (SIMD with AVX2):
```cpp
compare_int32_avx2(a_ptr_32, b_ptr_32, common_len);
```
- Compares **8 int32 elements or 4 int64 elements per iteration**
- Same ~4-8 cycles, but for 4-8x more elements

## How It Works

### 1. **Load 256 bits of data at once**
```cpp
__m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
__m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
```
- For int32: loads 8 elements (8 × 4 bytes = 32 bytes)
- For int64: loads 4 elements (4 × 8 bytes = 32 bytes)

### 2. **Compare all elements in parallel**
```cpp
__m256i cmp = _mm256_cmpeq_epi32(va, vb);  // or _mm256_cmpeq_epi64 for int64
```
- Single CPU instruction compares all 8/4 elements simultaneously
- Result: each lane is 0xFFFFFFFF if equal, 0x00000000 if not

### 3. **Check if all matched**
```cpp
int mask = _mm256_movemask_epi8(cmp);
if (mask != 0xFFFFFFFF) {
    // Found a mismatch, do scalar search to find exact position
}
```

### 4. **Handle remaining elements**
```cpp
for (; i < len; ++i) {
    if (a[i] != b[i]) return i;
}
```
- Process leftover elements (< 8 for int32, < 4 for int64) with scalar code

## Key Features

### ✅ **Runtime CPU Detection**
```cpp
inline bool has_avx2() {
    // Checks CPU capabilities at runtime
    // Caches result in static variable
}
```
- Uses CPUID instruction to detect AVX2 support
- Falls back to `std::mismatch` on older CPUs
- Zero overhead after first call (static caching)

### ✅ **Cross-platform Support**
```cpp
#if defined(__x86_64__) || defined(_M_X64)
    // SIMD code
#endif
    // Fallback code
```
- SIMD code only compiled on x86_64
- ARM/other architectures use standard library

### ✅ **Unaligned Memory Access**
```cpp
_mm256_loadu_si256(...)  // 'u' = unaligned
```
- Works with any memory address
- No alignment requirements on input tensors

## Expected Performance Gains

| Array Length | Match % | Speedup |
|--------------|---------|---------|
| 32 elements  | 10%     | ~2-3x   |
| 128 elements | 50%     | ~4-5x   |
| 512 elements | 80%     | ~5-7x   |
| 2048 elements| 95%     | ~6-8x   |

**Why variable speedup?**
- Early mismatches: Less benefit (fewer SIMD iterations)
- Late mismatches: More benefit (more SIMD iterations)
- Memory bandwidth: Can become bottleneck on very long arrays

## Testing

### Run Correctness Tests
```bash
python tests/kernel/test_radix.py
```

Tests cover:
- Exact matches
- Mismatches at various positions
- Boundary conditions (SIMD lane boundaries)
- Different array lengths
- Negative numbers
- Random data stress tests

### Run Performance Benchmarks
```bash
python tests/kernel/test_radix_benchmark.py
```

Compares baseline vs. optimized performance across various scenarios.

## How to Extend This

### Add AVX-512 Support
```cpp
#if defined(__AVX512F__)
inline size_t compare_int32_avx512(const int32_t *a, const int32_t *b, size_t len) {
    // Process 16 int32 or 8 int64 at once
    // Use _mm512_* intrinsics
}
#endif
```

### Add ARM NEON Support
```cpp
#if defined(__aarch64__)
#include <arm_neon.h>

inline size_t compare_int32_neon(const int32_t *a, const int32_t *b, size_t len) {
    // Use vld1q_s32, vceqq_s32, etc.
}
#endif
```

## SIMD Learning Resources

1. **Intel Intrinsics Guide**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
2. **SIMD for C++ Developers**: https://www.intel.com/content/www/us/en/developer/articles/technical/a-guide-to-auto-vectorization-with-intel-c-compilers.html
3. **Agner Fog's Optimization Manuals**: https://www.agner.org/optimize/

## Common SIMD Patterns

### Pattern 1: Element-wise Operations
```cpp
// Add 8 int32s at once
__m256i va = _mm256_loadu_si256(...);
__m256i vb = _mm256_loadu_si256(...);
__m256i result = _mm256_add_epi32(va, vb);
```

### Pattern 2: Horizontal Reductions
```cpp
// Sum all 8 int32s in a vector
__m256i sum = _mm256_hadd_epi32(v, v);
sum = _mm256_hadd_epi32(sum, sum);
```

### Pattern 3: Masked Operations
```cpp
// Conditional updates based on comparison
__m256i mask = _mm256_cmpgt_epi32(va, vb);
__m256i result = _mm256_blendv_epi8(va, vb, mask);
```

## Key Takeaways

1. **SIMD provides massive speedups** for data-parallel operations
2. **Always provide fallbacks** for older CPUs and other architectures
3. **Test thoroughly** - SIMD bugs can be subtle (alignment, edge cases)
4. **Profile first** - Not all code benefits from SIMD
5. **Consider memory bandwidth** - SIMD can saturate memory bus

## Next Steps

Apply similar SIMD optimizations to:
- `store_cache` kernel (vectorize K/V stores)
- Sampling operations (top-k with SIMD)
- Token embedding lookups
- Attention score calculations

---

**Author:** Mini-SGLang Contributors
**Date:** 2025-12-22
**Related Files:**
- `python/minisgl/kernel/csrc/src/radix.cpp` - Implementation
- `tests/kernel/test_radix.py` - Unit tests
- `tests/kernel/test_radix_benchmark.py` - Performance benchmarks
