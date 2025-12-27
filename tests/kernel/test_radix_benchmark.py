"""Benchmark for radix key comparison optimization."""
from __future__ import annotations

import time
import torch
from minisgl.kernel import fast_compare_key


def benchmark_compare_key(
    length: int,
    dtype: torch.dtype,
    match_ratio: float = 0.5,
    iterations: int = 10000
) -> float:
    """
    Benchmark fast_compare_key function.

    Args:
        length: Length of arrays to compare
        dtype: torch.int32 or torch.int64
        match_ratio: Fraction of elements that match (0.0 to 1.0)
        iterations: Number of iterations to run

    Returns:
        Average time per comparison in microseconds
    """
    # Create test arrays
    a = torch.randint(0, 10000, (length,), dtype=dtype, device="cpu")
    b = a.clone()

    # Modify b to control match ratio
    mismatch_idx = int(length * match_ratio)
    if mismatch_idx < length:
        b[mismatch_idx:] = torch.randint(10000, 20000, (length - mismatch_idx,), dtype=dtype)

    # Warmup
    for _ in range(100):
        fast_compare_key(a, b)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = fast_compare_key(a, b)
    end = time.perf_counter()

    avg_time_us = (end - start) / iterations * 1_000_000

    # Verify correctness
    expected = mismatch_idx if mismatch_idx < length else length
    assert result == expected, f"Expected {expected}, got {result}"

    return avg_time_us


def main():
    print("=" * 80)
    print("Radix Key Comparison Benchmark - BASELINE")
    print("=" * 80)

    test_cases = [
        # (length, match_ratio, description)
        (32, 0.1, "Short, early mismatch (cache miss)"),
        (32, 0.9, "Short, late mismatch (cache hit)"),
        (128, 0.5, "Medium, 50% match"),
        (512, 0.8, "Long, 80% match (common in radix cache)"),
        (1024, 0.95, "Very long, 95% match"),
        (2048, 0.99, "Max length, 99% match"),
    ]

    for dtype in [torch.int32, torch.int64]:
        print(f"\n{'='*80}")
        print(f"Data type: {dtype}")
        print(f"{'='*80}")
        print(f"{'Length':<10} {'Match%':<10} {'Time (μs)':<15} {'Description':<40}")
        print("-" * 80)

        for length, match_ratio, desc in test_cases:
            avg_time = benchmark_compare_key(length, dtype, match_ratio, iterations=10000)
            print(f"{length:<10} {match_ratio*100:<10.0f} {avg_time:<15.3f} {desc:<40}")

    print("\n" + "=" * 80)
    print("Benchmark complete! Save these numbers to compare after optimization.")
    print("=" * 80)


if __name__ == "__main__":
    main()
