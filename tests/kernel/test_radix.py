"""Tests for radix key comparison with SIMD optimization."""
from __future__ import annotations

import torch
import pytest
from minisgl.kernel import fast_compare_key


class TestRadixComparison:
    """Test suite for fast_compare_key function."""

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_exact_match(self, dtype):
        """Test when arrays are identical."""
        a = torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device="cpu")
        b = a.clone()
        result = fast_compare_key(a, b)
        assert result == 5, "Should return length when arrays match exactly"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_first_element_differs(self, dtype):
        """Test when first element differs."""
        a = torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device="cpu")
        b = torch.tensor([9, 2, 3, 4, 5], dtype=dtype, device="cpu")
        result = fast_compare_key(a, b)
        assert result == 0, "Should return 0 when first element differs"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_last_element_differs(self, dtype):
        """Test when last element differs."""
        a = torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device="cpu")
        b = torch.tensor([1, 2, 3, 4, 9], dtype=dtype, device="cpu")
        result = fast_compare_key(a, b)
        assert result == 4, "Should return 4 when last element differs"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_middle_element_differs(self, dtype):
        """Test when middle element differs."""
        a = torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device="cpu")
        b = torch.tensor([1, 2, 9, 4, 5], dtype=dtype, device="cpu")
        result = fast_compare_key(a, b)
        assert result == 2, "Should return 2 when middle element differs"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_different_lengths_shorter_a(self, dtype):
        """Test when array a is shorter."""
        a = torch.tensor([1, 2, 3], dtype=dtype, device="cpu")
        b = torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device="cpu")
        result = fast_compare_key(a, b)
        assert result == 3, "Should return length of shorter array when prefix matches"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_different_lengths_shorter_b(self, dtype):
        """Test when array b is shorter."""
        a = torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device="cpu")
        b = torch.tensor([1, 2, 3], dtype=dtype, device="cpu")
        result = fast_compare_key(a, b)
        assert result == 3, "Should return length of shorter array when prefix matches"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_empty_arrays(self, dtype):
        """Test with empty arrays."""
        a = torch.tensor([], dtype=dtype, device="cpu")
        b = torch.tensor([], dtype=dtype, device="cpu")
        result = fast_compare_key(a, b)
        assert result == 0, "Should return 0 for empty arrays"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_single_element_match(self, dtype):
        """Test single element arrays that match."""
        a = torch.tensor([42], dtype=dtype, device="cpu")
        b = torch.tensor([42], dtype=dtype, device="cpu")
        result = fast_compare_key(a, b)
        assert result == 1, "Should return 1 for single matching element"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_single_element_differ(self, dtype):
        """Test single element arrays that differ."""
        a = torch.tensor([42], dtype=dtype, device="cpu")
        b = torch.tensor([99], dtype=dtype, device="cpu")
        result = fast_compare_key(a, b)
        assert result == 0, "Should return 0 for single differing element"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("length", [7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65])
    def test_simd_boundary_exact(self, dtype, length):
        """Test lengths around SIMD boundaries (8 for int32, 4 for int64)."""
        a = torch.arange(length, dtype=dtype, device="cpu")
        b = a.clone()
        result = fast_compare_key(a, b)
        assert result == length, f"Failed for exact match at length {length}"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("length", [7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65])
    def test_simd_boundary_differ_at_end(self, dtype, length):
        """Test mismatch at last element for various lengths."""
        a = torch.arange(length, dtype=dtype, device="cpu")
        b = a.clone()
        b[-1] = -1
        result = fast_compare_key(a, b)
        assert result == length - 1, f"Failed for mismatch at end with length {length}"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_long_arrays(self, dtype):
        """Test with longer arrays (stress test)."""
        length = 1024
        a = torch.arange(length, dtype=dtype, device="cpu")
        b = a.clone()

        # Test exact match
        result = fast_compare_key(a, b)
        assert result == length, "Failed for long exact match"

        # Test mismatch at position 500
        b[500] = -1
        result = fast_compare_key(a, b)
        assert result == 500, "Failed to find mismatch at position 500"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_negative_numbers(self, dtype):
        """Test with negative numbers."""
        a = torch.tensor([-1, -2, -3, -4, -5], dtype=dtype, device="cpu")
        b = torch.tensor([-1, -2, -99, -4, -5], dtype=dtype, device="cpu")
        result = fast_compare_key(a, b)
        assert result == 2, "Should handle negative numbers correctly"

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_random_large_arrays(self, dtype):
        """Test with random data in large arrays."""
        length = 2048
        a = torch.randint(-10000, 10000, (length,), dtype=dtype, device="cpu")
        b = a.clone()

        # Should match exactly
        result = fast_compare_key(a, b)
        assert result == length, "Random arrays should match when cloned"

        # Modify at various positions and test
        for mismatch_pos in [0, 7, 8, 9, 63, 64, 65, 127, 128, 500, 1000, 2047]:
            if mismatch_pos < length:
                b = a.clone()
                b[mismatch_pos] = a[mismatch_pos] + 1
                result = fast_compare_key(a, b)
                assert result == mismatch_pos, f"Failed to detect mismatch at {mismatch_pos}"


def main():
    """Run all tests manually."""
    test = TestRadixComparison()

    print("Running radix comparison tests...")
    print("=" * 80)

    tests = [
        ("Exact match", lambda: test.test_exact_match(torch.int32)),
        ("First element differs", lambda: test.test_first_element_differs(torch.int32)),
        ("Last element differs", lambda: test.test_last_element_differs(torch.int32)),
        ("Middle element differs", lambda: test.test_middle_element_differs(torch.int32)),
        ("Different lengths (shorter a)", lambda: test.test_different_lengths_shorter_a(torch.int32)),
        ("Different lengths (shorter b)", lambda: test.test_different_lengths_shorter_b(torch.int32)),
        ("Empty arrays", lambda: test.test_empty_arrays(torch.int32)),
        ("Single element match", lambda: test.test_single_element_match(torch.int32)),
        ("Single element differ", lambda: test.test_single_element_differ(torch.int32)),
        ("Long arrays", lambda: test.test_long_arrays(torch.int32)),
        ("Negative numbers", lambda: test.test_negative_numbers(torch.int32)),
        ("Random large arrays", lambda: test.test_random_large_arrays(torch.int32)),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: Unexpected error: {e}")
            failed += 1

    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"Some tests failed. Please review.")

    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
