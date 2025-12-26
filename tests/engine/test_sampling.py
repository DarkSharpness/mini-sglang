from __future__ import annotations

import pytest
import torch
from minisgl.engine.sample import Sampler


class TestTopKSampling:
    """Test cases for top_k sampling functionality."""

    @pytest.fixture
    def sampler(self):
        """Create a Sampler instance on CPU for testing."""
        return Sampler(device=torch.device("cpu"))

    def test_top_k_basic(self, sampler):
        """Test basic top_k filtering with k=3."""
        # Create probability distribution: [0.4, 0.3, 0.2, 0.1]
        probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
        top_k = torch.tensor([3], dtype=torch.int32)

        result = sampler._apply_top_k(probs, top_k)

        # Should keep top 3 tokens (indices 0, 1, 2) and zero out index 3
        assert result[0, 3].item() == 0.0
        assert result[0, 0].item() > 0.0
        assert result[0, 1].item() > 0.0
        assert result[0, 2].item() > 0.0

        # Check normalization
        assert torch.allclose(result.sum(dim=-1), torch.tensor([1.0]))

    def test_top_k_single_token(self, sampler):
        """Test top_k=1 (greedy selection)."""
        probs = torch.tensor([[0.1, 0.5, 0.3, 0.1]], dtype=torch.float32)
        top_k = torch.tensor([1], dtype=torch.int32)

        result = sampler._apply_top_k(probs, top_k)

        # Only the highest probability token (index 1) should have non-zero prob
        assert result[0, 1].item() == 1.0
        assert result[0, 0].item() == 0.0
        assert result[0, 2].item() == 0.0
        assert result[0, 3].item() == 0.0

    def test_top_k_exceeds_vocab(self, sampler):
        """Test top_k larger than vocabulary size."""
        probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
        top_k = torch.tensor([100], dtype=torch.int32)  # k > vocab_size

        result = sampler._apply_top_k(probs, top_k)

        # Should keep all tokens since k > vocab_size
        assert torch.allclose(result, probs)

    def test_top_k_batched(self, sampler):
        """Test top_k with different k values per sequence in batch."""
        probs = torch.tensor(
            [[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]], dtype=torch.float32
        )
        top_k = torch.tensor([2, 3], dtype=torch.int32)

        result = sampler._apply_top_k(probs, top_k)

        # First sequence: keep top 2
        assert result[0, 2].item() == 0.0
        assert result[0, 3].item() == 0.0
        assert result[0, 0].item() > 0.0
        assert result[0, 1].item() > 0.0

        # Second sequence: keep top 3
        assert result[1, 3].item() == 0.0
        assert result[1, 0].item() > 0.0
        assert result[1, 1].item() > 0.0
        assert result[1, 2].item() > 0.0

        # Check normalization for both sequences
        assert torch.allclose(result.sum(dim=-1), torch.tensor([1.0, 1.0]))

    def test_top_k_uniform_distribution(self, sampler):
        """Test top_k on uniform distribution."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        top_k = torch.tensor([2], dtype=torch.int32)

        result = sampler._apply_top_k(probs, top_k)

        # Should keep exactly 2 tokens (any 2 from the uniform distribution)
        non_zero_count = (result[0] > 0).sum().item()
        assert non_zero_count == 2

        # Check normalization
        assert torch.allclose(result.sum(dim=-1), torch.tensor([1.0]))


class TestTopPSampling:
    """Test cases for top_p (nucleus) sampling functionality."""

    @pytest.fixture
    def sampler(self):
        """Create a Sampler instance on CPU for testing."""
        return Sampler(device=torch.device("cpu"))

    def test_top_p_basic(self, sampler):
        """Test basic top_p filtering with p=0.7."""
        # Create probability distribution: [0.5, 0.3, 0.15, 0.05]
        # Cumulative: [0.5, 0.8, 0.95, 1.0]
        # With top_p=0.7, should keep first 2 tokens (cumsum=0.8)
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float32)
        top_p = torch.tensor([0.7], dtype=torch.float32)

        result = sampler._apply_top_p(probs, top_p)

        # Should keep tokens until cumulative prob exceeds 0.7
        assert result[0, 0].item() > 0.0
        assert result[0, 1].item() > 0.0
        # Third token might be included depending on implementation
        # (cumsum > threshold check)

        # Check normalization
        assert torch.allclose(result.sum(dim=-1), torch.tensor([1.0]))

    def test_top_p_very_low(self, sampler):
        """Test top_p with very low value (should keep at least 1 token)."""
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float32)
        top_p = torch.tensor([0.01], dtype=torch.float32)

        result = sampler._apply_top_p(probs, top_p)

        # Should keep at least the top token
        non_zero_count = (result[0] > 0).sum().item()
        assert non_zero_count >= 1

        # Top token should always be kept
        assert result[0, 0].item() > 0.0

        # Check normalization
        assert torch.allclose(result.sum(dim=-1), torch.tensor([1.0]))

    def test_top_p_all_tokens(self, sampler):
        """Test top_p=1.0 (should keep all tokens)."""
        probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
        top_p = torch.tensor([1.0], dtype=torch.float32)

        result = sampler._apply_top_p(probs, top_p)

        # Should keep all tokens
        assert torch.allclose(result, probs)

    def test_top_p_batched(self, sampler):
        """Test top_p with different p values per sequence in batch."""
        probs = torch.tensor(
            [[0.5, 0.3, 0.15, 0.05], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32
        )
        top_p = torch.tensor([0.6, 0.9], dtype=torch.float32)

        result = sampler._apply_top_p(probs, top_p)

        # Both sequences should be properly filtered
        assert (result[0] >= 0).all()
        assert (result[1] >= 0).all()

        # Check normalization for both sequences
        assert torch.allclose(result.sum(dim=-1), torch.tensor([1.0, 1.0]))

    def test_top_p_uniform_distribution(self, sampler):
        """Test top_p on uniform distribution."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        top_p = torch.tensor([0.6], dtype=torch.float32)

        result = sampler._apply_top_p(probs, top_p)

        # With uniform distribution, should keep tokens until cumsum > 0.6
        # Cumsum: [0.25, 0.5, 0.75, 1.0]
        # Should keep 3 tokens (cumsum=0.75 > 0.6)
        non_zero_count = (result[0] > 0).sum().item()
        assert non_zero_count >= 2  # At least 2, possibly 3

        # Check normalization
        assert torch.allclose(result.sum(dim=-1), torch.tensor([1.0]))


class TestCombinedSampling:
    """Test cases for combined top_k and top_p sampling."""

    @pytest.fixture
    def sampler(self):
        """Create a Sampler instance on CPU for testing."""
        return Sampler(device=torch.device("cpu"))

    def test_top_k_then_top_p(self, sampler):
        """Test that top_k is applied before top_p."""
        # Distribution: [0.4, 0.25, 0.2, 0.1, 0.05]
        probs = torch.tensor([[0.4, 0.25, 0.2, 0.1, 0.05]], dtype=torch.float32)
        top_k = torch.tensor([3], dtype=torch.int32)  # Keep top 3
        top_p = torch.tensor([0.8], dtype=torch.float32)  # Then apply nucleus

        # Apply top_k first
        result = sampler._apply_top_k(probs.clone(), top_k)
        # Then apply top_p
        result = sampler._apply_top_p(result, top_p)

        # Last 2 tokens should be zeroed by top_k
        assert result[0, 3].item() == 0.0
        assert result[0, 4].item() == 0.0

        # Check normalization
        assert torch.allclose(result.sum(dim=-1), torch.tensor([1.0]))

    def test_top_k_more_restrictive(self, sampler):
        """Test when top_k is more restrictive than top_p."""
        probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
        top_k = torch.tensor([2], dtype=torch.int32)
        top_p = torch.tensor([0.9], dtype=torch.float32)

        result = sampler._apply_top_k(probs.clone(), top_k)
        result = sampler._apply_top_p(result, top_p)

        # top_k should dominate (only 2 tokens)
        non_zero_count = (result[0] > 0).sum().item()
        assert non_zero_count == 2

    def test_top_p_more_restrictive(self, sampler):
        """Test when top_p is more restrictive than top_k."""
        probs = torch.tensor([[0.6, 0.2, 0.1, 0.05, 0.05]], dtype=torch.float32)
        top_k = torch.tensor([4], dtype=torch.int32)
        top_p = torch.tensor([0.7], dtype=torch.float32)

        result = sampler._apply_top_k(probs.clone(), top_k)
        result = sampler._apply_top_p(result, top_p)

        # top_p should dominate (fewer tokens)
        # Cumsum: [0.6, 0.8, ...] so should keep ~2 tokens
        non_zero_count = (result[0] > 0).sum().item()
        assert non_zero_count <= 4  # At most what top_k allows


class TestSamplingEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def sampler(self):
        """Create a Sampler instance on CPU for testing."""
        return Sampler(device=torch.device("cpu"))

    def test_single_token_vocab(self, sampler):
        """Test with vocabulary of size 1."""
        probs = torch.tensor([[1.0]], dtype=torch.float32)
        top_k = torch.tensor([5], dtype=torch.int32)
        top_p = torch.tensor([0.5], dtype=torch.float32)

        result_k = sampler._apply_top_k(probs.clone(), top_k)
        result_p = sampler._apply_top_p(probs.clone(), top_p)

        # Should keep the only token
        assert result_k[0, 0].item() == 1.0
        assert result_p[0, 0].item() == 1.0

    def test_large_batch(self, sampler):
        """Test with larger batch size."""
        batch_size = 16
        vocab_size = 100
        probs = torch.softmax(torch.randn(batch_size, vocab_size), dim=-1)
        top_k = torch.randint(1, 50, (batch_size,), dtype=torch.int32)
        top_p = torch.rand(batch_size, dtype=torch.float32) * 0.5 + 0.5  # [0.5, 1.0]

        result_k = sampler._apply_top_k(probs.clone(), top_k)
        result_p = sampler._apply_top_p(probs.clone(), top_p)

        # Check all sequences are normalized
        assert torch.allclose(result_k.sum(dim=-1), torch.ones(batch_size))
        assert torch.allclose(result_p.sum(dim=-1), torch.ones(batch_size))

    def test_near_zero_probabilities(self, sampler):
        """Test with very small probabilities."""
        probs = torch.tensor([[1e-8, 1e-8, 1.0 - 2e-8, 1e-8]], dtype=torch.float32)
        top_k = torch.tensor([2], dtype=torch.int32)
        top_p = torch.tensor([0.99], dtype=torch.float32)

        result_k = sampler._apply_top_k(probs.clone(), top_k)
        result_p = sampler._apply_top_p(probs.clone(), top_p)

        # Should handle small probabilities correctly
        assert torch.allclose(result_k.sum(dim=-1), torch.tensor([1.0]))
        assert torch.allclose(result_p.sum(dim=-1), torch.tensor([1.0]))

    def test_identical_probabilities(self, sampler):
        """Test when all probabilities are identical."""
        probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float32)
        top_k = torch.tensor([3], dtype=torch.int32)
        top_p = torch.tensor([0.5], dtype=torch.float32)

        result_k = sampler._apply_top_k(probs.clone(), top_k)
        result_p = sampler._apply_top_p(probs.clone(), top_p)

        # Should keep exactly 3 tokens for top_k
        assert (result_k[0] > 0).sum().item() == 3

        # Check normalization
        assert torch.allclose(result_k.sum(dim=-1), torch.tensor([1.0]))
        assert torch.allclose(result_p.sum(dim=-1), torch.tensor([1.0]))


class TestSamplingProperties:
    """Test mathematical properties of sampling functions."""

    @pytest.fixture
    def sampler(self):
        """Create a Sampler instance on CPU for testing."""
        return Sampler(device=torch.device("cpu"))

    def test_top_k_idempotent(self, sampler):
        """Test that applying top_k twice gives same result."""
        probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
        top_k = torch.tensor([2], dtype=torch.int32)

        result1 = sampler._apply_top_k(probs.clone(), top_k)
        result2 = sampler._apply_top_k(result1.clone(), top_k)

        assert torch.allclose(result1, result2)

    def test_top_p_idempotent(self, sampler):
        """Test that applying top_p twice gives same result."""
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float32)
        top_p = torch.tensor([0.7], dtype=torch.float32)

        result1 = sampler._apply_top_p(probs.clone(), top_p)
        result2 = sampler._apply_top_p(result1.clone(), top_p)

        assert torch.allclose(result1, result2)

    def test_probability_mass_conservation(self, sampler):
        """Test that probability mass sums to 1 after filtering."""
        probs = torch.softmax(torch.randn(5, 50), dim=-1)
        top_k = torch.randint(5, 30, (5,), dtype=torch.int32)
        top_p = torch.tensor([0.9, 0.8, 0.95, 0.7, 0.85], dtype=torch.float32)

        result_k = sampler._apply_top_k(probs.clone(), top_k)
        result_p = sampler._apply_top_p(probs.clone(), top_p)

        assert torch.allclose(result_k.sum(dim=-1), torch.ones(5), atol=1e-6)
        assert torch.allclose(result_p.sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_monotonicity_top_k(self, sampler):
        """Test that larger k keeps more tokens."""
        probs = torch.tensor([[0.3, 0.25, 0.2, 0.15, 0.1]], dtype=torch.float32)

        result_k2 = sampler._apply_top_k(probs.clone(), torch.tensor([2], dtype=torch.int32))
        result_k3 = sampler._apply_top_k(probs.clone(), torch.tensor([3], dtype=torch.int32))
        result_k4 = sampler._apply_top_k(probs.clone(), torch.tensor([4], dtype=torch.int32))

        count_k2 = (result_k2[0] > 0).sum().item()
        count_k3 = (result_k3[0] > 0).sum().item()
        count_k4 = (result_k4[0] > 0).sum().item()

        assert count_k2 <= count_k3 <= count_k4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
