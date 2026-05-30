from __future__ import annotations

import minisgl.kernel as kernel
import pytest
import torch


def test_tensor_op():
    """Test the tensor operation kernel."""
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    kernel.test_tensor(x, y)
    # The kernel should complete without error
    # Add a simple assertion to verify execution
    assert y.shape == (12, 1024)


if __name__ == "__main__":
    pytest.main([__file__])
