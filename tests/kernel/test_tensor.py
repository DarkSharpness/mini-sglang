from __future__ import annotations

import pytest
import torch
from minisgl.kernel import test_tensor


@pytest.mark.cuda
def test_test_tensor(cuda_device):
    x_cpu = torch.randint(0, 100, (12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y_cuda = torch.empty((12, 1024), dtype=torch.int64, device=cuda_device)
    test_tensor(x_cpu, y_cuda)
    y_cpu = y_cuda.to("cpu")
    assert torch.all(y_cpu == x_cpu.long())
