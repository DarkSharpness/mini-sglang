from __future__ import annotations

import pytest
import torch
from minisgl.kernel import test_tensor as run_test_tensor


@pytest.mark.cuda
def test_test_tensor_ffi_check(cuda_device):
    x_cpu = torch.randint(0, 100, (12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y_cuda = torch.empty((12, 1024), dtype=torch.int64, device=cuda_device)
    run_test_tensor(x_cpu, y_cuda)
