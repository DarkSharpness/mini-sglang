from __future__ import annotations

import pytest
import torch
from minisgl.kernel import validate_tensor_ffi


@pytest.mark.cuda
def test_validate_tensor_ffi_success(cuda_device):
    x_cpu = torch.randint(0, 100, (12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y_cuda = torch.empty((12, 1024), dtype=torch.int64, device=cuda_device)
    ret = validate_tensor_ffi(x_cpu, y_cuda)
    assert ret == 0, f"FFI validation failed with return code {ret}"
