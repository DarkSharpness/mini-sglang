from __future__ import annotations

from minisgl.kernel import test_tensor
from minisgl.utils import call_if_main
import torch


@call_if_main()
def main():
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    test_tensor(x, y)
