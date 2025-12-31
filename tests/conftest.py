import pytest
import torch
import random
import numpy as np
import os


@pytest.fixture(autouse=True, scope="function")
def seed_fixing():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    yield


@pytest.fixture(scope="session")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def pytest_runtest_setup(item):
    # Skip @pytest.mark.cuda if no GPU
    if item.get_closest_marker("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test: No GPU detected")

    # Skip @pytest.mark.distributed on Windows
    if item.get_closest_marker("distributed"):
        if os.name == "nt":
            pytest.skip("Skipping distributed test: Not supported on Windows")
