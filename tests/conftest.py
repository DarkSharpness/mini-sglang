import pytest
import torch
import random
import numpy as np


@pytest.fixture(autouse=True, scope="function")
def seed_fixing():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


def pytest_runtest_setup(item):
    if "cuda" in [mark.name for mark in item.iter_markers()]:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
