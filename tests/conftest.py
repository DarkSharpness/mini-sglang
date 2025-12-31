import pytest
import torch
import random
import numpy as np
import os
from minisgl.distributed import set_tp_info


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
    return torch.device("cuda:0")


@pytest.fixture(scope="function")
def distributed_env():
    """Initializes torch.distributed and returns rank and world size."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("distributed tests require at least 2 GPUs")

    tp_rank = int(os.environ["RANK"])
    tp_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(tp_rank)
    torch.cuda.set_stream(torch.cuda.Stream(tp_rank))
    set_tp_info(tp_rank, tp_size)

    torch.distributed.init_process_group(
        world_size=tp_size,
        rank=tp_rank,
        backend="gloo",
        init_method="env://",
    )
    yield tp_rank, tp_size
    torch.distributed.destroy_process_group()


def pytest_runtest_setup(item):
    # Skip @pytest.mark.cuda if no GPU
    if item.get_closest_marker("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test: No GPU detected")

    # Skip @pytest.mark.distributed on Windows
    if item.get_closest_marker("distributed"):
        if os.name == "nt":
            pytest.skip("Skipping distributed test: Not supported on Windows")
