import os
import time
import torch
import pytest
from minisgl.distributed import set_tp_info
import minisgl.kernel as kernel

from minisgl.utils import init_logger


logger = init_logger(__name__)


class TestPyNCCLCommunicator:

    def _setup_distributed(self):
        """Initializes torch.distributed and returns rank and world size."""
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
        return tp_rank, tp_size

    def _get_comm(self, tp_rank, tp_size, K, dtype):
        """Initializes and returns the PyNCCL communicator."""
        tp_cpu_group = torch.distributed.group.WORLD
        assert tp_cpu_group is not None, "CPU group should not be None"
        USE_SYMM = 0
        return kernel.init_pynccl(
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_cpu_group=tp_cpu_group,
            max_size_bytes=8192 * K * dtype.itemsize if USE_SYMM else 0,
        )

    def _test_all_reduce_ones(self, comm, tp_rank, tp_size, K, dtype, N=4):
        """Verifies all_reduce with a tensor of ones."""
        f = lambda x: comm.all_reduce(x, "sum")
        x = torch.ones(8192 * K, dtype=dtype, device=f"cuda:{tp_rank}")
        for _ in range(N):
            f(x)
        ans = pow(tp_size, N)
        y = torch.full((8192 * K,), ans, dtype=dtype, device=f"cuda:{tp_rank}")
        assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"

    def _test_all_reduce_rank_sync(self, comm, tp_rank, tp_size, K, dtype):
        """Verifies all_reduce with a delay on rank 0 to test synchronization."""
        f = lambda x: comm.all_reduce(x, "sum")
        x = torch.full((8192 * K,), tp_rank, dtype=dtype, device=f"cuda:{tp_rank}")
        # sanity check: if one TP rank lags behind, the others should wait
        if tp_rank == 0:
            torch.cuda.synchronize()
            time.sleep(1)
        f(x)
        ans = (tp_size * (tp_size - 1)) // 2
        y = torch.full((8192 * K,), ans, dtype=dtype, device=f"cuda:{tp_rank}")
        assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"

    def _test_all_reduce_split_tensor(self, comm, tp_rank, tp_size, K, dtype):
        """Verifies all_reduce with a non-uniform tensor."""
        f = lambda x: comm.all_reduce(x, "sum")
        # to prevent overflow, we use a smaller value for this test
        x = torch.cat(
            [
                torch.zeros((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
                torch.ones((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
            ]
        )
        f(x)
        y = torch.cat(
            [
                torch.zeros((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
                torch.full((8192 * K // 2,), tp_size, dtype=dtype, device=f"cuda:{tp_rank}"),
            ]
        )
        assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"

    def _test_all_gather(self, comm, tp_rank, tp_size, K, dtype):
        """Verifies all_gather correctness."""
        src = torch.full((K,), tp_rank, dtype=dtype, device=f"cuda:{tp_rank}")
        torch.cuda.synchronize()
        dst = torch.empty((K * tp_size,), dtype=dtype, device=f"cuda:{tp_rank}")
        comm.all_gather(dst, src)
        torch.cuda.synchronize()
        expected = torch.arange(tp_size, dtype=dtype, device=f"cuda:{tp_rank}")
        expected = expected.repeat_interleave(K)
        assert torch.allclose(dst, expected), f"Rank {tp_rank} all-gather failed"

    @pytest.mark.distributed
    @torch.no_grad()
    def test_all_reduce_and_all_gather_correctness(self):
        """
        Tests correctness with a simple loop to catch basic race conditions and edge cases.
        """
        tp_rank, tp_size = self._setup_distributed()
        dtype = torch.float16
        K = 512
        comm = self._get_comm(tp_rank, tp_size, K, dtype)

        for i in range(10):
            if tp_rank == 0:
                logger.info(f"Simple correctness iteration {i+1}/10")
            self._test_all_reduce_ones(comm, tp_rank, tp_size, K, dtype)
            self._test_all_reduce_rank_sync(comm, tp_rank, tp_size, K, dtype)
            self._test_all_reduce_split_tensor(comm, tp_rank, tp_size, K, dtype)

        self._test_all_gather(comm, tp_rank, tp_size, K, dtype)
        torch.distributed.destroy_process_group()

    @pytest.mark.distributed
    @torch.no_grad()
    def test_all_reduce_and_all_gather_stress(self):
        """Full correctness + stress validation"""
        tp_rank, tp_size = self._setup_distributed()
        dtype = torch.float16
        K = 512
        comm = self._get_comm(tp_rank, tp_size, K, dtype)

        def run_correctness_phase():
            """Runs a subset of correctness checks for the stress test."""
            self._test_all_reduce_ones(comm, tp_rank, tp_size, K, dtype, N=4)
            self._test_all_gather(comm, tp_rank, tp_size, K, dtype)

        def run_stress_phase(duration_sec=5):
            """Simulates benchmark load without measuring performance."""
            if tp_rank == 0:
                logger.info(f"Running stress phase for {duration_sec}s...")
            start = time.time()
            stress_tensor = torch.randn(8192 * 512, dtype=dtype, device=f"cuda:{tp_rank}")
            while time.time() - start < duration_sec:
                comm.all_reduce(stress_tensor, "sum")
            if tp_rank == 0:
                logger.info("Stress phase complete.")

        # PHASE 1: Basic correctness
        for i in range(3):
            if tp_rank == 0:
                logger.info(f"Basic correctness iteration {i+1}/3")
            run_correctness_phase()

        # PHASE 2: Stress test
        run_stress_phase(duration_sec=10)
        torch.cuda.synchronize()

        # PHASE 3: Post-stress correctness
        for i in range(5):
            if tp_rank == 0:
                logger.info(f"Post-stress correctness iteration {i+1}/5")
            run_correctness_phase()

        if tp_rank == 0:
            logger.info("All phases passed!")

        torch.distributed.destroy_process_group()
