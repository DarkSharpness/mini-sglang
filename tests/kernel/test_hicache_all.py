from __future__ import annotations
import os
from typing import NamedTuple

import torch
import matplotlib.pyplot as plt

from minisgl.benchmark.perf import compare_memory_kernel_perf, perf_cuda
from minisgl.kernel import create_pin_tensor
from minisgl.kernel_v2 import transfer_hicache_all_layer
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def ref_hicache_impl(
    k_ptr_dst: torch.Tensor,
    v_ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_bytes: int,
    block_quota: int,
    num_layers: int,
) -> None:
    from sgl_kernel import transfer_kv_all_layer

    transfer_kv_all_layer(
        src_k_layers=k_ptr_src,
        src_v_layers=v_ptr_src,
        dst_k_layers=k_ptr_dst,
        dst_v_layers=v_ptr_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=item_bytes,
        block_quota=block_quota,
        num_layers=num_layers,
    )


class HicacheBenchArgs(NamedTuple):
    cache_item_size: int
    dtype: torch.dtype
    block_quota: int


@torch.inference_mode()
def test_hicache_kernel(
    args: HicacheBenchArgs,
    need_warmup: bool = True,
    need_plot: bool = True,
) -> None:
    CACHE_ITEM_SIZE, DTYPE, BLOCK_QUOTA = args

    CACHE_SIZE = 1024 * 1024
    HOST_CACHE_SIZE = CACHE_SIZE * 2
    NUM_LAYERS = 32

    _cuda_cache = torch.randn(
        (2, NUM_LAYERS, CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        device="cuda",
    )
    _host_cache = create_pin_tensor(
        (2, NUM_LAYERS, HOST_CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        numa=0,
    )
    ITEM_BYTES = _cuda_cache.element_size() * CACHE_ITEM_SIZE

    def _make_ptrs(tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [tensor[i].data_ptr() for i in range(NUM_LAYERS)],
            dtype=torch.uint64,
            device="cuda",
        )

    cuda_k_ptrs = _make_ptrs(_cuda_cache[0])
    cuda_v_ptrs = _make_ptrs(_cuda_cache[1])
    host_k_ptrs = _make_ptrs(_host_cache[0])
    host_v_ptrs = _make_ptrs(_host_cache[1])

    # test PCIe contiguous performance first
    if need_warmup:
        for size in [2**n for n in range(1, 21)]:
            assert size <= CACHE_SIZE
            MEM = size * ITEM_BYTES
            MEM_K = MEM // 1024
            if MEM_K == 0:
                continue
            if MEM_K < 1024:
                MEM_STR = f"{MEM_K:4d}KB"
            else:
                MEM_STR = f"{MEM_K // 1024:4d}MB"
            dur = perf_cuda(
                lambda: _host_cache[0, :size].copy_(_cuda_cache[0, :size], non_blocking=True)
            )
            bandwidth = MEM / (dur * 1e6)
            logger.info(f"PCIe D -> H | {MEM_STR} | Bandwidth: {bandwidth:6.2f} GB/s")
            dur = perf_cuda(
                lambda: _cuda_cache[0, :size].copy_(_host_cache[0, :size], non_blocking=True)
            )
            bandwidth = MEM / (dur * 1e6)
            logger.info(f"PCIe H -> D | {MEM_STR} | Bandwidth: {bandwidth:6.2f} GB/s")

    our_times = {
        "H->D": [],
        "D->H": [],
    }
    ref_times = {
        "H->D": [],
        "D->H": [],
    }

    BS_RANGE = [2**n for n in range(5, 17)]
    logger.info("=" * 60)
    logger.info("Start HiCache kernel performance test...")
    for bs in BS_RANGE:
        indices_dst = torch.randperm(CACHE_SIZE, dtype=torch.int64, device="cuda")[:bs] - 1
        indices_src = torch.randperm(HOST_CACHE_SIZE, dtype=torch.int64, device="cuda")[:bs] - 1
        indices_dst = indices_dst.sort().values
        indices_src = indices_src.sort().values
        MEM = bs * 2 * ITEM_BYTES * NUM_LAYERS

        t_ref, t_our = compare_memory_kernel_perf(
            our_impl=lambda: transfer_hicache_all_layer(
                k_ptr_dst=cuda_k_ptrs,
                v_ptr_dst=cuda_v_ptrs,
                indices_dst=indices_dst,
                k_ptr_src=host_k_ptrs,
                v_ptr_src=host_v_ptrs,
                indices_src=indices_src,
                block_quota=BLOCK_QUOTA,
                kv_cache_dst_stride_bytes=ITEM_BYTES,
                kv_cache_src_stride_bytes=ITEM_BYTES,
            ),
            baseline=lambda: ref_hicache_impl(
                k_ptr_dst=cuda_k_ptrs,
                v_ptr_dst=cuda_v_ptrs,
                indices_dst=indices_dst,
                k_ptr_src=host_k_ptrs,
                v_ptr_src=host_v_ptrs,
                indices_src=indices_src,
                item_bytes=ITEM_BYTES,
                block_quota=BLOCK_QUOTA,
                num_layers=NUM_LAYERS,
            ),
            memory_footprint=MEM,
            need_latency=False,
            description=f"H->D bs={bs:6d} | ",
        )
        our_times["H->D"].append(t_our)
        ref_times["H->D"].append(t_ref)

        indices_dst, indices_src = indices_src, indices_dst  # swap for D->H
        t_ref, t_our = compare_memory_kernel_perf(
            our_impl=lambda: transfer_hicache_all_layer(
                k_ptr_dst=host_k_ptrs,
                v_ptr_dst=host_v_ptrs,
                indices_dst=indices_dst,
                k_ptr_src=cuda_k_ptrs,
                v_ptr_src=cuda_v_ptrs,
                indices_src=indices_src,
                block_quota=BLOCK_QUOTA,
                kv_cache_dst_stride_bytes=ITEM_BYTES,
                kv_cache_src_stride_bytes=ITEM_BYTES,
            ),
            baseline=lambda: ref_hicache_impl(
                k_ptr_dst=host_k_ptrs,
                v_ptr_dst=host_v_ptrs,
                indices_dst=indices_dst,
                k_ptr_src=cuda_k_ptrs,
                v_ptr_src=cuda_v_ptrs,
                indices_src=indices_src,
                item_bytes=ITEM_BYTES,
                block_quota=BLOCK_QUOTA,
                num_layers=NUM_LAYERS,
            ),
            memory_footprint=MEM,
            need_latency=False,
            description=f"D->H bs={bs:6d} | ",
        )
        our_times["D->H"].append(t_our)
        ref_times["D->H"].append(t_ref)

    if not need_plot:
        return

    # plot the results all in one figure
    LABELS = ["Our H->D", "Ref H->D", "Our D->H", "Ref D->H"]
    COLORS = ["#c31e23", "#ff7c7e", "#0d7d87", "#99c6cc"]
    DATA = [our_times["H->D"], ref_times["H->D"], our_times["D->H"], ref_times["D->H"]]
    MARKERS = ["o", "s", "^", "*"]
    plt.figure(figsize=(8, 5))
    for label, color, data, marker in zip(LABELS, COLORS, DATA, MARKERS):
        plt.plot(BS_RANGE, data, label=label, color=color, marker=marker, linewidth=2, markersize=6)
    plt.xscale("log", base=2)
    plt.xlim(min(BS_RANGE) * 0.9, max(BS_RANGE) * 1.1)
    plt.ylim(0, 64)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Batch Size (log scale)", fontsize=14)
    plt.ylabel("Achieved PCIe Bandwidth (GB/s)", fontsize=14)
    plt.title(
        f"HiCache Kernel Performance (Item={ITEM_BYTES}B, Quota={BLOCK_QUOTA} Blocks)", fontsize=16
    )
    plt.legend(fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/hicache_all.png", dpi=300)
    plt.close()


@call_if_main()
def main():
    need_warmup = False
    for block_quota in [2, 3, 4]:
        for cache_item_size in [128, 256, 512, 1024]:
            args = HicacheBenchArgs(
                cache_item_size=cache_item_size,
                dtype=torch.float16,
                block_quota=block_quota,
            )
            test_hicache_kernel(args, need_warmup=need_warmup)
            need_warmup = False  # only need to warmup once
