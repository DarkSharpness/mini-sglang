from __future__ import annotations
import os

import torch
import matplotlib.pyplot as plt

from minisgl.benchmark.perf import compare_memory_kernel_perf
from minisgl.kernel_v2 import transfer_hicache
from minisgl.utils import call_if_main


def ref_hicache_impl(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_bytes: int,
    block_quota: int,
) -> None:
    from sgl_kernel import transfer_kv_per_layer

    transfer_kv_per_layer(
        src_k=k_cache_src,
        src_v=v_cache_src,
        dst_k=k_cache_dst,
        dst_v=v_cache_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=item_bytes,
        block_quota=block_quota,
    )


@call_if_main()
@torch.inference_mode()
def test_hicache_kernel():
    CACHE_ITEM_SIZE = 1024
    DTYPE = torch.float16
    CACHE_SIZE = 1024 * 1024
    HOST_CACHE_SIZE = CACHE_SIZE * 2
    BLOCK_QUOTA = 2

    cuda_cache = torch.empty(
        (2, CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        device="cuda",
    )
    host_cache = torch.empty(
        (2, HOST_CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        device="cpu",
        pin_memory=True,
    )

    ITEM_BYTES = cuda_cache.element_size() * CACHE_ITEM_SIZE
    our_times = {
        "H->D": [],
        "D->H": [],
    }
    ref_times = {
        "H->D": [],
        "D->H": [],
    }

    BS_RANGE = [2**n for n in range(5, 18)]
    for bs in BS_RANGE:
        indices_dst = torch.randperm(CACHE_SIZE, dtype=torch.int64, device="cuda")[:bs] - 1
        indices_src = torch.randperm(HOST_CACHE_SIZE, dtype=torch.int64, device="cuda")[:bs] - 1
        MEM = bs * 2 * ITEM_BYTES

        t_ref, t_our = compare_memory_kernel_perf(
            our_impl=lambda: transfer_hicache(
                k_cache_dst=cuda_cache[0],
                v_cache_dst=cuda_cache[1],
                indices_dst=indices_dst,
                k_cache_src=host_cache[0],
                v_cache_src=host_cache[1],
                indices_src=indices_src,
                block_quota=BLOCK_QUOTA,
            ),
            baseline=lambda: ref_hicache_impl(
                k_cache_dst=cuda_cache[0],
                v_cache_dst=cuda_cache[1],
                indices_dst=indices_dst,
                k_cache_src=host_cache[0],
                v_cache_src=host_cache[1],
                indices_src=indices_src,
                item_bytes=ITEM_BYTES,
                block_quota=BLOCK_QUOTA,
            ),
            memory_footprint=MEM,
            description=f"H->D bs={bs:6d} | ",
        )
        our_times["H->D"].append(t_our)
        ref_times["H->D"].append(t_ref)

        indices_dst, indices_src = indices_src, indices_dst  # swap for D->H
        t_ref, t_our = compare_memory_kernel_perf(
            our_impl=lambda: transfer_hicache(
                k_cache_dst=host_cache[0],
                v_cache_dst=host_cache[1],
                indices_dst=indices_dst,
                k_cache_src=cuda_cache[0],
                v_cache_src=cuda_cache[1],
                indices_src=indices_src,
                block_quota=BLOCK_QUOTA,
            ),
            baseline=lambda: ref_hicache_impl(
                k_cache_dst=host_cache[0],
                v_cache_dst=host_cache[1],
                indices_dst=indices_dst,
                k_cache_src=cuda_cache[0],
                v_cache_src=cuda_cache[1],
                indices_src=indices_src,
                item_bytes=ITEM_BYTES,
                block_quota=BLOCK_QUOTA,
            ),
            memory_footprint=MEM,
            description=f"D->H bs={bs:6d} | ",
        )
        our_times["D->H"].append(t_our)
        ref_times["D->H"].append(t_ref)

    # plot the results all in one figure
    LABELS = ["Our H->D", "Ref H->D", "Our D->H", "Ref D->H"]
    COLORS = ["#c31e23", "#ff5a5e", "#0d7d87", "#99c6cc"]
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
    plt.savefig(f"figures/hicache_{ITEM_BYTES}_{BLOCK_QUOTA}.png", dpi=300)
    print(f"Figure saved to figures/hicache_{ITEM_BYTES}_{BLOCK_QUOTA}.png")
