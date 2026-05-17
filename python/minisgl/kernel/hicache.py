from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_aot, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

DEFAULT_BLOCK_QUOTA = 2


@functools.cache
def _jit_hicache_module(*, element_size: int, unroll: int, block_quota: int) -> Module:
    args = make_cpp_args(
        element_size,
        unroll,
        block_quota,
        1024,  # num_threads, can be tuned for performance
    )
    return load_jit(
        "hicache",
        *args,
        cuda_files=["hicache.cu"],
        cuda_wrappers=[
            ("launch_one", f"&HiCacheKernel<{args}>::run_one"),
            ("launch_all", f"&HiCacheKernel<{args}>::run_all"),
        ],
    )


@functools.cache
def _jit_numa_module() -> Module:
    return load_aot("numa", cuda_files=["numa.cu"], extra_ldflags=["-lnuma"])


@functools.cache
def probe_numa_node() -> int:
    import subprocess

    import torch

    gpu_index = torch.cuda.current_device()
    target_gpu = f"GPU{gpu_index}"
    topo_output = subprocess.check_output(
        ["nvidia-smi", "topo", "-m"],
        text=True,
    )
    topo_output = topo_output.splitlines()
    # Find header to determine NUMA Affinity column index
    header = topo_output[0].split()
    try:
        numa_col_idx = header.index("NUMA")  # header contains: NUMA Affinity
    except ValueError:
        raise RuntimeError("NUMA Affinity column not found in topo output")
    # Data rows start after header
    for line in topo_output[1:]:
        cols = line.split()
        if not cols:
            continue
        if cols[0] == target_gpu:
            numa_value = cols[numa_col_idx]
            if numa_value == "N/A":
                raise RuntimeError("NUMA Affinity is N/A on this system")
            return int(numa_value)
    raise RuntimeError(f"{target_gpu} not found in topo output")


def _default_unroll(element_size: int) -> int:
    if element_size <= 512:
        return 4
    if element_size <= 1024:
        return 2
    # fallback: no unroll
    return 1


def transfer_hicache_one_layer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    element_dim = element_dim or k_cache_dst.size(-1)
    k_cache_src = k_cache_src.view(-1, element_dim)
    v_cache_src = v_cache_src.view(-1, element_dim)
    k_cache_dst = k_cache_dst.view(-1, element_dim)
    v_cache_dst = v_cache_dst.view(-1, element_dim)
    element_size = element_dim * k_cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
    )


def transfer_hicache_all_layer(
    k_ptr_dst: torch.Tensor,
    v_ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    kv_cache_src_stride_bytes: int,
    kv_cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    if element_size is None:  # assume both contiguous
        assert kv_cache_dst_stride_bytes == kv_cache_src_stride_bytes
        element_size = kv_cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_all(
        k_ptr_dst,
        v_ptr_dst,
        indices_dst,
        k_ptr_src,
        v_ptr_src,
        indices_src,
        kv_cache_src_stride_bytes,
        kv_cache_dst_stride_bytes,
    )


def allocate_host(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    import torch

    try:
        numa_node = probe_numa_node()
        module = _jit_numa_module()
    except Exception:
        return torch.empty(*shape, dtype=dtype, pin_memory=True)
    size_bytes = functools.reduce(lambda x, y: x * y, shape) * dtype.itemsize
    result = torch.from_dlpack(module.allocate_numa(size_bytes, numa_node))
    assert result.is_pinned(), "Expected pinned memory from NUMA allocator"
    return result.view(dtype).view(*shape)
