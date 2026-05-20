from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_qkvzba_split_reshape_cat_contiguous_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)

    v_per_group: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK

    total_q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    total_k: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    total_v: tl.constexpr = NUM_HEADS_V * HEAD_V
    total_qkvz: tl.constexpr = total_q + total_k + total_v + total_v
    total_ba: tl.constexpr = NUM_HEADS_V * 2

    qkv_dim_t: tl.constexpr = total_q + total_k + total_v

    blk_q_ptr = mixed_qkvz + i_bs * total_qkvz + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    blk_k_ptr = (
        mixed_qkvz
        + i_bs * total_qkvz
        + total_q
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_ptr = (
        mixed_qkvz
        + i_bs * total_qkvz
        + total_q
        + total_k
        + i_qk * v_per_group * HEAD_V
        + tl.arange(0, v_per_group * HEAD_V)
    )
    blk_z_ptr = (
        mixed_qkvz
        + i_bs * total_qkvz
        + total_q
        + total_k
        + total_v
        + i_qk * v_per_group * HEAD_V
        + tl.arange(0, v_per_group * HEAD_V)
    )

    blk_q_st_ptr = mixed_qkv + i_bs * qkv_dim_t + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * qkv_dim_t
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * qkv_dim_t
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * v_per_group * HEAD_V
        + tl.arange(0, v_per_group * HEAD_V)
    )
    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * v_per_group * HEAD_V
        + tl.arange(0, v_per_group * HEAD_V)
    )

    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))

    for i in tl.static_range(v_per_group):
        blk_b_ptr = mixed_ba + i_bs * total_ba + i_qk * v_per_group + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * v_per_group + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))

    for i in tl.static_range(v_per_group):
        blk_a_ptr = mixed_ba + i_bs * total_ba + NUM_HEADS_V + i_qk * v_per_group + i
        blk_a_st_ptr = a + i_bs * NUM_HEADS_V + i_qk * v_per_group + i
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = mixed_qkvz.shape[0]
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        (batch, qkv_dim_t),
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        (batch, num_heads_v, head_v),
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        (batch, num_heads_v),
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    grid = (batch, num_heads_qk)
    fused_qkvzba_split_reshape_cat_contiguous_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


__all__ = ["fused_qkvzba_split_reshape_cat_contiguous"]
