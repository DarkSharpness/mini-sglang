from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    out,
    state,
    state_indices,
    scale,
    stride_mixed_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    slot = tl.load(state_indices + i_n * stride_indices).to(tl.int64)

    p_o = out + (i_n * HV + i_hv) * V + o_v
    if slot < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_state = state + slot * stride_state_slot
    p_state = p_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    h = tl.load(p_state, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v

    q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    q = q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_val = tl.load(dt_bias + i_hv).to(tl.float32)

    x = a_val + dt_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g = -tl.exp(A_val) * softplus_x
    beta = tl.sigmoid(b_val)

    h = h * tl.exp(g)
    v = v - tl.sum(h * k[None, :], axis=1)
    v = v * beta
    h = h + v[:, None] * k[None, :]
    o = tl.sum(h * q[None, :], axis=1)

    tl.store(p_o, o.to(p_o.dtype.element_ty), mask=mask_v)
    tl.store(p_state, h.to(p_state.dtype.element_ty), mask=mask_h)


def packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    scale: float,
) -> torch.Tensor:
    if mixed_qkv.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("packed_decode expects [B, D] mixed_qkv and [B, HV] a/b.")
    if state.ndim != 4:
        raise ValueError("state must be [num_slots, HV, V, K].")
    if state_indices.ndim != 1:
        raise ValueError("state_indices must be [B].")
    if num_q_heads <= 0 or num_v_heads <= 0:
        raise ValueError("num_q_heads and num_v_heads must be positive.")

    B = mixed_qkv.shape[0]
    H = num_q_heads
    HV = num_v_heads
    K = head_k_dim
    V = head_v_dim
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)

    out = torch.empty((B, 1, HV, V), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    grid = (triton.cdiv(V, BV), B * HV)
    _packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        out=out,
        state=state,
        state_indices=state_indices,
        scale=scale,
        stride_mixed_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_state_slot=state.stride(0),
        stride_indices=state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=1,
        num_stages=3,
    )
    return out[:, 0]


__all__ = ["packed_decode"]
