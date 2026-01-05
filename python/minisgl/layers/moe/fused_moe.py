import triton
import torch
import triton.language as tl
from typing import Dict, Any, Optional, List
from sgl_kernel import gelu_and_mul, silu_and_mul
import functools
from typing import Tuple
from minisgl.layers.moe.topk import select_experts
from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size




def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y





@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)

def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda




def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    cumsum_buffer = torch.empty(
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
    )

    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad

@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(
                input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
            )
            accumulator += tmp
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


def moe_sum_reduce_triton(
    input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 8

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return

def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    is_marlin: bool,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    # A heuristic: fused marlin works faster with this config for small M
    if M <= E or (is_marlin and M <= 32):
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
    is_marlin: bool = False,
):
    E, _, N = w2_shape


    config = get_default_config(
        M, E, N, w1_shape[2], top_k, is_marlin
    )
    return config




@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,


    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )


    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # We accumulate along the K dimension.

        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    padded_size = 0
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=even_Ks,
        **config,

    )

def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):


    padded_size = 0
    assert (
        hidden_states.shape[1] == w1.shape[2] - padded_size
        ), "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    
    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk_ids.shape[1], w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)
    
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.shape[1]
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )


        invoke_fused_moe_kernel(
            curr_hidden_states,
            w1,
            intermediate_cache1,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
        )

        if activation == "silu":

            silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

        elif activation == "gelu":

            gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            (
                intermediate_cache3
                if not no_combine and topk_ids.shape[1] != 1
                else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
            ),
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
        )


        

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass

        if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
            pass  # we write directly into out_hidden_states
        elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
            torch.add(
                intermediate_cache3[:, 0],
                intermediate_cache3[:, 1],
                out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
            ).squeeze(dim=1)
        else:
            # According to micro benchmark results, torch.compile can get better performance for small token.
            if tokens_in_chunk <= 32:

                moe_sum_reduce_torch_compile(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    routed_scaling_factor,
                )
            else:
                moe_sum_reduce_triton(
                    intermediate_cache3,
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    routed_scaling_factor,
                )
    return out_hidden_states




def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> None:

    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        True,
        activation,
        apply_router_weight_on_input,
        False,
        routed_scaling_factor,
    )

def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,
        activation,
        apply_router_weight_on_input,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
    )




def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):

    if inplace:
        assert not no_combine, "no combine + inplace makes no sense"
        inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            routed_scaling_factor,
        )
        return hidden_states
    else:
        return outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )

        

def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    no_combine: bool = False,
) -> torch.Tensor:

    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=gating_output,
        top_k=topk,
        renormalize=renormalize,
    )
    return fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        no_combine=no_combine,

    )


    



    













