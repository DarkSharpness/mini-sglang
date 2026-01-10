from typing import Any, Optional

import torch

try:
    from sgl_kernel import topk_softmax
except ImportError:

    def topk_softmax(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("sgl_kernel not found.")


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    topk_softmax(
        topk_weights,
        topk_ids,
        gating_output.float(),
        renormalize,
    )

    return _fused_topk_postprocess(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        renormalize=renormalize,
        num_token_non_padded=num_token_non_padded,
    )


def _mask_topk_ids_padded_region(
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    if num_token_non_padded is None:
        return

    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    topk_ids[indices >= num_token_non_padded, :] = -1


def _fused_topk_postprocess(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor],
):
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    num_token_non_padded: Optional[torch.Tensor] = None,
    renormalize: bool = True,
):
    topk_weights, topk_ids = fused_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=top_k,
        renormalize=renormalize,
        num_token_non_padded=num_token_non_padded,
    )
    return topk_weights, topk_ids
