import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info, try_get_ep_info
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        ep_info = try_get_ep_info()

        self.tp_size = tp_info.size
        self.ep_size = ep_info.size if ep_info is not None else 1
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input

        if self.ep_size > 1:
            num_local_experts = div_even(num_experts, self.ep_size)
            self.gate_up_proj = torch.empty(num_local_experts, 2 * intermediate_size, hidden_size)
            self.down_proj = torch.empty(num_local_experts, hidden_size, intermediate_size)
        else:
            intermediate_per_tp = div_even(intermediate_size, tp_info.size)
            self.gate_up_proj = torch.empty(num_experts, 2 * intermediate_per_tp, hidden_size)
            self.down_proj = torch.empty(num_experts, hidden_size, intermediate_per_tp)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        if self.ep_size == 1 and self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
