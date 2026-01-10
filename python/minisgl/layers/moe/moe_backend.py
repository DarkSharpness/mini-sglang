from minisgl.layers.moe.fused_moe.layer import FusedMoE
from minisgl.models import ModelConfig

def get_moe_backend(moe_backend: str, config: ModelConfig, prefix: str = ""):
    if moe_backend == "fused_moe":
        return FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            prefix=prefix,)
    else:
        raise ValueError(f"Unsupported MoE backend: {moe_backend}")