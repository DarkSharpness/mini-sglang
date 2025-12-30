from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import safetensors
import torch
from huggingface_hub import snapshot_download
from minisgl.distributed import get_tp_info
from minisgl.quantization import QuantizationConfig
from minisgl.quantization.quantize import quantize_weight
from minisgl.utils import divide_up
from tqdm.asyncio import tqdm


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]
    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value
    return shard_state_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if key.count(".q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif key.count(".gate_proj"):
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            continue
        else:
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def _quantize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    quant_config: Optional[QuantizationConfig],
) -> Dict[str, torch.Tensor]:
    """Quantize weights in state dict and add scale/zero_point metadata.

    Args:
        state_dict: State dict with FP weights (after TP sharding)
        quant_config: Quantization configuration

    Returns:
        State dict with quantized weights and metadata
    """
    if quant_config is None or not quant_config.enabled:
        return state_dict

    quantized_state_dict: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        # Only quantize weight tensors (skip biases, norms, embeddings for now)
        if "weight" in key and value.dim() >= 2:
            # Skip layer norms and embedding layers (keep high precision)
            if any(skip in key for skip in ["norm", "embed_tokens"]):
                quantized_state_dict[key] = value
                continue

            # Quantize the weight
            result = quantize_weight(value, quant_config)
            if result is not None:
                quantized_w, scale, zero_point = result
                quantized_state_dict[key] = quantized_w
                quantized_state_dict[f"{key}.scale"] = scale
                quantized_state_dict[f"{key}.zero_point"] = zero_point
            else:
                quantized_state_dict[key] = value
        else:
            quantized_state_dict[key] = value

    return quantized_state_dict


def load_hf_weight(
    model_path: str,
    device: torch.device,
    quant_config: Optional[QuantizationConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Load HuggingFace weights with optional quantization.

    Args:
        model_path: Path to model (local dir or HF repo)
        device: Target device
        quant_config: Optional quantization configuration

    Returns:
        State dict with loaded (and optionally quantized) weights
    """
    if os.path.isdir(model_path):
        hf_folder = model_path
    else:
        try:
            hf_folder = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors"],
                tqdm_class=DisabledTqdm,
            )
        except Exception:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
            )

    # find the all *.pt files in the hf_folder
    files = glob.glob(f"{hf_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    # Apply TP sharding first (before quantization)
    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict)

    # Move to device
    state_dict = {k: v.to(device) for k, v in state_dict.items()}

    # Merge Q/K/V and gate/up projections
    state_dict = _merge_state_dict(state_dict)

    # Apply quantization AFTER sharding and merging
    # This ensures scales/zero_points match the sharded dimensions
    state_dict = _quantize_state_dict(state_dict, quant_config)

    return state_dict
