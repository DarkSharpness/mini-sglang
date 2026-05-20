from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Set, Tuple

import safetensors
import torch
from minisgl.distributed import get_ep_info, get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}

_EXPERT_KEY_RE = re.compile(r"experts\.(\d+)\.")
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(
    key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int, ep_size: int = 1
) -> torch.Tensor:
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if _EXPERT_KEY_RE.search(key) and ep_size > 1:
        return value
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _build_ep_skip_set(files: list[str], ep_size: int) -> Tuple[Set[str], int]:
    ep_info = get_ep_info()
    max_idx = -1
    expert_keys: list[tuple[str, int]] = []
    for file in files:
        with safetensors.safe_open(file, framework="pt") as f:
            for name in f.keys():
                m = _EXPERT_KEY_RE.search(name)
                if m:
                    idx = int(m.group(1))
                    max_idx = max(max_idx, idx)
                    expert_keys.append((name, idx))
    if max_idx < 0:
        return set(), 0

    num_local = (max_idx + 1) // ep_size
    local_start = ep_info.rank * num_local
    skip = {name for name, idx in expert_keys if not (local_start <= idx < local_start + num_local)}
    return skip, local_start


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(
    model_path: str, device: torch.device, ep_size: int = 1
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    skip_keys: Set[str] = set()
    ep_local_start = 0
    if ep_size > 1:
        skip_keys, ep_local_start = _build_ep_skip_set(files, ep_size)

    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    num_local_experts = config.num_experts // ep_size if ep_size > 1 else config.num_experts

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                if name in skip_keys:
                    continue
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue

                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                tensor = _shard_tensor(
                    name, raw, tp_info.rank, tp_info.size, config.num_kv_heads, ep_size
                )
                del raw

                out_name = name
                if ep_size > 1:
                    m = _EXPERT_KEY_RE.search(name)
                    if m:
                        local_idx = int(m.group(1)) - ep_local_start
                        out_name = name[: m.start(1)] + str(local_idx) + name[m.end(1) :]

                if (info := _get_merge_info(out_name)) is None:
                    out = (out_name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != num_local_experts:
                        continue
                    experts = [slots[idx] for idx in range(num_local_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
