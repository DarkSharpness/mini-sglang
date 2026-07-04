from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


# --------------------------------------------------------------- helpers
def _cos_sin_full_from_cache(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split the FlashInfer-style ``[cos | sin]`` cache and duplicate each
    half to reach the full ``rotary_dim`` width used by NeoX rotate_half.

    The cache is laid out as ``cat([cos, sin], dim=-1)`` with shape
    ``(max_position, rotary_dim)``, i.e. the first ``rotary_dim // 2`` columns
    hold ``cos`` and the second half holds ``sin``. Returns tensors of shape
    ``(T, rotary_dim)`` on the cache's device, with the cache's dtype.
    """
    half = rotary_dim // 2
    # Direct index — no dtype coercion, no CPU copy. Positions carry the
    # dtype the caller chose (int32 / int64); torch's advanced-indexing
    # accepts both without materialising a converted copy.
    selected = cos_sin_cache[positions]                  # (T, rotary_dim)
    cos_half = selected[..., :half]                      # (T, half)
    sin_half = selected[..., half:]                      # (T, half)
    cos_full = torch.cat((cos_half, cos_half), dim=-1)   # (T, rotary_dim)
    sin_full = torch.cat((sin_half, sin_half), dim=-1)   # (T, rotary_dim)
    return cos_full, sin_full


def _rope_neox_cpu(
    x: torch.Tensor, cos_full: torch.Tensor, sin_full: torch.Tensor
) -> torch.Tensor:
    """NeoX rotate_half applied in fp32; result cast back to ``x.dtype``.

    ``cos_full`` / ``sin_full`` broadcast over the head dimension and are
    expected to be fp32 with shape ``(T, 1, rotary_dim)``.
    """
    x_f = x.float()
    half = x_f.shape[-1] // 2
    x1 = x_f[..., :half]
    x2 = x_f[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x_f * cos_full + rotated * sin_full).to(x.dtype)


def _load_flashinfer_apply_rope():
    try:
        from flashinfer import apply_rope_with_cos_sin_cache_inplace
    except ImportError as exc:
        raise RuntimeError(
            "CUDA RoPE requires flashinfer to be importable; install "
            "flashinfer on the host to use RotaryEmbedding on CUDA tensors."
        ) from exc
    return apply_rope_with_cos_sin_cache_inplace


def _load_torch_npu():
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "NPU RoPE requires torch_npu to be importable; install "
            "torch_npu on the host to use RotaryEmbedding on NPU tensors."
        ) from exc
    return torch_npu


# ---------------------------------------------------------- RotaryEmbedding
class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        # No device-library imports at construction time — dispatch happens on
        # the first forward, keyed off ``query.device.type``.
        super().__init__()
        self.head_size = head_size
        # ``rotary_dim`` is asserted equal to ``head_size`` in the current
        # model zoo; kept as a distinct attribute so the NPU/CPU helpers can
        # split the cache along the rotary axis without recomputing ``//2``.
        self.rotary_dim = rotary_dim
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save. FlashInfer stores (cos | sin) along the
        # last dim; the NPU/CPU paths re-split and duplicate each half to
        # reach the full rotary_dim used by NeoX rotate_half.
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device_type = query.device.type

        if device_type == "cuda":
            apply_rope = _load_flashinfer_apply_rope()
            # FlashInfer semantics: rotates ``query`` and ``key`` in-place and
            # returns ``None``. We return the same objects so callers see the
            # identity contract that downstream attention relies on.
            apply_rope(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self._cos_sin_cache,
            )
            return query, key

        if device_type == "npu":
            torch_npu = _load_torch_npu()
            cos_full, sin_full = _cos_sin_full_from_cache(
                self._cos_sin_cache, positions, self.rotary_dim
            )
            cos_full = cos_full.to(dtype=query.dtype)
            sin_full = sin_full.to(dtype=query.dtype)
            T = query.shape[0]
            Hq = query.shape[1]
            Hk = key.shape[1]
            D = self.head_size
            q_4d = query.view(1, T, Hq, D)
            k_4d = key.view(1, T, Hk, D)
            cos_4d = cos_full.view(1, T, 1, D)
            sin_4d = sin_full.view(1, T, 1, D)
            # npu_rotary_mul with rotary_mode="half" is NeoX rotate_half; it
            # returns a fresh allocation, so inputs are untouched (no copy_,
            # no clone, no contiguous).
            q_rot = torch_npu.npu_rotary_mul(q_4d, cos_4d, sin_4d, rotary_mode="half")
            k_rot = torch_npu.npu_rotary_mul(k_4d, cos_4d, sin_4d, rotary_mode="half")
            return q_rot.squeeze(0), k_rot.squeeze(0)

        # CPU path — pure PyTorch NeoX rotate_half with fp32 mid-calc.
        cos_full, sin_full = _cos_sin_full_from_cache(
            self._cos_sin_cache, positions, self.rotary_dim
        )
        cos_b = cos_full.unsqueeze(1)                    # (T, 1, D)
        sin_b = sin_full.unsqueeze(1)                    # (T, 1, D)
        q_out = _rope_neox_cpu(query, cos_b, sin_b)
        k_out = _rope_neox_cpu(key, cos_b, sin_b)
        return q_out, k_out


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
