from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from minisgl.core import Req, get_global_ctx
from minisgl.kvcache import BaseCacheHandle
from minisgl.kernel.triton.causal_conv1d import (
    PAD_SLOT_ID,
    causal_conv1d_update as triton_causal_conv1d_update,
)
from minisgl.kernel.triton.gdn_decode import packed_decode as gdn_packed_decode

from .base import BaseAttnBackend


@dataclass
class _LayerRuntime:
    conv_cache: torch.Tensor
    ssm_cache: torch.Tensor


@dataclass
class _LayerStateSnapshot:
    conv_cache: torch.Tensor
    ssm_cache: torch.Tensor


class GDNAttnBackend:
    """Backend executor for GDN linear attention path."""

    def __init__(self) -> None:
        self._runtime: Dict[int, _LayerRuntime] = {}
        self._capture_active_bs: int | None = None
        self._capture_state_indices_i32: Dict[int, torch.Tensor] = {}
        self._prefix_state_cache: weakref.WeakKeyDictionary[
            object, Dict[int, _LayerStateSnapshot]
        ] = weakref.WeakKeyDictionary()

    def _get_prefix_node(self, handle: BaseCacheHandle) -> object | None:
        return getattr(handle, "node", None)

    def has_prefix_cache_state(self, handle: BaseCacheHandle) -> bool:
        node = self._get_prefix_node(handle)
        if node is None:
            return True
        return node in self._prefix_state_cache

    def on_prefix_cache_store(self, req: Req, handle: BaseCacheHandle) -> None:
        node = self._get_prefix_node(handle)
        if node is None:
            return
        slot = req.table_idx
        node_states: Dict[int, _LayerStateSnapshot] = {}
        for layer_id, rt in self._runtime.items():
            if slot < 0 or slot >= rt.ssm_cache.shape[0]:
                continue
            node_states[layer_id] = _LayerStateSnapshot(
                conv_cache=rt.conv_cache[slot].clone(),
                ssm_cache=rt.ssm_cache[slot].clone(),
            )
        if node_states:
            self._prefix_state_cache[node] = node_states

    def on_prefix_cache_match(self, handle: BaseCacheHandle, slot: int) -> None:
        node = self._get_prefix_node(handle)
        if node is None:
            return
        node_states = self._prefix_state_cache.get(node)
        if not node_states:
            return
        for layer_id, state in node_states.items():
            rt = self._runtime.get(layer_id)
            if rt is None or slot < 0 or slot >= rt.ssm_cache.shape[0]:
                continue
            if rt.conv_cache.shape[-1] != state.conv_cache.shape[-1]:
                continue
            rt.conv_cache[slot].copy_(state.conv_cache.to(device=rt.conv_cache.device))
            rt.ssm_cache[slot].copy_(
                state.ssm_cache.to(device=rt.ssm_cache.device, dtype=rt.ssm_cache.dtype)
            )

    def _ensure_runtime(self, layer, x: torch.Tensor) -> _LayerRuntime:
        ctx = get_global_ctx()
        num_slots = ctx.page_table.shape[0]

        conv_weights = layer.conv_weights
        if not isinstance(conv_weights, torch.Tensor):
            raise ValueError("conv_weights must be a Tensor in RadixLinearAttention.")
        conv_kernel = conv_weights.shape[-1]
        hist_len = max(0, conv_kernel - 1)

        rt = self._runtime.get(layer.layer_id)
        need_realloc = (
            rt is None
            or rt.conv_cache.shape[0] != num_slots
            or rt.conv_cache.shape[-1] != hist_len
            or rt.conv_cache.device != x.device
            or rt.conv_cache.dtype != x.dtype
        )
        if need_realloc:
            rt = _LayerRuntime(
                conv_cache=torch.zeros(
                    num_slots,
                    layer.q_dim + layer.k_dim + layer.v_dim,
                    hist_len,
                    dtype=x.dtype,
                    device=x.device,
                ),
                ssm_cache=torch.zeros(
                    num_slots,
                    layer.num_v_heads,
                    layer.head_v_dim,
                    layer.head_k_dim,
                    dtype=torch.float32,
                    device=x.device,
                ),
            )
            self._runtime[layer.layer_id] = rt
            return rt

        if rt.ssm_cache.device != x.device:
            rt.ssm_cache = rt.ssm_cache.to(device=x.device)
        return rt

    def _clear_slot_in_runtime(self, rt: _LayerRuntime, slot: int) -> None:
        if slot < 0 or slot >= rt.ssm_cache.shape[0]:
            return
        if rt.conv_cache.shape[-1] > 0:
            rt.conv_cache[slot].zero_()
        rt.ssm_cache[slot].zero_()

    def on_table_slot_allocated(self, slot: int) -> None:
        for rt in self._runtime.values():
            self._clear_slot_in_runtime(rt, slot)

    def _get_decode_state_indices(
        self,
        reqs: List[Req],
        device: torch.device,
        forward_batch,
    ) -> torch.Tensor:
        del forward_batch
        # During CUDA graph capture we must avoid creating new CUDA tensors.
        if self._capture_active_bs == len(reqs):
            state_i32 = self._capture_state_indices_i32.get(len(reqs))
            if state_i32 is None:
                raise RuntimeError(
                    "Missing cached decode state indices during capture. "
                    "prepare_for_capture() must run before graph capture."
                )
            return state_i32
        return torch.tensor([req.table_idx for req in reqs], dtype=torch.int32, device=device)

    def init_capture_graph(self, bs_list: List[int]) -> None:
        self._capture_active_bs = None
        self._capture_state_indices_i32 = {}

    def prepare_for_capture(self, batch) -> None:
        self._capture_active_bs = batch.size
        bs = batch.size
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        device = get_global_ctx().page_table.device
        state_i32 = self._capture_state_indices_i32.get(bs)
        if state_i32 is None or state_i32.device != device:
            state_i32 = torch.empty(bs, dtype=torch.int32, device=device)
            self._capture_state_indices_i32[bs] = state_i32
        for i, req in enumerate(reqs):
            state_i32[i] = req.table_idx

    def prepare_for_replay(self, batch) -> None:
        bs = batch.padded_size
        state_i32 = self._capture_state_indices_i32.get(bs)
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        if state_i32 is None:
            state_i32 = torch.empty(bs, dtype=torch.int32, device=get_global_ctx().page_table.device)
            self._capture_state_indices_i32[bs] = state_i32
        for i, req in enumerate(reqs):
            state_i32[i] = req.table_idx
        self._capture_active_bs = None

    def _apply_conv(
        self,
        token_states: torch.Tensor,
        slot: int,
        conv_weight: torch.Tensor,
        rt: _LayerRuntime,
    ) -> torch.Tensor:
        if conv_weight.shape[-1] <= 1:
            return token_states

        conv_hist = rt.conv_cache[slot]
        hist_len = conv_hist.shape[-1]

        if token_states.shape[0] == 1:
            full = torch.cat([conv_hist, token_states[0].unsqueeze(-1)], dim=-1)
            out = (full * conv_weight.squeeze(1)).sum(dim=-1)
            out = F.silu(out)
            if hist_len > 0:
                conv_hist[:, :-1] = conv_hist[:, 1:]
                conv_hist[:, -1] = token_states[0]
            return out.unsqueeze(0)

        inp = torch.cat([conv_hist, token_states.transpose(0, 1)], dim=-1).unsqueeze(0)
        out = F.conv1d(inp, conv_weight, bias=None, groups=conv_weight.shape[0])
        out = out.squeeze(0).transpose(0, 1)
        out = F.silu(out)

        if hist_len > 0:
            if token_states.shape[0] >= hist_len:
                conv_hist.copy_(token_states[-hist_len:].transpose(0, 1))
            else:
                conv_hist.copy_(
                    torch.cat(
                        [conv_hist[:, token_states.shape[0] :], token_states.transpose(0, 1)],
                        dim=1,
                    )
                )
        return out

    def _apply_conv_batch(
        self,
        mixed_qkv: torch.Tensor,
        reqs: List[Req],
        conv_weight: torch.Tensor,
        rt: _LayerRuntime,
    ) -> torch.Tensor:
        conv_qkv = torch.empty_like(mixed_qkv)
        offset = 0
        for req in reqs:
            length = req.extend_len
            if length == 0:
                continue
            seg = mixed_qkv[offset : offset + length]
            conv_qkv[offset : offset + length] = self._apply_conv(
                seg, req.table_idx, conv_weight, rt
            )
            offset += length
        return conv_qkv

    def _apply_conv_decode_batch(
        self,
        mixed_qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        rt: _LayerRuntime,
        state_indices_i32: torch.Tensor,
    ) -> torch.Tensor:
        if conv_weight.shape[-1] <= 1:
            return mixed_qkv

        weight_2d = conv_weight.squeeze(1) if conv_weight.ndim == 3 else conv_weight
        if not mixed_qkv.is_cuda:
            slots = state_indices_i32.to(dtype=torch.int64)
            conv_hist = rt.conv_cache.index_select(0, slots)
            full = torch.cat([conv_hist, mixed_qkv.unsqueeze(-1)], dim=-1)
            out = (full * weight_2d.unsqueeze(0)).sum(dim=-1)
            out = F.silu(out)
            hist_len = conv_hist.shape[-1]
            if hist_len > 0:
                if hist_len == 1:
                    updated_hist = mixed_qkv.unsqueeze(-1)
                else:
                    updated_hist = torch.cat(
                        [conv_hist[:, :, 1:], mixed_qkv.unsqueeze(-1)], dim=-1
                    )
                rt.conv_cache.index_copy_(0, slots, updated_hist)
            return out

        return triton_causal_conv1d_update(
            x=mixed_qkv,
            conv_state=rt.conv_cache,
            weight=weight_2d.contiguous(),
            bias=None,
            activation="silu",
            conv_state_indices=state_indices_i32,
            pad_slot_id=PAD_SLOT_ID,
        )

    def forward_decode(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer=None,
        forward_batch=None,
        save_kv_cache: bool = True,
        mixed_qkv: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del q, k, v, save_kv_cache, kwargs
        if layer is None or forward_batch is None:
            raise ValueError("layer and forward_batch are required for GDN decode.")
        if mixed_qkv is None or a is None or b is None:
            raise ValueError("mixed_qkv, a and b are required for GDN decode.")

        reqs = forward_batch.reqs
        if not reqs:
            return mixed_qkv.new_empty((1, 0, layer.num_v_heads, layer.head_v_dim))
        conv_weight = layer.conv_weights
        if not isinstance(conv_weight, torch.Tensor):
            raise ValueError("conv_weights must be a Tensor in RadixLinearAttention.")
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("A_log/dt_bias are required in RadixLinearAttention.")

        rt = self._ensure_runtime(layer, mixed_qkv)
        state_indices_i32 = self._get_decode_state_indices(reqs, mixed_qkv.device, forward_batch)
        conv_qkv = self._apply_conv_decode_batch(
            mixed_qkv,
            conv_weight,
            rt,
            state_indices_i32=state_indices_i32,
        )
        core = gdn_packed_decode(
            mixed_qkv=conv_qkv.contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            A_log=layer.A_log.contiguous(),
            dt_bias=layer.dt_bias.contiguous(),
            state=rt.ssm_cache,
            state_indices=state_indices_i32,
            num_q_heads=layer.num_q_heads,
            num_v_heads=layer.num_v_heads,
            head_k_dim=layer.head_k_dim,
            head_v_dim=layer.head_v_dim,
            scale=layer.head_k_dim**-0.5,
        )
        return core.unsqueeze(0)

    def forward_extend(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer=None,
        forward_batch=None,
        save_kv_cache: bool = True,
        mixed_qkv: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del q, k, v, save_kv_cache, kwargs
        if layer is None or forward_batch is None:
            raise ValueError("layer and forward_batch are required for GDN extend.")
        if mixed_qkv is None or a is None or b is None:
            raise ValueError("mixed_qkv, a and b are required for GDN extend.")

        reqs = forward_batch.reqs
        if not reqs:
            return mixed_qkv.new_empty((1, 0, layer.num_v_heads, layer.head_v_dim))

        conv_weight = layer.conv_weights
        if not isinstance(conv_weight, torch.Tensor):
            raise ValueError("conv_weights must be a Tensor in RadixLinearAttention.")
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("A_log/dt_bias are required in RadixLinearAttention.")

        rt = self._ensure_runtime(layer, mixed_qkv)
        conv_qkv = self._apply_conv_batch(mixed_qkv, reqs, conv_weight, rt)
        out = torch.empty(
            conv_qkv.shape[0],
            layer.num_v_heads,
            layer.head_v_dim,
            dtype=conv_qkv.dtype,
            device=conv_qkv.device,
        )
        req_layout: list[tuple[int, int, int]] = []
        max_extend_len = 0
        offset = 0
        for req in reqs:
            length = req.extend_len
            if length > 0:
                req_layout.append((req.table_idx, offset, length))
                max_extend_len = max(max_extend_len, length)
            offset += length

        for step in range(max_extend_len):
            token_indices = [start + step for _, start, length in req_layout if step < length]
            if not token_indices:
                continue
            indices = torch.tensor(token_indices, dtype=torch.int64, device=conv_qkv.device)
            state_slots = [slot for slot, _, length in req_layout if step < length]
            state_indices = torch.tensor(state_slots, dtype=torch.int32, device=conv_qkv.device)
            core_step = gdn_packed_decode(
                mixed_qkv=conv_qkv.index_select(0, indices).contiguous(),
                a=a.index_select(0, indices).contiguous(),
                b=b.index_select(0, indices).contiguous(),
                A_log=layer.A_log.contiguous(),
                dt_bias=layer.dt_bias.contiguous(),
                state=rt.ssm_cache,
                state_indices=state_indices,
                num_q_heads=layer.num_q_heads,
                num_v_heads=layer.num_v_heads,
                head_k_dim=layer.head_k_dim,
                head_v_dim=layer.head_v_dim,
                scale=layer.head_k_dim**-0.5,
            )
            out.index_copy_(0, indices, core_step.to(dtype=out.dtype))
        return out.unsqueeze(0)

    def forward(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer=None,
        forward_batch=None,
        save_kv_cache: bool = True,
        mixed_qkv: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if layer is None or forward_batch is None:
            raise ValueError("layer and forward_batch are required for GDN forward.")
        if mixed_qkv is None or a is None or b is None:
            raise ValueError("mixed_qkv, a and b are required for GDN forward.")

        reqs = forward_batch.reqs
        if not reqs:
            return mixed_qkv.new_empty((1, 0, layer.num_v_heads, layer.head_v_dim))

        if forward_batch.forward_mode.is_idle():
            return mixed_qkv.new_empty((mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim))
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                **kwargs,
            )
        return self.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )


class HybridLinearBackend(BaseAttnBackend):
    """Dispatches full attention and GDN linear attention like sglang hybrid backend."""

    def __init__(self, full_backend: BaseAttnBackend):
        self.full_backend = full_backend
        self.gdn_backend = GDNAttnBackend()

    def on_table_slot_allocated(self, slot: int) -> None:
        self.gdn_backend.on_table_slot_allocated(slot)

    def has_prefix_cache_state(self, handle: BaseCacheHandle) -> bool:
        return self.gdn_backend.has_prefix_cache_state(handle)

    def on_prefix_cache_store(self, req: Req, handle: BaseCacheHandle) -> None:
        self.gdn_backend.on_prefix_cache_store(req, handle)

    def on_prefix_cache_match(self, handle: BaseCacheHandle, slot: int) -> None:
        self.gdn_backend.on_prefix_cache_match(handle, slot)

    def forward(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer=None,
        forward_batch=None,
        save_kv_cache: bool = True,
        mixed_qkv: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if layer is None or forward_batch is None:
            raise ValueError("layer and forward_batch are required for hybrid attention backend.")
        if forward_batch.forward_mode.is_idle():
            if mixed_qkv is not None:
                return mixed_qkv.new_empty(
                    (mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim)
                )
            if q is None:
                raise ValueError("q is required in idle mode for full attention path.")
            return q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                **kwargs,
            )
        return self.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward_decode(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer=None,
        forward_batch=None,
        save_kv_cache: bool = True,
        mixed_qkv: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if mixed_qkv is not None:
            return self.gdn_backend.forward_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                **kwargs,
            )
        if q is None or k is None or v is None:
            raise ValueError("q/k/v are required for full attention decode path.")
        return self.full_backend.forward(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def forward_extend(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer=None,
        forward_batch=None,
        save_kv_cache: bool = True,
        mixed_qkv: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if mixed_qkv is not None:
            return self.gdn_backend.forward_extend(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                **kwargs,
            )
        if q is None or k is None or v is None:
            raise ValueError("q/k/v are required for full attention extend path.")
        return self.full_backend.forward(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def prepare_metadata(self, batch) -> None:
        return self.full_backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.gdn_backend.init_capture_graph(bs_list)
        return self.full_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch) -> None:
        self.gdn_backend.prepare_for_capture(batch)
        return self.full_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch) -> None:
        self.gdn_backend.prepare_for_replay(batch)
        return self.full_backend.prepare_for_replay(batch)


__all__ = ["GDNAttnBackend", "HybridLinearBackend"]
