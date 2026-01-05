from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue

            if isinstance(param, torch.Tensor):
                if "experts" in prefix:
                    mapped_name = name
                    matched_keys = []
                    for key in list(state_dict.keys()):
                        if prefix in key and mapped_name in key:
                            matched_keys.append(key)

                    def extract_expert_index(k):
                        match = re.search(r'experts\.(\d+)\.', k)
                        return int(match.group(1)) if match else 0

                    matched_keys.sort(key=extract_expert_index)

                    items = []
                    for k in matched_keys:
                        items.append(state_dict.pop(k))

                    if not items:
                        raise ValueError(f"No weights found in state_dict for {prefix} and {mapped_name}")

                    item = torch.stack(items, dim=0)
                else:
                    item = state_dict.pop(_concat_prefix(prefix, name))

                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape, f"Shape mismatch: param {param.shape} vs item {item.shape}"
                assert param.dtype == item.dtype, f"Dtype mismatch: param {param.dtype} vs item {item.dtype}"

                setattr(self, name, item)

            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict,
                    prefix=_concat_prefix(prefix, name),
                    _internal=True
                )

        if not _internal and state_dict:
            keys = list(state_dict.keys())
            raise RuntimeError(f"Unexpected keys in state_dict: {len(keys)} keys (first 10: {keys[:10]})")


class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            _ = prefix
            keys = list(state_dict.keys())
            raise RuntimeError(f"Unexpected keys in state_dict: {len(keys)} keys (first 10: {keys[:10]})")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        _ = prefix
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            keys = list(state_dict.keys())
            raise RuntimeError(f"Unexpected keys in state_dict: {len(keys)} keys (first 10: {keys[:10]})")