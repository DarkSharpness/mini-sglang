from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, List, final, override

import torch
from minisgl.utils import UNSET, Unset

_StatDict = Dict[str, torch.Tensor]


def _as_op(value: BaseOP | Callable) -> BaseOP:
    assert isinstance(value, (BaseOP, Callable))
    if not isinstance(value, BaseOP):
        value = _FuncOP(value)
    return value


class BaseOP:
    @final
    def __add__(self, value: BaseOP | Callable) -> _CatOP:
        return _CatOP(self, _as_op(value))

    @final
    def __radd__(self, value: BaseOP | Callable) -> _CatOP:
        return _CatOP(_as_op(value), self)

    @final
    def __or__(self, value: BaseOP | Callable) -> _ParOP:
        return _ParOP(self, _as_op(value))

    @final
    def __ror__(self, value: BaseOP | Callable) -> _ParOP:
        # or is commutative, so we can just call __or__
        return self.__or__(value)

    def _auto_forward(self, x: Any) -> Any:
        """
        A wrapper which accepts one single arg and tries to split it into multiple args.
        """
        if isinstance(x, (list, tuple)):
            return self.forward(*x)
        else:
            return self.forward(x)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement the forward method")

    def __repr__(self) -> str:
        import pprint  # on demand import

        msg = pprint.pformat(self.__dict__, indent=2, width=80, compact=True)
        return f"{self.__class__.__name__}(\n{msg}\n)"

    def load_state_dict(
        self,
        state_dict: _StatDict,
        *,
        prefix: str = "",
        strict: bool = True,
        partial: bool = False,
        _internal=False,
    ) -> None:
        if not _internal:
            state_dict = state_dict.copy()

        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, torch.Tensor):
                if partial and name not in state_dict:
                    continue
                tensor = state_dict.pop(name)
                if strict:
                    assert v.shape == tensor.shape, name
                    assert v.dtype == tensor.dtype, name
                # allow device to be different
                setattr(self, k, tensor)
                del tensor

            elif isinstance(v, BaseOP):
                v.load_state_dict(
                    state_dict, prefix=name, _internal=True, strict=strict, partial=partial
                )

        if not _internal:
            assert not state_dict, (
                f"State dict keys left after loading: {list(state_dict.keys())}. "
                "This usually means that the state dict does not match the operation."
            )

    def state_dict(self, prefix="") -> _StatDict:
        """
        Returns a state dict of the current operation.
        This is used for saving the state of the operation.
        """
        state_dict: _StatDict = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, torch.Tensor):
                state_dict[name] = v
            elif isinstance(v, BaseOP):
                sub_state_dict = v.state_dict(prefix=name)
                state_dict.update(sub_state_dict)
        return state_dict


@final
class _FuncOP(BaseOP):
    def __init__(self, func: Callable):
        self.func = func
        super().__init__()

    @override
    def forward(self, *args: Any) -> Any:
        return self.func(*args)

    def __repr__(self) -> str:
        name = getattr(self.func, "__name__", self.func)
        return f"function: {name}"


class _ProxyOP(BaseOP):
    def _auto_forward(self, x: Any) -> Any:
        """
        For proxy operations, do not unwrap the input when calling forward.
        This can reduce CPU overhead hopefully.
        """
        return self.forward(_packed_args=x)


@final
class _CatOP(_ProxyOP):
    def __init__(self, *ops: BaseOP):
        self._ops: List[BaseOP] = []
        for op in ops:
            if isinstance(op, _CatOP):
                self._ops.extend(op._ops)
            else:
                self._ops.append(op)
        super().__init__()

    @override
    def forward(self, *args: Any, _packed_args: Any = UNSET) -> Any:
        x = args if isinstance(_packed_args, Unset) else _packed_args
        for op in self._ops:
            x = op._auto_forward(x)
        return x

    def __repr__(self) -> str:
        return "(" + " + ".join(repr(op) for op in self._ops) + ")"


@final
class ObserverOP(BaseOP):
    def __init__(self, hook: Callable[[Any]]):
        super().__init__()
        self._hook = hook

    def _auto_forward(self, x: Any) -> Any:
        self._hook(x)
        return x

    @override
    def forward(self, *args: Any) -> Any:
        self._hook(args)
        return args


IDENTITY = ObserverOP(lambda _: None)


@final
class _ParOP(BaseOP):
    def __init__(self, *ops: BaseOP):
        self._ops: List[BaseOP] = []
        for op in ops:
            if isinstance(op, _ParOP):
                self._ops.extend(op._ops)
            else:
                self._ops.append(op)
        super().__init__()

    @override
    def forward(self, *args: Any, _packed_args: Any = UNSET) -> Any:
        x = args if isinstance(_packed_args, Unset) else _packed_args
        results = [op._auto_forward(x) for op in self._ops]
        results = [r if isinstance(r, (list, tuple)) else [r] for r in results]
        return [item for sublist in results for item in sublist]

    def __repr__(self) -> str:
        return "(" + " | ".join(repr(op) for op in self._ops) + ")"


class CustomOP(BaseOP):
    def __init__(self, model: BaseOP):
        self._model = model
        super().__init__()

    def _auto_forward(self, x: Any) -> Any:
        return self._model._auto_forward(x)

    @final
    @override
    def forward(self, *args: Any) -> Any:
        return self._model.forward(*args)


@final
class ListOP(BaseOP):
    def __init__(self, ops: Iterable[BaseOP]):
        self._ops = tuple(ops)
        super().__init__()

    @override
    def forward(self, *args: Any) -> Any:
        for op in self._ops:
            args = op._auto_forward(args)
        return args

    def load_state_dict(
        self,
        state_dict: _StatDict,
        *,
        prefix: str = "",
        strict: bool = True,
        partial: bool = False,
        _internal=False,
    ) -> None:
        if not _internal:
            state_dict = state_dict.copy()

        for i, op in enumerate(self._ops):
            name = f"{prefix}.{i}" if prefix else f"{i}"
            op.load_state_dict(
                state_dict, prefix=name, _internal=True, strict=strict, partial=partial
            )

        if not _internal:
            assert not state_dict, (
                f"State dict keys left after loading: {list(state_dict.keys())}. "
                "This usually means that the state dict does not match the operation."
            )

    def state_dict(self, prefix: str = "") -> _StatDict:
        """
        Returns a state dict of the current operation.
        This is used for saving the state of the operation.
        """
        state_dict: _StatDict = {}
        for i, op in enumerate(self._ops):
            name = f"{prefix}.{i}" if prefix else f"{i}"
            sub_state_dict = op.state_dict(prefix=name)
            state_dict.update(sub_state_dict)
        return state_dict


@final
class TakeOP(BaseOP):
    def __init__(self, which: int):
        self.which = which

    @override
    def forward(self, *args: Any) -> Any:
        return args[self.which]


__all__ = [
    "BaseOP",
    "CustomOP",
    "ListOP",
    "TakeOP",
    "IDENTITY",
    "ObserverOP",
]
