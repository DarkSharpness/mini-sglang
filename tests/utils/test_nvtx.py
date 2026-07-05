"""Gate 1.9j: hermetic tests for the CUDA/non-CUDA dispatch of nvtx_annotate.

The decorator must:
  * be a real no-op when torch.cuda.is_available() returns False
    (no attribute access on torch.cuda.nvtx, no RuntimeError)
  * preserve wraps metadata (__name__, __doc__)
  * transparently forward args/kwargs, return value, and exceptions
  * use torch.cuda.nvtx.range as a context manager when CUDA is available,
    and its __exit__ must run even when the wrapped function raises
  * not touch NVTX at import time nor at decorator construction time
  * not depend on torch_npu
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest


MODULE_PATH = "minisgl.utils.torch_utils"
SOURCE_PATH = Path(__file__).resolve().parents[2] / "python" / "minisgl" / "utils" / "torch_utils.py"


@pytest.fixture()
def nvtx_annotate():
    mod = importlib.import_module(MODULE_PATH)
    return mod.nvtx_annotate


# -------------------------------------------------------------------- helpers
class _FakeNvtxRange:
    """Context manager that records the exact ordering of __enter__/__exit__."""

    def __init__(self, log: list[tuple], name: str):
        self._log = log
        self._name = name

    def __enter__(self):
        self._log.append(("enter", self._name))
        return self

    def __exit__(self, exc_type, exc, tb):
        self._log.append(("exit", self._name, exc_type))
        return False   # do NOT swallow exceptions


class _FakeNvtx:
    def __init__(self, log: list[tuple]):
        self._log = log
        self.range_call_count = 0

    def range(self, name):
        self.range_call_count += 1
        return _FakeNvtxRange(self._log, name)


class _Host:
    """A container to hang decorated methods off of; exercises the ``self`` path."""


# ================================================================== 1. no-CUDA
def test_no_cuda_returns_value_and_forwards_args(nvtx_annotate):
    @nvtx_annotate("Op")
    def fn(self, a, b, *, kw):
        return (a, b, kw)

    host = _Host()
    with mock.patch("torch.cuda.is_available", return_value=False):
        result = fn(host, 1, 2, kw=99)
    assert result == (1, 2, 99)


def test_no_cuda_does_not_touch_torch_cuda_nvtx(nvtx_annotate):
    """Under no-CUDA the decorator must never access torch.cuda.nvtx at all."""
    @nvtx_annotate("Op")
    def fn(self):
        return "ok"

    host = _Host()

    # Sentinel that raises on ANY attribute access — proves the no-CUDA branch
    # doesn't reach into the nvtx submodule.
    class _Boom:
        def __getattr__(self, item):
            raise AssertionError(f"torch.cuda.nvtx.{item} was accessed under no-CUDA")

    with mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch("torch.cuda.nvtx", new=_Boom()):
        assert fn(host) == "ok"


def test_no_cuda_propagates_wrapped_exception(nvtx_annotate):
    @nvtx_annotate("Op")
    def fn(self):
        raise ValueError("boom-no-cuda")

    host = _Host()
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(ValueError, match="boom-no-cuda"):
            fn(host)


# ================================================================== 2. metadata
def test_functools_wraps_preserves_name_and_doc(nvtx_annotate):
    @nvtx_annotate("Op")
    def fn(self):
        """my docstring"""
        return None

    assert fn.__name__ == "fn"
    assert fn.__doc__ == "my docstring"


# ================================================================== 3. fake CUDA
def test_cuda_available_enters_and_exits_nvtx_range_once(nvtx_annotate):
    log: list[tuple] = []
    fake = _FakeNvtx(log)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def fn(self):
        log.append(("body", getattr(self, "_layer_id", None)))
        return "done"

    host = _Host()
    host._layer_id = 7

    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.nvtx", new=fake):
        out = fn(host)

    assert out == "done"
    assert fake.range_call_count == 1
    # ordering: enter → body → exit
    assert log == [("enter", "Layer_7"), ("body", 7), ("exit", "Layer_7", None)]


def test_cuda_available_exits_range_when_function_raises(nvtx_annotate):
    log: list[tuple] = []
    fake = _FakeNvtx(log)

    @nvtx_annotate("Op")
    def fn(self):
        log.append(("body-before-raise",))
        raise RuntimeError("boom-cuda")

    host = _Host()

    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.nvtx", new=fake):
        with pytest.raises(RuntimeError, match="boom-cuda"):
            fn(host)

    # __exit__ must still fire; and it must observe the RuntimeError type.
    assert ("enter", "Op") in log
    assert log[-1][0] == "exit"
    assert log[-1][1] == "Op"
    assert log[-1][2] is RuntimeError
    assert fake.range_call_count == 1


def test_cuda_available_forwards_args_and_return(nvtx_annotate):
    log: list[tuple] = []
    fake = _FakeNvtx(log)

    @nvtx_annotate("Op")
    def fn(self, a, b, *, kw):
        return (a * 10, b * 10, kw)

    host = _Host()
    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.nvtx", new=fake):
        assert fn(host, 1, 2, kw=3) == (10, 20, 3)


# ================================================================== 4. import-time
def test_import_does_not_touch_nvtx_or_query_is_available():
    """Fresh import of the module must not call NVTX or torch.cuda.is_available."""
    sys.modules.pop(MODULE_PATH, None)

    is_available_calls = 0
    nvtx_accesses: list[str] = []

    real_is_available = None
    try:
        import torch  # noqa: F401 — ensure torch is importable
        real_is_available = importlib.import_module("torch").cuda.is_available
    except Exception:
        pytest.skip("torch not importable in this environment")

    def _tracked_is_available():
        nonlocal is_available_calls
        is_available_calls += 1
        return False

    class _NvtxTracker:
        def __getattr__(self, item):
            nvtx_accesses.append(item)
            raise AssertionError(f"NVTX attr {item} accessed at import time")

    with mock.patch("torch.cuda.is_available", new=_tracked_is_available), \
         mock.patch("torch.cuda.nvtx", new=_NvtxTracker()):
        importlib.import_module(MODULE_PATH)
        # Construct a decorator (module-level effect equivalent) — must also
        # be inert until the wrapped function is *called*.
        mod = sys.modules[MODULE_PATH]
        deco = mod.nvtx_annotate("Op")
        @deco
        def _f(self):
            return None
    assert is_available_calls == 0, "torch.cuda.is_available called at import/decorator time"
    assert nvtx_accesses == [], f"NVTX touched at import/decorator time: {nvtx_accesses}"


# ================================================================== 5. no torch_npu
def test_source_does_not_import_torch_npu():
    src = SOURCE_PATH.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torch_npu"), \
                    f"unexpected import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("torch_npu"), \
                f"unexpected from-import: {mod}"
    # Belt-and-braces textual check
    assert "torch_npu" not in src
