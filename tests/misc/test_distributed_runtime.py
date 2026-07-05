"""Unit tests for :mod:`minisgl.distributed.runtime`.

The runtime initialiser touches ``torch`` and optionally ``torch_npu``. This
suite fakes both via ``sys.modules`` injection so it runs on a bare macOS host
without any real torch/CUDA/NPU install. As with the sibling suites the
modules under test are loaded via ``importlib.util`` to bypass the package
``__init__`` files that import the heavier optional runtime deps.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PY_ROOT = _REPO_ROOT / "python"
_DEVICE_PATH = _PY_ROOT / "minisgl" / "utils" / "device.py"
_BACKEND_PATH = _PY_ROOT / "minisgl" / "distributed" / "backend.py"
_RUNTIME_PATH = _PY_ROOT / "minisgl" / "distributed" / "runtime.py"


def _install_package_stub(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


def _load_isolated(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot build spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_install_package_stub("minisgl", _PY_ROOT / "minisgl")
_install_package_stub("minisgl.utils", _PY_ROOT / "minisgl" / "utils")
_install_package_stub("minisgl.distributed", _PY_ROOT / "minisgl" / "distributed")

device = _load_isolated("minisgl.utils.device", _DEVICE_PATH)
backend_mod = _load_isolated("minisgl.distributed.backend", _BACKEND_PATH)
runtime = _load_isolated("minisgl.distributed.runtime", _RUNTIME_PATH)


class _CallLog:
    """Ordered log of every fake torch entrypoint hit during a test."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def record(self, name: str, payload: Any = None) -> None:
        self.events.append((name, payload))


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    log: _CallLog,
    *,
    already_initialised: bool = False,
) -> types.ModuleType:
    """Inject a minimal fake ``torch`` (with ``.cuda`` / ``.npu`` / ``.distributed``)."""
    fake_torch = types.ModuleType("torch")

    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.set_device = lambda idx: log.record("cuda.set_device", idx)

    fake_npu = types.ModuleType("torch.npu")
    fake_npu.set_device = lambda idx: log.record("npu.set_device", idx)

    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.is_initialized = lambda: already_initialised
    fake_dist.init_process_group = lambda **kw: log.record("init_process_group", kw)

    fake_torch.cuda = fake_cuda
    fake_torch.npu = fake_npu
    fake_torch.distributed = fake_dist

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
    monkeypatch.setitem(sys.modules, "torch.npu", fake_npu)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    return fake_torch


def _set_env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    for k, v in values.items():
        monkeypatch.setenv(k, v)


def _install_torch_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake ``torch_npu`` — mere presence in ``sys.modules`` is enough."""
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))


def _install_torch_npu_import_hook(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "torch_npu" or name.startswith("torch_npu."):
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)


@pytest.fixture(autouse=True)
def _reset_device_cache():
    device.is_cuda_available.cache_clear()
    device.is_npu_available.cache_clear()
    device.get_device_type.cache_clear()
    yield
    device.is_cuda_available.cache_clear()
    device.is_npu_available.cache_clear()
    device.get_device_type.cache_clear()


# ---------------------------------------------------------------------------
# Happy paths per device
# ---------------------------------------------------------------------------


def test_cuda_binds_local_rank_and_selects_nccl(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cuda")
    _set_env(monkeypatch, LOCAL_RANK="1", RANK="1", WORLD_SIZE="4")

    result = runtime.initialize_distributed_from_env()

    assert result.device_type == "cuda"
    assert result.backend == "nccl"
    assert result.local_rank == 1
    assert result.rank == 1
    assert result.world_size == 4
    assert result.device == "cuda:1"

    # set_device must land before init_process_group and with the local rank.
    assert log.events == [
        ("cuda.set_device", 1),
        ("init_process_group", {"backend": "nccl"}),
    ]


def test_npu_imports_torch_npu_binds_local_rank_and_selects_hccl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "npu")
    _set_env(monkeypatch, LOCAL_RANK="2", RANK="2", WORLD_SIZE="8")

    result = runtime.initialize_distributed_from_env()

    assert result.device_type == "npu"
    assert result.backend == "hccl"
    assert result.device == "npu:2"
    assert result.local_rank == 2
    assert result.rank == 2
    assert result.world_size == 8

    assert log.events == [
        ("npu.set_device", 2),
        ("init_process_group", {"backend": "hccl"}),
    ]
    # torch_npu must have been imported dynamically.
    assert "torch_npu" in sys.modules


def test_cpu_does_not_touch_set_device_and_selects_gloo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cpu")
    _set_env(monkeypatch, LOCAL_RANK="0", RANK="0", WORLD_SIZE="1")

    result = runtime.initialize_distributed_from_env()

    assert result.device_type == "cpu"
    assert result.backend == "gloo"
    assert result.device == "cpu"

    # No set_device call — only init_process_group with the gloo backend.
    assert log.events == [("init_process_group", {"backend": "gloo"})]


# ---------------------------------------------------------------------------
# Env-var validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["LOCAL_RANK", "RANK", "WORLD_SIZE"])
def test_missing_env_var_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cpu")
    env = {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1"}
    env.pop(missing)
    _set_env(monkeypatch, **env)

    with pytest.raises(RuntimeError, match=missing):
        runtime.initialize_distributed_from_env()
    # Nothing was initialised.
    assert log.events == []


def test_non_integer_env_var_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cpu")
    _set_env(monkeypatch, LOCAL_RANK="0", RANK="oops", WORLD_SIZE="1")

    with pytest.raises(RuntimeError, match="RANK"):
        runtime.initialize_distributed_from_env()
    assert log.events == []


def test_world_size_zero_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cpu")
    _set_env(monkeypatch, LOCAL_RANK="0", RANK="0", WORLD_SIZE="0")

    with pytest.raises(ValueError, match="WORLD_SIZE"):
        runtime.initialize_distributed_from_env()
    assert log.events == []


def test_rank_out_of_range_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cpu")
    _set_env(monkeypatch, LOCAL_RANK="0", RANK="4", WORLD_SIZE="4")

    with pytest.raises(ValueError, match="RANK"):
        runtime.initialize_distributed_from_env()
    assert log.events == []


def test_negative_local_rank_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cpu")
    _set_env(monkeypatch, LOCAL_RANK="-1", RANK="0", WORLD_SIZE="1")

    with pytest.raises(ValueError, match="LOCAL_RANK"):
        runtime.initialize_distributed_from_env()
    assert log.events == []


# ---------------------------------------------------------------------------
# Double-init guard & backend forwarding
# ---------------------------------------------------------------------------


def test_double_init_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log, already_initialised=True)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cuda")
    _set_env(monkeypatch, LOCAL_RANK="0", RANK="0", WORLD_SIZE="1")

    with pytest.raises(RuntimeError, match="already initialised"):
        runtime.initialize_distributed_from_env()

    # Neither device binding nor process group creation may fire.
    assert log.events == []


def test_init_process_group_receives_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify backend is looked up via ``get_distributed_backend`` and forwarded."""
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cuda")
    _set_env(monkeypatch, LOCAL_RANK="0", RANK="0", WORLD_SIZE="1")

    runtime.initialize_distributed_from_env()

    init_events = [e for e in log.events if e[0] == "init_process_group"]
    assert len(init_events) == 1
    assert init_events[0][1] == {"backend": "nccl"}
    # No init_method / rank / world_size overrides leaked through.
    assert list(init_events[0][1].keys()) == ["backend"]


# ---------------------------------------------------------------------------
# torch_npu import failure
# ---------------------------------------------------------------------------


def test_npu_selected_but_torch_npu_import_fails_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "npu")
    _install_torch_npu_import_hook(monkeypatch, ImportError("no torch_npu wheel"))
    _set_env(monkeypatch, LOCAL_RANK="0", RANK="0", WORLD_SIZE="1")

    with pytest.raises(RuntimeError, match="torch_npu"):
        runtime.initialize_distributed_from_env()

    # Failure must land BEFORE init_process_group — no group must be created.
    assert not any(e[0] == "init_process_group" for e in log.events)


def test_runtime_module_does_not_import_torch_npu_at_module_scope() -> None:
    """Loading ``runtime`` must not pull in ``torch_npu`` — only device-npu path does."""
    # ``runtime`` was loaded at test-module import time above; if it had
    # ``import torch_npu`` at the top level we would see it in sys.modules
    # already (no test has run the NPU path yet under this specific assertion).
    # We therefore only check that the runtime module object itself has no
    # ``torch_npu`` attribute — a weaker but robust proxy given the shared
    # sys.modules state across parametrised tests.
    assert not hasattr(runtime, "torch_npu")


# ---------------------------------------------------------------------------
# bind_local_device: shared device-binding primitive (Gate 1.1b)
# ---------------------------------------------------------------------------


def test_bind_local_device_is_public_export() -> None:
    """``bind_local_device`` is part of the runtime's public surface."""
    assert hasattr(runtime, "bind_local_device")
    assert "bind_local_device" in runtime.__all__


def test_bind_local_device_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)

    device = runtime.bind_local_device("cuda", 3)

    assert device == "cuda:3"
    assert log.events == [("cuda.set_device", 3)]


def test_bind_local_device_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu(monkeypatch)

    device = runtime.bind_local_device("npu", 5)

    assert device == "npu:5"
    assert log.events == [("npu.set_device", 5)]
    # The dynamic import must have populated sys.modules.
    assert "torch_npu" in sys.modules


def test_bind_local_device_cpu_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)

    device = runtime.bind_local_device("cpu", 0)

    assert device == "cpu"
    # No set_device calls; no init_process_group.
    assert log.events == []


def test_bind_local_device_cpu_ignores_local_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU is a singleton — the rank must not leak into the returned spec."""
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)

    assert runtime.bind_local_device("cpu", 7) == "cpu"
    assert log.events == []


def test_bind_local_device_invalid_type_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)

    with pytest.raises(ValueError, match="tpu"):
        runtime.bind_local_device("tpu", 0)  # type: ignore[arg-type]
    assert log.events == []


def test_bind_local_device_npu_import_failure_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    _install_torch_npu_import_hook(monkeypatch, ImportError("no torch_npu wheel"))

    with pytest.raises(RuntimeError, match="torch_npu"):
        runtime.bind_local_device("npu", 0)
    # Failure must land BEFORE any set_device call.
    assert log.events == []


def test_bind_local_device_does_not_touch_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``bind_local_device`` must NOT touch ``torch.distributed`` — that's the caller's job."""
    log = _CallLog()
    _install_fake_torch(monkeypatch, log)

    runtime.bind_local_device("cuda", 0)

    assert all(name != "init_process_group" for name, _ in log.events)


def test_initialize_from_env_reuses_bind_local_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``initialize_distributed_from_env`` must route through the shared helper."""
    calls: list[tuple[str, int]] = []

    def spy_bind(device_type: str, local_rank: int) -> str:
        calls.append((device_type, local_rank))
        return f"{device_type}:{local_rank}" if device_type != "cpu" else "cpu"

    log = _CallLog()
    _install_fake_torch(monkeypatch, log)
    monkeypatch.setattr(runtime, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(runtime, "bind_local_device", spy_bind)
    _set_env(monkeypatch, LOCAL_RANK="6", RANK="6", WORLD_SIZE="8")

    result = runtime.initialize_distributed_from_env()

    assert calls == [("cuda", 6)]
    assert result.device == "cuda:6"
    # No direct set_device call escaped through the fake torch — the spy caught it.
    assert all(name != "cuda.set_device" for name, _ in log.events)


def test_runtime_has_no_private_bind_device_duplicate() -> None:
    """Gate 1.1b removed the ``_bind_device`` private duplicate."""
    assert not hasattr(runtime, "_bind_device")