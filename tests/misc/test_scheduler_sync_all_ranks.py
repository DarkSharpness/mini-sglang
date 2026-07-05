"""Gate 1.11h — SchedulerIOMixin.sync_all_ranks must be a no-op when there is
no CPU process group.

The guard is orthogonal to device type: single-rank / offline hosts never build
a ``torch.distributed.ProcessGroup`` regardless of accelerator, and calling
``.barrier()`` on ``None`` is what previously blocked ``Scheduler.shutdown()``
on a TP=1 NPU deployment. These tests pin three properties:

1. ``sync_all_ranks`` with ``tp_cpu_group=None`` does not touch a barrier.
2. ``sync_all_ranks`` with a real-looking group calls ``barrier`` and the
   returned work object's ``wait`` exactly once each — the original TP>1
   behaviour is preserved.
3. ``Scheduler.shutdown`` on a TP=1 NPU (group=None) still routes device
   synchronize through ``minisgl.utils.device_runtime.synchronize_device`` and
   still tears down the engine — the sync_all_ranks no-op does not short-circuit
   the rest of shutdown.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

# Ensure ``python/`` is discoverable so ``import minisgl.*`` resolves the real
# source tree even when this file is run standalone. Other tests in this dir
# rely on side-effects from ``test_ascend_fia_backend.py`` for the same insert.
_PY_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))


def _purge_minisgl_from_sys_modules() -> None:
    """Evict any ``minisgl*`` entries so behavioural tests get a clean re-import.

    ``tests/misc/test_device_runtime.py`` installs stub ``minisgl`` /
    ``minisgl.utils`` package objects at import time; leaving them in place
    shadows the real package for later ``from minisgl.scheduler.io import ...``
    resolution.
    """
    for name in list(sys.modules):
        if name == "minisgl" or name.startswith("minisgl."):
            del sys.modules[name]


# ---------------------------------------------------------------------------
# 1. sync_all_ranks with tp_cpu_group=None must not touch a barrier.
# ---------------------------------------------------------------------------


def test_sync_all_ranks_no_op_when_group_is_none() -> None:
    _purge_minisgl_from_sys_modules()

    from minisgl.scheduler.io import SchedulerIOMixin

    # Build a bare mixin without running __init__ — we only test the guard.
    holder = SchedulerIOMixin.__new__(SchedulerIOMixin)
    holder.tp_cpu_group = None  # type: ignore[assignment]

    # No exception, no attribute access on None — proves the guard fired.
    holder.sync_all_ranks()


# ---------------------------------------------------------------------------
# 2. sync_all_ranks preserves original barrier().wait() call when group exists.
# ---------------------------------------------------------------------------


def test_sync_all_ranks_calls_barrier_then_wait_when_group_present() -> None:
    _purge_minisgl_from_sys_modules()

    from minisgl.scheduler.io import SchedulerIOMixin

    call_log: List[str] = []

    class _FakeWork:
        def wait(self) -> None:
            call_log.append("wait")

    class _FakeGroup:
        def barrier(self) -> _FakeWork:
            call_log.append("barrier")
            return _FakeWork()

    holder = SchedulerIOMixin.__new__(SchedulerIOMixin)
    holder.tp_cpu_group = _FakeGroup()  # type: ignore[assignment]

    holder.sync_all_ranks()

    # barrier then wait, each exactly once, in that order.
    assert call_log == ["barrier", "wait"], call_log


# ---------------------------------------------------------------------------
# 3. Scheduler.shutdown on a TP=1 NPU (group=None) still synchronizes the
#    device and shuts the engine down. The sync_all_ranks no-op is transparent
#    to the surrounding shutdown sequence.
# ---------------------------------------------------------------------------


def test_scheduler_shutdown_with_none_group_still_runs_synchronize_and_engine_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    _purge_minisgl_from_sys_modules()

    import minisgl.scheduler.scheduler as sched_mod
    import minisgl.utils.device_runtime as dr

    sync_calls: List[str] = []

    def _fake_synchronize_device(dt: str) -> None:
        sync_calls.append(dt)

    monkeypatch.setattr(dr, "synchronize_device", _fake_synchronize_device)
    monkeypatch.setattr(sched_mod, "synchronize_device", _fake_synchronize_device)

    scheduler = sched_mod.Scheduler.__new__(sched_mod.Scheduler)
    scheduler.device_type = "npu"
    scheduler.engine = MagicMock()
    # Real production state on a TP=1 NPU: Engine._init_communication returned
    # None because tp_info.size == 1, so the mixin stored None into tp_cpu_group.
    scheduler.tp_cpu_group = None  # type: ignore[attr-defined]

    scheduler.shutdown()

    # Device synchronize routed through device_runtime with the engine's dtype.
    assert sync_calls == ["npu"], sync_calls
    # Engine still torn down after the no-op barrier.
    scheduler.engine.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Symmetric check — when a real group is attached, Scheduler.shutdown
#    invokes barrier().wait() exactly once between synchronize_device and
#    engine.shutdown. Guards against future regressions of the TP>1 path.
# ---------------------------------------------------------------------------


def test_scheduler_shutdown_with_real_group_still_barriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    _purge_minisgl_from_sys_modules()

    import minisgl.scheduler.scheduler as sched_mod
    import minisgl.utils.device_runtime as dr

    order: List[str] = []

    def _fake_synchronize_device(dt: str) -> None:
        order.append(f"sync:{dt}")

    monkeypatch.setattr(dr, "synchronize_device", _fake_synchronize_device)
    monkeypatch.setattr(sched_mod, "synchronize_device", _fake_synchronize_device)

    class _FakeWork:
        def wait(self) -> None:
            order.append("wait")

    class _FakeGroup:
        def barrier(self) -> _FakeWork:
            order.append("barrier")
            return _FakeWork()

    engine = MagicMock()

    def _record_engine_shutdown() -> None:
        order.append("engine_shutdown")

    engine.shutdown.side_effect = _record_engine_shutdown

    scheduler = sched_mod.Scheduler.__new__(sched_mod.Scheduler)
    scheduler.device_type = "npu"
    scheduler.engine = engine
    scheduler.tp_cpu_group = _FakeGroup()  # type: ignore[attr-defined]

    scheduler.shutdown()

    assert order == ["sync:npu", "barrier", "wait", "engine_shutdown"], order
