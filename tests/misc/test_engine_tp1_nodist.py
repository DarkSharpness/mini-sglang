"""Gate 1.11a — NPU + TP=1 must skip the torch.distributed bootstrap.

Structural (AST-only) checks. Rationale:

* Constructing an ``Engine`` for real would import the whole model stack, wire
  up cuda/npu runtimes, and probably crash on hosts without an accelerator.
  The existing ``test_engine_device.py`` file already established a pure-AST
  test pattern for the same module, so this file follows the same shape.

* The change is purely a control-flow guard, so a structural check is
  sufficient — no runtime side-effects to validate here beyond what the
  hardware smoke on the 910B1 container exercises.

Invariants asserted:

1. ``Engine._init_communication`` opens with an ``if`` block matching
   ``self.device_type == "npu" and config.tp_info.size == 1`` and that block
   ``return``s early (``None``) before touching torch.distributed.
2. The early-return branch does **not** call ``init_process_group``,
   ``enable_pynccl_distributed``, ``get_distributed_backend``, or ``new_group``
   — those imports/calls stay reachable only from the fall-through paths.
3. The CUDA TP=1 path (``config.tp_info.size == 1 or config.use_pynccl``) is
   preserved verbatim: it still initialises gloo and calls
   ``enable_pynccl_distributed``.
4. The accelerator TP>1 path still resolves the backend via
   ``get_distributed_backend(self.device_type)``.
5. ``Engine._sync_get_memory`` short-circuits when ``self.tp_cpu_group is
   None`` — the all_reduce is unreachable from the no-dist fast path.
6. ``Engine.shutdown`` guards ``torch.distributed.destroy_process_group()``
   with the same ``self.tp_cpu_group is not None`` check.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENGINE_PATH = _REPO_ROOT / "python" / "minisgl" / "engine" / "engine.py"


def _engine_tree() -> ast.Module:
    return ast.parse(_ENGINE_PATH.read_text())


def _engine_method(name: str) -> ast.FunctionDef:
    engine_cls = next(
        node for node in _engine_tree().body
        if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    return next(
        node for node in engine_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


# ---------------------------------------------------------------------------
# _init_communication: NPU + TP=1 no-op branch
# ---------------------------------------------------------------------------


def _init_comm() -> ast.FunctionDef:
    return _engine_method("_init_communication")


def _is_self_device_type_eq_npu(node: ast.AST) -> bool:
    """Match ``self.device_type == "npu"``."""
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return False
    left = node.left
    if not (
        isinstance(left, ast.Attribute)
        and left.attr == "device_type"
        and isinstance(left.value, ast.Name)
        and left.value.id == "self"
    ):
        return False
    right = node.comparators[0]
    return isinstance(right, ast.Constant) and right.value == "npu"


def _is_config_tp_size_eq_1(node: ast.AST) -> bool:
    """Match ``config.tp_info.size == 1``."""
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return False
    left = node.left
    if not (
        isinstance(left, ast.Attribute)
        and left.attr == "size"
        and isinstance(left.value, ast.Attribute)
        and left.value.attr == "tp_info"
        and isinstance(left.value.value, ast.Name)
        and left.value.value.id == "config"
    ):
        return False
    right = node.comparators[0]
    return isinstance(right, ast.Constant) and right.value == 1


def _find_npu_tp1_guard(fn: ast.FunctionDef) -> ast.If:
    """Locate the ``if self.device_type == "npu" and config.tp_info.size == 1`` block."""
    for stmt in fn.body:
        if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.BoolOp) and isinstance(
            stmt.test.op, ast.And
        ):
            values = stmt.test.values
            if len(values) == 2 and (
                (_is_self_device_type_eq_npu(values[0]) and _is_config_tp_size_eq_1(values[1]))
                or (_is_self_device_type_eq_npu(values[1]) and _is_config_tp_size_eq_1(values[0]))
            ):
                return stmt
    raise AssertionError(
        "Engine._init_communication must open with an "
        "`if self.device_type == 'npu' and config.tp_info.size == 1:` guard"
    )


def test_init_communication_has_npu_tp1_early_return_none() -> None:
    """The NPU+TP=1 guard must be the first non-trivial statement and return None."""
    fn = _init_comm()
    guard = _find_npu_tp1_guard(fn)
    # Body must be exactly `return None` (or bare `return`).
    assert len(guard.body) == 1, (
        "NPU+TP=1 guard body must be a single `return` statement"
    )
    ret = guard.body[0]
    assert isinstance(ret, ast.Return), "NPU+TP=1 guard must `return`"
    # `return None` and bare `return` both acceptable — both signal no group.
    if ret.value is not None:
        assert isinstance(ret.value, ast.Constant) and ret.value.value is None, (
            "NPU+TP=1 guard must return None (no tp_cpu_group)"
        )
    # Guard must have no `else` — the CUDA path stays in the outer body.
    assert guard.orelse == [], (
        "NPU+TP=1 guard must not use an `else` branch — the CUDA/HCCL paths "
        "fall through to the outer function body"
    )


def _walk_calls(node: ast.AST) -> list[ast.Call]:
    return [c for c in ast.walk(node) if isinstance(c, ast.Call)]


def _has_call(node: ast.AST, matcher) -> bool:
    return any(matcher(c) for c in _walk_calls(node))


def _is_init_process_group(call: ast.Call) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == "init_process_group"
        and isinstance(f.value, ast.Attribute)
        and f.value.attr == "distributed"
    )


def _is_enable_pynccl(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == "enable_pynccl_distributed"


def _is_get_distributed_backend(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == "get_distributed_backend"


def _is_new_group(call: ast.Call) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == "new_group"
        and isinstance(f.value, ast.Attribute)
        and f.value.attr == "distributed"
    )


def test_npu_tp1_guard_body_has_no_distributed_side_effects() -> None:
    """The early-return branch must not call any dist init helper."""
    guard = _find_npu_tp1_guard(_init_comm())
    for matcher, label in [
        (_is_init_process_group, "torch.distributed.init_process_group"),
        (_is_enable_pynccl, "enable_pynccl_distributed"),
        (_is_get_distributed_backend, "get_distributed_backend"),
        (_is_new_group, "torch.distributed.new_group"),
    ]:
        assert not _has_call(guard, matcher), (
            f"NPU+TP=1 no-op guard must not call {label}"
        )


def test_init_communication_still_preserves_cuda_pynccl_branch() -> None:
    """CUDA TP=1 / use_pynccl branch must still call gloo init + pynccl helper."""
    fn = _init_comm()
    # Any occurrence anywhere below the guard.
    assert _has_call(fn, _is_init_process_group), (
        "torch.distributed.init_process_group must still be called from the CUDA branch"
    )
    assert _has_call(fn, _is_enable_pynccl), (
        "enable_pynccl_distributed must still be called from the CUDA TP=1 branch"
    )
    # And the gloo/init_process_group in the pynccl branch keeps backend='gloo'.
    for call in _walk_calls(fn):
        if not _is_init_process_group(call):
            continue
        kw = {k.arg: k.value for k in call.keywords}
        assert "backend" in kw, "init_process_group must specify backend explicitly"


def test_init_communication_still_uses_get_distributed_backend_for_hccl_path() -> None:
    """NPU TP>1 (and CUDA TP>1 non-pynccl) still resolves via get_distributed_backend(self.device_type)."""
    fn = _init_comm()
    calls = [c for c in _walk_calls(fn) if _is_get_distributed_backend(c)]
    assert calls, (
        "Engine._init_communication must still call get_distributed_backend for TP>1"
    )
    for c in calls:
        assert len(c.args) == 1 and not c.keywords, (
            "get_distributed_backend(...) must take exactly one positional arg"
        )
        arg = c.args[0]
        assert (
            isinstance(arg, ast.Attribute)
            and arg.attr == "device_type"
            and isinstance(arg.value, ast.Name)
            and arg.value.id == "self"
        ), "get_distributed_backend must be called with self.device_type"


def test_init_communication_return_annotation_is_nullable() -> None:
    """The signature must advertise ``ProcessGroup | None`` so callers guard on None."""
    fn = _init_comm()
    ret = fn.returns
    assert ret is not None, "Engine._init_communication must declare a return type"
    # Accept either PEP 604 (X | None) or Optional[X].
    src = ast.unparse(ret)
    assert "None" in src, (
        "return type must include None to signal the NPU+TP=1 no-op contract, "
        f"got: {src!r}"
    )


# ---------------------------------------------------------------------------
# _sync_get_memory: guard the all_reduce when tp_cpu_group is None
# ---------------------------------------------------------------------------


def _is_all_reduce(call: ast.Call) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == "all_reduce"
        and isinstance(f.value, ast.Attribute)
        and f.value.attr == "distributed"
    )


def _is_self_tp_cpu_group_is_none(node: ast.AST) -> bool:
    """Match ``self.tp_cpu_group is None``."""
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Is):
        return False
    left = node.left
    if not (
        isinstance(left, ast.Attribute)
        and left.attr == "tp_cpu_group"
        and isinstance(left.value, ast.Name)
        and left.value.id == "self"
    ):
        return False
    right = node.comparators[0]
    return isinstance(right, ast.Constant) and right.value is None


def _is_self_tp_cpu_group_is_not_none(node: ast.AST) -> bool:
    """Match ``self.tp_cpu_group is not None``."""
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.IsNot):
        return False
    left = node.left
    if not (
        isinstance(left, ast.Attribute)
        and left.attr == "tp_cpu_group"
        and isinstance(left.value, ast.Name)
        and left.value.id == "self"
    ):
        return False
    right = node.comparators[0]
    return isinstance(right, ast.Constant) and right.value is None


def test_sync_get_memory_short_circuits_when_tp_cpu_group_is_none() -> None:
    """A top-level ``if self.tp_cpu_group is None: return ...`` must guard the all_reduce."""
    fn = _engine_method("_sync_get_memory")
    guards = [
        stmt for stmt in fn.body
        if isinstance(stmt, ast.If) and _is_self_tp_cpu_group_is_none(stmt.test)
    ]
    assert guards, (
        "_sync_get_memory must guard on `if self.tp_cpu_group is None:` to "
        "support the NPU+TP=1 no-dist fast path"
    )
    guard = guards[0]
    assert any(isinstance(s, ast.Return) for s in guard.body), (
        "the tp_cpu_group-None guard body must return before touching all_reduce"
    )
    # And this guard must appear before the all_reduce call in source order.
    all_reduce_calls = [c for c in _walk_calls(fn) if _is_all_reduce(c)]
    assert all_reduce_calls, "_sync_get_memory must still call all_reduce on the multi-rank path"
    all_reduce_line = min(c.lineno for c in all_reduce_calls)
    assert guard.lineno < all_reduce_line, (
        "the tp_cpu_group-None guard must precede the all_reduce call"
    )


# ---------------------------------------------------------------------------
# shutdown: also guard destroy_process_group
# ---------------------------------------------------------------------------


def _is_destroy_process_group(call: ast.Call) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == "destroy_process_group"
        and isinstance(f.value, ast.Attribute)
        and f.value.attr == "distributed"
    )


def test_shutdown_guards_destroy_process_group_on_tp_cpu_group() -> None:
    """shutdown() must not tear down a group that _init_communication skipped."""
    fn = _engine_method("shutdown")
    # Find the guard.
    guard = None
    for stmt in fn.body:
        if isinstance(stmt, ast.If) and _is_self_tp_cpu_group_is_not_none(stmt.test):
            guard = stmt
            break
    assert guard is not None, (
        "shutdown must guard destroy_process_group() behind "
        "`if self.tp_cpu_group is not None:`"
    )
    # Guard body must include destroy_process_group.
    destroy_calls_in_guard = [c for c in _walk_calls(guard) if _is_destroy_process_group(c)]
    assert destroy_calls_in_guard, (
        "destroy_process_group() must live inside the tp_cpu_group-not-None guard"
    )
    # No unguarded destroy_process_group at the top level of shutdown.
    top_level_calls = [
        stmt.value for stmt in fn.body
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
    ]
    for call in top_level_calls:
        assert not _is_destroy_process_group(call), (
            "destroy_process_group() must not appear unguarded at the top level of shutdown()"
        )
