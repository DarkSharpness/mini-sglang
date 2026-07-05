"""Gate 1.11d — Sampler.sample() must not touch torch.cuda.nvtx on non-CUDA hosts.

Structural + behavioural checks. Rationale:

* On the Ascend container the installed torch build has no CUDA/NVTX; the raw
  ``with torch.cuda.nvtx.range("Sampler"):`` used to raise
  ``RuntimeError: NVTX functions not installed.`` from inside the sampler.
* The outer ``@nvtx_annotate("Sampler")`` decorator (see
  ``minisgl/utils/torch_utils.py``) already guards on
  ``torch.cuda.is_available()`` and only imports ``torch.cuda.nvtx`` on CUDA,
  so the inline context manager was redundant and unsafe.

Invariants asserted:

1. ``engine/sample.py`` no longer contains any ``torch.cuda.nvtx`` reference.
2. ``Sampler.sample`` in greedy mode (``args.temperatures is None``) returns
   ``torch.argmax(logits, dim=-1)`` — bit-exact — even when
   ``torch.cuda`` is monkey-patched to make any nvtx attribute access blow up.
3. A grep across ``python/minisgl`` finds no unguarded
   ``torch.cuda.nvtx.range`` outside the already-guarded ``nvtx_annotate``
   helper in ``utils/torch_utils.py``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_PATH = _REPO_ROOT / "python" / "minisgl" / "engine" / "sample.py"
_MINISGL_ROOT = _REPO_ROOT / "python" / "minisgl"


# ---------------------------------------------------------------------------
# 1. Structural: sample.py must not reference torch.cuda.nvtx anywhere.
# ---------------------------------------------------------------------------


def test_sample_module_has_no_torch_cuda_nvtx_reference() -> None:
    src = _SAMPLE_PATH.read_text()
    code_lines = []
    for line in src.splitlines():
        # Ignore comments; the fix note may mention torch.cuda.nvtx in prose.
        stripped = line.split("#", 1)[0]
        code_lines.append(stripped)
    code_only = "\n".join(code_lines)
    assert "torch.cuda.nvtx" not in code_only, (
        "engine/sample.py must not contain any executable torch.cuda.nvtx "
        "reference — the outer @nvtx_annotate decorator handles NVTX safely"
    )


def test_sample_method_body_is_flat_and_uses_argmax_for_greedy() -> None:
    """Sampler.sample body: no `with` statement, greedy branch is argmax."""
    tree = ast.parse(_SAMPLE_PATH.read_text())
    sampler_cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Sampler"
    )
    sample_fn = next(
        n for n in sampler_cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "sample"
    )
    # No `with` block should remain in the body — the inline NVTX context is gone.
    with_stmts = [n for n in ast.walk(sample_fn) if isinstance(n, ast.With)]
    assert not with_stmts, (
        "Sampler.sample must not open a `with` block — the inline "
        "`torch.cuda.nvtx.range(...)` context is what breaks non-CUDA builds"
    )
    # Greedy branch must still be an argmax call.
    argmax_calls = [
        c for c in ast.walk(sample_fn)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "argmax"
    ]
    assert argmax_calls, "Sampler.sample greedy branch must call torch.argmax"


# ---------------------------------------------------------------------------
# 2. Behavioural: greedy Sampler.sample must not touch torch.cuda.nvtx.
# ---------------------------------------------------------------------------


class _ExplodingNVTX:
    """Any attribute access explodes — proves Sampler.sample never enters here."""

    def __getattr__(self, name):
        raise AssertionError(
            f"Sampler.sample must NOT access torch.cuda.nvtx.{name} — "
            "the inline NVTX context should be gone"
        )


def test_sampler_greedy_does_not_touch_cuda_nvtx(monkeypatch) -> None:
    from minisgl.engine.sample import BatchSamplingArgs, Sampler

    # Force the nvtx_annotate short-circuit: pretend CUDA is unavailable.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # And booby-trap the nvtx submodule for good measure — any access is a fail.
    monkeypatch.setattr(torch.cuda, "nvtx", _ExplodingNVTX(), raising=False)

    sampler = Sampler(device=torch.device("cpu"), vocab_size=8)
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
        ],
        dtype=torch.float32,
    )
    args = BatchSamplingArgs(temperatures=None)

    out = sampler.sample(logits, args)

    expected = torch.argmax(logits, dim=-1)
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    assert torch.equal(out, expected), (
        f"greedy Sampler.sample must equal torch.argmax(logits, dim=-1); "
        f"got {out.tolist()} vs {expected.tolist()}"
    )


# ---------------------------------------------------------------------------
# 3. Repo-wide scan: no other unguarded torch.cuda.nvtx.range.
# ---------------------------------------------------------------------------


_NVTX_RANGE_RE = re.compile(r"torch\.cuda\.nvtx\.range\b")


def test_no_unguarded_torch_cuda_nvtx_range_in_minisgl() -> None:
    """The only remaining torch.cuda.nvtx reference must live in the guarded
    nvtx_annotate helper (utils/torch_utils.py), which already checks
    torch.cuda.is_available() before importing torch.cuda.nvtx."""
    hits: list[tuple[Path, int, str]] = []
    for path in _MINISGL_ROOT.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if _NVTX_RANGE_RE.search(code):
                hits.append((path, lineno, line))

    # Filter out the acknowledged, guarded call site.
    unguarded = [
        (p, ln, line) for (p, ln, line) in hits
        if p != _MINISGL_ROOT / "utils" / "torch_utils.py"
    ]
    assert not unguarded, (
        "found unguarded torch.cuda.nvtx.range outside the guarded "
        f"nvtx_annotate helper: {unguarded}"
    )
