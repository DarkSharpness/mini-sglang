"""Static checks over ``pyproject.toml`` — Gate 0.3b.

These tests do not install anything, do not touch ``uv.lock``, and do not
execute any project code. They parse ``pyproject.toml`` with the stdlib
``tomllib`` and assert that CUDA-only wheels have been moved out of the
base ``[project].dependencies`` array and into a ``cuda`` extra.

Dependency names are compared after stripping the version specifier so the
tests remain robust to routine pin bumps (``flashinfer-python>=0.6``,
``sgl_kernel>=0.4``, ...).
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# CUDA-only PyPI distributions that must be extras, not base runtime deps.
_CUDA_ONLY = frozenset(
    {
        "flashinfer-python",
        "sgl_kernel",
        "quack-kernels",
    }
)

# Dependencies that MUST live in the base ``[project].dependencies`` because
# they are exercised on every supported device (CUDA, Ascend NPU, CPU).
# ``apache-tvm-ffi`` drives the AOT-compiled CPU helpers under
# ``minisgl.kernel`` (notably ``fast_compare_key`` invoked by the radix
# prefix cache on every prefill scheduling call), so it cannot be a
# device-specific extra.
_BASE_REQUIRED = frozenset(
    {
        "apache-tvm-ffi",
    }
)

# PEP 508 requirement string → distribution name (very small subset: we only
# need to strip the version specifier / extras / env markers on the trailing
# side; this is not a full parser but is sufficient for the entries we own).
_NAME_SPLIT = re.compile(r"[\s<>=!~;\[]")


def _pkg_name(requirement: str) -> str:
    return _NAME_SPLIT.split(requirement.strip(), 1)[0].lower()


@pytest.fixture(scope="module")
def project_table() -> dict:
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]


@pytest.fixture(scope="module")
def base_deps(project_table: dict) -> set[str]:
    return {_pkg_name(r) for r in project_table["dependencies"]}


@pytest.fixture(scope="module")
def extras(project_table: dict) -> dict[str, list[str]]:
    return project_table["optional-dependencies"]


# ---------------------------------------------------------------------------
# 1. CUDA-only deps must not appear in the base runtime list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dep", sorted(_CUDA_ONLY))
def test_cuda_only_dep_is_not_in_base_dependencies(base_deps: set[str], dep: str) -> None:
    assert dep.lower() not in base_deps, (
        f"{dep!r} must not appear in [project].dependencies; move it into the "
        f"'cuda' optional-dependencies extra"
    )


# ---------------------------------------------------------------------------
# 2. All four CUDA deps must live in the `cuda` extra
# ---------------------------------------------------------------------------


def test_cuda_extra_exists(extras: dict[str, list[str]]) -> None:
    assert "cuda" in extras, (
        "[project.optional-dependencies].cuda must exist and hold the CUDA-only wheels"
    )


@pytest.mark.parametrize("dep", sorted(_CUDA_ONLY))
def test_cuda_extra_contains_dep(extras: dict[str, list[str]], dep: str) -> None:
    cuda_names = {_pkg_name(r) for r in extras["cuda"]}
    assert dep.lower() in cuda_names, (
        f"{dep!r} missing from the 'cuda' extra; current members: "
        f"{sorted(cuda_names)}"
    )


def test_cuda_extra_holds_only_the_four_cuda_wheels(extras: dict[str, list[str]]) -> None:
    """The extra should not silently grow other packages under this Gate."""
    cuda_names = {_pkg_name(r) for r in extras["cuda"]}
    assert cuda_names == {name.lower() for name in _CUDA_ONLY}, (
        f"cuda extra membership drifted: expected {sorted(_CUDA_ONLY)}, "
        f"got {sorted(cuda_names)}"
    )


# ---------------------------------------------------------------------------
# 2b. Cross-device required deps must live in the base dependency list, not
#     only in the cuda extra — Gate 2.1g fix for Ascend Radix reuse.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dep", sorted(_BASE_REQUIRED))
def test_cross_device_required_dep_is_in_base_dependencies(
    base_deps: set[str], dep: str
) -> None:
    assert dep.lower() in base_deps, (
        f"{dep!r} must appear in [project].dependencies so that clean "
        f"Ascend installs receive it — it is required by kernel.radix "
        f"which underpins RadixPrefixCache on every device"
    )


@pytest.mark.parametrize("dep", sorted(_BASE_REQUIRED))
def test_cross_device_required_dep_not_only_in_cuda_extra(
    extras: dict[str, list[str]], dep: str
) -> None:
    cuda_names = {_pkg_name(r) for r in extras.get("cuda", [])}
    assert dep.lower() not in cuda_names, (
        f"{dep!r} must not live in the 'cuda' extra — it is a cross-device "
        f"requirement and belongs in the base dependency list. Duplicating "
        f"it in the cuda extra creates an invisible declaration split."
    )


def test_apache_tvm_ffi_pin_preserved(project_table: dict) -> None:
    """The declared floor (>=0.1.4) matches the smoke-tested wheel."""
    reqs = [r for r in project_table["dependencies"] if _pkg_name(r) == "apache-tvm-ffi"]
    assert reqs == ["apache-tvm-ffi>=0.1.4"], (
        f"apache-tvm-ffi requirement changed unexpectedly: {reqs!r}"
    )


# ---------------------------------------------------------------------------
# 3. `dev` extra is preserved
# ---------------------------------------------------------------------------


def test_dev_extra_is_preserved(extras: dict[str, list[str]]) -> None:
    assert "dev" in extras, "the existing 'dev' extra must not be removed"
    dev_names = {_pkg_name(r) for r in extras["dev"]}
    # Spot check a couple of well-known dev tools rather than pinning the full
    # list — this keeps the assertion useful without turning it into a rubber
    # stamp for future dev-extra edits.
    assert "pytest" in dev_names
    assert "ruff" in dev_names


# ---------------------------------------------------------------------------
# 4. Torch stays a base dependency
# ---------------------------------------------------------------------------


def test_torch_still_in_base_dependencies(
    project_table: dict, base_deps: set[str]
) -> None:
    assert "torch" in base_deps, (
        "'torch' must remain in the base [project].dependencies for Gate 0.3b"
    )
    # Pin range preserved verbatim — no accidental widening.
    torch_reqs = [r for r in project_table["dependencies"] if _pkg_name(r) == "torch"]
    assert torch_reqs == ["torch<2.10.0"], (
        f"torch requirement changed unexpectedly: {torch_reqs!r}"
    )


# ---------------------------------------------------------------------------
# 5. torch_npu must NOT be auto-added anywhere
# ---------------------------------------------------------------------------


def test_torch_npu_is_not_declared_anywhere(
    project_table: dict, extras: dict[str, list[str]]
) -> None:
    def scan(reqs: list[str]) -> list[str]:
        return [r for r in reqs if _pkg_name(r) in {"torch_npu", "torch-npu"}]

    assert scan(project_table["dependencies"]) == [], (
        "torch_npu must not be introduced in base dependencies during Gate 0.3b"
    )
    for extra_name, reqs in extras.items():
        assert scan(reqs) == [], (
            f"torch_npu unexpectedly declared in the {extra_name!r} extra"
        )
