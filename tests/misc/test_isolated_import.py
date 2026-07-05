"""Gate 0.3a — natural package imports must succeed without CUDA-only deps.

Each import is executed in a fresh Python subprocess with a ``sys.meta_path``
finder that raises ``ModuleNotFoundError`` for ``flashinfer``, ``sgl_kernel``,
``quack``, and ``torch_npu``. This guarantees:

1. The subject import truly runs against the real package layout (no
   ``importlib.util.spec_from_file_location`` bypass, no fake stub packages
   left in ``sys.modules`` by sibling test files).
2. Any transitive import of a blocked CUDA-only package surfaces as a hard
   failure, not a silently-swallowed exception.

Torch and other neutral runtime deps are *not* forcibly available in this
suite; the test host is expected to be a bare macOS dev box, so the subject
modules must additionally avoid module-scope ``import torch`` etc. That
guarantee is implicit in the requirement that they are Ascend-portable.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PY_ROOT = _REPO_ROOT / "python"


# The blocker is prepended to every subprocess script. It installs a
# ``MetaPathFinder`` that raises ``ModuleNotFoundError`` for any import whose
# top-level package is one of the four Ascend-forbidden CUDA deps. We also
# prepend ``python/`` to ``sys.path`` so subprocesses find the repo source
# without an ``pip install -e .``.
_HEADER = f"""\
import sys
sys.path.insert(0, {str(_PY_ROOT)!r})

import importlib.abc

_BLOCKED = {{"flashinfer", "sgl_kernel", "quack", "torch_npu"}}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.split(".", 1)[0]
        if top in _BLOCKED:
            raise ModuleNotFoundError(
                "Gate 0.3a: blocked CUDA-only dep " + repr(name)
            )
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run(body: str, *, timeout: float = 30.0) -> subprocess.CompletedProcess:
    script = _HEADER + body
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_ok(result: subprocess.CompletedProcess, marker: str) -> None:
    assert result.returncode == 0, (
        f"subprocess exited {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}"
    )
    assert marker in result.stdout, (
        f"expected marker {marker!r} not in stdout\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 1. Direct target modules: the three Gate 0.3a targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "minisgl.utils.device",
        "minisgl.distributed.backend",
        "minisgl.distributed.runtime",
    ],
)
def test_target_module_imports_without_cuda_deps(target: str) -> None:
    result = _run(f"import {target}\nprint('OK:{target}')\n")
    _assert_ok(result, f"OK:{target}")


# ---------------------------------------------------------------------------
# 2. Package __init__ itself must be safe (lazy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pkg", ["minisgl.utils", "minisgl.distributed"])
def test_package_init_does_not_pull_cuda_deps(pkg: str) -> None:
    """``import minisgl.utils`` alone must not load flashinfer/sgl_kernel/quack/torch_npu."""
    body = (
        f"import {pkg}\n"
        # ``sys.modules`` must not contain any blocked package after init.
        "import sys\n"
        "leaked = sorted(m for m in sys.modules if m.split('.', 1)[0] in "
        "{'flashinfer', 'sgl_kernel', 'quack', 'torch_npu'})\n"
        "assert leaked == [], f'leaked CUDA deps: {leaked!r}'\n"
        f"print('OK:{pkg}')\n"
    )
    result = _run(body)
    _assert_ok(result, f"OK:{pkg}")


# ---------------------------------------------------------------------------
# 3. Backward compatibility of public symbols (PEP 562 lazy resolution)
# ---------------------------------------------------------------------------


def test_utils_lazy_public_symbols_resolve() -> None:
    """``from minisgl.utils import <name>`` still returns the same callables."""
    body = (
        "from minisgl.utils import call_if_main, div_ceil, Registry\n"
        "assert callable(call_if_main)\n"
        "assert callable(div_ceil)\n"
        "assert Registry.__name__ == 'Registry'\n"
        # __dir__ must advertise the lazy names.
        "import minisgl.utils as u\n"
        "d = dir(u)\n"
        "assert 'init_logger' in d and 'Registry' in d and 'div_ceil' in d\n"
        "print('OK:utils-lazy')\n"
    )
    result = _run(body)
    _assert_ok(result, "OK:utils-lazy")


def test_utils_unknown_attribute_still_raises_attribute_error() -> None:
    """Lazy ``__getattr__`` must not mask typos."""
    body = (
        "import minisgl.utils\n"
        "try:\n"
        "    minisgl.utils.definitely_not_a_symbol\n"
        "except AttributeError as exc:\n"
        "    print('OK:attrerror:' + str(exc))\n"
        "else:\n"
        "    print('FAIL')\n"
    )
    result = _run(body)
    _assert_ok(result, "OK:attrerror")


def test_distributed_dir_advertises_lazy_names() -> None:
    body = (
        "import minisgl.distributed as d\n"
        "names = dir(d)\n"
        "for expected in ('DistributedInfo', 'get_tp_info', 'DistributedCommunicator',\n"
        "                 'enable_pynccl_distributed', 'destroy_distributed'):\n"
        "    assert expected in names, expected\n"
        "print('OK:distributed-dir')\n"
    )
    result = _run(body)
    _assert_ok(result, "OK:distributed-dir")


# ---------------------------------------------------------------------------
# 4. arch.py naturally imports and returns False on non-CUDA hosts
# ---------------------------------------------------------------------------


def test_arch_runs_under_natural_import() -> None:
    """``arch.is_sm90_supported()`` uses the device layer, must return False cleanly."""
    body = (
        "import minisgl.utils.arch as arch\n"
        # CUDA is not available on this host; the delegate must return False
        # without raising, and without importing any blocked CUDA package.
        "assert arch.is_sm90_supported() is False\n"
        "assert arch.is_sm100_supported() is False\n"
        "import sys\n"
        "leaked = sorted(m for m in sys.modules if m.split('.', 1)[0] in "
        "{'flashinfer', 'sgl_kernel', 'quack', 'torch_npu'})\n"
        "assert leaked == [], f'leaked CUDA deps: {leaked!r}'\n"
        "print('OK:arch')\n"
    )
    result = _run(body)
    _assert_ok(result, "OK:arch")


# ---------------------------------------------------------------------------
# 5. Runtime module still refuses to import ``torch_npu`` at module scope
# ---------------------------------------------------------------------------


def test_runtime_module_scope_does_not_import_torch_npu() -> None:
    body = (
        "import minisgl.distributed.runtime  # noqa: F401\n"
        "import sys\n"
        "assert 'torch_npu' not in sys.modules, sorted(sys.modules)\n"
        "print('OK:runtime-clean')\n"
    )
    result = _run(body)
    _assert_ok(result, "OK:runtime-clean")


# ---------------------------------------------------------------------------
# 6. Sanity: the blocker really does block direct imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocked", ["flashinfer", "sgl_kernel", "quack", "torch_npu"])
def test_blocker_actually_blocks_direct_import(blocked: str) -> None:
    body = (
        f"try:\n"
        f"    import {blocked}\n"
        f"except ModuleNotFoundError as exc:\n"
        f"    if 'Gate 0.3a' in str(exc):\n"
        f"        print('OK:blocked:{blocked}')\n"
        f"    else:\n"
        f"        print('FAIL:unexpected:' + str(exc))\n"
        f"else:\n"
        f"    print('FAIL:import-succeeded')\n"
    )
    result = _run(body)
    _assert_ok(result, f"OK:blocked:{blocked}")
