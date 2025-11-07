from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, List, NamedTuple, Tuple

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
DEFAULT_LDFLAGS = []


if TYPE_CHECKING:
    from tvm_ffi import Module


class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool

    @property
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


def _make_name(*args: Any) -> str:
    return "minisgl__" + "_".join(str(arg) for arg in args)


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def load_aot(
    *args: Any,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load

    # include paths
    extra_include_paths = extra_include_paths or []
    extra_include_paths.append(str(KERNEL_PATH / "include"))

    # cpp paths
    cpp_files = cpp_files or []
    cpp_files = [str(KERNEL_PATH / "src" / f) for f in cpp_files]

    # cuda paths
    cuda_files = cuda_files or []
    cuda_files = [str(KERNEL_PATH / "src" / f) for f in cuda_files]

    # flags
    extra_cflags = DEFAULT_CFLAGS + (extra_cflags or [])
    extra_cuda_cflags = DEFAULT_CUDA_CFLAGS + (extra_cuda_cflags or [])
    extra_ldflags = DEFAULT_LDFLAGS + (extra_ldflags or [])

    return load(
        _make_name(*args),
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
    )


def load_jit(
    *args: Any,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load_inline

    # include paths
    extra_include_paths = extra_include_paths or []
    extra_include_paths.append(str(KERNEL_PATH / "include"))

    # cpp files
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in (cpp_files or [])]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in (cpp_wrappers or [])]

    # cuda files
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files or []]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in (cuda_wrappers or [])]

    # flags
    extra_cflags = DEFAULT_CFLAGS + (extra_cflags or [])
    extra_cuda_cflags = DEFAULT_CUDA_CFLAGS + (extra_cuda_cflags or [])
    extra_ldflags = DEFAULT_LDFLAGS + (extra_ldflags or [])

    return load_inline(
        _make_name(*args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
    )
