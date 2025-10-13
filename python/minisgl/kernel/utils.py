from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable


@lru_cache()
def _prepare_for_load() -> str:
    import os
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def load_kernel_module(
    path: str | Iterable[str],
    name: str,
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
    ldflags: Iterable[str] | None = None,
) -> Any:
    from torch.utils.cpp_extension import load

    if isinstance(path, str):
        path = (path,)

    abs_path = _prepare_for_load()
    build_dir = f"{abs_path}/{build}"
    os.makedirs(build_dir, exist_ok=True)
    extra_cflags = list(cflags) if cflags is not None else ["-O3", "-std=c++17"]
    extra_cuda_cflags = list(cuda_flags) if cuda_flags is not None else ["-O3", "-std=c++17"]
    extra_ldflags = list(ldflags) if ldflags is not None else None
    return load(
        name=name,
        sources=[f"{abs_path}/csrc/{p}" for p in path],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        build_directory=build_dir,
    )
