# Testing Guide

This repository follows a **Strict Mirroring** strategy for testing. If you add or modify code in `python/minisgl/`, you must add or modify the corresponding test in `tests/`.

## 1. Directory Structure

Test files must exactly mirror the source file path.

**Rule:** If source is `python/minisgl/<DIR>/<FILE>.py`, test is `tests/<DIR>/test_<FILE>.py`.

```text
python/minisgl/                tests/
├── kernel/                    ├── kernel/
│   ├── index.py      ─────────┼──→ test_index.py
│   ├── pynccl.py     ─────────┼──→ test_pynccl.py
│   └── store.py      ─────────┼──→ test_store.py
├── message/                   ├── message/
│   └── utils.py      ─────────┼──→ test_utils.py
└── ...                        └── ...
```

**Exceptions:**

- **`tests/e2e/`**: For heavy, full-system generation tests that load real models (e.g., Llama/Qwen).
- **Declarative Files**: Files that contain only configuration, exports (like `__init__.py`), or simple data definitions do not require a test file.

## 2. Naming Conventions

- **Files:** `test_<source_filename>.py`
- **Functions:** `test_<source_function>_<scenario>`
- **Classes:** `Test<SourceClassName>` (for grouping method tests)

## 3. Writing Tests

### Design Principles: Transparency over Abstraction

Mini-SGLang is a reference implementation. Tests should be easy to read and debug by students and researchers.

1. **Avoid Mocking**: Do not use complex mocking libraries (like `unittest.mock`) unless absolutely necessary (e.g., network calls). Since the codebase is lightweight, instantiate **real objects** (e.g., `Scheduler`, `KVCache`) in your tests.

2. **Explicit Data**: Do not build complex "Data Factories" or builders. Instantiate test data explicitly within the test function so the data structure is visible.

3. **Local References**: Keep reference implementations (Python/CPU versions of kernels) **inside the test file** or in a strictly adjacent helper. Do not build a shared "Reference Library."

### GPU & CUDA Tests

Do **not** hardcode device strings like `"cuda:0"` or `"cuda:1"`. Use the `cuda_device` fixture.

**Exception:** Distributed/Multi-process tests (marked with `@pytest.mark.distributed`) may manually manage device indices to simulate multi-GPU environments.

```python
import pytest
import torch
from minisgl.kernel import store_cache

# Example of a local reference implementation
def ref_store_cache(cache, ...):
    # Pure Python/Torch CPU logic
    ...

@pytest.mark.cuda
def test_store_cache(cuda_device):
    # Arrange
    cache = torch.randn(..., device=cuda_device)

    # Act
    store_cache(cache, ...)

    # Assert (Strict correctness)
    # Use torch.allclose for floats, exact match (==) for integers
    assert torch.allclose(cache, ref_store_cache(cache, ...))
```

### Markers

Use markers to categorize tests so CI can filter them.

| Marker | Usage |
| --- | --- |
| `@pytest.mark.cuda` | Test requires a GPU. Skipped automatically if no GPU is found. |
| `@pytest.mark.distributed` | Test uses multi-GPU/multi-process (e.g., NCCL). |
| `@pytest.mark.e2e` | Heavy system tests (e.g., loading weights). Slower to run. |

## 4. Benchmarks vs. Tests

- **Tests (`tests/`)**: Verify **correctness** (`assert result == expected` or `torch.allclose`). Focus on logic, edge cases, and numerical precision.
- **Benchmarks (`benchmark/`)**: Measure **performance** (Tokens/sec).
- *Note: Performance code must be verified by a corresponding correctness test in `tests/` before benchmarking.*

## 5. Running Tests

```bash
# Run everything (CPU + GPU if available)
pytest

# Run only fast unit tests (Skip heavy E2E)
pytest -m "not e2e"

# Run specific module
pytest tests/kernel/

# Run heavy E2E tests
pytest tests/e2e/
```
