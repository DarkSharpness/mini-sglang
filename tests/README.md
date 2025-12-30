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

- **`tests/e2e/`**: For heavy, full-system generation tests that load real models (e.g., Llama/Qwen). These do not mirror a single source file.

## 2. Naming Conventions

- **Files:** `test_<source_filename>.py`
- **Functions:** `test_<source_function>_<scenario>`
- **Classes:** `Test<SourceClassName>` (for grouping method tests)

## 3. Writing Tests

### GPU & CUDA Tests

Do **not** hardcode device strings like `"cuda:0"` or `"cuda:1"`. Use the `cuda_device` fixture.

```python
import pytest
import torch
from minisgl.kernel import store_cache

@pytest.mark.cuda
def test_store_cache(cuda_device):
    # Arrange
    cache = torch.randn(..., device=cuda_device)

    # Act
    store_cache(cache, ...)

    # Assert (Strict correctness only)
    assert torch.allclose(cache, expected)
```

### Markers

Use markers to categorize tests so CI can filter them.

| Marker | Usage |
| --- | --- |
| `@pytest.mark.cuda` | Test requires a GPU. Skipped automatically if no GPU is found. |
| `@pytest.mark.distributed` | Test uses multi-GPU/multi-process (e.g., NCCL). |
| `@pytest.mark.e2e` | Heavy system tests (e.g., loading weights). Slower to run. |

### Testing Classes

For Object-Oriented code, group tests inside a class that mirrors the source class name.

```python
# Source: class Req
class TestReq:
    def test_complete_one_updates_len(self):
        # ...

    def test_append_host_extends_input(self):
        # ...
```

## 4. Benchmarks vs. Tests

- **Tests (`tests/`)**: Verify **correctness** (`assert result == expected` or `torch.allclose`). Focus on logic, edge cases, and error handling.
- **Benchmarks (`benchmark/`)**: Measure **performance** (Tokens/sec).
- *Note: For CUDA kernels, correctness tests should verify results against a CPU reference implementation (see `ref_indexing` in `test_index.py`).*

## 5. Running Tests

```bash
# Run everything (CPU + GPU if available)
pytest

# Run only fast CPU tests (Skip GPU)
pytest -m "not cuda"

# Run specific module
pytest tests/kernel/

# Run heavy E2E tests
pytest tests/e2e/
```
