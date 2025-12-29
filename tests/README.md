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
- **Functions:** `test_<source_function>_<condition>`
- *Bad:* `def test_ops():`
- *Good:* `def test_store_cache_correctness():`

## 3. Writing Tests

### GPU & CUDA Tests

Do **not** hardcode device strings like `"cuda:0"` or `"cuda:1"`. Use the `cuda_device` fixture.

```python
import pytest
import torch
from minisgl.kernel import store_cache

@pytest.mark.cuda
def test_store_cache_correctness(cuda_device):
    # Setup
    cache = torch.randn(..., device=cuda_device)

    # Run
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

## 4. Benchmarks vs. Tests

- **Tests (`tests/`)**: Verify **correctness** (`assert a == b`).
- **Benchmarks (`benchmark/`)**: Measure **performance** (Tokens/sec).
- *Do not put performance timers or throughput print statements in `tests/`.*

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
