<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang Ascend

**Status: Technical Preview — not production-ready.** An Ascend NPU port
of [`sgl-project/mini-sglang`](https://github.com/sgl-project/mini-sglang),
verified on **Ascend 910B1** with **TP=1** in **eager mode**. Every
capability listed below is backed by a signed Gate verdict under
[`docs/ascend_port/`](./docs/ascend_port).

**What works today (frozen and evidenced on 910B1):**

- Ascend Fused Infer Attention Score (**FIA**) attention backend
- Paged KV cache with radix prefix reuse
- Multi-step single-request generation
- Equal-length and ragged continuous batching
- Mixed-KV-length decode (per-request cached-length in one batch)
- Request lifecycle safety: allocation rollback, sampler commit
  atomicity, shutdown drain, overlap-safe abort, end-to-end abort
  acknowledgement

**Derived from:** [`sgl-project/mini-sglang`](https://github.com/sgl-project/mini-sglang)
(MIT). This fork retains the upstream MIT license — see [`LICENSE`](./LICENSE).
Upstream CUDA / FlashAttention / FlashInfer / H200 documentation is
preserved verbatim in the [Upstream documentation](#upstream-documentation)
section at the bottom of this file.

---

## Support matrix

```
Hardware:          Ascend 910B1
Model:             Qwen3-0.6B
Parallelism:       TP=1
Execution:         eager
Attention backend: npu_fia
Status:            validated
```

Gate freeze evidence:

| Capability | Verdict |
| --- | --- |
| Single-request eager on Ascend 910B1 | [`gate1_verdict.md`](./docs/ascend_port/gate1_verdict.md) |
| Single-request multistep generation, per-request stop tokens | [`gate2_1_multistep_verdict.md`](./docs/ascend_port/gate2_1_multistep_verdict.md) |
| Multi-request batching (equal-length, ragged, mixed-KV decode, dynamic admission) | [`gate2_2_multirequest_verdict.md`](./docs/ascend_port/gate2_2_multirequest_verdict.md) |
| Request lifecycle and cancel protocol (rollback, atomicity, drain, overlap abort, abort-ack) | [`gate2_3_request_lifecycle_verdict.md`](./docs/ascend_port/gate2_3_request_lifecycle_verdict.md) |

---

## Limitations

- **TP > 1 is not verified.** Gate 1 asserts HCCL init only; there is
  no attested cross-rank forward or decode.
- **Only Qwen3-0.6B has completed end-to-end validation on 910B1.**
  Other Qwen sizes, the Llama family, and MoE variants are not part of
  the current gate scope.
- **The full HTTP + ZMQ cross-process request path is not frozen.**
  Every lifecycle guarantee above was proven with the in-process
  offline driver plus committed hermetic tests.
- **No long-duration soak evidence.** There is no rolling-allocator or
  thousands-of-tick stability run.
- **No performance-leadership claim.** No benchmark numbers vs.
  SGLang / vLLM / any other framework on NPU are published.
- **Upstream acceptance status:** this fork is **not** merged into
  `sgl-project/mini-sglang`.

---

## Quick Start (Ascend 910B1)

### Prerequisites

- Ascend 910B1 host with **CANN** installed and a working
  `torch_npu` matching your `torch` version. See the Gate 1 verdict
  [§1](./docs/ascend_port/gate1_verdict.md) for the exact
  CANN / `torch` / `torch_npu` combination that has been validated.
- Linux (Ubuntu 22.04 was used for verification).
- Python 3.10+.

The upstream Quick Start assumed NVIDIA CUDA / FlashInfer /
`sgl-kernel`; **none of that is required for the Ascend path.**

### 1. Clone

```bash
git clone https://github.com/Ray-RP/mini-sglang-ascend.git
cd mini-sglang-ascend
git switch ascend-port
```

### 2. Environment

```bash
# Python 3.10+ recommended; matches the verification host.
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`pip install -e .` installs the base `minisgl` package. On the Ascend
target host you must have already installed CANN and `torch_npu`
outside this project (they are not on PyPI in a form the base extras
can pull in). The base dependencies (`torch`, `transformers`,
`apache-tvm-ffi`, `pyzmq`, `fastapi`, `uvicorn`, …) install cleanly on
top of a `torch_npu`-compatible `torch`.

### 3. Run

Deploy Qwen3-0.6B on a single Ascend die:

```bash
python -m minisgl \
    --model "Qwen/Qwen3-0.6B" \
    --attention-backend npu_fia \
    --tp 1
```

Or open the interactive shell:

```bash
python -m minisgl \
    --model "Qwen/Qwen3-0.6B" \
    --attention-backend npu_fia \
    --tp 1 \
    --shell
```

Use `/reset` in the shell to clear conversation history.

### 4. Tests

The Gate 2.3 hermetic test suite runs on the Ascend host under the
standard invocation (row counts recorded in
[`gate2_3_request_lifecycle_verdict.md`](./docs/ascend_port/gate2_3_request_lifecycle_verdict.md#11-test-surface-locked-at-this-gate)):

```bash
pytest -q -o addopts="" tests/misc/test_scheduler_prepare_batch_txn.py
pytest -q -o addopts="" tests/misc/test_engine_forward_sampler_atomic.py
pytest -q -o addopts="" tests/misc/test_scheduler_shutdown_drain.py
pytest -q -o addopts="" tests/misc/test_scheduler_overlap_abort_fence.py
pytest -q -o addopts="" tests/misc/test_scheduler_abort_ack.py
pytest -q -o addopts="" tests/misc/test_pyproject_config.py
```

---

## License and attribution

- This project is released under the **MIT License** — see
  [`LICENSE`](./LICENSE).
- The upstream copyright holder is `sgl-project` (Mini-SGLang, MIT).
- The Ascend port is a downstream fork; no upstream license change is
  made.

---

## Learn more

- **Ascend port verdicts:** [`docs/ascend_port/`](./docs/ascend_port)
  — one signed verdict per gate, describing exactly what has been
  proven on 910B1 and what has NOT.
- **Upstream feature reference:** [`docs/features.md`](./docs/features.md)
  (describes CUDA-target features; check each verdict for what the
  Ascend port actually supports).
- **Upstream architecture reference:** [`docs/structures.md`](./docs/structures.md).

---

## Upstream documentation

The sections below are preserved from the upstream Mini-SGLang README.
They describe the **CUDA / NVIDIA GPU** installation and benchmark
path. They are **not** the current Ascend Quick Start and are kept
here as historical reference for users tracking the upstream project.
For the Ascend 910B1 flow, use the [Quick Start (Ascend 910B1)](#quick-start-ascend-910b1)
section above.

### Upstream: Mini-SGLang overview (CUDA)

Mini-SGLang is a compact implementation of
[SGLang](https://github.com/sgl-project/sglang), designed to demystify
the complexities of modern LLM serving systems. With a compact codebase
of **~5,000 lines of Python**, it serves as both a capable inference
engine and a transparent reference for researchers and developers.

Upstream key features (CUDA target):

- **High Performance**: state-of-the-art throughput and latency with
  advanced optimizations on NVIDIA GPUs.
- **Lightweight & Readable**: a clean, modular, type-annotated codebase.
- **Advanced optimizations** (CUDA path):
  - **Radix Cache**: reuses KV cache for shared prefixes across requests.
  - **Chunked Prefill**: reduces peak memory usage for long-context
    serving.
  - **Overlap Scheduling**: hides CPU scheduling overhead with GPU
    computation.
  - **Tensor Parallelism**: scales inference across multiple GPUs.
  - **Optimized Kernels**: integrates **FlashAttention** and
    **FlashInfer** for maximum efficiency.

### Upstream: Quick Start (CUDA)

> **⚠️ Platform Support (upstream):** Mini-SGLang upstream targets
> **Linux only** (x86_64 and aarch64). Windows and macOS are not
> supported due to dependencies on Linux-specific CUDA kernels
> (`sgl-kernel`, `flashinfer`). Upstream recommends
> [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) on
> Windows or Docker for cross-platform compatibility.
>
> These prerequisites apply to the upstream CUDA path and are **not**
> the Ascend Quick Start above.

#### 1. Environment Setup (upstream CUDA)

Upstream recommends `uv` (compatible with `conda`).

```bash
# Create a virtual environment (Python 3.10+ recommended).
uv venv --python=3.12
source .venv/bin/activate
```

**Prerequisites (upstream CUDA):** upstream Mini-SGLang relies on
CUDA kernels that are JIT-compiled. Install the **NVIDIA CUDA Toolkit**
matching your driver, and verify with `nvidia-smi`.

#### 2. Installation (upstream CUDA)

Install upstream Mini-SGLang directly from its source:

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 Upstream: Installing on Windows (WSL2)</b></summary>

Since upstream Mini-SGLang requires Linux-specific dependencies,
Windows users should use WSL2:

1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Install CUDA on WSL2**:
   - Follow [NVIDIA's WSL2 CUDA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - Ensure your Windows GPU drivers support WSL2

3. **Install Mini-SGLang in WSL2**:
   ```bash
   # Inside WSL2 terminal
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **Access from Windows**: The server will be accessible at
   `http://localhost:8000` from Windows browsers and applications.

</details>

<details>
<summary><b>🐳 Upstream: Running with Docker (CUDA)</b></summary>

**Prerequisites**:
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **Build the Docker image**:
   ```bash
   docker build -t minisgl .
   ```

2. **Run the server**:
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **Run in interactive shell mode**:
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **Using Docker Volumes for persistent caches** (recommended for
   faster subsequent startups):
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

The Ascend port does not yet ship a validated Ascend-target
`Dockerfile`. The `Dockerfile` in this repository is the upstream
CUDA build and is **not** used for 910B1 verification.

</details>

### Upstream: Online serving (CUDA)

Launch an OpenAI-compatible API server with a single command (upstream
CUDA path):

```bash
# Deploy Qwen/Qwen3-0.6B on a single GPU
python -m minisgl --model "Qwen/Qwen3-0.6B"

# Deploy meta-llama/Llama-3.1-70B-Instruct on 4 GPUs with Tensor
# Parallelism, on port 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

The Ascend port has not attested the multi-GPU / `--tp 4` shape;
see [Limitations](#limitations).

### Upstream: Interactive shell (CUDA)

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

Use `/reset` to clear the chat history.

### Upstream: Benchmark (CUDA / H200)

The benchmark numbers below are from the upstream project on NVIDIA
H200 GPUs. **They are NOT the Ascend port's numbers, and the Ascend
port makes no performance claim.** See [Limitations](#limitations).

#### Upstream offline inference

See [bench.py](./benchmark/offline/bench.py) for more details. Set
`MINISGL_DISABLE_OVERLAP_SCHEDULING=1` for the ablation study on
overlap scheduling.

Upstream test configuration:

- Hardware: 1xH200 GPU.
- Model: Qwen3-0.6B, Qwen3-14B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100-1024 tokens
- Output Length: Randomly sampled between 100-1024 tokens

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

#### Upstream online inference

See [benchmark_qwen.py](./benchmark/online/bench_qwen.py) for more
details.

Upstream test configuration:

- Hardware: 4xH200 GPU, connected by NVLink.
- Model: Qwen3-32B
- Dataset:
  [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl),
  replaying first 1000 requests.

Upstream launch command:

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **Note**: If you encounter network issues when downloading models
> from HuggingFace, try using `--model-source modelscope` to download
> from ModelScope instead:
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)
