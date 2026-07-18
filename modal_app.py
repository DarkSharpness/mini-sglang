from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import modal

LOCAL_DIR = Path(__file__).parent
APP_DIR = Path("/root/mini-sglang")
PORT = 1919

app = modal.App("mini-sglang-benchmarks")

cache_volume = modal.Volume.from_name("mini-sglang-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("mini-sglang-results", create_if_missing=True)

# Keep dependency installation in a cached layer. The repository itself is mounted
# afterwards, so normal source edits do not trigger a full image rebuild.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install("uv==0.11.28")
    .add_local_file(LOCAL_DIR / "pyproject.toml", "/tmp/pyproject.toml", copy=True)
    .add_local_file(LOCAL_DIR / "uv.lock", "/tmp/uv.lock", copy=True)
    .run_commands(
        "cd /tmp && uv export --frozen --extra dev --no-emit-project"
        " --format requirements.txt | uv pip install --system -r -",
        "rm -rf /root/.cache && ln -s /cache /root/.cache",
    )
    # Thin layer so wandb can live-log without touching the locked dep set.
    .pip_install("wandb>=0.19.0")
    .env(
        {
            "PYTHONPATH": f"{APP_DIR}/python",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_PROJECT": "mini-sglang-spec",
        }
    )
    .add_local_dir(
        LOCAL_DIR,
        remote_path=str(APP_DIR),
        ignore=[
            ".git/**",
            ".venv/**",
            "**/__pycache__/**",
            "*.pdf",
            "claude/**",
        ],
    )
)

def _wandb_secrets() -> list[modal.Secret]:
    """Forward local wandb credentials into the container when present."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        return []
    payload = {"WANDB_API_KEY": api_key}
    for key in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_GROUP"):
        if value := os.environ.get(key):
            payload[key] = value
    return [modal.Secret.from_dict(payload)]


gpu_config = {
    "image": image,
    "gpu": "H100",
    "timeout": 8 * 60 * 60,
    "volumes": {
        "/cache": cache_volume,
        "/results": results_volume,
    },
    "secrets": _wandb_secrets(),
}


def _wait_for_server(server: subprocess.Popen[bytes], timeout: int = 20 * 60) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{PORT}/v1/models"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Mini-SGLang server exited with status {server.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Mini-SGLang server did not become ready within {timeout} seconds")


def _run_and_tee(command: list[str], output_path: Path) -> None:
    with output_path.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=APP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
            output.flush()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def _server_command(
    model: str,
    *,
    spec: bool,
    spec_num_draft: int = 4,
    spec_ngram_min: int = 1,
    spec_ngram_max: int = 3,
) -> list[str]:
    command = [
        "python",
        "-m",
        "minisgl",
        "--model",
        model,
        "--attn",
        "fi",
        "--page-size",
        "1",
        "--port",
        str(PORT),
    ]
    if spec:
        command += [
            "--spec-algorithm",
            "ngram",
            "--spec-num-draft",
            str(spec_num_draft),
            "--spec-ngram-min",
            str(spec_ngram_min),
            "--spec-ngram-max",
            str(spec_ngram_max),
        ]
    return command


def _stop_server(server: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def _bench_spec_command(
    *,
    workload: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    repeats: int,
    revision: str,
    spec: bool,
    overlap: bool,
    server_log: Path,
    spec_num_draft: int = 4,
    spec_ngram_min: int = 1,
    spec_ngram_max: int = 3,
) -> list[str]:
    command = [
        "python",
        "benchmark/online/bench_spec.py",
        "--workload",
        workload,
        "--batch-size",
        str(batch_size),
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--repeats",
        str(repeats),
        "--revision",
        revision,
        "--server-log",
        str(server_log),
        "--spec-num-draft",
        str(spec_num_draft),
        "--spec-ngram-min",
        str(spec_ngram_min),
        "--spec-ngram-max",
        str(spec_ngram_max),
        "--spec" if spec else "--no-spec",
        "--overlap" if overlap else "--no-overlap",
    ]
    # Live wandb when the container has credentials (forwarded from the laptop).
    if os.environ.get("WANDB_API_KEY"):
        command.append("--wandb")
    else:
        command.append("--no-wandb")
    return command


def _run_spec_cell(
    *,
    model: str,
    spec: bool,
    overlap: bool,
    workload: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    repeats: int,
    revision: str,
    spec_num_draft: int = 4,
    spec_ngram_min: int = 1,
    spec_ngram_max: int = 3,
) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = model.replace("/", "--")
    arm = "spec-on" if spec else "spec-off"
    overlap_name = "overlap-on" if overlap else "overlap-off"
    stem = (
        f"{arm}-{overlap_name}-{model_slug}-{workload}-bs{batch_size}-"
        f"in{input_len}-out{output_len}-{timestamp}"
    )
    client_path = Path("/results") / f"{stem}.log"
    server_path = Path("/results") / f"{stem}.server.log"
    env = os.environ.copy()
    # Match FI decode math to the prefill wrapper used by verify. Without this,
    # near-tied greedy logits can choose different tokens across the two kernels.
    env["MINISGL_FLASHINFER_USE_TENSOR_CORES"] = "true"
    if not overlap:
        env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"

    with server_path.open("w") as server_output:
        server_output.write(
            json.dumps(
                {
                    "revision": revision,
                    "model": model,
                    "spec": spec,
                    "overlap": overlap,
                    "workload": workload,
                    "batch_size": batch_size,
                    "input_len": input_len,
                    "output_len": output_len,
                    "repeats": repeats,
                    "server_command": _server_command(
                        model,
                        spec=spec,
                        spec_num_draft=spec_num_draft,
                        spec_ngram_min=spec_ngram_min,
                        spec_ngram_max=spec_ngram_max,
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )
        server_output.flush()
        server = subprocess.Popen(
            _server_command(
                model,
                spec=spec,
                spec_num_draft=spec_num_draft,
                spec_ngram_min=spec_ngram_min,
                spec_ngram_max=spec_ngram_max,
            ),
            cwd=APP_DIR,
            env=env,
            stdout=server_output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_for_server(server)
            _run_and_tee(
                _bench_spec_command(
                    workload=workload,
                    batch_size=batch_size,
                    input_len=input_len,
                    output_len=output_len,
                    repeats=repeats,
                    revision=revision,
                    spec=spec,
                    overlap=overlap,
                    server_log=server_path,
                    spec_num_draft=spec_num_draft,
                    spec_ngram_min=spec_ngram_min,
                    spec_ngram_max=spec_ngram_max,
                ),
                client_path,
            )
        finally:
            _stop_server(server)
    return client_path, server_path


def _run_spec_arm(
    *,
    model: str,
    spec: bool,
    workloads: tuple[str, ...],
    batch_sizes: list[int],
    input_len: int,
    output_len: int,
    repeats: int,
    revision: str,
) -> None:
    """Run many overlap-off cells; fresh server per cell so SPEC_METRICS stay local."""
    arm = "spec-on" if spec else "spec-off"
    for workload in workloads:
        for batch_size in batch_sizes:
            client_path, server_path = _run_spec_cell(
                model=model,
                spec=spec,
                overlap=False,
                workload=workload,
                batch_size=batch_size,
                input_len=input_len,
                output_len=output_len,
                repeats=repeats,
                revision=revision,
            )
            print(
                f"Completed {arm} {workload=} {batch_size=}: "
                f"client={client_path} server={server_path}"
            )


@app.function(image=image, timeout=30 * 60)
def cpu_tests() -> None:
    """Run deterministic CPU speculative tests in the Linux dependency image."""
    subprocess.run(
        [
            "pytest",
            "tests/core/test_speculative.py",
            "tests/core/test_cache_allocate.py",
            "-q",
            "--no-cov",
        ],
        cwd=APP_DIR,
        check=True,
    )


@app.function(**gpu_config)
def spec_e2e(
    model: str = "Qwen/Qwen3-0.6B",
    cases: str = "",
    serial: bool = False,
    ngram_min: int = 1,
    disable_cuda_graph: bool = False,
    fi_tensor_cores: bool = True,
) -> None:
    """Run token-for-token spec-off/spec-on equivalence on one H100."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("/results") / f"e2e-{model.replace('/', '--')}-{timestamp}"
    command = [
        "python",
        "tests/core/test_speculative_e2e.py",
        "--model",
        model,
        "--output-dir",
        str(output_dir),
        "--ngram-min",
        str(ngram_min),
    ]
    if cases:
        command.extend(["--cases", cases])
    if serial:
        command.append("--serial")
    if disable_cuda_graph:
        command.append("--disable-cuda-graph")
    command.append("--fi-tensor-cores" if fi_tensor_cores else "--no-fi-tensor-cores")
    subprocess.run(command, cwd=APP_DIR, check=True)
    env = os.environ.copy()
    env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
    if fi_tensor_cores:
        env["MINISGL_FLASHINFER_USE_TENSOR_CORES"] = "true"
    server_path = output_dir / "server-scenarios.log"
    with server_path.open("w") as server_output:
        server = subprocess.Popen(
            _server_command(model, spec=True),
            cwd=APP_DIR,
            env=env,
            stdout=server_output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_for_server(server)
            subprocess.run(
                ["python", "tests/core/test_speculative_server_e2e.py", "--port", str(PORT)],
                cwd=APP_DIR,
                check=True,
            )
        finally:
            _stop_server(server)
    cache_volume.commit()
    results_volume.commit()


@app.function(**gpu_config)
def benchmark_spec(
    model: str = "Qwen/Qwen3-8B",
    spec: bool = True,
    overlap: bool = False,
    workload: str = "friendly",
    batch_size: int = 32,
    input_len: int = 1024,
    output_len: int = 256,
    repeats: int = 3,
    revision: str = "working-tree-upload",
    spec_num_draft: int = 4,
    spec_ngram_min: int = 1,
    spec_ngram_max: int = 3,
) -> None:
    """Run one reproducible speculative benchmark cell."""
    if spec and overlap:
        raise ValueError("Speculation currently requires overlap=False.")
    if workload not in {"friendly", "adversarial"}:
        raise ValueError("workload must be friendly or adversarial")
    paths = _run_spec_cell(
        model=model,
        spec=spec,
        overlap=overlap,
        workload=workload,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        repeats=repeats,
        revision=revision,
        spec_num_draft=spec_num_draft,
        spec_ngram_min=spec_ngram_min,
        spec_ngram_max=spec_ngram_max,
    )
    print(f"Saved client/server logs: {paths}")
    cache_volume.commit()
    results_volume.commit()


@app.function(**gpu_config)
def spec_suite(
    model: str = "Qwen/Qwen3-8B",
    batch_sizes: str = "1,8,32,64",
    input_len: int = 1024,
    output_len: int = 256,
    repeats: int = 3,
    revision: str = "working-tree-upload",
) -> None:
    """Run the primary overlap-off A/B matrix; fresh server per cell."""
    sizes = [int(x) for x in batch_sizes.split(",")]
    for spec in (False, True):
        _run_spec_arm(
            model=model,
            spec=spec,
            workloads=("friendly", "adversarial"),
            batch_sizes=sizes,
            input_len=input_len,
            output_len=output_len,
            repeats=repeats,
            revision=revision,
        )
    cache_volume.commit()
    results_volume.commit()


@app.function(**gpu_config)
def baseline(
    model: str = "Qwen/Qwen3-8B",
    benchmark: str = "qwen",
) -> None:
    """Run an unmodified Mini-SGLang server and one online benchmark."""
    benchmark_scripts = {
        "qwen": "benchmark/online/bench_qwen.py",
        "simple": "benchmark/online/bench_simple.py",
    }
    if benchmark not in benchmark_scripts:
        choices = ", ".join(sorted(benchmark_scripts))
        raise ValueError(f"Unknown benchmark {benchmark!r}; choose one of: {choices}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = model.replace("/", "--")
    output_path = Path("/results") / f"baseline-{model_slug}-{benchmark}-{timestamp}.log"

    env = os.environ.copy()
    server = subprocess.Popen(
        [
            "python",
            "-m",
            "minisgl",
            "--model",
            model,
            "--attn",
            "fi",
            "--page-size",
            "1",
            "--port",
            str(PORT),
        ],
        cwd=APP_DIR,
        env=env,
        start_new_session=True,
    )
    try:
        _wait_for_server(server)
        print(f"Server ready; running {benchmark!r} baseline for {model}")
        _run_and_tee(["python", benchmark_scripts[benchmark]], output_path)
        print(f"Saved benchmark output to {output_path}")
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        cache_volume.commit()
        results_volume.commit()


@app.function(**gpu_config)
def gpu_shell() -> None:
    """Resource template for `modal shell modal_app.py::gpu_shell`."""
    pass
