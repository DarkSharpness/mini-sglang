from __future__ import annotations

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
    .pip_install("uv")
    .add_local_file(LOCAL_DIR / "pyproject.toml", "/tmp/pyproject.toml", copy=True)
    .run_commands(
        "uv pip install --system -r /tmp/pyproject.toml",
        "rm -rf /root/.cache && ln -s /cache /root/.cache",
    )
    .env(
        {
            "PYTHONPATH": f"{APP_DIR}/python",
            "TOKENIZERS_PARALLELISM": "false",
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

gpu_config = {
    "image": image,
    "gpu": "H100",
    "timeout": 8 * 60 * 60,
    "volumes": {
        "/cache": cache_volume,
        "/results": results_volume,
    },
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
