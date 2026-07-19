from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

APP_DIR = Path("/root/mini-sglang")
PORT = 1919


def wait_for_server(server: subprocess.Popen[str], timeout: int = 20 * 60) -> None:
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


def run_and_tee(command: list[str], output_path: Path) -> None:
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


def server_command(
    model: str,
    *,
    spec: bool,
    spec_num_draft: int = 4,
    spec_ngram_min: int = 1,
    spec_ngram_max: int = 3,
    attn: str = "fi",
) -> list[str]:
    command = [
        "python",
        "-m",
        "minisgl",
        "--model",
        model,
        "--attn",
        attn,
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


def _stop_server(server: subprocess.Popen[str]) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


@contextmanager
def running_server(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    output: IO[str] | None = None,
) -> Iterator[None]:
    """Start one server, wait for readiness, and always terminate its process group."""
    server = subprocess.Popen(
        command,
        cwd=APP_DIR,
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT if output else None,
        text=True,
        start_new_session=True,
    )
    try:
        wait_for_server(server)
        yield
    finally:
        _stop_server(server)


def bench_spec_command(
    *,
    workload: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    revision: str,
    spec: bool,
    overlap: bool,
    server_log: Path,
    spec_num_draft: int = 4,
    spec_ngram_min: int = 1,
    spec_ngram_max: int = 3,
) -> list[str]:
    return [
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
        "--wandb" if os.environ.get("WANDB_API_KEY") else "--no-wandb",
    ]
