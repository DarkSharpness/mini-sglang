from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

from utils import APP_DIR, PORT, bench_spec_command, run_and_tee, running_server, server_command

# Locally this file lives under benchmark/modal/. Modal hydrates it as
# /root/app.py, where the repository is already mounted at APP_DIR.
ENTRYPOINT = Path(__file__).resolve()
REPO_DIR = ENTRYPOINT.parents[2] if len(ENTRYPOINT.parents) > 2 else APP_DIR

app = modal.App("mini-sglang-benchmarks")
cache_volume = modal.Volume.from_name("mini-sglang-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("mini-sglang-results", create_if_missing=True)

# Dependencies stay cached while the repository is mounted as a thin source layer.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "libnuma1")
    .pip_install("uv==0.11.28")
    .add_local_file(REPO_DIR / "pyproject.toml", "/tmp/pyproject.toml", copy=True)
    .add_local_file(REPO_DIR / "uv.lock", "/tmp/uv.lock", copy=True)
    .run_commands(
        "cd /tmp && uv export --frozen --extra dev --no-emit-project"
        " --format requirements.txt | uv pip install --system -r -",
        "rm -rf /root/.cache && ln -s /cache /root/.cache",
    )
    .pip_install("wandb>=0.19.0")
    .env(
        {
            # benchmark/modal is on the path so the entrypoint's `import utils` resolves
            # to the mounted sibling module inside the container.
            "PYTHONPATH": f"{APP_DIR}/python:{APP_DIR}/benchmark/modal",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_PROJECT": "mini-sglang-spec",
        }
    )
    .add_local_dir(
        REPO_DIR,
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


def _maybe_prompt_wandb_env() -> None:
    """Prompt once for direct benchmark_spec launches; matrix scripts prompt themselves."""
    if os.environ.get("WANDB_API_KEY"):
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    if "benchmark_spec" not in " ".join(sys.argv):
        return

    import getpass

    try:
        api_key = getpass.getpass("wandb API key (blank to skip logging): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not api_key:
        return
    os.environ["WANDB_API_KEY"] = api_key

    project_default = os.environ.get("WANDB_PROJECT", "mini-sglang-spec")
    project = input(f"wandb project [{project_default}]: ").strip()
    os.environ["WANDB_PROJECT"] = project or project_default

    group_default = os.environ.get("WANDB_RUN_GROUP") or datetime.now().strftime(
        "bench-%Y%m%d-%H%M%S"
    )
    group = input(f"wandb run group [{group_default}]: ").strip()
    os.environ["WANDB_RUN_GROUP"] = group or group_default


def _wandb_secrets() -> list[modal.Secret]:
    _maybe_prompt_wandb_env()
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


def _commit_volumes() -> None:
    cache_volume.commit()
    results_volume.commit()


def _run_spec_cell(
    *,
    model: str,
    spec: bool,
    overlap: bool,
    workload: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    revision: str,
    spec_num_draft: int = 4,
    spec_ngram_min: int = 1,
    spec_ngram_max: int = 3,
) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arm = "spec-on" if spec else "spec-off"
    overlap_name = "overlap-on" if overlap else "overlap-off"
    stem = (
        f"{arm}-{overlap_name}-{model.replace('/', '--')}-{workload}-bs{batch_size}-"
        f"in{input_len}-out{output_len}-{timestamp}"
    )
    client_path = Path("/results") / f"{stem}.log"
    server_path = Path("/results") / f"{stem}.server.log"

    env = os.environ.copy()
    env["MINISGL_FLASHINFER_USE_TENSOR_CORES"] = "true"
    if not overlap:
        env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"

    command = server_command(
        model,
        spec=spec,
        spec_num_draft=spec_num_draft,
        spec_ngram_min=spec_ngram_min,
        spec_ngram_max=spec_ngram_max,
    )
    provenance = {
        "revision": revision,
        "model": model,
        "spec": spec,
        "overlap": overlap,
        "workload": workload,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "server_command": command,
    }

    with server_path.open("w") as server_output:
        server_output.write(json.dumps(provenance, sort_keys=True) + "\n")
        server_output.flush()
        with running_server(command, env=env, output=server_output):
            run_and_tee(
                bench_spec_command(
                    workload=workload,
                    batch_size=batch_size,
                    input_len=input_len,
                    output_len=output_len,
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
    return client_path, server_path


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
    attn: str = "fi",
    verbose_mismatches: bool = False,
) -> None:
    """Run offline equivalence and online server contracts on one H100."""
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
        "--attn",
        attn,
    ]
    if cases:
        command.extend(["--cases", cases])
    if serial:
        command.append("--serial")
    if disable_cuda_graph:
        command.append("--disable-cuda-graph")
    if verbose_mismatches:
        command.append("--verbose-mismatches")
    command.append("--fi-tensor-cores" if fi_tensor_cores else "--no-fi-tensor-cores")

    print("\n" + "=" * 88, flush=True)
    print("STAGE 1/2: GREEDY TOKEN EQUIVALENCE", flush=True)
    print("=" * 88, flush=True)
    equivalence = subprocess.run(command, cwd=APP_DIR, check=False)

    print("\n" + "=" * 88, flush=True)
    print("STAGE 2/2: ONLINE SERVER CONTRACTS", flush=True)
    print(
        "Checks: concurrent greedy+sampled traffic, client abort, post-abort recovery", flush=True
    )
    print("=" * 88, flush=True)

    env = os.environ.copy()
    env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
    if fi_tensor_cores:
        env["MINISGL_FLASHINFER_USE_TENSOR_CORES"] = "true"
    server_path = output_dir / "server-scenarios.log"
    server_error: str | None = None
    try:
        with server_path.open("w") as server_output:
            with running_server(
                server_command(model, spec=True, attn=attn),
                env=env,
                output=server_output,
            ):
                subprocess.run(
                    ["python", "tests/core/test_speculative_server_e2e.py", "--port", str(PORT)],
                    cwd=APP_DIR,
                    check=True,
                )
    except Exception as exc:
        server_error = f"{type(exc).__name__}: {exc}"
        print(f"FAIL: online server contracts did not complete: {server_error}", flush=True)
        if server_path.is_file():
            lines = server_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print(f"Last {min(40, len(lines))} server log lines:", flush=True)
            print("\n".join(lines[-40:]), flush=True)

    _commit_volumes()

    comparison_path = output_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text()) if comparison_path.is_file() else None
    equivalence_ok = equivalence.returncode == 0
    server_ok = server_error is None

    print("\n" + "=" * 88, flush=True)
    print("SPECULATIVE E2E FINAL SUMMARY", flush=True)
    print("=" * 88, flush=True)
    if comparison is not None:
        print(
            f"  [{'PASS' if equivalence_ok else 'FAIL'}] Greedy token equivalence: "
            f"{comparison['matched_cases']}/{comparison['total_cases']} cases matched",
            flush=True,
        )
        for failure in comparison["failures"]:
            print(
                f"         {failure['name']}: first difference at output token "
                f"{failure['token_index']} "
                f"({failure['spec_off_token']} != {failure['spec_on_token']})",
                flush=True,
            )
        metrics = comparison.get("spec_metrics", {})
        if metrics:
            print(
                "         Speculation activity: "
                f"{metrics.get('proposal_hits', 0)}/{metrics.get('proposal_requests', 0)} "
                f"proposal hits; {metrics.get('accepted_tokens', 0)}/"
                f"{metrics.get('drafted_tokens', 0)} drafted tokens accepted",
                flush=True,
            )
    else:
        print(
            f"  [FAIL] Greedy token equivalence: process exited {equivalence.returncode} "
            "before writing comparison.json",
            flush=True,
        )
    print(
        f"  [{'PASS' if server_ok else 'FAIL'}] Online server contracts: "
        + ("mixed traffic and abort recovery passed" if server_ok else str(server_error)),
        flush=True,
    )
    print(f"  Artifacts: {output_dir}", flush=True)
    print("=" * 88, flush=True)
    if not (equivalence_ok and server_ok):
        raise RuntimeError("speculative E2E failed; see the final summary above")


@app.function(**gpu_config)
def benchmark_spec(
    model: str = "Qwen/Qwen3-8B",
    spec: bool = True,
    overlap: bool = False,
    workload: str = "friendly",
    batch_size: int = 32,
    input_len: int = 1024,
    output_len: int = 1024,
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
        revision=revision,
        spec_num_draft=spec_num_draft,
        spec_ngram_min=spec_ngram_min,
        spec_ngram_max=spec_ngram_max,
    )
    print(f"Saved client/server logs: {paths}")
    _commit_volumes()


@app.function(**gpu_config)
def baseline(
    model: str = "Qwen/Qwen3-8B",
    benchmark: str = "qwen",
    spec: bool = False,
    overlap: bool = True,
) -> None:
    """Run one stock online benchmark against a configurable server arm."""
    benchmark_scripts = {
        "qwen": "benchmark/online/bench_qwen.py",
        "simple": "benchmark/online/bench_simple.py",
    }
    if benchmark not in benchmark_scripts:
        choices = ", ".join(sorted(benchmark_scripts))
        raise ValueError(f"Unknown benchmark {benchmark!r}; choose one of: {choices}")
    if spec and overlap:
        raise ValueError("Speculation currently requires overlap=False.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arm = "spec-on" if spec else "spec-off"
    overlap_name = "overlap-on" if overlap else "overlap-off"
    output_path = (
        Path("/results")
        / f"{benchmark}-{arm}-{overlap_name}-{model.replace('/', '--')}-{timestamp}.log"
    )
    env = os.environ.copy()
    env["MINISGL_FLASHINFER_USE_TENSOR_CORES"] = "true"
    if not overlap:
        env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
    try:
        with running_server(server_command(model, spec=spec), env=env):
            print(f"Server ready; running {benchmark!r} {arm}, {overlap_name} for {model}")
            run_and_tee(["python", benchmark_scripts[benchmark]], output_path)
            print(f"Saved benchmark output to {output_path}")
    finally:
        _commit_volumes()


@app.function(**gpu_config)
def gpu_shell() -> None:
    """Resource template for `modal shell benchmark/modal/app.py::gpu_shell`."""
    pass
