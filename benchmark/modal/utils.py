from __future__ import annotations

import json
import math
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


QWEN_TRACE_SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # bench_qwen.py replay order


SPEC_SUMMARY_KEYS = (
    "drafted_tokens",
    "accepted_tokens",
    "acceptance_rate",
    "proposal_requests",
    "proposal_hits",
    "draft_hit_rate",
)


def _trace_spec_summary(server_log: Path | None, n_requests: int) -> dict[str, float]:
    """Acceptance summary for a trace run, keyed like the fixed spec cells."""
    metrics = _last_spec_metrics(server_log) if server_log else None
    if not metrics:
        return {}
    summary = {k: float(metrics[k]) for k in SPEC_SUMMARY_KEYS if k in metrics}
    accepted, hits = metrics.get("accepted_tokens"), metrics.get("proposal_hits")
    if accepted is not None:
        if hits:
            summary["mean accepted len (tok)"] = float(accepted) / float(hits)
        if n_requests:
            summary["mean accepted tok per request"] = float(accepted) / float(n_requests)
    return summary


def log_qwen_trace_to_wandb(
    log_path: Path,
    *,
    model: str,
    spec: bool,
    overlap: bool,
    n_requests: int = 1000,
    server_log: Path | None = None,
) -> None:
    """Parse a bench_qwen output log and log one wandb run (a row per scale).

    bench_qwen.py stays wandb-free; this reads the per-scale summary blocks that
    process_benchmark_results prints. Skips quietly when wandb is unconfigured
    and never raises: the log in /results remains the source of truth.

    With server_log, acceptance stats are added to the run summary. One server
    serves the whole trace, so these are process totals across every scale, not
    per-scale values.
    """
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB: no API key in environment; skipping qwen-trace logging", flush=True)
        return
    try:
        import re

        import wandb

        text = log_path.read_text(errors="replace")
        tputs = [float(x) for x in re.findall(r"Throughput:\s+([\d.]+)\s+token/s", text)]
        rates = [float(x) for x in re.findall(r"Per-request output:\s+([\d.]+)\s+token/s", text)]
        e2es = [float(x) for x in re.findall(r"E2E:\s+([\d.]+)\s+s \(", text)]
        durs = [float(x) for x in re.findall(r"Duration:\s+([\d.]+)\s+s", text)]
        n_scales = len(QWEN_TRACE_SCALES)
        if not (len(tputs) == len(rates) == len(e2es) == len(durs) == n_scales):
            print(
                f"WANDB: expected {n_scales} summary blocks in {log_path.name}, found "
                f"{len(tputs)}/{len(rates)}/{len(e2es)}/{len(durs)}; skipping",
                flush=True,
            )
            return

        arm = ("spec-on" if spec else "spec-off") + ("-overlap-on" if overlap else "-overlap-off")
        model_slug = model.replace("/", "--")
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "mini-sglang-spec"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            group=os.environ.get("WANDB_RUN_GROUP") or None,
            name=f"{arm}-qwen-trace-{model_slug}",
            job_type="benchmark_cell",
            tags=["qwen-trace", arm, "workload:qwen-trace", f"model:{model_slug}"],
            config={
                "arm": arm,
                "workload": "qwen-trace",
                "model": model,
                "spec": spec,
                "overlap": overlap,
                "n_requests": n_requests,
                "trace_scales": QWEN_TRACE_SCALES,
                "source_log": log_path.name,
            },
            reinit=True,
        )
        for scale, tput, rate, e2e, dur in sorted(
            zip(QWEN_TRACE_SCALES, tputs, rates, e2es, durs)
        ):
            run.log(
                {
                    "trace_scale": scale,
                    "throughput (tok/s)": tput,
                    "avg request output (tok/s)": rate,
                    "duration (s)": dur,
                    "offered load (req/s)": n_requests / dur,
                    "avg in-flight requests": n_requests * e2e / dur,
                }
            )
        for key, value in _trace_spec_summary(server_log, n_requests).items():
            run.summary[key] = value
        run.finish()
        print(f"WANDB: logged qwen-trace run {run.name} ({run.url})", flush=True)
    except Exception as exc:  # never fail the benchmark over logging
        print(f"WANDB: qwen-trace logging failed: {exc}", flush=True)


def server_command(
    model: str,
    *,
    spec: bool,
    spec_num_draft: int = 4,
    spec_ngram_min: int = 1,
    spec_ngram_max: int = 3,
    attn: str = "fi",
    max_running_requests: int | None = None,
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
    if max_running_requests is not None:
        command += ["--max-running-requests", str(max_running_requests)]
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


def _latest_arm_dir(root: Path, group: str, arm: str) -> Path | None:
    """Newest lm-eval results dir for one arm, matched by the run-group stem."""
    prefix = f"quality-{group}-" if group else "quality-"
    matches = [
        path for path in root.glob(f"{prefix}*") if path.is_dir() and f"-{arm}-overlap" in path.name
    ]
    return max(matches, key=lambda path: path.name) if matches else None


def _load_accuracies(run_dir: Path) -> dict[tuple[str, str], float]:
    """(task, filter) -> exact_match from the newest results_*.json (any nesting)."""
    files = sorted(run_dir.rglob("results_*.json"))
    if not files:
        raise FileNotFoundError(f"No results_*.json under {run_dir}")
    accuracies: dict[tuple[str, str], float] = {}
    for task, metrics in json.loads(files[-1].read_text())["results"].items():
        for key, value in metrics.items():
            metric, _, filter_name = key.partition(",")
            if metric == "exact_match" and filter_name:
                accuracies[(task, filter_name)] = float(value)
    return accuracies


def _load_samples(run_dir: Path) -> dict[tuple[str, str, int], dict]:
    """(task, filter, doc_id) -> sample row from every samples_*.jsonl in the dir."""
    samples: dict[tuple[str, str, int], dict] = {}
    skipped = 0
    for path in sorted(run_dir.rglob("samples_*.jsonl")):
        task = path.name[len("samples_") :].rsplit("_", 1)[0]
        # Split on \n only: lm-eval writes generations with ensure_ascii=False, and
        # str.splitlines() would break records on \u2028/\x85 inside the JSON strings.
        for line in path.read_text(errors="replace").split("\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            samples[(task, row.get("filter", ""), int(row["doc_id"]))] = row
    if skipped:
        print(f"WARN: skipped {skipped} unparseable sample lines under {run_dir.name}")
    return samples


def _sign_test_p(b: int, c: int) -> float:
    """Two-sided exact sign test on discordant pairs (H0: flips are symmetric)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _last_spec_metrics(log_path: Path) -> dict | None:
    """Last SPEC_METRICS payload in a server log, if any (process totals)."""
    if not log_path.is_file():
        return None
    latest = None
    for line in log_path.read_text(errors="replace").splitlines():
        if "SPEC_METRICS" in line:
            latest = line
    if latest is None:
        return None
    try:
        return json.loads(latest.split("SPEC_METRICS", 1)[1].strip())
    except json.JSONDecodeError:
        return None


def _mean_output_chars(samples: dict[tuple[str, str, int], dict]) -> float:
    """Mean raw generation length, one sample per (task, doc_id)."""
    lengths: dict[tuple[str, int], int] = {}
    for (task, _filter, doc_id), row in samples.items():
        try:
            lengths[(task, doc_id)] = len(row["resps"][0][0])
        except (KeyError, IndexError, TypeError):
            continue
    return sum(lengths.values()) / len(lengths) if lengths else 0.0


def print_eval_comparison(root: Path, group: str = "") -> None:
    """Print a spec-off vs spec-on comparison of two lm-eval runs (any tasks).

    Aggregate accuracy alone hides direction, so the paired block splits every
    shared question into both-right / both-wrong / one-arm-only and sign-tests
    the discordant pairs: symmetric flips are kernel-numerics noise, one-sided
    flips are a real regression.
    """
    dirs: dict[str, Path] = {}
    for arm in ("spec-off", "spec-on"):
        arm_dir = _latest_arm_dir(root, group, arm)
        if arm_dir is None:
            raise FileNotFoundError(
                f"No {arm} results under {root} for group {group or '<any>'!r}; "
                "run benchmark/modal/app.py::quality for both arms first"
            )
        dirs[arm] = arm_dir

    accuracies = {arm: _load_accuracies(path) for arm, path in dirs.items()}
    samples = {arm: _load_samples(path) for arm, path in dirs.items()}

    print("=" * 88)
    print(f"SPEC QUALITY A/B — group {group or '<latest runs>'}")
    for arm, path in dirs.items():
        print(f"  {arm}: {path.name}")
    print("=" * 88)

    print(f"{'task':<30}{'filter':<20}{'spec-off':>10}{'spec-on':>10}{'delta':>10}")
    rows = sorted(set(accuracies["spec-off"]) | set(accuracies["spec-on"]))
    for task, filter_name in rows:
        off = accuracies["spec-off"].get((task, filter_name))
        on = accuracies["spec-on"].get((task, filter_name))
        off_text = f"{off:.4f}" if off is not None else "-"
        on_text = f"{on:.4f}" if on is not None else "-"
        delta = f"{on - off:+.4f}" if off is not None and on is not None else "-"
        print(f"{task:<30}{filter_name:<20}{off_text:>10}{on_text:>10}{delta:>10}")

    print("\nPaired answers (same question, both arms):")
    for task, filter_name in rows:
        both_right = both_wrong = off_only = on_only = 0
        for key, off_row in samples["spec-off"].items():
            if key[0] != task or key[1] != filter_name:
                continue
            on_row = samples["spec-on"].get(key)
            if on_row is None or "exact_match" not in off_row or "exact_match" not in on_row:
                continue
            off_hit, on_hit = bool(off_row["exact_match"]), bool(on_row["exact_match"])
            both_right += off_hit and on_hit
            both_wrong += not off_hit and not on_hit
            off_only += off_hit and not on_hit
            on_only += on_hit and not off_hit
        total = both_right + both_wrong + off_only + on_only
        if total == 0:
            print(f"  {task}/{filter_name}: no shared samples")
            continue
        p_value = _sign_test_p(off_only, on_only)
        print(
            f"  {task}/{filter_name}: both-right {both_right}  both-wrong {both_wrong}  "
            f"off-only-right {off_only}  on-only-right {on_only}  sign-test p={p_value:.3f}"
        )

    for arm in ("spec-off", "spec-on"):
        mean_chars = _mean_output_chars(samples[arm])
        run_dir = dirs[arm]
        metrics = _last_spec_metrics(run_dir.parent / f"{run_dir.name}.server.log")
        line = f"{arm}: mean output {mean_chars:.0f} chars"
        if metrics and metrics.get("drafted_tokens"):
            accepted, drafted = metrics.get("accepted_tokens", 0), metrics["drafted_tokens"]
            line += f"; {accepted}/{drafted} drafted tokens accepted ({accepted / drafted:.1%})"
        print(line)
    print("=" * 88)
