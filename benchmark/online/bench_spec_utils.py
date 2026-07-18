from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from minisgl.benchmark.client import RawResult


COPY_TEXT = (
    "The paged cache maps each logical token position to a physical KV slot. "
    "Verification scores a frontier token followed by draft tokens in one forward pass. "
    "Matching drafts become committed output, while rejected slots return to the allocator. "
)


def friendly_prompt(tokenizer: Any, target_tokens: int, request_id: int) -> str:
    """Copy-heavy prompt: instruct the model to reproduce a document verbatim."""
    instruction = (
        f"Request {request_id}: reproduce the document exactly, changing no words.\n<document>\n"
    )
    text = instruction + COPY_TEXT * max(2, target_tokens // 35) + "\n</document>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < target_tokens:
        ids += tokenizer.encode(COPY_TEXT * 8, add_special_tokens=False)
    return tokenizer.decode(ids[:target_tokens])


def output_prompt_overlap(prompt_ids: list[int], output_ids: list[int], *, n: int = 3) -> float:
    """Fraction of length-n output windows that also occur verbatim in the prompt.

    This is the prompt-lookup premise made measurable: high overlap means the
    drafter can copy long spans out of the prompt, so acceptance should be high.
    """
    if len(output_ids) < n or len(prompt_ids) < n:
        return 0.0
    prompt_windows = {tuple(prompt_ids[i : i + n]) for i in range(len(prompt_ids) - n + 1)}
    total = len(output_ids) - n + 1
    hits = sum(tuple(output_ids[i : i + n]) in prompt_windows for i in range(total))
    return hits / total


def mean_output_prompt_overlap(
    results: list[RawResult],
    tokenizer: Any,
    *,
    n: int = 3,
) -> float | None:
    """Mean prompt-copy overlap across results that captured their output text."""
    fractions: list[float] = []
    for result in results:
        if result.output_text is None:
            continue
        prompt_ids = tokenizer.encode(result.message, add_special_tokens=False)
        output_ids = tokenizer.encode(result.output_text, add_special_tokens=False)
        fractions.append(output_prompt_overlap(prompt_ids, output_ids, n=n))
    if not fractions:
        return None
    return statistics.mean(fractions)


def summarize(results: list[RawResult], output_len: int) -> dict[str, float]:
    """Compact machine keys for BENCH_REPEAT / BENCH_RESULT JSON lines."""
    start = min(r.tics[0] for r in results)
    end = max(r.tics[-1] for r in results)
    duration = end - start
    request_rates = [output_len / (r.tics[-1] - r.tics[0]) for r in results]
    return {
        "duration_s": duration,
        "aggregate_output_tps": len(results) * output_len / duration,
        "mean_request_output_tps": statistics.mean(request_rates),
        "p50_request_output_tps": statistics.median(request_rates),
    }


def maybe_wandb_run(config: dict[str, Any], *, enabled: bool):
    """Start a wandb run when requested and credentials exist; otherwise None."""
    if not enabled:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB: --wandb set but WANDB_API_KEY missing; skipping", flush=True)
        return None
    try:
        import wandb
    except ImportError:
        print("WANDB: package not installed; skipping", flush=True)
        return None

    project = os.environ.get("WANDB_PROJECT", "mini-sglang-spec")
    entity = os.environ.get("WANDB_ENTITY") or None
    arm = "spec-on" if config.get("spec") else "spec-off"
    model_slug = str(config.get("model", "unknown")).replace("/", "--")
    name = f"{arm}-{config.get('workload')}-bs{config.get('batch_size')}-{model_slug}"
    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        job_type="benchmark_cell",
        reinit=True,
    )


def wandb_summary_payload(
    final: dict[str, Any],
    *,
    batch_wall_median_s: float,
    spec_metrics: dict[str, int | float] | None,
    output_prompt_overlap: float | None = None,
) -> dict[str, float]:
    """Summary metrics for bar-chart compare (one value per cell)."""
    payload: dict[str, float] = {
        "throughput (tok/s)": float(final["aggregate_output_tps_median"]),
        "latency (s)": float(batch_wall_median_s),
    }
    if output_prompt_overlap is not None:
        payload["prompt_overlap"] = float(output_prompt_overlap)
    if spec_metrics is not None:
        for key in ("drafted_tokens", "accepted_tokens", "acceptance_rate"):
            if key in spec_metrics:
                payload[key] = float(spec_metrics[key])
    return payload


def latest_spec_metrics(server_log: str) -> dict[str, int | float] | None:
    """Parse the newest SPEC_METRICS JSON line from a server log."""
    if not server_log:
        return None
    path = Path(server_log)
    if not path.is_file():
        return None

    latest: dict[str, int | float] | None = None
    marker = "SPEC_METRICS "
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        try:
            payload = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            latest = {
                str(key): value
                for key, value in payload.items()
                if isinstance(value, (int, float))
            }
    return latest


def poll_spec_metrics(
    server_log: str,
    *,
    timeout_s: float = 30.0,
) -> dict[str, int | float] | None:
    """Wait briefly for a drained SPEC_METRICS line (fresh server ⇒ cell-local)."""
    deadline = time.monotonic() + timeout_s
    latest: dict[str, int | float] | None = None
    while time.monotonic() < deadline:
        latest = latest_spec_metrics(server_log)
        if latest is not None and int(latest.get("proposal_requests", 0)) > 0:
            return latest
        time.sleep(0.05)
    return latest


def log_server_artifact(wb: Any, server_log: str) -> None:
    """Attach the server log as a wandb artifact."""
    if not server_log:
        return
    path = Path(server_log)
    if not path.is_file():
        print(f"WANDB: server log not found: {path}", flush=True)
        return

    import wandb

    artifact = wandb.Artifact(f"server-log-{wb.id}", type="server-log")
    artifact.add_file(str(path))
    wb.log_artifact(artifact)
