from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

from minisgl.benchmark.client import (
    RawResult,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from openai import AsyncOpenAI
from transformers import AutoTokenizer


COPY_TEXT = (
    "The paged cache maps each logical token position to a physical KV slot. "
    "Verification scores a frontier token followed by draft tokens in one forward pass. "
    "Matching drafts become committed output, while rejected slots return to the allocator. "
)


def _friendly_prompt(tokenizer: Any, target_tokens: int, request_id: int) -> str:
    instruction = (
        f"Request {request_id}: reproduce the document exactly, changing no words.\n<document>\n"
    )
    text = instruction + COPY_TEXT * max(2, target_tokens // 35) + "\n</document>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < target_tokens:
        ids += tokenizer.encode(COPY_TEXT * 8, add_special_tokens=False)
    return tokenizer.decode(ids[:target_tokens])


def _summarize(results: list[RawResult], output_len: int) -> dict[str, float]:
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


def _maybe_wandb_run(config: dict[str, Any], *, enabled: bool):
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


def _log_server_artifact(wb: Any, server_log: str) -> None:
    """Attach the server log and expose its latest cumulative spec snapshot."""
    if not server_log:
        return
    path = Path(server_log)
    if not path.is_file():
        print(f"WANDB: server log not found: {path}", flush=True)
        return

    latest: dict[str, int | float] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "SPEC_METRICS "
        if marker not in line:
            continue
        try:
            payload = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            latest = payload

    if latest is not None:
        for key, value in latest.items():
            wb.summary[f"spec_cumulative/{key}"] = value

    import wandb

    artifact = wandb.Artifact(f"server-log-{wb.id}", type="server-log")
    artifact.add_file(str(path))
    wb.log_artifact(artifact)


async def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    async with AsyncOpenAI(
        base_url=f"http://127.0.0.1:{args.port}/v1", api_key="dummy"
    ) as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        if args.workload == "friendly":
            prompts = [
                _friendly_prompt(tokenizer, args.input_len, i) for i in range(args.batch_size)
            ]
        else:
            prompts = [
                generate_prompt(tokenizer, args.input_len) for _ in range(args.batch_size)
            ]

        config = {
            "model": model,
            "workload": args.workload,
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "repeats": args.repeats,
            "seed": args.seed,
            "spec": args.spec,
            "overlap": args.overlap,
            "revision": args.revision,
            "spec_num_draft": args.spec_num_draft,
            "spec_ngram_min": args.spec_ngram_min,
            "spec_ngram_max": args.spec_ngram_max,
        }
        print("BENCH_CONFIG " + json.dumps(config, sort_keys=True), flush=True)

        wb = _maybe_wandb_run(config, enabled=args.wandb)
        try:
            warmup_bs = min(args.batch_size, 8)
            await benchmark_one_batch(
                client,
                prompts[:warmup_bs],
                min(args.output_len, 32),
                model,
                pbar=False,
            )

            summaries: list[dict[str, float]] = []
            for repeat in range(args.repeats):
                results = await benchmark_one_batch(
                    client, prompts, args.output_len, model, pbar=False
                )
                process_benchmark_results(results)
                summary = _summarize(results, args.output_len)
                summary["repeat"] = repeat
                summaries.append(summary)
                print("BENCH_REPEAT " + json.dumps(summary, sort_keys=True), flush=True)
                if wb is not None:
                    wb.log(
                        {k: v for k, v in summary.items() if k != "repeat"},
                        step=repeat,
                    )

            aggregate = [r["aggregate_output_tps"] for r in summaries]
            per_request = [r["mean_request_output_tps"] for r in summaries]
            final = {
                **config,
                "aggregate_output_tps_median": statistics.median(aggregate),
                "aggregate_output_tps_min": min(aggregate),
                "aggregate_output_tps_max": max(aggregate),
                "mean_request_output_tps_median": statistics.median(per_request),
                "mean_request_output_tps_min": min(per_request),
                "mean_request_output_tps_max": max(per_request),
            }
            print("BENCH_RESULT " + json.dumps(final, sort_keys=True), flush=True)
            if wb is not None:
                for key in (
                    "aggregate_output_tps_median",
                    "aggregate_output_tps_min",
                    "aggregate_output_tps_max",
                    "mean_request_output_tps_median",
                    "mean_request_output_tps_min",
                    "mean_request_output_tps_max",
                ):
                    wb.summary[key] = final[key]
        finally:
            if wb is not None:
                _log_server_artifact(wb, args.server_log)
                wb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["friendly", "adversarial"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=bool(os.environ.get("WANDB_API_KEY")),
        help="Live-log this cell to wandb (default: on when WANDB_API_KEY is set)",
    )
    parser.add_argument("--spec", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--revision", default=os.environ.get("BENCH_REVISION", "unknown"))
    parser.add_argument(
        "--server-log",
        default="",
        help="Server log to attach to the same wandb run (empty disables)",
    )
    parser.add_argument("--spec-num-draft", type=int, default=4)
    parser.add_argument("--spec-ngram-min", type=int, default=1)
    parser.add_argument("--spec-ngram-max", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.input_len, args.output_len, args.repeats) < 1:
        parser.error("batch-size, input-len, output-len, and repeats must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
