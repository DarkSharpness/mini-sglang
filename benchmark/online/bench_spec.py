from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
from typing import Any

from bench_spec_utils import (
    friendly_prompt,
    log_server_artifact,
    maybe_wandb_run,
    poll_spec_metrics,
    summarize,
    wandb_summary_payload,
)
from minisgl.benchmark.client import (
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from openai import AsyncOpenAI
from transformers import AutoTokenizer


async def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    async with AsyncOpenAI(
        base_url=f"http://127.0.0.1:{args.port}/v1", api_key="dummy"
    ) as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        if args.workload == "friendly":
            prompts = [
                friendly_prompt(tokenizer, args.input_len, i) for i in range(args.batch_size)
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

        wb = maybe_wandb_run(config, enabled=args.wandb)
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
                summary = summarize(results, args.output_len)
                summary["repeat"] = repeat
                summaries.append(summary)
                print("BENCH_REPEAT " + json.dumps(summary, sort_keys=True), flush=True)

            aggregate = [r["aggregate_output_tps"] for r in summaries]
            per_request = [r["mean_request_output_tps"] for r in summaries]
            batch_wall = [r["duration_s"] for r in summaries]
            final: dict[str, Any] = {
                **config,
                "aggregate_output_tps_median": statistics.median(aggregate),
                "aggregate_output_tps_min": min(aggregate),
                "aggregate_output_tps_max": max(aggregate),
                "mean_request_output_tps_median": statistics.median(per_request),
                "mean_request_output_tps_min": min(per_request),
                "mean_request_output_tps_max": max(per_request),
            }
            # Fresh server per cell (Modal) ⇒ process totals are this cell only.
            spec_metrics = (
                poll_spec_metrics(args.server_log)
                if args.spec and args.server_log
                else None
            )
            if spec_metrics is not None:
                final["spec_metrics"] = spec_metrics
                print("SPEC_METRICS " + json.dumps(spec_metrics, sort_keys=True), flush=True)
            elif args.spec and args.server_log:
                print("WARN: no SPEC_METRICS in server log after bench", flush=True)
            print("BENCH_RESULT " + json.dumps(final, sort_keys=True), flush=True)
            if wb is not None:
                payload = wandb_summary_payload(
                    final,
                    batch_wall_median_s=statistics.median(batch_wall),
                    spec_metrics=spec_metrics,
                )
                # One aggregated history row creates scalar charts without noisy
                # repeat-index plots; summary keeps the same values for run tables.
                wb.log(payload)
                for key, value in payload.items():
                    wb.summary[key] = value
        finally:
            if wb is not None:
                log_server_artifact(wb, args.server_log)
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
        help="Server log for SPEC_METRICS scrape + wandb artifact (empty disables)",
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
