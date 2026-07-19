"""Manual GPU E2E: compare genuine greedy decode with n-gram speculation.

Run from the repository root:
    python tests/core/test_speculative_e2e.py --model Qwen/Qwen3-0.6B

The parent launches each arm in a fresh process because Mini-SGLang's global
context, distributed process group, and CUDA engine are process-scoped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Distinct, non-repeating sentences so the copy case has genuine (not degenerate
# self-referential) prompt overlap, mirroring the friendly benchmark document.
_LONG_DOCUMENT = " ".join(
    (
        "Paged attention stores the key-value cache in fixed-size blocks instead of one contiguous buffer.",
        "A block table maps each logical position of a sequence to the physical block that holds its tokens.",
        "Prefill processes every prompt token in a single forward pass and writes their keys and values.",
        "Decoding then advances one token at a time, reading the whole cache to attend over the context.",
        "The decode phase is memory bound, since each step reloads the model weights to emit a single token.",
        "Speculative decoding proposes several future tokens and verifies them together in one pass.",
        "Prompt lookup decoding skips the draft model and copies candidate tokens from earlier text.",
        "It searches the prompt and the tokens generated so far for a matching n-gram suffix.",
        "Verification runs the target model over the frontier token followed by every drafted token.",
        "The scheduler keeps the longest correct prefix and returns rejected cache pages to the allocator.",
        "A bonus token is always emitted from the last verified position, guaranteeing forward progress.",
        "Continuous batching interleaves prefill and decode work so the accelerator rarely sits idle.",
        "CUDA graphs capture a fixed sequence of kernels once and replay them to remove launch overhead.",
        "Grouped-query attention lets several query heads share a single key-value head to cut cache size.",
        "Acceptance rate measures the fraction of drafted tokens that the target model actually keeps.",
        "A high acceptance rate turns each verification pass into several tokens of real progress.",
    )
)


CASES = [
    {
        # remain_len=1 makes max_draft=0, so this must use ordinary decode and
        # stop after exactly one token. It covers the smallest output budget.
        "name": "one-token-boundary",
        "prompt": "Continue with one short word: The sky is",
        "max_tokens": 1,
        "ignore_eos": True,
    },
    {
        # The repeated "alpha beta" suffix gives the drafter a continuation to
        # propose. Because max_tokens=3, the scheduler may draft at most two
        # tokens initially and must shrink that budget as the request finishes.
        "name": "three-token-boundary",
        "prompt": "Complete this sequence: alpha beta gamma alpha beta",
        "max_tokens": 3,
        "ignore_eos": True,
    },
    {
        # The requested output repeats prompt text, so generated suffixes should
        # match prompt n-grams and produce frequent multi-token verify commits.
        # The 128-token comparison catches bad acceptance or KV rollback state.
        "name": "copy-friendly",
        "prompt": (
            "Repeat the text between <doc> tags exactly.\n"
            "<doc>The scheduler verifies several draft tokens in one target-model "
            "forward pass. Accepted tokens become committed state; rejected KV slots "
            "return to the allocator. The scheduler verifies several draft tokens in "
            "one target-model forward pass.</doc>"
        ),
        "max_tokens": 128,
        "ignore_eos": True,
    },
    {
        # Novel output has little prompt overlap, exercising no-draft fallback
        # and rejected-draft correction rather than mostly accepted proposals.
        "name": "open-ended",
        "prompt": "Invent a new scientific instrument and explain how it works.",
        "max_tokens": 128,
        "ignore_eos": True,
    },
    {
        # Unlike the fixed-length cases, this allows EOS. Spec-on must stop at
        # the same EOS as spec-off even if EOS appears inside a verified run.
        "name": "eos-enabled",
        "prompt": "Answer with only the word yes.",
        "max_tokens": 128,
        "ignore_eos": False,
    },
    {
        # This pair shares a radix-cacheable prefix but has different suffixes.
        # Running both in one batch checks that verify state for one request does
        # not contaminate its sibling's token history or KV mapping.
        "name": "shared-prefix-a",
        "prompt": (
            "Shared system context: be concise and factual. "
            "Shared system context: be concise and factual. Question: What is CUDA?"
        ),
        "max_tokens": 17,
        "ignore_eos": True,
    },
    {
        # Second half of the shared-prefix pair; see shared-prefix-a above.
        "name": "shared-prefix-b",
        "prompt": (
            "Shared system context: be concise and factual. "
            "Shared system context: be concise and factual. Question: What is a KV cache?"
        ),
        "max_tokens": 17,
        "ignore_eos": True,
    },
    {
        # Benchmark-like edit task: most output tokens come from the document,
        # keeping active drafting/acceptance exercised for up to 1024 tokens.
        # This catches late state corruption or numerical divergence missed by
        # the shorter copy-friendly case.
        "name": "long-copy-edit",
        "prompt": (
            "Correct any spelling mistakes in the document below and return the full "
            "corrected text, changing nothing else.\n<document>\n"
            + _LONG_DOCUMENT
            + "\n</document>"
        ),
        "max_tokens": 1024,
        "ignore_eos": True,
    },
    {
        # Long low-overlap counterpart to long-copy-edit. In the same mixed
        # batch, drafts from another request can still route this request through
        # verify, exposing batch-coupled kernel divergence over 1024 tokens.
        "name": "long-open-ended",
        "prompt": (
            "Write a detailed technical essay explaining how a modern GPU inference "
            "server schedules, batches, and serves large language model requests."
        ),
        "max_tokens": 1024,
        "ignore_eos": True,
    },
]


def _worker(
    model: str,
    algorithm: str,
    output: Path,
    *,
    case_names: list[str] | None = None,
    serial: bool = False,
    ngram_min: int = 1,
    disable_cuda_graph: bool = False,
    attn: str = "fi",
) -> None:
    """Run one isolated engine arm and save token IDs/text for parent comparison."""
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    cases = CASES if not case_names else [c for c in CASES if c["name"] in set(case_names)]
    llm = LLM(
        model_path=model,
        dtype=torch.bfloat16,
        attention_backend=attn,
        page_size=1,
        spec_algorithm=algorithm,
        spec_num_draft=4,
        spec_ngram_min=ngram_min,
        spec_ngram_max=max(3, ngram_min),
        cuda_graph_max_bs=0 if disable_cuda_graph else None,
    )
    try:
        params = [
            SamplingParams(
                temperature=0.0,
                top_k=1,
                max_tokens=case["max_tokens"],
                ignore_eos=case["ignore_eos"],
            )
            for case in cases
        ]
        prompts = [case["prompt"] for case in cases]
        generated: list[dict[str, Any]] = []
        if serial:
            # Isolation diagnostic: one generate() call per case removes batch coupling.
            for prompt, sampling_params in zip(prompts, params, strict=True):
                generated.extend(llm.generate([prompt], [sampling_params]))
        else:
            # Primary gate: all cases share one mixed batch, matching server behavior.
            generated = llm.generate(prompts, params)
        payload = {
            "model": model,
            "algorithm": algorithm,
            "serial": serial,
            "cases": [
                {
                    "name": case["name"],
                    "max_tokens": case["max_tokens"],
                    "prompt": case["prompt"],
                    "text": result["text"],
                    "token_ids": result["token_ids"],
                }
                for case, result in zip(cases, generated, strict=True)
            ],
            "spec_metrics": llm.spec_metrics.as_dict(),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        llm.shutdown()


def _run_arm(
    model: str,
    algorithm: str,
    output: Path,
    *,
    case_names: list[str] | None = None,
    serial: bool = False,
    ngram_min: int = 1,
    disable_cuda_graph: bool = False,
    fi_tensor_cores: bool = True,
    attn: str = "fi",
) -> None:
    """Launch an arm in a fresh process so CUDA/global engine state cannot leak."""
    env = os.environ.copy()
    env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
    if fi_tensor_cores:
        env["MINISGL_FLASHINFER_USE_TENSOR_CORES"] = "true"
    command = [
        sys.executable,
        __file__,
        "--worker",
        "--model",
        model,
        "--algorithm",
        algorithm,
        "--output",
        str(output),
        "--case-names",
        ",".join(case_names or []),
        "--ngram-min",
        str(ngram_min),
        "--attn",
        attn,
    ]
    if serial:
        command.append("--serial-worker")
    if disable_cuda_graph:
        command.append("--disable-cuda-graph-worker")
    log_path = output.with_suffix(".log")
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            check=False,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print(
            f"ERROR: {algorithm} worker exited {result.returncode}; "
            f"last {min(40, len(lines))} log lines follow:",
            flush=True,
        )
        print("\n".join(lines[-40:]), flush=True)
        raise RuntimeError(f"{algorithm} worker failed; full log: {log_path}")


def _first_mismatch(left: list[int], right: list[int]) -> tuple[int, int | None, int | None] | None:
    """Return the first differing token, including unequal-length output tails."""
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i, a, b
    if len(left) != len(right):
        i = min(len(left), len(right))
        return i, left[i] if i < len(left) else None, right[i] if i < len(right) else None
    return None


def _print_mismatch(
    off_case: dict[str, Any],
    on_case: dict[str, Any],
    index: int,
    expected: int | None,
    actual: int | None,
    *,
    verbose: bool = False,
) -> None:
    """Print a compact mismatch; full generations already live in JSON artifacts."""
    off_ids, on_ids = off_case["token_ids"], on_case["token_ids"]
    start = max(0, index - 4)
    stop = index + 5
    print(
        f"  DIVERGED {off_case['name']}: output token {index} "
        f"(spec-off={expected}, spec-on={actual})",
        flush=True,
    )
    print(f"    spec-off token context: {off_ids[start:stop]}", flush=True)
    print(f"    spec-on  token context: {on_ids[start:stop]}", flush=True)
    if not verbose:
        return

    bar = "-" * 88
    print(bar, flush=True)
    print("PROMPT:", flush=True)
    print(off_case.get("prompt", "<unavailable>"), flush=True)
    print(bar, flush=True)
    print(f"SPEC-OFF generation ({len(off_ids)} tokens):", flush=True)
    print(off_case.get("text", "<unavailable>"), flush=True)
    print(bar, flush=True)
    print(f"SPEC-ON generation ({len(on_ids)} tokens):", flush=True)
    print(on_case.get("text", "<unavailable>"), flush=True)
    print(bar, flush=True)


def _compare(
    off: dict[str, Any],
    on: dict[str, Any],
    *,
    verbose_mismatches: bool = False,
) -> list[dict[str, Any]]:
    """Compare every case and return structured mismatches."""
    failures: list[dict[str, Any]] = []
    print("\nCASE RESULTS", flush=True)
    for off_case, on_case in zip(off["cases"], on["cases"], strict=True):
        assert off_case["name"] == on_case["name"]
        mismatch = _first_mismatch(off_case["token_ids"], on_case["token_ids"])
        if mismatch is None:
            print(
                f"  PASS     {off_case['name']}: " f"{len(off_case['token_ids'])} tokens match",
                flush=True,
            )
            continue
        index, expected, actual = mismatch
        _print_mismatch(
            off_case,
            on_case,
            index,
            expected,
            actual,
            verbose=verbose_mismatches,
        )
        failures.append(
            {
                "name": off_case["name"],
                "token_index": index,
                "spec_off_token": expected,
                "spec_on_token": actual,
            }
        )
    return failures


def main() -> None:
    """Run worker mode for one arm, or parent mode to compare fresh off/on arms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case names to run (default: all). Example: open-ended",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run each selected case in its own generate() batch (isolation diagnostic)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=["none", "ngram"], help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case-names", default="", help=argparse.SUPPRESS)
    parser.add_argument("--serial-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument(
        "--attn",
        choices=["fi", "fa"],
        default="fi",
        help="Attention backend for both arms (default: fi)",
    )
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--verbose-mismatches",
        action="store_true",
        help="Print full prompts and generations for mismatches (normally use JSON artifacts)",
    )
    parser.add_argument(
        "--fi-tensor-cores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align FI decode with prefill tensor-core math (default: enabled)",
    )
    parser.add_argument("--disable-cuda-graph-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        # Internal mode used by _run_arm; users normally invoke the parent mode.
        assert args.algorithm is not None and args.output is not None
        names = [n for n in args.case_names.split(",") if n] or None
        _worker(
            args.model,
            args.algorithm,
            args.output,
            case_names=names,
            serial=args.serial_worker,
            ngram_min=args.ngram_min,
            disable_cuda_graph=args.disable_cuda_graph_worker,
            attn=args.attn,
        )
        return

    if args.output_dir is None:
        temp = tempfile.TemporaryDirectory(prefix="minisgl-spec-e2e-")
        output_dir = Path(temp.name)
    else:
        temp = None
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    selected = [n for n in args.cases.split(",") if n] or [c["name"] for c in CASES]
    unknown = set(selected) - {c["name"] for c in CASES}
    if unknown:
        parser.error(f"Unknown case name(s): {sorted(unknown)}")

    off_path, on_path = output_dir / "spec-off.json", output_dir / "spec-on.json"
    print("=" * 88, flush=True)
    print("GREEDY TOKEN-EQUIVALENCE GATE", flush=True)
    print(
        f"model={args.model} cases={len(selected)} mode="
        f"{'serial' if args.serial else 'mixed-batch'} attn={args.attn} "
        f"ngram_min={args.ngram_min} cuda_graph={'off' if args.disable_cuda_graph else 'on'} "
        f"fi_tensor_cores={args.fi_tensor_cores}",
        flush=True,
    )
    print("Meaning: spec-on must emit exactly the same token IDs as genuine spec-off.", flush=True)
    print("=" * 88, flush=True)
    # Genuine baseline first, then active n-gram speculation in a separate process.
    print("[1/3] Running genuine spec-off baseline in a fresh process...", flush=True)
    _run_arm(
        args.model,
        "none",
        off_path,
        case_names=selected,
        serial=args.serial,
        ngram_min=args.ngram_min,
        disable_cuda_graph=args.disable_cuda_graph,
        fi_tensor_cores=args.fi_tensor_cores,
        attn=args.attn,
    )
    print("[2/3] Running n-gram spec-on in a fresh process...", flush=True)
    _run_arm(
        args.model,
        "ngram",
        on_path,
        case_names=selected,
        serial=args.serial,
        ngram_min=args.ngram_min,
        disable_cuda_graph=args.disable_cuda_graph,
        fi_tensor_cores=args.fi_tensor_cores,
        attn=args.attn,
    )
    print("[3/3] Comparing generated token IDs...", flush=True)
    off_payload = json.loads(off_path.read_text())
    on_payload = json.loads(on_path.read_text())
    failures = _compare(
        off_payload,
        on_payload,
        verbose_mismatches=args.verbose_mismatches,
    )
    matched = len(selected) - len(failures)
    result = {
        "status": "pass" if not failures else "fail",
        "model": args.model,
        "matched_cases": matched,
        "total_cases": len(selected),
        "failures": failures,
        "spec_metrics": on_payload.get("spec_metrics", {}),
    }
    result_path = output_dir / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print("\nGREEDY EQUIVALENCE SUMMARY", flush=True)
    print(
        f"  {'PASS' if not failures else 'FAIL'}: "
        f"{matched}/{len(selected)} cases match token-for-token",
        flush=True,
    )
    if failures:
        print(
            "  Full prompts, generations, token IDs, and the machine-readable comparison "
            f"are in {output_dir}",
            flush=True,
        )
        print(
            "  Re-run with --verbose-mismatches only when full console output is useful.",
            flush=True,
        )
    else:
        print(f"  Artifacts: {output_dir}", flush=True)
    if temp is not None:
        temp.cleanup()
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
