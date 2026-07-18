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


CASES = [
    {
        "name": "one-token-boundary",
        "prompt": "Continue with one short word: The sky is",
        "max_tokens": 1,
        "ignore_eos": True,
    },
    {
        "name": "three-token-boundary",
        "prompt": "Complete this sequence: alpha beta gamma alpha beta",
        "max_tokens": 3,
        "ignore_eos": True,
    },
    {
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
        "name": "open-ended",
        "prompt": "Invent a new scientific instrument and explain how it works.",
        "max_tokens": 128,
        "ignore_eos": True,
    },
    {
        "name": "eos-enabled",
        "prompt": "Answer with only the word yes.",
        "max_tokens": 128,
        "ignore_eos": False,
    },
    {
        "name": "shared-prefix-a",
        "prompt": (
            "Shared system context: be concise and factual. "
            "Shared system context: be concise and factual. Question: What is CUDA?"
        ),
        "max_tokens": 17,
        "ignore_eos": True,
    },
    {
        "name": "shared-prefix-b",
        "prompt": (
            "Shared system context: be concise and factual. "
            "Shared system context: be concise and factual. Question: What is a KV cache?"
        ),
        "max_tokens": 17,
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
) -> None:
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    cases = CASES if not case_names else [c for c in CASES if c["name"] in set(case_names)]
    llm = LLM(
        model_path=model,
        dtype=torch.bfloat16,
        attention_backend="fi",
        page_size=1,
        spec_algorithm=algorithm,
        spec_num_draft=4,
        spec_ngram_min=ngram_min,
        spec_ngram_max=max(3, ngram_min),
        cuda_graph_max_bs=0 if disable_cuda_graph else None,
    )
    try:
        generated: list[dict[str, Any]] = []
        if serial:
            for case in cases:
                params = SamplingParams(
                    temperature=0.0,
                    top_k=1,
                    max_tokens=case["max_tokens"],
                    ignore_eos=case["ignore_eos"],
                )
                generated.extend(llm.generate([case["prompt"]], [params]))
        else:
            params = [
                SamplingParams(
                    temperature=0.0,
                    top_k=1,
                    max_tokens=case["max_tokens"],
                    ignore_eos=case["ignore_eos"],
                )
                for case in cases
            ]
            generated = llm.generate([case["prompt"] for case in cases], params)
        payload = {
            "model": model,
            "algorithm": algorithm,
            "serial": serial,
            "cases": [
                {
                    "name": case["name"],
                    "max_tokens": case["max_tokens"],
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
) -> None:
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
    ]
    if serial:
        command.append("--serial-worker")
    if disable_cuda_graph:
        command.append("--disable-cuda-graph-worker")
    subprocess.run(command, check=True, env=env)


def _first_mismatch(left: list[int], right: list[int]) -> tuple[int, int | None, int | None] | None:
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i, a, b
    if len(left) != len(right):
        i = min(len(left), len(right))
        return i, left[i] if i < len(left) else None, right[i] if i < len(right) else None
    return None


def _compare(off: dict[str, Any], on: dict[str, Any]) -> None:
    for off_case, on_case in zip(off["cases"], on["cases"], strict=True):
        assert off_case["name"] == on_case["name"]
        mismatch = _first_mismatch(off_case["token_ids"], on_case["token_ids"])
        if mismatch is not None:
            index, expected, actual = mismatch
            raise AssertionError(
                f"{off_case['name']}: token {index} differs "
                f"(spec-off={expected}, spec-on={actual}); outputs preserved for diagnosis"
            )


def main() -> None:
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
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--fi-tensor-cores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align FI decode with prefill tensor-core math (default: enabled)",
    )
    parser.add_argument("--disable-cuda-graph-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
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
    _run_arm(
        args.model,
        "none",
        off_path,
        case_names=selected,
        serial=args.serial,
        ngram_min=args.ngram_min,
        disable_cuda_graph=args.disable_cuda_graph,
        fi_tensor_cores=args.fi_tensor_cores,
    )
    _run_arm(
        args.model,
        "ngram",
        on_path,
        case_names=selected,
        serial=args.serial,
        ngram_min=args.ngram_min,
        disable_cuda_graph=args.disable_cuda_graph,
        fi_tensor_cores=args.fi_tensor_cores,
    )
    _compare(json.loads(off_path.read_text()), json.loads(on_path.read_text()))
    print(f"PASS: {len(selected)} greedy cases match token-for-token")
    print(f"Artifacts: {output_dir}")
    if temp is not None:
        temp.cleanup()


if __name__ == "__main__":
    main()
