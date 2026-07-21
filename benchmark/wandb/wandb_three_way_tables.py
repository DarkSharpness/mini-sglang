"""Log derived three-way comparison tables to wandb.

For a three-way benchmark group (spec-on/overlap-off, spec-off/overlap-off,
spec-off/overlap-on), logs one wandb.Table per workload plus one for the qwen
arrival trace, each with raw per-arm throughput and the derived metrics:

    overlap penalty (%)      = (T_off_no_overlap / T_off_overlap - 1) * 100
    fair spec gain (%)       = (T_spec_no_overlap / T_off_no_overlap - 1) * 100
    net deployment gain (%)  = (T_spec_no_overlap / T_off_overlap - 1) * 100

Usage:
    python benchmark/wandb/wandb_three_way_tables.py --group fi-spec-three-way-20260719-060855
"""

from __future__ import annotations

import argparse
import os
import sys

import wandb

DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "mini-sglang-spec")
METRIC = "throughput (tok/s)"

BASE = "spec-off-overlap-off"
OVERLAP = "spec-off-overlap-on"
SPEC = "spec-on-overlap-off"
DERIVED = ["overlap penalty (%)", "fair spec gain (%)", "net deployment gain (%)"]


def arm_name(config: dict) -> str:
    spec = "on" if config.get("spec") else "off"
    overlap = "on" if config.get("overlap") else "off"
    return f"spec-{spec}-overlap-{overlap}"


def table_rows(cells: dict, x_values: list) -> list[list]:
    rows = []
    for x in x_values:
        arms = cells[x]
        missing = {BASE, OVERLAP, SPEC} - set(arms)
        if missing:
            print(f"skipping x={x}: missing arms {sorted(missing)}")
            continue
        base, overlap, spec = arms[BASE], arms[OVERLAP], arms[SPEC]
        rows.append(
            [
                x,
                round(base, 1),
                round(overlap, 1),
                round(spec, 1),
                round((base / overlap - 1) * 100, 1),
                round((spec / base - 1) * 100, 1),
                round((spec / overlap - 1) * 100, 1),
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", required=True, help="wandb group to tabulate")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity", default=None, help="defaults to API default entity")
    parser.add_argument("--run-name", default="three-way-tables", help="name for the analysis run")
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity

    # (workload, x) -> arm -> throughput; newest run wins on duplicates.
    fixed: dict[tuple, dict] = {}
    trace: dict[float, dict] = {}
    for r in api.runs(f"{entity}/{args.project}", filters={"group": args.group}):
        if r.job_type == "analysis":
            continue
        if "qwen-trace" in r.tags:
            arm = arm_name(r.config)
            for h in r.scan_history(keys=["trace_scale", METRIC]):
                trace.setdefault(h["trace_scale"], {}).setdefault(arm, h[METRIC])
            continue
        workload, x, val = r.config.get("workload"), r.config.get("batch_size"), r.summary.get(METRIC)
        if workload and x is not None and val is not None:
            fixed.setdefault((workload, x), {}).setdefault(arm_name(r.config), float(val))

    if not fixed and not trace:
        sys.exit(f"no usable runs in group {args.group!r}")

    payload = {}
    for workload in sorted({w for w, _ in fixed}):
        cells = {x: arms for (w, x), arms in fixed.items() if w == workload}
        rows = table_rows(cells, sorted(cells))
        columns = ["batch_size", f"{BASE} (tok/s)", f"{OVERLAP} (tok/s)", f"{SPEC} (tok/s)", *DERIVED]
        payload[f"derived/{workload}"] = wandb.Table(columns=columns, data=rows)
    if trace:
        rows = table_rows(trace, sorted(trace))
        columns = ["trace_scale", f"{BASE} (tok/s)", f"{OVERLAP} (tok/s)", f"{SPEC} (tok/s)", *DERIVED]
        payload["derived/qwen_trace"] = wandb.Table(columns=columns, data=rows)

    run = wandb.init(
        entity=entity,
        project=args.project,
        group=args.group,
        name=args.run_name,
        job_type="analysis",
        tags=["analysis", "table"],
    )
    run.log(payload)
    print(f"logged {len(payload)} tables -> {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
