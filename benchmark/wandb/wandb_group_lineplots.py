"""Log per-group line charts (metric vs batch size) to wandb.

For every metric, creates one chart per workload plus a combined all-workloads
chart, rendered with the native wandb custom-chart engine via the private
"line-colored" preset (solid lines, one color per series, per-series tooltip).

Usage:
    python benchmark/wandb/wandb_group_lineplots.py --group fi-spec-vs-overlap-20260718-161144
    python benchmark/wandb/wandb_group_lineplots.py --group <group> \
        --metric "throughput (tok/s)" --metric "latency (s)"
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import wandb

DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "mini-sglang-spec")
DEFAULT_METRICS = ["throughput (tok/s)", "avg request output (tok/s)"]
PRESET_NAME = "line-colored"

# Colorblind-validated categorical palette, assigned in fixed order.
COLORS = [
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
]

VEGA_SPEC = {
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "title": {
        "text": "${string:title}",
        "anchor": "middle",
        "fontSize": 17,
        "fontWeight": "bold",
        "color": "#111111",
    },
    "data": {"name": "wandb"},
    "mark": {"type": "line", "point": {"filled": True, "size": 45}, "strokeWidth": 2},
    "encoding": {
        "x": {
            "field": "${field:x}",
            "type": "quantitative",
            "axis": {"gridColor": "#e6e6e6", "titleFontWeight": "bold"},
        },
        "y": {
            "field": "${field:y}",
            "type": "quantitative",
            "scale": {"zero": True},
            "axis": {"gridColor": "#e6e6e6", "titleFontWeight": "bold"},
        },
        "color": {
            "field": "${field:stroke}",
            "type": "nominal",
            "scale": {"range": COLORS},
            "legend": {"title": None},
        },
        "tooltip": [
            {"field": "${field:stroke}", "type": "nominal"},
            {"field": "${field:x}", "type": "quantitative"},
            {"field": "${field:y}", "type": "quantitative", "format": ",.1f"},
        ],
    },
    "background": "#ffffff",
}


def ensure_preset(api: wandb.Api, entity: str) -> str:
    try:
        return api.create_custom_chart(
            entity=entity,
            name=PRESET_NAME,
            display_name="Line (colored by series)",
            spec_type="vega2",
            access="private",
            spec=VEGA_SPEC,
        )
    except Exception as exc:  # already exists (or no permission to recreate)
        print(f"reusing existing preset {entity}/{PRESET_NAME} ({exc})")
        return f"{entity}/{PRESET_NAME}"


def metric_slug(metric: str) -> str:
    slug = re.sub(r"\s*\(.*?\)", "", metric).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", slug).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", required=True, help="wandb group to chart")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity", default=None, help="defaults to API default entity")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help=f"summary key to chart; repeatable (default: {DEFAULT_METRICS})",
    )
    parser.add_argument("--x-key", default="batch_size", help="config key for the x axis")
    parser.add_argument("--run-name", default="lineplots", help="name for the analysis run")
    args = parser.parse_args()
    metrics = args.metrics or DEFAULT_METRICS

    api = wandb.Api()
    entity = args.entity or api.default_entity
    chart_id = ensure_preset(api, entity)

    rows: dict[str, list] = {m: [] for m in metrics}  # (workload, arm, x, value)
    for r in api.runs(f"{entity}/{args.project}", filters={"group": args.group}):
        if r.job_type == "analysis":
            continue
        workload = r.config.get("workload") or "all"
        x = r.config.get(args.x_key)
        if x is None:
            continue
        arm = r.config.get("arm") or r.name.split(f"-{workload}-")[0]
        for metric in metrics:
            val = r.summary.get(metric)
            if val is not None:
                rows[metric].append((workload, arm, x, float(val)))

    if not any(rows.values()):
        sys.exit(f"no runs in group {args.group!r} had metrics {metrics}")

    def chart(title: str, metric: str, data: list[list]):
        n_series = len({d[0] for d in data})
        if n_series > len(COLORS):
            print(f"WARNING: {title!r} has {n_series} series; colors repeat past {len(COLORS)}")
        table = wandb.Table(columns=["arm", args.x_key, metric], data=sorted(data))
        return wandb.plot_table(
            vega_spec_name=chart_id,
            data_table=table,
            fields={"x": args.x_key, "y": metric, "stroke": "arm"},
            string_fields={"title": title},
        )

    run = wandb.init(
        entity=entity,
        project=args.project,
        group=args.group,
        name=args.run_name,
        job_type="analysis",
        tags=["analysis", "lineplot"],
    )

    payload = {}
    for metric in metrics:
        slug = metric_slug(metric)
        suffix = "" if metric == metrics[0] else f" ({slug.replace('_', ' ')})"
        workloads = sorted({w for w, _, _, _ in rows[metric]})
        for workload in workloads:
            payload[f"{slug}/{workload}"] = chart(
                f"{workload.capitalize()}{suffix}",
                metric,
                [[a, x, v] for w, a, x, v in rows[metric] if w == workload],
            )
        if len(workloads) > 1:
            payload[f"{slug}/all"] = chart(
                f"All workloads{suffix}",
                metric,
                [[f"{a} / {w}", x, v] for w, a, x, v in rows[metric]],
            )

    run.log(payload)
    print(f"logged {len(payload)} charts -> {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
