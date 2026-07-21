from __future__ import annotations

import json
import os
import sys

import wandb

PROJECT = os.environ.get("WANDB_ENTITY_PROJECT", "alaydshah/mini-sglang-spec")


def main() -> None:
    api = wandb.Api()
    runs = api.runs(PROJECT)
    rows = []
    for run in runs:
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
        summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
        rows.append(
            {
                "name": run.name,
                "id": run.id,
                "group": run.group,
                "tags": list(run.tags),
                "state": run.state,
                "created_at": str(run.created_at),
                "config": cfg,
                "summary": summary,
            }
        )
    print(f"pulled {len(rows)} runs from {PROJECT}", file=sys.stderr)
    out = os.environ.get("OUT", "modal-results/wandb_runs.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
