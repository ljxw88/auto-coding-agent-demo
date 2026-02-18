"""
Research Logger — records experiment results to disk.

Usage:
    python agents/research_logger.py --experiment-json '{ ... }'
    python agents/research_logger.py --experiment-file /path/to/exp.json

Writes:
  - experiments/exp_NNN.json  (structured record)
  - research_log.md           (human-readable append)
  - ideas_backlog.json        (marks idea as done)

If --update-rolling-best is set, also updates baselines/rolling_best.json.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_backlog, save_backlog, save_baseline, load_baseline,
    next_experiment_id, timestamp, EXPERIMENTS_DIR, RESEARCH_LOG
)


def write_experiment(exp: dict) -> str:
    exp_id = exp.get("id") or next_experiment_id()
    exp["id"] = exp_id
    path = EXPERIMENTS_DIR / f"{exp_id}.json"
    with open(path, "w") as f:
        json.dump(exp, f, indent=2)
    print(f"[logger] Written: {path}")
    return exp_id


def append_research_log(exp: dict) -> None:
    outcome = exp.get("outcome", "?")
    emoji = {"improved": "✅", "regression": "❌", "no_change": "❌", "too_slow": "⏱️", "failed": "💥"}.get(outcome, "❓")

    idea_title = exp.get("idea_title", exp.get("idea_id", "?"))
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    branch = exp.get("git_branch", "N/A")

    vs_orig = exp.get("vs_original_baseline", {})
    vs_roll = exp.get("vs_rolling_best", {})
    primary = exp.get("metrics", {}).get("primary", {})

    lines = [
        f"\n## {exp['id']} — {idea_title}",
        f"**Date**: {date_str}  **Branch**: `{branch}`",
        f"**Result**: {emoji} {outcome}  |  {primary.get('name','metric')}: {primary.get('value', 'N/A')}",
    ]

    if vs_orig.get("delta") is not None:
        lines.append(f"**vs Original**: {vs_orig['delta']:+.4f} ({vs_orig.get('delta_pct', 0):+.2f}%)")
    if vs_roll.get("delta") is not None:
        lines.append(f"**vs Rolling Best**: {vs_roll['delta']:+.4f} ({vs_roll.get('delta_pct', 0):+.2f}%)")

    if exp.get("smoke_test"):
        st = exp["smoke_test"]
        lines.append(f"**Smoke test**: {st.get('avg_seconds_per_epoch', '?'):.2f}s/epoch")

    if exp.get("conclusion"):
        lines.append(f"**Conclusion**: {exp['conclusion']}")

    lines.append(f"**Branch kept**: {'Yes' if exp.get('branch_kept') else 'No'}")
    lines.append("")

    with open(RESEARCH_LOG, "a") as f:
        f.write("\n".join(lines))

    print(f"[logger] Appended to research_log.md")


def mark_idea_done(idea_id: str) -> None:
    backlog = load_backlog()
    for idea in backlog.get("ideas", []):
        if idea["id"] == idea_id:
            idea["status"] = "done"
            break
    save_backlog(backlog)
    print(f"[logger] Marked idea {idea_id} as done")


def update_rolling_best(exp: dict) -> None:
    metrics = exp.get("metrics", {})
    data = {
        "_comment": "Updated whenever an experiment beats the current best. Starts as a copy of original_baseline.json.",
        "recorded_at": timestamp(),
        "source_experiment": exp.get("id"),
        "metrics": metrics,
        "notes": f"Set by experiment {exp.get('id')}: {exp.get('idea_title', '')}",
    }
    save_baseline("rolling_best", data)
    print(f"[logger] Updated rolling_best.json (new best: {metrics.get('primary', {}).get('value')})")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment-json", type=str, help="Experiment record as JSON string")
    group.add_argument("--experiment-file", type=str, help="Path to experiment JSON file")
    parser.add_argument("--update-rolling-best", action="store_true",
                        help="Also update baselines/rolling_best.json")
    args = parser.parse_args()

    if args.experiment_file:
        with open(args.experiment_file) as f:
            exp = json.load(f)
    else:
        exp = json.loads(args.experiment_json)

    exp_id = write_experiment(exp)
    append_research_log(exp)

    if exp.get("idea_id"):
        mark_idea_done(exp["idea_id"])

    if args.update_rolling_best:
        update_rolling_best(exp)

    print(f"\n[logger] Experiment {exp_id} logged successfully.")
    print(json.dumps({"status": "ok", "experiment_id": exp_id}))


if __name__ == "__main__":
    main()
