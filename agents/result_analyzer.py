"""
Result Analyzer — compares experiment metrics against both baselines.

Usage:
    python agents/result_analyzer.py --metrics '{"primary": {"name": "accuracy", "value": 0.87}, "supporting": {}}'
    python agents/result_analyzer.py --metrics-file tmp_metrics.json

Prints a JSON analysis result to stdout (last block after '--- ANALYSIS JSON ---').
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_baseline, load_config, timestamp


def compare(new_value: float, baseline_value: float, direction: str) -> dict:
    if baseline_value is None:
        return {"delta": None, "improved": None, "note": "baseline not set"}

    delta = new_value - baseline_value
    if direction == "higher_is_better":
        improved = delta > 0
    else:
        improved = delta < 0

    return {
        "baseline_value": round(baseline_value, 6),
        "new_value": round(new_value, 6),
        "delta": round(delta, 6),
        "delta_pct": round(100 * delta / abs(baseline_value), 3) if baseline_value != 0 else None,
        "improved": improved,
    }


def analyze(metrics: dict) -> dict:
    config = load_config()
    direction = config.get("metric_direction", "higher_is_better")
    primary_name = config.get("primary_metric", "accuracy")

    primary_value = metrics.get("primary", {}).get("value")
    if primary_value is None:
        return {"status": "error", "message": f"Primary metric '{primary_name}' not found in metrics"}

    original = load_baseline("original_baseline")
    rolling = load_baseline("rolling_best")

    original_val = original.get("metrics", {}).get("primary", {}).get("value")
    rolling_val = rolling.get("metrics", {}).get("primary", {}).get("value")

    vs_original = compare(primary_value, original_val, direction)
    vs_rolling = compare(primary_value, rolling_val, direction)

    # Determine overall outcome
    if vs_rolling.get("improved"):
        outcome = "improved"
    elif vs_rolling.get("improved") is False:
        outcome = "regression" if vs_rolling["delta"] < 0 else "no_change"
    else:
        outcome = "unknown"

    return {
        "status": "ok",
        "timestamp": timestamp(),
        "primary_metric": primary_name,
        "new_value": primary_value,
        "vs_original_baseline": vs_original,
        "vs_rolling_best": vs_rolling,
        "outcome": outcome,
        "keep_branch": vs_rolling.get("improved", False),
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metrics", type=str, help="JSON string of metrics")
    group.add_argument("--metrics-file", type=str, help="Path to metrics JSON file")
    args = parser.parse_args()

    if args.metrics_file:
        with open(args.metrics_file) as f:
            raw = json.load(f)
        config = load_config()
        primary_name = config.get("primary_metric", "accuracy")
        metrics = {
            "primary": {"name": primary_name, "value": raw.get(primary_name)},
            "supporting": {k: v for k, v in raw.items() if k != primary_name},
        }
    else:
        metrics = json.loads(args.metrics)

    result = analyze(metrics)
    print("\n--- ANALYSIS JSON ---")
    print(json.dumps(result, indent=2))

    if result.get("outcome") == "improved":
        sys.exit(0)
    elif result.get("status") == "error":
        sys.exit(1)
    else:
        sys.exit(0)  # No improvement but not an error


if __name__ == "__main__":
    main()
