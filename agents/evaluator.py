"""
Evaluator — runs training (smoke test or full) and evaluation.

Usage:
    python agents/evaluator.py --mode smoke_test
    python agents/evaluator.py --mode full_train
    python agents/evaluator.py --mode evaluate

Smoke test:
  - Runs training for smoke_test_epochs (default: 10)
  - Measures per-epoch time
  - Exits with code 2 if avg per-epoch time > max_seconds_per_epoch

Full train:
  - Runs training for full_train_epochs (default: 500)

Evaluate:
  - Runs eval_command
  - Reads metrics_output_file
  - Prints JSON metrics to stdout

All modes write a structured JSON result to stdout (last line).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, timestamp, REPO_ROOT


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TOO_SLOW = 2  # smoke test: per-epoch time exceeded threshold


def run_command(cmd: str, timeout_seconds: int = None) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=str(REPO_ROOT),
            capture_output=True, text=True,
            timeout=timeout_seconds,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return EXIT_ERROR, "", f"Command timed out after {timeout_seconds}s"
    except Exception as e:
        return EXIT_ERROR, "", str(e)


def smoke_test(config: dict) -> dict:
    """
    Run training for smoke_test_epochs epochs.
    Measure per-epoch wall-clock time.
    Returns result dict with status and timing info.
    """
    epochs = config.get("smoke_test_epochs", 10)
    max_sec = config.get("max_seconds_per_epoch", 10)
    template = config.get("train_command_template", "")

    if not template:
        return {"status": "error", "message": "train_command_template not set in research-config.json"}

    cmd = template.format(epochs=epochs)
    print(f"[evaluator] Smoke test: running {epochs} epochs...")
    print(f"[evaluator] Command: {cmd}")

    start = time.time()
    # Timeout: allow max_sec * epochs * 3x buffer
    timeout = int(max_sec * epochs * 3)
    code, stdout, stderr = run_command(cmd, timeout_seconds=timeout)
    elapsed = time.time() - start

    if code != 0:
        return {
            "status": "failed",
            "message": f"Training failed (exit {code})",
            "stderr": stderr[-2000:],
            "stdout": stdout[-2000:],
        }

    avg_sec_per_epoch = elapsed / epochs
    print(f"[evaluator] Smoke test done in {elapsed:.1f}s ({avg_sec_per_epoch:.2f}s/epoch)")

    if avg_sec_per_epoch > max_sec:
        return {
            "status": "too_slow",
            "avg_seconds_per_epoch": round(avg_sec_per_epoch, 2),
            "threshold": max_sec,
            "message": f"Per-epoch time {avg_sec_per_epoch:.2f}s exceeds threshold {max_sec}s — rejecting idea",
        }

    return {
        "status": "ok",
        "avg_seconds_per_epoch": round(avg_sec_per_epoch, 2),
        "threshold": max_sec,
        "smoke_test_epochs": epochs,
        "total_elapsed": round(elapsed, 2),
    }


def full_train(config: dict) -> dict:
    """Run full training. Returns result dict."""
    epochs = config.get("full_train_epochs", 500)
    max_sec = config.get("max_seconds_per_epoch", 10)
    template = config.get("train_command_template", "")

    if not template:
        return {"status": "error", "message": "train_command_template not set in research-config.json"}

    cmd = template.format(epochs=epochs)
    # Timeout: 1.5x expected max
    timeout = int(max_sec * epochs * 1.5)
    print(f"[evaluator] Full training: {epochs} epochs (timeout {timeout}s)...")
    print(f"[evaluator] Command: {cmd}")

    start = time.time()
    code, stdout, stderr = run_command(cmd, timeout_seconds=timeout)
    elapsed = time.time() - start

    if code != 0:
        return {
            "status": "failed",
            "message": f"Training failed (exit {code})",
            "stderr": stderr[-2000:],
            "elapsed": round(elapsed, 2),
        }

    return {
        "status": "ok",
        "full_train_epochs": epochs,
        "total_elapsed": round(elapsed, 2),
        "avg_seconds_per_epoch": round(elapsed / epochs, 2),
    }


def evaluate(config: dict) -> dict:
    """Run eval command and read metrics from output file."""
    eval_cmd = config.get("eval_command", "")
    metrics_file = REPO_ROOT / config.get("metrics_output_file", "tmp_metrics.json")

    if not eval_cmd:
        return {"status": "error", "message": "eval_command not set in research-config.json"}

    print(f"[evaluator] Running eval: {eval_cmd}")
    code, stdout, stderr = run_command(eval_cmd, timeout_seconds=300)

    if code != 0:
        return {
            "status": "failed",
            "message": f"Eval failed (exit {code})",
            "stderr": stderr[-2000:],
        }

    if not metrics_file.exists():
        return {
            "status": "error",
            "message": f"Metrics file not found: {metrics_file}",
        }

    with open(metrics_file) as f:
        raw_metrics = json.load(f)

    primary_name = config.get("primary_metric", "accuracy")
    primary_value = raw_metrics.get(primary_name)

    return {
        "status": "ok",
        "metrics": {
            "primary": {"name": primary_name, "value": primary_value},
            "supporting": {k: v for k, v in raw_metrics.items() if k != primary_name},
        },
        "raw": raw_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Run training or evaluation")
    parser.add_argument(
        "--mode",
        choices=["smoke_test", "full_train", "evaluate"],
        required=True,
    )
    args = parser.parse_args()

    config = load_config()
    result = {"mode": args.mode, "timestamp": timestamp()}

    if args.mode == "smoke_test":
        result.update(smoke_test(config))
        if result.get("status") == "too_slow":
            print("\n--- EVALUATOR RESULT JSON ---")
            print(json.dumps(result, indent=2))
            sys.exit(EXIT_TOO_SLOW)

    elif args.mode == "full_train":
        result.update(full_train(config))
        if result.get("status") == "failed":
            print("\n--- EVALUATOR RESULT JSON ---")
            print(json.dumps(result, indent=2))
            sys.exit(EXIT_ERROR)

    elif args.mode == "evaluate":
        result.update(evaluate(config))
        if result.get("status") in ("failed", "error"):
            print("\n--- EVALUATOR RESULT JSON ---")
            print(json.dumps(result, indent=2))
            sys.exit(EXIT_ERROR)

    print("\n--- EVALUATOR RESULT JSON ---")
    print(json.dumps(result, indent=2))
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
