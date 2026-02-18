"""
Evaluator — runs training (smoke test or full) and evaluation.

Usage:
    python agents/evaluator.py --mode smoke_test
    python agents/evaluator.py --mode full_train [--wandb-name MY_RUN_NAME]
    python agents/evaluator.py --mode evaluate

Smoke test:
  - Runs training for smoke_test_epochs (default: 10) using smoke_test_command_template
  - No wandb — fast iteration check only
  - Measures per-epoch time
  - Exits with code 2 if avg per-epoch time > max_seconds_per_epoch

Full train:
  - Runs training for full_train_epochs (default: 600) using full_train_command_template
  - Launched inside a named tmux session (autoexp_train) for observability
  - Training output is teed to a temp log file; evaluator polls for completion
  - Optionally pass --wandb-name to set the W&B run name (defaults to autoexp_<timestamp>)

Evaluate:
  - Runs eval_command
  - Reads metrics_output_file
  - Prints JSON metrics to stdout

All modes write a structured JSON result to stdout (last line).
"""

import argparse
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, timestamp, REPO_ROOT


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TOO_SLOW = 2  # smoke test: per-epoch time exceeded threshold

# Temp files used during full training (in repo root for easy inspection)
TRAIN_LOG_FILE = REPO_ROOT / "tmp_train_log.txt"
TRAIN_SCRIPT_FILE = REPO_ROOT / "tmp_train_script.sh"
TMUX_SESSION = "autoexp_train"


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
    Uses smoke_test_command_template (no wandb).
    Measures per-epoch wall-clock time.
    Returns result dict with status and timing info.
    """
    epochs = config.get("smoke_test_epochs", 10)
    max_sec = config.get("max_seconds_per_epoch", 10)
    # Use dedicated smoke template; fall back to old train_command_template
    template = config.get("smoke_test_command_template", config.get("train_command_template", ""))

    if not template:
        return {"status": "error", "message": "smoke_test_command_template not set in research-config.json"}

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


def full_train(config: dict, wandb_name: str = "") -> dict:
    """
    Run full training via a tmux session for observability.
    Uses full_train_command_template with {epochs} and {wandb_name} substitutions.
    Polls a log file for TRAIN_EXIT_CODE: marker to detect completion.
    """
    epochs = config.get("full_train_epochs", 600)
    max_sec = config.get("max_seconds_per_epoch", 10)
    template = config.get("full_train_command_template", config.get("train_command_template", ""))

    if not template:
        return {"status": "error", "message": "full_train_command_template not set in research-config.json"}

    if not wandb_name:
        wandb_name = f"autoexp_{timestamp().replace(':', '').replace('-', '')[:15]}"

    cmd = template.format(epochs=epochs, wandb_name=wandb_name)
    # Timeout: 1.5x expected max
    timeout = int(max_sec * epochs * 1.5)

    print(f"[evaluator] Full training: {epochs} epochs (timeout {timeout}s)...")
    print(f"[evaluator] W&B run name: {wandb_name}")
    print(f"[evaluator] Command: {cmd}")
    print(f"[evaluator] Launching in tmux session '{TMUX_SESSION}'...")
    print(f"[evaluator] Attach with: tmux attach -t {TMUX_SESSION}")

    # Clean up any stale log file from a previous run
    if TRAIN_LOG_FILE.exists():
        TRAIN_LOG_FILE.unlink()

    # Write a shell script so we avoid nested quote escaping
    script_content = f"""#!/bin/bash
set -o pipefail
{cmd} 2>&1 | tee {TRAIN_LOG_FILE}
echo "TRAIN_EXIT_CODE:$?" >> {TRAIN_LOG_FILE}
"""
    TRAIN_SCRIPT_FILE.write_text(script_content)
    TRAIN_SCRIPT_FILE.chmod(TRAIN_SCRIPT_FILE.stat().st_mode | stat.S_IEXEC)

    # Kill any existing session with the same name before starting
    run_command(f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null || true")

    launch_code, _, launch_err = run_command(
        f"tmux new-session -d -s {TMUX_SESSION} 'bash {TRAIN_SCRIPT_FILE}'"
    )
    if launch_code != 0:
        return {
            "status": "failed",
            "message": f"Failed to launch tmux session: {launch_err}",
        }

    # Poll log file for completion marker
    start = time.time()
    poll_interval = 30  # seconds
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            run_command(f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null || true")
            return {
                "status": "failed",
                "message": f"Training timed out after {timeout}s ({elapsed:.0f}s elapsed)",
                "elapsed": round(elapsed, 2),
            }

        if TRAIN_LOG_FILE.exists():
            content = TRAIN_LOG_FILE.read_text(errors="replace")
            exit_lines = [l for l in content.splitlines() if l.startswith("TRAIN_EXIT_CODE:")]
            if exit_lines:
                exit_code = int(exit_lines[-1].split(":")[-1].strip())
                if exit_code != 0:
                    return {
                        "status": "failed",
                        "message": f"Training failed (exit {exit_code})",
                        "log_tail": content[-3000:],
                        "elapsed": round(elapsed, 2),
                    }
                break  # success

        print(f"[evaluator] Training in progress... {elapsed:.0f}s elapsed (polling every {poll_interval}s)")
        time.sleep(poll_interval)

    elapsed = time.time() - start
    return {
        "status": "ok",
        "full_train_epochs": epochs,
        "total_elapsed": round(elapsed, 2),
        "avg_seconds_per_epoch": round(elapsed / epochs, 2),
        "wandb_name": wandb_name,
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
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="",
        help="W&B run name for full_train mode (e.g. idea ID). Auto-generated if not set.",
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
        result.update(full_train(config, wandb_name=args.wandb_name))
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

