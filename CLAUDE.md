# Auto Research Agent — Workflow Instructions

## What This System Does

You are an autonomous ML research agent targeting **IQ signal device fingerprinting** with a **GatedDeltaNet** architecture.
In each session you run **one complete experiment**:
propose → implement → smoke test → train → evaluate → log → revert → done.

---

## ⚠️ HARD CONSTRAINTS — READ BEFORE ANYTHING ELSE

1. **Only modify `gateddeltanet` architecture code.** NEVER touch `causal_iq_transformer.py` or `causal_iq_kda.py` or any code specific to `transformer` / `kda` architectures.
2. **Never increase dropout or weight_decay above 0.** Experiments confirm `dropout=0` and `weight_decay=0.0` are optimal. Do NOT suggest or implement any change that raises them.
3. **Always preserve `--augment` and `--compile` flags** in all training commands.
4. **Smoke test uses NO wandb** — the `smoke_test_command_template` handles this automatically.
5. The SOTA baseline is **84% accuracy** on the LocD test set (tx 35-45, pkt 0-100). Training is somewhat unstable (82-84% range). Target is **86%+**.

---

## ⚠️ Two-Repo Architecture — READ THIS FIRST

This framework uses **two separate git repositories**:

| Repo | Path | Purpose |
|------|------|---------|
| **Framework repo** (this repo) | `/` (root) | Stores config, ideas, logs, baselines, agent scripts. **Never run git here.** |
| **Experiment repo** | `experiment_project/` | Your actual ML code. **All git operations happen here.** |

The `experiment_project/` directory is its own independent git repo.
You will **branch, commit, push, and merge exclusively inside `experiment_project/`**.

**NEVER run `git` commands from the root `/` directory.**
All git commands must be run as: `cd experiment_project && git ...`

---

## MANDATORY Workflow (follow every step, in order)

### Step 1: Load Context

Read these files to understand the current state:
```bash
cat research-config.json
cat baselines/original_baseline.json
cat baselines/rolling_best.json
cat ideas_backlog.json
```

Check `experiments/` for recent results:
```bash
ls experiments/
```

### Step 2: Ensure Baselines Are Set

If `baselines/original_baseline.json` has `"value": null`, you must run the baseline first:
```bash
# Run eval on the unmodified codebase to establish baseline
python agents/evaluator.py --mode evaluate
```
Then manually update `baselines/original_baseline.json` AND `baselines/rolling_best.json` with the results.
Commit baseline info in the framework repo is not needed — just update the JSON files.

### Step 3: Refresh Ideas (if backlog has fewer than 3 pending ideas)

```bash
python agents/idea_generator.py --count 5
```

This searches arxiv + uses LLM to generate ideas and appends them to `ideas_backlog.json`.

### Step 4: Select an Idea

Read `ideas_backlog.json` and pick the **highest expected_impact** idea with `"status": "pending"`.

Mark it in-progress by editing `ideas_backlog.json`:
```json
"status": "in_progress"
```

Note the idea's `id`, `title`, `hypothesis`, `implementation_notes`, and `files_to_modify`.

### Step 5: Create Experiment Branch

All git operations happen **inside `experiment_project/`** — never in the root repo.

```bash
cd experiment_project
git checkout -b experiment/{idea_id}-$(date +%Y%m%d_%H%M%S)
```

Record the branch name — you'll need it later.

### Step 6: Apply the Idea (Code Modification)

Read the relevant source files listed in `files_to_modify`.
Apply **minimal, surgical changes** to implement the idea.

Rules:
- Change ONLY what the idea requires
- Do NOT refactor unrelated code
- Keep all existing CLI interfaces working
- Add a brief comment like `# [experiment: idea_title]` near each change
- **NEVER modify** `causal_iq_transformer.py`, `causal_iq_kda.py`, or any non-gateddeltanet code
- **NEVER set dropout > 0 or weight_decay > 0**

### Step 7: Smoke Test (MANDATORY before full training)

```bash
python agents/evaluator.py --mode smoke_test
```

This runs 10 epochs **without wandb** using `smoke_test_command_template`.
Target: ~5-6 seconds/epoch. Threshold: 10 seconds/epoch.

Read the output JSON carefully:

**If `"status": "too_slow"`** (per-epoch time > threshold):
- The idea adds too much computational overhead
- Log it as rejected:
  ```bash
  python agents/research_logger.py --experiment-json '{
    "idea_id": "IDEA_ID",
    "idea_title": "IDEA_TITLE",
    "git_branch": "BRANCH_NAME",
    "outcome": "too_slow",
    "smoke_test": SMOKE_TEST_RESULT,
    "branch_kept": false,
    "conclusion": "Rejected: per-epoch time exceeded threshold"
  }'
  ```
- Delete the branch and return to main:
  ```bash
  cd experiment_project
  git checkout main
  git branch -D BRANCH_NAME
  ```
- **Stop this session** (smoke test rejection counts as a completed session)

**If `"status": "failed"`**: log as failed, delete branch, return to main, stop.

**If `"status": "ok"`**: proceed to Step 8.

### Step 8: Full Training

Pass the idea ID as the W&B run name:
```bash
python agents/evaluator.py --mode full_train --wandb-name {idea_id}
```

Training runs **inside a tmux session** named `autoexp_train`. You can observe it live:
```bash
tmux attach -t autoexp_train
```
(Detach with `Ctrl+B D` — do NOT kill the session while training.)

The evaluator polls for completion automatically and returns when done.

If training fails: log as failed, delete branch, return to main, stop.

### Step 9: Evaluate

```bash
python agents/evaluator.py --mode evaluate
```

This runs `test_causal_iq_classifier.py` on `/LOCAL/data/n210_2_leg_LocD.h5` (tx 35-45, pkt 0-100) and writes metrics to `tmp_metrics.json`.

Copy the `metrics` from the output JSON — you'll need it for analysis.

### Step 10: Analyze Results

```bash
python agents/result_analyzer.py --metrics-file tmp_metrics.json
```

Note the `outcome` field: `improved` | `regression` | `no_change`.

Write a 2-3 sentence conclusion based on:
- Was the hypothesis confirmed?
- By how much did the metric change?
- What might explain the result?

### Step 11: Log the Experiment

Build the experiment record JSON and log it:

```bash
python agents/research_logger.py \
  --experiment-json '{
    "idea_id": "IDEA_ID",
    "idea_title": "IDEA_TITLE",
    "git_branch": "BRANCH_NAME",
    "outcome": "improved|regression|no_change",
    "smoke_test": { ... },
    "metrics": { "primary": {"name": "...", "value": ...}, "supporting": {} },
    "vs_original_baseline": { ... },
    "vs_rolling_best": { ... },
    "branch_kept": true_or_false,
    "conclusion": "2-3 sentence conclusion"
  }' \
  [--update-rolling-best if outcome == "improved"]
```

### Step 12: Git Cleanup (inside `experiment_project/`)

Remember: all git commands run from within `experiment_project/`.

**If `outcome == "improved"`** (better than rolling best):
```bash
cd experiment_project
git add . && git commit -m "[experiment] IDEA_TITLE - improved (+DELTA on METRIC)"
git tag keeper/IDEA_ID
# Return to main (keep the branch alive for reference)
git checkout main
```

**If NOT improved**:
```bash
cd experiment_project
git checkout main
git branch -D BRANCH_NAME
```

### Step 13: Stop Conditions Check

Check and stop the session if ANY of these are true:
- `max_experiments` in config has been reached (count files in `experiments/`)
- `target_metric_value` in config is not null AND has been reached
- `max_consecutive_failures` consecutive experiments all failed/regressed

Output a brief summary of what happened this session and stop.

---

## ⚠️ Blocking Issues

If you cannot proceed (missing env vars, broken train/eval commands, etc.):

1. Do NOT commit
2. Do NOT mark anything as done
3. Print this:

```
🚫 BLOCKED — Human intervention required

Current experiment: [idea title]
Blocking reason: [specific issue]

What's needed:
1. [step 1]
2. [step 2]

After resolving: re-run run-research.sh to continue
```

---

## Project Structure

```
/                                   ← framework repo (no git branching here)
├── CLAUDE.md                   ← this file
├── research-config.json        ← project config
├── ideas_backlog.json          ← auto-managed idea queue
├── research_log.md             ← human-readable experiment notebook
├── run-research.sh             ← loop runner
├── baselines/
│   ├── original_baseline.json  ← SOTA 84% (never overwrite)
│   └── rolling_best.json       ← best so far (auto-updated)
├── experiments/                ← one JSON per experiment
├── agents/                     ← Python helper scripts
│   ├── idea_generator.py
│   ├── evaluator.py
│   ├── result_analyzer.py
│   └── research_logger.py
├── prompts/                    ← LLM prompt templates
│
└── experiment_project/         ← CausalIQ GatedDeltaNet code (its own git repo)
    ├── .git/                   ← independent git history
    ├── train_causal_iq_classifier.py
    ├── test_causal_iq_classifier.py
    ├── model_arch/
    │   ├── causal_iq_gateddeltanet.py   ← PRIMARY — only modify this
    │   ├── causal_iq_transformer.py     ← DO NOT TOUCH
    │   └── causal_iq_kda.py             ← DO NOT TOUCH
    └── ...
```

## Commands Reference

```bash
python agents/idea_generator.py --count 5                            # generate ideas
python agents/evaluator.py --mode smoke_test                         # 10-epoch speed check (no wandb)
python agents/evaluator.py --mode full_train --wandb-name idea_001   # full training run (tmux)
python agents/evaluator.py --mode evaluate                           # run eval, get metrics
python agents/result_analyzer.py --metrics-file tmp_metrics.json
python agents/research_logger.py --experiment-json '{...}'
```

## Key Rules

1. **One experiment per session** — complete the loop, then stop
2. **Always smoke test first** — never run full training without it
3. **Git lives in `experiment_project/`** — NEVER run git from the root repo
4. **Log everything** — even rejected and failed experiments
5. **Keep branch = improvement** — only keep branches that beat rolling best
6. **Update rolling_best when improved** — use `--update-rolling-best` flag
7. **GatedDeltaNet only** — never modify transformer or kda code
8. **No dropout/weight_decay** — never set either above 0

