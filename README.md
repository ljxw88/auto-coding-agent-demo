# Auto Research Agent

An autonomous ML research loop that proposes ideas, implements them, trains/evaluates, records findings, and iterates — all driven by Claude Code.

> All orchestration files and agent scripts in this repo are AI-generated.

## How It Works

Each session runs one complete experiment:

```
Propose idea (LLM + arxiv)
    → Apply to code (Claude Code modifies files)
    → Smoke test (10 epochs, check per-epoch speed)
    → Full training (500 epochs)
    → Evaluate
    → Compare vs. original baseline AND rolling best
    → Log findings
    → Revert to main (keep branch only if improved)
    → Repeat
```

### Smoke Test Gate

Before committing to a full 500-epoch run, the agent runs 10 epochs and checks per-epoch wall-clock time. If it exceeds the threshold (default: 10s/epoch), the idea is **rejected as too slow** and the next idea is tried. This prevents wasted compute on ideas that add too much overhead.

### Two Baselines

- **Original baseline** — fixed at the start, never overwritten. All experiments are compared against it.
- **Rolling best** — updated whenever an experiment improves on the current best. Ideas are prioritized to beat this.

### Git Strategy

- Each experiment runs on its own branch: `experiment/{idea_id}-{timestamp}`
- If an experiment **improves** the rolling best: branch is kept and tagged
- If no improvement: branch is deleted, returning cleanly to `main`

---

## Setup

### Prerequisites

- Python 3.10+
- [Claude Code CLI](https://claude.ai/code) (`claude` command)
- `ANTHROPIC_API_KEY` environment variable

### Install

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./init.sh
```

### Configure Your Research Project

1. **Edit `research-config.json`** with your project's details:
   ```json
   {
     "project_name": "MyModel",
     "research_goal": "Improve accuracy on CIFAR-10",
     "primary_metric": "accuracy",
     "metric_direction": "higher_is_better",
     "train_command_template": "cd my-project && python train.py --epochs {epochs}",
     "eval_command": "cd my-project && python eval.py --output ../tmp_metrics.json",
     "metrics_output_file": "tmp_metrics.json",
     "smoke_test_epochs": 10,
     "full_train_epochs": 500,
     "max_seconds_per_epoch": 10
   }
   ```

2. **Add your research project** as a subdirectory (e.g., `my-project/`)

3. **Set the original baseline**:
   ```bash
   python agents/evaluator.py --mode evaluate
   # Copy the metrics value into baselines/original_baseline.json and rolling_best.json
   ```

### Run

```bash
# Run 10 experiment iterations
./run-research.sh 10

# Or run Claude Code manually (one experiment)
claude -p --dangerously-skip-permissions
```

---

## File Structure

```
/
├── CLAUDE.md                   ← agent workflow (auto-read by Claude Code)
├── research-config.json        ← your project config
├── ideas_backlog.json          ← auto-managed idea queue
├── research_log.md             ← human-readable experiment notebook
├── run-research.sh             ← loop runner
├── init.sh                     ← environment setup
├── requirements.txt
│
├── baselines/
│   ├── original_baseline.json  ← fixed original (never overwrite)
│   └── rolling_best.json       ← best so far (auto-updated)
│
├── experiments/                ← one JSON per experiment
│
├── agents/
│   ├── idea_generator.py       ← LLM + arxiv → ideas
│   ├── evaluator.py            ← smoke test + training + eval
│   ├── result_analyzer.py      ← compare metrics vs baselines
│   ├── research_logger.py      ← write experiment logs
│   └── utils.py                ← shared: LLM client, git helpers, arxiv
│
├── prompts/                    ← LLM prompt templates
│   ├── idea_generation.md
│   ├── code_modification.md
│   └── result_analysis.md
│
└── [your-research-project/]    ← attach here
```

## Viewing Results

```bash
# Quick overview
cat research_log.md

# Full experiment record
cat experiments/exp_001.json

# Current best
cat baselines/rolling_best.json
```

## Running Modes

| Mode | Command |
|------|---------|
| Full auto loop | `./run-research.sh 10` |
| Single experiment | `claude -p --dangerously-skip-permissions` |
| Just generate ideas | `python agents/idea_generator.py --count 5` |
| Just run smoke test | `python agents/evaluator.py --mode smoke_test` |
| Just evaluate | `python agents/evaluator.py --mode evaluate` |
