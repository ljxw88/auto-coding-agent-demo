#!/bin/bash

# =============================================================================
# run-research.sh - Auto Research Agent Loop Runner
# =============================================================================
# Runs Claude Code in a loop to execute research experiments.
# Each session = one experiment (idea → implement → smoke test → train → eval → log → revert)
#
# Usage:
#   ./run-research.sh <number_of_experiments>
#   ./run-research.sh 10
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

LOG_DIR="./automation-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/research-$(date +%Y%m%d_%H%M%S).log"

log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" >> "$LOG_FILE"
    case $level in
        INFO)     echo -e "${BLUE}[INFO]${NC} ${message}" ;;
        SUCCESS)  echo -e "${GREEN}[SUCCESS]${NC} ${message}" ;;
        WARNING)  echo -e "${YELLOW}[WARNING]${NC} ${message}" ;;
        ERROR)    echo -e "${RED}[ERROR]${NC} ${message}" ;;
        PROGRESS) echo -e "${CYAN}[PROGRESS]${NC} ${message}" ;;
    esac
}

count_experiments() {
    ls experiments/exp_*.json 2>/dev/null | wc -l | tr -d ' '
}

count_pending_ideas() {
    python3 -c "
import json
with open('ideas_backlog.json') as f:
    b = json.load(f)
print(sum(1 for i in b.get('ideas', []) if i.get('status') == 'pending'))
" 2>/dev/null || echo "0"
}

if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_experiments>"
    echo "Example: $0 10"
    exit 1
fi

if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument must be a positive integer"
    exit 1
fi

TOTAL_RUNS=$1

echo ""
echo "========================================"
echo "   Auto Research Agent Runner"
echo "========================================"
echo ""

log "INFO" "Starting research loop: $TOTAL_RUNS experiments"
log "INFO" "Log: $LOG_FILE"

if [ ! -f "research-config.json" ]; then
    log "ERROR" "research-config.json not found. Please configure it first."
    exit 1
fi

if [ ! -f "baselines/original_baseline.json" ]; then
    log "ERROR" "baselines/original_baseline.json not found."
    exit 1
fi

INITIAL_EXP=$(count_experiments)
log "INFO" "Experiments completed at start: $INITIAL_EXP"

for ((run=1; run<=TOTAL_RUNS; run++)); do
    echo ""
    echo "========================================"
    log "PROGRESS" "Experiment run $run of $TOTAL_RUNS"
    echo "========================================"

    PENDING=$(count_pending_ideas)
    log "INFO" "Pending ideas: $PENDING"

    RUN_START=$(date +%s)
    RUN_LOG="$LOG_DIR/run-${run}-$(date +%Y%m%d_%H%M%S).log"

    PROMPT_FILE=$(mktemp)
    cat > "$PROMPT_FILE" << 'PROMPT_EOF'
Follow the workflow in CLAUDE.md exactly.

Run ONE complete research experiment this session:
1. Load context (research-config.json, baselines, ideas_backlog.json)
2. Refresh ideas if needed (< 3 pending)
3. Select the best pending idea
4. Create an experiment git branch
5. Apply the idea (modify code)
6. Run smoke test — if too slow, reject and stop
7. Run full training
8. Evaluate
9. Analyze results
10. Log the experiment
11. Git cleanup (keep branch if improved, delete if not)
12. Check stop conditions

Complete exactly ONE experiment then stop.
If blocked for any reason, print the blocking message and stop.
PROMPT_EOF

    log "INFO" "Launching Claude Code session..."

    if claude -p \
        --dangerously-skip-permissions \
        --allowed-tools "Bash,Edit,Read,Write,Glob,Grep,Task,WebSearch,WebFetch" \
        < "$PROMPT_FILE" 2>&1 | tee "$RUN_LOG"; then

        RUN_END=$(date +%s)
        log "SUCCESS" "Run $run completed in $((RUN_END - RUN_START))s"
    else
        RUN_END=$(date +%s)
        log "WARNING" "Run $run exited with code $? after $((RUN_END - RUN_START))s"
    fi

    rm -f "$PROMPT_FILE"

    AFTER_EXP=$(count_experiments)
    NEW_EXP=$((AFTER_EXP - INITIAL_EXP - (run - 1)))

    if [ "$NEW_EXP" -gt 0 ]; then
        log "SUCCESS" "New experiment(s) logged this run"
    else
        log "WARNING" "No new experiments logged this run"
    fi

    # Check if max_experiments reached
    MAX_EXP=$(python3 -c "import json; c=json.load(open('research-config.json')); print(c.get('max_experiments', 9999))" 2>/dev/null || echo "9999")
    if [ "$AFTER_EXP" -ge "$MAX_EXP" ]; then
        log "SUCCESS" "Reached max_experiments ($MAX_EXP). Stopping."
        break
    fi

    echo "---" >> "$LOG_FILE"

    if [ $run -lt $TOTAL_RUNS ]; then
        sleep 2
    fi
done

echo ""
echo "========================================"
log "SUCCESS" "Research loop finished!"
echo "========================================"

FINAL_EXP=$(count_experiments)
log "INFO" "Total experiments: $FINAL_EXP (was $INITIAL_EXP)"
log "INFO" "Full log: $LOG_FILE"
log "INFO" "Research notebook: research_log.md"
