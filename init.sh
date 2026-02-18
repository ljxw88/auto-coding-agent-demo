#!/bin/bash

# =============================================================================
# init.sh - Research Agent Initialization
# =============================================================================
# Run at the start of each session to verify the environment is ready.
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Initializing Auto Research Agent...${NC}"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -r requirements.txt

# Verify required files
echo "Checking required files..."
for f in research-config.json ideas_backlog.json baselines/original_baseline.json baselines/rolling_best.json; do
    if [ ! -f "$f" ]; then
        echo -e "${RED}✗ Missing: $f${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ $f${NC}"
done

# Verify ANTHROPIC_API_KEY
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}⚠ ANTHROPIC_API_KEY not set — idea generation will fail${NC}"
    echo "  Export it: export ANTHROPIC_API_KEY=sk-ant-..."
else
    echo -e "${GREEN}✓ ANTHROPIC_API_KEY is set${NC}"
fi

# Make scripts executable
chmod +x run-research.sh

echo ""
echo -e "${GREEN}✓ Initialization complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit research-config.json with your project's train/eval commands"
echo "  2. Attach your research project directory here"
echo "  3. Run: ./run-research.sh 10"
