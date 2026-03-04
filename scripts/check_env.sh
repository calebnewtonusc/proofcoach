#!/usr/bin/env bash
# ProofCoach — Environment Check
set -euo pipefail

ERRORS=0

check() {
  local name="$1"
  local cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo "[OK] $name"
  else
    echo "[FAIL] $name"
    ERRORS=$((ERRORS + 1))
  fi
}

echo "=== ProofCoach Environment Check ==="

# Python
check "Python 3.11+" "python --version | grep -E 'Python 3\.(11|12|13)'"

# Core packages
check "torch" "python -c 'import torch; assert torch.cuda.is_available()'"
check "transformers" "python -c 'import transformers'"
check "peft" "python -c 'import peft'"
check "trl" "python -c 'import trl'"
check "deepspeed" "python -c 'import deepspeed'"
check "vllm" "python -c 'import vllm'"
check "datasets" "python -c 'import datasets'"
check "anthropic" "python -c 'import anthropic'"
check "loguru" "python -c 'import loguru'"

# Lean 4
check "lean (Lean 4)" "lean --version"

# GPUs
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [ "$GPU_COUNT" -ge 18 ]; then
  echo "[OK] GPUs: $GPU_COUNT × detected (need 18 for training)"
elif [ "$GPU_COUNT" -gt 0 ]; then
  echo "[WARN] GPUs: $GPU_COUNT detected (need 18 for full training, $GPU_COUNT for testing)"
else
  echo "[FAIL] No GPUs detected"
  ERRORS=$((ERRORS + 1))
fi

# Check VRAM
if command -v nvidia-smi &>/dev/null; then
  TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
  echo "     Total VRAM: ${TOTAL_VRAM} MiB"
fi

# Environment variables
check "ANTHROPIC_API_KEY set" "[ -n '${ANTHROPIC_API_KEY:-}' ]"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[WARN] WANDB_API_KEY not set (wandb logging will be disabled)"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[WARN] HF_TOKEN not set (may be needed for gated models)"
fi

# Disk space
AVAILABLE_GB=$(df -BG . | awk 'NR==2{print $4}' | tr -d 'G')
echo "     Disk space: ${AVAILABLE_GB} GB free"
if [ "$AVAILABLE_GB" -lt 500 ]; then
  echo "[WARN] Less than 500 GB free. Need ~2 TB for data + checkpoints."
fi

# Data directories
for dir in data/raw data/synthesized data/train checkpoints; do
  mkdir -p "$dir"
done
echo "[OK] Data directories ready"

echo "==================================="
if [ "$ERRORS" -eq 0 ]; then
  echo "All checks passed."
else
  echo "$ERRORS check(s) failed."
  exit 1
fi
