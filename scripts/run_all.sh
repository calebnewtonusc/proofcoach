#!/usr/bin/env bash
# ProofCoach — Full Pipeline Runner
# Runs the complete pipeline: collect → synthesize → verify → train → evaluate
set -euo pipefail

echo "======================================================="
echo "ProofCoach Full Pipeline"
echo "======================================================="

# 1. Check environment
echo "[1/12] Checking environment..."
bash scripts/check_env.sh

# 2. Download olympiad archives
echo "[2/12] Downloading AMC/AIME/USAMO/IMO archives..."
python discovery/olympiad_downloader.py

# 3. Crawl AoPS solutions
echo "[3/12] Crawling AoPS wiki and forum solutions..."
python discovery/aops_crawler.py

# 4. Download Putnam archives
echo "[4/12] Downloading Putnam archives..."
python discovery/olympiad_downloader.py --putnam

# 5. Start vLLM synthesis servers
echo "[5/12] Starting vLLM synthesis servers..."
bash scripts/start_vllm.sh
sleep 30  # Wait for vLLM to be ready

# 6. Synthesize teaching dialogues
echo "[6/12] Synthesizing Socratic tutoring dialogues..."
python synthesis/teaching_synthesizer.py \
  --backend vllm \
  --vllm-urls http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 \
  --workers 40

# 7. Generate misconception pairs
echo "[7/12] Generating misconception diagnosis pairs..."
python synthesis/misconception_generator.py \
  --backend vllm \
  --vllm-urls http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004

# 8. Generate DPO pairs
echo "[8/12] Generating DPO preference pairs..."
python synthesis/generate_dpo_pairs.py \
  --backend vllm \
  --vllm-urls http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004

# 9. Kill vLLM servers before training
echo "[9/12] Stopping vLLM servers..."
pkill -f "vllm serve" || true
sleep 5

# 10. Stage 1: SFT
echo "[10/12] Stage 1: SFT on 180k tutoring pairs..."
deepspeed --num_gpus=18 training/train.py \
  --deepspeed training/configs/ds_config.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-dir data/train \
  --output-dir checkpoints/proofcoach-sft \
  --epochs 3

# 11. Stage 2: GRPO RL
echo "[11/12] Stage 2: GRPO RL with Lean 4 reward..."
deepspeed --num_gpus=18 training/train_rl.py \
  --base-model checkpoints/proofcoach-sft/final \
  --output-dir checkpoints/proofcoach-rl \
  --steps 2000

# Stage 3: DPO
echo "[12/12] Stage 3: DPO on teaching quality..."
deepspeed --num_gpus=18 training/train_dpo.py \
  --base-model checkpoints/proofcoach-rl/final \
  --output-dir checkpoints/proofcoach-final \
  --epochs 1

echo "======================================================="
echo "Training complete! Final model: checkpoints/proofcoach-final"
echo "Running CoachBench evaluation..."
python evaluation/coachbench.py --model checkpoints/proofcoach-final --all
echo "Pipeline complete."
