# ProofCoach — GPU Setup Guide

## Hardware Target

| Phase | Config | VRAM | Time |
|-------|--------|------|------|
| Data collection | CPU only | — | ~2h |
| Lean 4 verification (batch) | CPU or 1× GPU | — | ~4h |
| Synthesis (vLLM) | 8× A6000 48GB | 384GB | ~10h |
| Stage 1 SFT | 18× A6000 48GB | 864GB | ~6h |
| Stage 2 GRPO RL | 18× A6000 48GB | 864GB | ~12h |
| Stage 3 DPO | 18× A6000 48GB | 864GB | ~4h |
| Inference | 1× A100 80GB | 80GB | <200ms p50 |

---

## Driver and CUDA Setup

```bash
# Verify NVIDIA drivers
nvidia-smi

# Required: CUDA 12.1+
nvcc --version

# Install CUDA 12.1 if needed (Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# Install PyTorch with CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

---

## GPU Allocation Strategy

### Synthesis Phase (8 GPUs, ports 8001-8004)

Run Qwen2.5-72B across 4 vLLM instances, 2 GPUs each:

```bash
bash scripts/start_vllm.sh
# Starts 4 instances:
# Instance 1: GPUs 0-1, port 8001
# Instance 2: GPUs 2-3, port 8002
# Instance 3: GPUs 4-5, port 8003
# Instance 4: GPUs 6-7, port 8004
```

### Training Phase (18 GPUs)

```bash
# Stage 1 SFT
deepspeed --num_gpus=18 training/train.py \
  --deepspeed training/configs/ds_config.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-dir data/train \
  --output-dir checkpoints/proofcoach-sft

# Stage 2 GRPO RL
deepspeed --num_gpus=18 training/train_rl.py \
  --base-model checkpoints/proofcoach-sft/final \
  --output-dir checkpoints/proofcoach-rl

# Stage 3 DPO
deepspeed --num_gpus=18 training/train_dpo.py \
  --base-model checkpoints/proofcoach-rl/final \
  --output-dir checkpoints/proofcoach-final
```

---

## Lean 4 Setup

Lean 4 runs on CPU for proof verification. Install separately from PyTorch:

```bash
# Install elan (Lean version manager)
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source ~/.profile

# Install Lean 4
elan toolchain install leanprover/lean4:stable
elan default leanprover/lean4:stable

# Verify installation
lean --version
# Expected: Lean (version 4.x.x, ...)

# Install Mathlib (for competition math theorems)
# In your Lean project directory:
lake init proofcoach_lean
cd proofcoach_lean
# Add to lakefile.lean:
# require mathlib from git "https://github.com/leanprover-community/mathlib4"
lake update
lake build

# Test Lean 4 interface
python -c "from core.lean4_interface import Lean4Interface; l = Lean4Interface(); print(l.verify('theorem test (n : ℤ) : n^2 - 1 = (n-1)*(n+1) := by ring'))"
```

### Lean 4 Performance

Lean 4 type-checking is CPU-bound. For batch verification during RL training:

```bash
# Scale Lean 4 workers
export LEAN4_WORKERS=8  # one per CPU core, up to 32
```

On a 32-core CPU server: ~200 proof checks/second, sufficient for RL training at batch_size=4 × N=4 completions = 16 checks per step.

---

## Memory Requirements

### Synthesis (Qwen2.5-72B, bfloat16)
- Total: 72B × 2 bytes = 144GB → 2 GPUs per instance minimum
- 4 instances × 2 GPUs × 48GB = 384GB total

### Training (Qwen2.5-7B, DeepSpeed ZeRO-3 + CPU offload)
- Model parameters: 7.6B × 2 bytes = 15.2GB
- With ZeRO-3: params sharded across 18 GPUs → ~0.85GB per GPU for params
- Gradients + optimizer states: offloaded to CPU (192GB CPU RAM required)
- Activations: ~8GB per GPU at sequence length 2048

### Inference (1× A6000 or A100)
- Qwen2.5-7B bfloat16: ~15GB VRAM
- With KV cache at 4096 tokens: ~20GB
- Fits on a single 48GB A6000

---

## Environment Verification

```bash
bash scripts/check_env.sh
```

Expected output:
```
[OK] Python 3.11.x
[OK] torch 2.2.0+cu121
[OK] transformers 4.44.x
[OK] peft 0.12.x
[OK] trl 0.9.x
[OK] deepspeed 0.14.x
[OK] vllm 0.5.x
[OK] lean 4.x.x
[OK] GPUs: 18 × NVIDIA A6000 (48564 MiB each)
[OK] Total VRAM: 873,152 MiB
[OK] ANTHROPIC_API_KEY set
[OK] Disk space: 8.2 TB free (need ~2 TB for data + checkpoints)
```

---

## Common Issues

### "CUDA out of memory" during training

```bash
# Reduce per-GPU batch size
export MICRO_BATCH_SIZE=1  # default is 2

# Increase gradient accumulation
export GRAD_ACCUM_STEPS=8  # default is 4

# Enable gradient checkpointing
export USE_GRADIENT_CHECKPOINTING=1
```

### Lean 4 timeout during RL training

```bash
# Increase Lean 4 timeout
export LEAN4_TIMEOUT=30  # seconds (default 10)

# Use simulated rewards for debugging (skips Lean 4)
export LEAN4_SIMULATED=1
```

### DeepSpeed CPU offload OOM

```bash
# Verify CPU RAM (need 192GB+)
free -h

# Enable NUMA-aware memory allocation
export DEEPSPEED_NUMA_AFFINITY=1
```

### Lean 4 "lake: command not found"

```bash
source ~/.profile
# or
export PATH="$HOME/.elan/bin:$PATH"
```
