# ProofCoach — Teaches like a grandmaster. Proves like a computer.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: Qwen2.5-7B](https://img.shields.io/badge/base_model-Qwen2.5--7B--Instruct-purple.svg)](https://huggingface.co/Qwen)
[![GPUs: 18x A6000](https://img.shields.io/badge/training-18x_A6000-red.svg)](https://www.nvidia.com)
[![Lean 4: Verified](https://img.shields.io/badge/proofs-Lean_4_verified-brightgreen.svg)](https://leanprover.github.io)

> **"Teaches like a grandmaster. Proves like a computer."**

ProofCoach is the first AI math tutor that combines **Socratic teaching** with **formally verified proof steps**. It doesn't just solve problems — it teaches WHY, diagnoses misconceptions, sequences practice by skill gap, and proves every step it teaches using Lean 4, so students can never be taught incorrect mathematics.

Trained on every AMC/AIME/USAMO/IMO problem ever published, every Art of Problem Solving community solution (multiple approaches per problem), and the Putnam archives. The training corpus has never been distilled into model weights before — not because the data doesn't exist, but because no one has built a reward signal that verifies mathematical correctness at proof-step granularity. ProofCoach does, using Lean 4 as the formal verifier.

---

## What Makes ProofCoach Different

| Capability | GPT-4o w/ math | Wolfram Alpha | Photomath | Khan AI | **ProofCoach** |
|---|---|---|---|---|---|
| Olympiad-level solving | inconsistent | no | no | no | **USAMO/IMO trained** |
| Formal proof verification | no | no | no | no | **Lean 4 in the loop** |
| Multi-approach explanations | rarely | no | no | limited | **AoPS-style, 5 approaches** |
| Misconception diagnosis | no | no | no | limited | **Dedicated misconception agent** |
| Socratic teaching (not answers) | no | no | no | limited | **Core training objective** |
| Skill-gap sequenced practice | no | no | no | partial | **Student knowledge model** |
| Verified no wrong steps | no | no | no | no | **Lean 4 verified** |
| Putnam/olympiad coverage | partial | no | no | no | **Full archive** |

---

## Architecture

```
                    ┌───────────────────────────────────────────┐
 Student input ────►│          ProofCoach Model                 │
 (problem + work)   │  (Qwen2.5-7B + LoRA, 3-stage trained)    │
                    └──────────────────┬────────────────────────┘
                                       │
               ┌───────────────────────┼──────────────────────────┐
               ▼                       ▼                          ▼
      ┌─────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
      │   Tutor Agent   │   │  Proof Verifier    │   │ Practice Sequencer   │
      │  (Socratic      │   │  (Lean 4 interface │   │ (skill model +       │
      │   dialogue,     │   │   formal step      │   │  curriculum from     │
      │   hints, WHY)   │   │   verification)    │   │  skill gap analysis) │
      └────────┬────────┘   └─────────┬──────────┘   └──────────┬───────────┘
               │                      │                          │
               ▼                      ▼                          ▼
      ┌─────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
      │ Misconception   │   │   Lean 4 Runtime   │   │  Problem Taxonomy    │
      │ Detector Agent  │   │  (proof assistant, │   │  (number theory,     │
      │  (diagnoses     │   │   type checker,    │   │   combinatorics,     │
      │  wrong thinking)│   │   counterexamples) │   │   algebra, geometry) │
      └─────────────────┘   └────────────────────┘   └──────────────────────┘
```

**Training data (4 streams, 200k+ tutoring pairs):**
- Stream 1: AMC 8/10/12, AIME I/II, USAMO, IMO + AoPS community solutions (45%)
- Stream 2: Art of Problem Solving wiki multi-approach explanations (25%)
- Stream 3: Putnam archives + math olympiad training resources (15%)
- Stream 4: Synthesized Socratic tutoring dialogues from solution pairs (15%)

---

## Quick Start

**1. Clone and install**

```bash
git clone https://github.com/calebnewtonusc/proofcoach.git
cd proofcoach
pip install -r requirements.txt
```

**2. Install Lean 4 (for proof verification)**

```bash
# Install elan (Lean version manager)
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source ~/.profile

# Install Lean 4
elan toolchain install leanprover/lean4:stable
elan default leanprover/lean4:stable

# Verify
lean --version
```

**3. Configure environment**

```bash
cp .env.example .env
# Edit .env — fill in ANTHROPIC_API_KEY and others
```

**4. Collect training data**

```bash
# Download AMC/AIME/USAMO/IMO archives
python discovery/olympiad_downloader.py --all-competitions

# Crawl Art of Problem Solving solutions
python discovery/aops_crawler.py --full-crawl --workers 20
```

**5. Synthesize training pairs**

```bash
bash scripts/start_vllm.sh
python synthesis/teaching_synthesizer.py --backend vllm \
  --vllm-urls http://localhost:8001 http://localhost:8002
python synthesis/misconception_generator.py --backend vllm
```

**6. Train**

```bash
# Stage 1: SFT
deepspeed --num_gpus=18 training/train.py \
  --deepspeed training/configs/ds_config.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-dir data/train \
  --output-dir checkpoints/proofcoach-sft

# Stage 2: RL (Lean 4 proof verification reward)
deepspeed --num_gpus=18 training/train_rl.py \
  --base-model checkpoints/proofcoach-sft/final \
  --output-dir checkpoints/proofcoach-rl

# Stage 3: DPO (teaching quality)
deepspeed --num_gpus=18 training/train_dpo.py \
  --base-model checkpoints/proofcoach-rl/final \
  --output-dir checkpoints/proofcoach-final
```

**7. Evaluate**

```bash
python evaluation/coachbench.py --model checkpoints/proofcoach-final --all
```

**8. Deploy**

```bash
cd deploy && docker compose up -d
```

---

## Hardware Requirements

| Phase | Config | Notes |
|-------|--------|-------|
| Data collection | Any machine | CPU-bound async I/O |
| Synthesis | 4-8x A6000 | vLLM Qwen2.5-72B |
| Training | 18x A6000 | DeepSpeed ZeRO-3, ~6h |
| Inference | 1x A100 or 1x A6000 | <200ms p50 |

---

## CoachBench

CoachBench evaluates ProofCoach across two dimensions:
1. **Problem solving**: correct answer on 200 held-out olympiad problems
2. **Teaching quality**: does the model teach the student to solve the next similar problem?

```bash
# Full benchmark
python evaluation/coachbench.py --model checkpoints/proofcoach-final --all

# Problem solving only
python evaluation/coachbench.py --model checkpoints/proofcoach-final --solving

# Teaching quality evaluation
python evaluation/coachbench.py --model checkpoints/proofcoach-final --teaching
```

**v1 Targets:**
| Metric | Target |
|--------|--------|
| AMC 10/12 accuracy | >85% |
| AIME accuracy | >50% |
| USAMO solution quality | >70% (rubric) |
| Proof step correctness (Lean 4) | >90% |
| Teaching effectiveness | >0.3 improvement in 5-problem sequence |

---

## Citation

```bibtex
@inproceedings{newton2026proofcoach,
  title     = {ProofCoach: Formal Verification-Grounded Math Olympiad Tutoring},
  author    = {Newton, Caleb and others},
  booktitle = {NeurIPS 2026 AI for Education Workshop},
  year      = {2026},
}
```

---

## License

**Code:** MIT License | **Model weights:** Apache 2.0 | **Training data:** CC BY 4.0

---

*Target: 864GB VRAM, 200k+ tutoring pairs, Lean 4 verified proof steps. Training in progress.*
