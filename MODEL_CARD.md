# ProofCoach — Model Card

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | ProofCoach v1 |
| **Base model** | Qwen/Qwen2.5-7B-Instruct |
| **Fine-tuning method** | LoRA (SFT) → GRPO RL → DPO |
| **Parameters** | 7.6B (base) + LoRA adapters |
| **Context window** | 8,192 tokens |
| **Languages** | English (math notation universal) |
| **License** | Apache 2.0 |
| **Developer** | Caleb Newton (calebnewtonusc) |
| **Version** | 1.0.0 |
| **Release date** | Q2 2026 (projected) |

---

## Training Details

### Stage 1: Supervised Fine-Tuning

- **Algorithm**: SFT via TRL SFTTrainer
- **Data**: 180,000 tutoring interaction pairs (4 streams)
- **LoRA rank**: 64, alpha: 128, dropout: 0.05
- **Target modules**: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
- **Learning rate**: 2e-4 with WarmupDecayLR
- **Epochs**: 3
- **Batch size**: 2 per GPU × 4 gradient accumulation = 8 effective
- **Hardware**: 18× A6000 48GB, DeepSpeed ZeRO-3

### Stage 2: RL with Lean 4 Verification Reward

- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Reward signal**: Lean 4 type-checker — +1.0 if proof step verified, -1.0 if rejected
- **Partial reward**: +0.3 for correct numerical answer without formal proof
- **KL penalty**: 0.02 vs. SFT checkpoint
- **Group size N**: 4 completions per prompt
- **Learning rate**: 1e-6

### Stage 3: DPO on Teaching Quality

- **Algorithm**: DPO (Direct Preference Optimization), beta=0.1
- **Chosen**: Socratic dialogue that leads student to discover insight
- **Rejected**: Direct answer / mechanical explanation
- **Preference signal sources**: AoPS upvotes, student self-report, problem success rate
- **Pairs**: 20,000 preference pairs

---

## Intended Use

### Primary Use Cases

- **Olympiad math tutoring**: AMC 8/10/12, AIME, USAMO, IMO problem tutoring
- **Proof verification assistance**: Checking correctness of mathematical arguments
- **Misconception diagnosis**: Identifying where a student's reasoning went wrong
- **Practice sequencing**: Suggesting the next practice problem based on skill gaps
- **Self-study**: Students working through olympiad problems with AI guidance

### Out-of-Scope Uses

- **Not a calculator**: Not designed for numerical computation (use Wolfram Alpha)
- **Not a theorem prover**: Cannot generate arbitrary Lean 4 proofs (it tutors with verifiable steps, but is not a full proof search system like AlphaProof)
- **Not for cheating**: Designed for learning, not for submitting others' work as one's own
- **K-12 standard curriculum**: Optimized for competition math, not aligned to any curriculum standard

---

## Evaluation

### CoachBench Results (v1 targets)

| Benchmark | Metric | Target | Status |
|-----------|--------|--------|--------|
| CoachBench-AMC | AMC 10/12 accuracy | >85% | In training |
| CoachBench-AIME | AIME accuracy | >50% | In training |
| CoachBench-USAMO | Solution quality (rubric) | >70% | In training |
| CoachBench-Lean | Proof step verification rate | >90% | In training |
| CoachBench-Teaching | Teaching improvement (5 problems) | >0.3 | In training |

### Comparison vs. Baselines

| Model | AMC 12 | AIME | Lean-verified | Socratic |
|-------|--------|------|---------------|---------|
| GPT-4o | ~72% | ~23% | No | No |
| Claude 3.5 Sonnet | ~75% | ~28% | No | No |
| Qwen2.5-7B (base) | ~45% | ~12% | No | No |
| **ProofCoach v1** | **>85%** | **>50%** | **>90%** | **Yes** |

*GPT-4o and Claude figures from published benchmarks as of early 2026.*

---

## Limitations

1. **Lean 4 scope**: Not all mathematical claims can be easily expressed as Lean 4 theorems. The interface handles common proof patterns; novel or complex proofs may fail verification incorrectly.

2. **Socratic patience**: The model is trained to ask questions rather than give answers. For students who need direct explanations before Socratic questioning, this may feel slow.

3. **English only**: Training data is primarily English. Non-English mathematical notation is understood (LaTeX is universal), but prose tutoring is English-only.

4. **Knowledge cutoff**: Competition problems through 2025 are included. 2026+ competition problems are not in the training set.

5. **Higher mathematics**: Optimized for olympiad difficulty (high school competition level). Graduate-level mathematics (topology, abstract algebra beyond competition level) is not the target domain.

6. **Hallucination**: Despite Lean 4 reward signal, the model can still generate incorrectly reasoned steps that happen to correspond to correct final answers. The Lean 4 verification catches most errors in formally expressible claims.

---

## Ethical Considerations

### Academic Integrity
ProofCoach is designed as a learning tool. It defaults to Socratic questioning rather than answer provision. However, we acknowledge that it can be used to obtain solutions — this is an inherent tension in any math AI system.

### Access and Equity
Competition mathematics preparation is historically expensive (tutors, AoPS courses). ProofCoach aims to democratize access to olympiad-level instruction.

### Bias
The training corpus (AoPS) skews toward a particular mathematical culture and problem-solving style. Problems from Asian olympiads (CMO, APMO) and other traditions are underrepresented.

### Data Attribution
AoPS community members contributed the solutions in the training corpus. We credit the community collectively.

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
