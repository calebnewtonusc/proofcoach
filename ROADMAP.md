# ProofCoach — Roadmap

## Current Status: v1 (In Development)

---

## v1 — TEACH (Target: Q2 2026)

Core tutoring capability: Socratic dialogue + Lean 4 verified proof steps.

### Milestones

- [x] Architecture design and data source identification
- [x] Training pipeline design (SFT → GRPO → DPO)
- [ ] AoPS wiki and forum crawler (aops_crawler.py)
- [ ] AMC/AIME/USAMO/IMO downloader (olympiad_downloader.py)
- [ ] Putnam archive downloader (putnam_downloader.py)
- [ ] Lean 4 installation and interface (lean4_interface.py)
- [ ] Teaching dialogue synthesizer (teaching_synthesizer.py)
- [ ] Misconception generator (misconception_generator.py)
- [ ] Stage 1 SFT on 200k tutoring pairs
- [ ] Stage 2 GRPO with Lean 4 reward signal
- [ ] Stage 3 DPO on teaching quality
- [ ] CoachBench evaluation harness
- [ ] Deployment API + Docker Compose

### v1 Target Metrics

| Metric | Target |
|--------|--------|
| AMC 10/12 accuracy | >85% |
| AIME accuracy | >50% |
| USAMO solution quality | >70% (rubric) |
| Proof step correctness (Lean 4) | >90% |
| Teaching effectiveness | >0.3 improvement (5-problem) |

---

## v1.5 — SEQUENCE (Target: Q3 2026)

Persistent student knowledge model + adaptive curriculum sequencing.

### Milestones

- [ ] Student knowledge model per topic node (skill_model.py)
- [ ] Problem taxonomy with prerequisite graph (problem_taxonomy.py)
- [ ] Session persistence (Redis or Postgres backend)
- [ ] Practice sequencer agent trained on skill gap targeting
- [ ] Multi-session evaluation: improvement across 10 sessions
- [ ] Student dashboard (session history, mastery map)

### v1.5 Target Metrics

| Metric | Target |
|--------|--------|
| Practice sequence effectiveness | >0.5 (CoachBench-Sequence) |
| Correct skill gap identification | >80% |
| Problem difficulty calibration | ±0.5 difficulty level |

---

## v2 — ASSESS (Target: Q4 2026)

Formal mastery gating: students cannot advance without demonstrating mastery.

### Milestones

- [ ] Formal student model with mastery thresholds per node
- [ ] Novel problem generation at calibrated difficulty
- [ ] Held-out mastery assessment (3 problems per node)
- [ ] Prerequisite enforcement: cannot unlock generating functions without counting mastery
- [ ] Mastery certificates (cryptographically signed assertions)
- [ ] Teacher dashboard: class-level mastery heatmaps

### v2 Target Metrics

| Metric | Target |
|--------|--------|
| Mastery assessment precision | >85% |
| False positive mastery gates | <5% |
| Novel problem quality (rubric) | >80% |

---

## v3 — COMPETE (Target: Q1 2027)

AMC/AIME competition coach with timed practice and strategy.

### Milestones

- [ ] Timed practice mode (AMC: 75 min, AIME: 3h)
- [ ] Triage strategy: skip/return/solve decision model
- [ ] Mental math shortcut curriculum
- [ ] Post-contest review with counterfactual analysis
- [ ] Competition calendar and prep schedule planner
- [ ] Score prediction model (predicts AMC score from practice performance)
- [ ] Integration with AoPS contest room archives

### v3 Target Metrics

| Metric | Target |
|--------|--------|
| AMC score improvement (10 sessions) | +5 points avg |
| AIME qualification rate | +15% over baseline |
| Score prediction accuracy | ±2 points |

---

## Long-Term Vision

- **ProofCoach API**: hosted endpoint for any math education platform
- **Lean 4 proof search**: given a problem, generate a fully formal Lean 4 proof, not just tutoring steps
- **IMO Gold Medal target**: solve all 6 IMO problems with verified proofs — the formal AI math benchmark
- **Multilingual**: Chinese, Spanish, French olympiad content
- **K-12 mode**: scale down from olympiad to AMC 8 / precalculus difficulty
