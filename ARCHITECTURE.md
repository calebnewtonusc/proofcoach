# ProofCoach — Full System Architecture
## "Teaches like a grandmaster. Proves like a computer."

---

## THE VISION

Student asks: "I don't understand why the sum of the first n odd numbers equals n²."
ProofCoach: understands the student's knowledge state → sequences the right questions → walks through the visual proof (pairing), the algebraic proof, and the inductive proof → verifies each step in Lean 4 → offers practice problems at the right difficulty.

Not "here's the formula." Not "here's the proof." A conversation that makes the student understand WHY — and verifiably teaches correct mathematics at every step.

---

## 4-PHASE PRODUCT VISION

```
Phase 1 (v1):   TEACH           Olympiad-level Socratic tutoring + Lean 4 verification
Phase 2 (v1.5): SEQUENCE        Personalized curriculum from skill gap model
Phase 3 (v2):   ASSESS          Formal student model, mastery gating, adaptive difficulty
Phase 4 (v3):   COMPETE         AMC/AIME prep coach, competition strategy, timed practice
```

### Phase 1 — v1: TEACH (Current)

Given any olympiad problem or student question, ProofCoach teaches through Socratic dialogue — not by providing answers, but by asking targeted questions that lead the student to discover the insight themselves. Every mathematical claim is verifiable via Lean 4. The model knows 5+ solution approaches per problem and selects the one best matched to the student's current knowledge.

### Phase 2 — v1.5: SEQUENCE

ProofCoach builds and maintains a student knowledge model across sessions. After each interaction, it updates beliefs about student mastery in number theory, combinatorics, algebra, and geometry subtopics. The next problem offered is chosen to target the specific skill gap — not the next problem in a textbook sequence.

### Phase 3 — v2: ASSESS

Full formal student model with mastery gating. Before moving from counting techniques to generating functions, ProofCoach verifies (via held-out problems) that the student has genuinely mastered the prerequisite. No advancing until mastery is demonstrated. The model generates novel problems at exactly the right difficulty.

### Phase 4 — v3: COMPETE

AMC/AIME competition coach: timed practice, strategy under time pressure, triage (which problems to skip), mental math shortcuts, and post-competition review. Learns the student's specific weakness categories and prioritizes them.

---

## TARGET METRICS

| Version | Task | Target | Benchmark |
|---------|------|--------|-----------|
| v1 | AMC 10/12 accuracy | >85% | CoachBench-AMC |
| v1 | AIME accuracy | >50% | CoachBench-AIME |
| v1 | Proof step Lean 4 check rate | >90% | CoachBench-Lean |
| v1 | Teaching improvement (5-problem) | >0.3 | CoachBench-Teaching |
| v1.5 | Practice sequence effectiveness | >0.5 | CoachBench-Sequence |
| v2 | Mastery assessment precision | >85% | CoachBench-Mastery |
| v3 | AMC score improvement (10 sessions) | +5 points avg | CompBench |

---

## 5 TECHNICAL DIFFERENTIATORS

### 1. AoPS Multi-Perspective Corpus (Never Distilled Before)

Art of Problem Solving (AoPS) is the premier online math competition community. For every olympiad problem, AoPS has 5-20 different solution approaches submitted by different users — each explaining their reasoning differently, targeting different mathematical tools. This multi-perspective corpus has never been systematically distilled into model weights. ProofCoach extracts all solutions for each problem and trains on the diversity of approaches, not just one "correct" answer.

### 2. Lean 4 as the Reward Signal (Formally Verified Tutoring)

The key innovation: ProofCoach's RL reward comes from Lean 4 proof verification. When the model generates a tutoring step ("The next step is to show that n²-1 = (n-1)(n+1)"), the claim is automatically expressed in Lean 4 and type-checked. If Lean accepts it, reward = +1. If Lean rejects it, reward = -1. This means the model is literally trained to never teach incorrect mathematics — not as a soft preference, but as the hard reward signal.

### 3. Socratic Process Training

Existing math AI is trained to produce correct answers. ProofCoach is trained on the TEACHING PROCESS — the art of asking the right question at the right moment that makes a student discover the insight themselves. The AoPS forum contains thousands of examples of expert tutors guiding students through problems via targeted questions. That Socratic process corpus has been extracted and distilled.

### 4. Misconception Detection as a First-Class Task

ProofCoach diagnoses WHERE a student's reasoning went wrong — not just that it's wrong. The model is trained on misconception pair data: (student's wrong approach, correct diagnosis of the error, corrective question). This turns error correction from "that's incorrect" into "I see you're assuming the function is continuous — can you think of a case where that fails?"

### 5. Problem Taxonomy with Prereq Graph

Problems are tagged to a topic taxonomy (number theory subtopics, combinatorics techniques, algebra tools, geometry theorems) with a prerequisite graph. The student knowledge model tracks mastery per node. This enables genuinely adaptive sequencing: the model never teaches pigeonhole principle before it teaches the basics of counting.

---

## 3-STAGE TRAINING PIPELINE

### Stage 1 — SFT: train.py

```
Input:  (problem_statement, student_work_attempt, knowledge_state)
Output: (socratic_questions, explanation, proof_steps, next_problem_hint)

Data mix:
  45% Stream 1: AMC/AIME/USAMO/IMO + AoPS community solutions
  25% Stream 2: AoPS wiki multi-approach explanations
  15% Stream 3: Putnam + olympiad training resources
  15% Stream 4: Synthesized Socratic tutoring dialogues

Target: ~200k tutoring interaction pairs
Base: Qwen2.5-7B-Instruct (general reasoning, not coder variant)
Training: DeepSpeed ZeRO-3, LoRA rank 64, 3 epochs, ~6h on 18xA6000
```

### Stage 2 — Lean 4 Verification RL: train_rl.py

```
For each generated tutoring step:
  1. Extract mathematical claims as Lean 4 propositions
  2. Lean 4 type-checker verifies the claim
  3. Reward = +1.0 if Lean accepts, -1.0 if Lean rejects
  4. Partial reward for correct numerical answer without formal proof

Algorithm: GRPO with KL penalty vs. SFT checkpoint
Target: >90% of generated proof steps pass Lean 4 verification
```

### Stage 3 — DPO on Teaching Quality: train_dpo.py

```
Chosen:   Step-by-step Socratic guidance → student discovers insight
Rejected: Just give the answer / overly mechanical explanation

Preference signals:
  - AoPS community "upvotes" on tutoring posts (chosen = high-voted)
  - Student self-reported understanding improvement
  - Problem success rate after being tutored vs. not tutored

Algorithm: DPO, beta=0.1
Data: ~20k preference pairs
```

---

## FILE STRUCTURE

```
proofcoach/
│
├── DISCOVERY
│   └── discovery/
│       ├── aops_crawler.py         Art of Problem Solving wiki + forum
│       ├── olympiad_downloader.py  AMC/AIME/USAMO/IMO archives
│       ├── putnam_downloader.py    Putnam competition archives
│       └── imo_solutions.py        IMO official + shortlist solutions
│
├── SYNTHESIS
│   └── synthesis/
│       ├── prompts.py              All system prompts
│       ├── synthesize_bulk.py      Async multi-problem synthesis
│       ├── teaching_synthesizer.py Generate tutoring dialogues from solutions
│       ├── misconception_generator.py  Wrong approach → diagnosis pairs
│       └── generate_dpo_pairs.py   Teaching quality preference pairs
│
├── CORE
│   └── core/
│       ├── lean4_interface.py      Lean 4 proof assistant interface
│       ├── problem_taxonomy.py     Topic taxonomy + prereq graph
│       └── skill_model.py          Student knowledge state tracker
│
├── AGENTS
│   └── agents/
│       ├── tutor_agent.py              Socratic tutor
│       ├── proof_verifier_agent.py     Lean 4 verification
│       ├── misconception_detector_agent.py  Error diagnosis
│       └── practice_sequencer_agent.py  Adaptive practice selection
│
├── TRAINING
│   └── training/
│       ├── train_prep.py
│       ├── train.py                Stage 1: SFT
│       ├── train_rl.py             Stage 2: GRPO + Lean 4 reward
│       ├── train_dpo.py            Stage 3: DPO teaching quality
│       └── configs/
│           ├── ds_config.json
│           └── ds_config_rl.json
│
├── EVALUATION
│   └── evaluation/
│       └── coachbench.py           CoachBench: 200 problems + teaching eval
│
├── KNOWLEDGE
│   └── knowledge/
│       ├── number_theory.md        Divisibility, primes, modular arithmetic, ...
│       ├── combinatorics.md        Counting, pigeonhole, generating functions, ...
│       ├── algebra.md              Polynomials, inequalities, sequences, ...
│       └── geometry.md             Euclidean, trigonometric, projective, ...
│
├── DEPLOYMENT
│   └── deploy/
│       ├── docker-compose.yml
│       ├── api_server.py
│       └── lean_server.py          Lean 4 verification microservice
│
└── SCRIPTS
    └── scripts/
        ├── run_all.sh
        ├── start_vllm.sh
        ├── check_env.sh
        └── health_check.py
```

---

## LEAN 4 INTEGRATION

The core technical innovation. ProofCoach generates a proof step; the Lean 4 interface translates it to a Lean 4 proposition; Lean type-checks it.

### Example flow:

Model output:
```
"The key insight is that n² - 1 = (n-1)(n+1) by the difference of squares identity."
```

Lean 4 proposition generated:
```lean
theorem diff_of_squares (n : ℤ) : n^2 - 1 = (n - 1) * (n + 1) := by ring
```

Lean 4 response:
```
Goals accomplished: 'diff_of_squares' is correct.
```

Reward: +1.0

If the model had generated:
```
"n² - 1 = (n-1)(n-1) by the difference of squares identity."
```

The Lean proposition would fail to typecheck (counterexample: n=3, n²-1=8, (n-1)(n-1)=4), and reward = -1.0.

This is why ProofCoach cannot teach wrong mathematics. The punishment is built into the training loop.

---

## DEPLOYMENT API

```
POST /v1/tutor
  { "problem": "...", "student_work": "...", "session_id": "..." }
  → { "question": "What do you notice about the parity of n²?",
      "hint_level": 2, "verified_steps": [...] }

POST /v1/verify
  { "claim": "For all n, n² ≥ n for n ≥ 1", "proof_attempt": "..." }
  → { "verified": true, "lean4_proof": "...", "explanation": "..." }

POST /v1/diagnose
  { "problem": "...", "student_work": "...", "student_answer": "..." }
  → { "correct": false, "misconception": "Assumed convergence without checking",
      "corrective_question": "..." }

POST /v1/sequence
  { "student_id": "...", "session_history": [...] }
  → { "next_problem": {...}, "reason": "targeting weak area: AM-GM inequality" }

GET  /v1/problems/{competition}/{year}
  → List of problems from that competition year
```
