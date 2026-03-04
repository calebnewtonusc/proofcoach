# ProofCoach — Data Sources

## Overview

ProofCoach is trained on 4 data streams totaling 200k+ tutoring interaction pairs. The key insight: AoPS has 5-20 community-submitted solutions per problem, each explaining the same insight differently. This multi-perspective corpus has never been systematically distilled into model weights.

---

## Stream 1: AMC/AIME/USAMO/IMO + AoPS Solutions (45% of training data)

### Source
- **Art of Problem Solving wiki**: aops.com/wiki — official problem statements + community solution pages
- **AoPS forum**: aops.com/Forum — for each problem, 5-20 individual user solution posts
- **AMC archives**: AMC 8 (1999–2025), AMC 10 (2000–2025), AMC 12 (2000–2025), AIME I/II (1983–2025)
- **USAMO archives**: 1972–2025
- **IMO archives**: 1959–2025, including shortlist problems
- **HMMT, PUMAC, ARML**: Regional and invitational competitions

### Format
Each problem becomes multiple training examples — one per community solution approach:

```json
{
  "conversations": [
    {"role": "system", "content": "You are ProofCoach..."},
    {"role": "user", "content": "Problem: [problem_statement]\nStudent work: [student_attempt]"},
    {"role": "assistant", "content": "[socratic_questions + proof_steps + next_hint]"}
  ],
  "metadata": {
    "problem_id": "AMC12-2023-A15",
    "approach": "casework_by_parity",
    "difficulty": 7,
    "topics": ["number_theory", "modular_arithmetic"],
    "source_user": "mathgeek2023",
    "community_votes": 47
  }
}
```

### Quality Filters
- Minimum solution length: 100 words (filters one-liners)
- Minimum community votes: 3 upvotes (filters low-quality posts)
- LaTeX rendering validation: all math expressions parse correctly
- Manual review sample: 500 random examples checked for mathematical correctness

### Scale
- ~8,000 unique problems across all competitions
- ~40,000 community solutions (5 per problem average)
- ~90k synthesized tutoring dialogues from these solutions

---

## Stream 2: AoPS Wiki Multi-Approach Explanations (25%)

### Source
AoPS wiki articles contain structured multi-approach explanations for each topic — not just problems, but the underlying mathematical tools.

- **Topic articles**: pigeonhole principle, AM-GM inequality, Cauchy-Schwarz, Vieta's formulas (500+ articles)
- **Solution articles**: for famous olympiad problems, the wiki has extended solution writeups with multiple methods
- **Proof articles**: formal and informal proofs of key theorems used in competitions

### Format
Converted to Socratic tutoring dialogues:

```json
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "I don't understand why AM-GM works."},
    {"role": "assistant", "content": "Let's discover it together..."}
  ],
  "metadata": {
    "topic": "AM-GM_inequality",
    "approach": "visual_geometric",
    "source": "aops_wiki"
  }
}
```

### Scale
- ~500 wiki articles
- ~50k synthesized explanation dialogues

---

## Stream 3: Putnam + Olympiad Training Resources (15%)

### Source
- **Putnam archives**: 1938–2024, all problems + official solutions
- **Mathematical Olympiad in China (CMO)**: 1985–2024
- **Balkan MO, Nordic MO, Asian Pacific MO**: regional olympiad archives
- **Art of Problem Solving books**: Problem Solving Strategies (Engel), The Art of Problem Solving (Rusczyk), Introduction to Counting and Probability — problem sets and solutions from the accompanying courses
- **Olympiad training handouts**: USAMO training materials from coaches (publicly shared)
- **Evan Chen's olympiad handouts**: evan.eyvanshu.com — freely available problem sets with solutions

### Scale
- ~3,000 Putnam problems with solutions
- ~5,000 international olympiad problems
- ~30k synthesized tutoring pairs

---

## Stream 4: Synthesized Socratic Tutoring Dialogues (15%)

### Generation Pipeline
For each (problem, solution_approach) pair:
1. Extract the key insight / "aha moment" of the solution
2. Prompt Qwen2.5-72B to generate a Socratic tutoring dialogue where:
   - The student starts with a partial attempt or misconception
   - The tutor asks targeted questions to guide discovery
   - The dialogue ends with the student articulating the insight themselves
3. Lean 4 verify all mathematical claims in the dialogue
4. Filter to keep only dialogues where >90% of claims verify

### Misconception Pairs
Separately, generate misconception correction examples:

```json
{
  "conversations": [
    {"role": "user", "content": "[problem] + [student wrong approach]"},
    {"role": "assistant", "content": "[diagnosis of error] + [corrective question]"}
  ],
  "metadata": {
    "misconception_type": "assumed_continuity",
    "correct_diagnosis": true
  }
}
```

### Scale
- ~30k Socratic dialogue pairs
- ~10k misconception diagnosis pairs

---

## Data Pipeline Summary

```
Raw Sources → discovery/ crawlers → data/raw/
Raw data    → synthesis/ pipeline → data/synthesized/
Both        → deduplication       → data/train/
            → Lean 4 verification → data/verified/
            → quality filter      → data/final/
```

### Deduplication
- MinHash LSH for near-duplicate problem detection (threshold: 0.85 Jaccard)
- Exact match removal on problem statement after LaTeX normalization
- Cross-stream deduplication: same problem from different sources → keep highest-quality version

### LaTeX Processing
- All problems stored with raw LaTeX
- Rendered to ASCII for model input (using sympy or custom renderer)
- Lean 4 propositions generated from LaTeX proof steps

### Final Dataset Stats

| Split | Problems | Tutoring Pairs | Verified |
|-------|----------|----------------|---------|
| Train | 14,000 | 180,000 | 85% |
| Val | 500 | 10,000 | 100% |
| Test (CoachBench) | 200 | held-out | 100% |
| DPO pairs | — | 20,000 | 100% |

---

## Legal and Ethical Notes

- AoPS data: community-contributed, used under AoPS terms for research
- AMC/AIME problems: published by MAA, used for research under fair use
- IMO problems: published by IMO, used for research
- Putnam problems: published by MAA, used for research
- No personal data: all AoPS usernames anonymized in training data
- No paywall bypass: only publicly accessible pages crawled
