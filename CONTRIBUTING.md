# Contributing to ProofCoach

## Welcome

ProofCoach is an open research project. Contributions are welcome in all areas: data collection, model training, evaluation, and deployment.

---

## Ways to Contribute

### 1. Data Quality
- Add problems from competitions not yet in our corpus (regional olympiads, university competitions)
- Improve LaTeX parsing and normalization
- Write additional Socratic tutoring dialogues for underrepresented topics
- Fix incorrect solutions in the training data (submit PRs with corrections)

### 2. Lean 4 Interface
- Expand the set of proof patterns the interface can handle
- Improve the natural language → Lean 4 translation
- Add counterexample generation for failed proofs
- Fix Lean 4 version compatibility issues

### 3. Evaluation
- Add problems to CoachBench
- Implement additional teaching quality metrics
- Human evaluation studies (tutoring quality rubric)
- Comparison benchmarks against other math AI systems

### 4. Agents
- Improve Socratic question quality in the tutor agent
- Expand misconception taxonomy
- Improve practice sequencer's difficulty calibration
- Add new specialized agents (e.g., geometry visualization)

### 5. Infrastructure
- Docker improvements for deployment
- API optimizations
- Documentation improvements

---

## Development Setup

```bash
git clone https://github.com/calebnewtonusc/proofcoach.git
cd proofcoach
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"

# Install Lean 4
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source ~/.profile
elan toolchain install leanprover/lean4:stable

# Copy environment config
cp .env.example .env
# Fill in your API keys
```

---

## Code Standards

### Python Style
- Python 3.11+
- Type hints on all function signatures
- `loguru` for all logging (no `print()`)
- `dataclasses` for structured data
- Docstrings on all public functions and classes

```python
from loguru import logger
from dataclasses import dataclass
from typing import Optional

@dataclass
class TutoringStep:
    """A single step in a Socratic tutoring dialogue."""
    question: str
    hint_level: int
    lean4_claim: Optional[str] = None
    verified: bool = False

def format_step(step: TutoringStep) -> str:
    """Format a tutoring step for display to the student."""
    logger.debug(f"Formatting step at hint_level={step.hint_level}")
    ...
```

### Commit Messages
- `feat: add putnam_downloader for Putnam archive`
- `fix: handle LaTeX parsing edge case in aops_crawler`
- `refactor: extract lean4 claim parser to separate module`
- `data: add 500 additional IMO problems from 1990-2000`
- `eval: add USAMO rubric scoring to coachbench`

### Testing
Run before submitting a PR:

```bash
python -m pytest tests/ -v
python scripts/check_env.sh
python scripts/health_check.py  # requires running server
```

---

## Submitting a PR

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/add-putnam-downloader`
3. Make changes with tests
4. Run the test suite and ensure it passes
5. Submit a PR with a clear description of what you changed and why

### PR Description Template

```
## What this PR does
[1-2 sentence summary]

## Data impact (if applicable)
- Problems added: N
- Solutions added: N
- Lean 4 verification rate: X%

## Testing
- [ ] Python tests pass
- [ ] check_env.sh passes
- [ ] Manual testing of changed functionality
```

---

## Reporting Issues

Use GitHub Issues for:
- Bug reports (include stack trace, Python version, OS)
- Feature requests
- Data quality issues (wrong solutions, missing problems)
- Lean 4 verification failures on correct proofs

---

## Mathematical Standards

For any math-related contribution:
- All claimed proof steps must be Lean 4 verifiable (or clearly marked as not yet formalized)
- Solutions should reference standard olympiad techniques with correct names
- Difficulty ratings should follow AMC/AIME scale (1-10)

---

## License

By contributing, you agree that your contributions will be licensed under MIT (code) and CC BY 4.0 (data/content).
