# TurnKV — Project Context

Shared workspace for the three-person TurnKV team. Code lives in the rest of the repo; this folder is where we coordinate.

## Folder layout

```
context/
├── README.md                    # this file
├── decisions/                   # one .md per design decision (ADR-style)
├── experiments/                 # one .md per experiment run or analysis
├── logs/                        # one .md per person per week (worklog)
├── papers/                      # reference PDFs (already populated)
├── Project Proposal/            # proposal PDF (already populated)
├── mlsys2025style/              # paper template (already populated)
└── gitingest/                   # source dumps of external repos for reference
```

## How we work together

### Branches & PRs

- `main` is protected — never push directly.
- Each person works on feature branches: `gaurav/<topic>`, `pradyut/<topic>`, `yagneek/<topic>`.
- Open a PR for anything non-trivial. Self-assign a reviewer from the other two.
- `make test` must pass before merge.

### Where things go

- **Press classes & benchmarks** → `kvpress/kvpress/presses/`, `kvpress/evaluation/benchmarks/`
- **Design decisions (durable)** → `context/decisions/` (ADR-style, see template)
- **Experiment records** → `context/experiments/` (see template)
- **Running observations / questions / "huh, that's odd"** → `documentation/journal.md` (append-only log)
- **Consolidated insights worth remembering** → `documentation/findings.md` (curated)
- **What I did this week** → `context/logs/YYYY-WW-<name>.md`
- **Ephemeral notes** → PR description, not here

### Before merging something non-trivial

If the change encodes a design choice that affects other people's work (e.g. the shape of a `Press` class, the multi-turn harness API, the metrics format), drop a one-page ADR in `decisions/` first. Templates below.

## Templates

### Decision record — `decisions/NNN-<slug>.md`

```markdown
# NNN: <title>

- **Status**: Proposed | Accepted | Rejected | Superseded by NNN
- **Date**: YYYY-MM-DD
- **Authors**: Name, Name

## Context
What problem are we solving? What constraints apply?

## Decision
What did we decide?

## Alternatives considered
What else did we look at and why not those?

## Consequences
What does this change about how we build/test/report?
```

### Experiment record — `experiments/NNN-<slug>.md`

```markdown
# NNN: <experiment name>

- **Date**: YYYY-MM-DD
- **Owner**: Name
- **Branch / commit**: <sha>

## Question
What are we trying to find out?

## Setup
- Model: ...
- Benchmark subset: ...
- Press + compression ratio: ...
- Hardware: ...

## Results
Numbers, with a link to the predictions.csv / metrics.json in the results dir.

## Takeaway
One or two sentences. What do we now believe that we didn't before?
```

### Worklog — `logs/YYYY-WW-<name>.md`

Informal. One bullet per working session. Link PRs, ADRs, experiments by filename.

## Useful references

- **Proposal**: [Project Proposal/MLSys_Project_Proposal.pdf](./Project%20Proposal/MLSys_Project_Proposal.pdf)
- **Grading & deadlines**: [project_logistics_grading.pdf](./project_logistics_grading.pdf) — final presentation 2026-04-30, final report 2026-05-03
- **Paper template**: [mlsys2025style/](./mlsys2025style/)
- **kvpress Press API**: [../kvpress/kvpress/presses/scorer_press.py](../kvpress/kvpress/presses/scorer_press.py) (override `score()`)
- **SCBench dataset**: https://huggingface.co/datasets/microsoft/SCBench — see [`../kvpress/evaluation/benchmarks/scbench/README.md`](../kvpress/evaluation/benchmarks/scbench/README.md) for integration notes
- **EpiCache** (baseline, separate venv): `../epicache/` — cloned locally per `README.md`; evaluated on LongMemEval / LoCoMo / Realtalk
- **Journal & findings**: [../documentation/journal.md](../documentation/journal.md), [../documentation/findings.md](../documentation/findings.md)
