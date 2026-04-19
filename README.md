# turnkv

Turn-aware KV-cache eviction for multi-turn LLM conversations. CMU 15-642 MLSys Spring 2026 project.

**Team**: Gaurav Shah · Pradyut Ganesh · Yagneek Patlola
**Proposal**: [`context/Project Proposal/MLSys_Project_Proposal.pdf`](./context/Project%20Proposal/MLSys_Project_Proposal.pdf)

## Repo layout

```
turnkv/
├── kvpress/          # NVIDIA kvpress fork — our 3 new Press classes and benchmarks land here
├── epicache/         # (gitignored) Apple's EpiCache baseline — clone locally, see below
├── context/          # papers, proposal, decisions, experiments, team logs
├── documentation/    # journal.md (observations) + findings.md (consolidated insights)
└── .claude/          # Claude Code project memory (CLAUDE.md)
```

## First-time setup

```bash
# 1. Clone this repo
git clone https://github.com/Gaurav-Shah05/turnkv.git
cd turnkv

# 2. Install kvpress (our fork, with SCBench scaffolding and TurnKV presses)
cd kvpress
uv sync --all-groups                          # main venv
cd ..

# 3. Clone EpiCache baseline into a sibling folder (gitignored)
git clone --depth 1 https://github.com/apple/ml-epicache.git epicache

# 4. Set up a SEPARATE venv for EpiCache (it pins older torch/transformers/numpy
#    that would conflict with kvpress — do NOT share a venv)
cd epicache
python -m venv .venv-epicache
source .venv-epicache/bin/activate            # Windows: .venv-epicache\Scripts\activate
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
make i
deactivate
cd ..
```

## Where things go

| Change type                                | Location                                                   |
|--------------------------------------------|------------------------------------------------------------|
| New Press class                            | `kvpress/kvpress/presses/<name>_press.py` + register in `kvpress/evaluation/evaluate_registry.py` |
| New benchmark                              | `kvpress/evaluation/benchmarks/<name>/`                    |
| Architecture / API decision                | `context/decisions/NNN-<slug>.md` (ADR)                    |
| Experiment result table                    | `context/experiments/NNN-<slug>.md`                        |
| "I worked on this today"                   | `context/logs/YYYY-WW-<firstname>.md`                      |
| Live observations, surprises, questions    | `documentation/journal.md` (append-only)                   |
| Consolidated insights (→ final paper)      | `documentation/findings.md`                                |

## Running things

```bash
# Tests (kvpress)
cd kvpress && make test

# Style / lint (kvpress)
cd kvpress && make style

# kvpress eval example (once SCBench integration is runnable — see kvpress/evaluation/benchmarks/scbench/README.md)
cd kvpress/evaluation
python evaluate.py --dataset scbench --data_dir scbench_kv \
    --press_name snapkv --compression_ratio 0.5 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct

# EpiCache baseline run (separate venv)
cd epicache
source .venv-epicache/bin/activate
bash scripts/run_epicache_eval_llama.sh 0 8 pair 4096 2048 False longmemeval 100000
```

## Team workflow

- `main` is protected. Never push directly.
- Feature branches: `<firstname>/<topic>` (e.g. `gaurav/turn-floor-press`).
- PR into `main`; `make test` (in `kvpress/`) must pass.
- For any change that affects the API other teammates build against (press class signatures, metrics format, multi-turn harness interface), drop an ADR in `context/decisions/` first.

## Status (2026-04-18)

- **Scaffolded**: SCBench benchmark loader + metrics (`kvpress/evaluation/benchmarks/scbench/`), team collaboration folders (`context/`, `documentation/`).
- **Next up** (roughly in order):
  1. Port SCBench metrics from upstream `microsoft/MInference/scbench/compute_scores.py` verbatim.
  2. Implement the multi-turn harness in `kvpress/evaluation/multi_turn_evaluate.py`.
  3. Scaffold the three turn-aware Press classes as `ScorerPress` subclasses with failing tests.
  4. First EpiCache baseline run on LongMemEval or LoCoMo with DeepSeek-R1-Distill-Llama-8B (or Llama-3.1-8B-Instruct for paper-comparable numbers).
  5. ADR: which model + budgets + datasets we standardize on.
