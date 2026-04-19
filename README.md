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

# 4. (Optional but required for LongMemEval) Download the cleaned LongMemEval data
# Link: https://drive.google.com/file/d/1zo5C2sKsN3-2TUZt7kiRd2wsZLmyd-4y/view
# Save it to the project root as longmemeval-data-cleaned.tar.gz, then:
tar -xzf longmemeval-data-cleaned.tar.gz -C context/datasets/
mv context/datasets/data context/datasets/longmemeval
# Both the tarball and the extracted JSON are gitignored.
# See context/experiments/longmemeval_cleanup_status.md for schema + cleanup notes.

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

# kvpress eval example (once the multi-turn harness ships)
# See kvpress/evaluation/benchmarks/{convcodeworld,scbench,longmemeval,topiocqa}/README.md
cd kvpress/evaluation
python evaluate.py --dataset convcodeworld --data_dir CF_EF_UNIT_SNF \
    --press_name snapkv --compression_ratio 0.875 \
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

## Benchmark plan (revised 2026-04-19)

Per-problem inspection of SCBench revealed its "multi-turn" mode is independent queries over a shared context, not conversational. The three proposed presses assume turn-to-turn dependency — SCBench doesn't exercise that. Full rationale in `documentation/findings.md`.

| Role | Benchmark | Why |
|------|-----------|-----|
| Primary conversational | **LongMemEval_S** | ~115K-token histories, evidence placed many turns before probe. MIT. EpiCache ships a data converter. |
| Primary coding | **ConvCodeWorld / ConvCodeBench** | 1,140 BigCodeBench × 5 feedback configs × 10-turn refinement trajectories. Verifiable pass/fail labels. |
| Topic-shift validation | **TopiOCQA** | Ground-truth topic labels per turn — isolates Cross-Turn Accumulation's topic-shift detection. |
| Appendix | **SCBench** (reframed) | Cross-query KV retention on shared contexts. Numbers still useful; claim rewritten. |

## Status (2026-04-19)

- **Scaffolded**: SCBench (demoted-appendix), ConvCodeWorld (primary coding), team collaboration folders.
- **Not yet scaffolded**: LongMemEval, TopiOCQA — next.
- **Shared blocker across all four**: the multi-turn harness (`kvpress/evaluation/multi_turn_evaluate.py`). ADR for its API lives at `context/decisions/001-multi-turn-harness.md` (TBD).
- **Next up** (roughly in order):
  1. ADR `001-multi-turn-harness.md` — lock down the turn-aware runner API before anyone codes against it.
  2. Scaffold LongMemEval + TopiOCQA benchmarks following the ConvCodeWorld/SCBench template.
  3. Port per-benchmark metrics to match upstream reference implementations.
  4. Scaffold the three turn-aware Press classes as `ScorerPress` subclasses with failing tests.
  5. First EpiCache baseline run on LongMemEval and LoCoMo with Llama-3.1-8B-Instruct (paper-comparable numbers) plus DeepSeek-R1-Distill-Llama-8B (stretch).
