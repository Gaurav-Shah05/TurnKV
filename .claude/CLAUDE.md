# TurnKV project context (for Claude Code)

This repo — `turnkv` — is the top-level workspace for a CMU 15-642 MLSys class project.

**Team**: Gaurav Shah, Pradyut Ganesh, Yagneek Patlola
**Deadline**: final presentation 2026-04-30, final report 2026-05-03

## What we're building

Three new KV-cache eviction "Press" classes for NVIDIA's kvpress, designed to be **turn-aware** for multi-turn LLM conversations:

1. **Cross-Turn Accumulation Press** — running score tensor accumulated across turns with exponential decay; topic-shift detection via cosine similarity between turn key-vectors.
2. **Turn-Floor Press** — per-turn minimum budget (length-proportional) so early turns aren't fully evicted.
3. **Role-Boundary Anchor Press** — retention bonus around role boundaries (user/assistant tokens) to preserve intent-bearing tokens.

Primary benchmark: **SCBench** (Li et al., ICLR 2025). Primary baselines: SnapKV, StreamingLLM, ObservedAttention (H2O), KVzip, ExpectedAttention, all already in kvpress. External comparison: **EpiCache** (Apple, 2025) — the only existing multi-turn-aware method — run separately in `epicache/`.

Full proposal: `context/Project Proposal/MLSys_Project_Proposal.pdf`.

## Repo structure

```
turnkv/
├── kvpress/              # NVIDIA kvpress fork — all the Python code lives here
│   ├── kvpress/          # the package
│   ├── evaluation/       # benchmark harness; new presses register in evaluate_registry.py
│   ├── tests/            # pytest suite
│   └── pyproject.toml    # install with `cd kvpress && uv sync --all-groups`
├── epicache/             # gitignored — clone `apple/ml-epicache` here for baseline runs (own venv)
├── context/              # team coordination — papers, decisions, experiments, logs
├── documentation/        # journal.md (observations) and findings.md (curated insights)
└── .claude/              # this folder — Claude Code config + project brief
```

## How to work here

### Default to editing the kvpress/ subfolder

Any code change — new press class, metric tweak, eval harness, test — happens inside `kvpress/`. Python imports are kvpress-rooted (`from kvpress.presses.scorer_press import ScorerPress`). The package's own `pyproject.toml`, `Makefile`, and `tests/` are at `kvpress/` level.

### Workflow for non-trivial work

1. Write/update a decision record in `context/decisions/` if the change affects how others build against it (new Press API, metrics shape, multi-turn harness interface). Use the template in `context/decisions/000-template.md`.
2. Open a feature branch `<firstname>/<topic>`.
3. PR into `main`; `make test` must pass.
4. Log observations to `documentation/journal.md` as you go. Promote to `documentation/findings.md` when consolidated.

### SCBench integration status (as of 2026-04-18)

- Scaffolded at `kvpress/evaluation/benchmarks/scbench/` (loader, metrics, create_huggingface_dataset.py).
- Registered in `kvpress/evaluation/evaluate_registry.py` as `"scbench"`.
- **NOT yet runnable end-to-end.** Three TODOs in `kvpress/evaluation/benchmarks/scbench/README.md`:
  1. Port metrics from `microsoft/MInference/scbench/compute_scores.py` verbatim.
  2. Run the flattening script + either publish the reshaped HF dataset or load locally.
  3. Implement `multi_turn_evaluate.py` — the turn-aware runner. This is the heart of the project.

### EpiCache baseline

Clone upstream separately, keep it in its own venv (it pins transformers 4.51.3, numpy 1.26.4, torch 2.3.0 — incompatible with kvpress). Edit `scripts/run_epicache_eval_llama.sh` to override `MODEL` if targeting DeepSeek-R1-Distill-Llama-8B (the Llama monkeypatch activates on substring match, so the architecture works).

See `context/decisions/` for the ADR recording the EpiCache run configuration we've agreed on.

## Style & conventions

- **Commit messages**: imperative; reference the file(s) changed; if the change implements an ADR, reference it.
- **SPDX headers**: every new `.py` file starts with the NVIDIA Apache-2.0 header (see any existing file in `kvpress/kvpress/presses/` for the exact line).
- **Line length**: 120 (black config in `kvpress/pyproject.toml`).
- **No emojis in code or docs** unless a teammate requests them.

## What NOT to do

- Don't edit `kvpress/kvzap/` — it's an upstream kvpress satellite (KVzap paper), not our code.
- Don't push to `main` directly.
- Don't add EpiCache files to the turnkv repo — it's gitignored for a reason (license + dependency conflicts).
- Don't vendor the SCBench repo; the dataset is on HF (`microsoft/SCBench`), their code is vLLM-coupled and not reusable.
- Don't rewrite NVIDIA's CI in `.github/workflows/` — just make them run from `kvpress/` subdirectory (`working-directory: kvpress`).
