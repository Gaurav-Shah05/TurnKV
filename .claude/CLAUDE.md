# TurnKV project context (for Claude Code)

This repo — `turnkv` — is the top-level workspace for a CMU 15-642 MLSys class project.

**Team**: Gaurav Shah, Pradyut Ganesh, Yagneek Patlola
**Deadline**: final presentation 2026-04-30, final report 2026-05-03

## What we're building

Three new KV-cache eviction "Press" classes for NVIDIA's kvpress, designed to be **turn-aware** for multi-turn LLM conversations:

1. **Cross-Turn Accumulation Press** — running score tensor accumulated across turns with exponential decay; topic-shift detection via cosine similarity between turn key-vectors.
2. **Turn-Floor Press** — per-turn minimum budget (length-proportional) so early turns aren't fully evicted.
3. **Role-Boundary Anchor Press** — retention bonus around role boundaries (user/assistant tokens) to preserve intent-bearing tokens.

### Benchmark plan (final as of 2026-04-19 — see `documentation/findings.md` + ADR 001)

Core assumption: **long context is permanent (KEEP bucket), only turn content gets compressed (COMPRESS bucket).**

Two benchmarks cover two different signals:

- **Primary coding (per-turn accuracy curve)**: **ConvCodeWorld / ConvCodeBench** (1,140 BigCodeBench tasks × 5 feedback configs × 10-turn refinement trajectories). Mode 2 replay — teacher-force reference prior iters, our model generates at each iter, execute against BigCodeBench tests for pass/fail. Gives a real per-turn curve.
- **Primary conversational (single-probe accuracy on compressed long history)**: **LongMemEval_S** (500 probes × ~103K-token multi-session histories). Y1 setup — teacher-force the full prior history (~490 user/assistant turns), apply press at session boundaries, generate only at the probe. One score per probe. Direct head-to-head with EpiCache.
- **Topic-shift validation (Loyalty/topic-shift detection only)**: **TopiOCQA**. Ground-truth topic labels per turn. Not used for compression sweeps — validates one component.

SCBench and MultiDoc2Dial **dropped**: SCBench because turns are independent queries (no cross-turn dependency signal); MultiDoc2Dial because only ~230 tokens of compressible turn content per dialogue (padding docs doesn't help under the "context permanent" rule).

Primary baselines: SnapKV, StreamingLLM, ObservedAttention (H2O), KVzip, ExpectedAttention — all already in kvpress. External comparison: **EpiCache** (Apple, 2025) — the only existing multi-turn-aware method — run separately in `epicache/`.

Full proposal: `context/Project Proposal/MLSys_Project_Proposal.pdf`. Findings that justify this benchmark pivot are in `documentation/findings.md`.

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

### Benchmark scaffolding status (as of 2026-04-23)

| Benchmark | Status | Location |
|-----------|--------|----------|
| SCBench | Scaffolded (loader + metrics + flattening script). Demoted to appendix — see `documentation/findings.md`. | `kvpress/evaluation/benchmarks/scbench/` |
| ConvCodeWorld | Scaffolded. `live_loop.py` fully expanded: static-replay + live-loop modes, VRAM guards, FA3 flashdecode tracking, tokenizer-artefact normalisation, compilable-prefix fallback. `modal_app.py` rewritten with extracted constants + reproducible FA3 build layer. `modal_run.sh` rewritten. `MODAL_HYPERPARAMS.md` added. `executor.py` has `normalize_tokenizer_artifacts`, `normalize_candidate_code`, `_longest_compilable_prefix`. | `kvpress/evaluation/benchmarks/convcodeworld/` |
| LongMemEval | **Not yet scaffolded.** Primary conversational. Reuse EpiCache's `data/longmemeval/convert_longmemeval.py` as the data-prep entry point. | TBD `kvpress/evaluation/benchmarks/longmemeval/` |
| TopiOCQA | **Not yet scaffolded.** Topic-shift validation. Load via `datasets.load_dataset("McGill-NLP/TopiOCQA", "plain_text")`. | TBD `kvpress/evaluation/benchmarks/topiocqa/` |

**Week-1 press primitives landed** (2026-04-22): `TurnBoundary`, `TurnAwareMixin`, `TurnFloorPress` (policy A), `RoleBoundaryAnchorPress` (policy B), `TurnAwareGlobalPress` (composer). Tests green. Registry updated with `turnkv_*` and `baseline_*` entries. `LoyaltyPress` (policy C) is the remaining press primitive.

**ConvCodeWorld execution infrastructure** (2026-04-23): `live_loop.py` expanded with static-replay/live-loop modes, VRAM guards, FA3 flashdecode tracking, tokenizer-artefact normalisation, compilable-prefix fallback. `attention_patch.py` adds `reset_flashdecode_tracking` / `flashdecode_used_layers`. `executor.py` adds `normalize_tokenizer_artifacts`, `normalize_candidate_code`, `_longest_compilable_prefix`. `modal_app.py` rewritten with extracted constants + reproducible FA3 build layer. `modal_run.sh` rewritten to run `no_press` full-cache baseline. `MODAL_HYPERPARAMS.md` added as CLI flag reference.

**Shared blocker**: the multi-turn harness (`kvpress/evaluation/multi_turn_evaluate.py`). The existing `evaluate.py` treats each question independently; all four benchmarks need a harness that preserves the KV cache across turns and re-applies the press at each boundary. This is the heart of the project. **Architecture and API locked down in `context/decisions/001-multi-turn-harness.md`** (2026-04-19). Teammates should implement against that ADR.

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
