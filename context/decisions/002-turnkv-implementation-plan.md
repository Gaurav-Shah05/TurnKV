# 002: TurnKV implementation plan

- **Status**: Accepted
- **Date**: 2026-04-21
- **Authors**: Gaurav Shah, Pradyut Ganesh, Yagneek Patlola
- **Depends on**: [001-multi-turn-harness.md](001-multi-turn-harness.md)

## Context

ADR 001 locked the multi-turn API (`TurnBoundary`, `TurnAwareMixin`, `on_turn_end`), the two compression cadences (local via `DecodingPress`, global via a new turn-boundary press), the three weight-producing policies, the four base techniques, the per-benchmark `global_budget`s, and the evaluation protocols for the three primary benchmarks. That ADR intentionally stopped short of naming files.

This ADR names the files, the work order, the correctness process, and the judge infrastructure. It is the executable companion to 001. The three of us should be able to start coding without further design discussion.

## Decision

### 1. Model, precision, and judge

**Model**: DeepSeek-R1-Distill-Llama-3.1-8B, bf16. 8B bf16 weights (~16 GB) + 128K KV (~16 GB) fits on a single A100-80GB.

**Judge**: Gemma 4 E4B, self-hosted on Modal. No GPT, no OpenRouter, no OpenAI. The MLSys narrative is cleaner without a closed-model dependency (reproducible weights, no rate limits, no outage risk), and cost drops to ~$6 for the full ablation matrix on an L4.

Sanity-gated by a one-time human meta-evaluation before we trust the judge across ~20,000 yes/no verdicts (40 runs × 500 probes). One labeler is enough — the LongMemEval authors validated GPT-4o with expert annotators and got 97% agreement; we only need to confirm a 4B-class open judge clears 90% on closed-form QA. **Runbook** (whoever owns LongMemEval):

1. Stand up the Modal Gemma endpoint (`modal_judge_app.py`) and confirm it responds to a single call.
2. Generate one baseline predictions CSV: `python multi_turn_evaluate.py --benchmark longmemeval_s --press_name baseline_snapkv --compression_ratio 0.25`. Uses real model predictions, not fresh ones.
3. Sample 50 probes stratified by question type: 7 each from single-session-user, single-session-assistant, multi-session, knowledge-update, temporal-reasoning; 5 from single-session-preference; 5 from abstention. Save to `meta_eval/sample.jsonl`.
4. Label manually. Open a small CLI / notebook that shows `question`, `golden_answer`, `model_prediction` one at a time; you enter `y`, `n`, or `?` (uncertain). `?` cases get escalated to the other two teammates on Slack — aim for ≤ 5 escalations. Save to `meta_eval/human.jsonl`. Budget ~1 hour for the 50.
5. Run Gemma on the same 50 via `judge.py`. Save to `meta_eval/gemma.jsonl`.
6. Compute agreement: `python -c "import json; h=[json.loads(l) for l in open('meta_eval/human.jsonl')]; g=[json.loads(l) for l in open('meta_eval/gemma.jsonl')]; print(sum(a['label']==b['label'] for a,b in zip(h,g))/len(h))"`.
7. Write up the result in `context/experiments/meta_eval_report.md`: the 50 probes, human labels, Gemma verdicts, disagreements (so we can see *where* Gemma fails), final agreement rate. Commit this file to the repo.
8. **If agreement ≥ 90%**: adopt. Proceed to the full matrix.
9. **If < 90%**: inspect the disagreements, refine the Gemma prompt (four variants are in LongMemEval paper Appendix A.4 — try them), re-run step 5 on the same 50. Up to three prompt revisions. If still below 90% after three tries, flag in Slack and we decide together; do **not** silently switch judge models.

**CoT**: off by default on every benchmark, including ConvCodeWorld. `--cot` is an explicit flag. This is deliberately a second experiment axis (Group 9 in TurnKV.pdf §Potential experiments), not a benchmark property. Teammates should not assume it's on.

### 2. File plan

All paths relative to repo root.

#### Press primitives — `kvpress/kvpress/presses/`

| File | LOC | Role |
|------|----:|------|
| `turn_aware_base.py` | 80 | `TurnBoundary` dataclass and `TurnAwareMixin` with defaults for `on_turn_start`, `on_turn_end`, `update_loyalty`, and `compute_weights(kv_len) -> Tensor`. |
| `turn_floor_press.py` | 120 | Policy A (ADR 001 §3-A). `floor_k = max(c=5, α_len=0.3 × G × \|T_k\| / Σ\|T_j\|) × γ^(current−k)` with γ=0.9. Turns shorter than 10 tokens are exempt. Inherits `TurnAwareMixin` only; does not evict. |
| `role_boundary_anchor_press.py` | 80 | Policy B (ADR 001 §3-B). `w_k = max(3, ⌊β × \|T_k\|⌋)` with β=0.15. User turns reserve first + last w tokens; assistant and feedback turns reserve last w only. |
| `loyalty_press.py` | 180 | Policy C (ADR 001 §3-C). Per-token integer counter. Recomputes attention internally (`softmax(Q·Kᵀ/√d)` from `hidden_states` and `keys`, RoPE from `module.rotary_emb`). Increments on positions in top-25% per query. Updates during both prefill and decode of the current turn; past-turn counts frozen. Follows the pattern in `snapkv_press.py`. **Flash-attention-2 compatible** — never requests `output_attentions=True`. |
| `turn_aware_global_press.py` | 220 | The weighted composer. Fields: `base_press: ScorerPress \| AdaKVPress`, `policies: list[TurnAwareMixin]`, `alphas: dict[str, float]`, `global_budget: int`. Exposes `run_global_compression(cache, target_len)` — harness-invoked, not hook-invoked. Final score: `base_scorer(t) × (1 + Σ αᵢ × wᵢ(t))`. Its `forward_hook` drives `LoyaltyPress.update_loyalty` on current-turn attentions; does not evict. |

**Reuse** (do not reimplement):
- `ScorerPress.compress()` eviction pattern — `kvpress/kvpress/presses/scorer_press.py:76`
- `DecodingPress` — `kvpress/kvpress/presses/decoding_press.py:22`, unchanged
- `BasePress.__call__` context manager — `kvpress/kvpress/presses/base_press.py:158`
- SnapKV-style attention recompute — `kvpress/kvpress/presses/snapkv_press.py:17`

#### Registry

- `kvpress/evaluation/evaluate_registry.py`: add `turnkv_{snapkv, adakv_snapkv, streaming_llm, expected_attention}` (all αs = 1.0) plus `baseline_*` twins (all αs = 0.0). Ablation toggle = single registry key swap.
- `kvpress/kvpress/__init__.py`: export the five new classes.

#### Harness — `kvpress/evaluation/`

| File | LOC | Role |
|------|----:|------|
| `multi_turn_evaluate.py` | 350 (2 chunks) | Fire CLI sibling to `evaluate.py`. Chunk 1 (~175 LOC): `MultiTurnConfig`, `MultiTurnRunner`, `_setup_*` methods (subclass existing `EvaluationRunner`). Chunk 2 (~175 LOC): `run()` loop, per-turn driver, `run_global_compression` invocation, metrics dispatch. Config fields: `global_budget`, `local_budget=4096`, `alpha_floor`, `alpha_anchor`, `alpha_loyalty`, `benchmark`, `cot: bool=False`. |

#### Benchmark loaders

| File | LOC | Role |
|------|----:|------|
| `benchmarks/longmemeval/multi_turn_loader.py` | 150 | Loads `context/datasets/longmemeval/longmemeval_s_cleaned.json`. Y1 teacher-forced protocol. Yields `TurnSpec(role, text, teacher_force, is_generation_target, metadata)`. |
| `benchmarks/longmemeval/judge.py` | 100 | Modal-hosted Gemma client; local process → Modal RPC. Prompts from LongMemEval paper Appendix A.4 (four variants). Cache to `~/.cache/turnkv/judge/{sha256}.json`. |
| `benchmarks/longmemeval/modal_judge_app.py` | 80 | Modal app: Gemma 4 E4B, `@app.function(gpu="L4")`, HF weights baked into the image. |
| `benchmarks/convcodeworld/multi_turn_loader.py` | 170 | Mode 2 static replay. Turn 1 = problem statement (teacher-forced). Turns 2-10 alternate our model's code generation and teacher-forced reference feedback. `--cot` prepends a reasoning prompt. |
| `benchmarks/convcodeworld/executor.py` | 180 | Linux subprocess sandbox: `resource.setrlimit(RLIMIT_AS, 1GB)`, 30s timeout, `unshare -n`, empty `env`. Windows: mocked. |
| `benchmarks/convcodeworld/calculate_metrics.py` | +60 | Extend the existing file with per-turn Pass@1, MRR, Recall (ConvCodeBench §5.1). No new module; a `turn_index` groupby is enough. |
| `benchmarks/topiocqa/multi_turn_loader.py` | 100 | `(passage, Q, A)` triples. Topic labels carried in metadata. |
| `benchmarks/topiocqa/eviction_quality.py` | 100 | Group 10 diagnostic. Per-boundary `current_topic_evicted_pct` and `retired_topic_evicted_pct`. |

#### Sweep runner

| File | LOC | Role |
|------|----:|------|
| `kvpress/evaluation/sweep.py` | 200 | Fire CLI: `--group {1..10}`. Emits bash commands wrapping `multi_turn_evaluate.py`. |
| `configs/sweeps/group{1..10}.yaml` | 10 × 30 | YAML presets for the ten ablation groups from TurnKV.pdf §Potential experiments. |

#### Tests

| File | LOC | Scope |
|------|----:|-------|
| `kvpress/tests/presses/test_turn_aware.py` | 200 | Floor allocation sums; anchor window bounds; loyalty increments only on top-25%; composer output shape; `run_global_compression` hits target; **all-αs-zero produces bit-identical evictions to un-weighted base** (the critical regression guard). |
| `kvpress/tests/integration/test_multi_turn_harness.py` | 100 | 10-turn fake session on the `unit_test_model` fixture. Cache stays ≤ budget; callbacks fire in order; no OOM; predictions non-empty. |

**Totals**: ~2,500 LOC + ~300 LOC tests across ~22 files.

### 3. Per-file correctness process

Every file ≤ 300 LOC is written as one chunk. Files > 300 LOC are split into ≤ 300 LOC chunks; only `multi_turn_evaluate.py` is affected, split into two chunks.

After each chunk is written:
1. Spawn two independent code-review agents in parallel.
2. Each agent reviews the chunk against ADR 001 §N (for the relevant policy) and the existing kvpress patterns in `scorer_press.py`, `decoding_press.py`, and `snapkv_press.py`.
3. Agents explicitly flag: tensor shape mismatches, flash-attention-2 incompatibilities, any behaviour that differs between `all-αs=0` and the un-weighted base, RoPE application, GQA head broadcasting.
4. Only advance to the next chunk or file when both pass.

This process is non-negotiable for the press primitives in Week 1. It is advisory for Week 2 and 3 files, where standard PR review is acceptable.

### 4. Work order

**Week 1 — Gaurav solo.** Press primitives in dependency order: `turn_aware_base` → `turn_floor_press` + `role_boundary_anchor_press` → first three unit tests → `loyalty_press` → `turn_aware_global_press` → remaining unit tests → integration test on `unit_test_model` → registry updates. Exit criterion: `turnkv_snapkv` CLI-usable, tests green on the tiny model fixture.

**Week 2 — three-way parallel.**
- Gaurav: `multi_turn_evaluate.py` (two chunks) + `topiocqa/` loader + `eviction_quality.py`.
- Pradyut: `longmemeval/` loader → `modal_judge_app.py` → `judge.py`. Stand up Modal first so the human meta-evaluation can run end-to-end on the 50-probe sample before the full runs start.
- Yagneek: `convcodeworld/` loader → `executor.py` (slowest piece). Verify the sandbox against BigCodeBench's own unit tests on a 10-task subset before committing.

Mid-Week-2 checkpoint: one baseline press × one policies-on press × one compression ratio × all three benchmarks produces a 6-row table. The regression guard must pass: `baseline_snapkv` matches stock `snapkv` on LongBench single-turn within ±0.5%.

**Week 3 — ablations.** `sweep.py` + YAML presets first. Then Group 1 → 2 → 3 in order (headline numbers are in Group 3). Groups 4–7 (hyperparameter sweeps) as GPU allows. Group 9 (CoT ConvCodeWorld) piggybacks on 2/3 runs via the `--cot` flag. Group 10 (TopiOCQA eviction diagnostic) is a post-hoc log read. Group 8 (local budget sweep) last; skip if time-tight.

### 5. Compute and cost

| Benchmark | Time per run | 40 runs | Notes |
|-----------|-------------:|--------:|-------|
| LongMemEval_S | ~1 h | 40 GPU-h | 500 probes × ~105K prefill |
| ConvCodeWorld (CF_EF_UNIT_SNF) | ~3 h | 120 GPU-h | 1,140 tasks × 10 turns. Use 200-task subset for sweeps; full 1,140 only for headline. |
| TopiOCQA (val) | ~20 min | 13 GPU-h | 205 conversations × 12 turns |

Headline: ~170 GPU-h. Ablations: +50-80 GPU-h. Judge: ~$6 total.

### 6. Team contract

- Any turn-aware press subclasses `TurnAwareMixin`.
- Any benchmark loader yields an iterator of `TurnSpec`.
- Any new metric plugs into `SCORER_REGISTRY` and consumes a DataFrame with `session_id`, `turn_index`, `predicted_answer` columns.
- `multi_turn_evaluate.py` is the only way to invoke turn-aware runs. `evaluate.py` stays single-turn-only.
- `--cot` defaults off. Full stop.

### 7. Correctness-critical tests (must land in Week 1)

- `test_all_alphas_zero_equivalent_to_base`: with αs = 0, `TurnAwareGlobalPress(SnapKVPress())` produces bit-identical evictions to stock `SnapKVPress`. If this fails, the whole ablation matrix is meaningless.
- `test_loyalty_updates_during_prefill`: replay a three-turn fake session, assert loyalty > 0 on positions attended to in top-25% during turn 2's prefill.
- `test_turn_floor_exempts_short_turns`: a five-token turn gets zero floor allocation.
- `test_role_boundary_assistant_last_only`: an assistant-role turn has ones at `[end−w, end)` and zeros at `[start, start+w)`.
- `test_budget_hit`: `run_global_compression(target=N)` always exits with `kv_len == N`.

### 8. Non-goals for v1

KVzip integration, CoT-aware loyalty, dynamic alpha, ConvCodeWorld Mode 1 (live loop), Slurm or Modal auto-scheduling beyond the bash script that `sweep.py` emits.

## Consequences

**What lands in the repo by end of Week 1**
Five new press files, a test suite that fails-first then passes, registry updates. The ablation machinery exists but isn't wired to real data yet.

**What lands by end of Week 2**
A harness that can run one (benchmark, press, ratio) cell end-to-end on all three benchmarks. Modal Gemma judge responds. Sandbox executes ConvCodeWorld attempts.

**What lands by end of Week 3**
Headline table (3 benchmarks × 4 base × 2 variants × 5 ratios). Per-turn accuracy plots. Eviction quality diagnostic on TopiOCQA. Draft writing using the existing paper scaffold.

**What fails the plan**
If `test_all_alphas_zero_equivalent_to_base` fails: ablation matrix is meaningless, rework the composer before proceeding.
If Modal Gemma falls below 90% human agreement on the 50-probe meta-eval: iterate on the prompt, or as last resort switch judge model. **Do not** silently fall back to GPT — that re-introduces the closed-model dependency we just removed.
If ConvCodeWorld sandbox leaks network access: kill runs, fix the `unshare -n` invocation, re-verify before proceeding.

## Verification

**Unit**: `cd kvpress && uv run pytest tests/presses/test_turn_aware.py -v`
**Integration**: `cd kvpress && uv run pytest tests/integration/test_multi_turn_harness.py -v`
**Smoke end-to-end** (Week 2 exit):
```bash
cd kvpress/evaluation
python multi_turn_evaluate.py \
    --benchmark topiocqa \
    --press_name turnkv_snapkv \
    --compression_ratio 0.25 \
    --fraction 0.01
```
**Regression guard** (Week 2 exit): `baseline_snapkv` on LongBench within ±0.5% of stock `snapkv`.
