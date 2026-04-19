# Journal

Append new entries at the **top**. Each entry is dated and signed.

Format:
```
## YYYY-MM-DD — short title — @author
body (1-N paragraphs)
```

---

## 2026-04-19 — final benchmark lineup, "context permanent" assumption, Mode 2 / Y1 setups — @gaurav

Several discussions locked down:

- **Long context is permanent (KEEP bucket).** kvpress has no sparse-retrieval; everything in cache is what the model sees. We reserve the initial document/prompt as permanent, and compression applies only to accumulated turn content (COMPRESS bucket: Q + A across turns). This changes which benchmarks make sense: padding a short doc doesn't help because docs are never compressed.
- **ConvCodeWorld evaluation is Mode 2 replay.** At each iter, teacher-force the reference model's prior code + feedback; our model generates this iter's code; we execute against BigCodeBench unit tests for pass/fail. All 10 iters always run. Requires subprocess-based code execution.
- **LongMemEval_S setup is Y1 (teacher-forced history).** Prefill the full ~490-turn chat history verbatim (no model generation until probe). Apply press at session boundaries. One accuracy score per probe. Matches EpiCache's evaluation exactly.
- **Loyalty updates during both prefill and decode.** Any time the model runs attention over past-turn tokens (not just from decode), past-turn tokens in the top-25% get +1 loyalty. Current-turn tokens never accumulate.
- **MultiDoc2Dial dropped.** Measured: only ~230 tokens of compressible turn content per dialogue. Too small to stress compression under "context permanent" rule.
- **Final benchmark lineup**: ConvCodeWorld (per-turn curve) + LongMemEval_S (long-history single-probe) + TopiOCQA (targeted loyalty validation).

Next: implement the two-regime harness, the three press classes, and the ConvCodeWorld execution module. Team split TBD.

## 2026-04-19 — ADR 001 drafted: multi-turn harness architecture — @gaurav

Wrote `context/decisions/001-multi-turn-harness.md`. Locks down the API between the multi-turn harness and the turn-aware presses before anyone starts coding. Key decisions:

- **Two-regime compression**: local (intra-turn via `DecodingPress`, unchanged) + global (at turn boundary, three-policy weighting × base scorer).
- **Per-benchmark `global_budget`** calibrated to fire at turn 4: `median_initial_context + 4 × median_per_turn_tokens`. ~105K for LongMemEval_S, ~4.5K for ConvCodeWorld, ~2K for TopiOCQA.
- **Three policies act as multiplicative weights** on the base press's scorer. Turn-Floor (across-turn), Role-Boundary Anchor (within-turn positional), Loyalty (top-25% attention persistence from past turns only).
- **Four base techniques**: SnapKV, Ada-SnapKV, StreamingLLM, ExpectedAttention — all flash_attention_2 compatible. KVzip dropped from core (incompatible scoring mechanism); kept as stretch via bespoke subclass.
- **Turn-boundary protocol**: harness owns metadata (list of `TurnBoundary` entries); calls `press.on_turn_end(turn_idx, role, start_kv, end_kv)` after each turn. Tokenizer-agnostic.
- **Baselines** use the same eviction surface and same budget — only difference is whether the three-policy weights are applied.
- **Work split**: Pradyut → policies A+B, Yagneek → policy C + loyalty integration, Gaurav → harness + global wrapper.

Open questions deferred: ConvCodeWorld no-op-turn handling, TopiOCQA aggressive-compression feasibility, α tuning, KVzip stretch.

## 2026-04-19 — benchmark pivot: SCBench out, LongMemEval + ConvCodeWorld in — @gaurav

Finished measuring SCBench per-problem structure (stats + CSVs in `context/experiments/`). Finding: **turns within a session are independent queries over a shared long context, not conversational.** The three proposed presses (Cross-Turn Accumulation, Turn-Floor, Role-Boundary Anchor) all assume turn-to-turn dependency — SCBench doesn't exercise that.

Surveyed replacement benchmarks. New plan:
- **Primary conversational**: **LongMemEval_S** — ~115K-token multi-session histories, evidence deliberately placed many turns before the probe. MIT. EpiCache ships a converter so data-prep is cheap.
- **Primary coding**: **ConvCodeWorld / ConvCodeBench** — 1,140 BigCodeBench tasks × 5 feedback configs × 10-turn refinement trajectories. Median ~4.5K tokens per trajectory (measured on 50-task sample). Per-turn pass/fail labels from the dataset, so no LLM judge needed. Fits compression sweeps at 1/2, 1/4, 1/8, 1/16, 1/32.
- **Topic-shift validation**: **TopiOCQA** — has ground-truth topic labels per turn; cleanly isolates Cross-Turn Accumulation's topic-shift detection.
- **Appendix**: SCBench reframed as "cross-query KV retention on shared contexts" — the press effect is still measurable there, just doesn't justify the "turn-aware" framing.

Scaffolded `kvpress/evaluation/benchmarks/convcodeworld/` this session (README, metrics, flattening script). Registered in `evaluate_registry.py` alongside existing `scbench` entry. LongMemEval + TopiOCQA still need scaffolds (next session).

Confirmed ConvCodeWorld dataset structure by actually loading it: 5 dict-typed columns (one per feedback combo), each with `ITER=1..10` keys, each iteration holding task-indexed arrays of code + feedback + pass/fail. 1,140 tasks × 5 configs = 5,700 distinct 10-turn trajectories. License unstated on the HF card — flag to verify before publication.

## 2026-04-18 — repo restructured to turnkv layout — @gaurav

Moved all kvpress-native code (including evaluation/, tests/, notebooks/, kvzap/, pyproject.toml) into `kvpress/` subfolder. Added `documentation/` (this file + `findings.md`), `.claude/CLAUDE.md`, and the parent README. `epicache/` is gitignored — each teammate clones `apple/ml-epicache` locally. `context/` stays where it was, now with scaffolding (decisions/, experiments/, logs/, templates). GitHub repo will be renamed `turnkv` via settings; the fork network was already detached.

SCBench benchmark scaffold landed at `kvpress/evaluation/benchmarks/scbench/` — loader + approximate metrics + register in `evaluate_registry.py`. **Not yet runnable end-to-end**: (a) microsoft/SCBench schema needs flattening via `create_huggingface_dataset.py`, (b) metrics need reconciliation with upstream `compute_scores.py`, (c) true multi-turn harness (preserving KV cache across turns) is the next substantial piece of work.
