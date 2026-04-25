# Journal

Append new entries at the **top**. Each entry is dated and signed.

Format:
```
## YYYY-MM-DD — short title — @author
body (1-N paragraphs)
```

---

## 2026-04-25 — smoke #2: live-loop on the same 228-task tune split — @gaurav

Re-ran the same TurnKV(α=1,1,1) vs SnapKV smoke under `--benchmark-mode live`
(model generates each iter, executor runs it, Gemma-3-4b-it writes verbal
feedback — the actual Mode-1 ConvCodeWorld pipeline). Same model, same alphas,
same 228-task split; only the harness mode flipped. Dispatched at 01:42 EDT,
landed ~02:30 EDT (~46 min/shard with 10-way parallelism).

Headline: **TurnKV and SnapKV stay tied in live too** (recall 36.59 each on
the 205-task common subset; baseline shard 4 OOM'd on first dispatch and is
re-running). Live mode beats static by +3.91 pp recall on *both* presses
(32.68 → 36.59), so the lift is harness×model not policy. The compile_error
spike that worried us in static (+88) does *not* compound in live (+2 only,
because the model regenerates fresh code each iter rather than amending a
teacher-forced trajectory). The new structural signal in live is timeouts:
4 → 17 (+13) under TurnKV, same root cause we suspected — eviction
occasionally removes import statements / function signatures, forcing
regenerate-from-scratch loops that hit the 30-s executor cap.

Direct answer to the "is live better for TurnKV?" question: **no, not at
α=(1,1,1).** Predictable next step: ablate to find which policy is helping
vs hurting. Loyalty (data-driven) is the prime suspect for "the helpful one";
anchor over-protection is the prime suspect for "the hurting one". Plan to
test α=(0,0,1) (Loyalty-only) next on live mode — if it wins, anchor is
indeed the culprit; if it doesn't, we need to dig deeper.

Smoke #2 writeup: `context/experiments/002-smoke-tune20-alpha111-live.md`
plus interim 205-task metrics at
`context/experiments/smoke_002_tune20_alpha111_live/metrics_205task_subset.json`.

Also along the way: `BENCHMARK_MODE` env-var hook on the smoke scripts
(default `static`, `=live` to switch); subdir prefix encodes mode so result
trees never collide.

---

## 2026-04-25 — first smoke test on the 228-task tune split — @gaurav

Ran TurnKV (α=1,1,1, γ=0.1, β=0.25, K=0.25, update_every=5) vs plain SnapKV on
228 tasks (= 20% random split of CF_EF_UNIT_SNF, seed 42) under Llama-3.1-8B-Instruct
+ static replay + CoT + global=4096/local=2048. Fanned out 10 H100 shards per
profile in parallel (baseline → `gauravmshah2004`, turnkv → `docmanish2312`).

Headline: **tied at +0.05 pp overall (22.19 → 22.24); final-iter -0.44 pp**.
Recall 32.02 either way. Iters 1-4 are bit-identical, divergence starts at iter 5
which lines up with the TurnFloor `exp(-γ(T-i))` term beginning to bite.

The interesting signal is the **status mix**: 87 runtime_errors became
compile_errors (95→183, +93%) under TurnKV. So the policy is *changing what the
model sees* in a structural way without converting any of those changes into
extra passes. Hypothesis: anchor=1.0 over-protects role-boundary tokens and the
floor scale-down evicts code-body tokens (function signatures / imports /
indentation) that were being kept by plain SnapKV. Worth diagnosing before any
α-sweep.

Plumbing built along the way: deterministic 80/20 split + 10-way interleaved
shard builder, `--task-ids @<path>` JSON-list support in `live_loop.py`,
`output_subdir` param in `modal_app.py` to keep the 10 parallel writers in
race-free per-shard subtrees. Smoke run scripts under
`kvpress/evaluation/benchmarks/convcodeworld/modal_run_smoke_*.sh`. Full writeup
in `context/experiments/001-smoke-tune20-alpha111.md`; raw metrics bundle in
`context/experiments/smoke_001_tune20_alpha111/metrics.json`.

Operational notes (Windows + Modal):
- Cisco AnyConnect (CMU VPN) hijacks WSL2 networking — its adapter has metric 1
  vs WSL's 15, so all WSL outbound dies. Pivoted to running Modal CLI from
  Windows-side Python 3.13 + Git Bash, mirroring `~/.modal.toml`.
- `modal run -d ::run_convcodeworld_live` skips `main()`, so the launcher path
  translation never fires — passed the container path directly.
- Git Bash MSYS rewrites `/root/...` → `C:/Program Files/Git/root/...`; need
  `MSYS_NO_PATHCONV=1` and `MSYS2_ARG_CONV_EXCL="*"` in env.
- Modal CLI on Windows hits a charmap codec on the ✓ glyph; need
  `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`.
- Windows `MAX_PATH=260` blew up `modal volume get` for the long turnkv result
  subdir name; pulled per-shard `predictions.jsonl` into a flat dir under
  `E:\sm\t-flat\` instead.

---

## 2026-04-23 — live_loop.py major expansion, Modal rewrite, flashdecode tracking — @yagneek

Large batch of ConvCodeWorld infrastructure work landed today (all unstaged, branch `liveloop`):

**`live_loop.py` — major expansion (~650 LOC added)**
- Added `benchmark_mode` field (`live` / `static`, with aliases `live_loop` / `static_replay`). `_normalize_benchmark_mode` validates and normalises the string.
- Added `full_kv_cache`, `require_flashdecode`, and `error_on_kv_cache_vram_exhaustion` config fields.
- New VRAM-safety helpers: `infer_device`, `_cache_tensor_devices`, `_assert_cache_on_device`, `_estimate_kv_cache_bytes`, `_assert_kv_cache_fits_available_vram`. These guard against silent OOM when `full_kv_cache=True` keeps the entire KV cache resident.
- Utility helpers: `_is_flash_attention_3`, `_model_uses_flash_attention_3`, `_text_config`, `_format_num_bytes`.
- Imports `flashdecode_used_layers` / `reset_flashdecode_tracking` from `attention_patch.py` to verify FA3 decode path is actually active when `require_flashdecode=True`.
- Default feedback model updated from `gemma-4-E2B-it` → `gemma-3-4b-it` (available without gated access).

**`attention_patch.py` — flashdecode tracking (~180 LOC added)**
- New module-level tracking: `reset_flashdecode_tracking(model)` clears `_kvpress_flashdecode_used` flags; `flashdecode_used_layers(model)` returns sorted list of layer indices that actually used the FA3 decode path.
- `_flashdecode_forward` internal helper wires the `flash_attn_with_kvcache` path and sets the flag on the module.
- Lazy import of `flash_attn_interface.flash_attn_with_kvcache` via `lru_cache` so the module loads cleanly when FA3 is absent.

**`executor.py` — code normalisation helpers (~45 LOC added)**
- `normalize_tokenizer_artifacts(text)`: translates byte-level BPE artefacts (`Ċ`→`\n`, `Ġ`→` `, `ĉ`→`\t`) that appear when the model emits raw tokenizer vocab tokens instead of decoded text.
- `_contains_entry_point(code, entry_point)`: regex check for a named function definition.
- `_longest_compilable_prefix(code, entry_point)`: walks candidate code backwards from the last line to find the longest prefix that (a) compiles and (b) contains the entry-point function. Handles truncated generation gracefully.
- `normalize_candidate_code` and `normalize_tokenizer_artifacts` exported for use in `live_loop.py`.

**`modal_app.py` — full rewrite (~186 LOC added)**
- All constants extracted to module level: `DEFAULT_MODEL`, `DEFAULT_FEEDBACK_MODEL`, `CUDA_BASE_IMAGE`, `MODAL_TORCH_VERSION`, `FLASH_ATTN3_REF`, `TRANSFORMERS_GIT_REF`, `MODAL_EVAL_REQUIREMENTS`.
- FA3 build flags extracted to `FLASH_ATTN3_BUILD_ENV` (disables backward, SM80, split, FP16, FP8 kernels to keep the H100-only wheel small and the Modal layer cacheable).
- `MODAL_EVAL_REQUIREMENTS` tuple drives `_shell_requirements()` → single `uv pip install` invocation; no more inline string concatenation.
- `base_image` build chain now explicit and reproducible.
- Feedback model updated to `gemma-3-4b-it` to match `live_loop.py`.

**`modal_run.sh` — full rewrite**
- Now a proper bash script (`set -euo pipefail`, `cd` to repo root relative to script location).
- Targets `run_convcodeworld_live` entrypoint (not `main`).
- Default run: `no_press` at `compression_ratio=0.0`, `full_kv_cache`, `require_flashdecode`, `error_on_kv_cache_vram_exhaustion` — i.e. a baseline full-cache run to establish upper-bound numbers and verify the FA3 decode path is active.
- `--fraction 0.05` + `--num-eval-examples -1` for a quick smoke sweep.
- Runs detached (`-d`) with `MODAL_HF_SECRET_NAME=hf-secret`.

**`MODAL_HYPERPARAMS.md` — new file**
- Complete reference table for every CLI flag exposed by `modal_app.py::main`, grouped by: dispatch, model/runtime, benchmark sampling, generation/budget, base-press, turn-aware, and Modal infrastructure constants.
- Documents `modal_run.sh` preset args.

Open questions from today:
- `gemma-3-4b-it` vs `gemma-4-E2B-it` for verbal feedback: the 3B is ungated but weaker; need to measure feedback quality difference on a 20-task sample before committing to it for headline runs.
- `_longest_compilable_prefix` falls back to the raw string if nothing compiles — need a test that exercises the entry-point guard on a truncated generation.
- `require_flashdecode` check fires after generation; if FA3 silently falls back mid-run we only find out at the end. Consider checking after the first decode step instead.

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
