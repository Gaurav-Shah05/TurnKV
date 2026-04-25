# 002: Smoke test #2 — TurnKV (α=1,1,1) vs SnapKV in **live loop** on the 228-task tune split

- **Date**: 2026-04-25 (dispatched 01:42 EDT, completed ~02:30 EDT)
- **Owner**: Gaurav
- **Branch / commit**: `gaurav/harness-fixes` @ `<this commit>`
- **Related decisions**: `decisions/001-multi-turn-harness.md`, `decisions/002-implementation-plan.md`
- **Companion experiment**: `001-smoke-tune20-alpha111.md` (same configuration, static-replay mode)

## Question

Does live-loop mode change the picture from smoke #1, where TurnKV (α=1,1,1)
tied plain SnapKV at +0.05 pp under static replay? Two competing forces in
play: (a) Loyalty's cumulative attention signal has actual cross-turn
dependency to measure when the model sees its own prior generation, possibly
helping; (b) the +88 compile_error spike we saw in static could compound when
iter N's broken code becomes iter N+1's input, possibly hurting.

## Setup

Identical to smoke #1, only `--benchmark-mode live` flips:

| | |
|---|---|
| Model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Feedback model (live verbal feedback path) | `google/gemma-3-4b-it` |
| Attention | `flash_attention_3` (FA3 flashdecode required) |
| Benchmark mode | **`live`** (model generates each iter, executor runs the code, Gemma writes verbal feedback) |
| Feedback config | `CF_EF_UNIT_SNF` |
| Tasks | 228 (= same `tune_20pct_seed42.json` as smoke #1) |
| Iterations / task | up to 10 (early-stop on first pass, `early_stop_on_pass=True`) |
| Compression | global=4096, local=2048, `--compression-ratio 0.0`, `--max-new-tokens 1024`, `--cot` |
| TurnKV α | floor=1, anchor=1, loyalty=1 (γ=0.1, β=0.25, K=0.25, update_every=5) — same as smoke #1 |
| Hardware | 10 × H100 per profile, baseline → `gauravmshah2004`, turnkv → `docmanish2312` |
| Result tree | volume `kvpress-convcodeworld-results`, `{baseline,turnkv}_snapkv_live_smoke_20260425_*/shard_*_of_10/live__.../predictions.jsonl` |

### Parameterising the smoke scripts

`modal_run_smoke_*.sh` now read `BENCHMARK_MODE` from env (default `static`).
Live runs are `BENCHMARK_MODE=live bash modal_run_smoke_*.sh`. The output
subdir prefix encodes the mode (`baseline_snapkv_live_smoke_*` vs
`baseline_snapkv_static_smoke_*`) so result trees never collide.

### Operational hiccups

- **Baseline shard 4 OOM'd at model load** on first dispatch — Modal H100
  contention manifested as `torch.AcceleratorError: CUDA error: out of memory`
  inside `caching_allocator_warmup` before any task ran. Re-dispatched as a
  one-off; numbers below are the **205-task intersection** of the 9 surviving
  baseline shards × the matching 10 turnkv shards. Full 228-task numbers will
  be regenerated when shard-4-retry lands.
- Live runs took ~46 min/shard (vs ~25 min/shard for static), so wall-clock
  for the full fan-out was ~48 min thanks to 10-way parallelism.

## Results

Headline (205-task common subset, *same task universe in all four cells*):

```
config                 overall   final-iter    mrr   recall
baseline_static          22.68       19.02   28.52    32.68
turnkv_static            22.73       18.54   28.50    32.68
baseline_live             5.30       36.59   29.81    36.59
turnkv_live               5.29       36.59   29.56    36.59
```

> Why does `overall` collapse from ~22.7 to ~5.3 going static→live? Because
> `early_stop_on_pass=True` marks subsequent iters as `metric_excluded` once
> a task passes — `overall` is the pooled mean over all *kept* rows, so the
> denominator at iter ≥ 2 only contains tasks that haven't passed yet. In
> static replay we never early-stop (the per-iter label is the dataset's
> reference label; there's no model generation to "succeed and skip"). So
> static `overall` ≈ avg per-iter rate, live `overall` is biased down by the
> shrinking denominator. **`recall` is the apples-to-apples cross-mode
> headline** (% of tasks where any iter passes).

Static→live recall: **+3.91 pp on both presses (32.68 → 36.59)**. So the live
harness genuinely helps the model solve harder tasks (it can iterate on
*its own* code with real feedback, rather than being teacher-forced into a
particular reference trajectory). But the lift is identical for SnapKV and
TurnKV — the press isn't the lever.

Per-iteration Pass@1 in live mode (% of tasks not yet passed):

```
iter   B-live   T-live   delta
   1    24.88   24.88   +0.00
   2     9.74    7.79   -1.95
   3     4.32    6.34   +2.02
   4     0.75    0.75   +0.00
   5     0.76    0.76   +0.00
   6     0.76    0.00   -0.76
   7     0.00    0.76   +0.76
   8     0.00    0.00   +0.00
   9     0.00    0.00   +0.00
  10     0.00    0.00   +0.00
```

Total tasks passing at SOME iter ends up identical at 75/205 (36.59%) — turnkv
just shifts a couple of late-pass tasks from iter 2 → iter 3 (and vice-versa).

Status mix (live, before `metric_excluded` filter):

```
status                  baseline   turnkv    delta
  compile_error              83       85       +2
  pass                       75       75        0
  runtime_error            1252     1241      -11
  skipped_after_pass        636      632       -4
  timeout                     4       17      +13
```

Live mode mostly produces `runtime_error` (75% of pre-stop iters) — the
generated code parses fine but doesn't pass the test cases. The
`compile_error` story we worried about under static (+88 at iter 10) does **not**
show up in live: only +2 compile_errors total. **Timeouts** are
3.25× higher under turnkv (+13), which is consistent with eviction
occasionally removing import statements or function-signature tokens, leading
the model to regenerate the function from scratch with extra retries that hit
the 30-s executor timeout.

## Takeaway

1. **Live loop does not unlock TurnKV.** At α=(1,1,1), TurnKV and SnapKV are
   bit-tied at 36.59 recall, identical iter-1 (24.88), and offsetting moves
   between iter 2 and iter 3. Mrr is essentially the same. So my prediction
   from smoke #1 — that live loop wouldn't make turnkv look better against
   baseline without α tuning — was correct on the headline. (My specific
   compile_error-compounding hypothesis was *wrong*: live mode produces
   barely any compile_errors at all because the model is generating fresh
   code each iter, not amending a teacher-forced trajectory.)

2. **Live mode beats static on recall by ~4 pp (32.68 → 36.59) for both
   presses.** That delta is identical across SnapKV and TurnKV, so it's the
   harness×model interaction (verbal-feedback-driven iteration on the model's
   own code) doing the work, not the eviction policy. This matches the
   intuition that ConvCodeWorld's static replay is a harder evaluation regime
   for the *model* (it has no agency over the code it's "fixing") — but
   doesn't tell us anything new about the press.

3. **Timeout count tripled under TurnKV (4 → 17, +13).** This is the new
   structural signal in live: tasks where TurnKV evictions force the model
   to regenerate code that ends up looping or otherwise hitting the 30-s
   executor cap. Same root cause we suspected from the static `compile_error`
   spike — the press is removing tokens whose absence forces the model to
   regenerate from scratch.

4. **No decision change to ADRs.** ADR 001/002 still stand; we still need a
   winning α before claiming anything.

## Implications for the project

- **The case for ablation is stronger now**, not weaker. Both modes show the
  policy does *something* structural (status-mix shifts), but at α=(1,1,1)
  those changes don't convert to better Pass@1. We need to identify which
  individual policy is helping vs hurting before we can trust a combined α.
- **Live loop is the right primary mode for the per-turn-curve story**
  (better recall and a real per-turn signal), but we should keep static
  available for fast α sweeps because static gets clean per-iter numbers
  without the early-stop denominator complication.
- **Either eviction signal is genuinely useful, or none is** — the current
  α=(1,1,1) choice doesn't let us distinguish. Next: ablations.

## Next steps

- Wait for shard-4 retry → regenerate the full 228-task version of this
  bundle (likely <0.5 pp difference from the 205-task subset numbers).
- α-ablation on the 228-task tune split: (1,0,0), (0,1,0), (0,0,1), all under
  the same live-mode harness. This is the single most informative thing to
  run next.
- Investigate the 13 extra timeouts under turnkv — pull the task IDs that
  hit timeout under turnkv but not baseline, look at the cache state at the
  iteration where turnkv first diverges.
