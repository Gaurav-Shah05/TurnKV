# 001: Smoke test — TurnKV (α=1,1,1) vs SnapKV on the 228-task tune split

- **Date**: 2026-04-24 (dispatched 22:27 EDT, completed ~23:00 EDT)
- **Owner**: Gaurav
- **Branch / commit**: `gaurav/harness-fixes` @ `<this commit>`
- **Related decisions**: `decisions/001-multi-turn-harness.md`, `decisions/002-implementation-plan.md`

## Question

Does the cherry-picked TurnKV configuration (α=(1,1,1), γ=0.1, β=0.25, K=0.25,
update_every=5) wrapping SnapKV beat plain SnapKV on the 228-task tune split,
when both run static replay with Llama-3.1-8B-Instruct + CoT and an identical
budget (global=4096 / local=2048)?

## Setup

| | |
|---|---|
| Model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Feedback model (live verbal feedback path) | `google/gemma-3-4b-it` |
| Attention | `flash_attention_3` (FA3 flashdecode required) |
| Benchmark mode | static replay (matches ADR 001 §8) |
| Feedback config | `CF_EF_UNIT_SNF` (compile + exec + unit + simulated-novice verbal) |
| Tasks | 228 (= 20% of 1140 CF_EF_UNIT_SNF task IDs, `random.Random(42).sample` over sorted-lex universe; no stratification) |
| Iterations / task | 10 |
| Compression | global=4096, local=2048, `--compression-ratio 0.0`, `--max-new-tokens 1024`, `--cot` |
| TurnKV α | floor=1, anchor=1, loyalty=1 |
| TurnKV tunables | γ=0.1, β=0.25, K=0.25, update_every=5 (decode-step subsampling) |
| Baseline | `snapkv` press (no TurnKV wrapper, same SnapKV under the hood) |
| TurnKV | `turnkv_snapkv` press (wraps SnapKV with the three policies) |
| Hardware | 10 × H100 per profile (1 GPU per shard, 10 shards × 23/22 tasks each) |
| Compute split | baseline → Modal profile `gauravmshah2004`; turnkv → `docmanish2312` |
| Result tree | volume `kvpress-convcodeworld-results`, see `metrics.json` for both subdirs |

## Design choices (and why)

1. **Static replay, not live loop.** ADR 001 §8 specifies static for compression-ratio
   sweeps because every iteration sees the same teacher-forced "previous_code" history,
   making Pass@1 deltas attributable to the press rather than to compounding generation
   noise. We may revisit live loop once we have a winning α (see "Live loop?" below).

2. **20% / 80% random split, seed=42.** Three options were on the table: (a)
   stratify by `canonical_solution` line-count quartile, (b) stratify by `libs` domain,
   (c) pure random. We picked (c) for simplicity — n=228 from 1140 is large enough
   that the law of large numbers gives reasonable expected coverage of every
   subpopulation, and (a)/(b) added cognitive load with no proven payoff yet. The
   "random stratification" descriptor in chat just meant "random sampling, no special
   stratification". Universe is sorted lexicographically before sampling so the
   split is stable regardless of dataset row order.

3. **Cherry-picked α=(1,1,1).** Equal weighting of all three policies. Smoke purpose
   is to see if turn-aware eviction does *anything* on top of SnapKV; not to win.
   `loyalty_update_every=5` is the decode subsampling default we landed on in
   commit `44c2699` to keep wall-clock parity with plain SnapKV.

4. **10-way sharding, interleaved by task index.** Each Modal container gets ~23
   tasks across 1 GPU. Interleaved (idx % num_shards) rather than contiguous chunks
   so each shard sees a representative cross-section of BigCodeBench task IDs and
   per-shard wall-clock is roughly balanced. 228 = 8×23 + 2×22, so two shards have
   one fewer task. Per-shard `output_subdir` keeps the result tree race-free across
   the 10 parallel writers (Modal volume's eventual consistency would otherwise have
   the race in `_results_dir`'s `while exists()` loop fire on duplicate paths).

5. **Two Modal profiles, in parallel.** Baseline on `gauravmshah2004`, TurnKV on
   `docmanish2312`. Different profiles have separate Modal workspaces (and hence
   separate volumes), so the two configs don't share infrastructure — clean
   isolation, no contention. Cost: ~10 H100s on each side simultaneously.

6. **Container-side `--task-ids` path.** `modal_app.py::main` has a translation
   helper `_translate_task_ids()` that rewrites a launcher `@<local_path>` to
   `@/root/kvpress/<rel>`, but `modal run -d ::run_convcodeworld_live` skips
   `main()` and dispatches the function directly — translation never fires. Fix:
   the smoke shell scripts pass the *container path* directly
   (`@/root/kvpress/.../shards/<file>.json`). The shard JSONs ship inside the image
   via `image.add_local_dir(REPO_ROOT, "/root/kvpress")`, so the path resolves at
   container start. The `_translate_task_ids` helper is kept for the
   `modal run ::main` path (which `--detach-remote=True` users would use).

## Results

Full metrics: `smoke_001_tune20_alpha111/metrics.json`. Headline:

```
                 baseline   turnkv    delta
overall pass     22.19      22.24    +0.05 pp
final-iter pass  19.30      18.86    -0.44 pp
mrr              27.34      27.32    -0.02
recall           32.02      32.02     0.00
```

Per-iteration Pass@1 (% of 228 tasks):

```
iter   baseline   turnkv    delta
  1     23.68     23.68    +0.00
  2     23.68     23.68    +0.00
  3     22.81     22.81    +0.00
  4     22.81     22.81    +0.00
  5     23.68     23.25    -0.43
  6     22.81     22.81    +0.00
  7     21.93     22.37    +0.44
  8     20.61     20.61    +0.00
  9     20.61     21.49    +0.88
 10     19.30     18.86    -0.44
```

Status mix (across 2,280 rows = 228 tasks × 10 iter):

```
                  baseline   turnkv     delta
pass               506        507       +1
runtime_error    1,676      1,589      -87
compile_error       95        183      +88
timeout              3          1       -2
```

## Takeaway

1. **TurnKV at α=(1,1,1) is statistically tied with plain SnapKV** on the 228-task
   tune split (+0.05 pp overall, well within sampling noise for n=228). Recall
   is identical at 32.02; mrr is identical at ~27.3. So the policy is neither
   helping nor hurting on the headline.

2. **Iters 1-4 are bit-identical, iters 5+ diverge.** The TurnFloor `exp(-γ(T-i))`
   term doesn't bite until late turns and the cache hits the global budget; until
   then both presses' eviction sets coincide. Confirms the floor is acting where
   designed.

3. **The status mix is the actually interesting signal: 87 runtime_errors became
   compile_errors.** TurnKV is changing *which* tokens the model sees in a way that
   shifts failures from "code runs but wrong answer" to "code doesn't even parse".
   Hypothesis: alpha_anchor=1.0 with anchor_beta=0.25 protects role-boundary tokens
   so aggressively that some critical *code* tokens (function signatures, imports,
   indentation) get evicted instead. Worth diagnosing before further sweeps.

4. **No decision change to ADRs.** ADR 001 / 002 still stand. We need a winning α
   before we promote anything.

## Next steps

- **Diagnose the compile_error spike.** Pull a handful of tasks where baseline
  passes iter N but turnkv compile-errors iter N, diff the cached token sets at
  the boundary. Most likely culprit: anchor over-protecting role markers at the
  cost of code-body tokens.
- **α-sweep on the 228-task tune split** once the diagnosis suggests a saner
  prior. Probably try (1,0,0), (0,1,0), (0,0,1) ablations first to see which
  policy is doing what before recombining.
- **Hold-out evaluation only after we have a winning α** — the 912-task hold-out
  is for the chosen config, not for sweeping.

## Live-loop?

> "do you think we get better performance on liveloop?"

**Honest take: probably not without α-tuning first.** Two competing forces:

- **Likely worse**: in static replay every iter sees the same teacher-forced prior
  code, so eviction errors don't compound — iter 5 doesn't *depend* on what we
  evicted in iter 4. In live loop, iter 5 sees iter 4's actual model output, so
  any compile_error the eviction caused at iter 4 becomes the input context for
  iter 5. The +88 compile_error spike we just saw would compound through the
  trajectory rather than being independent across iters.
- **Possibly better**: the Loyalty signal is designed against cumulative attention
  patterns from real generation. Static replay's "previous_code" is a different
  attempted solution per iter (not coherent across iters), so Loyalty has thin
  signal to act on. Live loop gives Loyalty actual cross-turn dependency to
  measure, which is the regime the policy was designed for.

Net: live loop will likely amplify whatever's wrong with the current α. With
α=(1,1,1) tied here and the compile_error symptom suggesting anchor over-protection,
I'd expect live loop to make turnkv *worse* before a tuned α makes it better. Plan:
diagnose compile_error → ablate to find the helpful α component → only then run
live loop on the chosen config.
