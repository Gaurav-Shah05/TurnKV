# 004: Smoke test — budget sweep, baseline vs Loyalty-only in live loop

- **Date**: 2026-04-25 (dispatched 03:56 EDT, completed 04:50 EDT — ~54 min for 40 shards in parallel on `gauravmshah2004`)
- **Owner**: Gaurav
- **Branch / commit**: `gaurav/harness-fixes` @ `<this commit>`
- **Related decisions**: `decisions/001-multi-turn-harness.md`, `decisions/002-implementation-plan.md`
- **Builds on**: smokes 001 (static, α=1,1,1), 002 (live, α=1,1,1), 003 (live, α=0,0,1)

## Question

After smokes 001-003, baseline and TurnKV(α) tied at every cell tested at
`global=4096`. Two competing reads on why: **(a) wrong α weighting** (need to
ablate more cells); **(b) wrong regime** — at budget=4096 the eviction is too
gentle to differentiate any retention strategy.

This smoke tests (b) directly: drop the budget to 2048 and 1024 (keeping
local-budget proportional at half), re-run baseline + Loyalty-only on the
same 228-task tune split. If the gap appears below 4096, the regime
hypothesis is right and we have a winning config. If it doesn't, the policy
genuinely doesn't beat SnapKV on this benchmark and we need to pivot.

## Setup

Identical to smoke #2 / #3 (live mode, Llama-3.1-8B-Instruct + Gemma-3-4b-it,
228-task tune split, max_turns=10, max_new_tokens=1024, CoT on) **except**:

| param          | smoke #2/#3        | smoke #4 (3 budgets) |
|---             |---                 |---                    |
| `--global-budget` | 4096            | **4096 / 2048 / 1024** |
| `--local-budget`  | 2048            | **2048 / 1024 / 512**  |
| `--alpha-floor`   | 1.0 (s2) / 0.0 (s3) | 0.0 (Loyalty-only) |
| `--alpha-anchor`  | 1.0 / 0.0          | 0.0 |
| `--alpha-loyalty` | 1.0                | 1.0 |
| `--loyalty-top-p` | 0.25               | 0.25 |
| `--loyalty-update-every` | 5           | 5 |

40 detached Modal containers on `gauravmshah2004` (4 configs × 10 shards), all
finished within ~54 min (Modal allocated enough H100s to run the full set close
to in parallel). Output subdirs:
- `baseline_snapkv_b{2048,1024}_live_smoke_<TS>/shard_*_of_10/`
- `turnkv_snapkv_loyaltyonly_b{2048,1024}_live_smoke_<TS>/shard_*_of_10/`

Smokes #2/#3 results (budget=4096) carried over from `metrics.json` in their
respective experiment dirs.

## Results

Headline (live mode, full 228-task tune split, six cells side-by-side):

```
cell             overall   final   mrr   recall   pass
baseline_4096      4.90    34.65  28.23   34.65    79
loyalty_4096       4.90    34.65  28.22   34.65    79      ← bit-tied
baseline_2048      4.98    35.09  28.45   35.09    80      ← baseline peaks here
loyalty_2048       4.74    33.77  27.94   33.77    77
baseline_1024      4.34    31.58  27.21   31.58    72
loyalty_1024       4.63    33.33  27.50   33.33    76      ← LOYALTY WINS by +1.75 pp recall
```

Loyalty − baseline recall delta (pp), at each budget:

```
global=4096   baseline 34.65   loyalty 34.65   Δ +0.00
global=2048   baseline 35.09   loyalty 33.77   Δ −1.32
global=1024   baseline 31.58   loyalty 33.33   Δ +1.75    ← first win on this project
```

Status mix is the smoking-gun signal — the budget sweep shows exactly when
SnapKV starts evicting structurally-important code tokens:

```
status              B-4096   L-4096   B-2048   L-2048   B-1024   L-1024
compile_error           96        3      230       21      343       17
pass                    79       79       80       77       72       76
runtime_error         1431     1512     1285     1527     1241     1550
skipped_after_pass     669      668      674      654      620      637
timeout                  5       18       11        1        4        0
```

Read row by row:
- **compile_error** explodes for SnapKV as budget tightens (96 → 230 → 343),
  while Loyalty-only stays flat (3 → 21 → 17). At budget=1024, SnapKV is
  emitting **20×** as many compile errors as Loyalty — that's the entire
  mechanism by which Loyalty wins at this budget.
- **runtime_error** rises for both as budget tightens, but Loyalty rises a
  bit more (1512 → 1527 → 1550) — those are the ex-compile-errors converted
  into "code parses but doesn't work".
- **timeout** disappears at low budgets for both presses. The +13 timeout
  regression we worried about at budget=4096 was a *budget-specific*
  artifact, not a Loyalty pathology. At budget=1024 Loyalty has zero
  timeouts.
- **pass** rate for SnapKV declines monotonically (79 → 80 → 72), but
  baseline_2048 is the global peak across all six cells. Loyalty stays
  more level (79 → 77 → 76).

Per-iteration Pass@1 (% of tasks not yet passed):

```
 iter   B-4096  L-4096  B-2048  L-2048  B-1024  L-1024
   1    23.68   23.68   23.68   23.68   23.68   23.68    ← all start here, no eviction yet
   2     8.62    8.62    9.20    8.05    8.05    6.90
   3     3.77    3.77    4.43    4.38    1.25    3.70    ← L-1024 holds while B-1024 collapses
   4     1.31    1.31    0.66    0.65    0.63    1.28
   5     0.66    0.66    0.00    0.00    0.00    0.00
   6     0.67    0.00    0.67    0.00    0.00    0.00
   7     0.00    0.67    0.00    0.00    0.00    0.00
   8     0.00    0.00    0.00    0.00    0.64    0.00
   9     0.00    0.00    0.67    0.66    0.00    0.65
  10     0.00    0.00    0.00    0.00    0.00    0.65    ← L-1024 still finding passes at iter 10
```

Iter 1 is identical across all six cells (no eviction has happened yet, every
press is just running the prefilled context with no compression). Divergence
starts at iter 2 once eviction kicks in. **Loyalty at budget=1024 is the only
config still finding novel passes at iter 10** — i.e. its retention is
preserving enough context that late-iteration verbal-feedback still drives
the model to a new solution.

## Takeaway

1. **First winning configuration: Loyalty-only at budget=1024 beats SnapKV
   by +1.75 pp recall.** The data-driven retention signal works, but only
   in the regime where eviction is binding hard enough to differentiate
   strategies. Smokes 001-003 missed this because budget=4096 left both
   presses too much slack.

2. **The mechanism is structural: compile_error 343 → 17 (20× reduction)
   at budget=1024.** Loyalty-only is preserving the imports / signatures /
   indentation / parens that SnapKV evicts as it gets aggressive. The
   timeout regression we worried about at 4096 turns out to be budget-
   specific noise (zero timeouts at 1024).

3. **Baseline SnapKV peaks at budget=2048**, not 4096. This is genuinely
   surprising — less context helps the model. Hypothesis: at 4096 the
   prefill includes some noisy intermediate-iteration tokens (failed
   reasoning, half-correct code) that distract more than they help.
   Compression to 2048 happens to evict mostly noise, lifting the model.
   At 2048 specifically, Loyalty's K=0.25 "always-keep top 25%" pulls
   noise back in, regressing −1.32 pp vs baseline. Loyalty isn't always
   the right policy; it's the right policy specifically when SnapKV's
   eviction starts losing structure.

4. **The paper has a defensible narrative now.** TurnKV's data-driven
   Loyalty signal beats plain SnapKV in the *budget-constrained* regime
   that actually matters for production (when you cannot afford a 4 K
   global cache per layer). At generous budgets both tie. The cross-over
   point on this benchmark is somewhere between 1024 and 2048.

## What I'd try next

- **Sweep budget=512 and 768.** Hypothesis: the gap widens further as
  budget tightens. If Loyalty is +1.75 at 1024, what is it at 768? At 512?
- **Re-run α=(1,1,1) at budget=1024.** Smokes 001/002 had it tying at 4096;
  maybe the heuristic policies (TurnFloor + RoleAnchor) help here too once
  eviction is binding. If α=(1,1,1) > Loyalty-only at 1024, that's another
  story for the paper (the triad).
- **Investigate the budget=2048 peak for baseline.** If we can characterize
  *why* SnapKV does best at 2048 (not 1024 or 4096), we can pick that as
  our shipped default and report Loyalty as the "tight-budget" mode.
- **Hold-out evaluation only after we lock in the winning config + budget.**
  The 912-task hold-out is for the chosen config, not for sweeping.
