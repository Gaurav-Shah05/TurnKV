# 003: Smoke test — Loyalty-only TurnKV (α=0,0,1) vs SnapKV in live loop

- **Date**: 2026-04-25 (dispatched 02:46 EDT, completed 03:36 EDT)
- **Owner**: Gaurav
- **Branch / commit**: `gaurav/harness-fixes` @ `<this commit>`
- **Related decisions**: `decisions/001-multi-turn-harness.md`, `decisions/002-implementation-plan.md`
- **Supersedes**: nothing yet (still in smoke phase)

## Question

After smoke #2 showed α=(1,1,1) tying SnapKV in static and slightly **regressing** in
live (-0.44 pp recall, +12 timeouts), with status-mix evidence pointing at heuristic
over-protection of role-boundary tokens, does turning off the heuristic policies
(TurnFloor + RoleAnchor) and running pure data-driven Loyalty-only (α=(0,0,1)) win?

## Setup

Identical to smoke #2 (live mode, Llama-3.1-8B-Instruct + Gemma-3-4b-it,
228-task tune split, global=4096 / local=2048, max_turns=10, max_new_tokens=1024,
CoT on) **except**:

| param | smoke #2 | smoke #3 |
|---|---|---|
| `--alpha-floor` | 1.0 | **0.0** |
| `--alpha-anchor` | 1.0 | **0.0** |
| `--alpha-loyalty` | 1.0 | 1.0 |
| `--floor-gamma` | 0.1 | 0.1 (unused) |
| `--anchor-beta` | 0.25 | 0.25 (unused) |
| `--loyalty-top-p` | 0.25 | 0.25 |
| `--loyalty-update-every` | 5 | 5 |

10 detached Modal containers on `docmanish2312`, ~50 min wall-clock end-to-end (10-way
parallelism). Log prefix: `smoke_turnkv_snapkv_loyaltyonly_live_20260425_024647_*`.
Output subdir on Modal volume: `turnkv_snapkv_loyaltyonly_live_smoke_20260425_024647/`.

## Results

Headline (live mode, full 228-task tune split, three configs side-by-side):

```
config                          overall   final-iter   mrr   recall    pass
baseline_live                     4.90       34.65   28.23    34.65      79
turnkv_live  α=(1,1,1)            4.81       34.21   27.89    34.21      78
turnkv_live  α=(0,0,1) Loyalty    4.90       34.65   28.22    34.65      79   ← bit-tied with baseline on recall
```

Per-iteration Pass@1 (live, % of tasks not yet passed):

```
iter   B-live   T-a111   T-loyal   L vs B
   1    23.68    23.68     23.68    +0.00
   2     8.62     6.90      8.62    +0.00
   3     3.77     5.56      3.77    +0.00
   4     1.31     0.65      1.31    +0.00
   5     0.66     0.66      0.66    +0.00
   6     0.67     0.00      0.00    -0.67
   7     0.00     0.66      0.67    +0.67
   8     0.00     0.00      0.00    +0.00
   9     0.00     0.00      0.00    +0.00
  10     0.00     0.00      0.00    +0.00
```

Iter 1-5 are **bit-identical** between Loyalty-only and baseline_live. Iter 6 →
iter 7 shifts a single task by one iteration. Net recall identical: 79 / 228 in
both.

Per-task pass divergence vs baseline:
```
both pass:                       77
baseline pass, loyalty fail:      2
baseline fail, loyalty pass:      2
both fail:                      147
```

So Loyalty-only doesn't *literally* solve the same 79 tasks as baseline — it
swaps two pairs of tasks in and out — but the count is identical.

## The actually interesting signal: status mix

```
status                B-live   T-a111   T-loyal   loyal-B
compile_error             96      110         3       -93   ← Loyalty almost eliminates compile errors
pass                      79       78        79         0
runtime_error           1431     1416      1512       +81   ← absorbed into runtime errors
skipped_after_pass       669      659       668        -1
timeout                    5       17        18       +13   ← still get the timeout regression
```

**Loyalty-only drops compile_error from 96 → 3** (a 32× reduction). Those 93 lost
compile_errors don't become passes — they become 81 runtime_errors and 13
timeouts. So the data-driven retention signal IS doing exactly what we'd hoped:
it preserves the structurally-important tokens (imports, function signatures,
indentation, parens) that make code parse. The model is now generating
*syntactically valid* code that just doesn't *work*.

This confirms one half of the smoke #2 hypothesis: the compile_error spike under
α=(1,1,1) was indeed coming from the heuristic policies (RoleAnchor / TurnFloor)
evicting code-body tokens. Loyalty-only doesn't have that failure mode.

But the **timeout regression** persists (+13 vs baseline, basically the same
+12 we saw under α=(1,1,1)). So timeouts aren't from heuristic over-protection —
they're from Loyalty's own behavior. Loyalty's K=0.25 + update_every=5 still
sometimes evicts tokens that force the model into regenerate-from-scratch loops.

## Takeaway

1. **Loyalty-only EXACTLY ties SnapKV on recall (34.65 = 34.65, 79 = 79
   passes).** This is the cleanest result yet — pure data-driven retention
   matches SnapKV's prefill-attention-based eviction at this budget.

2. **The eviction signal is real but isn't translating to wins.** Loyalty
   knows what to keep (compile_error: 96 → 3 is *not* noise) but at
   global=4096 the eviction isn't binding hard enough to differentiate. The
   tokens Loyalty preserves are tokens SnapKV would also have kept; the
   tokens Loyalty evicts produce different runtime errors than the tokens
   SnapKV evicts, but the *count* of passing tasks is the same.

3. **Doesn't kill the paper's three-policy thesis, but does narrow it.**
   The triad-as-currently-written (positional floor + positional anchor +
   data-driven loyalty) doesn't beat plain SnapKV at α=(1,1,1) or α=(0,0,1)
   on Llama-3.1-8B + ConvCodeWorld at budget=4096. We have two interpretations:
   (a) wrong α — needs intermediate weighting like (0.5, 0.25, 1.0); (b) wrong
   *regime* — at budget=4096 the eviction is too gentle to reveal what TurnKV
   is doing differently from SnapKV. Should test (b) with a tighter budget
   before claiming the policy doesn't work.

4. **No decision change to ADRs.** ADR 001/002 still stand; we still have no
   winning α.

## What I'd try next

Three orthogonal directions, ordered by EV:

1. **Tighter global budget (smoke #4)**: drop to global=1024 or 2048 and re-run
   α=(0,0,1) and SnapKV on the same 228-task split. Hypothesis: at budget=4096
   neither policy is evicting hard enough to differentiate. At budget=1024
   we might actually see Loyalty's data-driven signal beat SnapKV's positional
   one (or definitively lose, which is also informative).

2. **Sharper Loyalty top-p (smoke #5)**: K=0.10 instead of 0.25. K=0.25 keeps
   the top 25% of tokens with a +1.0 score boost — that's a lot of tokens at
   budget=4096. K=0.10 would be a tighter "loyalty" definition.

3. **Mixed α (smoke #6)**: (0.5, 0.0, 1.0) — half-floor + full-loyalty, no
   anchor. Tests whether the "preserve recent turn content" signal from
   TurnFloor adds anything to data-driven Loyalty.

The paper's thesis lives or dies on whether *some* TurnKV configuration beats
SnapKV. Right now we have two configs that tie or slightly regress. A budget
sweep is the highest-EV next step because it tests whether we're even in a
regime where the eviction matters.

## Live-loop status (per the original "is live better for TurnKV?" question)

After three smokes:

- α=(1,1,1) static: +0.05 pp overall vs baseline (tied)
- α=(1,1,1) live:   −0.09 pp overall, −0.44 pp recall vs baseline (slight regression)
- α=(0,0,1) live:   ±0.00 pp on every metric vs baseline (exact tie)

**Live mode does not currently unlock TurnKV vs SnapKV.** Live gives both presses
a +2.2 to +2.6 pp recall lift over static, but the *delta* between presses is
indistinguishable from zero. The "is live better for TurnKV?" question now has
a clean answer: no, not at any α we've tested. The harness×model interaction
(verbal-feedback iteration on the model's own code) is doing the lift; the
press is irrelevant at this budget.
