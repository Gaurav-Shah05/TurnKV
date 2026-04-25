# TurnKV experiment plan — from origin to final report

This is the running roadmap for the TurnKV class project. It records what we
*originally* set out to do (per the proposal + ADR 001/002), what we've done
so far, and the planned sequence forward up to the 2026-05-03 final report.
Updated when the next phase changes.

- **Last updated**: 2026-04-25 (after smoke #4)
- **Owner**: Gaurav
- **Team**: Gaurav, Pradyut, Yagneek
- **Final presentation**: 2026-04-30 (5 days)
- **Final report**: 2026-05-03 (8 days)

---

## Origin: the proposal-era plan

Ship three turn-aware Press classes for kvpress and demonstrate that they
beat SnapKV (the strongest existing single-turn baseline already in kvpress)
on multi-turn benchmarks where existing presses are blind to turn structure.

The three policies:

| # | Press                  | Signal                                                      | Hypothesis |
|---|---                     |---                                                          |---         |
| 1 | **TurnFloor**          | per-turn minimum budget with `exp(-γ(T-i))` decay           | Early turns shouldn't be fully evicted; recent turns should be near-fully retained. |
| 2 | **RoleBoundaryAnchor** | +β score boost on `w` tokens around user/assistant boundaries | Intent-bearing tokens at role edges deserve protection. |
| 3 | **LoyaltyPress**       | per-position attention-score accumulator across turns       | Tokens that downstream turns actually use are the ones to keep — pure data-driven retention. |

Composed via `TurnAwareGlobalPress`: `score = α_floor·f_floor + α_anchor·f_anchor + α_loyalty·f_loyalty`, then SnapKV's natural top-K eviction over the residual budget. Per-turn budget cap derived from `α`s and γ.

Benchmark plan from ADR 001 (locked 2026-04-19):
- **ConvCodeWorld / ConvCodeBench** — primary coding, per-turn accuracy curve. 1140 BigCodeBench tasks × 5 feedback configs × 10-iter trajectories.
- **LongMemEval_S** — primary conversational, single-probe on ~103 K-token compressed history. Direct head-to-head with EpiCache.
- **TopiOCQA** — secondary, validates the Loyalty topic-shift signal only.

Original baseline assumption: **TurnKV with α=(1,1,1) cherry-pick beats plain SnapKV out of the box**. (This turned out to be wrong; see Phase 1.)

---

## Where we are right now

- **Phase 0 (Week-1 implementation, DONE)**: All three Press classes shipped, registered, tested. `TurnAwareGlobalPress` composer, decode subsampling for LoyaltyPress, full pytest suite green. Static replay + live loop both wired through `live_loop.py` + Modal.
- **Phase 1 (Smokes 001-004, DONE)**: Established that the (1,1,1) cherry-pick *ties* SnapKV at generous budget but Loyalty-only at a tight budget *beats* it. Detailed below.
- **Phase 2 (Characterize the win, NEXT)**: tighter budget sweep + α-ablation at the winning budget. Locks in the configuration we ship.
- **Phase 3 (Hold-out + external)**: 912-task hold-out with the locked config; head-to-head vs EpiCache.
- **Phase 4 (Secondary benchmarks, stretch)**: LongMemEval / TopiOCQA. Compete-for-time against writing.
- **Phase 5 (Writeup + presentation)**: Paper draft, presentation, final report.

---

## Phase 1: smokes (DONE — 2026-04-24 / 2026-04-25)

| # | Mode    | α            | Budget (g/l) | Key result | Doc |
|---|---      |---           |---           |---         |---  |
| 001 | static  | (1,1,1)    | 4096 / 2048  | Tied, +0.05 pp overall. +88 compile_errors (status mix shift). | `001-smoke-tune20-alpha111.md` |
| 002 | live    | (1,1,1)    | 4096 / 2048  | Tied/slightly worse, −0.44 pp recall. +12 timeouts. Compile spike muted (+14). | `002-smoke-tune20-alpha111-live.md` |
| 003 | live    | (0,0,1)    | 4096 / 2048  | Exact tie on recall (34.65 = 34.65). Compile_error dropped 96 → 3 (32×). Timeouts persist (+13). | `003-smoke-tune20-loyalty-only-live.md` |
| 004 | live    | (0,0,1)    | sweep 4096/2048/1024 | **+1.75 pp recall at budget=1024**. Baseline peaks at 2048. Compile-error gap is the mechanism. | `004-smoke-tune20-budget-sweep-live.md` |

**Locked findings going into Phase 2**:
- Loyalty's data-driven retention does what we hoped — preserves structurally-important tokens (imports, signatures, indentation) under eviction pressure.
- The win lives **specifically** in the budget-constrained regime. At generous budgets both presses tie. The crossover on this benchmark is between 1024 and 2048.
- Heuristic policies (TurnFloor + RoleAnchor) at α=(1,1,1) introduce structural failures (compile_error and timeout regressions) at budget=4096 — needs revisit at lower budgets, where they may help.
- All experiments use the same 228-task tune split (`splits/tune_20pct_seed42.json`). The 912-task hold-out is reserved.

---

## Phase 2: characterize the win (planned — 2026-04-26)

Two parallel sweeps on the 228-task tune split, both live mode, on `gauravmshah2004` workspace:

### 2a. Tighter budget sweep — extend to 512 and 768

Extends smoke #4 down. Tests whether the +1.75 pp gap at budget=1024 grows monotonically as budget tightens, or whether it has a single sweet spot.

| Cell | Press | α | Budget (g/l) |
|---   |---    |---|---           |
| Aa   | snapkv | — | 768 / 384 |
| Ab   | snapkv | — | 512 / 256 |
| Ba   | turnkv (Loyalty) | (0,0,1) | 768 / 384 |
| Bb   | turnkv (Loyalty) | (0,0,1) | 512 / 256 |

40 shards (4 cells × 10 shards), one fan-out per workspace, ~50 min wall.

**Decision logic**:
- If gap grows monotonically → use budget=512 as canonical "tight-budget" config; report whole curve.
- If gap is unimodal (peak at 1024) → report 1024 as canonical; note non-monotonic behavior.
- If gap collapses below 1024 (model degrades too much for any policy to help) → report 1024 as the "sweet spot" with a note on the lower bound.

### 2b. α-ablation at budget=1024 — does the triad add anything?

Loyalty alone beats SnapKV at budget=1024 by +1.75. Does adding the heuristic policies (Floor, Anchor) make it *better*, or do they hurt as they did at budget=4096?

| Cell | α              | Notes |
|---   |---             |---    |
| C1   | (0,0,1)        | already done in smoke #4 |
| C2   | (1,0,1)        | floor + loyalty, no anchor |
| C3   | (0,1,1)        | anchor + loyalty, no floor |
| C4   | (1,1,1)        | full triad — re-test at the tight budget |
| C5   | (1,0,0)        | floor only — diagnostic |
| C6   | (0,1,0)        | anchor only — diagnostic |

60 shards (6 cells × 10), distribute across both `gauravmshah2004` and `docmanish2312`, ~50 min wall if all parallel.

**Decision logic**:
- Best cell wins. Paper claim becomes "α=(*) at budget=1024 beats SnapKV by Δ pp recall on ConvCodeWorld."
- If C4=(1,1,1) tops the table, the paper's three-policy thesis is fully supported.
- If C1=(0,0,1) wins, the paper claim is "the data-driven Loyalty signal is what matters; positional heuristics don't add value at the budget where compression bites."
- Either is publishable, but the second is a different (smaller) story.

### 2c. Hyperparameter sensitivity at the chosen (α*, budget*)

Once Phase 2a + 2b lock in the winner, sweep the inner hyperparameters one-at-a-time:

| Param                    | Default | Sweep |
|---                       |---      |---    |
| `loyalty_top_p` (K)      | 0.25    | 0.10, 0.25, 0.50 |
| `loyalty_update_every`   | 5       | 1, 5, 10 (does decode subsampling cost recall?) |
| `floor_gamma` (γ)        | 0.1     | 0.05, 0.1, 0.2 (only if floor in winning α) |
| `anchor_beta` (β)        | 0.25    | 0.1, 0.25, 0.5 (only if anchor in winning α) |

Cheap: each sweep is 1-2 cells × 10 shards. Drop any param that doesn't move the needle ≥0.5 pp.

### Phase 2 deliverable

A single experiment doc (smoke #5 / 6) reporting the chosen `(α*, budget*, K*, update_every*, γ*, β*)` with full decision trail. Targets locking in the canonical TurnKV config by EOD 2026-04-26.

---

## Phase 3: hold-out evaluation + external comparison (planned — 2026-04-27)

### 3a. Hold-out (912 tasks)

The 912-task hold-out (`splits/holdout_80pct_seed42.json`) has been frozen since 2026-04-24 and never seen during sweeping. Run **only** the chosen winning config + the matching baseline (plain SnapKV at the same budget) on it. This is the headline number for the paper.

| Cell | Press | α* | Budget* | Tasks |
|---   |---    |---|---      |---    |
| H1   | snapkv | — | budget* | 912 |
| H2   | turnkv | α* | budget* | 912 |

20 shards (2 cells × 10 each), ~50 min if 20 H100s parallel; ~100 min on quota=10. **Numbers reported in the paper**: H1 vs H2 recall delta, per-iter curve, status mix.

### 3b. EpiCache external baseline

EpiCache (Apple, NeurIPS 2025) is the only existing multi-turn-aware press. Cloned separately at `epicache/` (gitignored, has its own venv with pinned `transformers 4.51.3`, `numpy 1.26.4`, `torch 2.3.0`).

Per the existing ADR (decisions/), run EpiCache on:
- The same 228-task tune split — sanity check that EpiCache also runs at budget* on Llama-3.1.
- The same 912-task hold-out — direct head-to-head with our chosen config.

Use `apple/ml-epicache`'s `scripts/run_epicache_eval_llama.sh` with `MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct` and matching budget. Output gets converted to ConvCodeWorld's `predictions.jsonl` schema for `calculate_metrics.py`.

### Phase 3 deliverable

Final ConvCodeWorld result table:
```
                        recall (228 tune)    recall (912 hold-out)
SnapKV @ budget*           ?                  ?
TurnKV(α*) @ budget*       ?                  ? (our headline number)
EpiCache @ budget*         ?                  ?
```

---

## Phase 4: secondary benchmarks (stretch — 2026-04-28)

Both are scaffolded only enough to load + run the harness, NOT yet end-to-end runnable. If Phase 3 finishes early on 04-27, attempt these. Otherwise drop to appendix-only or out of scope.

### 4a. LongMemEval_S (primary conversational, 500 probes × ~103 K tokens each)

- Reuse EpiCache's `data/longmemeval/convert_longmemeval.py` for data prep.
- Y1 setup: teacher-force the full prior history (~490 turns), apply our press at session boundaries, generate only at the probe.
- One score per probe → recall@1.
- Apples-to-apples comparison with EpiCache on the same 500 probes.

### 4b. TopiOCQA (Loyalty topic-shift validation)

- Smaller scope: only validates whether the Loyalty signal correctly identifies turn-by-turn topic shifts (vs the dataset's ground-truth topic labels).
- Doesn't need full eviction harness — instrument LoyaltyPress's accumulator + emit per-turn shift detection, then compute precision/recall vs ground truth.

### Phase 4 deliverable

LongMemEval result row + a small TopiOCQA validation plot. If only TopiOCQA finishes, that's still a paper-worthy "Loyalty's signal mechanically tracks topic shifts" claim.

---

## Phase 5: writeup + presentation (2026-04-29 → 2026-05-03)

### Working backward from deadlines:

- **2026-04-30 — final presentation** (10–15 min team talk).
  - Slides: motivation, three policies, smoke story (the +1.75 pp finding), hold-out result, future work.
  - Drafted by Gaurav, reviewed by team.
- **2026-05-03 — final report**.
  - Sections: motivation, related work (kvpress, EpiCache, ConvCodeWorld), policy design, harness implementation, smoke methodology, full ConvCodeWorld results (228 + 912), EpiCache comparison, LongMemEval (if landed), discussion (per-budget regime story), limitations, future work.
  - Use `documentation/findings.md` for already-curated insights.

### Sub-tasks on the writeup:

- **Methodology section** — record the budget-sweep insight as a methodological contribution: "evaluating KV compression policies at a single budget can mask interactions; we recommend evaluating across budgets."
- **Limitations** — model is Llama-3.1-8B; budget regime where TurnKV wins isn't universal; we haven't yet tested longer histories or different model families.
- **Reproducibility appendix** — every smoke is on `main` with deterministic seeds + checked-in `splits/`; readers can re-run.

---

## Risks + open questions

1. **Modal H100 quota / cost.** ~80 GPU-hours used so far across smokes 001-004. Phase 2 + Phase 3 add maybe another ~120 GPU-hours. Watch the bill.
2. **GPU contention causing OOMs.** Smoke #2 baseline shard 4 OOM'd at model load (Modal contention, not real OOM). Have a pattern for retry. May happen again in Phase 3 hold-out — the 912-task pull is bigger.
3. **Path-related Modal hiccups already paid down**:
   - WSL2 networking dies under Cisco AnyConnect (routing metric 1 vs 15) → use Windows-side `modal.exe` from Git Bash with `MSYS_NO_PATHCONV=1`, `PYTHONIOENCODING=utf-8`.
   - Windows MAX_PATH blows up `modal volume get` for the long turnkv result subdir → pull `predictions.jsonl` per-shard into a flat `/e/sm/...` dir.
4. **EpiCache is gnarly.** Pinned numpy 1.26.4 vs our 2.4 venv; needs full separate venv. Plan a half-day buffer in Phase 3.
5. **Whether the +1.75 pp at budget=1024 holds on the 912-task hold-out.** If it doesn't, paper story is much weaker. Mitigation: smoke #4 was on a 228-task subset of the same dataset, no leakage from sweeping; reasonable expectation that the gap holds. If it shrinks, report honestly.
6. **TopiOCQA / LongMemEval may not land.** Both need scaffolding work; if Phase 3 takes longer than planned, drop these. The paper still works on ConvCodeWorld alone.
7. **Tight schedule for the team.** Yagneek owns the live-loop harness, Pradyut the metrics scaffolding. Coordinate so I'm not sole-pathing the experiments.

---

## Calendar

| Date    | Day      | Plan |
|---      |---       |---   |
| 04-25   | (today)  | Smoke #4 done + pushed. Plan written + pushed. |
| 04-26   | Sat      | Phase 2: budget sweep (512/768) + α-ablation at budget=1024. Lock in `(α*, budget*)`. |
| 04-27   | Sun      | Phase 3: hold-out (912) + EpiCache. End of day = headline result table. |
| 04-28   | Mon      | Phase 4 stretch: TopiOCQA validation; start LongMemEval scaffolding if time. Begin slides outline. |
| 04-29   | Tue      | Slides + practice run. Begin paper draft. |
| 04-30   | Wed      | **Final presentation.** Polish paper draft. |
| 05-01   | Thu      | Paper draft, full pass. |
| 05-02   | Fri      | Paper revisions, figures. |
| 05-03   | Sat      | **Final report due.** |

This is tight but achievable. Phase 2 and Phase 3 are the load-bearing experiments; everything else can compress if needed.
