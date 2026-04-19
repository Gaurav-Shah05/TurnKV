# 001: Multi-turn harness and turn-aware eviction API

- **Status**: Proposed
- **Date**: 2026-04-19
- **Authors**: TurnKV team

## Context

kvpress's current `evaluate.py` handles one query at a time. It prefills a context, applies a press, and answers one or more independent questions — all sharing the same compressed cache. That matches SCBench's actual structure (turns are independent queries) but not real conversations (where later turns depend on earlier ones).

To evaluate our three turn-aware presses, we need a harness that:

1. Keeps the KV cache across turns.
2. Re-runs the press at each turn boundary.
3. Lets the press know when a turn ended, which role produced it, and where its tokens sit in the cache.

None of the four benchmark scaffolds (SCBench, ConvCodeWorld, LongMemEval, TopiOCQA) are runnable end-to-end until this harness exists. This ADR locks down its architecture and the API between it and the presses so the three of us can code in parallel.

## Decision

### 0. Key assumption: long context is permanent

Every token in the KV cache belongs to one of two buckets:
- **KEEP** — permanent. Never evicted. Typically the initial document / prompt at turn 0. Must fit within a fixed prefix of the model's context window.
- **COMPRESS** — everything the user and model say during the conversation. Press operates here.

This is a consequence of kvpress not supporting sparse retrieval from a backing store — whatever's in the cache is what the model sees. We reserve the initial context so it's always accessible; all compression budget applies to accumulated Q + A content only.

Per benchmark:
- ConvCodeWorld: KEEP = task prompt (~500 tok). COMPRESS = code + feedback across 10 iters (~4.5K tok).
- LongMemEval_S (Y1 setup): KEEP = empty. COMPRESS = all ~103K tokens of the prior chat history — see §8.
- TopiOCQA: KEEP = empty. COMPRESS = accumulated Q + A + retrieved passages (~1.7K tok).

### 1. Two compression cadences

Compression fires at two different moments:

**Local (within a turn, during decode)** — runs kvpress's existing `DecodingPress` unchanged, wrapped around the base eviction press. Every 128 decode steps, if `|current_turn_prefill| + |current_turn_decoded_so_far| > local_budget`, compress only the current turn's decoded tokens. Never touches prior turns. Local budget is fixed at **4096 tokens** for all benchmarks.

**Global (at turn boundary, once per turn)** — fires after turn k finishes generating. If the full cache (all turns so far) exceeds `global_budget`, we apply our three-policy weighting to the base press's scorer and evict down to `global_budget`. Does not fire mid-turn.

The two stack: `TurnAwareGlobalPress(DecodingPress(base_press))`.

### 2. Global budget: one value per benchmark

No single number works across benchmarks (LongMemEval's context is ~100K, TopiOCQA's is ~0). Calibrate so global compression first fires around turn 4:

```
global_budget = median_initial_context + 4 × median_per_turn_tokens
```

Per-benchmark values read from `context/experiments/*_per_session_aggregated.csv`:

| Benchmark | median context | median per-turn | `global_budget` |
|-----------|---------------:|----------------:|----------------:|
| LongMemEval_S | 103,168 | ~500 | **~105,000** |
| ConvCodeWorld | ~500 | ~1,000 | **~4,500** |
| TopiOCQA | 0 | ~500 | **~2,000** |

Reported compression ratios (1/2, 1/4, 1/8, 1/16, 1/32) are computed relative to `global_budget`, not to the model's 128K window, so cross-benchmark numbers stay comparable.

### 3. Three policies as weights on the base scorer

The three policies don't evict. They produce per-token weights that multiply the base press's score. Final ranking is:

```
final_score(t) = base_scorer(t) × weight(t)

weight(t) = 1
          + α_floor   × 𝟙[t belongs to a turn's reserved floor]
          + α_anchor  × 𝟙[t sits in a role-boundary window]
          + α_loyalty × loyalty(t) / max_loyalty
```

Evict the bottom-ranked tokens until the budget is met. All α default to 1.0.

**A — Turn-Floor** (reserve budget per turn). Each completed turn k gets at least:

```
floor_k = max(c, α_floor_len × global_budget × |T_k| / Σ_j |T_j|) × γ^(current_turn − k)
```

`α_floor_len ≈ 0.3`, `γ ≈ 0.9` (old turns shrink but never vanish), `c = 5` (hard minimum). Turns shorter than 10 tokens (single-sentence acknowledgments) are exempt.

**B — Role-Boundary Anchor** (protect tokens at turn edges). Reserve `w_k = max(3, ⌊β × |T_k|⌋)` tokens at role boundaries, `β ≈ 0.15`. For user turns, keep both the first *w* tokens (intent) and last *w* tokens (full request). For assistant turns, keep only the last *w* (answer crux — assistant openings are often boilerplate).

**C — Loyalty score** (reward tokens other turns still need). Per-token counter. **Updates during BOTH prefill and decode of the current turn** — wherever the model runs attention over the cache, past-turn tokens can accumulate loyalty. When a past-turn token receives attention in the **top 25%** of a query's distribution, its loyalty += 1. Current turn's own tokens never accumulate loyalty — they become eligible only when turn k+1 starts. During LongMemEval history prefill (Y1), each subsequent session's prefill attends over earlier sessions' tokens — loyalty accumulates session-by-session so semantically-relevant early-session tokens get reinforced many times; pure filler accumulates nothing.

### 4. Turn-boundary API

The harness owns turn-boundary metadata. The press consumes it via callbacks. Tokenizer-agnostic — no scanning chat-template tokens.

Shared dataclass:

```python
@dataclass
class TurnBoundary:
    turn_idx: int         # 0 = static context, 1..N = turns
    start_kv: int         # inclusive KV cache position where this span begins
    end_kv: int           # exclusive KV cache position where it ends
    role: str             # "context" | "user" | "assistant" | "feedback"
```

Press mixin (all turn-aware presses inherit this):

```python
@dataclass
class TurnAwareMixin:
    turn_boundaries: list[TurnBoundary] = field(default_factory=list)
    loyalty: dict[int, int] = field(default_factory=dict)
    current_turn: int = 0

    def on_turn_start(self, turn_idx, role, start_kv): ...
    def on_turn_end(self, turn_idx, role, start_kv, end_kv): ...
    def update_loyalty(self, query_attentions, query_turn_idx): ...
    def compute_weights(self, kv_len) -> torch.Tensor: ...
```

Harness calls pattern:

```python
# In multi_turn_evaluate.py
press.on_turn_start(turn_idx, role=turn.role, start_kv=len(kv))
# ... prefill + generate with DecodingPress active (local regime) ...
press.on_turn_end(turn_idx, role="user",      start_kv=..., end_kv=...)
press.on_turn_end(turn_idx, role="assistant", start_kv=..., end_kv=len(kv))

if len(kv) > global_budget:
    press.run_global_compression(target=global_budget)
```

The press doesn't parse tokens to figure out turn boundaries. The harness tells it.

### 5. Base eviction techniques

Four base techniques, all flash_attention_2 compatible:

| Technique | Query-aware | Basis | Head-adaptive |
|-----------|:-----------:|-------|:-------------:|
| SnapKV | yes | attention (observation window) | no |
| Ada-SnapKV | yes | attention | **yes** |
| StreamingLLM | no | **position** | no |
| ExpectedAttention | **no** | attention (expected over future queries) | no |

Each base press gets two variants in the experiment matrix:
- **raw** — the baseline; no three-policy weighting.
- **+turn-aware** — `TurnAwareGlobalPress(base_press)`; policies applied at global compression.

Eight total rows per (benchmark, compression ratio) cell.

### 6. Baselines

- Same eviction surface as our methods (both touch prefill + decode).
- Same calibrated `global_budget` per benchmark.
- Same `local_budget = 4096` and `DecodingPress` wrapping.
- Only difference from turn-aware variants: whether the three policies' weights are applied during global compression.

This is the cleanest possible apples-to-apples comparison.

### 7. Evaluation plan

**Headline table**: accuracy (or benchmark-specific metric) vs. compression ratio, broken out by {benchmark, technique, variant, ratio}:
- 3 benchmarks × 4 techniques × 2 variants × 5 ratios = **120 cells per benchmark**

**Per-turn accuracy plots**: one line per technique variant, x-axis = turn number, y-axis = accuracy. Shape of the decay curve is what a reviewer will look at first.

**Ablations (on LongMemEval_S only)**: individual policy on/off (3 × 8 cells), α sweep for each policy (3 × 5 values × 8 cells). Cap at one compression ratio to keep the matrix tractable.

**External comparison**: EpiCache (in its own venv) on LongMemEval_S at matching budgets (2K/4K/6K/8K). Side-by-side table.

**Regression guard**: LongBench (single-turn) on each technique at one compression ratio. Confirms our turn-aware weighting doesn't hurt single-turn performance.

## Alternatives considered

**SCBench as primary benchmark.** Rejected. Per-problem inspection showed its "multi-turn" mode is actually independent queries over a shared context. Kept as an appendix with a reframed claim about cross-query KV retention.

**Growing budget (64K → 128K, +16K per event).** Rejected from v1. Adds a design axis and a debugging variable before we've validated the static calibration. Revisit as an ablation if a reviewer asks.

**Single static budget across all benchmarks.** Rejected. Context sizes span 60× across our three benchmarks; no single value works.

**Turn-boundary detection via tokenizer chat-template markers.** Rejected. The harness already has the information; parsing tokens is tokenizer-specific and error-prone.

**KVzip in the core matrix.** Rejected. Its importance comes from a separate context-reconstruction forward pass, which doesn't fit the per-token weight multiplication pattern. Replaced with ExpectedAttention. KVzip can still be a stretch goal via a bespoke subclass.

**H2O / ObservedAttentionPress in the core.** Rejected. Duplicates SnapKV's quadrant and forces eager attention (slower). Fine to include as a fifth technique for citation breadth, but not required.

**Eager attention as default.** Rejected. SDPA is faster than eager and gives us the attention weights our loyalty update needs. Flash is even faster but hides the weights. SDPA is the right default; eager only where a press actually forces it (H2O).

## Consequences

**New code (in dependency order):**
- `kvpress/kvpress/presses/turn_aware_base.py` — `TurnBoundary`, `TurnAwareMixin`
- `kvpress/kvpress/presses/turn_floor_press.py` — policy A
- `kvpress/kvpress/presses/role_boundary_anchor_press.py` — policy B
- `kvpress/kvpress/presses/loyalty_press.py` — policy C (also performs loyalty updates during `score()`)
- `kvpress/kvpress/presses/turn_aware_global_press.py` — wraps a base press + the three policies, runs the weighted scoring at global compression time
- `kvpress/evaluation/multi_turn_evaluate.py` — the harness (new CLI entry point)
- `kvpress/evaluation/benchmarks/<name>/multi_turn_loader.py` — one per benchmark, handles schema quirks (ConvCodeWorld no-op turns, TopiOCQA per-turn passages, LongMemEval session→turn flattening)

**Modified:**
- `kvpress/evaluation/evaluate_registry.py` — register the new presses

**Tests (failing-first):**
- `compress()` honors the budget.
- Loyalty increments correctly across turn boundaries.
- Turn-floor allocations sum to the expected total.
- Role-boundary windows are sized correctly.
- Integration: replay one LongMemEval_S session at 1/4, check state is sane after 10 turn boundaries.

**Paper narrative:** the primary claim is "turn-aware weighting of a base scorer improves accuracy on conversational multi-turn at aggressive compression, without hurting single-turn." Everything above is what makes that falsifiable.

## 8. ConvCodeWorld Mode 2 replay + LongMemEval Y1 teacher-forcing

Both benchmarks need a specific evaluation setup that's not obvious from the schema:

**ConvCodeWorld — Mode 2 (static replay)**. At every iteration k, teacher-force the reference model's prior code and prior feedback (from the dataset). Our model generates code at that iteration. We execute the generated code against BigCodeBench unit tests to get pass/fail. All 10 iterations always run, regardless of whether our model solved it earlier — because the next iter's context is fixed from the reference trajectory. Requires a code-execution sandbox (subprocess with timeout is sufficient for our purposes; Docker is cleaner but heavier).

Alternative Mode 1 (live loop) would have our model's own code fed back through real feedback regeneration. We reject it: compression could change iter-k's output, which would cascade into different iter-(k+1) context, compounding differences and making compression-ratio comparisons apples-to-oranges.

**LongMemEval_S — Y1 (teacher-forced history)**. We don't have the model generate intermediate assistant responses — we teacher-force all ~490 user/assistant turns from the dataset verbatim during prefill. Our model generates only at the final probe. One accuracy score per probe.

Y1 is the only setup that:
1. Cleanly isolates the compression effect (the only way the probe answer can be wrong is compression losing evidence, not generation drift).
2. Matches EpiCache's evaluation protocol — apples-to-apples with their numbers.
3. Is computationally tractable (500 generations, not 500 × 490).

Under Y1, the chat history is treated as **turn content in the COMPRESS bucket** (not as a permanent document). It's prefilled session-by-session so loyalty updates, global compression checks, and press state can fire at session boundaries.

## Open questions

1. **ConvCodeWorld no-op turns** — 39.6% of turns have empty feedback because the task already passed. Default for v1: harness skips them. Revisit if it biases the per-turn accuracy plots.
2. **TopiOCQA aggressive ratios** — median trajectory is 1.7K tokens; at 1/32 we keep ~53 tokens, probably unusable. Option: skip 1/16 and 1/32 for TopiOCQA, or filter to the 500 longest conversations.
3. **α defaults** — start at 1.0 each. Sweep once on LongMemEval_S; if the optimum varies a lot across benchmarks, that's itself a finding.
4. **KVzip as stretch goal** — bespoke subclass intercepting its score tensor. Target: paper appendix if v1 leaves time.
5. **Qwen2.5-7B-Instruct-1M on LongMemEval truncation outliers** — 6 probes exceed 128K. Single stretch run to report as a footnote.
6. **ConvCodeWorld execution infra** — local subprocess with timeout for v1; revisit if Docker is needed for reliability.
7. **CoT for ConvCodeWorld** — prompt model to reason through feedback before emitting code. Expected 2–4× longer decodes, stresses local regime more. Default: on for ConvCodeWorld, off for LongMemEval.
