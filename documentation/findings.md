# Findings

Curated insights worth remembering. Rewrite when wrong; cross-reference to the journal entries that produced them.

Organize by topic. When a topic grows, split it out into its own section.

---

## Benchmarks

### SCBench does not test conversational multi-turn (2026-04-19)
Within each SCBench "session", the turns are **independent queries over the same long context** (e.g. multiple MCQs over one reading passage, multiple key lookups over one dictionary). Turn k never references the answer of turn k-1 — so SCBench evaluates *KV-cache reuse under shifting queries*, not *conversational memory*. Our three turn-aware presses (Cross-Turn Accumulation, Turn-Floor, Role-Boundary Anchor) assume turn-to-turn dependency; running them on SCBench exercises the cross-*query* accumulation idea only, and leaves role-boundary and turn-floor signals inert. **Implication**: demote SCBench to an appendix benchmark reframed as "cross-query KV retention on shared contexts"; make LongMemEval and ConvCodeWorld the primary benchmarks.

### Per-problem context length is heavily skewed in SCBench (2026-04-19)
Synthetic tasks (scbench_kv / prefix_suffix / vt / repoqa / mf / many_shot / summary*) have uniform context lengths within the model's 128K window. Three natural-document tasks (`scbench_qa_eng`, `scbench_choice_eng`, `scbench_qa_chn`) have heavily skewed distributions: **`scbench_qa_chn` ranges 27K to 5.9M tokens** (one session at 5.9M). Any fixed `--max_context_length=131072` truncates 114/922 sessions (~12%). Truncation is unavoidable with Llama-3.1-8B; switching to Qwen2.5-7B-Instruct-1M removes truncation for all but the 5.9M outlier. Full per-session stats in `context/experiments/scbench_per_session_aggregated.csv`.

### Three latent schema quirks in SCBench that silently zero out the prefix (2026-04-19)
1. `scbench_vt` puts the long context under `input`, NOT `context`. Blind `row["context"]` returns empty.
2. `scbench_mf` context is a `list[int]` of ~30,000 numbers. Must coerce with `" ".join(map(str, ...))` before tokenizing.
3. `scbench_mf` per-turn `answer` is ALSO a list, not a string. Answer-comparison needs `str()` coercion.
All three coercions should live in one loader helper so presses never see raw heterogeneity.

### ConvCodeWorld is the right primary coding benchmark (2026-04-19)
10-turn refinement trajectories on 1,140 BigCodeBench problems × 5 feedback configurations. Per-turn pass/fail labels give verifiable ground truth without running code at eval time. Measured token counts (50-task sample, Llama-3.1 tokenizer): min 1.6K, median 4.5K, max 21.8K. At 1/32 compression the median trajectory keeps ~140 tokens — aggressive enough that press quality dominates. Dataset license is unstated on HF — verify before publication.

### Compression ratio, not absolute context size, is what makes eviction methods distinguishable (2026-04-19)
I initially dismissed Code-Feedback and ConvCodeWorld for "being too short" (2–8K tokens vs SCBench's 60K+). That's wrong for our research question. At 1/32 compression, an 8K trajectory keeps 250 tokens — a discriminative regime where SnapKV vs. StreamingLLM vs. our presses produce measurably different outputs. The long-context story is about *memory savings at scale*; the discriminative story is about *ratio*. Our paper claim is the second.

## KV eviction methods

*(empty — populate as we learn)*

## Multi-turn vs single-turn behavior

*(empty — populate as we learn)*

## EpiCache baseline

*(empty — populate as we learn)*

## Our three policies

*(empty — populate as we learn)*
