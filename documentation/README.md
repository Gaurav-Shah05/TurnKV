# Documentation

Two files. Two purposes.

## `journal.md` — running log, append-only

Where you write **as you work**. Observations, questions, surprises, hypotheses you want to test, things that confused you for 20 minutes. Dated entries, most recent on top.

Low bar to write. If it's only half-formed, that's fine — write it anyway. Your teammates can catch things you missed.

**Good journal entries:**
- "Turn-Floor with α=0.3 beats α=0.5 on `scbench_vt` — might be because shorter turns dominate; need to check turn-length distribution."
- "EpiCache clustering embeddings are Qwen3-0.6B by default, but their sentence-model variant (MiniLM) is reported as faster. Test both?"
- "Q: does SnapKV re-score the KVs of prior turns when a new turn arrives, or only the fresh tokens? Reading [kvpress/kvpress/presses/snapkv_press.py](../kvpress/kvpress/presses/snapkv_press.py) now."

**Not journal entries:**
- "Set up my venv today." (not interesting)
- "Fixed bug in press." (belongs in a commit message)

## `findings.md` — consolidated insights, curated

Where we **lift the good stuff out of `journal.md`** once it hardens into something we want to remember. Think of it as the "what we actually learned" document.

Should end up as the raw material for the Related Work, Method, and Discussion sections of the final paper. Refactor aggressively — when an entry becomes wrong or gets superseded, rewrite it.

**Good findings entries:**
- "Attention sinks (StreamingLLM) are about position, not content — the first 4 tokens dominate regardless of what they are. Implication: our Role-Boundary Anchor press needs to verify that role tokens actually inherit sink behavior on Llama-3.1's chat template."
- "SCBench scores KV retrieval tasks with exact string match on the value field, but QA tasks use F1. Mixing both in the same ablation table requires care."

## When to promote from journal → findings

- The observation repeats across multiple experiments.
- You'd reference it in a design decision or the paper.
- You'd want a new teammate to know this on day 1.

## What belongs where (decision table)

| Thought                                      | Write it in            |
|----------------------------------------------|------------------------|
| "I tried X, it did Y"                        | `journal.md`           |
| "I've now tried X five ways and Y always wins"| `findings.md`         |
| "We're going to implement Z because …"       | `context/decisions/`   |
| "Here's the full experiment result table"    | `context/experiments/` |
| "I worked on this today"                     | `context/logs/`        |
