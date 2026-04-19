# Journal

Append new entries at the **top**. Each entry is dated and signed.

Format:
```
## YYYY-MM-DD — short title — @author
body (1-N paragraphs)
```

---

## 2026-04-18 — repo restructured to turnkv layout — @gaurav

Moved all kvpress-native code (including evaluation/, tests/, notebooks/, kvzap/, pyproject.toml) into `kvpress/` subfolder. Added `documentation/` (this file + `findings.md`), `.claude/CLAUDE.md`, and the parent README. `epicache/` is gitignored — each teammate clones `apple/ml-epicache` locally. `context/` stays where it was, now with scaffolding (decisions/, experiments/, logs/, templates). GitHub repo will be renamed `turnkv` via settings; the fork network was already detached.

SCBench benchmark scaffold landed at `kvpress/evaluation/benchmarks/scbench/` — loader + approximate metrics + register in `evaluate_registry.py`. **Not yet runnable end-to-end**: (a) microsoft/SCBench schema needs flattening via `create_huggingface_dataset.py`, (b) metrics need reconciliation with upstream `compute_scores.py`, (c) true multi-turn harness (preserving KV cache across turns) is the next substantial piece of work.
