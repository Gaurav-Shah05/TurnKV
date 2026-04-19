# `context/experiments/`

All dataset data in four CSVs, one doc to explain them.

| File | Purpose |
|------|---------|
| `benchmarks.md` | Mental model, columns, and gotchas for all 4 benchmarks. **Start here.** |
| `<name>.csv` (×3 — scbench, convcodeworld, topiocqa) | Per-turn content CSVs. One row per turn with actual field values (truncated). |
| `<name>_stats.csv` (×3 — scbench, convcodeworld, topiocqa) | Per-session aggregated stats. One row per problem with turn count, Q/A token min/avg/max, and context/total tokens. Llama-3.1 tokenizer. |
| `longmemeval_s.csv` | Combined content + stats, one row per probe (500 total). |
| `longmemeval_cleanup_status.md` | LongMemEval Drive link, extraction steps, and cleanup log for the 21 flagged sessions. |
| `scripts/` | Regeneration scripts. |

Open any CSV in Excel or `pandas.read_csv(...)` to see actual field content for all problems in that dataset (content truncated to 400–600 chars per cell for file-size sanity).
