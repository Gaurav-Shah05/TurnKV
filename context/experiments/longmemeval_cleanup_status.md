# LongMemEval cleanup status (as shipped in `longmemeval-data-cleaned.tar.gz`)

## Where to get the data

- **Google Drive download** (cleaned, ready to use): https://drive.google.com/file/d/1zo5C2sKsN3-2TUZt7kiRd2wsZLmyd-4y/view
- Upstream source (raw, uncleaned): https://github.com/xiaowu0162/LongMemEval

The Google Drive file is `longmemeval-data-cleaned.tar.gz` (847 MB). Download it, drop it at the project root, then extract:

```bash
tar -xzf longmemeval-data-cleaned.tar.gz -C context/datasets/
mv context/datasets/data context/datasets/longmemeval
```

The tarball and extracted JSON are both gitignored — too big for git.

## What's in the tarball

The tarball at the project root (`longmemeval-data-cleaned.tar.gz`, 847 MB) contains three files under `data/`:

- `longmemeval_s_cleaned.json` (277 MB) — 500-probe S variant; cleaned per the table below.
- `longmemeval_m_cleaned.json` (2.7 GB) — 500-probe M variant (~500 sessions per probe); same cleanup applied.
- `longmemeval_oracle.json` (15 MB) — oracle subset (evidence sessions only, no fillers).

These files are **gitignored** (too big for GitHub's 100 MB per-file limit). Each teammate must either:

1. Copy `longmemeval-data-cleaned.tar.gz` into their local `E:\15642_MLSys\kvpress-multi-turn\` checkout, then
2. `tar -xzf longmemeval-data-cleaned.tar.gz -C context/datasets/ && mv context/datasets/data context/datasets/longmemeval` (matches the layout the analysis scripts expect).

## Session-level cleanup (applied upstream before tarball was built)

The following per-session issues were identified in the source LongMemEval release. Actions are reflected in the `_cleaned.json` files:

| Question ID | Session ID | Issue | Action |
|-------------|------------|-------|--------|
| 118b2229 | 5cd6ab1b | ambiguous reference | session removed |
| ad7109d1 | 26a0ee23 | answer interference | session removed |
| c8c3f81d | 19ec83c5_1 | info leak (no conflict) | session removed |
| 0862e8bf | 8f532b13_1 | info leak (no conflict) | session removed |
| 6d550036 | e255d6fc_2 | answer interference | session removed |
| gpt4_d12ceb0e | 157dc93d | answer interference | session removed |
| 60472f9c | a8fc1154_3 | answer interference | session removed |
| 60472f9c | 541ecc45_1 | answer interference | session removed |
| 1c0ddc50 | 2aa70c9c_1 | info leak (no conflict) | no action |
| 4f54b7c9 | f508f11f_2 | answer interference | session removed |
| 5025383b | d4230511_4 | answer interference | session removed |
| gpt4_7abb270c | 253742b4_1 | answer interference | session removed |
| 2ebe6c90 | 4e9524c7_4 | answer interference | session removed |
| 370a8ff4 | 3fe482ad | ambiguous reference | session removed |
| gpt4_d6585ce8 | c2e2d770 | answer interference | session removed |
| gpt4_d31cdae3 | 1dbe5e0c_2 | answer interference | session removed |
| 9ea5eabc | 6e2cca63_1 | ambiguous reference | no action |
| 41698283 | 431ae25c | info leak (no conflict) | no action |
| a2f3aa27 | eb30ba3d_1 | ambiguous reference | session removed |
| dad224aa | 6de8645d_1 | info leak (no conflict) | session removed |
| 2133c1b5_abs | a864e7aa_5 | ambiguous reference | no action |

### Issue definitions

- **ambiguous reference** — the filler session contains a scenario similar to the probe question; might confuse retrieval evaluation.
- **answer interference** — a filler session accidentally changes or overrides the ground-truth answer.
- **info leak (no conflict)** — a filler session leaks the answer text but doesn't contradict it; affects retrieval scoring but not answer correctness.

## Per-probe token stats (S variant)

After cleanup, the S split is 500 probes, distribution below (Llama-3.1 tokenizer, `add_special_tokens=False`):

| question_type | N | sessions (min/med/max) | total_haystack_tokens (min/med/max) | turns (med) |
|---------------|--:|-------------------------|--------------------------------------|------------:|
| knowledge-update | 78 | 39 / 48 / 55 | 100,721 / 103,081 / 105,120 | 487 |
| multi-session | 133 | 38 / 47 / 54 | 96,541 / 103,167 / 105,065 | 490 |
| single-session-assistant | 56 | 42 / 48 / 62 | 99,640 / 103,513 / 104,983 | 495 |
| single-session-preference | 30 | 42 / 47 / 55 | 100,635 / 103,598 / 104,881 | 494 |
| single-session-user | 70 | 41 / 50 / 57 | 99,077 / 103,200 / 104,852 | 498 |
| temporal-reasoning | 133 | 41 / 47 / 61 | 99,458 / 103,043 / 105,124 | 489 |

Overall **median 103,168 haystack tokens** (p5=100,919, p95=104,764, max=105,124). Distribution is tight by design — the benchmark pads each probe's history to target a fixed token budget.

Compression sweep at the median:

| Ratio | Tokens kept |
|------:|------------:|
| 1/2 | 51,584 |
| 1/4 | 25,792 |
| 1/8 | 12,896 |
| 1/16 | 6,448 |
| 1/32 | 3,224 |

Full per-probe data: [`longmemeval_s_per_session_aggregated.csv`](longmemeval_s_per_session_aggregated.csv). Per-turn data is not in git (too large); run [`scripts/analyze_longmemeval.py`](scripts/analyze_longmemeval.py) locally to regenerate it — output lands at `longmemeval_s_per_turn_tokens.csv` (gitignored).
