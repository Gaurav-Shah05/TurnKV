# SCBench

[SCBench](https://arxiv.org/abs/2412.10319) (Li et al., ICLR 2025) evaluates long-context methods — including KV-cache eviction — across 12 task types under two modes: **multi-turn** (KV cache reused across turns within a session) and **multi-request** (cache reused across sessions with different queries). It is the primary benchmark for the TurnKV project.

- Paper: https://arxiv.org/abs/2412.10319
- Dataset: https://huggingface.co/datasets/microsoft/SCBench
- Reference evaluation code: https://github.com/microsoft/MInference/tree/main/scbench

## Dataset schema

Each row in `microsoft/SCBench` is one **session**:

```python
{
    "id": int,                              # session id
    "context": str,                         # shared long context (299k – 3.17M chars)
    "multi_turns": [                        # 2–4 turns per session
        {"input": str, "answer": str, "options": list[str] | None},
        ...
    ],
}
```

There are 12 subsets (922 sessions total):

| Subset                           | Sessions | Category            |
|----------------------------------|---------:|---------------------|
| `scbench_kv`                     | 100      | String retrieval    |
| `scbench_prefix_suffix`          | 100      | String retrieval    |
| `scbench_vt` (variable tracing)  | 90       | String retrieval (multi-hop) |
| `scbench_repoqa`                 | 88       | Semantic retrieval  |
| `scbench_qa_eng`                 | 69       | Semantic retrieval  |
| `scbench_qa_chn`                 | 35       | Semantic retrieval  |
| `scbench_choice_eng`             | 58       | Semantic retrieval  |
| `scbench_many_shot`              | 54       | Global processing   |
| `scbench_mf` (Math.Find)         | 100      | Global processing   |
| `scbench_summary`                | 70       | Global processing   |
| `scbench_summary_with_needles`   | 70       | Multi-tasking       |
| `scbench_repoqa_and_kv`          | 88       | Multi-tasking       |

## Two evaluation modes

**Multi-request (supported by the existing `evaluate.py` runner).** Flatten each session into N rows, one per turn; treat turns as independent queries against the same compressed context. Multiple turns per row with `questions=[...]` list — kvpress's current pipeline does this natively.

**Multi-turn (NOT yet supported — requires a new runner).** Turn-by-turn iteration where each turn appends the previous Q/A to the running context, re-applies the press at the boundary, and preserves the prior KV cache. This is the setting where turn-aware eviction policies (TurnKV's contribution) should matter. A `multi_turn_evaluate.py` harness will be added separately — see `context/decisions/` for the design note.

## Usage

### Prepare the flattened dataset (for multi-request mode)

```bash
# Fetches microsoft/SCBench, flattens multi_turns → one row per turn with
# answer_prefix, max_new_tokens, and task filled in. Saves locally by default;
# set PUSH_TO_HUB=1 + an HF repo id to publish.
python evaluation/benchmarks/scbench/create_huggingface_dataset.py
```

### Run (multi-request mode, once dataset is registered)

```bash
# From evaluation/
python evaluate.py \
    --dataset scbench \
    --data_dir scbench_kv \
    --press_name snapkv \
    --compression_ratio 0.5 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

Use `--data_dir <subset>` to pick one of the 12 subsets above.

## TODO (before this integration is useful)

1. **Port `compute_scores.py` metrics verbatim** from `microsoft/MInference/scbench/` — the stubs in [`calculate_metrics.py`](calculate_metrics.py) use reasonable defaults (F1 / ROUGE-L / substring-match) but haven't been reconciled with the upstream scoring exactly. Required for reporting numbers comparable to the SCBench paper.
2. **Publish the flattened dataset to HF** (or accept local-path loading) so `DATASET_REGISTRY["scbench"]` resolves without running the prep script every time.
3. **Implement `multi_turn_evaluate.py`** — the turn-aware harness. This is where the project's core measurements happen. Document the design in `context/decisions/` before implementing.
4. **Add per-turn-number reporting** in `calculate_metrics` — group by `turn_index` column and emit `turn_1_score`, `turn_2_score`, etc. Matches SCBench paper Tables 7–8.
