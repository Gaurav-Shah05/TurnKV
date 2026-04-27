# ColBench (Backend)

[ColBench](https://arxiv.org/abs/2503.15478) is the multi-turn collaborative coding benchmark introduced with [SWEET-RL](https://arxiv.org/abs/2503.15478) (Meta FAIR, March 2025). The Backend split asks an agent to solve a Python task by chatting with a *simulated human* over up to 10 turns. The simulator has access to the reference solution and hidden tests; the agent does not. At each turn the agent emits either a clarifying question (natural language) or a final code submission (fenced ```python``` block); the loop terminates on the first submission and pass/fail is decided by the hidden tests.

- Paper: https://arxiv.org/abs/2503.15478
- Dataset: https://huggingface.co/datasets/facebook/collaborative_agent_bench
- Scope of this folder: **Backend Programming only** (Python + unit tests). The Frontend Design split (HTML/CSS + vision-LLM judge) is out of scope.

## Why it complements ConvCodeWorld

This folder is structurally parallel to `evaluation/benchmarks/convcodeworld/`: the same Modal image, the same FA3 + vLLM Triton dual-model setup (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` + `google/gemma-4-26B-A4B-it`), the same KV-cache carry-over harness, the same press registry. What changes is the *feedback distribution*: ConvCodeWorld feeds the agent compile + execute traces from real BigCodeBench tests; ColBench feeds it natural-language replies from a reference-aware human simulator. That stresses presses differently — the pages of cache that matter shift from compiler diagnostics + tracebacks to free-form clarification text — without us having to maintain two sets of execution / Modal infrastructure.

## Differences from convcodeworld

| Dimension | ConvCodeWorld | ColBench (this folder) |
|---|---|---|
| Conversation shape | 10 fixed iter-of-refinement turns | up to 10 turns of question-or-submit; loop ends at first submit |
| Feedback source | compile + execute on BigCodeBench tests + LLM verbal | NL reply from a reference-aware human simulator |
| Reference trajectories | Yes (5 feedback configs × 10 iters precomputed) | No — *live mode only*, no static replay |
| Headline metric | per-turn pass-rate curve | session pass-rate + `mean_questions_before_submit` |
| Benchmark mode flag | `--benchmark-mode live|static` | (no flag — always live) |

## Quick start

After running the splits + shards builders once (see *Splits* below), the three smoke scripts mirror convcodeworld's no_press / baseline_snapkv / turnkv_snapkv pattern:

```bash
cd kvpress

evaluation/benchmarks/colbench/modal_run_smoke_no_press.sh        --split 100
evaluation/benchmarks/colbench/modal_run_smoke_baseline_snapkv.sh --split 100
evaluation/benchmarks/colbench/modal_run_smoke_turnkv_snapkv.sh   --split 100
```

Each script fans out 10 detached Modal H200 containers (one per shard) and writes per-shard logs under `.modal_diag/`. See `MODAL_SETUP.md` for the full runbook and `MODAL_HYPERPARAMS.md` for the CLI reference.

## Splits

Build the tune/holdout split and shard it (one-time, requires HF access to the ColBench dataset):

```bash
cd kvpress
python evaluation/benchmarks/colbench/scripts/build_split.py
python evaluation/benchmarks/colbench/scripts/build_shards.py \
    --input evaluation/benchmarks/colbench/splits/tune_20pct_seed42.json \
    --num-shards 10
```

The smoke scripts fail-fast with a clear regeneration command if the splits or shards are missing.

## Output schema

Each run writes to its results directory:

- `predictions.jsonl` — one row per turn, columns include `task_id`, `iteration`, `is_question`, `is_submission`, `agent_message`, `user_reply`, `predicted_answer`, `passed`, `status`, `compilation_feedback`, `execution_feedback`, `cache_len_before_global`, `cache_len_after_global`. The column set is a strict superset of convcodeworld's.
- `predictions.csv` — same content, CSV-flat.
- `metrics.json` — `overall`, `per_iteration`, `mrr`, `recall`, `final_pass_rate`, `mean_questions_before_submit`, `status_counts`, plus the resolved config and the git revision.
- `config.yaml` — the resolved `ColBenchLiveConfig` dataclass.

## Local Fire CLI

Outside of Modal, `live_loop.py` is callable directly via Fire (useful for CPU dry-runs and tokenizer-template validation):

```bash
cd kvpress/evaluation
python benchmarks/colbench/live_loop.py \
    --press_name=snapkv \
    --compression_ratio=0.5 \
    --model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --feedback_model=google/gemma-4-26B-A4B-it \
    --feedback_attn_implementation=vllm_triton \
    --num_eval_examples=1 \
    --max_turns=4
```

## Status

Scaffolded 2026-04-27. Live-loop + Modal infrastructure landed; smoke runs require regenerating the splits (the scaffold ships only a placeholder `splits/split_manifest.json`). Headline-quality numbers depend on:

1. Confirming the upstream HF schema field names match what our loader expects (`description`, `reference_solution`, `private_tests`, `entry_point`, `code_prompt`). The loader is field-name-resilient across the common aliases but a snapshot bump may need a one-line tweak.
2. A first end-to-end Modal smoke run to validate the simulator's behavior on this prompt template (the system prompt asking it to never reveal the reference is new — convcodeworld's simulator was just generating verbal critique).
