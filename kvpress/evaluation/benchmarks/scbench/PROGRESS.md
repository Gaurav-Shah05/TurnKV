# SCBench + kvpress — progress notes

Last updated: 2026-04-18 (maintenance doc; update when behavior changes.)

## What’s implemented

- **Multi-turn SCDQ loop** (`loop.py`): long shared context + per-turn questions; KV cache carries across turns (not reset to context-only each turn).
- **Two KV compression policies** (CLI / `ScbenchConfig.kv_compression`):
  - **`context_prefill`**: compress only the **initial long-context** prefill (original integration).
  - **`decode_only`** (default in `ScbenchConfig`): **no** compression on long context or question prefills; compress **assistant decode** KV only via `AnswerSuffixDecodingPress` (`kvpress/presses/answer_suffix_decoding_press.py`) — periodic attempts every `decode_compression_interval` steps when above `decode_token_limit`, plus a final pass.
- **CLI** (`run_scbench.py`): Fire entrypoint, `PRESS_REGISTRY` presses, metrics to `predictions.jsonl` + `metrics.json`.
- **Modal** (`modal_app.py`): GPU worker runs the same CLI in-container.

## Defaults worth knowing

| Setting | Typical value | Notes |
|--------|----------------|--------|
| Dataset config | `scbench_kv` | `microsoft/SCBench` config name; `test` split. |
| `num_eval_examples` (CLI) | `-1` = all rows | Modal defaults to **`1`** for smoke runs (see `modal_app.py`). |
| `kv_compression` | `decode_only` | Switch to `context_prefill` for legacy behavior. |
| Modal model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Open weights; override for Llama 3 if your HF account has access. |
| `KV_PRESS_SCBENCH_MODEL` | Set in Modal subprocess env | Fire misparses `model=org/name`; the CLI reads this env for the full model id. |
| `max_seq_length` (Modal) | `8192` | Keeps first-prompt truncation in a sane range for smoke runs. |

## Reliability fixes applied

1. **Subprocess Python on Modal**: use `/root/kvpress/.venv/bin/python` so deps from `uv sync` (e.g. pandas) are available — not bare `/usr/local/bin/python`.
2. **Repo root in `modal_app.py`**: walk parents until `pyproject.toml` is found (avoid `Path(__file__).parents[3]` `IndexError` on Modal).
3. **`.env` without `python-dotenv`**: parse `HF_TOKEN` from repo `.env` manually for Modal secrets.
4. **First-prompt length**: `max_input_length` clamped; `run_scdq_example` gets **`max_context_tokens=max_input_length`** (never `None` as “unlimited”); `_first_prompt_token_cap()` ensures `None` still maps to a finite cap from `max_position_embeddings`.
5. **Modal**: do not write logs into the repo during `modal run` (Modal treats that as changing the build context).

## How to run

**Local (uses your GPU if CUDA is available):**

```bash
cd evaluation
python benchmarks/scbench/run_scbench.py num_eval_examples=1 press_name=snapkv
```

**Modal (cloud GPU):**

```bash
modal run evaluation/benchmarks/scbench/modal_app.py::run_scbench
```

Set `HF_TOKEN` (or use repo-root `.env`) before `modal run` for gated datasets/models if needed.

## Limitations / follow-ups

- **`decode_only`** requires a **`ScorerPress`** (e.g. SnapKV, Knorm); not `AdaKVPress` / `KVzipPress` on the decode path as wired today.
- **Full `scbench_kv`** rows can be very long; truncation + caps are required for small models (e.g. TinyLlama 2048 ctx).
- **Modal** must be run from your machine where `modal` is logged in; automated agents may hit environment/spawn limits.

## Key files

| File | Role |
|------|------|
| `loop.py` | SCDQ inference, `kv_compression` modes, greedy decode + optional suffix press |
| `run_scbench.py` | Config, dataset load, pipeline, metrics |
| `scdq_prompts.py` | MInference-style prompts |
| `calculate_metrics.py` | Task metrics |
| `modal_app.py` | Modal image, secrets, subprocess invocation |
| `kvpress/presses/answer_suffix_decoding_press.py` | Decode-only suffix compression |
