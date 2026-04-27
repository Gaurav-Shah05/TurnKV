# ColBench Modal Hyperparameters

Source of truth: `modal_app.py`. Invoke from the `kvpress/` directory:

```bash
modal run evaluation/benchmarks/colbench/modal_app.py::main [flags]
```

Modal exposes Python `snake_case` parameters as CLI flags with hyphens.

## Dispatch Parameters

| Flag | Type | Default | Meaning |
|---|---:|---|---|
| `--press-names` | `str` | `snapkv,streaming_llm,expected_attention` | Comma-separated list of presses. The entrypoint launches one remote run per press and passes each value as `press_name`. |
| `--detach-remote` / `--no-detach-remote` | `bool` | `False` | If true, spawn remote jobs and print Modal object IDs/dashboard URLs instead of waiting. |

Note: ColBench has no `--benchmark-mode` flag because there is no static-replay
mode (no reference dialogues in the upstream dataset). Every run is a live loop.

## Model And Runtime Parameters

| Flag | Type | Default | Meaning |
|---|---:|---|---|
| `--model` | `str` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Agent (code-generation) model. Stored in `KV_PRESS_COLBENCH_MODEL` inside Modal. |
| `--feedback-model` | `str` or `None` | `google/gemma-4-26B-A4B-it` | Human-simulator model. Always required (ColBench has no static reference). |
| `--attn-implementation` | `str` or `None` | `flash_attention_3` | Attention backend for the agent. Use `eager` only for debugging fallback. |
| `--feedback-attn-implementation` | `str` or `None` | `vllm_triton` | Attention backend for the simulator. `vllm_triton` launches a local vLLM server with `VLLM_ATTENTION_BACKEND=TRITON_ATTN`. |
| `--feedback-vllm-port` | `int` | `8001` | Local vLLM OpenAI-compatible server port. |
| `--feedback-vllm-cuda-visible-devices` | `str` or `None` | `None` | Optional CUDA device mask for the feedback vLLM subprocess. Default leaves both models visible on GPU 0. |
| `--feedback-vllm-max-model-len` | `int` | `32768` | vLLM `--max-model-len` for the feedback server. |
| `--feedback-vllm-gpu-memory-utilization` | `float` | `0.75` | vLLM `--gpu-memory-utilization`. |
| `--feedback-vllm-start-timeout-s` | `int` | `1800` | Seconds to wait for the feedback vLLM server health check. |
| `--cot` / `--no-cot` | `bool` | `False` | Adds a brief "think first" instruction to the agent system prompt. Off by default (DeepSeek-R1-Distill already reasons inside its template; double-CoT burns tokens on meta-commentary). |
| `--network-isolation` | `str` | `auto` | Passed to the code executor. `auto` tries `unshare -n` and falls back. |
| `--early-stop-on-pass` / `--no-early-stop-on-pass` | `bool` | `True` | Stop appending synthetic rows once the agent submits passing code. |

## Benchmark Sampling Parameters

| Flag | Type | Default | Meaning |
|---|---:|---|---|
| `--colbench-split` | `str` | `backend` | Logical split tag baked into the results directory name. Currently only `backend` is wired; frontend is out of scope. |
| `--dataset-name` | `str` | `facebook/collaborative_agent_bench` | HF dataset id loaded by `live_loop.py`. |
| `--dataset-subset` | `str` | `backend` | HF config name passed to `load_dataset`. |
| `--hf-split` | `str` | `train` | HF split name. |
| `--num-eval-examples` | `int` | `1` | Number of tasks to run after filtering. `0` or `-1` runs all selected rows. |
| `--fraction` | `float` | `1.0` | Fraction of the filtered rows to sample before `num_eval_examples`. |
| `--task-ids` | `str` or `None` | `None` | CSV list or `@<path>` to a JSON list. The smoke scripts pass one shard JSON per worker. |
| `--max-turns` | `int` | `10` | Maximum total turns per task (questions + the submission). |
| `--max-questions-before-submit` | `int` | `9` | After this many clarifying questions the agent is forced to submit on the next turn. Must be `< max_turns`. |

## Generation And Budget Parameters

| Flag | Type | Default | Meaning |
|---|---:|---|---|
| `--max-new-tokens` | `int` | `1024` | Max agent tokens per turn (question or submission body). Greedy decode. |
| `--code-generation-until-eos` / `--no-code-generation-until-eos` | `bool` | `False` | If true, the agent decodes until EOS or a stop sequence rather than capping at `max_new_tokens`. |
| `--verbal-feedback-max-new-tokens` | `int` | `256` | Max tokens emitted by the simulator per reply. |
| `--global-budget` | `int` | `4500` | Cache length threshold for turn-boundary global compression. Compression fires only when the cache exceeds this. |
| `--compression-ratio` | `float` | `0.5` | Fraction of KV positions to evict for scorer presses. Live-loop global target = `global_budget * (1 - compression_ratio)`. |
| `--local-budget` | `int` | `4096` | Target size for answer-suffix decode compression (assistant-generated tokens only). |
| `--decode-compression-interval` | `int` | `128` | Decode-token interval between local-compression attempts. |
| `--decode-hidden-states-buffer-size` | `int` | `256` | Recent decode hidden states retained for scorer queries during answer-suffix compression. |

## Base-Press Parameters

`None` means the Modal runner does not override the press object's registry
default. As with convcodeworld, the live-loop global path requires a
`ScorerPress`-compatible base — typically `snapkv`, `streaming_llm`,
`expected_attention`, `knorm`, a `turnkv_*` variant, a `baseline_*` variant, or
`no_press`.

| Flag | Type | Default | Applies To |
|---|---:|---|---|
| `--key-channel-compression-ratio` | `float` or `None` | `None` | `think`, `snap_think` |
| `--threshold` | `float` or `None` | `None` | `DMSPress` wrappers |
| `--snapkv-window-size` | `int` or `None` | `None` | `snapkv`, `pyramidkv` |
| `--snapkv-kernel-size` | `int` or `None` | `None` | `snapkv`, `pyramidkv` |
| `--streaming-llm-n-sink` | `int` or `None` | `None` | `streaming_llm` |
| `--expected-attention-n-future-positions` | `int` or `None` | `None` | `expected_attention` |
| `--expected-attention-n-sink` | `int` or `None` | `None` | `expected_attention` |
| `--expected-attention-use-covariance` / `--no-...` | `bool` or `None` | `None` | `expected_attention` |
| `--expected-attention-use-vnorm` / `--no-...` | `bool` or `None` | `None` | `expected_attention` |
| `--expected-attention-epsilon` | `float` or `None` | `None` | `expected_attention` (Modal default constructor uses `epsilon=0.01` if omitted) |

## Turn-Aware Parameters

Same semantics as convcodeworld. Passing any policy-specific value can create
the missing turn-aware policy when the selected base press is wrapped into
`TurnAwareGlobalPress`. Set the corresponding `alpha_*` above zero for that
policy to affect ranking.

| Flag | Type | Default | Meaning |
|---|---:|---|---|
| `--alpha-floor` | `float` or `None` | `None` | Blend coefficient for `TurnFloorPress` weights. |
| `--alpha-anchor` | `float` or `None` | `None` | Blend coefficient for `RoleBoundaryAnchorPress` weights. |
| `--alpha-loyalty` | `float` or `None` | `None` | Blend coefficient for `LoyaltyPress` weights. |
| `--anchor-beta` | `float` or `None` | `None` | Fraction of each role span protected by anchoring. `[0, 1]`; class default 0.15. |
| `--floor-gamma` | `float` or `None` | `None` | Per-turn exponential decay for floor reservations. `(0, 1]`; class default 0.9. |
| `--loyalty-top-p` | `float` or `None` | `None` | Fraction of past keys per query counted as loyalty hits. `(0, 1]`; class default 0.25. |
| `--alpha-floor-len` | `float` or `None` | `None` | Length-proportional coefficient for per-turn floor size. Non-negative; class default 0.3. |
| `--min-floor-tokens` | `int` or `None` | `None` | Hard minimum number of floor tokens before decay. Non-negative; class default 5. |

## Modal Infrastructure Constants

These are constants in `modal_app.py`, not CLI flags. They mirror convcodeworld
exactly so the FA3 image layer is reused.

| Name | Value | Meaning |
|---|---|---|
| Modal app | `kvpress-colbench-live` | App name registered with Modal. |
| GPU | `H200` | Modal GPU request, configured by `KV_PRESS_COLBENCH_MODAL_GPU` before direct `modal run`, or by `--gpu-spec` in the smoke scripts. |
| Timeout | `86400` seconds | One-day remote function timeout. |
| CUDA base image | `nvidia/cuda:12.9.1-devel-ubuntu24.04` | Base image. |
| Python | `3.11` | Python version added to the CUDA image. |
| Torch seed version | `2.8.0` | Initial Torch installed into `/root/kvpress/.venv`. |
| FlashAttention-3 ref | `v2.8.3` | Git tag used to build the reduced Hopper FA3 wheel. |
| Transformers ref | `bc4b330451d0e3e33f4ac63593ed9f245227712e` | Upstream commit installed for Gemma4 support. |
| HF cache volume | `kvpress-hf-cache` | Mounted at `/root/.cache/huggingface`. **Shared with convcodeworld.** |
| Results volume | `kvpress-colbench-results` | Mounted at `/root/kvpress/evaluation/results_colbench_live_modal`. |

## Smoke Run Script Presets

Local shell dispatchers, not direct `modal_app.py::main` flags. They launch
`NUM_SHARDS` (default 10) detached calls to
`evaluation/benchmarks/colbench/modal_app.py::run_colbench_live`, one per shard
JSON. Defaults: H200, Gemma4-26B-A4B feedback, vLLM Triton attention.

| Script | Press | Mode | Notes |
|---|---|---|---|
| `modal_run_smoke_no_press.sh` | `no_press` | live | Full-KV control. Passes `--full-kv-cache` and `--code-generation-until-eos`. |
| `modal_run_smoke_baseline_snapkv.sh` | `snapkv` | live | **Live mode** — divergence from convcodeworld's smoke #1, which runs static-replay. ColBench has no reference dialogues. Compression ratio 0.5, GLOBAL_BUDGET=2048, LOCAL_BUDGET=1536, max_new_tokens=2048. |
| `modal_run_smoke_turnkv_snapkv.sh` | `turnkv_snapkv` | live | TurnKV-wrapped SnapKV with all three policies on at default alphas (1, 1, 1). Same env-var override surface as convcodeworld's turnkv smoke (`ALPHA_FLOOR`, `FLOOR_GAMMA`, `ALPHA_FLOOR_LEN`, `MIN_FLOOR_TOKENS`, `ALPHA_ANCHOR`, `ANCHOR_BETA`, `ALPHA_LOYALTY`, `LOYALTY_TOP_P`, `LOYALTY_UPDATE_EVERY`). |

All three smoke scripts accept:

```bash
./evaluation/benchmarks/colbench/modal_run_smoke_turnkv_snapkv.sh --split 20pct
./evaluation/benchmarks/colbench/modal_run_smoke_turnkv_snapkv.sh --split 100
```

| Flag | Meaning |
|---|---|
| `--gpu-spec <spec>` | Modal GPU request for each shard, default `H200`. Forwarded as `KV_PRESS_COLBENCH_MODAL_GPU`. |
| `--background-modal-cli` | Background each `modal run -d`. Default. |
| `--foreground-modal-cli` / `--no-background-modal-cli` | Wait for each Modal CLI process before submitting the next shard. Use during image rebuilds. |

The scripts accept Hugging Face credentials from `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, `MODAL_HF_SECRET_NAME`, or an existing local
`huggingface-cli login` token.

| `--split` value | Shard stem | With `NUM_SHARDS=10` |
|---|---|---:|
| `20pct`, `tune20` | `tune_20pct_seed42` | ~20-23 tasks per shard |
| `100`, `100tasks`, `small`, `tune100` | `tune_100tasks_seed42` | 10 tasks per shard |
| Any other value | Used directly as the shard stem | Depends on the shard files |

The split selector changes which files under
`evaluation/benchmarks/colbench/splits/shards/` are passed via `--task-ids @...`.
It does not change `NUM_SHARDS`; if you override `NUM_SHARDS`, matching shard
files must already exist.

## `modal_run.sh` Preset

`modal_run.sh` runs one detached full-KV ColBench command with
`MODAL_HF_SECRET_NAME=hf-secret`:

```bash
--model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
--feedback-model google/gemma-4-26B-A4B-it
--attn-implementation flash_attention_3
--feedback-attn-implementation vllm_triton
--colbench-split backend
--press-name no_press
--compression-ratio 0.0
--fraction 0.05
--num-eval-examples -1
--full-kv-cache
--require-flashdecode
--error-on-kv-cache-vram-exhaustion
--max-new-tokens 6144
--no-cot
--log-level DEBUG
```
