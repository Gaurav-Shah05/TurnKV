# ColBench Modal Setup

This runs ColBench (Backend) on Modal in live-loop mode. The default uses
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` for the agent (code generation),
`google/gemma-4-26B-A4B-it` for the human simulator (verbal feedback), CoT
disabled, and early-stop-on-pass enabled. The Modal worker requests one H200
GPU by default and uses `flash_attention_3` for the agent. The simulator runs
on a local vLLM server with `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, so its
prefill and decode both use vLLM Triton attention.

ColBench has no reference dialogues, so unlike convcodeworld there is no
`--benchmark-mode static` option — every run is a live loop.

## Local Setup

Run these from the kvpress repo root:

```bash
cd kvpress
python -m pip install "modal>=1.4.0,<2"
modal token new
modal profile current
```

The local Modal profile observed during convcodeworld setup was `ypatlola`;
both folders share that default.

## Hugging Face Access

Gemma and Llama-family weights require an HF token with model access. Use one
of these:

```bash
# Option A: local shell env, forwarded into Modal at app definition time
export HF_TOKEN=hf_...

# Option B: repo-local .env parsed by modal_app.py
printf 'HF_TOKEN=hf_...\n' > .env

# Option C: named Modal secret
modal secret create hf-secret HF_TOKEN=hf_...
export MODAL_HF_SECRET_NAME=hf-secret
```

You also need access to `facebook/collaborative_agent_bench` on HF for the
dataset itself; if it is gated, accept the terms once on the dataset page.

## Splits

Before any smoke run you must generate the tune/holdout split and shards.
This is a one-time step (the resulting JSON files commit to the repo):

```bash
cd kvpress
python evaluation/benchmarks/colbench/scripts/build_split.py
python evaluation/benchmarks/colbench/scripts/build_shards.py \
    --input evaluation/benchmarks/colbench/splits/tune_20pct_seed42.json \
    --num-shards 10
```

To also build the smaller 100-task split, manually subset
`tune_20pct_seed42.json` (the convcodeworld convention preserves the source
split order), save as `tune_100tasks_seed42.json`, and re-run `build_shards.py`.

## Smoke Run

Build the image and run one task with one press to verify wiring end-to-end:

```bash
cd kvpress
modal run evaluation/benchmarks/colbench/modal_app.py::main \
    --press-names snapkv \
    --num-eval-examples 1
```

The first run builds the image: the FlashAttention-3 layer is the slow one
(~20-25 minutes the first time) but is reusable across the convcodeworld and
colbench folders because they pin identical CUDA / Torch / vLLM / FA3 / Transformers
versions. Modal will reuse this layer until any of those pins change.

The image installs:
- CUDA 12.9.1 dev image, Python 3.11
- Torch 2.8.0 seed → vLLM nightly (CUDA 12.9 wheels)
- A reduced BF16-only Hopper FA3 wheel built from `Dao-AILab/flash-attention@v2.8.3`,
  cached in `/opt/fa3-wheelhouse/` (paged-KV, split-KV, backward, FP16, FP8,
  and SM80 kernels disabled to keep the image small)
- Pinned Transformers commit `bc4b330451d0e3e33f4ac63593ed9f245227712e` for
  Gemma4 support
- Repo source copy + editable install of the kvpress package

## Sharded Smoke Scripts

The sharded smoke scripts launch one detached Modal function per shard JSON.
Each function requests one H200 by default, and the local dispatcher backgrounds
each `modal run -d` by default:

```bash
cd kvpress

CONFIG_LABEL=h200_gemma4_all10 \
evaluation/benchmarks/colbench/modal_run_smoke_no_press.sh \
    --split 100

CONFIG_LABEL=h200_gemma4_all10 \
evaluation/benchmarks/colbench/modal_run_smoke_baseline_snapkv.sh \
    --split 100

CONFIG_LABEL=h200_gemma4_all10 \
evaluation/benchmarks/colbench/modal_run_smoke_turnkv_snapkv.sh \
    --split 100
```

| Flag | Meaning |
|---|---|
| `--split 20pct` / `--split 100` | Selects the 20% tune split or the 100-task split. |
| `--gpu-spec H200` | Modal GPU request for each shard. Maps to `KV_PRESS_COLBENCH_MODAL_GPU` inside each `modal run`. |
| `--background-modal-cli` | Submit all shards without waiting for each local Modal CLI process (default). |
| `--foreground-modal-cli` / `--no-background-modal-cli` | Wait for each Modal CLI process before submitting the next shard. Use this when rebuilding images. |

Logs go to `.modal_diag/`, with each run writing an `index.txt` mapping shards
to local log files and Modal output subdirectories.

## Defaults

| Setting | Value |
|---|---|
| Agent model | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Simulator model | `google/gemma-4-26B-A4B-it` |
| GPU | one H200 (override via `KV_PRESS_COLBENCH_MODAL_GPU` or `--gpu-spec`) |
| Agent attention | `flash_attention_3` |
| Simulator attention | `vllm_triton` |
| `cot` | `False` (DeepSeek-R1-Distill already reasons inside its chat template) |
| `early_stop_on_pass` | `True` |
| `max_turns` | `10` |
| `max_questions_before_submit` | `9` (last turn forced-submit) |
| `global_budget` | `4500` |
| `local_budget` | `4096` |

## Pull Results Back

Modal stores outputs in `kvpress-colbench-results`. The app writes each run as
a directory containing:

- `predictions.jsonl`
- `predictions.csv`
- `metrics.json`
- `config.yaml`

Use the Modal dashboard volume browser, or a small one-off Modal function, to
download from:

```text
/root/kvpress/evaluation/results_colbench_live_modal
```

## Notes

- Run Modal commands from `kvpress/`, not the parent `TurnKV/` directory.
- `network_isolation=auto` tries `unshare -n` inside the worker and falls back
  if the container cannot create a network namespace.
- Synthetic rows after early stop are marked `metric_excluded=True`; metrics are
  computed only from turns that actually ran.
- The H200 fits agent BF16 weights, Gemma4-26B-A4B BF16 weights, vLLM KV cache,
  and the agent's live-loop cache on one GPU. If you override the GPU spec to
  a smaller card, also reduce `--feedback-vllm-max-model-len` and
  `--feedback-vllm-gpu-memory-utilization`.
