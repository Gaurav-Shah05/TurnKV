# ConvCodeWorld Modal Setup

This runs ConvCodeWorld on Modal in either `--benchmark-mode live` or
`--benchmark-mode static`. The default live-loop mode uses
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` for code generation,
`google/gemma-4-26B-A4B-it` for LLM-simulated verbal feedback, CoT enabled, and
early-stop-on-pass enabled. The Modal worker requests one H200 GPU by default and uses
`flash_attention_3` for the code model by default. The feedback model defaults
to a local vLLM server with `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, so feedback
prefill and decode both use vLLM Triton attention.

## Local Setup

Run these from the kvpress repo root:

```bash
cd kvpress
python -m pip install "modal>=1.4.0,<2"
modal token new
modal profile current
```

This checkout already has a Modal CLI available in the active environment; the
local profile observed during setup was `ypatlola`.

## Hugging Face Access

Gemma and Llama-family model weights may require a Hugging Face token with model
access. Use one of these:

```bash
# Option A: local shell env, forwarded into Modal at app definition time
export HF_TOKEN=hf_...

# Option B: repo-local .env parsed by modal_app.py
printf 'HF_TOKEN=hf_...\n' > .env

# Option C: named Modal secret
modal secret create hf-secret HF_TOKEN=hf_...
export MODAL_HF_SECRET_NAME=hf-secret
```

## Smoke Run

Build the image and run one task with one press:

```bash
cd kvpress
modal run evaluation/benchmarks/convcodeworld/modal_app.py::main \
    --benchmark-mode live \
    --press-names snapkv \
    --num-eval-examples 1
```

The first run builds the image, installs eval dependencies, downloads model and
dataset files, and writes results under the Modal volume
`kvpress-convcodeworld-results`.

The FlashAttention-3 build is the slow layer. The image builds a reduced
BF16-only Hopper FA3 wheel from `Dao-AILab/flash-attention` before copying the
repo, stores it in `/opt/fa3-wheelhouse`, and installs it into the runtime venv.
Paged-KV, split-KV, backward, FP16, FP8, and SM80 kernels are disabled to keep
the Modal build reusable and small for this benchmark path. Modal can reuse
that layer across code changes as long as the CUDA base image, Torch seed
version, vLLM wheel, and FA3 ref stay unchanged. The dependency layer installs
the repo's runtime/eval requirements after the vLLM pre-release wheel.
Transformers is then
installed from a pinned upstream GitHub commit because the released wheel tested
during setup did not recognize Gemma4 yet. The source package is installed with
`--no-deps` after repo copy so those pins are preserved.
The image also installs the vLLM pre-release CUDA wheel and the live runner
starts an OpenAI-compatible local server for Gemma 4 feedback when
`--feedback-attn-implementation vllm_triton` is selected.

## Sharded Smoke Scripts

The sharded smoke scripts launch one detached Modal function per shard JSON.
Each function requests one H200 by default, and the local dispatcher backgrounds
each `modal run -d` by default, so a plain `--split 100` run submits all 10
shards and lets Modal place them on separate GPUs when capacity is available.

```bash
cd kvpress

CONFIG_LABEL=h200_gemma4_all10 \
evaluation/benchmarks/convcodeworld/modal_run_smoke_no_press.sh \
    --split 100

CONFIG_LABEL=h200_gemma4_all10 \
evaluation/benchmarks/convcodeworld/modal_run_smoke_baseline_snapkv.sh \
    --split 100

CONFIG_LABEL=h200_gemma4_all10 \
evaluation/benchmarks/convcodeworld/modal_run_smoke_turnkv_snapkv.sh \
    --split 100
```

Shared script flags:

| Flag | Meaning |
|---|---|
| `--split 228` / `--split 100` | Selects the prebuilt 228-task or 100-task split shards. |
| `--gpu-spec H200` | Sets the Modal GPU request for each shard. This maps to `KV_PRESS_CONVCODEWORLD_MODAL_GPU` inside each `modal run`. |
| `--background-modal-cli` | Starts all local `modal run -d` commands in the background, so all shards are submitted immediately. This is the default. |
| `--foreground-modal-cli` / `--no-background-modal-cli` | Waits for each local Modal CLI process before submitting the next shard. Use this when rebuilding images. |

The scripts still accept environment defaults such as `MODAL_GPU_SPEC=...` and
`BACKGROUND_MODAL_CLI=false`, but `H200` and background dispatch are the default
script behavior. Logs are written under `.modal_diag/`, and each run also writes an
`index.txt` mapping shards to local log files and Modal output subdirectories.

## Base Press Run

```bash
cd kvpress
modal run evaluation/benchmarks/convcodeworld/modal_app.py::main \
    --benchmark-mode live \
    --press-names snapkv,streaming_llm,expected_attention \
    --compression-ratio 0.5 \
    --snapkv-window-size 64 \
    --snapkv-kernel-size 5 \
    --streaming-llm-n-sink 4 \
    --expected-attention-n-future-positions 512 \
    --expected-attention-n-sink 4 \
    --expected-attention-use-covariance \
    --expected-attention-use-vnorm \
    --expected-attention-epsilon 0.01 \
    --feedback-config CF_EF_UNIT_SNF \
    --fraction 0.1 \
    --num-eval-examples -1 \
    --local-budget 4096
```

Defaults:

| Setting | Value |
|---|---|
| benchmark mode | `live` (`static` is also supported) |
| model | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| feedback model | `google/gemma-4-26B-A4B-it` |
| GPU | one H200 via Modal `gpu="H200"` by default; override direct Modal runs with `KV_PRESS_CONVCODEWORLD_MODAL_GPU` or smoke scripts with `--gpu-spec` |
| code attention implementation | `flash_attention_3` |
| feedback attention implementation | `vllm_triton` |
| `cot` | `True` |
| `early_stop_on_pass` | `True` |
| `max_turns` | `10` |
| `global_budget` | `4500` |
| `local_budget` | `4096` |
| verbal simulator | Gemma 4 26B-A4B via local vLLM server, separate fresh KV cache |

Press hyperparameters exposed by the top-level Modal command:

| Press | Flags |
|---|---|
| all scorer presses | `--compression-ratio` |
| ThinK/composed ThinK | `--key-channel-compression-ratio` |
| DMS | `--threshold` |
| SnapKV/PyramidKV | `--snapkv-window-size`, `--snapkv-kernel-size` |
| StreamingLLM | `--streaming-llm-n-sink` |
| ExpectedAttention | `--expected-attention-n-future-positions`, `--expected-attention-n-sink`, `--expected-attention-use-covariance`, `--no-expected-attention-use-covariance`, `--expected-attention-use-vnorm`, `--no-expected-attention-use-vnorm`, `--expected-attention-epsilon` |
| TurnAwareGlobalPress | `--alpha-floor`, `--alpha-anchor`, `--alpha-loyalty` |
| RoleBoundaryAnchorPress | `--anchor-beta` |
| TurnFloorPress | `--floor-gamma`, `--alpha-floor-len`, `--min-floor-tokens` |
| LoyaltyPress | `--loyalty-top-p` |
| answer-suffix decode press | `--decode-compression-interval`, `--decode-hidden-states-buffer-size` |

`--local-budget` is also exposed by the top-level Modal command and controls
the answer-suffix decode compression target.

Use `--attn-implementation eager` only as a debugging fallback. The default FA3
path forces BF16 model loading to match the reduced FA3 wheel. Use
`--feedback-attn-implementation` to override the feedback model attention path;
the default is `vllm_triton` for Gemma 4 26B-A4B feedback.

The H200 memory budget is intended to fit the code-model weights, Gemma 4 BF16
weights, vLLM KV cache, and the live-loop code cache on one GPU. If you override
the GPU spec to a smaller card, also reduce `--feedback-vllm-max-model-len` and
`--feedback-vllm-gpu-memory-utilization`.

## Pull Results Back

Modal stores outputs in `kvpress-convcodeworld-results`. The app writes each run
as a directory containing:

- `predictions.jsonl`
- `predictions.csv`
- `metrics.json`
- `config.yaml`

Use the Modal dashboard volume browser, or a small one-off Modal function, to
download result files from:

```text
/root/kvpress/evaluation/results_convcodeworld_live_modal
```

## Notes

- Run Modal commands from `kvpress/`, not the parent `TurnKV/` directory.
- `network_isolation=auto` tries `unshare -n` inside the worker and falls back
  if the container cannot create a network namespace.
- Synthetic rows after early stop are marked `metric_excluded=True`; metrics are
  computed only from turns that actually ran.
