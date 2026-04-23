# ConvCodeWorld Modal Setup

This runs ConvCodeWorld on Modal in either `--benchmark-mode live` or
`--benchmark-mode static`. The default live-loop mode uses
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` for code generation,
`google/gemma-4-E2B-it` for LLM-simulated verbal feedback, CoT enabled, and
early-stop-on-pass enabled. The Modal worker requests exact H100 GPUs and uses
`flash_attention_3` for the code model by default. The feedback model defaults
to `sdpa` for compatibility with Gemma attention head dimensions.

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
BF16-only H100 FA3 wheel from `Dao-AILab/flash-attention` before copying the
repo, stores it in `/opt/fa3-wheelhouse`, and installs it into the runtime venv.
Paged-KV, split-KV, backward, FP16, FP8, and SM80 kernels are disabled to keep
the Modal build reusable and small for this benchmark path. Modal can reuse
that layer across code changes as long as the CUDA base image, Torch pin, and
FA3 ref stay unchanged. The dependency layer installs the repo's runtime/eval
requirements explicitly with Torch pinned to 2.8.0. Transformers is then
installed from a pinned upstream GitHub commit because the released wheel tested
during setup did not recognize Gemma4 yet. The source package is installed with
`--no-deps` after repo copy so those pins are preserved.

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
| feedback model | `google/gemma-4-E2B-it` |
| GPU | exact H100 via Modal `gpu="H100!"` |
| code attention implementation | `flash_attention_3` |
| feedback attention implementation | `sdpa` |
| `cot` | `True` |
| `early_stop_on_pass` | `True` |
| `max_turns` | `10` |
| `global_budget` | `4500` |
| `local_budget` | `4096` |
| verbal simulator | separate Gemma 4 E2B model, separate fresh KV cache |

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
the default is `sdpa` for Gemma compatibility.

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
