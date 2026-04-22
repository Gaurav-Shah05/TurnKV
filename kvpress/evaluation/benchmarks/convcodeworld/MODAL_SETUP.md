# ConvCodeWorld Modal Setup

This runs the live-loop ConvCodeWorld benchmark on Modal with
`meta-llama/Meta-Llama-3.1-8B-Instruct`, CoT enabled, LLM-simulated verbal
feedback, and early-stop-on-pass enabled.

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

Llama 3.1 is gated, so the Modal worker needs a Hugging Face token with model
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
    --press-names snapkv \
    --num-eval-examples 1
```

The first run builds the image, installs eval dependencies, downloads model and
dataset files, and writes results under the Modal volume
`kvpress-convcodeworld-results`.

## Base Press Run

```bash
cd kvpress
modal run evaluation/benchmarks/convcodeworld/modal_app.py::main \
    --press-names snapkv,streaming_llm,expected_attention \
    --compression-ratio 0.5 \
    --feedback-config CF_EF_UNIT_SNF \
    --num-eval-examples 10
```

Defaults:

| Setting | Value |
|---|---|
| model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `cot` | `True` |
| `early_stop_on_pass` | `True` |
| `max_turns` | `10` |
| `global_budget` | `4500` |
| `local_budget` | `4096` |
| verbal simulator | same loaded Llama model, separate fresh KV cache |

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
