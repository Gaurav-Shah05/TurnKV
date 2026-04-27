#!/usr/bin/env bash
# Single detached full-KV ColBench (Backend) Modal run. Useful for sanity
# checking the image build + the live-loop wiring before launching the sharded
# smoke runs. Mirrors convcodeworld/modal_run.sh.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."

common_args=(
  evaluation/benchmarks/colbench/modal_app.py::run_colbench_live
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
)

MODAL_HF_SECRET_NAME=hf-secret modal run -d "${common_args[@]}"
