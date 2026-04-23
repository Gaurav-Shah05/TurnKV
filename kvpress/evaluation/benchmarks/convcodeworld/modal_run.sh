#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."

common_args=(
  evaluation/benchmarks/convcodeworld/modal_app.py::run_convcodeworld_live
  --benchmark-mode live
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  --feedback-model google/gemma-3-4b-it
  --attn-implementation flash_attention_3
  --feedback-attn-implementation flash_attention_3
  --feedback-config CF_EF_UNIT_SNF
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
