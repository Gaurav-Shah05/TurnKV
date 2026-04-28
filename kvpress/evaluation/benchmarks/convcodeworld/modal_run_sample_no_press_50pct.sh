#!/usr/bin/env bash
# Sample run: shard 0 of the 50% tune split (57 tasks) with full KV cache
# (no_press). Used to surface and fix executor environment issues before
# dispatching the full 10-shard 50% run.
#
# Run from kvpress/:
#   MODAL_HF_SECRET_NAME=hf-secret \
#     evaluation/benchmarks/convcodeworld/modal_run_sample_no_press_50pct.sh
#
# After the run completes, retrieve predictions.jsonl with:
#   modal volume get kvpress-convcodeworld-results \
#     sample_50pct_no_press_fullkv_shard0/predictions.jsonl \
#     ./predictions_sample_shard0.jsonl
#
# Profile: ypatlola (override via MODAL_PROFILE=...).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."

MODAL_PROFILE="${MODAL_PROFILE:-pganesh}"
MODAL_GPU_SPEC="${MODAL_GPU_SPEC:-H200}"
FEEDBACK_MODEL="${FEEDBACK_MODEL:-google/gemma-4-26B-A4B-it}"
FEEDBACK_ATTN_IMPLEMENTATION="${FEEDBACK_ATTN_IMPLEMENTATION:-vllm_triton}"
FEEDBACK_VLLM_MAX_MODEL_LEN="${FEEDBACK_VLLM_MAX_MODEL_LEN:-32768}"
FEEDBACK_VLLM_GPU_MEMORY_UTILIZATION="${FEEDBACK_VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES="${FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES:-}"
GLOBAL_BUDGET="${GLOBAL_BUDGET:-4096}"
LOCAL_BUDGET="${LOCAL_BUDGET:-2048}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-sample_50pct_no_press_fullkv_shard0}"

SHARD_JSON="$script_dir/splits/shards/tune_50pct_seed42_shard_0_of_10.json"
CONTAINER_SHARD_JSON="/root/kvpress/evaluation/benchmarks/convcodeworld/splits/shards/tune_50pct_seed42_shard_0_of_10.json"

if [[ ! -f "$SHARD_JSON" ]]; then
  echo "Shard JSON missing: $SHARD_JSON" >&2
  echo "  Run build_shards.py first (see Phase 1 of the 50% run plan)." >&2
  exit 1
fi

MODAL_BIN="${MODAL_BIN:-$(command -v modal || true)}"
if [[ -z "$MODAL_BIN" ]]; then
  echo "Modal CLI not found. Install Modal or set MODAL_BIN=/path/to/modal." >&2
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${MODAL_HF_SECRET_NAME:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "No Hugging Face credential found. Set HF_TOKEN, HUGGING_FACE_HUB_TOKEN, MODAL_HF_SECRET_NAME, or run huggingface-cli login." >&2
  exit 1
fi

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}"
export MSYS2_ARG_CONV_EXCL="${MSYS2_ARG_CONV_EXCL:-*}"
# Downgrade the build-validation check from error to warning so that
# pre-existing LF/CRLF line-ending discrepancies on Windows don't block the run.
export MODAL_BUILD_VALIDATION="${MODAL_BUILD_VALIDATION:-warn}"

LOG_ROOT="$script_dir/../../../../.modal_diag"
mkdir -p "$LOG_ROOT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/sample_no_press_50pct_shard0_${RUN_TS}.log"

echo "Dispatching sample run: shard 0 of tune_50pct_seed42 (57 tasks), full KV cache"
echo "  output_subdir : $OUTPUT_SUBDIR"
echo "  log           : $LOG_FILE"
echo "  profile       : $MODAL_PROFILE"
echo "  gpu           : $MODAL_GPU_SPEC"

env \
  MODAL_PROFILE="$MODAL_PROFILE" \
  KV_PRESS_CONVCODEWORLD_MODAL_GPU="$MODAL_GPU_SPEC" \
  HF_TOKEN="${HF_TOKEN:-}" \
  HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
  MODAL_HF_SECRET_NAME="${MODAL_HF_SECRET_NAME:-}" \
  PYTHONIOENCODING="${PYTHONIOENCODING}" \
  PYTHONUTF8="${PYTHONUTF8}" \
  MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV}" \
  MSYS2_ARG_CONV_EXCL="${MSYS2_ARG_CONV_EXCL}" \
  "$MODAL_BIN" run -d \
  evaluation/benchmarks/convcodeworld/modal_app.py::run_convcodeworld_live \
  --benchmark-mode live \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attn-implementation flash_attention_3 \
  --feedback-model "$FEEDBACK_MODEL" \
  --feedback-attn-implementation "$FEEDBACK_ATTN_IMPLEMENTATION" \
  --feedback-vllm-cuda-visible-devices "$FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES" \
  --feedback-vllm-max-model-len "$FEEDBACK_VLLM_MAX_MODEL_LEN" \
  --feedback-vllm-gpu-memory-utilization "$FEEDBACK_VLLM_GPU_MEMORY_UTILIZATION" \
  --feedback-config CF_EF_UNIT_SNF \
  --press-name no_press \
  --compression-ratio 0.0 \
  --global-budget "$GLOBAL_BUDGET" \
  --local-budget "$LOCAL_BUDGET" \
  --full-kv-cache \
  --code-generation-until-eos \
  --cot \
  --max-turns 10 \
  --num-eval-examples 0 \
  --task-ids "@$CONTAINER_SHARD_JSON" \
  --output-subdir "$OUTPUT_SUBDIR" \
  --require-flashdecode \
  --log-level INFO \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Run dispatched. Monitor on the Modal dashboard."
echo ""
echo "When complete, retrieve predictions.jsonl with:"
echo "  modal volume get kvpress-convcodeworld-results \\"
echo "    ${OUTPUT_SUBDIR}/predictions.jsonl \\"
echo "    ./predictions_sample_shard0.jsonl"
