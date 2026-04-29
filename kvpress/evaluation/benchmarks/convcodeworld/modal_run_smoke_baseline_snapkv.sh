#!/usr/bin/env bash
# Smoke run #1 (baseline): plain SnapKV on the 228-task 20% tune split,
# fanned out to 10 detached Modal containers (one per shard, ~23 tasks each
# on its own H200). Same model + prompt config as the turnkv smoke run so
# the two are directly comparable.
#
# Profile: gauravmshah2004 (override via MODAL_PROFILE=...).
# 'Win' is defined as: turnkv_snapkv >= baseline Pass@1 within ~1 pp on
# this 228-task split, with comparable wall-clock.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."

MODAL_PROFILE="${MODAL_PROFILE:-ypatlola}"
NUM_SHARDS="${NUM_SHARDS:-10}"
# BENCHMARK_MODE: "static" (teacher-forced reference prior code per iter, ADR 001 §8)
# or "live" (model's own generated code is the prior context per iter; needs the
# feedback model + executor sandbox).
BENCHMARK_MODE="${BENCHMARK_MODE:-live}"
# Cache budgets — override for budget sweeps.
GLOBAL_BUDGET="${GLOBAL_BUDGET:-2048}"
LOCAL_BUDGET="${LOCAL_BUDGET:-4096}"
MODAL_GPU_SPEC="${MODAL_GPU_SPEC:-H200}"
FEEDBACK_MODEL="${FEEDBACK_MODEL:-google/gemma-4-26B-A4B-it}"
FEEDBACK_ATTN_IMPLEMENTATION="${FEEDBACK_ATTN_IMPLEMENTATION:-vllm_triton}"
FEEDBACK_VLLM_MAX_MODEL_LEN="${FEEDBACK_VLLM_MAX_MODEL_LEN:-32768}"
FEEDBACK_VLLM_GPU_MEMORY_UTILIZATION="${FEEDBACK_VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES="${FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES:-}"
BACKGROUND_MODAL_CLI="${BACKGROUND_MODAL_CLI:-true}"
# Optional label baked into output_subdir + log filename so multiple runs
# with different budgets / configs don't collide on the Modal volume.
CONFIG_LABEL="${CONFIG_LABEL:-baseline_snapkv_global2048_local1536_maxtokens2048_compressratio0.5_split228}"
DEFAULT_SHARD_STEM="tune_20pct_seed42"
SHARD_STEM="${SHARD_STEM:-$DEFAULT_SHARD_STEM}"

usage() {
  echo "Usage: $(basename "$0") [--split 228|100|<split-stem>] [--gpu-spec H200] [--background-modal-cli|--foreground-modal-cli]" >&2
  echo "  228 -> tune_20pct_seed42 (228 tasks, ~23 per shard)" >&2
  echo "  100     -> tune_100tasks_seed42 (100 tasks, 10 per shard)" >&2
  echo "  --gpu-spec sets the Modal GPU request for each shard (default: ${MODAL_GPU_SPEC})" >&2
  echo "  --background-modal-cli submits all shards without waiting for each Modal CLI process (default: ${BACKGROUND_MODAL_CLI})" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      SHARD_STEM="$2"
      shift 2
      ;;
    --split=*)
      SHARD_STEM="${1#--split=}"
      shift
      ;;
    --gpu-spec)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      MODAL_GPU_SPEC="$2"
      shift 2
      ;;
    --gpu-spec=*)
      MODAL_GPU_SPEC="${1#--gpu-spec=}"
      shift
      ;;
    --background-modal-cli)
      BACKGROUND_MODAL_CLI=true
      shift
      ;;
    --background-modal-cli=*)
      BACKGROUND_MODAL_CLI="${1#--background-modal-cli=}"
      shift
      ;;
    --foreground-modal-cli|--no-background-modal-cli)
      BACKGROUND_MODAL_CLI=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "$SHARD_STEM" in
  228|20pct|tune20)
    SHARD_STEM="$DEFAULT_SHARD_STEM"
    ;;
  100|100tasks|small|tune100)
    SHARD_STEM="tune_100tasks_seed42"
    ;;
esac

case "$BACKGROUND_MODAL_CLI" in
  true|false)
    ;;
  *)
    echo "Invalid --background-modal-cli value: $BACKGROUND_MODAL_CLI (expected true or false)" >&2
    exit 1
    ;;
esac

if [[ -z "$MODAL_GPU_SPEC" ]]; then
  echo "--gpu-spec must not be empty" >&2
  exit 1
fi
SHARD_DIR="$script_dir/splits/shards"
SPLIT_FILE="$script_dir/splits/${SHARD_STEM}.json"

# Modal mounts REPO_ROOT (kvpress/) at /root/kvpress; the shard JSONs ship
# with the image, so use the container-side path directly to avoid any
# launcher-vs-container path-translation drift.
CONTAINER_SHARD_DIR="/root/kvpress/evaluation/benchmarks/convcodeworld/splits/shards"

if [[ ! -d "$SHARD_DIR" ]]; then
  echo "Shard dir missing: $SHARD_DIR" >&2
  echo "  Build shards first: python evaluation/benchmarks/convcodeworld/scripts/build_shards.py --input evaluation/benchmarks/convcodeworld/splits/${SHARD_STEM}.json --num-shards $NUM_SHARDS" >&2
  exit 1
fi
if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "Split file missing: $SPLIT_FILE" >&2
  exit 1
fi

MODAL_BIN="${MODAL_BIN:-$(command -v modal || true)}"
if [[ -z "$MODAL_BIN" ]]; then
  echo "Modal CLI not found. Install Modal or set MODAL_BIN=/path/to/modal before running this smoke test." >&2
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${MODAL_HF_SECRET_NAME:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "No Hugging Face credential found. Set HF_TOKEN, HUGGING_FACE_HUB_TOKEN, MODAL_HF_SECRET_NAME, or run huggingface-cli login." >&2
  exit 1
fi

# When running through the Windows-side modal.exe, the Modal CLI tries to
# write Unicode to a charmap-only stdout and dies. Force utf-8 to be safe;
# this is harmless under WSL/Linux too.
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUTF8="${PYTHONUTF8:-1}"

LOG_ROOT="$script_dir/../../../../.modal_diag"
mkdir -p "$LOG_ROOT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="$CONFIG_LABEL"
if [[ "$SHARD_STEM" != "$DEFAULT_SHARD_STEM" ]]; then
  RUN_LABEL="${RUN_LABEL:+${RUN_LABEL}_}${SHARD_STEM}"
fi
if [[ -n "$RUN_LABEL" ]]; then
  RUN_TAG="baseline_snapkv_${RUN_LABEL}_${BENCHMARK_MODE}_smoke"
  LOG_PREFIX="smoke_baseline_snapkv_${RUN_LABEL}_${BENCHMARK_MODE}"
else
  RUN_TAG="baseline_snapkv_${BENCHMARK_MODE}_smoke"
  LOG_PREFIX="smoke_baseline_snapkv_${BENCHMARK_MODE}"
fi
LOG_DIR="$LOG_ROOT/${RUN_TAG}_${RUN_TS}"
mkdir -p "$LOG_DIR"
INDEX_FILE="$LOG_DIR/index.txt"
echo "# baseline_snapkv smoke - mode=$BENCHMARK_MODE split=$SHARD_STEM label='$CONFIG_LABEL' profile=$MODAL_PROFILE gpu=$MODAL_GPU_SPEC feedback=$FEEDBACK_MODEL/$FEEDBACK_ATTN_IMPLEMENTATION budget=$GLOBAL_BUDGET/$LOCAL_BUDGET - $RUN_TS" > "$INDEX_FILE"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_json="$SHARD_DIR/${SHARD_STEM}_shard_${shard}_of_${NUM_SHARDS}.json"
  if [[ ! -f "$shard_json" ]]; then
    echo "Missing shard JSON: $shard_json" >&2
    exit 1
  fi
  shard_json_container="$CONTAINER_SHARD_DIR/${SHARD_STEM}_shard_${shard}_of_${NUM_SHARDS}.json"
  output_subdir="${RUN_TAG}_${RUN_TS}/shard_${shard}_of_${NUM_SHARDS}"
  log_file="$LOG_DIR/${LOG_PREFIX}_shard${shard}.log"

  shard_args=(
    evaluation/benchmarks/convcodeworld/modal_app.py::run_convcodeworld_live
    --benchmark-mode "$BENCHMARK_MODE"
    --model meta-llama/Meta-Llama-3.1-8B-Instruct
    --attn-implementation flash_attention_3
    --feedback-model "$FEEDBACK_MODEL"
    --feedback-attn-implementation "$FEEDBACK_ATTN_IMPLEMENTATION"
    --feedback-vllm-cuda-visible-devices "$FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES"
    --feedback-vllm-max-model-len "$FEEDBACK_VLLM_MAX_MODEL_LEN"
    --feedback-vllm-gpu-memory-utilization "$FEEDBACK_VLLM_GPU_MEMORY_UTILIZATION"
    --feedback-config CF_EF_UNIT_SNF
    --press-name snapkv
    --compression-ratio 0.5
    --global-budget "$GLOBAL_BUDGET"
    --local-budget "$LOCAL_BUDGET"
    --max-turns 10
    --max-new-tokens 4096
    --num-eval-examples 0
    --task-ids "@$shard_json_container"
    --output-subdir "$output_subdir"
    --cot
    --require-flashdecode
    --log-level INFO
  )

  echo "[shard $shard/$((NUM_SHARDS - 1))] dispatching detached run -> $log_file"
  modal_env=(
    env
    MODAL_PROFILE="$MODAL_PROFILE"
    KV_PRESS_CONVCODEWORLD_MODAL_GPU="$MODAL_GPU_SPEC"
    HF_TOKEN="${HF_TOKEN:-}"
    HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
    MODAL_HF_SECRET_NAME="${MODAL_HF_SECRET_NAME:-}"
    PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
    PYTHONUTF8="${PYTHONUTF8:-1}"
    MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}"
    MSYS2_ARG_CONV_EXCL="${MSYS2_ARG_CONV_EXCL:-*}"
  )
  if [[ "$BACKGROUND_MODAL_CLI" == "true" ]]; then
    (
      nohup "${modal_env[@]}" "$MODAL_BIN" run -d "${shard_args[@]}" \
        > "$log_file" 2>&1 < /dev/null &
      disown $! 2>/dev/null || true
    )
    sleep 3
  else
    "${modal_env[@]}" "$MODAL_BIN" run -d "${shard_args[@]}" \
      > "$log_file" 2>&1
  fi

  echo "shard=$shard json=$shard_json log=$log_file output_subdir=$output_subdir" >> "$INDEX_FILE"
done

echo "Index of all shards: $INDEX_FILE"
