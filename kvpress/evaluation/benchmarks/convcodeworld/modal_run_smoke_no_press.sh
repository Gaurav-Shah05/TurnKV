#!/usr/bin/env bash
# Smoke run #0 (full KV): no press / no KV-cache compression on the 228-task
# 20% tune split, fanned out to 10 detached Modal containers (one per shard,
# ~23 tasks each on its own H100). Same model + prompt config as the SnapKV
# smoke runs so this is the uncompressed control.
#
# Profile: ypatlola (override via MODAL_PROFILE=...).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."

MODAL_PROFILE="${MODAL_PROFILE:-ypatlola}"
NUM_SHARDS="${NUM_SHARDS:-10}"
# BENCHMARK_MODE: "live" uses the model's own generated code as prior context
# and live feedback; "static" uses teacher-forced reference prior code/feedback.
BENCHMARK_MODE="${BENCHMARK_MODE:-live}"
# Kept for config parity with the compressed smoke runs. With no_press these
# budgets do not trigger eviction.
GLOBAL_BUDGET="${GLOBAL_BUDGET:-4096}"
LOCAL_BUDGET="${LOCAL_BUDGET:-2048}"
# Optional label baked into output_subdir + log filename so multiple runs
# with different configs don't collide on the Modal volume.
CONFIG_LABEL="${CONFIG_LABEL:-}"
DEFAULT_SHARD_STEM="tune_20pct_seed42"
SHARD_STEM="${SHARD_STEM:-$DEFAULT_SHARD_STEM}"

usage() {
  echo "Usage: $(basename "$0") [--split 228|100|<split-stem>]" >&2
  echo "  228 -> tune_20pct_seed42 (228 tasks, ~23 per shard)" >&2
  echo "  100     -> tune_100tasks_seed42 (100 tasks, 10 per shard)" >&2
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

: "${HF_TOKEN:?HF_TOKEN must be set before running this smoke test}"
MODAL_BIN="${MODAL_BIN:-$(command -v modal || true)}"
if [[ -z "$MODAL_BIN" ]]; then
  echo "Modal CLI not found. Install Modal or set MODAL_BIN=/path/to/modal before running this smoke test." >&2
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
  RUN_TAG="no_press_${RUN_LABEL}_${BENCHMARK_MODE}_smoke"
  LOG_PREFIX="smoke_no_press_${RUN_LABEL}_${BENCHMARK_MODE}"
else
  RUN_TAG="no_press_${BENCHMARK_MODE}_smoke"
  LOG_PREFIX="smoke_no_press_${BENCHMARK_MODE}"
fi
LOG_DIR="$LOG_ROOT/${RUN_TAG}_${RUN_TS}"
mkdir -p "$LOG_DIR"
INDEX_FILE="$LOG_DIR/index.txt"
echo "# no_press smoke - mode=$BENCHMARK_MODE split=$SHARD_STEM label='$CONFIG_LABEL' profile=$MODAL_PROFILE budget=$GLOBAL_BUDGET/$LOCAL_BUDGET - $RUN_TS" > "$INDEX_FILE"

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
    --feedback-config CF_EF_UNIT_SNF
    --press-name no_press
    --compression-ratio 0.0
    --global-budget "$GLOBAL_BUDGET"
    --local-budget "$LOCAL_BUDGET"
    --max-turns 10
    --code-generation-until-eos
    --num-eval-examples 0
    --task-ids "@$shard_json_container"
    --output-subdir "$output_subdir"
    --cot
    --require-flashdecode
    --log-level INFO
  )
  if [[ "$BENCHMARK_MODE" == "live" ]]; then
    shard_args+=(--full-kv-cache)
  fi

  echo "[shard $shard/$((NUM_SHARDS - 1))] dispatching detached run -> $log_file"
  # Background the modal CLI: with -d, the function call registers on Modal
  # within a few seconds of image upload, after which the CLI just streams
  # logs uselessly until task completion. Backgrounding lets the dispatcher
  # loop continue immediately; the function call survives even if we kill
  # the CLI later. nohup + redirected stdin keeps it alive past the parent
  # shell's exit.
  (
    nohup env \
      MODAL_PROFILE="$MODAL_PROFILE" \
      HF_TOKEN="$HF_TOKEN" \
      PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}" \
      PYTHONUTF8="${PYTHONUTF8:-1}" \
      MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}" \
      MSYS2_ARG_CONV_EXCL="${MSYS2_ARG_CONV_EXCL:-*}" \
      "$MODAL_BIN" run -d "${shard_args[@]}" \
      > "$log_file" 2>&1 < /dev/null &
    disown $! 2>/dev/null || true
  )
  # Tiny pause so successive dispatches don't race the gRPC channel setup;
  # Modal de-dupes image builds across concurrent calls so no extra cost.
  sleep 3

  echo "shard=$shard json=$shard_json log=$log_file output_subdir=$output_subdir" >> "$INDEX_FILE"
done

echo "Index of all shards: $INDEX_FILE"
