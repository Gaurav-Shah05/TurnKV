#!/usr/bin/env bash
# Smoke run #1 (baseline): plain SnapKV on the 228-task 20% tune split,
# fanned out to 10 detached Modal containers (one per shard, ~23 tasks each
# on its own H100). Same model + prompt config as the turnkv smoke run so
# the two are directly comparable.
#
# Profile: gauravmshah2004 (override via MODAL_PROFILE=...).
# 'Win' is defined as: turnkv_snapkv >= baseline Pass@1 within ~1 pp on
# this 228-task split, with comparable wall-clock.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."

MODAL_PROFILE="${MODAL_PROFILE:-gauravmshah2004}"
NUM_SHARDS="${NUM_SHARDS:-10}"
SHARD_DIR="$script_dir/splits/shards"
SHARD_STEM="tune_20pct_seed42"

# Modal mounts REPO_ROOT (kvpress/) at /root/kvpress; the shard JSONs ship
# with the image, so use the container-side path directly to avoid any
# launcher-vs-container path-translation drift.
CONTAINER_SHARD_DIR="/root/kvpress/evaluation/benchmarks/convcodeworld/splits/shards"

if [[ ! -d "$SHARD_DIR" ]]; then
  echo "Shard dir missing: $SHARD_DIR" >&2
  echo "  Build shards first: python evaluation/benchmarks/convcodeworld/scripts/build_shards.py --input evaluation/benchmarks/convcodeworld/splits/tune_20pct_seed42.json --num-shards $NUM_SHARDS" >&2
  exit 1
fi

: "${HF_TOKEN:?HF_TOKEN must be set before running this smoke test}"

# When running through the Windows-side modal.exe, the Modal CLI tries to
# write Unicode to a charmap-only stdout and dies. Force utf-8 to be safe;
# this is harmless under WSL/Linux too.
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUTF8="${PYTHONUTF8:-1}"

LOG_DIR="$script_dir/../../../../.modal_diag"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
INDEX_FILE="$LOG_DIR/smoke_baseline_snapkv_${RUN_TS}_index.txt"
echo "# baseline_snapkv smoke - profile=$MODAL_PROFILE - $RUN_TS" > "$INDEX_FILE"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_json="$SHARD_DIR/${SHARD_STEM}_shard_${shard}_of_${NUM_SHARDS}.json"
  if [[ ! -f "$shard_json" ]]; then
    echo "Missing shard JSON: $shard_json" >&2
    exit 1
  fi
  shard_json_container="$CONTAINER_SHARD_DIR/${SHARD_STEM}_shard_${shard}_of_${NUM_SHARDS}.json"
  output_subdir="baseline_snapkv_smoke_${RUN_TS}/shard_${shard}_of_${NUM_SHARDS}"
  log_file="$LOG_DIR/smoke_baseline_snapkv_${RUN_TS}_shard${shard}.log"

  shard_args=(
    evaluation/benchmarks/convcodeworld/modal_app.py::run_convcodeworld_live
    --benchmark-mode static
    --model meta-llama/Meta-Llama-3.1-8B-Instruct
    --attn-implementation flash_attention_3
    --feedback-config CF_EF_UNIT_SNF
    --press-name snapkv
    --compression-ratio 0.0
    --global-budget 4096
    --local-budget 2048
    --max-turns 10
    --max-new-tokens 1024
    --num-eval-examples 0
    --task-ids "@$shard_json_container"
    --output-subdir "$output_subdir"
    --cot
    --require-flashdecode
    --log-level INFO
  )

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
      "${MODAL_BIN:-/home/anaconda3/bin/modal}" run -d "${shard_args[@]}" \
      > "$log_file" 2>&1 < /dev/null &
    disown $! 2>/dev/null || true
  )
  # Tiny pause so successive dispatches don't race the gRPC channel setup;
  # Modal de-dupes image builds across concurrent calls so no extra cost.
  sleep 3

  echo "shard=$shard json=$shard_json log=$log_file output_subdir=$output_subdir" >> "$INDEX_FILE"
done

echo "Index of all shards: $INDEX_FILE"
