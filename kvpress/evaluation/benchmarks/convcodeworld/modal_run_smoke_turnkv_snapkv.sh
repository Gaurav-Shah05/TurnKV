#!/usr/bin/env bash
# Smoke run #2 (turnkv): TurnKV-wrapped SnapKV with all 3 policies on,
# cherry-picked alphas (alpha=(1,1,1), floor_gamma=0.1, anchor_beta=0.25,
# loyalty_top_p=0.25, loyalty_update_every=5).
#
# Fanned out to 10 detached Modal containers (one per shard, ~23 tasks
# each on its own H100). Profile: docmanish2312 (override via MODAL_PROFILE=...).
#
# Compared head-to-head against modal_run_smoke_baseline_snapkv.sh on the
# same 228-task 20% tune split.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."

MODAL_PROFILE="${MODAL_PROFILE:-docmanish2312}"
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
INDEX_FILE="$LOG_DIR/smoke_turnkv_snapkv_${RUN_TS}_index.txt"
echo "# turnkv_snapkv smoke - profile=$MODAL_PROFILE - $RUN_TS" > "$INDEX_FILE"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_json="$SHARD_DIR/${SHARD_STEM}_shard_${shard}_of_${NUM_SHARDS}.json"
  if [[ ! -f "$shard_json" ]]; then
    echo "Missing shard JSON: $shard_json" >&2
    exit 1
  fi
  shard_json_container="$CONTAINER_SHARD_DIR/${SHARD_STEM}_shard_${shard}_of_${NUM_SHARDS}.json"
  output_subdir="turnkv_snapkv_smoke_${RUN_TS}/shard_${shard}_of_${NUM_SHARDS}"
  log_file="$LOG_DIR/smoke_turnkv_snapkv_${RUN_TS}_shard${shard}.log"

  shard_args=(
    evaluation/benchmarks/convcodeworld/modal_app.py::run_convcodeworld_live
    --benchmark-mode static
    --model meta-llama/Meta-Llama-3.1-8B-Instruct
    --attn-implementation flash_attention_3
    --feedback-config CF_EF_UNIT_SNF
    --press-name turnkv_snapkv
    --compression-ratio 0.0
    --global-budget 4096
    --local-budget 2048
    --max-turns 10
    --max-new-tokens 1024
    --num-eval-examples 0
    --task-ids "@$shard_json_container"
    --output-subdir "$output_subdir"
    --cot
    --alpha-floor 1.0
    --alpha-anchor 1.0
    --alpha-loyalty 1.0
    --floor-gamma 0.1
    --anchor-beta 0.25
    --loyalty-top-p 0.25
    --loyalty-update-every 5
    --require-flashdecode
    --log-level INFO
  )

  echo "[shard $shard/$((NUM_SHARDS - 1))] dispatching detached run -> $log_file"
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
  sleep 3

  echo "shard=$shard json=$shard_json log=$log_file output_subdir=$output_subdir" >> "$INDEX_FILE"
done

echo "Index of all shards: $INDEX_FILE"
