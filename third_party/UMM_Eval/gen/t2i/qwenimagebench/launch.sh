#!/usr/bin/env bash
# Qwen-Image-Bench (t2i) — generation (EN+ZH) + local Q-Judger (ms-swift) scoring.
# Conda hooks read unset vars, so no -u.
set -eo pipefail

QIB_BENCH_DIR_OVERRIDE="${QIB_BENCH_DIR_OVERRIDE:-}"
if [[ -n "$QIB_BENCH_DIR_OVERRIDE" ]]; then
  BENCH_DIR="$(cd "$QIB_BENCH_DIR_OVERRIDE" && pwd)"
else
  BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${BENCH_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/config/conda_envs.env"
source "${CONDA_SH}"
export PYTHONPATH="${REPO_ROOT}/gen/common:${PYTHONPATH:-}"
# Activate a python env up front so top-level helpers (build_output_root -> output_paths)
# work even on eval-only runs. The per-stage functions re-activate GEN_ENV / the judge env.
umm_conda_activate "${GEN_ENV}"

MODEL_NAME="${MODEL_NAME:-model}"
STEP_NAME="${STEP_NAME:-step}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_BASE="${OUTPUT_FOLDER:-${OUTPUT_PATH:-${OUTPUT_BASE:-${REPO_ROOT}/outputs}}}"
LAUNCH_MODE="${LAUNCH_MODE:-device_map_multi}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RUN_GENERATION="${RUN_GENERATION:-1}"
if [[ -z "${RUN_EVAL+x}" ]]; then
  if (( NUM_NODES > 1 )); then RUN_EVAL=0; else RUN_EVAL=1; fi
fi

GPUS="${GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-2}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
MAX_MEMORY_PER_GPU_GB="${MAX_MEMORY_PER_GPU_GB:-70}"

IMAGE_SIZE="${IMAGE_SIZE:-1024}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
CFG_NORM="${CFG_NORM:-global}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
COLLECT_EXPERT_USAGE="${COLLECT_EXPERT_USAGE:-1}"
EXPERT_USAGE_ROOT="${EXPERT_USAGE_ROOT:-}"
TARGET_PROMPT_IDS="${TARGET_PROMPT_IDS:-}"
RESIZE_TO="${RESIZE_TO:-}"
# Generation prompt root. Judge prompts are intentionally fixed to the bundled raw QIB prompts.
DATA_ROOT="${DATA_ROOT:-${BENCH_DIR}/prompts}"

# Languages to run (always both by default).
QIB_LANGS="${QIB_LANGS:-en zh}"
QIB_LANGS="${QIB_LANGS//,/ }"

# Opt-in thinking mode (wise/rise-style): THINK_MODE=1 routes outputs to separate _think
# subdirs so a think run never overwrites the normal run. Generation itself reads THINK_MODE
# in inference.py (t2i_generate think_mode).
THINK_MODE="${THINK_MODE:-false}"
THINK_SUFFIX=""
case "${THINK_MODE,,}" in 1|true|on|yes) THINK_SUFFIX="_think";; esac

# Q-Judger (local ms-swift) knobs.
QJUDGER_MODEL_PATH="${QJUDGER_MODEL_PATH:-${BENCH_DIR}/eval/models/q_judger}"
# Per-replica batch. Data-parallel judge puts ONE full ~54G model on each card, leaving
# only ~26G headroom. 24 OOMs; 12 is the conservative first step above the old value of 8.
# judge.py automatically halves a dimension's batch and retries after a CUDA OOM.
QJUDGER_BATCH_SIZE="${QJUDGER_BATCH_SIZE:-12}"
QJUDGER_MAX_NEW_TOKENS="${QJUDGER_MAX_NEW_TOKENS:-4096}"
QJUDGER_DIM_BATCH_SIZES="${QJUDGER_DIM_BATCH_SIZES:-}"
QJUDGER_MIN_BATCH_SIZE="${QJUDGER_MIN_BATCH_SIZE:-8}"
QJUDGER_REQUIRE_FLA="${QJUDGER_REQUIRE_FLA:-1}"
QJUDGER_SYNC_TIMEOUT_SECS="${QJUDGER_SYNC_TIMEOUT_SECS:-${WORKER_SYNC_TIMEOUT_SECS:-86400}}"
QJUDGER_SYNC_POLL_SECS="${QJUDGER_SYNC_POLL_SECS:-${WORKER_SYNC_POLL_SECS:-5}}"

build_device_groups() {
  local devices_csv="$1"; local per_worker="$2"; local joined_group
  if (( per_worker <= 0 )); then echo "GPUS_PER_WORKER must be positive, got $per_worker"; exit 1; fi
  IFS=',' read -r -a ALL_VISIBLE_DEVICES <<< "$devices_csv"
  if (( ${#ALL_VISIBLE_DEVICES[@]} == 0 )); then echo "No visible devices were provided."; exit 1; fi
  if (( ${#ALL_VISIBLE_DEVICES[@]} % per_worker != 0 )); then
    echo "Visible GPU count ${#ALL_VISIBLE_DEVICES[@]} is not divisible by GPUS_PER_WORKER=$per_worker"; exit 1; fi
  DEVICE_GROUPS=()
  for ((i=0; i<${#ALL_VISIBLE_DEVICES[@]}; i+=per_worker)); do
    joined_group=$(IFS=,; echo "${ALL_VISIBLE_DEVICES[*]:i:per_worker}")
    DEVICE_GROUPS+=("$joined_group")
  done
}

build_output_root() {
  python - "$IMAGE_SIZE" "$CFG_SCALE" "$TIMESTEP_SHIFT" "$CFG_NORM" <<'PY'
import sys, output_paths
print(output_paths.derive_t2i_root(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
PY
}

if (( NUM_NODES <= 0 )); then echo "NUM_NODES must be positive, got $NUM_NODES"; exit 1; fi
if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "NODE_RANK must be in [0, NUM_NODES), got NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES"; exit 1; fi

mkdir -p "$OUTPUT_BASE"
cd "$OUTPUT_BASE"
OUTPUT_ROOT="$(build_output_root)"

run_generation_for_lang() {
  local QIB_MODE="$1"
  umm_conda_activate "${GEN_ENV}"
  local OUTPUT_SUBDIR="qwenimagebench_${QIB_MODE}${THINK_SUFFIX}"
  local DATA_PATH="${DATA_ROOT}/qwenimagebench_${QIB_MODE}.jsonl"
  local LOG_DIR="${OUTPUT_BASE}/logs/qwenimagebench_worker_logs/${MODEL_NAME}/${STEP_NAME}/node_${NODE_RANK}/${QIB_MODE}${THINK_SUFFIX}"

  local COMMON_ARGS=(
    --model_name "$MODEL_NAME" --step_name "$STEP_NAME" --output_subdir "$OUTPUT_SUBDIR"
    --resolution "$IMAGE_SIZE" --data_path "$DATA_PATH" --data_root "$DATA_ROOT" --mode "$QIB_MODE"
    --num_steps "$NUM_STEPS" --cfg_scale "$CFG_SCALE" --timestep_shift "$TIMESTEP_SHIFT"
    --cfg_norm "$CFG_NORM" --batch_size "$BATCH_SIZE" --micro_batch_size "$MICRO_BATCH_SIZE"
  )
  [[ -n "$MODEL_PATH" ]] && COMMON_ARGS+=(--model_path "$MODEL_PATH")
  local EXTRA_ARGS=()
  if (( COLLECT_EXPERT_USAGE != 0 )); then
    EXTRA_ARGS+=(--collect_expert_usage)
    [[ -n "$EXPERT_USAGE_ROOT" ]] && EXTRA_ARGS+=(--expert_usage_root "$EXPERT_USAGE_ROOT")
  fi
  [[ -n "$TARGET_PROMPT_IDS" ]] && EXTRA_ARGS+=(--target_prompt_ids "$TARGET_PROMPT_IDS")
  if [[ -n "${RESUME:-}" ]]; then
    case "${RESUME,,}" in
      1|true|t|yes|y|on) EXTRA_ARGS+=(--resume) ;;
    esac
  fi

  if [[ "$LAUNCH_MODE" == "device_map" ]]; then
    export CUDA_VISIBLE_DEVICES
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
    TRANSFORMERS_VERBOSITY=error python "${BENCH_DIR}/inference.py" \
      "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
      --device_map "$DEVICE_MAP" --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB" \
      --num_shards "$NUM_NODES" --shard_rank "$NODE_RANK"
  elif [[ "$LAUNCH_MODE" == "device_map_multi" ]]; then
    mkdir -p "$LOG_DIR"
    build_device_groups "$CUDA_VISIBLE_DEVICES" "$GPUS_PER_WORKER"
    local LOCAL_NUM_SHARDS="${#DEVICE_GROUPS[@]}"
    local TOTAL_NUM_SHARDS=$((NUM_NODES * LOCAL_NUM_SHARDS))
    local NODE_SHARD_OFFSET=$((NODE_RANK * LOCAL_NUM_SHARDS))
    local pids=()
    for shard_rank in "${!DEVICE_GROUPS[@]}"; do
      local worker_devices="${DEVICE_GROUPS[$shard_rank]}"
      local global_shard_rank=$((NODE_SHARD_OFFSET + shard_rank))
      local worker_log="$LOG_DIR/qwenimagebench_shard_${global_shard_rank}.log"
      echo "Launching ${QIB_MODE} global shard ${global_shard_rank}/${TOTAL_NUM_SHARDS} on GPUs ${worker_devices}. Log: ${worker_log}"
      (
        export CUDA_VISIBLE_DEVICES="$worker_devices"
        export HF_MODULES_CACHE="${LOG_DIR}/hf_modules/shard_${global_shard_rank}"
        mkdir -p "$HF_MODULES_CACHE"
        unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
        TRANSFORMERS_VERBOSITY=error python "${BENCH_DIR}/inference.py" \
          "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
          --device_map "$DEVICE_MAP" --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB" \
          --num_shards "$TOTAL_NUM_SHARDS" --shard_rank "$global_shard_rank"
      ) >"$worker_log" 2>&1 &
      pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    if (( failed != 0 )); then echo "At least one ${QIB_MODE} worker failed. Check logs under $LOG_DIR."; exit 1; fi
  elif [[ "$LAUNCH_MODE" == "ddp" ]]; then
    TRANSFORMERS_VERBOSITY=error torchrun --nproc_per_node="$GPUS" "${BENCH_DIR}/inference.py" \
      "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
  else
    echo "Unsupported LAUNCH_MODE=$LAUNCH_MODE. Use device_map, device_map_multi, or ddp."; exit 1
  fi

  if [[ -n "$RESIZE_TO" ]]; then
    bash "${REPO_ROOT}/gen/common/resize_outputs.sh" \
      "${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}/${OUTPUT_SUBDIR}/samples" "$RESIZE_TO"
  fi
}

sync_judge_phase() {
  local phase="$1"
  local local_status="$2"
  if (( NUM_NODES <= 1 )); then
    return "$local_status"
  fi
  if [[ -z "${WORKER_SYNC_DIR:-}" ]]; then
    echo "WORKER_SYNC_DIR is required for multi-node QIB judge synchronization."
    return 1
  fi

  local sync_root="$WORKER_SYNC_DIR"
  local phase_dir="${sync_root}/qib_judge/${phase}"
  local status_file="${phase_dir}/node_${NODE_RANK}.status"
  local deadline=0
  local completed=0
  mkdir -p "$phase_dir"
  if (( local_status == 0 )); then
    echo "success" >"${status_file}.tmp"
  else
    echo "failed" >"${status_file}.tmp"
  fi
  mv "${status_file}.tmp" "$status_file"

  if (( QJUDGER_SYNC_TIMEOUT_SECS > 0 )); then
    deadline=$(( $(date +%s) + QJUDGER_SYNC_TIMEOUT_SECS ))
  fi
  while true; do
    completed="$(find "$phase_dir" -maxdepth 1 -type f -name 'node_*.status' | wc -l)"
    completed="${completed//[[:space:]]/}"
    if (( completed >= NUM_NODES )); then
      break
    fi
    if (( deadline > 0 && $(date +%s) >= deadline )); then
      echo "QIB judge sync timed out at ${phase}: ${completed}/${NUM_NODES} nodes."
      return 1
    fi
    sleep "$QJUDGER_SYNC_POLL_SECS"
  done
  if grep -q '^failed$' "${phase_dir}"/node_*.status; then
    echo "QIB judge phase ${phase} failed on at least one node."
    return 1
  fi
}

run_eval_for_lang() {
  local QIB_MODE="$1"
  umm_conda_activate "${QWENIMAGEBENCH_EVAL_ENV}"
  # Reduce fragmentation OOMs for the per-card model replica (qwen3_5 gated-delta-rule attn).
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  local OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}/qwenimagebench_${QIB_MODE}${THINK_SUFFIX}"
  local SAMPLE_DIR="${OUTPUT_DIR}/samples"
  local EVAL_DIR="${OUTPUT_DIR}/eval"
  mkdir -p "$EVAL_DIR"
  local JUDGE_INPUT="${EVAL_DIR}/judge_input.jsonl"

  local input_status=0
  if (( NODE_RANK == 0 )); then
    if ! python "${BENCH_DIR}/eval/build_judge_input.py" \
      --lang "$QIB_MODE" \
      --samples-dir "$SAMPLE_DIR" --out "$JUDGE_INPUT"; then
      input_status=1
    fi
  fi
  sync_judge_phase "${QIB_MODE}${THINK_SUFFIX}_input" "$input_status"

  # Data-parallel judge: one model replica per GPU (no tensor-parallel), each scores
  # rows[rank::N], then merge_shard_scores.py recombines into one bench_scores.json.
  # The 27B Q-Judger fits on a single 80G card, so DP gives ~N x throughput + full GPU use.
  local JUDGE_ARGS=(
    --input "$JUDGE_INPUT" --model "$QJUDGER_MODEL_PATH"
    --local-metadata "${BENCH_DIR}/eval/metadata/bench_metadata.json"
    --max-batch-size "$QJUDGER_BATCH_SIZE" --max-new-tokens "$QJUDGER_MAX_NEW_TOKENS"
    --min-batch-size "$QJUDGER_MIN_BATCH_SIZE"
  )
  [[ -n "$QJUDGER_DIM_BATCH_SIZES" ]] && JUDGE_ARGS+=(--dimension-batch-sizes "$QJUDGER_DIM_BATCH_SIZES")
  if (( QJUDGER_REQUIRE_FLA == 0 )); then JUDGE_ARGS+=(--no-require-fla); fi

  local IFS_SAVE="$IFS"; IFS=',' read -r -a JUDGE_DEVS <<< "$CUDA_VISIBLE_DEVICES"; IFS="$IFS_SAVE"
  local LOCAL_NSHARDS="${QJUDGER_NUM_SHARDS:-${#JUDGE_DEVS[@]}}"
  if (( LOCAL_NSHARDS <= 0 || LOCAL_NSHARDS > ${#JUDGE_DEVS[@]} )); then
    echo "QJUDGER_NUM_SHARDS must be in [1, ${#JUDGE_DEVS[@]}], got ${LOCAL_NSHARDS}."
    return 1
  fi
  local TOTAL_NSHARDS=$((NUM_NODES * LOCAL_NSHARDS))
  if (( TOTAL_NSHARDS == 1 )); then
    ( cd "${BENCH_DIR}/eval" && python judge.py "${JUDGE_ARGS[@]}" )
  else
    local JLOG_DIR="${EVAL_DIR}/judge_logs/node_${NODE_RANK}"; mkdir -p "$JLOG_DIR"
    local jpids=()
    for (( r=0; r<LOCAL_NSHARDS; r++ )); do
      local dev="${JUDGE_DEVS[$r]}"
      local global_rank=$((NODE_RANK * LOCAL_NSHARDS + r))
      local jlog="${JLOG_DIR}/judge_${QIB_MODE}_shard${global_rank}.log"
      echo "judge ${QIB_MODE} global shard ${global_rank}/${TOTAL_NSHARDS} on GPU ${dev} -> ${jlog}"
      ( cd "${BENCH_DIR}/eval" && CUDA_VISIBLE_DEVICES="$dev" python judge.py \
          "${JUDGE_ARGS[@]}" --num-shards "$TOTAL_NSHARDS" --shard-rank "$global_rank" ) >"$jlog" 2>&1 &
      jpids+=("$!")
    done
    local jfail=0
    for p in "${jpids[@]}"; do wait "$p" || jfail=1; done
    if ! sync_judge_phase "${QIB_MODE}${THINK_SUFFIX}_shards" "$jfail"; then
      echo "A judge shard failed for ${QIB_MODE}; see ${EVAL_DIR}/judge_logs."
      return 1
    fi

    local merge_status=0
    if (( NODE_RANK == 0 )); then
      if ! ( cd "${BENCH_DIR}/eval" && python merge_shard_scores.py \
        --input "$JUDGE_INPUT" --num-shards "$TOTAL_NSHARDS" ); then
        merge_status=1
      fi
    fi
    sync_judge_phase "${QIB_MODE}${THINK_SUFFIX}_merge" "$merge_status"
  fi
}

# ---- generation (all requested languages) ----
if (( RUN_GENERATION != 0 )); then
  for QIB_MODE in $QIB_LANGS; do run_generation_for_lang "$QIB_MODE"; done
fi

if (( RUN_EVAL == 0 )); then exit 0; fi

# ---- eval (all requested languages on every node) then node-0 bilingual aggregate ----
AGG_ARGS=()
for QIB_MODE in $QIB_LANGS; do
  run_eval_for_lang "$QIB_MODE"
  SCORES_JSON="${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}/qwenimagebench_${QIB_MODE}${THINK_SUFFIX}/eval/judge_input_bench_scores.json"
  AGG_ARGS+=(--"$QIB_MODE" "$SCORES_JSON")
done
if (( NODE_RANK != 0 )); then exit 0; fi
SUMMARY_DIR="${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}"
python "${BENCH_DIR}/eval/aggregate_bilingual.py" \
  "${AGG_ARGS[@]}" --out "${SUMMARY_DIR}/qwenimagebench_bilingual_summary${THINK_SUFFIX}.json"
