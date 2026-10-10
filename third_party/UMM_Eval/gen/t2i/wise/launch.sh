#!/usr/bin/env bash
# WISE (t2i) — world-knowledge reasoning generation + GPT judge (gpt-4o via tokenhub).
set -eo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/gen/common:${PYTHONPATH:-}"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

MODEL_NAME="${MODEL_NAME:-model}"
STEP_NAME="${STEP_NAME:-step}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_BASE="${OUTPUT_FOLDER:-${OUTPUT_PATH:-${REPO_ROOT}/outputs}}"
LAUNCH_MODE="${LAUNCH_MODE:-device_map_multi}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RUN_GENERATION="${RUN_GENERATION:-1}"
if [[ -z "${RUN_EVAL+x}" ]]; then
  if (( NUM_NODES > 1 )); then
    RUN_EVAL=0
  else
    RUN_EVAL=1
  fi
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
THINK_MODE="${THINK_MODE:-false}"
MODE_SUFFIX="raw"
if is_truthy "$THINK_MODE"; then
  MODE_SUFFIX="think"
fi
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-wise_${MODE_SUFFIX}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs/wise_worker_logs/${MODEL_NAME}/${STEP_NAME}/${MODE_SUFFIX}/node_${NODE_RANK}}"
COLLECT_EXPERT_USAGE="${COLLECT_EXPERT_USAGE:-1}"
EXPERT_USAGE_ROOT="${EXPERT_USAGE_ROOT:-}"

WISE_DATA_ROOT="${WISE_DATA_ROOT:-${BENCH_DIR}/eval/data}"
CATEGORIES="${CATEGORIES:-cultural_common_sense,natural_science,spatio-temporal_reasoning}"
TARGET_PROMPT_IDS="${TARGET_PROMPT_IDS:-}"
RESIZE_TO="${RESIZE_TO:-}"
PE_MODE="${PE_MODE:-raw}"
PE_PROMPT_PATH="${REPO_ROOT}/tools/external_gen/pe_enhanced/wise_en.jsonl"

# GPT judge (gpt-4o via tokenhub). API_KEY optional (else resolved from api_key.yaml).
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
EVAL_MODEL="${EVAL_MODEL:-gpt-4o}"
EVAL_BASE_URL="${EVAL_BASE_URL:-${EVAL_API_BASE_URL:-${OPENAI_BASE_URL:-https://tokenhub.sensetime.com/v1}}}"
MAX_WORKERS="${MAX_WORKERS:-32}"

COMMON_ARGS=(
  --model_name "$MODEL_NAME"
  --step_name "$STEP_NAME"
  --wise_data_root "$WISE_DATA_ROOT"
  --categories "$CATEGORIES"
  --image_size "$IMAGE_SIZE"
  --num_steps "$NUM_STEPS"
  --cfg_scale "$CFG_SCALE"
  --timestep_shift "$TIMESTEP_SHIFT"
  --cfg_norm "$CFG_NORM"
  --batch_size "$BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --think_mode "$THINK_MODE"
  --output_subdir "$OUTPUT_SUBDIR"
)
case "$PE_MODE" in
  raw) ;;
  enhanced)
    [[ -f "$PE_PROMPT_PATH" ]] || { echo "Enhanced prompt cache not found: $PE_PROMPT_PATH"; exit 1; }
    COMMON_ARGS+=(--pe_prompt_path "$PE_PROMPT_PATH") ;;
  *) echo "Unsupported PE_MODE=$PE_MODE. Use raw or enhanced."; exit 1 ;;
esac
if [[ -n "$MODEL_PATH" ]]; then
  COMMON_ARGS+=(--model_path "$MODEL_PATH")
fi

EXTRA_ARGS=()
if (( COLLECT_EXPERT_USAGE != 0 )); then
  EXTRA_ARGS+=(--collect_expert_usage)
  if [[ -n "$EXPERT_USAGE_ROOT" ]]; then
    EXTRA_ARGS+=(--expert_usage_root "$EXPERT_USAGE_ROOT")
  fi
fi
if [[ -n "$TARGET_PROMPT_IDS" ]]; then
  EXTRA_ARGS+=(--target_prompt_ids "$TARGET_PROMPT_IDS")
fi

build_device_groups() {
  local devices_csv="$1"
  local per_worker="$2"
  local joined_group

  if (( per_worker <= 0 )); then
    echo "GPUS_PER_WORKER must be positive, got $per_worker"
    exit 1
  fi

  IFS=',' read -r -a ALL_VISIBLE_DEVICES <<< "$devices_csv"
  if (( ${#ALL_VISIBLE_DEVICES[@]} == 0 )); then
    echo "No visible devices were provided."
    exit 1
  fi
  if (( ${#ALL_VISIBLE_DEVICES[@]} % per_worker != 0 )); then
    echo "Visible GPU count ${#ALL_VISIBLE_DEVICES[@]} is not divisible by GPUS_PER_WORKER=$per_worker"
    exit 1
  fi

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

if (( NUM_NODES <= 0 )); then
  echo "NUM_NODES must be positive, got $NUM_NODES"
  exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "NODE_RANK must be in [0, NUM_NODES), got NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES"
  exit 1
fi

mkdir -p "$OUTPUT_BASE"
cd "$OUTPUT_BASE"
RESULT_ROOT_DIR="${RESULT_ROOT_DIR:-${RESULT_DIR:-}}"
if [[ -z "$RESULT_ROOT_DIR" ]]; then
  OUTPUT_ROOT="$(build_output_root)"
  RESULT_ROOT_DIR="${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}"
fi
WISE_RESULT_DIR="$RESULT_ROOT_DIR"
WISE_IMAGE_DIR="${WISE_RESULT_DIR}/${OUTPUT_SUBDIR}"

if (( RUN_GENERATION != 0 )); then
  source activate bagel
  export PATH=/opt/conda/envs/bagel/bin:$PATH
  if [[ "$LAUNCH_MODE" == "device_map" ]]; then
    export CUDA_VISIBLE_DEVICES
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
    TRANSFORMERS_VERBOSITY=error python "${BENCH_DIR}/inference.py" \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" \
      --device_map "$DEVICE_MAP" \
      --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB" \
      --num_shards "$NUM_NODES" \
      --shard_rank "$NODE_RANK"
  elif [[ "$LAUNCH_MODE" == "device_map_multi" ]]; then
    mkdir -p "$LOG_DIR"
    build_device_groups "$CUDA_VISIBLE_DEVICES" "$GPUS_PER_WORKER"
    LOCAL_NUM_SHARDS="${#DEVICE_GROUPS[@]}"
    TOTAL_NUM_SHARDS=$((NUM_NODES * LOCAL_NUM_SHARDS))
    NODE_SHARD_OFFSET=$((NODE_RANK * LOCAL_NUM_SHARDS))
    pids=()

    for shard_rank in "${!DEVICE_GROUPS[@]}"; do
      worker_devices="${DEVICE_GROUPS[$shard_rank]}"
      global_shard_rank=$((NODE_SHARD_OFFSET + shard_rank))
      worker_log="$LOG_DIR/wise_shard_${global_shard_rank}.log"
      echo "Launching global shard ${global_shard_rank}/${TOTAL_NUM_SHARDS} on node ${NODE_RANK}/${NUM_NODES} using GPUs ${worker_devices}. Log: ${worker_log}"
      (
        export CUDA_VISIBLE_DEVICES="$worker_devices"
        # Per-worker module cache so concurrent trust_remote_code loads do not race.
        export HF_MODULES_CACHE="${LOG_DIR}/hf_modules/shard_${global_shard_rank}"
        mkdir -p "$HF_MODULES_CACHE"
        unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
        TRANSFORMERS_VERBOSITY=error python "${BENCH_DIR}/inference.py" \
          "${COMMON_ARGS[@]}" \
          "${EXTRA_ARGS[@]}" \
          --device_map "$DEVICE_MAP" \
          --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB" \
          --num_shards "$TOTAL_NUM_SHARDS" \
          --shard_rank "$global_shard_rank"
      ) >"$worker_log" 2>&1 &
      pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if (( failed != 0 )); then
      echo "At least one WISE worker failed. Check logs under $LOG_DIR."
      exit 1
    fi
  elif [[ "$LAUNCH_MODE" == "ddp" ]]; then
    TRANSFORMERS_VERBOSITY=error torchrun --nproc_per_node="$GPUS" "${BENCH_DIR}/inference.py" \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  else
    echo "Unsupported LAUNCH_MODE=$LAUNCH_MODE. Use device_map, device_map_multi, or ddp."
    exit 1
  fi
fi

if [[ -n "$RESIZE_TO" ]]; then
  bash "${REPO_ROOT}/gen/common/resize_outputs.sh" "${WISE_IMAGE_DIR}" "$RESIZE_TO"
fi

if (( RUN_EVAL == 0 )); then
  exit 0
fi
if (( NUM_NODES > 1 )) && (( NODE_RANK != 0 )); then
  echo "Skipping WISE eval on node ${NODE_RANK}; eval runs only on node 0."
  exit 0
fi

# ---- GPT judge (gpt-4o via tokenhub) + WiScore aggregation ----
source activate geneval
export PATH=/opt/conda/envs/geneval/bin:$PATH
IFS=',' read -r -a CATEGORY_LIST <<< "$CATEGORIES"
SCORE_FILES=()

for category in "${CATEGORY_LIST[@]}"; do
  [[ -z "$category" ]] && continue

  result_full="${category}_full_results_${OUTPUT_SUBDIR}.json"
  result_scores="${category}_scores_results_${OUTPUT_SUBDIR}.jsonl"

  python "${BENCH_DIR}/eval/gpt_eval.py" \
    --image_dir "$WISE_IMAGE_DIR" \
    --json_path "${WISE_DATA_ROOT}/${category}.json" \
    --output_dir "$WISE_RESULT_DIR" \
    --api_key "$API_KEY" \
    --api_base "$EVAL_BASE_URL" \
    --model "$EVAL_MODEL" \
    --result_full "$result_full" \
    --result_scores "$result_scores" \
    --max_workers "$MAX_WORKERS"

  SCORE_FILES+=("${WISE_RESULT_DIR}/${result_scores}")
done

python "${BENCH_DIR}/eval/Calculate.py" \
  "${SCORE_FILES[@]}" \
  --category all | tee "${WISE_RESULT_DIR}/wise_results_${OUTPUT_SUBDIR}.txt"
