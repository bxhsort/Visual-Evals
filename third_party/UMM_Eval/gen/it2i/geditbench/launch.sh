#!/usr/bin/env bash
# GEdit-Bench (it2i) — image-editing generation + multi-GPU GPT-4o judge scoring.
# Conda hooks read unset vars, so no -u.
set -eo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/config/conda_envs.env"
source "${CONDA_SH}"
umm_conda_activate "${GEN_ENV}"
export PYTHONPATH="${REPO_ROOT}/gen/common:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

MODEL_NAME="${MODEL_NAME:-model}"
STEP_NAME="${STEP_NAME:-step}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}}"
MODEL_PATH="${MODEL_PATH:-}"
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

TXT_CFG="${TXT_CFG:-4.0}"
IMG_CFG="${IMG_CFG:-1.0}"
CFG_RENORM="${CFG_RENORM:-global}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
INSTRUCTION_LANGUAGE="${INSTRUCTION_LANGUAGE:-en}"
TASK_TYPE="${TASK_TYPE:-all}"
NUM_STEPS="${NUM_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gedit_bench}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs/geditbench_worker_logs/node_${NODE_RANK}}"
COLLECT_EXPERT_USAGE="${COLLECT_EXPERT_USAGE:-1}"
EXPERT_USAGE_ROOT="${EXPERT_USAGE_ROOT:-}"
TARGET_KEYS="${TARGET_KEYS:-}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"

GEDIT_ROOT="${GEDIT_ROOT:-${BENCH_DIR}/eval}"
GEDIT_EVAL_ROOT="${GEDIT_EVAL_ROOT:-${GEDIT_ROOT}/Step1X-Edit/GEdit-Bench}"
DATASET_PATH="${DATASET_PATH:-${GEDIT_ROOT}/GEdit-Bench}"
EVAL_ENV="${EVAL_ENV:-${GEDITBENCH_EVAL_ENV}}"
EVAL_BACKBONE="${EVAL_BACKBONE:-gpt4o}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-neo}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
EVAL_GPUS="${EVAL_GPUS:-}"
API_KEY="${API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}}"

build_output_folder() {
  python - "$TXT_CFG" "$IMG_CFG" "$CFG_RENORM" "$TIMESTEP_SHIFT" "$IMAGE_SIZE" <<'PY'
import sys, output_paths
print(output_paths.derive_it2i_folder(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
PY
}

if [[ -z "$OUTPUT_FOLDER" ]]; then
  OUTPUT_FOLDER="$(build_output_folder)"
fi

mkdir -p "$OUTPUT_BASE"
cd "$OUTPUT_BASE"

COMMON_ARGS=(
  --model_root "$MODEL_ROOT"
  --gedit_root "$GEDIT_ROOT"
  --dataset_path "$DATASET_PATH"
  --output_folder "$OUTPUT_FOLDER"
  --output_subdir "$OUTPUT_SUBDIR"
  --txt_cfg "$TXT_CFG"
  --img_cfg "$IMG_CFG"
  --cfg_renorm "$CFG_RENORM"
  --timestep_shift "$TIMESTEP_SHIFT"
  --image_size "$IMAGE_SIZE"
  --instruction_language "$INSTRUCTION_LANGUAGE"
  --task_type "$TASK_TYPE"
  --num_steps "$NUM_STEPS"
  --batch_size "$BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
)

if [[ -n "$MODEL_PATH" ]]; then
  COMMON_ARGS+=(--model_path "$MODEL_PATH")
fi
if [[ -n "$MODEL_NAME" ]]; then
  COMMON_ARGS+=(--model_name "$MODEL_NAME")
fi
if [[ -n "$STEP_NAME" ]]; then
  COMMON_ARGS+=(--step_name "$STEP_NAME")
fi
if [[ -n "$OUTPUT_PATH" ]]; then
  COMMON_ARGS+=(--output_path "$OUTPUT_PATH")
fi

EXTRA_ARGS=()
if (( COLLECT_EXPERT_USAGE != 0 )); then
  EXTRA_ARGS+=(--collect_expert_usage)
  if [[ -n "$EXPERT_USAGE_ROOT" ]]; then
    EXTRA_ARGS+=(--expert_usage_root "$EXPERT_USAGE_ROOT")
  fi
fi
if [[ -n "$TARGET_KEYS" ]]; then
  EXTRA_ARGS+=(--target_keys "$TARGET_KEYS")
fi
if [[ -n "${RESUME:-}" ]]; then
  case "${RESUME,,}" in
    1|true|t|yes|y|on) EXTRA_ARGS+=(--resume) ;;
  esac
fi
if (( SAVE_VISUALIZATIONS != 0 )); then
  EXTRA_ARGS+=(--save_visualizations)
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

build_eval_devices() {
  local devices_csv="$1"
  IFS=',' read -r -a ALL_EVAL_VISIBLE_DEVICES <<< "$devices_csv"
  if (( ${#ALL_EVAL_VISIBLE_DEVICES[@]} == 0 )); then
    echo "No visible devices were provided for evaluation."
    exit 1
  fi

  if [[ -z "$EVAL_GPUS" ]]; then
    EVAL_GPUS="${#ALL_EVAL_VISIBLE_DEVICES[@]}"
  fi
  if ! [[ "$EVAL_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_GPUS must be a positive integer, got $EVAL_GPUS"
    exit 1
  fi
  if (( EVAL_GPUS > ${#ALL_EVAL_VISIBLE_DEVICES[@]} )); then
    echo "EVAL_GPUS=$EVAL_GPUS exceeds available eval devices: ${devices_csv}"
    exit 1
  fi

  EVAL_DEVICE_LIST=("${ALL_EVAL_VISIBLE_DEVICES[@]:0:EVAL_GPUS}")
}

ensure_eval_inputs_exist() {
  if [[ ! -d "$RESULT_IMAGE_DIR" ]]; then
    echo "GEdit-Bench edited image directory does not exist: $RESULT_IMAGE_DIR"
    echo "If you are running eval-only, generate GEdit-Bench outputs first or pass OUTPUT_PATH/OUTPUT_SUBDIR to the existing GEdit-Bench result directory."
    exit 1
  fi

  if ! find "$RESULT_IMAGE_DIR" -type f -name '*.png' -print -quit | grep -q .; then
    echo "No edited images were found under: $RESULT_IMAGE_DIR"
    echo "GEdit-Bench eval expects images organized like <result_dir>/<task_type>/<language>/*.png."
    echo "If you are running eval-only, make sure RESULT_IMAGE_DIR points to GEdit-Bench outputs instead of another benchmark directory."
    exit 1
  fi
}

if (( NUM_NODES <= 0 )); then
  echo "NUM_NODES must be positive, got $NUM_NODES"
  exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "NODE_RANK must be in [0, NUM_NODES), got NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES"
  exit 1
fi

if (( RUN_GENERATION != 0 )); then
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
      worker_log="$LOG_DIR/geditbench_shard_${global_shard_rank}.log"
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
      echo "At least one GEdit-Bench worker failed. Check logs under $LOG_DIR."
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

if [[ -n "$OUTPUT_PATH" ]]; then
  RESULT_IMAGE_DIR="$OUTPUT_PATH"
else
  RESULT_IMAGE_DIR="${OUTPUT_BASE}/${OUTPUT_FOLDER}/${MODEL_NAME}/${STEP_NAME}/${OUTPUT_SUBDIR}"
fi
RESULT_ROOT_DIR="$(dirname "$RESULT_IMAGE_DIR")"

if (( RUN_EVAL != 0 )); then
  # API_KEY is optional for the gpt4o backbone: when empty, run_gedit_score.py resolves
  # the key from api_key.yaml:token_hub_eval (tokenhub gpt-4.1 judge).
  ensure_eval_inputs_exist
  build_eval_devices "$EVAL_CUDA_VISIBLE_DEVICES"

  umm_conda_activate "$EVAL_ENV"
  cd "$GEDIT_EVAL_ROOT"
  # run_gedit_score.py reads the GEdit-Bench dataset from this env (defaults to the original path otherwise).
  export GEDIT_DATASET_PATH="$DATASET_PATH"

  pids=()
  for shard_id in "${!EVAL_DEVICE_LIST[@]}"; do
    eval_device="${EVAL_DEVICE_LIST[$shard_id]}"
    echo "Launching GEdit-Bench eval shard ${shard_id}/${EVAL_GPUS} on GPU ${eval_device}"
    (
      export CUDA_VISIBLE_DEVICES="$eval_device"
      python run_gedit_score.py \
        --model_name "$EVAL_MODEL_NAME" \
        --save_dir "$RESULT_ROOT_DIR" \
        --backbone "$EVAL_BACKBONE" \
        --edited_images_dir "$RESULT_IMAGE_DIR" \
        --instruction_language "$INSTRUCTION_LANGUAGE" \
        --task_type "$TASK_TYPE" \
        --num_shards "$EVAL_GPUS" \
        --api_key "$API_KEY" \
        --shard_id "$shard_id"
    ) &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "At least one GEdit-Bench eval shard failed."
    exit 1
  fi

  TARGET_CSV_DIR="${RESULT_ROOT_DIR}/${EVAL_MODEL_NAME}/${EVAL_BACKBONE}"
  if (( EVAL_GPUS > 1 )); then
    python merge_csv.py --target_dir "$TARGET_CSV_DIR"
  fi

  if [[ "$TASK_TYPE" == "all" ]]; then
    python calculate_statistics.py \
      --model_name "$EVAL_MODEL_NAME" \
      --save_path "$RESULT_ROOT_DIR" \
      --backbone "$EVAL_BACKBONE" \
      --language "$INSTRUCTION_LANGUAGE"
  else
    echo "TASK_TYPE=$TASK_TYPE is not all, so calculate_statistics.py is skipped because it expects all groups."
  fi
fi
