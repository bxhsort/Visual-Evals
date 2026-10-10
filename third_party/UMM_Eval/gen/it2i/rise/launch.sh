#!/usr/bin/env bash
# RISE (it2i) — reasoning image-editing generation + gpt-4o (tokenhub) judge scoring.
# RISE supports a think_mode; outputs land under rise_raw / rise_think.
# Conda hooks read unset vars, so no -u.
set -eo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/config/conda_envs.env"
source "${CONDA_SH}"
umm_conda_activate "${GEN_ENV}"
export PYTHONPATH="${REPO_ROOT}/gen/common:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

is_truthy() {
  case "${1,,}" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

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
IMG_CFG="${IMG_CFG:-1.5}"
CFG_RENORM="${CFG_RENORM:-${CFG_NORM:-none}}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-1.0}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
NUM_STEPS="${NUM_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
THINK_MODE="${THINK_MODE:-false}"
MODE_SUFFIX="raw"
if is_truthy "$THINK_MODE"; then
  MODE_SUFFIX="think"
fi
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-rise_${MODE_SUFFIX}}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs/rise_worker_logs/${MODEL_NAME}/${STEP_NAME}/${MODE_SUFFIX}/node_${NODE_RANK}}"
COLLECT_EXPERT_USAGE="${COLLECT_EXPERT_USAGE:-1}"
EXPERT_USAGE_ROOT="${EXPERT_USAGE_ROOT:-}"
TARGET_INDICES="${TARGET_INDICES:-}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-1}"

RISE_DATA_DIR="${RISE_DATA_DIR:-${BENCH_DIR}/eval/data}"
RISE_DATA_JSON="${RISE_DATA_JSON:-${RISE_DATA_DIR}/datav2_total_w_subtask.json}"
RISE_INPUT_ROOT="${RISE_INPUT_ROOT:-${RISE_DATA_DIR}}"
RISE_NPROC="${RISE_NPROC:-${EVAL_NUM_PROCESSES:-96}}"
RISE_PREFIX="${RISE_PREFIX:-${OUTPUT_SUBDIR}}"
RISE_EVAL_ENV="${RISE_EVAL_ENV:-${GEN_ENV}}"
# API_KEY is optional: when empty, gpt_eval.py resolves the key from api_key.yaml:token_hub_eval.
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
  --data_path "$RISE_DATA_JSON"
  --image_root "$RISE_INPUT_ROOT"
  --output_folder "$OUTPUT_FOLDER"
  --output_subdir "$OUTPUT_SUBDIR"
  --txt_cfg "$TXT_CFG"
  --img_cfg "$IMG_CFG"
  --cfg_renorm "$CFG_RENORM"
  --timestep_shift "$TIMESTEP_SHIFT"
  --image_size "$IMAGE_SIZE"
  --num_steps "$NUM_STEPS"
  --batch_size "$BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --think_mode "$THINK_MODE"
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
if [[ -n "$TARGET_INDICES" ]]; then
  EXTRA_ARGS+=(--target_indices "$TARGET_INDICES")
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

ensure_eval_inputs_exist() {
  if [[ ! -d "$RESULT_IMAGE_DIR" ]]; then
    echo "RISE output directory does not exist: $RESULT_IMAGE_DIR"
    echo "If you are running eval-only, generate RISE outputs first or pass OUTPUT_PATH/OUTPUT_SUBDIR to an existing RISE result directory."
    exit 1
  fi

  if ! find "$RESULT_IMAGE_DIR" -type f -name '*.png' -print -quit | grep -q .; then
    echo "No generated RISE images were found under: $RESULT_IMAGE_DIR"
    echo "RISE eval expects images organized like <result_dir>/<category>/*.png."
    echo "If you are running eval-only, make sure RESULT_IMAGE_DIR points to RISE outputs."
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
      worker_log="$LOG_DIR/rise_shard_${global_shard_rank}.log"
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
      echo "At least one RISE worker failed. Check logs under $LOG_DIR."
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

if (( RUN_EVAL != 0 )); then
  if (( NUM_NODES > 1 )) && (( NODE_RANK != 0 )); then
    echo "Skipping RISE eval on node ${NODE_RANK}; eval runs only on node 0."
    exit 0
  fi

  if [[ ! -f "$RISE_DATA_JSON" ]]; then
    echo "RISE_DATA_JSON does not exist: $RISE_DATA_JSON"
    exit 1
  fi
  if [[ ! -d "$RISE_INPUT_ROOT" ]]; then
    echo "RISE_INPUT_ROOT does not exist: $RISE_INPUT_ROOT"
    exit 1
  fi

  ensure_eval_inputs_exist

  umm_conda_activate "${RISE_EVAL_ENV}"
  cd "${BENCH_DIR}/eval/judge"

  python gpt_eval.py \
    --data "$RISE_DATA_JSON" \
    --output "$RESULT_IMAGE_DIR" \
    --input "$RISE_INPUT_ROOT" \
    --prefix "$RISE_PREFIX" \
    --api_key "$API_KEY" \
    --nproc "$RISE_NPROC"
fi
