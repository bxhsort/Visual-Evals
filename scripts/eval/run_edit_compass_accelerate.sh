#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-bagel}"
STEP_NAME="${STEP_NAME:-bagel_edit_compass}"
DATA_ROOT="${EDIT_COMPASS_DATA_ROOT:-${DATA_ROOT:-${BAGEL_ROOT}/third_party/Edit-Compass-data}}"
BENCHMARK_ROOT="${EDIT_COMPASS_BENCHMARK_ROOT:-${BENCHMARK_ROOT:-${BAGEL_ROOT}/third_party/Edit-Compass-and-EditReward-Compass}}"
OUTPUT_BASE="${EDIT_COMPASS_OUTPUT_BASE:-${EDIT_COMPASS_OUTPUT_ROOT:-${OUTPUT_ROOT:-${BAGEL_ROOT}/outputs/editcompass}}}"
OUTPUT_ROOT="${OUTPUT_BASE%/}/${STEP_NAME}"
LANGUAGE="${EDIT_COMPASS_LANGUAGE:-${LANGUAGE:-en}}"
case "$LANGUAGE" in
  en|cn) ;;
  *) LANGUAGE="en" ;;
esac
PARTS="${PARTS:-Part1 Part2}"
METRICS="${METRICS:-IC URC VQ WA IF}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_MEMORY="${MAX_MEMORY:-80GiB}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-${BAGEL_ROOT}/offload_edit_compass/${STEP_NAME}}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
OVERWRITE="${OVERWRITE:-0}"
THINK_MODE="${THINK_MODE:-0}"
MASTER_PORT="${MASTER_PORT:-12460}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-${EVAL_NUM_PROCESSES:-8}}"
CFG_TEXT_SCALE="${CFG_TEXT_SCALE:-4.0}"
CFG_IMG_SCALE="${CFG_IMG_SCALE:-1.5}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1024}"
MIN_IMAGE_SIZE="${MIN_IMAGE_SIZE:-512}"
EDIT_COMPASS_API_MAX_IMAGE_SIDE="${EDIT_COMPASS_API_MAX_IMAGE_SIDE:-2048}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required"
  exit 1
fi
if (( RUN_GENERATION != 0 )) && [[ ! -d "$DATA_ROOT" ]]; then
  echo "EditCompass data not found: $DATA_ROOT" >&2
  echo "See DG2_EVAL.md for the official dataset download command." >&2
  exit 1
fi
if (( RUN_EVAL != 0 )) && [[ ! -d "${BENCHMARK_ROOT}/Prompt" ]]; then
  echo "EditCompass evaluator not found: ${BENCHMARK_ROOT}/Prompt" >&2
  echo "Clone the official EditCompass repository as described in DG2_EVAL.md." >&2
  exit 1
fi

export PYTHONPATH="${BAGEL_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}/logs"
cd "$BAGEL_ROOT"

echo "[edit-compass] model=${MODEL_PATH}"
echo "[edit-compass] output=${OUTPUT_ROOT}/${MODEL_NAME}/${LANGUAGE}"
echo "[edit-compass] data_root=${DATA_ROOT}"
echo "[edit-compass] parts=${PARTS} language=${LANGUAGE}"
echo "[edit-compass] run_generation=${RUN_GENERATION} run_eval=${RUN_EVAL} run_summary=${RUN_SUMMARY}"
echo "[edit-compass] gpu_list=${GPU_LIST} think_mode=${THINK_MODE}"

read -r -a PART_ARGS <<< "$PARTS"
read -r -a METRIC_ARGS <<< "$METRICS"
OVERWRITE_ARGS=()
if (( OVERWRITE != 0 )); then
  OVERWRITE_ARGS=(--overwrite)
fi
THINK_ARGS=()
case "${THINK_MODE,,}" in
  1|true|yes|on) THINK_ARGS=(--think) ;;
esac

if (( RUN_GENERATION != 0 )); then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  pids=()
  for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    log="${OUTPUT_ROOT}/logs/gen_worker_$(printf '%02d' "$idx")_gpu_${gpu}.log"
    shard_result="${OUTPUT_ROOT}/${MODEL_NAME}/${LANGUAGE}/logs/generated_shard_$(printf '%02d' "$idx").jsonl"
    echo "[edit-compass] worker ${idx}/${#GPUS[@]} gpu=${gpu} log=${log}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python ./eval/gen/edit_compass/gen_images_edit_compass.py \
        --mode generate \
        --data_root "$DATA_ROOT" \
        --save_root "$OUTPUT_ROOT" \
        --model_name "$MODEL_NAME" \
        --language "$LANGUAGE" \
        --parts "${PART_ARGS[@]}" \
        --model_path "$MODEL_PATH" \
        --device 0 \
        --shard_id "$idx" \
        --total_shards "${#GPUS[@]}" \
        --shard_result "$shard_result" \
        --accelerate_device_map \
        --max_memory "$MAX_MEMORY" \
        --offload_folder "${OFFLOAD_FOLDER}/worker_${idx}" \
        --cfg_text_scale "$CFG_TEXT_SCALE" \
        --cfg_img_scale "$CFG_IMG_SCALE" \
	        --num_timesteps "$NUM_TIMESTEPS" \
	        --max_image_size "$MAX_IMAGE_SIZE" \
	        --min_image_size "$MIN_IMAGE_SIZE" \
	        "${THINK_ARGS[@]}" \
	        "${OVERWRITE_ARGS[@]}"
    ) >"$log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[edit-compass] worker ${idx} failed; tailing log:"
      tail -80 "${OUTPUT_ROOT}/logs/gen_worker_$(printf '%02d' "$idx")_gpu_${GPUS[$idx]}.log"
      failed=1
    fi
  done
  if (( failed != 0 )); then
    exit 1
  fi

  python ./eval/gen/edit_compass/gen_images_edit_compass.py \
    --mode finalize \
    --data_root "$DATA_ROOT" \
    --save_root "$OUTPUT_ROOT" \
    --model_name "$MODEL_NAME" \
    --language "$LANGUAGE" \
    --parts "${PART_ARGS[@]}"
fi

if (( RUN_EVAL != 0 )); then
  export API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
  export API_BASE="${API_BASE:-${OPENAI_BASE_URL:-}}"
  if [[ -z "$API_KEY" ]]; then
    echo "API_KEY or OPENAI_API_KEY is required for EditCompass scoring." >&2
    exit 1
  fi
  python ./eval/gen/edit_compass/eval_edit_compass.py \
    --benchmark_root "$BENCHMARK_ROOT" \
    --data_root "$DATA_ROOT" \
    --save_root "${OUTPUT_ROOT}/${MODEL_NAME}" \
    --language "$LANGUAGE" \
    --parts "${PART_ARGS[@]}" \
    --metrics "${METRIC_ARGS[@]}" \
    --num_workers "$EVAL_NUM_WORKERS" \
    --api_key "$API_KEY" \
    --api_base "${API_BASE:-${OPENAI_BASE_URL:-}}" \
    --eval_model_name "${EDIT_COMPASS_EVAL_MODEL_NAME:-gemini-3.1-pro-preview}" \
    --api_max_image_side "$EDIT_COMPASS_API_MAX_IMAGE_SIDE"
fi

if (( RUN_SUMMARY != 0 )); then
  python ./eval/gen/edit_compass/summary_edit_compass.py \
    --data_root "$DATA_ROOT" \
    --save_dir "${OUTPUT_ROOT}/${MODEL_NAME}" \
    --parts "${PART_ARGS[@]}" \
    --languages "$LANGUAGE"
fi

echo "[edit-compass] done"
