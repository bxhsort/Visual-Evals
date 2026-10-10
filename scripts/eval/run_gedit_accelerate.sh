#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UMM_ROOT="${UMM_ROOT:-${BAGEL_ROOT}/third_party/UMM_Eval}"

MODEL_PATH="${MODEL_PATH:-${model_path:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${output_dir:-}}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-}}"
N_GPU="${N_GPU:-8}"
MAX_MEMORY="${MAX_MEMORY:-80GiB}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-${BAGEL_ROOT}/offload_gedit}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
N_GPT_PARALLEL="${N_GPT_PARALLEL:-10}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-bagel}"
INSTRUCTION_LANGUAGES="${INSTRUCTION_LANGUAGES:-en cn}"
FINALIZE_EVAL="${FINALIZE_EVAL:-1}"
THINK_MODE="${THINK_MODE:-0}"
GEDIT_DATASET_PATH="${GEDIT_DATASET_PATH:-${UMM_ROOT}/gen/it2i/geditbench/eval/GEdit-Bench}"

CFG_TEXT_SCALE="${CFG_TEXT_SCALE:-4.0}"
CFG_IMG_SCALE="${CFG_IMG_SCALE:-1.5}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH/model_path is required"
  exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR/output_dir is required"
  exit 1
fi

if [[ -z "$GPU_LIST" ]]; then
  gpu_values=()
  for ((i=0; i<N_GPU; i++)); do
    gpu_values+=("$i")
  done
else
  IFS=',' read -r -a gpu_values <<< "$GPU_LIST"
  N_GPU="${#gpu_values[@]}"
fi

GEN_DIR="${OUTPUT_DIR}/gen_image"
LOG_DIR="${OUTPUT_DIR}/logs"
EVAL_IMAGE_DIR="${GEN_DIR}/fullset"

export PYTHONPATH="${BAGEL_ROOT}:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR" "$GEN_DIR" "$LOG_DIR"
cd "$BAGEL_ROOT"

echo "[gedit-accelerate] model=${MODEL_PATH}"
echo "[gedit-accelerate] output_dir=${OUTPUT_DIR}"
echo "[gedit-accelerate] gen_dir=${GEN_DIR}"
echo "[gedit-accelerate] eval_image_dir=${EVAL_IMAGE_DIR}"
echo "[gedit-accelerate] gpu_list=$(IFS=,; echo "${gpu_values[*]}")"
echo "[gedit-accelerate] max_memory=${MAX_MEMORY} think_mode=${THINK_MODE}"

if (( RUN_GENERATION != 0 )); then
  pids=()
  THINK_ARGS=()
  case "${THINK_MODE,,}" in
    1|true|yes|on) THINK_ARGS=(--use_think) ;;
  esac
  for idx in "${!gpu_values[@]}"; do
    gpu="${gpu_values[$idx]}"
    log="${LOG_DIR}/gen_worker_$(printf '%02d' "$idx")_gpu_${gpu}.log"
    echo "[gedit-accelerate] worker ${idx}/${N_GPU} gpu=${gpu} log=${log}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python eval/gen/gen_images_mp_it2i.py \
        --benchmark geditbench \
        --dataset_path "$GEDIT_DATASET_PATH" \
        --model_path "$MODEL_PATH" \
        --output_dir "$GEN_DIR" \
        --shard_id "$idx" \
        --total_shards "$N_GPU" \
        --device 0 \
        --cfg_text_scale "$CFG_TEXT_SCALE" \
        --cfg_img_scale "$CFG_IMG_SCALE" \
	        --accelerate_device_map \
	        --max_memory "$MAX_MEMORY" \
	        --offload_folder "${OFFLOAD_FOLDER}/worker_${idx}" \
	        ${TARGET_INDICES:+--target_indices "$TARGET_INDICES"} \
	        "${THINK_ARGS[@]}"
    ) >"$log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[gedit-accelerate] worker ${idx} failed; tailing log:"
      tail -80 "${LOG_DIR}/gen_worker_$(printf '%02d' "$idx")_gpu_${gpu_values[$idx]}.log"
      failed=1
    fi
  done
  if (( failed != 0 )); then
    exit 1
  fi
fi

if (( RUN_EVAL != 0 )); then
  if [[ -n "${openai_api_key:-}" ]]; then
    export API_KEY="${openai_api_key}"
  else
    export API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
  fi
  if [[ -z "$API_KEY" && -z "${EVAL_API_KEY_FILE:-}" ]]; then
    echo "API_KEY (or EVAL_API_KEY_FILE) is required for GEdit-Bench scoring."
    exit 1
  fi

  for language in $INSTRUCTION_LANGUAGES; do
    echo "[gedit-accelerate] evaluating language=${language}"
    STEP_NAME="$(basename "$OUTPUT_DIR")" \
    EVAL_MODEL_NAME="$EVAL_MODEL_NAME" \
    INSTRUCTION_LANGUAGE="$language" \
    OUTPUT_PATH="$EVAL_IMAGE_DIR" \
    EVAL_GPUS="$N_GPU" \
    EVAL_CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_values[*]}")" \
    DATASET_PATH="$GEDIT_DATASET_PATH" \
    FINALIZE_EVAL="$FINALIZE_EVAL" \
    API_KEY="$API_KEY" \
    RUN_GENERATION=0 \
    RUN_EVAL=1 \
    bash "${UMM_ROOT}/gen/it2i/geditbench/launch.sh"
  done
fi

echo "[gedit-accelerate] done"
