#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UMM_ROOT="${UMM_ROOT:-${BAGEL_ROOT}/third_party/UMM_Eval}"

MODEL_PATH="${MODEL_PATH:-${model_path:-}}"
OUTPUT_PATH="${OUTPUT_PATH:-${output_path:-}}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}"
MAX_MEMORY="${MAX_MEMORY:-80GiB}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-${BAGEL_ROOT}/offload_imgedit}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_NUM_PROCESSES="${EVAL_NUM_PROCESSES:-4}"
THINK_MODE="${THINK_MODE:-0}"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-${UMM_ROOT}/gen/it2i/imgedit/eval/Benchmark}"
METADATA_FILE="${METADATA_FILE:-${BENCHMARK_ROOT}/singleturn/singleturn.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${BENCHMARK_ROOT}/singleturn}"
PROMPTS_JSON="${PROMPTS_JSON:-${BENCHMARK_ROOT}/singleturn/judge_prompt.json}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH/model_path is required"
  exit 1
fi
if [[ -z "$OUTPUT_PATH" ]]; then
  echo "OUTPUT_PATH/output_path is required"
  exit 1
fi
if [[ ! -f "$METADATA_FILE" ]]; then
  echo "ImgEdit metadata does not exist: $METADATA_FILE"
  echo "Install the benchmark assets or set BENCHMARK_ROOT."
  exit 1
fi

export PYTHONPATH="${BAGEL_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_PATH}/bagel"

cd "$BAGEL_ROOT"

echo "[imgedit-accelerate] model=${MODEL_PATH}"
echo "[imgedit-accelerate] output=${OUTPUT_PATH}/bagel"
echo "[imgedit-accelerate] metadata=${METADATA_FILE}"
echo "[imgedit-accelerate] max_latent_size=${MAX_LATENT_SIZE}"
echo "[imgedit-accelerate] max_memory=${MAX_MEMORY}"
echo "[imgedit-accelerate] gpu_list=${GPU_LIST} think_mode=${THINK_MODE}"

if (( RUN_GENERATION != 0 )); then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  N_GPU="${#GPUS[@]}"
  LOG_DIR="${OUTPUT_PATH}/logs"
  mkdir -p "$LOG_DIR"

  pids=()
  THINK_ARGS=()
  case "${THINK_MODE,,}" in
    1|true|yes|on) THINK_ARGS=(--think) ;;
  esac
  for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    log="${LOG_DIR}/gen_worker_$(printf '%02d' "$idx")_gpu_${gpu}.log"
    echo "[imgedit-accelerate] worker ${idx}/${N_GPU} gpu=${gpu} log=${log}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python eval/gen/gen_images_mp_it2i.py \
        --benchmark imgedit \
        --metadata_file "${METADATA_FILE}" \
        --image_root "${IMAGE_ROOT}" \
        --output_dir "${OUTPUT_PATH}/bagel" \
        --shard_id "$idx" \
        --total_shards "$N_GPU" \
        --device 0 \
        --max_latent_size "${MAX_LATENT_SIZE}" \
        --model-path "${MODEL_PATH}" \
        --accelerate_device_map \
        --max_memory "${MAX_MEMORY}" \
        --offload_folder "${OFFLOAD_FOLDER}/worker_${idx}" \
        ${TARGET_INDICES:+--target_indices "$TARGET_INDICES"} \
        "${THINK_ARGS[@]}"
    ) >"$log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[imgedit-accelerate] worker ${idx} failed; tailing log:"
      tail -80 "${LOG_DIR}/gen_worker_$(printf '%02d' "$idx")_gpu_${GPUS[$idx]}.log"
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
    echo "API_KEY (or EVAL_API_KEY_FILE) is required for ImgEdit scoring."
    exit 1
  fi

  RUN_GENERATION=0 \
  RUN_EVAL=1 \
  MODEL_NAME="bagel" \
  STEP_NAME="$(basename "$OUTPUT_PATH")" \
  OUTPUT_PATH="${OUTPUT_PATH}/bagel" \
  IMGEDIT_ROOT="${UMM_ROOT}/gen/it2i/imgedit/eval" \
  IMGEDIT_BENCHMARK_ROOT="${BENCHMARK_ROOT}/singleturn" \
  DATA_PATH="$METADATA_FILE" \
  IMAGE_ROOT="$IMAGE_ROOT" \
  PROMPTS_JSON="$PROMPTS_JSON" \
  EVAL_NUM_PROCESSES="${EVAL_NUM_PROCESSES}" \
  FINALIZE_EVAL="${FINALIZE_EVAL:-1}" \
  API_KEY="${API_KEY}" \
  bash "${UMM_ROOT}/gen/it2i/imgedit/launch.sh"
fi

echo "[imgedit-accelerate] done"
