#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UMM_ROOT="${UMM_ROOT:-${BAGEL_ROOT}/third_party/UMM_Eval}"

MODEL_PATH="${MODEL_PATH:-${model_path:-}}"
MODEL_NAME="${MODEL_NAME:-bagel}"
STEP_NAME="${STEP_NAME:-$(basename "${MODEL_PATH:-step}")}"
for legacy_output in OUTPUT_PATH output_path OUTPUT_DIR output_dir OUTPUT_ROOT RESULT_ROOT_DIR IMGEDIT_OUTPUT_PATH RESULT_DIR OUTPUT_SUBDIR OUTPUT_FOLDER; do
  if [[ -n "${!legacy_output:-}" ]]; then
    echo "${legacy_output} bypasses the experiment layout; use OUTPUT_BASE and EXPERIMENT_NAME instead." >&2
    exit 1
  fi
done
OUTPUT_BASE="${OUTPUT_BASE:-${BAGEL_ROOT}/outputs}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}"
MAX_MEMORY="${MAX_MEMORY:-80GiB}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-${BAGEL_ROOT}/offload_imgedit}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_NUM_PROCESSES="${EVAL_NUM_PROCESSES:-4}"
THINK_MODE="${THINK_MODE:-0}"
TARGET_INDICES="${TARGET_INDICES:-}"
CFG_TEXT_SCALE="${CFG_TEXT_SCALE:-${TXT_CFG:-4.0}}"
CFG_IMG_SCALE="${CFG_IMG_SCALE:-${IMG_CFG:-1.5}}"
# Values not exposed by this wrapper retain the unified generator's defaults.
for fixed_spec in NUM_STEPS:50 NUM_TIMESTEPS:50 IMAGE_SIZE:1024 MAX_IMAGE_SIZE:1024 MIN_IMAGE_SIZE:512 \
    TIMESTEP_SHIFT:3.0 CFG_TYPE:serial_text_img CFG_RENORM:text_channel CFG_RENORM_TYPE:text_channel SEED:42 MAX_LENGTH:2048; do
  fixed_name="${fixed_spec%%:*}"; fixed_value="${fixed_spec#*:}"
  if [[ -n "${!fixed_name:-}" && "${!fixed_name}" != "$fixed_value" \
      && "${!fixed_name}.0" != "$fixed_value" && "${fixed_value}.0" != "${!fixed_name}" ]]; then
    echo "ImgEdit wrapper uses ${fixed_name}=${fixed_value}; this wrapper cannot apply ${!fixed_name}." >&2
    exit 1
  fi
done

BENCHMARK_ROOT="${BENCHMARK_ROOT:-${UMM_ROOT}/gen/it2i/imgedit/eval/Benchmark}"
METADATA_FILE="${METADATA_FILE:-${BENCHMARK_ROOT}/singleturn/singleturn.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${BENCHMARK_ROOT}/singleturn}"
PROMPTS_JSON="${PROMPTS_JSON:-${BENCHMARK_ROOT}/singleturn/judge_prompt.json}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH/model_path is required"
  exit 1
fi
if [[ ! -f "$METADATA_FILE" ]]; then
  echo "ImgEdit metadata does not exist: $METADATA_FILE"
  echo "Install the benchmark assets or set BENCHMARK_ROOT."
  exit 1
fi

export PYTHONPATH="${BAGEL_ROOT}:${PYTHONPATH:-}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
GENERATION_SHARDS="${#GPUS[@]}"
DG2_EFFECTIVE_CONFIG_JSON="$(python - "$CFG_TEXT_SCALE" "$CFG_IMG_SCALE" "$MAX_LATENT_SIZE" \
    "$GENERATION_SHARDS" "$METADATA_FILE" "$IMAGE_ROOT" "$PROMPTS_JSON" \
    "$TARGET_INDICES" "${IMGEDIT_EVAL_MODEL:-gpt-4o}" <<'PY'
import json, sys
txt, img, latent, shards, metadata, images, prompts, target, judge = sys.argv[1:]
print(json.dumps(dict(CFG_TEXT_SCALE=float(txt), CFG_IMG_SCALE=float(img), CFG_TYPE="serial_text_img",
    CFG_RENORM="text_channel", CFG_INTERVAL=[0, 1], TIMESTEP_SHIFT=3.0, NUM_STEPS=50,
    MAX_IMAGE_SIZE=1024, MIN_IMAGE_SIZE=512, MAX_LATENT_SIZE=int(latent), SEED=42,
    SEED_STRATEGY="base_plus_stable_index", MAX_LENGTH=2048, TEMPERATURE=0.3, CROP_BBOX=True,
    GENERATOR="gen_images_mp_it2i.py", GENERATION_SHARDS=int(shards), METADATA_FILE=metadata,
    IMAGE_ROOT=images, PROMPTS_JSON=prompts, TARGET_INDICES=target, IMGEDIT_EVAL_MODEL=judge)))
PY
)"
SETTING_LABEL="$(PYTHONPATH="${UMM_ROOT}/gen/common:${PYTHONPATH:-}" python - "$CFG_TEXT_SCALE" "$CFG_IMG_SCALE" <<'PY'
import sys, output_paths
print(output_paths.derive_it2i_folder(sys.argv[1], sys.argv[2], "text_channel", 3.0, 1024))
PY
)"
source "${SCRIPT_DIR}/experiment_layout.sh"
dg2_prepare_experiment imgedit "$SETTING_LABEL"
OUTPUT_PATH="$DG2_RESULT_DIR"
mkdir -p "${OUTPUT_PATH}/bagel"

source "${SCRIPT_DIR}/coverage_report.sh"
dg2_write_coverage_reports() {
  dg2_run_coverage_audit --bench imgedit --results-root "$OUTPUT_PATH" \
      --metadata-file "${COVERAGE_METADATA_FILE:-$METADATA_FILE}" \
      --image-dir "${COVERAGE_IMAGE_DIR:-${OUTPUT_PATH}/bagel}" --judge-root "${COVERAGE_JUDGE_ROOT:-$OUTPUT_PATH}"
}
dg2_install_coverage_trap

cd "$BAGEL_ROOT"

echo "[imgedit-accelerate] model=${MODEL_PATH}"
echo "[imgedit-accelerate] result_dir=${DG2_RESULT_DIR}"
echo "[imgedit-accelerate] output=${OUTPUT_PATH}/bagel"
echo "[imgedit-accelerate] metadata=${METADATA_FILE}"
echo "[imgedit-accelerate] max_latent_size=${MAX_LATENT_SIZE}"
echo "[imgedit-accelerate] max_memory=${MAX_MEMORY}"
echo "[imgedit-accelerate] gpu_list=${GPU_LIST} think_mode=${THINK_MODE}"

if (( RUN_GENERATION != 0 )); then
  source activate bagel
  export PATH=/opt/conda/envs/bagel/bin:$PATH
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  N_GPU="${#GPUS[@]}"
  LOG_DIR="${OUTPUT_PATH}/logs"
  mkdir -p "$LOG_DIR"

  pids=()
  THINK_ARGS=()
  case "$(printf '%s' "$THINK_MODE" | tr '[:upper:]' '[:lower:]')" in
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
        --cfg_text_scale "$CFG_TEXT_SCALE" \
        --cfg_img_scale "$CFG_IMG_SCALE" \
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
  source activate geneval
  export PATH=/opt/conda/envs/geneval/bin:$PATH
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
  RESULT_ROOT_DIR="$DG2_RESULT_DIR" \
  LOG_DIR="${DG2_RESULT_DIR}/logs" \
  IMGEDIT_ROOT="${UMM_ROOT}/gen/it2i/imgedit/eval" \
  IMGEDIT_BENCHMARK_ROOT="${BENCHMARK_ROOT}/singleturn" \
  DATA_PATH="$METADATA_FILE" \
  IMAGE_ROOT="$IMAGE_ROOT" \
  PROMPTS_JSON="$PROMPTS_JSON" \
  EVAL_NUM_PROCESSES="${EVAL_NUM_PROCESSES}" \
  FINALIZE_EVAL="${FINALIZE_EVAL:-1}" \
  API_KEY="${API_KEY}" \
  IMGEDIT_EVAL_MODEL="${IMGEDIT_EVAL_MODEL:-gpt-4o}" \
  bash "${UMM_ROOT}/gen/it2i/imgedit/launch.sh"
fi

echo "[imgedit-accelerate] done"
