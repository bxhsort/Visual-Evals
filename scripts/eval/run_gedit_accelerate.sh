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
for legacy_output in OUTPUT_DIR output_dir OUTPUT_PATH output_path OUTPUT_ROOT RESULT_ROOT_DIR GEDIT_OUTPUT_DIR RESULT_DIR OUTPUT_SUBDIR OUTPUT_FOLDER; do
  if [[ -n "${!legacy_output:-}" ]]; then
    echo "${legacy_output} bypasses the experiment layout; use OUTPUT_BASE and EXPERIMENT_NAME instead." >&2
    exit 1
  fi
done
OUTPUT_BASE="${OUTPUT_BASE:-${BAGEL_ROOT}/outputs}"
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
TARGET_INDICES="${TARGET_INDICES:-}"
GEDIT_DATASET_PATH="${GEDIT_DATASET_PATH:-${UMM_ROOT}/gen/it2i/geditbench/eval/GEdit-Bench}"

CFG_TEXT_SCALE="${CFG_TEXT_SCALE:-4.0}"
CFG_IMG_SCALE="${CFG_IMG_SCALE:-1.5}"
# Values not exposed by this wrapper retain the unified generator's defaults.
for fixed_spec in NUM_STEPS:50 NUM_TIMESTEPS:50 IMAGE_SIZE:1024 MAX_IMAGE_SIZE:1024 MIN_IMAGE_SIZE:512 \
    TIMESTEP_SHIFT:3.0 CFG_TYPE:serial_text_img CFG_RENORM:text_channel CFG_RENORM_TYPE:text_channel \
    MAX_LATENT_SIZE:64 SEED:42 MAX_LENGTH:2048; do
  fixed_name="${fixed_spec%%:*}"; fixed_value="${fixed_spec#*:}"
  if [[ -n "${!fixed_name:-}" && "${!fixed_name}" != "$fixed_value" \
      && "${!fixed_name}.0" != "$fixed_value" && "${fixed_value}.0" != "${!fixed_name}" ]]; then
    echo "GEdit wrapper uses ${fixed_name}=${fixed_value}; this wrapper cannot apply ${!fixed_name}." >&2
    exit 1
  fi
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH/model_path is required"
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

GENERATION_SHARDS="$N_GPU"
DG2_EFFECTIVE_CONFIG_JSON="$(python - "$CFG_TEXT_SCALE" "$CFG_IMG_SCALE" \
    "$GENERATION_SHARDS" "$GEDIT_DATASET_PATH" "$INSTRUCTION_LANGUAGES" \
    "$EVAL_MODEL_NAME" "${EVAL_BACKBONE:-gpt4o}" "$TARGET_INDICES" <<'PY'
import json, sys
txt, img, shards, dataset, languages, judge_name, backbone, target = sys.argv[1:]
print(json.dumps(dict(CFG_TEXT_SCALE=float(txt), CFG_IMG_SCALE=float(img), CFG_TYPE="serial_text_img",
    CFG_RENORM="text_channel", CFG_INTERVAL=[0.4, 1], TIMESTEP_SHIFT=3.0, NUM_STEPS=50,
    MAX_IMAGE_SIZE=1024, MIN_IMAGE_SIZE=512, MAX_LATENT_SIZE=64, SEED=42, TEMPERATURE=0.3,
    SEED_STRATEGY="base_plus_stable_index", MAX_LENGTH=2048, CROP_BBOX=False,
    GENERATOR="gen_images_mp_it2i.py", GENERATION_SHARDS=int(shards), GEDIT_DATASET_PATH=dataset,
    INSTRUCTION_LANGUAGES=languages.split(), EVAL_MODEL_NAME=judge_name, EVAL_BACKBONE=backbone,
    TARGET_INDICES=target)))
PY
)"
SETTING_LABEL="$(PYTHONPATH="${UMM_ROOT}/gen/common:${PYTHONPATH:-}" python - "$CFG_TEXT_SCALE" "$CFG_IMG_SCALE" <<'PY'
import sys, output_paths
print(output_paths.derive_it2i_folder(sys.argv[1], sys.argv[2], "text_channel", 3.0, 1024))
PY
)"
source "${SCRIPT_DIR}/experiment_layout.sh"
dg2_prepare_experiment geditbench "$SETTING_LABEL"
OUTPUT_DIR="$DG2_RESULT_DIR"
GEN_DIR="${OUTPUT_DIR}/gen_image"
LOG_DIR="${OUTPUT_DIR}/logs"
EVAL_IMAGE_DIR="${GEN_DIR}/fullset"

source "${SCRIPT_DIR}/coverage_report.sh"
dg2_write_coverage_reports() {
  local language audit_root metadata
  audit_root="$OUTPUT_DIR"
  metadata="${COVERAGE_METADATA_FILE:-${audit_root}/coverage_expected_geditbench.json}"
  if [[ -z "${COVERAGE_METADATA_FILE:-}" ]]; then
    if ! dg2_export_gedit_expected "$GEDIT_DATASET_PATH" "$metadata"; then
      echo "[coverage] WARNING: GEdit expected IDs could not be exported from the local dataset; set COVERAGE_METADATA_FILE to an exported key/task_type/instruction_language JSON. Denominator is unknown." >&2
      # Do not certify coverage using an old export from another run or dataset.
      metadata="$GEDIT_DATASET_PATH"
    fi
  fi
  for language in $INSTRUCTION_LANGUAGES; do
    dg2_run_coverage_audit --bench geditbench --results-root "$audit_root" --language "$language" \
      --metadata-file "$metadata" --image-dir "${COVERAGE_IMAGE_DIR:-$EVAL_IMAGE_DIR}" \
      --judge-root "${COVERAGE_JUDGE_ROOT:-${audit_root}/${EVAL_MODEL_NAME}/${EVAL_BACKBONE:-gpt4o}}" --model-name "$EVAL_MODEL_NAME"
  done
}
dg2_install_coverage_trap

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
  source activate bagel
  export PATH=/opt/conda/envs/bagel/bin:$PATH
  pids=()
  THINK_ARGS=()
  case "$(printf '%s' "$THINK_MODE" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) THINK_ARGS=(--think) ;;
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
  source activate geneval
  export PATH=/opt/conda/envs/geneval/bin:$PATH
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
    RESULT_ROOT_DIR="$OUTPUT_DIR" \
    LOG_DIR="${OUTPUT_DIR}/logs" \
    EVAL_BACKBONE="${EVAL_BACKBONE:-gpt4o}" \
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
