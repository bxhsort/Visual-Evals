#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-bagel}"
STEP_NAME="${STEP_NAME:-$(basename "${MODEL_PATH:-step}")}"
for legacy_output in EDIT_COMPASS_OUTPUT_BASE EDIT_COMPASS_OUTPUT_ROOT OUTPUT_ROOT RESULT_DIR RESULT_ROOT_DIR OUTPUT_PATH output_path OUTPUT_DIR output_dir OUTPUT_SUBDIR OUTPUT_FOLDER; do
  if [[ -n "${!legacy_output:-}" ]]; then
    echo "${legacy_output} bypasses the experiment layout; use OUTPUT_BASE and EXPERIMENT_NAME instead." >&2
    exit 1
  fi
done
DATA_ROOT="${EDIT_COMPASS_DATA_ROOT:-${DATA_ROOT:-${BAGEL_ROOT}/third_party/Edit-Compass-data}}"
BENCHMARK_ROOT="${EDIT_COMPASS_BENCHMARK_ROOT:-${BENCHMARK_ROOT:-${BAGEL_ROOT}/third_party/Edit-Compass-and-EditReward-Compass}}"
OUTPUT_BASE="${OUTPUT_BASE:-${BAGEL_ROOT}/outputs}"
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
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
EDIT_COMPASS_API_MAX_IMAGE_SIDE="${EDIT_COMPASS_API_MAX_IMAGE_SIDE:-2048}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
GENERATION_SHARDS="${#GPUS[@]}"
DG2_EFFECTIVE_CONFIG_JSON="$(python - "$CFG_TEXT_SCALE" "$CFG_IMG_SCALE" "$NUM_TIMESTEPS" \
    "$MAX_IMAGE_SIZE" "$MIN_IMAGE_SIZE" "$TIMESTEP_SHIFT" "$GENERATION_SHARDS" \
    "$DATA_ROOT" "$BENCHMARK_ROOT" "$LANGUAGE" "$PARTS" "$METRICS" \
    "${EDIT_COMPASS_EVAL_MODEL_NAME:-gemini-3.1-pro-preview}" "$EDIT_COMPASS_API_MAX_IMAGE_SIDE" <<'PY'
import json, sys
txt, img, steps, size, minimum, shift, shards, data, benchmark, language, parts, metrics, judge, judge_side = sys.argv[1:]
print(json.dumps(dict(CFG_TEXT_SCALE=float(txt), CFG_IMG_SCALE=float(img), CFG_TYPE="serial_text_img",
    CFG_RENORM="text_channel", CFG_INTERVAL=[0, 1], NUM_TIMESTEPS=int(steps),
    MAX_IMAGE_SIZE=int(size), MIN_IMAGE_SIZE=int(minimum), TIMESTEP_SHIFT=float(shift),
    MAX_LATENT_SIZE=64, SEED=42, GENERATION_SHARDS=int(shards), DATA_ROOT=data,
    BENCHMARK_ROOT=benchmark, EDIT_COMPASS_LANGUAGE=language, PARTS=parts.split(), METRICS=metrics.split(),
    EDIT_COMPASS_EVAL_MODEL_NAME=judge, EDIT_COMPASS_API_MAX_IMAGE_SIDE=int(judge_side))))
PY
)"
SETTING_LABEL="$(PYTHONPATH="${BAGEL_ROOT}/third_party/UMM_Eval/gen/common:${PYTHONPATH:-}" python - \
    "$CFG_TEXT_SCALE" "$CFG_IMG_SCALE" "$TIMESTEP_SHIFT" "$MAX_IMAGE_SIZE" <<'PY'
import sys, output_paths
print(output_paths.derive_it2i_folder(sys.argv[1], sys.argv[2], "text_channel", sys.argv[3], sys.argv[4]))
PY
)"
source "${SCRIPT_DIR}/experiment_layout.sh"
dg2_prepare_experiment editcompass "$SETTING_LABEL"
OUTPUT_ROOT="$DG2_RESULT_DIR"

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
echo "[edit-compass] result_dir=${DG2_RESULT_DIR}"
echo "[edit-compass] output=${OUTPUT_ROOT}/${MODEL_NAME}/${LANGUAGE}"
echo "[edit-compass] data_root=${DATA_ROOT}"
echo "[edit-compass] parts=${PARTS} language=${LANGUAGE}"
echo "[edit-compass] run_generation=${RUN_GENERATION} run_eval=${RUN_EVAL} run_summary=${RUN_SUMMARY}"
echo "[edit-compass] gpu_list=${GPU_LIST} think_mode=${THINK_MODE}"

read -r -a PART_ARGS <<< "$PARTS"
read -r -a METRIC_ARGS <<< "$METRICS"

source "${SCRIPT_DIR}/coverage_report.sh"
dg2_write_coverage_reports() {
  local audit_root="${OUTPUT_ROOT}/${MODEL_NAME}/${LANGUAGE}"
  dg2_run_coverage_audit --bench editcompass --results-root "$OUTPUT_ROOT" --language "$LANGUAGE" \
      --out-prefix "${OUTPUT_ROOT}/coverage_editcompass_${LANGUAGE}" \
      --model-name "$MODEL_NAME" --parts "${PART_ARGS[@]}" --metadata-file "${COVERAGE_METADATA_FILE:-$DATA_ROOT}" \
      --image-dir "${COVERAGE_IMAGE_DIR:-$audit_root}" --judge-root "${COVERAGE_JUDGE_ROOT:-${OUTPUT_ROOT}/${MODEL_NAME}}"
}
dg2_install_coverage_trap

OVERWRITE_ARGS=()
if (( OVERWRITE != 0 )); then
  OVERWRITE_ARGS=(--overwrite)
fi
THINK_ARGS=()
case "$(printf '%s' "$THINK_MODE" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) THINK_ARGS=(--think) ;;
esac

if (( RUN_GENERATION != 0 )); then
  source activate bagel
  export PATH=/opt/conda/envs/bagel/bin:$PATH
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
        --timestep_shift "$TIMESTEP_SHIFT" \
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
  source activate geneval
  export PATH=/opt/conda/envs/geneval/bin:$PATH
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
