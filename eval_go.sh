#!/usr/bin/env bash
# Run BAGEL image-editing evaluations with the local wrappers.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DG2_CONFIG="${DG2_CONFIG:-${SCRIPT_DIR}/config/eval.env}"
if [[ -f "$DG2_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$DG2_CONFIG"
fi

# Environments are activated in each generation/evaluation stage.

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/models/BAGEL-7B-MoT}"
MODEL_NAME="${MODEL_NAME:-bagel}"
STEP_NAME="${STEP_NAME:-$(basename "$MODEL_PATH")}"
TASK="${TASK:-all}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
MAX_MEMORY="${MAX_MEMORY:-80GiB}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FINALIZE_EVAL="${FINALIZE_EVAL:-1}"
THINK_MODE="${THINK_MODE:-0}"

OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/outputs}"
if [[ -z "${EXPERIMENT_NAME:-}" ]]; then
  echo "EXPERIMENT_NAME is required; set it to the name of this experiment." >&2
  exit 1
fi
for legacy_output in IMGEDIT_OUTPUT_PATH GEDIT_OUTPUT_DIR RISE_OUTPUT_BASE EDIT_COMPASS_OUTPUT_BASE \
    EDIT_COMPASS_OUTPUT_ROOT OUTPUT_ROOT OUTPUT_PATH output_path OUTPUT_DIR output_dir RESULT_DIR RESULT_ROOT_DIR OUTPUT_SUBDIR OUTPUT_FOLDER; do
  if [[ -n "${!legacy_output:-}" ]]; then
    echo "${legacy_output} bypasses the experiment layout; use OUTPUT_BASE and EXPERIMENT_NAME instead." >&2
    exit 1
  fi
done

# Values sourced from eval.env are ordinary shell variables; expose the editing
# configuration explicitly so each child records and uses the requested values.
for configuration_name in EXPERIMENT_SETTING COVERAGE_REPORT COVERAGE_PYTHON COVERAGE_METADATA_FILE COVERAGE_IMAGE_DIR COVERAGE_JUDGE_ROOT \
    CFG_TEXT_SCALE CFG_IMG_SCALE CFG_TYPE CFG_RENORM_TYPE TXT_CFG IMG_CFG CFG_RENORM \
    NUM_STEPS NUM_TIMESTEPS IMAGE_SIZE MAX_IMAGE_SIZE MIN_IMAGE_SIZE MAX_LATENT_SIZE TIMESTEP_SHIFT \
    BENCHMARK_ROOT METADATA_FILE IMAGE_ROOT PROMPTS_JSON GEDIT_DATASET_PATH INSTRUCTION_LANGUAGES TARGET_INDICES \
    RISE_DATA_DIR RISE_DATA_JSON RISE_INPUT_ROOT EVAL_MODEL_NAME EVAL_BACKBONE IMGEDIT_EVAL_MODEL JUDGE_MODEL \
    EDIT_COMPASS_EVAL_MODEL_NAME EDIT_COMPASS_API_MAX_IMAGE_SIDE METRICS EVAL_NUM_PROCESSES EVAL_NUM_WORKERS \
    UMM_ROOT; do
  if [[ -n "${!configuration_name:-}" ]]; then
    export "$configuration_name"
  fi
done

run_imgedit() {
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  GPU_LIST="$GPU_LIST" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
  OUTPUT_BASE="$OUTPUT_BASE" EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  MAX_MEMORY="$MAX_MEMORY" \
  RUN_GENERATION="$RUN_GENERATION" \
  RUN_EVAL="$RUN_EVAL" \
  FINALIZE_EVAL="$FINALIZE_EVAL" \
  THINK_MODE="$THINK_MODE" \
  TARGET_INDICES="${TARGET_INDICES:-}" \
  bash scripts/eval/run_imgedit_accelerate.sh
}

run_gedit() {
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  GPU_LIST="$GPU_LIST" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
  OUTPUT_BASE="$OUTPUT_BASE" EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  MAX_MEMORY="$MAX_MEMORY" \
  RUN_GENERATION="$RUN_GENERATION" \
  RUN_EVAL="$RUN_EVAL" \
  FINALIZE_EVAL="$FINALIZE_EVAL" \
  THINK_MODE="$THINK_MODE" \
  TARGET_INDICES="${TARGET_INDICES:-}" \
  bash scripts/eval/run_gedit_accelerate.sh
}

run_rise() {
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  GPU_LIST="$GPU_LIST" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  STEP_NAME="$STEP_NAME" \
  OUTPUT_BASE="$OUTPUT_BASE" EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  RUN_GENERATION="$RUN_GENERATION" \
  RUN_EVAL="$RUN_EVAL" \
  THINK_MODE="$THINK_MODE" \
  TXT_CFG="${TXT_CFG:-4.0}" \
  IMG_CFG="${IMG_CFG:-2.0}" \
  CFG_RENORM="${CFG_RENORM:-text_channel}" \
  TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}" \
  IMAGE_SIZE="${IMAGE_SIZE:-1024}" \
  NUM_STEPS="${NUM_STEPS:-50}" \
  MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}" \
  MAX_MEMORY="$MAX_MEMORY" \
  TARGET_INDICES="${TARGET_INDICES:-}" \
  API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}" \
  bash scripts/eval/run_rise_accelerate.sh
}

run_editcompass() {
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  GPU_LIST="$GPU_LIST" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  STEP_NAME="$STEP_NAME" \
  OUTPUT_BASE="$OUTPUT_BASE" EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  EDIT_COMPASS_DATA_ROOT="${EDIT_COMPASS_DATA_ROOT:-}" \
  EDIT_COMPASS_BENCHMARK_ROOT="${EDIT_COMPASS_BENCHMARK_ROOT:-}" \
  EDIT_COMPASS_LANGUAGE="${EDIT_COMPASS_LANGUAGE:-en}" \
  PARTS="${PARTS:-Part1 Part2}" \
  MAX_MEMORY="$MAX_MEMORY" \
  RUN_GENERATION="$RUN_GENERATION" \
  RUN_EVAL="$RUN_EVAL" \
  RUN_SUMMARY="${RUN_SUMMARY:-1}" \
  THINK_MODE="$THINK_MODE" \
  API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}" \
  API_BASE="${API_BASE:-${OPENAI_BASE_URL:-}}" \
  bash scripts/eval/run_edit_compass_accelerate.sh
}

echo "[eval-go] model=${MODEL_PATH}"
echo "[eval-go] experiment=${EXPERIMENT_NAME} output_base=${OUTPUT_BASE}"
echo "[eval-go] task=${TASK} gpu_list=${GPU_LIST} max_memory=${MAX_MEMORY}"
echo "[eval-go] run_generation=${RUN_GENERATION} run_eval=${RUN_EVAL} think_mode=${THINK_MODE}"

case "$TASK" in
  imgedit)
    run_imgedit
    ;;
  gedit|geditbench|geeditbench|gedit_bench)
    run_gedit
    ;;
  rise|risebench|rise_bench)
    run_rise
    ;;
  editcompass|edit_compass|edit-compass)
    run_editcompass
    ;;
  both)
    run_imgedit
    run_gedit
    ;;
  all)
    run_imgedit
    run_gedit
    run_rise
    run_editcompass
    ;;
  *)
    echo "Unsupported TASK=${TASK}; expected imgedit, geditbench, rise, editcompass, both, or all"
    exit 1
    ;;
esac
