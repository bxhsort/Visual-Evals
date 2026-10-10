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

BAGEL_ENV="${BAGEL_ENV:-${SCRIPT_DIR}/.venv}"
if [[ -f "${BAGEL_ENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${BAGEL_ENV}/bin/activate"
elif [[ -d "${BAGEL_ENV}/conda-meta" ]]; then
  if [[ -z "${CONDA_SH:-}" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  fi
  [[ -f "${CONDA_SH:-}" ]] || { echo "CONDA_SH is required to activate ${BAGEL_ENV}"; exit 1; }
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$BAGEL_ENV"
fi

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

IMGEDIT_OUTPUT_PATH="${IMGEDIT_OUTPUT_PATH:-${SCRIPT_DIR}/outputs/imgedit/${STEP_NAME}}"
GEDIT_OUTPUT_DIR="${GEDIT_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/geditbench/${STEP_NAME}}"
RISE_OUTPUT_BASE="${RISE_OUTPUT_BASE:-${SCRIPT_DIR}/outputs}"
EDIT_COMPASS_OUTPUT_BASE="${EDIT_COMPASS_OUTPUT_BASE:-${SCRIPT_DIR}/outputs/editcompass}"

run_imgedit() {
  echo "[eval-go] ImgEdit -> ${IMGEDIT_OUTPUT_PATH}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  MODEL_PATH="$MODEL_PATH" \
  OUTPUT_PATH="$IMGEDIT_OUTPUT_PATH" \
  MAX_MEMORY="$MAX_MEMORY" \
  RUN_GENERATION="$RUN_GENERATION" \
  RUN_EVAL="$RUN_EVAL" \
  FINALIZE_EVAL="$FINALIZE_EVAL" \
  THINK_MODE="$THINK_MODE" \
  TARGET_INDICES="${TARGET_INDICES:-}" \
  bash scripts/eval/run_imgedit_accelerate.sh
}

run_gedit() {
  echo "[eval-go] GEdit -> ${GEDIT_OUTPUT_DIR}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  MODEL_PATH="$MODEL_PATH" \
  OUTPUT_DIR="$GEDIT_OUTPUT_DIR" \
  MAX_MEMORY="$MAX_MEMORY" \
  RUN_GENERATION="$RUN_GENERATION" \
  RUN_EVAL="$RUN_EVAL" \
  FINALIZE_EVAL="$FINALIZE_EVAL" \
  THINK_MODE="$THINK_MODE" \
  TARGET_INDICES="${TARGET_INDICES:-}" \
  bash scripts/eval/run_gedit_accelerate.sh
}

run_rise() {
  echo "[eval-go] RISE -> ${RISE_OUTPUT_BASE}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  GPU_LIST="$GPU_LIST" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  STEP_NAME="$STEP_NAME" \
  OUTPUT_BASE="$RISE_OUTPUT_BASE" \
  RUN_GENERATION="$RUN_GENERATION" \
  RUN_EVAL="$RUN_EVAL" \
  THINK_MODE="${THINK_MODE:-1}" \
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
  echo "[eval-go] EditCompass -> ${EDIT_COMPASS_OUTPUT_BASE}/${STEP_NAME}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  GPU_LIST="$GPU_LIST" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  STEP_NAME="$STEP_NAME" \
  EDIT_COMPASS_OUTPUT_BASE="$EDIT_COMPASS_OUTPUT_BASE" \
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
