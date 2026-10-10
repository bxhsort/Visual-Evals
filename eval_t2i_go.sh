#!/usr/bin/env bash
# One-click BAGEL text-to-image evaluation.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DG2_CONFIG="${DG2_CONFIG:-${SCRIPT_DIR}/config/eval.env}"
if [[ -f "$DG2_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$DG2_CONFIG"
fi

: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME to the name of this experiment}"
GENEVAL_PROMPT_MODE="${GENEVAL_PROMPT_MODE:-long}"
case "$GENEVAL_PROMPT_MODE" in
  short|long|both) ;;
  *) echo "GENEVAL_PROMPT_MODE must be short, long, or both" >&2; exit 1 ;;
esac
if [[ "$GENEVAL_PROMPT_MODE" == "both" && -n "${METADATA_FILE:-}" ]]; then
  echo "For GENEVAL_PROMPT_MODE=both, use GENEVAL_SHORT_METADATA_FILE and GENEVAL_LONG_METADATA_FILE instead of METADATA_FILE." >&2
  exit 1
fi

# Environments are activated in each generation/evaluation stage.

MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_user/wangdianyi-240107110022/model_zoo/BAGEL-7B-MoT}"
MODEL_NAME="${MODEL_NAME:-baseline}"
STEP_NAME="${STEP_NAME:-$(basename "$MODEL_PATH")}"
TASK="${TASK:-all}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/outputs}"

RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
MAX_MEMORY="${MAX_MEMORY:-80GiB}"

NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
CFG_NORM="${CFG_NORM:-global}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-}"

GENEVAL_IMAGE_SIZE="${GENEVAL_IMAGE_SIZE:-1024}"
DPGBENCH_IMAGE_SIZE="${DPGBENCH_IMAGE_SIZE:-1024}"
LONGTEXTBENCH_IMAGE_SIZE="${LONGTEXTBENCH_IMAGE_SIZE:-1024}"
CVTG_2K_IMAGE_SIZE="${CVTG_2K_IMAGE_SIZE:-1024}"
QWENIMAGEBENCH_IMAGE_SIZE="${QWENIMAGEBENCH_IMAGE_SIZE:-1024}"
TIIFBENCH_IMAGE_SIZE="${TIIFBENCH_IMAGE_SIZE:-1024}"
ONEIGBENCH_IMAGE_SIZE="${ONEIGBENCH_IMAGE_SIZE:-1024}"
WISE_IMAGE_SIZE="${WISE_IMAGE_SIZE:-1024}"
BIZGENEVAL_IMAGE_SIZE="${BIZGENEVAL_IMAGE_SIZE:-1024}"
QIB_LANGS="${QIB_LANGS:-en zh}"
LONGTEXTBENCH_LANGS="${LONGTEXTBENCH_LANGS:-en}"
ONEIGBENCH_LANGS="${ONEIGBENCH_LANGS:-en zh}"

run_one() {
  local benchmark="$1"
  local image_size="$2"
  local geneval_prompt_mode="${3:-$GENEVAL_PROMPT_MODE}"
  local max_latent_size="$MAX_LATENT_SIZE"
  local batch_size="${BATCH_SIZE:-4}"
  [[ "$benchmark" == "wise" || "$benchmark" == "tiifbench" ]] && batch_size="${BATCH_SIZE:-1}"
  if [[ -z "$max_latent_size" ]]; then
    max_latent_size=$(( image_size > 1024 ? image_size / 16 : 64 ))
  fi
  echo "[bagel-t2i-go] ${benchmark} -> IMAGE_SIZE=${image_size}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  GPU_LIST="$GPU_LIST" \
  BENCHMARK="$benchmark" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  STEP_NAME="$STEP_NAME" \
  OUTPUT_BASE="$OUTPUT_BASE" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  EXPERIMENT_SETTING="${EXPERIMENT_SETTING:-}" \
  EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-}" \
  GENEVAL_PROMPT_MODE="$geneval_prompt_mode" \
  GENEVAL_SHORT_METADATA_FILE="${GENEVAL_SHORT_METADATA_FILE:-}" \
  GENEVAL_LONG_METADATA_FILE="${GENEVAL_LONG_METADATA_FILE:-}" \
  METADATA_FILE="${METADATA_FILE:-}" \
  QIB_METADATA_FILE_EN="${QIB_METADATA_FILE_EN:-}" \
  QIB_METADATA_FILE_ZH="${QIB_METADATA_FILE_ZH:-}" \
  LONGTEXTBENCH_METADATA_FILE_EN="${LONGTEXTBENCH_METADATA_FILE_EN:-}" \
  LONGTEXTBENCH_METADATA_FILE_ZH="${LONGTEXTBENCH_METADATA_FILE_ZH:-}" \
  TEXTBENCH_ROOT="${TEXTBENCH_ROOT:-}" \
  TEXTCRAFTER_ROOT="${TEXTCRAFTER_ROOT:-}" \
  RUN_GENERATION="$RUN_GENERATION" \
  RUN_EVAL="$RUN_EVAL" \
  COVERAGE_REPORT="${COVERAGE_REPORT:-1}" \
  COVERAGE_PYTHON="${COVERAGE_PYTHON:-}" \
  COVERAGE_METADATA_FILE="${COVERAGE_METADATA_FILE:-}" \
  COVERAGE_IMAGE_DIR="${COVERAGE_IMAGE_DIR:-}" \
  COVERAGE_JUDGE_ROOT="${COVERAGE_JUDGE_ROOT:-}" \
  IMAGE_SIZE="$image_size" \
  NUM_STEPS="$NUM_STEPS" \
  CFG_SCALE="$CFG_SCALE" \
  TIMESTEP_SHIFT="$TIMESTEP_SHIFT" \
  CFG_NORM="$CFG_NORM" \
  BATCH_SIZE="$batch_size" \
  NUM_IMAGES="${NUM_IMAGES:-}" \
  SEED="${SEED:-42}" \
  MAX_LATENT_SIZE="$max_latent_size" \
  MAX_MEMORY="$MAX_MEMORY" \
  TARGET_INDICES="${TARGET_INDICES:-}" \
  TARGET_PROMPT_IDS="${TARGET_PROMPT_IDS:-}" \
  TARGET_KEYS="${TARGET_KEYS:-}" \
  LONGTEXTBENCH_LANGS="$LONGTEXTBENCH_LANGS" \
  ONEIGBENCH_LANGS="$ONEIGBENCH_LANGS" \
  GRID_ROWS="${GRID_ROWS:-}" \
  GRID_COLS="${GRID_COLS:-}" \
  ONEIGBENCH_EVAL_EXECUTOR="${ONEIGBENCH_EVAL_EXECUTOR:-}" \
  ONEIGBENCH_API_EVAL="${ONEIGBENCH_API_EVAL:-}" \
  CVTG_SUBSETS="${CVTG_SUBSETS:-}" \
  CVTG_AREAS="${CVTG_AREAS:-}" \
  BIZGENEVAL_DEFAULT_ASPECT_RATIO="${BIZGENEVAL_DEFAULT_ASPECT_RATIO:-}" \
  THINK_MODE="${THINK_MODE:-0}" \
  OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}" \
  WISE_SUBDIR="${WISE_SUBDIR:-}" \
  WISE_CATEGORIES="${WISE_CATEGORIES:-}" \
  WISE_DATA_ROOT="${WISE_DATA_ROOT:-}" \
  EVAL_MODEL="${EVAL_MODEL:-}" \
  EVAL_BASE_URL="${EVAL_BASE_URL:-}" \
  EVAL_API_BASE_URL="${EVAL_API_BASE_URL:-}" \
  OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
  JUDGE_MODEL="${JUDGE_MODEL:-}" \
  JUDGE_API_BASE="${JUDGE_API_BASE:-}" \
  EVAL_MAX_ITEMS="${EVAL_MAX_ITEMS:-}" \
  RESIZE_TO="${RESIZE_TO:-}" \
  TIIF_BENCH_ROOT="${TIIF_BENCH_ROOT:-}" \
  TIIFBENCH_SPLIT="${TIIFBENCH_SPLIT:-}" \
  TIIFBENCH_PROMPT_DIR="${TIIFBENCH_PROMPT_DIR:-}" \
  TIIFBENCH_EVAL_PROMPT_DIR="${TIIFBENCH_EVAL_PROMPT_DIR:-}" \
  TIIFBENCH_SPECIFIC_FILE="${TIIFBENCH_SPECIFIC_FILE:-}" \
  TIIFBENCH_EVAL_MODEL="${TIIFBENCH_EVAL_MODEL:-}" \
  TIIFBENCH_EVAL_BASE_URL="${TIIFBENCH_EVAL_BASE_URL:-}" \
  TIIFBENCH_API_VERSION="${TIIFBENCH_API_VERSION:-}" \
  API_KEY="${API_KEY:-}" \
  QIB_LANGS="$QIB_LANGS" \
  QIB_EVAL_BACKEND="${QIB_EVAL_BACKEND:-}" \
  QIB_API_ROOT="${QIB_API_ROOT:-}" \
  QIB_API_LOCAL_BENCH_ROOT="${QIB_API_LOCAL_BENCH_ROOT:-}" \
  INPUT_JSON_en="${INPUT_JSON_en:-}" \
  INPUT_JSON_cn="${INPUT_JSON_cn:-}" \
  QIB_REWARD_BASE_URL="${QIB_REWARD_BASE_URL:-}" \
  QIB_REWARD_BASE_URLS="${QIB_REWARD_BASE_URLS:-}" \
  REWARD_BASE_URL="${REWARD_BASE_URL:-}" \
  REWARD_BASE_URLS="${REWARD_BASE_URLS:-}" \
  QIB_EVAL_WORKERS="${QIB_EVAL_WORKERS:-}" \
  QIB_DIM_MODE="${QIB_DIM_MODE:-}" \
  QIB_REWARD_TIMEOUT="${QIB_REWARD_TIMEOUT:-}" \
  QIB_REWARD_MAX_RETRIES="${QIB_REWARD_MAX_RETRIES:-}" \
  QIB_REWARD_IMAGE_SOURCE="${QIB_REWARD_IMAGE_SOURCE:-}" \
  QJUDGER_MODEL_PATH="${QJUDGER_MODEL_PATH:-}" \
  QJUDGER_BATCH_SIZE="${QJUDGER_BATCH_SIZE:-}" \
  QJUDGER_MAX_NEW_TOKENS="${QJUDGER_MAX_NEW_TOKENS:-}" \
  QJUDGER_NUM_SHARDS="${QJUDGER_NUM_SHARDS:-}" \
  bash scripts/eval/run_t2i_accelerate.sh
}

run_geneval_modes() {
  case "$GENEVAL_PROMPT_MODE" in
    short|long) run_one geneval "$GENEVAL_IMAGE_SIZE" "$GENEVAL_PROMPT_MODE" ;;
    both)
      run_one geneval "$GENEVAL_IMAGE_SIZE" short
      run_one geneval "$GENEVAL_IMAGE_SIZE" long
      ;;
  esac
}

echo "[bagel-t2i-go] model=${MODEL_PATH}"
echo "[bagel-t2i-go] model_name=${MODEL_NAME} step_name=${STEP_NAME}"
echo "[bagel-t2i-go] task=${TASK} gpu_list=${GPU_LIST}"
echo "[bagel-t2i-go] output_base=${OUTPUT_BASE}"
echo "[bagel-t2i-go] experiment=${EXPERIMENT_NAME} geneval_prompt_mode=${GENEVAL_PROMPT_MODE}"
echo "[bagel-t2i-go] run_generation=${RUN_GENERATION} run_eval=${RUN_EVAL}"
echo "[bagel-t2i-go] cfg_scale=${CFG_SCALE} timestep_shift=${TIMESTEP_SHIFT} cfg_norm=${CFG_NORM} steps=${NUM_STEPS}"

case "$TASK" in
  geneval)
    run_geneval_modes
    ;;
  dpgbench)
    run_one dpgbench "$DPGBENCH_IMAGE_SIZE"
    ;;
  longtextbench|longtext|longcontext)
    run_one longtextbench "$LONGTEXTBENCH_IMAGE_SIZE"
    ;;
  cvtg_2k|cvtg|cvtg2k)
    run_one cvtg_2k "$CVTG_2K_IMAGE_SIZE"
    ;;
  qwenimagebench|qib|qwen_image_bench|qwenimage)
    run_one qwenimagebench "$QWENIMAGEBENCH_IMAGE_SIZE"
    ;;
  tiifbench|tiif|tiif_bench)
    run_one tiifbench "$TIIFBENCH_IMAGE_SIZE"
    ;;
  oneigbench|oneig|oneig_bench)
    run_one oneigbench "$ONEIGBENCH_IMAGE_SIZE"
    ;;
  wise|wisebench|wise_bench)
    run_one wise "$WISE_IMAGE_SIZE"
    ;;
  bizgeneval|biz)
    run_one bizgeneval "$BIZGENEVAL_IMAGE_SIZE"
    ;;
  all|both)
    run_geneval_modes
    run_one dpgbench "$DPGBENCH_IMAGE_SIZE"
    run_one longtextbench "$LONGTEXTBENCH_IMAGE_SIZE"
    run_one cvtg_2k "$CVTG_2K_IMAGE_SIZE"
    run_one tiifbench "$TIIFBENCH_IMAGE_SIZE"
    run_one oneigbench "$ONEIGBENCH_IMAGE_SIZE"
    run_one wise "$WISE_IMAGE_SIZE"
    run_one bizgeneval "$BIZGENEVAL_IMAGE_SIZE"
    run_one qwenimagebench "$QWENIMAGEBENCH_IMAGE_SIZE"
    ;;
  *)
    echo "Unsupported TASK=${TASK}"
    echo "Supported: geneval, dpgbench, longtextbench, cvtg_2k, tiifbench, oneigbench, wise, bizgeneval, qwenimagebench, all"
    exit 1
    ;;
esac
