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
BATCH_SIZE="${BATCH_SIZE:-4}"
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
  local max_latent_size="$MAX_LATENT_SIZE"
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
  RUN_GENERATION="$RUN_GENERATION" \
  RUN_EVAL="$RUN_EVAL" \
  IMAGE_SIZE="$image_size" \
  NUM_STEPS="$NUM_STEPS" \
  CFG_SCALE="$CFG_SCALE" \
  TIMESTEP_SHIFT="$TIMESTEP_SHIFT" \
  CFG_NORM="$CFG_NORM" \
  BATCH_SIZE="$BATCH_SIZE" \
  MAX_LATENT_SIZE="$max_latent_size" \
  MAX_MEMORY="$MAX_MEMORY" \
  TARGET_INDICES="${TARGET_INDICES:-}" \
  TARGET_PROMPT_IDS="${TARGET_PROMPT_IDS:-}" \
  TARGET_KEYS="${TARGET_KEYS:-}" \
  LONGTEXTBENCH_LANGS="$LONGTEXTBENCH_LANGS" \
  ONEIGBENCH_LANGS="$ONEIGBENCH_LANGS" \
  CVTG_SUBSETS="${CVTG_SUBSETS:-}" \
  CVTG_AREAS="${CVTG_AREAS:-}" \
  BIZGENEVAL_DEFAULT_ASPECT_RATIO="${BIZGENEVAL_DEFAULT_ASPECT_RATIO:-}" \
  THINK_MODE="${THINK_MODE:-0}" \
  WISE_SUBDIR="${WISE_SUBDIR:-}" \
  WISE_CATEGORIES="${WISE_CATEGORIES:-}" \
  TIIF_BENCH_ROOT="${TIIF_BENCH_ROOT:-}" \
  TIIFBENCH_SPLIT="${TIIFBENCH_SPLIT:-}" \
  TIIFBENCH_PROMPT_DIR="${TIIFBENCH_PROMPT_DIR:-}" \
  TIIFBENCH_EVAL_PROMPT_DIR="${TIIFBENCH_EVAL_PROMPT_DIR:-}" \
  TIIFBENCH_SPECIFIC_FILE="${TIIFBENCH_SPECIFIC_FILE:-}" \
  TIIFBENCH_EVAL_ENV="${TIIFBENCH_EVAL_ENV:-}" \
  TIIFBENCH_EVAL_MODEL="${TIIFBENCH_EVAL_MODEL:-}" \
  TIIFBENCH_EVAL_BASE_URL="${TIIFBENCH_EVAL_BASE_URL:-}" \
  TIIFBENCH_API_VERSION="${TIIFBENCH_API_VERSION:-}" \
  API_KEY="${API_KEY:-}" \
  QIB_LANGS="$QIB_LANGS" \
  QIB_EVAL_BACKEND="${QIB_EVAL_BACKEND:-}" \
  QIB_API_ENV="${QIB_API_ENV:-}" \
  QIB_API_ROOT="${QIB_API_ROOT:-}" \
  QIB_API_LOCAL_BENCH_ROOT="${QIB_API_LOCAL_BENCH_ROOT:-}" \
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

echo "[bagel-t2i-go] model=${MODEL_PATH}"
echo "[bagel-t2i-go] model_name=${MODEL_NAME} step_name=${STEP_NAME}"
echo "[bagel-t2i-go] task=${TASK} gpu_list=${GPU_LIST}"
echo "[bagel-t2i-go] output_base=${OUTPUT_BASE}"
echo "[bagel-t2i-go] run_generation=${RUN_GENERATION} run_eval=${RUN_EVAL}"
echo "[bagel-t2i-go] cfg_scale=${CFG_SCALE} timestep_shift=${TIMESTEP_SHIFT} cfg_norm=${CFG_NORM} steps=${NUM_STEPS}"

case "$TASK" in
  geneval)
    run_one geneval "$GENEVAL_IMAGE_SIZE"
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
    run_one geneval "$GENEVAL_IMAGE_SIZE"
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
