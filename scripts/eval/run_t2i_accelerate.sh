#!/usr/bin/env bash
# Run BAGEL text-to-image generation/evaluation with local multi-GPU workers.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UMM_ROOT="${UMM_ROOT:-${BAGEL_ROOT}/third_party/UMM_Eval}"

BENCHMARK="${BENCHMARK:-${TASK:-geneval}}"
MODEL_PATH="${MODEL_PATH:-${model_path:-}}"
MODEL_NAME="${MODEL_NAME:-bagel}"
STEP_NAME="${STEP_NAME:-$(basename "${MODEL_PATH:-step}")}"
OUTPUT_BASE="${OUTPUT_BASE:-${BAGEL_ROOT}/outputs}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FINALIZE_EVAL="${FINALIZE_EVAL:-0}"

IMAGE_SIZE="${IMAGE_SIZE:-}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
CFG_NORM="${CFG_NORM:-global}"
BATCH_SIZE="${BATCH_SIZE:-}"
SEED="${SEED:-42}"
NUM_IMAGES="${NUM_IMAGES:-}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-}"
MAX_MEMORY="${MAX_MEMORY:-80GiB}"
MASTER_PORT="${MASTER_PORT:-12370}"
TARGET_INDICES="${TARGET_INDICES:-}"
TARGET_PROMPT_IDS="${TARGET_PROMPT_IDS:-}"
TARGET_KEYS="${TARGET_KEYS:-}"
QIB_LANGS="${QIB_LANGS:-en zh}"
LONGTEXTBENCH_LANGS="${LONGTEXTBENCH_LANGS:-en}"
ONEIGBENCH_LANGS="${ONEIGBENCH_LANGS:-en zh}"
CVTG_SUBSETS="${CVTG_SUBSETS:-CVTG,CVTG-Style}"
CVTG_AREAS="${CVTG_AREAS:-2,3,4,5}"
BIZGENEVAL_DEFAULT_ASPECT_RATIO="${BIZGENEVAL_DEFAULT_ASPECT_RATIO:-1:1}"
THINK_MODE="${THINK_MODE:-0}"
GENEVAL_PROMPT_MODE="${GENEVAL_PROMPT_MODE:-long}"
WISE_CATEGORIES="${WISE_CATEGORIES:-cultural_common_sense,natural_science,spatio-temporal_reasoning}"
WISE_SUBDIR="${WISE_SUBDIR:-${OUTPUT_SUBDIR:-wise_raw}}"

QJUDGER_MODEL_PATH="${QJUDGER_MODEL_PATH:-${UMM_ROOT}/gen/t2i/qwenimagebench/eval/models/q_judger}"
QJUDGER_BATCH_SIZE="${QJUDGER_BATCH_SIZE:-8}"
QJUDGER_MAX_NEW_TOKENS="${QJUDGER_MAX_NEW_TOKENS:-4096}"
QJUDGER_NUM_SHARDS="${QJUDGER_NUM_SHARDS:-}"
QIB_EVAL_BACKEND="${QIB_EVAL_BACKEND:-local}"
QIB_API_ROOT="${QIB_API_ROOT:-}"
QIB_API_LOCAL_BENCH_ROOT="${QIB_API_LOCAL_BENCH_ROOT:-}"
QIB_EVAL_WORKERS="${QIB_EVAL_WORKERS:-8}"
QIB_DIM_MODE="${QIB_DIM_MODE:-metadata}"
QIB_REWARD_TIMEOUT="${QIB_REWARD_TIMEOUT:-900}"
QIB_REWARD_MAX_RETRIES="${QIB_REWARD_MAX_RETRIES:-2}"
QIB_REWARD_IMAGE_SOURCE="${QIB_REWARD_IMAGE_SOURCE:-image_path}"
TIIF_BENCH_ROOT="${TIIF_BENCH_ROOT:-${UMM_ROOT}/gen/t2i/tiifbench/eval/TIIF-Bench}"
TIIFBENCH_SPLIT="${TIIFBENCH_SPLIT:-testmini}"
TIIFBENCH_PROMPT_DIR="${TIIFBENCH_PROMPT_DIR:-${TIIF_BENCH_ROOT}/data/${TIIFBENCH_SPLIT}_prompts}"
TIIFBENCH_EVAL_PROMPT_DIR="${TIIFBENCH_EVAL_PROMPT_DIR:-${TIIF_BENCH_ROOT}/data/${TIIFBENCH_SPLIT}_eval_prompts}"
TIIFBENCH_SPECIFIC_FILE="${TIIFBENCH_SPECIFIC_FILE:-}"
TIIFBENCH_EVAL_MODEL="${TIIFBENCH_EVAL_MODEL:-gpt-4o}"
TIIFBENCH_EVAL_BASE_URL="${TIIFBENCH_EVAL_BASE_URL:-https://tokenhub.sensetime.com/v1}"
TIIFBENCH_API_VERSION="${TIIFBENCH_API_VERSION:-2025-01-01-preview}"
API_KEY="${API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}}"

if [[ -z "$MODEL_PATH" && "$RUN_GENERATION" != "0" ]]; then
  echo "MODEL_PATH/model_path is required when RUN_GENERATION=1"
  exit 1
fi
if [[ -n "$MODEL_PATH" && ! -e "$MODEL_PATH" ]]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH"
  exit 1
fi

normalize_image_size() {
  local raw_value="$1"
  local width height
  if [[ "$raw_value" =~ ^[0-9]+$ ]]; then echo "$raw_value"; return 0; fi
  if [[ "$raw_value" =~ ^([0-9]+)[xX\*]([0-9]+)$ ]]; then
    width="${BASH_REMATCH[1]}"; height="${BASH_REMATCH[2]}"
    if [[ "$width" != "$height" ]]; then echo "IMAGE_SIZE must be square, got: $raw_value" >&2; return 1; fi
    echo "$width"; return 0
  fi
  case "$(printf '%s' "$raw_value" | tr '[:lower:]' '[:upper:]')" in
    1K) echo "1024" ;;
    1.5K) echo "1536" ;;
    2K) echo "2048" ;;
    4K) echo "4096" ;;
    *) echo "IMAGE_SIZE must be an integer, 1K/1.5K/2K/4K, or square spec, got: $raw_value" >&2; return 1 ;;
  esac
}

default_image_size_for_benchmark() {
  case "$1" in
    longtextbench|longtext|longcontext) echo "512" ;;
    cvtg_2k|cvtg|cvtg2k|bizgeneval|biz) echo "2048" ;;
    geneval|dpgbench|qwenimagebench|qib|qwen_image_bench|qwenimage|tiifbench|tiif|tiif_bench|oneigbench|oneig|oneig_bench|wise|wisebench|wise_bench) echo "1024" ;;
    *) echo "Unsupported BENCHMARK=$1" >&2; return 1 ;;
  esac
}

canonical_benchmark() {
  case "$1" in
    geneval) echo "geneval" ;;
    dpgbench) echo "dpgbench" ;;
    longtextbench|longtext|longcontext) echo "longtextbench" ;;
    cvtg_2k|cvtg|cvtg2k) echo "cvtg_2k" ;;
    qwenimagebench|qib|qwen_image_bench|qwenimage) echo "qwenimagebench" ;;
    tiifbench|tiif|tiif_bench) echo "tiifbench" ;;
    oneigbench|oneig|oneig_bench) echo "oneigbench" ;;
    wise|wisebench|wise_bench) echo "wise" ;;
    bizgeneval|biz) echo "bizgeneval" ;;
    *) echo "Unsupported BENCHMARK=$1" >&2; return 1 ;;
  esac
}

derive_t2i_root() {
  PYTHONPATH="${UMM_ROOT}/gen/common:${PYTHONPATH:-}" python - "$IMAGE_SIZE" "$CFG_SCALE" "$TIMESTEP_SHIFT" "$CFG_NORM" <<'PY'
import sys, output_paths
print(output_paths.derive_t2i_root(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
PY
}

BENCHMARK="$(canonical_benchmark "$BENCHMARK")"
case "$(printf '%s' "$THINK_MODE" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    THINK_MODE=1
    echo "THINK_MODE=1 is unsupported by the current unified ${BENCHMARK} generator; no thinking experiment will be created." >&2
    exit 1
    ;;
  0|false|no|off) THINK_MODE=0 ;;
  *) echo "THINK_MODE must be 0/1 or false/true" >&2; exit 1 ;;
esac
if [[ -z "$IMAGE_SIZE" ]]; then
  IMAGE_SIZE="$(default_image_size_for_benchmark "$BENCHMARK")"
else
  IMAGE_SIZE="$(normalize_image_size "$IMAGE_SIZE")"
fi
if [[ -z "$MAX_LATENT_SIZE" ]]; then
  MAX_LATENT_SIZE=$(( IMAGE_SIZE > 1024 ? IMAGE_SIZE / 16 : 64 ))
fi

case "$BENCHMARK" in
  geneval)
    case "$GENEVAL_PROMPT_MODE" in
      short) METADATA_FILE="${GENEVAL_SHORT_METADATA_FILE:-${METADATA_FILE:-${BAGEL_ROOT}/eval/gen/geneval/prompts/evaluation_metadata.jsonl}}" ;;
      long) METADATA_FILE="${GENEVAL_LONG_METADATA_FILE:-${METADATA_FILE:-${BAGEL_ROOT}/eval/gen/geneval/prompts/evaluation_metadata_long.jsonl}}" ;;
      *) echo "A single wrapper run requires GENEVAL_PROMPT_MODE=short or long; use eval_t2i_go.sh for both." >&2; exit 1 ;;
    esac
    NUM_IMAGES="${NUM_IMAGES:-4}"
    ;;
  dpgbench)
    METADATA_FILE="${METADATA_FILE:-${UMM_ROOT}/gen/t2i/dpgbench/eval/prompts.jsonl}"
    NUM_IMAGES="${NUM_IMAGES:-4}"
    ;;
  qwenimagebench)
    METADATA_FILE="${METADATA_FILE:-}"
    NUM_IMAGES="${NUM_IMAGES:-1}"
    case "$QIB_EVAL_BACKEND" in local|api) ;; *) echo "QIB_EVAL_BACKEND must be local or api" >&2; exit 1 ;; esac
    [[ " $QIB_LANGS " == *" en "* ]] || unset QIB_METADATA_FILE_EN
    [[ " $QIB_LANGS " == *" zh "* ]] || unset QIB_METADATA_FILE_ZH
    for mode in $QIB_LANGS; do
      case "$mode" in
        en) QIB_METADATA_FILE_EN="${QIB_METADATA_FILE_EN:-${UMM_ROOT}/gen/t2i/qwenimagebench/prompts/qwenimagebench_en.jsonl}" ;;
        zh) QIB_METADATA_FILE_ZH="${QIB_METADATA_FILE_ZH:-${UMM_ROOT}/gen/t2i/qwenimagebench/prompts/qwenimagebench_zh.jsonl}" ;;
        *) echo "Unsupported QIB_LANGS language: $mode" >&2; exit 1 ;;
      esac
    done
    ;;
  tiifbench)
    METADATA_FILE="${METADATA_FILE:-${TIIFBENCH_PROMPT_DIR}}"
    NUM_IMAGES="${NUM_IMAGES:-1}"
    ;;
  longtextbench)
    METADATA_FILE="${METADATA_FILE:-}"
    NUM_IMAGES="${NUM_IMAGES:-4}"
    [[ " $LONGTEXTBENCH_LANGS " == *" en "* ]] || unset LONGTEXTBENCH_METADATA_FILE_EN
    [[ " $LONGTEXTBENCH_LANGS " == *" zh "* ]] || unset LONGTEXTBENCH_METADATA_FILE_ZH
    for mode in $LONGTEXTBENCH_LANGS; do
      case "$mode" in
        en) LONGTEXTBENCH_METADATA_FILE_EN="${LONGTEXTBENCH_METADATA_FILE_EN:-${UMM_ROOT}/gen/t2i/longtextbench/eval/text_prompts.jsonl}" ;;
        zh) LONGTEXTBENCH_METADATA_FILE_ZH="${LONGTEXTBENCH_METADATA_FILE_ZH:-${UMM_ROOT}/gen/t2i/longtextbench/eval/text_prompts_zh.jsonl}" ;;
        *) echo "Unsupported LONGTEXTBENCH_LANGS language: $mode" >&2; exit 1 ;;
      esac
    done
    ;;
  cvtg_2k)
    METADATA_FILE="${METADATA_FILE:-${UMM_ROOT}/gen/t2i/cvtg_2k/eval/CVTG-2K}"
    NUM_IMAGES="${NUM_IMAGES:-1}"
    ;;
  oneigbench)
    METADATA_FILE="${METADATA_FILE:-${UMM_ROOT}/gen/t2i/oneigbench/eval/OneIG-Benchmark}"
    NUM_IMAGES="${NUM_IMAGES:-4}"
    ;;
  wise)
    METADATA_FILE="${METADATA_FILE:-${BAGEL_ROOT}/eval/gen/wise/final_data.json}"
    NUM_IMAGES="${NUM_IMAGES:-1}"
    OUTPUT_SUBDIR="$WISE_SUBDIR"
    ;;
  bizgeneval)
    METADATA_FILE="${METADATA_FILE:-${UMM_ROOT}/gen/t2i/bizgeneval/eval/test.jsonl}"
    NUM_IMAGES="${NUM_IMAGES:-1}"
    OUTPUT_SUBDIR="bizgeneval"
    ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
N_GPU="${#GPUS[@]}"
if (( N_GPU <= 0 )); then
  echo "No GPU was provided."
  exit 1
fi
if [[ "$SEED" != "42" ]]; then
  echo "The current T2I generators use SEED=42; another SEED is unsupported." >&2
  exit 1
fi
if [[ "$BENCHMARK" == "wise" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
  if [[ "$NUM_IMAGES" != "1" ]]; then
    echo "The current unified WISE generator saves one image per prompt; NUM_IMAGES=1 is required." >&2
    exit 1
  fi
  if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer." >&2
    exit 1
  fi
  BATCH_SIZE=1
  unset GENERATION_SHARDS
elif [[ "$BENCHMARK" == "tiifbench" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
  if [[ "$NUM_IMAGES" != "1" || "$BATCH_SIZE" != "1" ]]; then
    echo "The current TIIFBench generator produces one image per description with BATCH_SIZE=1; NUM_IMAGES=1 is required." >&2
    exit 1
  fi
  unset GENERATION_SHARDS
else
  BATCH_SIZE="${BATCH_SIZE:-4}"
  if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ || ! "$NUM_IMAGES" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE and NUM_IMAGES must be positive integers." >&2
    exit 1
  fi
  if (( BATCH_SIZE > NUM_IMAGES )); then BATCH_SIZE="$NUM_IMAGES"; fi
  unset GENERATION_SHARDS
fi

# Record only settings used by this benchmark. The shared helper adds the model,
# prompt mode and thinking mode, and hashes the selected input paths separately.
CFG_INTERVAL='[0,1.0]'
CONFIG_KEYS="IMAGE_SIZE CFG_SCALE CFG_NORM NUM_STEPS NUM_IMAGES BATCH_SIZE MAX_LATENT_SIZE TIMESTEP_SHIFT SEED CFG_INTERVAL"
CONFIG_KEYS+=" TARGET_INDICES"
case "$BENCHMARK" in
  geneval) CONFIG_KEYS+=" METADATA_FILE" ;;
  dpgbench)
    EVAL_BACKBONE=mplug
    CONFIG_KEYS+=" METADATA_FILE TARGET_PROMPT_IDS EVAL_BACKBONE"
    ;;
  qwenimagebench)
    CONFIG_KEYS+=" QIB_LANGS QIB_METADATA_FILE_EN QIB_METADATA_FILE_ZH TARGET_PROMPT_IDS QIB_EVAL_BACKEND"
    if [[ "$QIB_EVAL_BACKEND" == "api" ]]; then
      INPUT_JSON_en="${INPUT_JSON_en:-${QIB_API_LOCAL_BENCH_ROOT}/data_en.json}"
      INPUT_JSON_cn="${INPUT_JSON_cn:-${QIB_API_LOCAL_BENCH_ROOT}/data_cn.json}"
      CONFIG_KEYS+=" QIB_API_ROOT QIB_API_LOCAL_BENCH_ROOT QIB_DIM_MODE QIB_REWARD_TIMEOUT QIB_REWARD_MAX_RETRIES QIB_REWARD_IMAGE_SOURCE QIB_REWARD_BASE_URL QIB_REWARD_BASE_URLS REWARD_BASE_URL REWARD_BASE_URLS"
      if [[ " $QIB_LANGS " == *" en "* ]]; then CONFIG_KEYS+=" INPUT_JSON_en"; fi
      if [[ " $QIB_LANGS " == *" zh "* ]]; then CONFIG_KEYS+=" INPUT_JSON_cn"; fi
    else
      CONFIG_KEYS+=" QJUDGER_MODEL_PATH QJUDGER_BATCH_SIZE QJUDGER_MAX_NEW_TOKENS"
    fi
    ;;
  tiifbench)
    CONFIG_KEYS+=" METADATA_FILE TIIFBENCH_SPLIT TIIFBENCH_SPECIFIC_FILE TIIFBENCH_EVAL_PROMPT_DIR TIIFBENCH_EVAL_MODEL TIIFBENCH_EVAL_BASE_URL TIIFBENCH_API_VERSION"
    ;;
  longtextbench)
    TEXTBENCH_ROOT="${TEXTBENCH_ROOT:-${UMM_ROOT}/gen/t2i/longtextbench/eval}"
    SAMPLE_FORMAT=prompt_repeat_v1
    CONFIG_KEYS+=" LONGTEXTBENCH_LANGS LONGTEXTBENCH_METADATA_FILE_EN LONGTEXTBENCH_METADATA_FILE_ZH TARGET_PROMPT_IDS TEXTBENCH_ROOT RESIZE_TO SAMPLE_FORMAT"
    ;;
  cvtg_2k)
    TEXTCRAFTER_ROOT="${TEXTCRAFTER_ROOT:-${UMM_ROOT}/gen/t2i/cvtg_2k/eval/TextCrafter_Eval}"
    CVTG_CLIP_MODEL_PATH="${CVTG_CLIP_MODEL_PATH:-/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/ckpt/ViT-L-14.pt}"
    CVTG_AESTHETIC_MODEL_PATH="${CVTG_AESTHETIC_MODEL_PATH:-/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/ckpt/sa_0_4_vit_l_14_linear.pth}"
    CVTG_VQA_MODEL_PATH="${CVTG_VQA_MODEL_PATH:-/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/ckpt/clip-flant5-xxl}"
    CVTG_VQA_TOKENIZER_PATH="${CVTG_VQA_TOKENIZER_PATH:-${CVTG_VQA_MODEL_PATH}}"
    CVTG_VQA_VISION_MODEL_PATH="${CVTG_VQA_VISION_MODEL_PATH:-/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/ckpt/clip-vit-large-patch14-336}"
    CVTG_OCR_MODEL_ROOT="${CVTG_OCR_MODEL_ROOT:-/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/ckpt/paddleocr}"
    CONFIG_KEYS+=" METADATA_FILE CVTG_SUBSETS CVTG_AREAS TARGET_KEYS TEXTCRAFTER_ROOT RESIZE_TO CVTG_CLIP_MODEL_PATH CVTG_AESTHETIC_MODEL_PATH CVTG_VQA_MODEL_PATH CVTG_VQA_VISION_MODEL_PATH CVTG_OCR_MODEL_ROOT CVTG_VQA_TOKENIZER_PATH"
    ;;
  oneigbench)
    GRID_ROWS="${GRID_ROWS:-2}"
    GRID_COLS="${GRID_COLS:-2}"
    ONEIGBENCH_EVAL_EXECUTOR="${ONEIGBENCH_EVAL_EXECUTOR:-auto}"
    ONEIGBENCH_API_EVAL="${ONEIGBENCH_API_EVAL:-0}"
    CONFIG_KEYS+=" METADATA_FILE ONEIGBENCH_LANGS GRID_ROWS GRID_COLS ONEIGBENCH_EVAL_EXECUTOR ONEIGBENCH_API_EVAL RESIZE_TO"
    ;;
  wise)
    WISE_DATA_ROOT="${WISE_DATA_ROOT:-${UMM_ROOT}/gen/t2i/wise/eval/data}"
    EVAL_MODEL="${EVAL_MODEL:-gpt-4o}"
    EVAL_BASE_URL="${EVAL_BASE_URL:-${EVAL_API_BASE_URL:-${OPENAI_BASE_URL:-https://tokenhub.sensetime.com/v1}}}"
    CONFIG_KEYS+=" METADATA_FILE WISE_SUBDIR WISE_CATEGORIES WISE_DATA_ROOT EVAL_MODEL EVAL_BASE_URL RESIZE_TO"
    ;;
  bizgeneval)
    JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.1-pro-preview}"
    JUDGE_API_BASE="${JUDGE_API_BASE:-${EVAL_API_BASE_URL:-${OPENAI_BASE_URL:-https://tokenhub.sensetime.com/v1}}}"
    CONFIG_KEYS+=" METADATA_FILE TARGET_PROMPT_IDS BIZGENEVAL_DEFAULT_ASPECT_RATIO JUDGE_MODEL JUDGE_API_BASE EVAL_MAX_ITEMS"
    ;;
esac
CONFIG_ARGS=()
for config_key in $CONFIG_KEYS; do
  if [[ -n "${!config_key:-}" ]]; then CONFIG_ARGS+=("${config_key}=${!config_key}"); fi
done
DG2_EFFECTIVE_CONFIG_JSON="$(python - "${CONFIG_ARGS[@]}" <<'PY'
import json, sys
print(json.dumps({key.upper(): value for key, value in (item.split('=', 1) for item in sys.argv[1:])}))
PY
)"

OUTPUT_ROOT="$(derive_t2i_root)"
if [[ "$BENCHMARK" == "geneval" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT}_prompt_${GENEVAL_PROMPT_MODE}"
fi
source "${SCRIPT_DIR}/experiment_layout.sh"
dg2_prepare_experiment "$BENCHMARK" "$OUTPUT_ROOT"
RESULT_DIR="$DG2_RESULT_DIR"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/logs/${BENCHMARK}}"

source "${SCRIPT_DIR}/coverage_report.sh"
dg2_write_coverage_reports() {
  local mode metadata image_dir judge_dir audit_root upper
  local audit_args=(--bench "$BENCHMARK" --num-images "$NUM_IMAGES" --model-name "$MODEL_NAME")
  [[ -n "$TARGET_INDICES" ]] && audit_args+=(--target-indices "$TARGET_INDICES")
  [[ -n "$TARGET_PROMPT_IDS" ]] && audit_args+=(--target-prompt-ids "$TARGET_PROMPT_IDS")
  [[ -n "$TARGET_KEYS" ]] && audit_args+=(--target-keys "$TARGET_KEYS")
  case "$BENCHMARK" in
    qwenimagebench)
      for mode in $QIB_LANGS; do
        metadata="${QIB_METADATA_FILE_EN:-}"
        [[ "$mode" == "zh" ]] && metadata="$QIB_METADATA_FILE_ZH"
        image_dir="${RESULT_DIR}/qwenimagebench_${mode}/samples"
        judge_dir="${RESULT_DIR}/qwenimagebench_${mode}/eval"
        [[ "$QIB_EVAL_BACKEND" == "api" ]] && judge_dir="${RESULT_DIR}/qwenimagebench_${mode}/eval_api"
        dg2_run_coverage_audit "${audit_args[@]}" --results-root "$RESULT_DIR" --language "$mode" \
      --metadata-file "${COVERAGE_METADATA_FILE:-$metadata}" --image-dir "${COVERAGE_IMAGE_DIR:-$image_dir}" \
      --judge-root "${COVERAGE_JUDGE_ROOT:-$judge_dir}"
      done
      ;;
    longtextbench)
      for mode in $LONGTEXTBENCH_LANGS; do
        metadata="${LONGTEXTBENCH_METADATA_FILE_EN:-}"
        [[ "$mode" == "zh" ]] && metadata="$LONGTEXTBENCH_METADATA_FILE_ZH"
        audit_root="${RESULT_DIR}/longtextbench_${mode}"
        dg2_run_coverage_audit "${audit_args[@]}" --results-root "$audit_root" --language "$mode" \
      --metadata-file "${COVERAGE_METADATA_FILE:-$metadata}" --image-dir "${COVERAGE_IMAGE_DIR:-${audit_root}/samples}" \
      --judge-root "${COVERAGE_JUDGE_ROOT:-${audit_root}/eval}"
      done
      ;;
    oneigbench)
      for mode in $ONEIGBENCH_LANGS; do
        upper="$(printf '%s' "$mode" | tr '[:lower:]' '[:upper:]')"
        dg2_run_coverage_audit "${audit_args[@]}" --results-root "$RESULT_DIR" --language "$mode" \
      --metadata-file "${COVERAGE_METADATA_FILE:-$METADATA_FILE}" \
      --image-dir "${COVERAGE_IMAGE_DIR:-${RESULT_DIR}/OneIGBench-$upper}" \
      --judge-root "${COVERAGE_JUDGE_ROOT:-${RESULT_DIR}/oneigbench-${upper}_results}"
      done
      ;;
    *)
      metadata="$METADATA_FILE"
      judge_dir="$RESULT_DIR"
      case "$BENCHMARK" in
        geneval) image_dir="${RESULT_DIR}/gen_eval" ;;
        dpgbench) image_dir="${RESULT_DIR}/dpgbench" ;;
        cvtg_2k)
          image_dir="${RESULT_DIR}/CVTG"
          audit_args+=(--cvtg-subsets "$CVTG_SUBSETS" --cvtg-areas "$CVTG_AREAS")
          ;;
        tiifbench)
          image_dir="${RESULT_DIR}/TIIFBench-${TIIFBENCH_SPLIT}"
          judge_dir="${RESULT_DIR}/tiifbench-${TIIFBENCH_SPLIT}_results/eval_json"
          audit_args+=(--split "$TIIFBENCH_SPLIT")
          [[ -n "$TIIFBENCH_SPECIFIC_FILE" ]] && audit_args+=(--tiifbench-specific-file "$TIIFBENCH_SPECIFIC_FILE")
          ;;
        wise)
          mode="raw"
          case "$(printf '%s' "$THINK_MODE" | tr '[:upper:]' '[:lower:]')" in 1|true|yes|on) mode="think" ;; esac
          image_dir="${RESULT_DIR}/${OUTPUT_SUBDIR:-wise_${mode}}"
          ;;
        bizgeneval)
          image_dir="${RESULT_DIR}/${OUTPUT_SUBDIR:-bizgeneval}"
          judge_dir="${image_dir}_eval"
          ;;
      esac
      dg2_run_coverage_audit "${audit_args[@]}" --results-root "$RESULT_DIR" \
      --metadata-file "${COVERAGE_METADATA_FILE:-$metadata}" --image-dir "${COVERAGE_IMAGE_DIR:-$image_dir}" \
      --judge-root "${COVERAGE_JUDGE_ROOT:-$judge_dir}"
      ;;
  esac
}
dg2_install_coverage_trap


export PYTHONPATH="${BAGEL_ROOT}:${UMM_ROOT}/gen/common:${PYTHONPATH:-}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "[bagel-t2i] benchmark=${BENCHMARK}"
echo "[bagel-t2i] model=${MODEL_PATH}"
echo "[bagel-t2i] model_name=${MODEL_NAME} step_name=${STEP_NAME}"
echo "[bagel-t2i] output_base=${OUTPUT_BASE}"
echo "[bagel-t2i] result_dir=${RESULT_DIR}"
echo "[bagel-t2i] gpu_list=${GPU_LIST}"
echo "[bagel-t2i] run_generation=${RUN_GENERATION} run_eval=${RUN_EVAL}"
echo "[bagel-t2i] image_size=${IMAGE_SIZE} cfg_scale=${CFG_SCALE} timestep_shift=${TIMESTEP_SHIFT} cfg_norm=${CFG_NORM}"
echo "[bagel-t2i] num_steps=${NUM_STEPS} num_images=${NUM_IMAGES} batch_size=${BATCH_SIZE}"

run_generation_one() {
  local mode="$1"
  local metadata_file="$METADATA_FILE"
  if [[ "$BENCHMARK" == "qwenimagebench" ]]; then
    metadata_file="${QIB_METADATA_FILE_EN:-}"
    [[ "$mode" == "zh" ]] && metadata_file="$QIB_METADATA_FILE_ZH"
  fi
  if [[ "$BENCHMARK" == "longtextbench" ]]; then
    if [[ "$mode" == "zh" ]]; then
      metadata_file="$LONGTEXTBENCH_METADATA_FILE_ZH"
    else
      metadata_file="$LONGTEXTBENCH_METADATA_FILE_EN"
    fi
  fi
  if [[ ! -f "$metadata_file" ]]; then
    if [[ ! -d "$metadata_file" ]]; then
      echo "Metadata file/dir not found: $metadata_file"
      exit 1
    fi
  fi

  local extra_args=()
  [[ -n "$TARGET_INDICES" ]] && extra_args+=(--target_indices "$TARGET_INDICES")
  [[ -n "$TARGET_PROMPT_IDS" ]] && extra_args+=(--target_prompt_ids "$TARGET_PROMPT_IDS")
  [[ -n "$TARGET_KEYS" ]] && extra_args+=(--target_keys "$TARGET_KEYS")
  if [[ "$BENCHMARK" == "tiifbench" ]]; then
    extra_args+=(--tiifbench_split "$TIIFBENCH_SPLIT")
    [[ -n "$TIIFBENCH_SPECIFIC_FILE" ]] && extra_args+=(--tiifbench_specific_file "$TIIFBENCH_SPECIFIC_FILE")
  fi

  local pids=()
  for idx in "${!GPUS[@]}"; do
    local gpu="${GPUS[$idx]}"
    local log="${LOG_DIR}/gen_${mode}_worker_$(printf '%02d' "$idx")_gpu_${gpu}.log"
    echo "[bagel-t2i] worker ${idx}/${N_GPU} mode=${mode} gpu=${gpu} log=${log}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python "${BAGEL_ROOT}/eval/gen/gen_images_mp_t2i.py" \
        --benchmark "$BENCHMARK" \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --metadata_file "$metadata_file" \
        --output_dir "$RESULT_DIR" \
        --mode "$mode" \
        --shard_id "$idx" \
        --total_shards "$N_GPU" \
        --device 0 \
        --seed 42 \
        --resolution "$IMAGE_SIZE" \
        --num_images "$NUM_IMAGES" \
        --batch_size "$BATCH_SIZE" \
        --num_timesteps "$NUM_STEPS" \
        --cfg_scale "$CFG_SCALE" \
        --cfg_norm "$CFG_NORM" \
        --timestep_shift "$TIMESTEP_SHIFT" \
        --max_latent_size "$MAX_LATENT_SIZE" \
        --accelerate_device_map \
        --max_memory "$MAX_MEMORY" \
        --offload_folder "${RESULT_DIR}/offload/${BENCHMARK}_${mode}_worker_${idx}" \
        --cvtg_subsets "$CVTG_SUBSETS" \
        --cvtg_areas "$CVTG_AREAS" \
        --default_aspect_ratio "$BIZGENEVAL_DEFAULT_ASPECT_RATIO" \
        --wise_subdir "$WISE_SUBDIR" \
        "${extra_args[@]}"
    ) >"$log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[bagel-t2i] worker ${idx} failed; tailing log:"
      tail -100 "${LOG_DIR}/gen_${mode}_worker_$(printf '%02d' "$idx")_gpu_${GPUS[$idx]}.log"
      failed=1
    fi
  done
  if (( failed != 0 )); then
    exit 1
  fi
}

build_qib_api_generated_json() {
  local input_json="$1"
  local sample_dir="$2"
  local output_json="$3"
  python - "$input_json" "$sample_dir" "$output_json" "${TARGET_PROMPT_IDS:-}" <<'PY'
import json
import sys
from pathlib import Path

input_json, sample_dir, output_json, target_ids_raw = sys.argv[1:5]
sample_dir = Path(sample_dir)
target_ids = None
if target_ids_raw:
    target_ids = {part.strip() for part in target_ids_raw.split(",") if part.strip()}

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list):
    raise ValueError(f"QwenImageBench API input must be a list: {input_json}")

rows = []
missing = []
for item in data:
    item_id = str(item["ID"])
    if target_ids is not None and item_id not in target_ids:
        continue
    image_path = sample_dir / f"{item_id}.png"
    if not image_path.exists():
        missing.append(str(image_path))
        continue
    row = dict(item)
    row["image_path"] = str(image_path)
    rows.append(row)

if missing:
    preview = "\n".join(missing[:10])
    more = "" if len(missing) <= 10 else f"\n... {len(missing) - 10} more"
    raise FileNotFoundError(f"Missing QwenImageBench samples:\n{preview}{more}")

Path(output_json).parent.mkdir(parents=True, exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=4)
print(f"Wrote {len(rows)} rows to {output_json}")
PY
}

write_qib_api_summary() {
  python - "${RESULT_DIR}/qwenimagebench_api_summary.json" "$@" <<'PY'
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
pairs = sys.argv[2:]
summary = {}
totals = []
for i in range(0, len(pairs), 2):
    lang, path = pairs[i], Path(pairs[i + 1])
    with open(path, "r", encoding="utf-8") as f:
        scores = json.load(f)
    summary[lang] = scores
    if isinstance(scores.get("total"), (int, float)):
        totals.append(float(scores["total"]))
summary["average_total"] = statistics.mean(totals) if totals else None
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)
print(f"Wrote API summary to {out}")
PY
}

if (( RUN_GENERATION != 0 )); then
  source activate bagel
  export PATH=/opt/conda/envs/bagel/bin:$PATH
  case "$BENCHMARK" in
    qwenimagebench)
      for mode in $QIB_LANGS; do run_generation_one "$mode"; done
      ;;
    longtextbench)
      for mode in $LONGTEXTBENCH_LANGS; do run_generation_one "$mode"; done
      ;;
    oneigbench)
      for mode in $ONEIGBENCH_LANGS; do run_generation_one "$mode"; done
      ;;
    wise)
      run_generation_one "all"
      ;;
    *)
      run_generation_one "all"
      ;;
  esac
fi

if (( RUN_EVAL != 0 )); then
  case "$BENCHMARK" in
    geneval)
      source activate geneval
      export PATH=/opt/conda/envs/geneval/bin:$PATH
      cd "${UMM_ROOT}/gen/t2i/geneval/eval"
      env -i \
        PATH="$PATH" \
        LD_LIBRARY_PATH="/opt/conda/envs/geneval/lib" \
        CONDA_PREFIX="/opt/conda/envs/geneval" \
        PYTHONNOUSERSITE=1 \
        HOME="$HOME" \
        "/opt/conda/envs/geneval/bin/python" evaluation/evaluate_images.py \
        "${RESULT_DIR}/gen_eval" \
        --outfile "${RESULT_DIR}/results_geneval.jsonl" \
        --model-path "models"
      python evaluation/summary_scores.py "${RESULT_DIR}/results_geneval.jsonl" | tee "${RESULT_DIR}/summary_geneval.txt"
      ;;
    dpgbench)
      source activate geneval
      export PATH=/opt/conda/envs/geneval/bin:$PATH
      cd "${UMM_ROOT}/gen/t2i/dpgbench/eval"
      EVAL_GPUS="${EVAL_GPUS:-${N_GPU}}"
      accelerate launch --num_machines 1 --num_processes "$EVAL_GPUS" --multi_gpu --mixed_precision "fp16" --main_process_port "$((MASTER_PORT + 100))" \
        ./compute_dpg_bench.py \
        --image-root-path "${RESULT_DIR}/dpgbench" \
        --resolution "$IMAGE_SIZE" \
        --pic-num "$NUM_IMAGES" \
        --res-path "${RESULT_DIR}/dpgbench_results.txt" \
        --vqa-model mplug
      ;;
    qwenimagebench)
      source activate geneval
      export PATH=/opt/conda/envs/geneval/bin:$PATH
      if [[ "$QIB_EVAL_BACKEND" == "api" ]]; then
        if [[ -z "$QIB_API_ROOT" || ! -f "${QIB_API_ROOT}/api_eval.py" ]]; then
          echo "QIB_API_ROOT must contain api_eval.py when QIB_EVAL_BACKEND=api"
          exit 1
        fi
        if [[ -z "$QIB_API_LOCAL_BENCH_ROOT" ]]; then
          echo "QIB_API_LOCAL_BENCH_ROOT is required when QIB_EVAL_BACKEND=api"
          exit 1
        fi
        if [[ -z "${QIB_REWARD_BASE_URL:-}" && -z "${QIB_REWARD_BASE_URLS:-}" && -z "${REWARD_BASE_URL:-}" && -z "${REWARD_BASE_URLS:-}" ]]; then
          echo "QIB_REWARD_BASE_URL/QIB_REWARD_BASE_URLS is required when QIB_EVAL_BACKEND=api"
          exit 1
        fi
        API_SUMMARY_ARGS=()
        for mode in $QIB_LANGS; do
          api_lang="$mode"
          if [[ "$api_lang" == "zh" ]]; then api_lang="cn"; fi
          input_var="INPUT_JSON_${api_lang}"
          api_input_json="${!input_var:-${QIB_API_LOCAL_BENCH_ROOT}/data_${api_lang}.json}"
          QIB_DIR="${RESULT_DIR}/qwenimagebench_${mode}"
          SAMPLE_DIR="${QIB_DIR}/samples"
          EVAL_DIR="${QIB_DIR}/eval_api"
          GENERATED_JSON="${EVAL_DIR}/data_${api_lang}_gen_image.json"
          JUDGED_JSON="${EVAL_DIR}/data_${api_lang}_gen_image_judged.json"
          SCORES_JSON="${EVAL_DIR}/data_${api_lang}_gen_image_bench_scores.json"
          CACHE_DIR="${EVAL_DIR}/reward_cache/${QIB_DIM_MODE}/${api_lang}"
          mkdir -p "$EVAL_DIR"
          build_qib_api_generated_json "$api_input_json" "$SAMPLE_DIR" "$GENERATED_JSON"
          python "${QIB_API_ROOT}/api_eval.py" \
            --input_json "$GENERATED_JSON" \
            --output_judged_json "$JUDGED_JSON" \
            --output_scores_json "$SCORES_JSON" \
            --run_name "${STEP_NAME}_${api_lang}" \
            --cache_dir "$CACHE_DIR" \
            --dim_mode "$QIB_DIM_MODE" \
            --image_source "$QIB_REWARD_IMAGE_SOURCE" \
            --num_workers "$QIB_EVAL_WORKERS" \
            --timeout "$QIB_REWARD_TIMEOUT" \
            --max_retries "$QIB_REWARD_MAX_RETRIES" \
            --resume
          API_SUMMARY_ARGS+=("$api_lang" "$SCORES_JSON")
        done
        write_qib_api_summary "${API_SUMMARY_ARGS[@]}"
      else
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        IFS=',' read -r -a JUDGE_GPUS <<< "$GPU_LIST"
        NSHARDS="${QJUDGER_NUM_SHARDS:-${#JUDGE_GPUS[@]}}"
        AGG_ARGS=()
        for mode in $QIB_LANGS; do
          QIB_DIR="${RESULT_DIR}/qwenimagebench_${mode}"
          SAMPLE_DIR="${QIB_DIR}/samples"
          EVAL_DIR="${QIB_DIR}/eval"
          JUDGE_INPUT="${EVAL_DIR}/judge_input.jsonl"
          mkdir -p "$EVAL_DIR"
          python "${UMM_ROOT}/gen/t2i/qwenimagebench/eval/build_judge_input.py" \
            --lang "$mode" \
            --samples-dir "$SAMPLE_DIR" \
            --out "$JUDGE_INPUT"
          if (( NSHARDS <= 1 )); then
            ( cd "${UMM_ROOT}/gen/t2i/qwenimagebench/eval" && python judge.py \
              --input "$JUDGE_INPUT" \
              --model "$QJUDGER_MODEL_PATH" \
              --local-metadata "${UMM_ROOT}/gen/t2i/qwenimagebench/eval/metadata/bench_metadata.json" \
              --max-batch-size "$QJUDGER_BATCH_SIZE" \
              --max-new-tokens "$QJUDGER_MAX_NEW_TOKENS" )
          else
            JLOG_DIR="${EVAL_DIR}/judge_logs"
            mkdir -p "$JLOG_DIR"
            jpids=()
            for (( rank=0; rank<NSHARDS; rank++ )); do
              dev="${JUDGE_GPUS[$(( rank % ${#JUDGE_GPUS[@]} ))]}"
              jlog="${JLOG_DIR}/judge_${mode}_shard${rank}.log"
              echo "[bagel-t2i] judge ${mode} shard ${rank}/${NSHARDS} gpu=${dev} log=${jlog}"
              ( cd "${UMM_ROOT}/gen/t2i/qwenimagebench/eval" && CUDA_VISIBLE_DEVICES="$dev" python judge.py \
                --input "$JUDGE_INPUT" \
                --model "$QJUDGER_MODEL_PATH" \
                --local-metadata "${UMM_ROOT}/gen/t2i/qwenimagebench/eval/metadata/bench_metadata.json" \
                --max-batch-size "$QJUDGER_BATCH_SIZE" \
                --max-new-tokens "$QJUDGER_MAX_NEW_TOKENS" \
                --num-shards "$NSHARDS" \
                --shard-rank "$rank" ) >"$jlog" 2>&1 &
              jpids+=("$!")
            done
            jfailed=0
            for pid in "${jpids[@]}"; do
              if ! wait "$pid"; then jfailed=1; fi
            done
            if (( jfailed != 0 )); then
              echo "A Qwen-Image-Bench judge shard failed. Check logs under ${JLOG_DIR}."
              exit 1
            fi
            ( cd "${UMM_ROOT}/gen/t2i/qwenimagebench/eval" && python merge_shard_scores.py --input "$JUDGE_INPUT" --num-shards "$NSHARDS" )
          fi
          AGG_ARGS+=(--"$mode" "${EVAL_DIR}/judge_input_bench_scores.json")
        done
        python "${UMM_ROOT}/gen/t2i/qwenimagebench/eval/aggregate_bilingual.py" \
          "${AGG_ARGS[@]}" \
          --out "${RESULT_DIR}/qwenimagebench_bilingual_summary.json"
      fi
      ;;
    tiifbench)
      if [[ ! -d "$TIIF_BENCH_ROOT" ]]; then
        echo "TIIF_BENCH_ROOT not found: $TIIF_BENCH_ROOT"
        exit 1
      fi
      if [[ ! -d "$TIIFBENCH_EVAL_PROMPT_DIR" ]]; then
        echo "TIIFBENCH_EVAL_PROMPT_DIR not found: $TIIFBENCH_EVAL_PROMPT_DIR"
        exit 1
      fi
      source activate geneval
      export PATH=/opt/conda/envs/geneval/bin:$PATH
      export PYTHONPATH="${UMM_ROOT}/gen/common:${PYTHONPATH:-}"
      IMAGE_DIR="${RESULT_DIR}/TIIFBench-${TIIFBENCH_SPLIT}"
      RESULTS_DIR="${RESULT_DIR}/tiifbench-${TIIFBENCH_SPLIT}_results"
      EVAL_JSON_DIR="${RESULTS_DIR}/eval_json"
      mkdir -p "$EVAL_JSON_DIR"
      cd "$TIIF_BENCH_ROOT"
      python "${UMM_ROOT}/gen/t2i/tiifbench/eval/eval_with_vlm_mp_tokenhub.py" \
        --jsonl_dir "$TIIFBENCH_EVAL_PROMPT_DIR" \
        --image_dir "$IMAGE_DIR" \
        --eval_model "$MODEL_NAME" \
        --output_dir "$EVAL_JSON_DIR" \
        --model "$TIIFBENCH_EVAL_MODEL" \
        --api_key "$API_KEY" \
        --base_url "$TIIFBENCH_EVAL_BASE_URL" \
        --api_version "$TIIFBENCH_API_VERSION"
      python eval/summary_results.py --input_dir "$EVAL_JSON_DIR"
      python eval/summary_dimension_results.py \
        --input_excel "${EVAL_JSON_DIR}/result_summary.xlsx" \
        --output_txt "${RESULTS_DIR}/result_summary_dimension.txt"
      ;;
    longtextbench)
      for mode in $LONGTEXTBENCH_LANGS; do
        metadata="${LONGTEXTBENCH_METADATA_FILE_EN:-}"
        [[ "$mode" == "zh" ]] && metadata="$LONGTEXTBENCH_METADATA_FILE_ZH"
        RUN_GENERATION=0 RUN_EVAL=1 \
        RESULT_ROOT_DIR="$RESULT_DIR" RESULT_DIR="$RESULT_DIR" DATA_PATH="$metadata" \
        OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
        IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
        TIMESTEP_SHIFT="$TIMESTEP_SHIFT" MODE="$mode" OUTPUT_SUBDIR="longtextbench_${mode}" TEXTBENCH_ROOT="$TEXTBENCH_ROOT" \
        TARGET_PROMPT_IDS="$TARGET_PROMPT_IDS" \
        bash "${UMM_ROOT}/gen/t2i/longtextbench/launch.sh"
      done
      ;;
    cvtg_2k)
      RUN_GENERATION=0 RUN_EVAL=1 \
      RESULT_ROOT_DIR="$RESULT_DIR" BENCHMARK_ROOT="$METADATA_FILE" TEXTCRAFTER_ROOT="$TEXTCRAFTER_ROOT" \
      CVTG_CLIP_MODEL_PATH="$CVTG_CLIP_MODEL_PATH" CVTG_AESTHETIC_MODEL_PATH="$CVTG_AESTHETIC_MODEL_PATH" CVTG_VQA_MODEL_PATH="$CVTG_VQA_MODEL_PATH" CVTG_VQA_VISION_MODEL_PATH="$CVTG_VQA_VISION_MODEL_PATH" CVTG_OCR_MODEL_ROOT="$CVTG_OCR_MODEL_ROOT" CVTG_VQA_TOKENIZER_PATH="$CVTG_VQA_TOKENIZER_PATH" \
      OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
      IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
      TIMESTEP_SHIFT="$TIMESTEP_SHIFT" CVTG_SUBSETS="$CVTG_SUBSETS" CVTG_AREAS="$CVTG_AREAS" \
      TARGET_KEYS="$TARGET_KEYS" \
      bash "${UMM_ROOT}/gen/t2i/cvtg_2k/launch.sh"
      ;;
    oneigbench)
      for mode in $ONEIGBENCH_LANGS; do
        RUN_GENERATION=0 RUN_EVAL=1 \
        RESULT_ROOT_DIR="$RESULT_DIR" RESULT_DIR="$RESULT_DIR" ONEIG_BENCH_ROOT="$METADATA_FILE" \
        OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
        IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
        TIMESTEP_SHIFT="$TIMESTEP_SHIFT" LANGUAGE="$mode" GRID_ROWS="$GRID_ROWS" GRID_COLS="$GRID_COLS" \
        ONEIGBENCH_EVAL_EXECUTOR="$ONEIGBENCH_EVAL_EXECUTOR" ONEIGBENCH_API_EVAL="$ONEIGBENCH_API_EVAL" \
        bash "${UMM_ROOT}/gen/t2i/oneigbench/launch.sh"
      done
      ;;
    wise)
      RUN_GENERATION=0 RUN_EVAL=1 \
      RESULT_ROOT_DIR="$RESULT_DIR" RESULT_DIR="$RESULT_DIR" \
      OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
      IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
      TIMESTEP_SHIFT="$TIMESTEP_SHIFT" THINK_MODE="$THINK_MODE" OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
      CATEGORIES="$WISE_CATEGORIES" WISE_DATA_ROOT="$WISE_DATA_ROOT" EVAL_MODEL="$EVAL_MODEL" EVAL_BASE_URL="$EVAL_BASE_URL" API_KEY="$API_KEY" \
      bash "${UMM_ROOT}/gen/t2i/wise/launch.sh"
      ;;
    bizgeneval)
      RUN_GENERATION=0 RUN_EVAL=1 \
      RESULT_ROOT_DIR="$RESULT_DIR" RESULT_DIR="$RESULT_DIR" DATA_PATH="$METADATA_FILE" \
      OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
      IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
      TIMESTEP_SHIFT="$TIMESTEP_SHIFT" TARGET_PROMPT_IDS="$TARGET_PROMPT_IDS" OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
      JUDGE_MODEL="$JUDGE_MODEL" JUDGE_API_BASE="$JUDGE_API_BASE" \
      JUDGE_API_KEY="$API_KEY" \
      bash "${UMM_ROOT}/gen/t2i/bizgeneval/launch.sh"
      ;;
  esac
fi

echo "[bagel-t2i] done"
