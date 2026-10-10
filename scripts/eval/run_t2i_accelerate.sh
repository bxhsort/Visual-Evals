#!/usr/bin/env bash
# Run BAGEL text-to-image generation/evaluation with local multi-GPU workers.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UMM_ROOT="${UMM_ROOT:-${BAGEL_ROOT}/third_party/UMM_Eval}"
UMM_ENV_CONFIG="${UMM_ENV_CONFIG:-${UMM_ROOT}/config/conda_envs.env}"

if [[ -f "$UMM_ENV_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$UMM_ENV_CONFIG"
fi

BENCHMARK="${BENCHMARK:-${TASK:-geneval}}"
MODEL_PATH="${MODEL_PATH:-${model_path:-}}"
MODEL_NAME="${MODEL_NAME:-bagel}"
STEP_NAME="${STEP_NAME:-$(basename "${MODEL_PATH:-step}")}"
OUTPUT_BASE="${OUTPUT_BASE:-${BAGEL_ROOT}/outputs}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FINALIZE_EVAL="${FINALIZE_EVAL:-0}"

BAGEL_ENV="${BAGEL_ENV:-${BAGEL_ROOT}/.venv}"
QIB_API_ENV="${QIB_API_ENV:-${GEN_ENV}}"

IMAGE_SIZE="${IMAGE_SIZE:-}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
CFG_NORM="${CFG_NORM:-global}"
BATCH_SIZE="${BATCH_SIZE:-4}"
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
WISE_CATEGORIES="${WISE_CATEGORIES:-cultural_common_sense,natural_science,spatio-temporal_reasoning}"
case "${THINK_MODE,,}" in
  1|true|yes|on) WISE_SUBDIR="${WISE_SUBDIR:-wise_think}" ;;
  *) WISE_SUBDIR="${WISE_SUBDIR:-wise_raw}" ;;
esac

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
  case "${raw_value^^}" in
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

activate_conda_env() {
  local env_name="$1"
  if [[ "$env_name" == /* && ! -d "$env_name" ]]; then
    echo "Conda env path not found: $env_name"
    echo "Set the matching *_EVAL_ENV variable or UMM_ENV_CONFIG to a valid config/conda_envs.env."
    exit 1
  fi
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "CONDA_SH not found: $CONDA_SH"
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$env_name"
}

BENCHMARK="$(canonical_benchmark "$BENCHMARK")"
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
    METADATA_FILE="${METADATA_FILE:-${BAGEL_ROOT}/eval/gen/geneval/prompts/evaluation_metadata_long.jsonl}"
    NUM_IMAGES="${NUM_IMAGES:-4}"
    ;;
  dpgbench)
    METADATA_FILE="${METADATA_FILE:-${UMM_ROOT}/gen/t2i/dpgbench/eval/prompts.jsonl}"
    NUM_IMAGES="${NUM_IMAGES:-4}"
    ;;
  qwenimagebench)
    METADATA_FILE="${METADATA_FILE:-}"
    NUM_IMAGES="${NUM_IMAGES:-1}"
    ;;
  tiifbench)
    METADATA_FILE="${METADATA_FILE:-${TIIFBENCH_PROMPT_DIR}}"
    NUM_IMAGES="${NUM_IMAGES:-1}"
    ;;
  longtextbench)
    METADATA_FILE="${METADATA_FILE:-}"
    NUM_IMAGES="${NUM_IMAGES:-1}"
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
    ;;
  bizgeneval)
    METADATA_FILE="${METADATA_FILE:-${UMM_ROOT}/gen/t2i/bizgeneval/eval/test.jsonl}"
    NUM_IMAGES="${NUM_IMAGES:-1}"
    ;;
esac

OUTPUT_ROOT="$(derive_t2i_root)"
RESULT_DIR="${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/logs/${BENCHMARK}}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
N_GPU="${#GPUS[@]}"
if (( N_GPU <= 0 )); then
  echo "No GPU was provided."
  exit 1
fi

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
    metadata_file="${UMM_ROOT}/gen/t2i/qwenimagebench/prompts/qwenimagebench_${mode}.jsonl"
  fi
  if [[ "$BENCHMARK" == "tiifbench" ]]; then
    metadata_file="${TIIFBENCH_PROMPT_DIR}"
  fi
  if [[ "$BENCHMARK" == "longtextbench" ]]; then
    if [[ "$mode" == "zh" ]]; then
      metadata_file="${UMM_ROOT}/gen/t2i/longtextbench/eval/text_prompts_zh.jsonl"
    else
      metadata_file="${UMM_ROOT}/gen/t2i/longtextbench/eval/text_prompts.jsonl"
    fi
  fi
  if [[ ! -f "$metadata_file" ]]; then
    if [[ ! -d "$metadata_file" ]]; then
      echo "Metadata file/dir not found: $metadata_file"
      exit 1
    fi
  fi

  if [[ -f "${BAGEL_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${BAGEL_ENV}/bin/activate"
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
      activate_conda_env "$GENEVAL_EVAL_ENV"
      cd "${UMM_ROOT}/gen/t2i/geneval/eval"
      env -i \
        PATH="$CONDA_PREFIX/bin:/usr/bin:/bin" \
        LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
        CONDA_PREFIX="$CONDA_PREFIX" \
        PYTHONNOUSERSITE=1 \
        HOME="$HOME" \
        "$CONDA_PREFIX/bin/python" evaluation/evaluate_images.py \
        "${RESULT_DIR}/gen_eval" \
        --outfile "${RESULT_DIR}/results_geneval.jsonl" \
        --model-path "models"
      python evaluation/summary_scores.py "${RESULT_DIR}/results_geneval.jsonl" | tee "${RESULT_DIR}/summary_geneval.txt"
      ;;
    dpgbench)
      activate_conda_env "$DPGBENCH_EVAL_ENV"
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
        activate_conda_env "$QIB_API_ENV"
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
        activate_conda_env "$QWENIMAGEBENCH_EVAL_ENV"
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
      activate_conda_env "$TIIFBENCH_EVAL_ENV"
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
        RUN_GENERATION=0 RUN_EVAL=1 \
        OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
        IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
        TIMESTEP_SHIFT="$TIMESTEP_SHIFT" MODE="$mode" \
        TARGET_PROMPT_IDS="$TARGET_PROMPT_IDS" \
        bash "${UMM_ROOT}/gen/t2i/longtextbench/launch.sh"
      done
      ;;
    cvtg_2k)
      RUN_GENERATION=0 RUN_EVAL=1 \
      OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
      IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
      TIMESTEP_SHIFT="$TIMESTEP_SHIFT" CVTG_SUBSETS="$CVTG_SUBSETS" CVTG_AREAS="$CVTG_AREAS" \
      TARGET_KEYS="$TARGET_KEYS" \
      bash "${UMM_ROOT}/gen/t2i/cvtg_2k/launch.sh"
      ;;
    oneigbench)
      for mode in $ONEIGBENCH_LANGS; do
        RUN_GENERATION=0 RUN_EVAL=1 \
        OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
        IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
        TIMESTEP_SHIFT="$TIMESTEP_SHIFT" LANGUAGE="$mode" \
        bash "${UMM_ROOT}/gen/t2i/oneigbench/launch.sh"
      done
      ;;
    wise)
      RUN_GENERATION=0 RUN_EVAL=1 \
      OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
      IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
      TIMESTEP_SHIFT="$TIMESTEP_SHIFT" THINK_MODE="$THINK_MODE" \
      CATEGORIES="$WISE_CATEGORIES" API_KEY="$API_KEY" \
      bash "${UMM_ROOT}/gen/t2i/wise/launch.sh"
      ;;
    bizgeneval)
      RUN_GENERATION=0 RUN_EVAL=1 \
      OUTPUT_FOLDER="$OUTPUT_BASE" MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" \
      IMAGE_SIZE="$IMAGE_SIZE" CFG_SCALE="$CFG_SCALE" CFG_NORM="$CFG_NORM" \
      TIMESTEP_SHIFT="$TIMESTEP_SHIFT" TARGET_PROMPT_IDS="$TARGET_PROMPT_IDS" \
      JUDGE_API_KEY="$API_KEY" \
      bash "${UMM_ROOT}/gen/t2i/bizgeneval/launch.sh"
      ;;
  esac
fi

echo "[bagel-t2i] done"
