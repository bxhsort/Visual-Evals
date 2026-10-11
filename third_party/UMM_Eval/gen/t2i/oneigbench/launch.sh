#!/usr/bin/env bash
# OneIG-Bench (t2i, EN/ZH) — generation + local multi-metric scoring
# (alignment / text / diversity / style / reasoning).
set -eo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/gen/common:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

ONEIG_BENCH_ROOT="${ONEIG_BENCH_ROOT:-${BENCH_DIR}/eval/OneIG-Benchmark}"
ONEIGBENCH_PATCHES_DIR="${ONEIGBENCH_PATCHES_DIR:-${BENCH_DIR}/eval/oneigbench_patches}"

MODEL_NAME="${MODEL_NAME:-model}"
STEP_NAME="${STEP_NAME:-step}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_BASE="${OUTPUT_FOLDER:-${OUTPUT_PATH:-${REPO_ROOT}/outputs}}"
LAUNCH_MODE="${LAUNCH_MODE:-device_map_multi}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RUN_GENERATION="${RUN_GENERATION:-1}"
if [[ -z "${RUN_EVAL+x}" ]]; then
  if (( NUM_NODES > 1 )); then
    RUN_EVAL=0
  else
    RUN_EVAL=1
  fi
fi

GPUS="${GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-2}"
DEVICE_MAP="${DEVICE_MAP:-balanced}"
MAX_MEMORY_PER_GPU_GB="${MAX_MEMORY_PER_GPU_GB:-70}"

IMAGE_SIZE="${IMAGE_SIZE:-1024}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
CFG_NORM="${CFG_NORM:-global}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRID_ROWS="${GRID_ROWS:-2}"
GRID_COLS="${GRID_COLS:-2}"
LANGUAGE="${LANGUAGE:-${MODE:-en}}"
# Guard against locale env leak (e.g. container default LANGUAGE=en_US:en).
case "$LANGUAGE" in
  en|zh) ;;
  *) LANGUAGE="en" ;;
esac
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs/oneigbench_worker_logs/${MODEL_NAME}/${STEP_NAME}/${LANGUAGE}/node_${NODE_RANK}}"
RESIZE_TO="${RESIZE_TO:-}"
PE_MODE="${PE_MODE:-raw}"

ONEIGBENCH_EVAL_EXECUTOR="${ONEIGBENCH_EVAL_EXECUTOR:-auto}"
ONEIGBENCH_EVAL_NUM_WORKERS="${ONEIGBENCH_EVAL_NUM_WORKERS:-8}"
ONEIGBENCH_API_EVAL="${ONEIGBENCH_API_EVAL:-0}"

if [[ "$LANGUAGE" == "zh" ]]; then
  ONEIG_MODE="ZH"
  BENCH_SUBDIR="OneIGBench-ZH"
  ANALYSIS_SCRIPT="fine_grained_analysis_zh.py"
  RESULT_MD_NAME="result_zh.md"
  DATA_FILE="${ONEIG_BENCH_ROOT}/OneIG-Bench-ZH.csv"
  ALIGNMENT_CLASSES=(anime human object multilingualism)
  DIVERSITY_CLASSES=(anime human object text reasoning multilingualism)
  RESULTS_SUBDIR="oneigbench-ZH_results"
  PE_PROMPT_PATH="${REPO_ROOT}/tools/external_gen/pe_enhanced/oneigbench_zh.jsonl"
else
  ONEIG_MODE="EN"
  BENCH_SUBDIR="OneIGBench-EN"
  ANALYSIS_SCRIPT="fine_grained_analysis_en.py"
  RESULT_MD_NAME="result_en.md"
  DATA_FILE="${ONEIG_BENCH_ROOT}/OneIG-Bench.csv"
  ALIGNMENT_CLASSES=(anime human object)
  DIVERSITY_CLASSES=(anime human object text reasoning)
  RESULTS_SUBDIR="oneigbench-EN_results"
  PE_PROMPT_PATH="${REPO_ROOT}/tools/external_gen/pe_enhanced/oneigbench_en.jsonl"
fi

count_visible_gpus() {
  local devices_csv="$1"
  IFS=',' read -r -a _VISIBLE_DEVICES <<< "$devices_csv"
  echo "${#_VISIBLE_DEVICES[@]}"
}

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

should_resume() {
  [[ -n "${RESUME:-}" ]] && is_truthy "$RESUME"
}

build_device_groups() {
  local devices_csv="$1"
  local per_worker="$2"
  local joined_group

  if (( per_worker <= 0 )); then
    echo "GPUS_PER_WORKER must be positive, got $per_worker"
    exit 1
  fi

  IFS=',' read -r -a ALL_VISIBLE_DEVICES <<< "$devices_csv"
  if (( ${#ALL_VISIBLE_DEVICES[@]} == 0 )); then
    echo "No visible devices were provided."
    exit 1
  fi
  if (( ${#ALL_VISIBLE_DEVICES[@]} % per_worker != 0 )); then
    echo "Visible GPU count ${#ALL_VISIBLE_DEVICES[@]} is not divisible by GPUS_PER_WORKER=$per_worker"
    exit 1
  fi

  DEVICE_GROUPS=()
  for ((i=0; i<${#ALL_VISIBLE_DEVICES[@]}; i+=per_worker)); do
    joined_group=$(IFS=,; echo "${ALL_VISIBLE_DEVICES[*]:i:per_worker}")
    DEVICE_GROUPS+=("$joined_group")
  done
}

build_output_root() {
  python - "$IMAGE_SIZE" "$CFG_SCALE" "$TIMESTEP_SHIFT" "$CFG_NORM" <<'PY'
import sys, output_paths
print(output_paths.derive_t2i_root(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
PY
}

if (( NUM_NODES <= 0 )); then
  echo "NUM_NODES must be positive, got $NUM_NODES"
  exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
  echo "NODE_RANK must be in [0, NUM_NODES), got NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES"
  exit 1
fi

mkdir -p "$OUTPUT_BASE"
RESULT_ROOT_DIR="${RESULT_ROOT_DIR:-${RESULT_DIR:-}}"
if [[ -z "$RESULT_ROOT_DIR" ]]; then
  OUTPUT_ROOT="$(build_output_root)"
  RESULT_ROOT_DIR="${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}"
fi
RESULT_ROOT="$RESULT_ROOT_DIR"
IMAGE_DIR="${RESULT_ROOT}/${BENCH_SUBDIR}"
RESULTS_DIR="${RESULT_ROOT}/${RESULTS_SUBDIR}"
IMAGE_GRID="${GRID_ROWS},${GRID_COLS}"

if (( RUN_GENERATION != 0 || RUN_EVAL != 0 )); then
  source activate bagel
  export PATH=/opt/conda/envs/bagel/bin:$PATH
fi

if (( RUN_GENERATION != 0 )); then
  if [[ -z "$MODEL_PATH" ]]; then
    echo "MODEL_PATH is required for generation."
    exit 1
  fi

  COMMON_GEN_ARGS=(
    --model_path "$MODEL_PATH"
    --model_name "$MODEL_NAME"
    --output_path "$IMAGE_DIR"
    --data_root "$ONEIG_BENCH_ROOT"
    --resolution "$IMAGE_SIZE"
    --cfg_scale "$CFG_SCALE"
    --cfg_norm "$CFG_NORM"
    --timestep_shift "$TIMESTEP_SHIFT"
    --num_steps "$NUM_STEPS"
    --batch_size "$BATCH_SIZE"
    --language "$LANGUAGE"
    --grid_rows "$GRID_ROWS"
    --grid_cols "$GRID_COLS"
  )
  case "$PE_MODE" in
    raw) ;;
    enhanced)
      [[ -f "$PE_PROMPT_PATH" ]] || { echo "Enhanced prompt cache not found: $PE_PROMPT_PATH"; exit 1; }
      COMMON_GEN_ARGS+=(--pe_prompt_path "$PE_PROMPT_PATH") ;;
    *) echo "Unsupported PE_MODE=$PE_MODE. Use raw or enhanced."; exit 1 ;;
  esac
  if should_resume; then
    COMMON_GEN_ARGS+=(--resume)
  fi
  if [[ -n "$RESIZE_TO" ]]; then
    COMMON_GEN_ARGS+=(--resize_to "$RESIZE_TO")
  fi

  if [[ "$LAUNCH_MODE" == "device_map" ]]; then
    export CUDA_VISIBLE_DEVICES
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
    TRANSFORMERS_VERBOSITY=error python "${BENCH_DIR}/inference.py" \
      "${COMMON_GEN_ARGS[@]}" \
      --device_map "$DEVICE_MAP" \
      --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB" \
      --num_shards "$NUM_NODES" \
      --shard_rank "$NODE_RANK"
  elif [[ "$LAUNCH_MODE" == "device_map_multi" ]]; then
    mkdir -p "$LOG_DIR"
    build_device_groups "$CUDA_VISIBLE_DEVICES" "$GPUS_PER_WORKER"
    LOCAL_NUM_SHARDS="${#DEVICE_GROUPS[@]}"
    TOTAL_NUM_SHARDS=$((NUM_NODES * LOCAL_NUM_SHARDS))
    NODE_SHARD_OFFSET=$((NODE_RANK * LOCAL_NUM_SHARDS))
    pids=()

    for shard_rank in "${!DEVICE_GROUPS[@]}"; do
      worker_devices="${DEVICE_GROUPS[$shard_rank]}"
      global_shard_rank=$((NODE_SHARD_OFFSET + shard_rank))
      worker_log="$LOG_DIR/oneigbench_shard_${global_shard_rank}.log"
      echo "Launching global shard ${global_shard_rank}/${TOTAL_NUM_SHARDS} on node ${NODE_RANK}/${NUM_NODES} using GPUs ${worker_devices}. Log: ${worker_log}"
      (
        export CUDA_VISIBLE_DEVICES="$worker_devices"
        # Per-worker module cache so concurrent trust_remote_code loads do not race.
        export HF_MODULES_CACHE="${LOG_DIR}/hf_modules/shard_${global_shard_rank}"
        mkdir -p "$HF_MODULES_CACHE"
        unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
        TRANSFORMERS_VERBOSITY=error python "${BENCH_DIR}/inference.py" \
          "${COMMON_GEN_ARGS[@]}" \
          --device_map "$DEVICE_MAP" \
          --max_memory_per_gpu_gb "$MAX_MEMORY_PER_GPU_GB" \
          --num_shards "$TOTAL_NUM_SHARDS" \
          --shard_rank "$global_shard_rank"
      ) >"$worker_log" 2>&1 &
      pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if (( failed != 0 )); then
      echo "At least one OneIGBench worker failed. Check logs under $LOG_DIR."
      exit 1
    fi
  elif [[ "$LAUNCH_MODE" == "ddp" ]]; then
    TRANSFORMERS_VERBOSITY=error torchrun --nproc_per_node="$GPUS" "${BENCH_DIR}/inference.py" \
      "${COMMON_GEN_ARGS[@]}"
  else
    echo "Unsupported LAUNCH_MODE=$LAUNCH_MODE. Use device_map, device_map_multi, or ddp."
    exit 1
  fi
fi

if [[ -n "$RESIZE_TO" ]]; then
  bash "${REPO_ROOT}/gen/common/resize_outputs.sh" "$IMAGE_DIR" "$RESIZE_TO" || true
fi

if (( RUN_EVAL == 0 )); then
  exit 0
fi
if (( NUM_NODES > 1 )) && (( NODE_RANK != 0 )); then
  echo "Skipping OneIGBench eval on node ${NODE_RANK}; eval runs only on node 0."
  exit 0
fi

cd "$ONEIG_BENCH_ROOT"
export PYTHONPATH="${ONEIGBENCH_PATCHES_DIR}:${PYTHONPATH:-}"

mkdir -p "$RESULTS_DIR"
CACHE_ROOT="${RESULTS_DIR}/tmp"
mkdir -p "$CACHE_ROOT"
EVAL_GPUS="${EVAL_GPUS:-$(count_visible_gpus "$CUDA_VISIBLE_DEVICES")}"

run_oneig_eval() {
  local module="$1"
  shift
  local common_args=(
    --mode "$ONEIG_MODE"
    --model_names "$MODEL_NAME"
    --image_grid "$IMAGE_GRID"
    --executor_mode "$ONEIGBENCH_EVAL_EXECUTOR"
    --num_workers "$ONEIGBENCH_EVAL_NUM_WORKERS"
    --output_dir "$RESULTS_DIR"
    --cache_root "$CACHE_ROOT"
  )

  if [[ "$ONEIGBENCH_API_EVAL" == "1" ]]; then
    python -m "$module" "${common_args[@]}" --api_eval "$@"
  else
    torchrun --nproc_per_node="$EVAL_GPUS" -m "$module" "${common_args[@]}" "$@"
  fi
}

run_oneig_eval scripts.alignment.alignment_score \
  --image_dirname "$IMAGE_DIR" \
  --class_items "${ALIGNMENT_CLASSES[@]}"

run_oneig_eval scripts.text.text_score \
  --image_dirname "${IMAGE_DIR}/text"

run_oneig_eval scripts.diversity.diversity_score \
  --image_dirname "$IMAGE_DIR" \
  --class_items "${DIVERSITY_CLASSES[@]}"

run_oneig_eval scripts.style.style_score \
  --image_dirname "${IMAGE_DIR}/anime"

run_oneig_eval scripts.reasoning.reasoning_score \
  --image_dirname "${IMAGE_DIR}/reasoning"

ANALYSIS_SCRIPT_PATCHED="${ONEIGBENCH_PATCHES_DIR}/${ANALYSIS_SCRIPT}"
python "$ANALYSIS_SCRIPT_PATCHED" \
  --mode "$ONEIG_MODE" \
  --result_dir "$RESULTS_DIR" \
  --data_file "$DATA_FILE" \
  --output_md "${RESULTS_DIR}/${RESULT_MD_NAME%.md}_${MODEL_NAME}.md"
