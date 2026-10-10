#!/usr/bin/env bash
# BizGenEval (t2i) — generation + VLM checklist judging (gemini via tokenhub).
# Conda hooks read unset vars, so no -u.
set -eo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/config/conda_envs.env"
source "${CONDA_SH}"
umm_conda_activate "${GEN_ENV}"
export PYTHONPATH="${REPO_ROOT}/gen/common:${PYTHONPATH:-}"

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

IMAGE_SIZE="${IMAGE_SIZE:-2048}"
DEFAULT_ASPECT_RATIO="${DEFAULT_ASPECT_RATIO:-1:1}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
CFG_NORM="${CFG_NORM:-global}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-bizgeneval}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs/bizgeneval_worker_logs/node_${NODE_RANK}}"
COLLECT_EXPERT_USAGE="${COLLECT_EXPERT_USAGE:-0}"
EXPERT_USAGE_ROOT="${EXPERT_USAGE_ROOT:-}"
TARGET_PROMPT_IDS="${TARGET_PROMPT_IDS:-}"

DATA_PATH="${DATA_PATH:-${BENCH_DIR}/eval/test.jsonl}"
PROMPT_FIELD="${PROMPT_FIELD:-prompt}"
PE_MODE="${PE_MODE:-raw}"
PE_PROMPT_PATH="${REPO_ROOT}/tools/external_gen/pe_enhanced/bizgeneval_en.jsonl"

# VLM checklist judge (gemini via tokenhub). JUDGE_API_KEY optional (else resolved from api_key.yaml).
JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.1-pro-preview}"
JUDGE_API_BASE="${JUDGE_API_BASE:-${EVAL_API_BASE_URL:-${OPENAI_BASE_URL:-https://tokenhub.sensetime.com/v1}}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-8}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-240}"
EVAL_FORCE_RERUN="${EVAL_FORCE_RERUN:-0}"
EVAL_MAX_ITEMS="${EVAL_MAX_ITEMS:-}"

COMMON_ARGS=(
  --model_name "$MODEL_NAME"
  --step_name "$STEP_NAME"
  --output_subdir "$OUTPUT_SUBDIR"
  --resolution "$IMAGE_SIZE"
  --default_aspect_ratio "$DEFAULT_ASPECT_RATIO"
  --data_path "$DATA_PATH"
  --num_steps "$NUM_STEPS"
  --cfg_scale "$CFG_SCALE"
  --cfg_norm "$CFG_NORM"
  --timestep_shift "$TIMESTEP_SHIFT"
  --batch_size "$BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --prompt_field "$PROMPT_FIELD"
)
case "$PE_MODE" in
  raw) ;;
  enhanced)
    [[ -f "$PE_PROMPT_PATH" ]] || { echo "Enhanced prompt cache not found: $PE_PROMPT_PATH"; exit 1; }
    COMMON_ARGS+=(--pe_prompt_path "$PE_PROMPT_PATH") ;;
  *) echo "Unsupported PE_MODE=$PE_MODE. Use raw or enhanced."; exit 1 ;;
esac
if [[ -n "$MODEL_PATH" ]]; then
  COMMON_ARGS+=(--model_path "$MODEL_PATH")
fi

EXTRA_ARGS=()
if (( COLLECT_EXPERT_USAGE != 0 )); then
  EXTRA_ARGS+=(--collect_expert_usage)
  if [[ -n "$EXPERT_USAGE_ROOT" ]]; then
    EXTRA_ARGS+=(--expert_usage_root "$EXPERT_USAGE_ROOT")
  fi
fi
if [[ -n "$TARGET_PROMPT_IDS" ]]; then
  EXTRA_ARGS+=(--target_prompt_ids "$TARGET_PROMPT_IDS")
fi

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
if [[ ! -f "$DATA_PATH" ]]; then
  echo "BizGenEval data file not found: $DATA_PATH"
  echo "Set DATA_PATH to point at test.jsonl."
  exit 1
fi

mkdir -p "$OUTPUT_BASE"
cd "$OUTPUT_BASE"
OUTPUT_ROOT="$(build_output_root)"
BENCH_OUTPUT_PATH="${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}/${OUTPUT_SUBDIR}"

if (( RUN_GENERATION != 0 )); then
  if [[ "$LAUNCH_MODE" == "device_map" ]]; then
    export CUDA_VISIBLE_DEVICES
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
    TRANSFORMERS_VERBOSITY=error python "${BENCH_DIR}/inference.py" \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" \
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
      worker_log="$LOG_DIR/bizgeneval_shard_${global_shard_rank}.log"
      echo "Launching global shard ${global_shard_rank}/${TOTAL_NUM_SHARDS} on node ${NODE_RANK}/${NUM_NODES} using GPUs ${worker_devices}. Log: ${worker_log}"
      (
        export CUDA_VISIBLE_DEVICES="$worker_devices"
        # Per-worker module cache so concurrent trust_remote_code loads do not race.
        export HF_MODULES_CACHE="${LOG_DIR}/hf_modules/shard_${global_shard_rank}"
        mkdir -p "$HF_MODULES_CACHE"
        unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK
        TRANSFORMERS_VERBOSITY=error python "${BENCH_DIR}/inference.py" \
          "${COMMON_ARGS[@]}" \
          "${EXTRA_ARGS[@]}" \
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
      echo "At least one bizgeneval worker failed. Check logs under $LOG_DIR."
      exit 1
    fi
  elif [[ "$LAUNCH_MODE" == "ddp" ]]; then
    TRANSFORMERS_VERBOSITY=error torchrun --nproc_per_node="$GPUS" "${BENCH_DIR}/inference.py" \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  else
    echo "Unsupported LAUNCH_MODE=$LAUNCH_MODE. Use device_map, device_map_multi, or ddp."
    exit 1
  fi
fi

if (( RUN_EVAL == 0 )); then
  exit 0
fi
if (( NUM_NODES > 1 )) && (( NODE_RANK != 0 )); then
  echo "Skipping BizGenEval eval on node ${NODE_RANK}; eval runs only on node 0."
  exit 0
fi

# ---- VLM checklist judging (gemini via tokenhub) ----
umm_conda_activate "${BIZGENEVAL_EVAL_ENV:-$GEN_ENV}"

EVAL_OUTPUT_DIR="${BENCH_OUTPUT_PATH}_eval"
mkdir -p "$EVAL_OUTPUT_DIR"

EVAL_ARGS=(
  --image-dir "$BENCH_OUTPUT_PATH"
  --output-dir "$EVAL_OUTPUT_DIR"
  --data-path "$DATA_PATH"
  --api-base "$JUDGE_API_BASE"
  --judge-model "$JUDGE_MODEL"
  --concurrency "$EVAL_CONCURRENCY"
  --eval-timeout "$EVAL_TIMEOUT"
)
if [[ -n "$JUDGE_API_KEY" ]]; then
  EVAL_ARGS+=(--api-key "$JUDGE_API_KEY")
fi
if (( EVAL_FORCE_RERUN != 0 )); then
  EVAL_ARGS+=(--force-rerun)
fi
if [[ -n "$EVAL_MAX_ITEMS" ]]; then
  EVAL_ARGS+=(--max-items "$EVAL_MAX_ITEMS")
fi

python "${BENCH_DIR}/eval/eval_bizgeneval.py" "${EVAL_ARGS[@]}"
