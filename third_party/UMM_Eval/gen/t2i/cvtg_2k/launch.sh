#!/usr/bin/env bash
# CVTG-2K (t2i) — generation + local TextCrafter scoring (PaddleOCR / CLIP / VQA / aesthetic).
# No API judge.
set -eo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/gen/common:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

MODEL_NAME="${MODEL_NAME:-model}"
STEP_NAME="${STEP_NAME:-step}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_BASE="${OUTPUT_FOLDER:-${OUTPUT_PATH:-${REPO_ROOT}/outputs}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${BENCH_DIR}/eval/CVTG-2K}"
TEXTCRAFTER_ROOT="${TEXTCRAFTER_ROOT:-${BENCH_DIR}/eval/TextCrafter_Eval}"
PADDLEOCR_SOURCE_DIR="${PADDLEOCR_SOURCE_DIR:-/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/ckpt}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-${REPO_ROOT}/../../../.cache/huggingface}}"
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
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
CFG_NORM="${CFG_NORM:-global}"
LOG_DIR="${LOG_DIR:-${OUTPUT_BASE}/logs/cvtg_worker_logs/${MODEL_NAME}/${STEP_NAME}/node_${NODE_RANK}}"
COLLECT_EXPERT_USAGE="${COLLECT_EXPERT_USAGE:-1}"
EXPERT_USAGE_ROOT="${EXPERT_USAGE_ROOT:-}"
CVTG_SUBSETS="${CVTG_SUBSETS:-CVTG,CVTG-Style}"
CVTG_AREAS="${CVTG_AREAS:-2,3,4,5}"
TARGET_KEYS="${TARGET_KEYS:-}"
RESIZE_TO="${RESIZE_TO:-}"

COMMON_ARGS=(
  --model_name "$MODEL_NAME"
  --step_name "$STEP_NAME"
  --model_root "$MODEL_ROOT"
  --benchmark_root "$BENCHMARK_ROOT"
  --image_size "$IMAGE_SIZE"
  --num_steps "$NUM_STEPS"
  --cfg_scale "$CFG_SCALE"
  --cfg_norm "$CFG_NORM"
  --timestep_shift "$TIMESTEP_SHIFT"
  --subsets "$CVTG_SUBSETS"
  --areas "$CVTG_AREAS"
)
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
if [[ -n "$TARGET_KEYS" ]]; then
  EXTRA_ARGS+=(--target_keys "$TARGET_KEYS")
fi

count_visible_gpus() {
  local devices_csv="$1"
  IFS=',' read -r -a _VISIBLE_DEVICES <<< "$devices_csv"
  echo "${#_VISIBLE_DEVICES[@]}"
}

ensure_libgl1() {
  if ldconfig -p 2>/dev/null | grep -q 'libGL\.so\.1'; then
    return 0
  fi

  echo "libGL.so.1 not found. Installing libgl1 for CVTG evaluation..."
  yes | apt-get install libgl1
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
cd "$OUTPUT_BASE"
RESULT_ROOT_DIR="${RESULT_ROOT_DIR:-${RESULT_DIR:-}}"
if [[ -z "$RESULT_ROOT_DIR" ]]; then
  OUTPUT_ROOT="$(build_output_root)"
  RESULT_ROOT_DIR="${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}"
fi
RESULT_DIR="${RESULT_ROOT_DIR}/CVTG"
RESULT_FILE="${RESULT_ROOT_DIR}/CVTG_results.json"

if (( RUN_GENERATION != 0 )); then
  source activate bagel
  export PATH=/opt/conda/envs/bagel/bin:$PATH
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
      worker_log="$LOG_DIR/cvtg_shard_${global_shard_rank}.log"
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
      echo "At least one CVTG worker failed. Check logs under $LOG_DIR."
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

if [[ -n "$RESIZE_TO" ]]; then
  bash "${REPO_ROOT}/gen/common/resize_outputs.sh" "$RESULT_DIR" "$RESIZE_TO"
fi

if (( RUN_EVAL != 0 )); then
  ensure_libgl1
  if [[ ! -d "$BENCHMARK_ROOT" ]]; then
    echo "BENCHMARK_ROOT does not exist: $BENCHMARK_ROOT (see docs/data_setup.md)"
    exit 1
  fi
  if [[ ! -d "$TEXTCRAFTER_ROOT" ]]; then
    echo "TEXTCRAFTER_ROOT does not exist: $TEXTCRAFTER_ROOT (see docs/data_setup.md)"
    exit 1
  fi
  if [[ ! -d "$HOME/.paddleocr" ]]; then
    if [[ -d "$PADDLEOCR_SOURCE_DIR" ]]; then
      cp -r "$PADDLEOCR_SOURCE_DIR" "$HOME/.paddleocr"
    else
      echo "PaddleOCR cache is absent; PaddleOCR will download its public models on first use."
    fi
  fi

  EVAL_GPUS="${EVAL_GPUS:-$(count_visible_gpus "$CUDA_VISIBLE_DEVICES")}"
  source activate geneval
  export PATH=/opt/conda/envs/geneval/bin:$PATH
  cd "$TEXTCRAFTER_ROOT"
  TRANSFORMERS_VERBOSITY=error torchrun --nproc_per_node="$EVAL_GPUS" unified_metrics_eval.py \
    --benchmark_dir "$BENCHMARK_ROOT" \
    --result_dir "$RESULT_DIR" \
    --output_file "$RESULT_FILE" \
    --cache_dir "$HF_CACHE_DIR"
fi
