#!/usr/bin/env bash
# TIIF-Bench (t2i) — generation + gpt-4o (tokenhub) VLM-judge scoring.
# Conda hooks read unset vars, so no -u.
set -eo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/config/conda_envs.env"
source "${CONDA_SH}"
umm_conda_activate "${GEN_ENV}"
export PYTHONPATH="${REPO_ROOT}/gen/common:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TRANSFORMERS_VERBOSITY=error

TIIF_BENCH_ROOT="${TIIF_BENCH_ROOT:-${BENCH_DIR}/eval/TIIF-Bench}"

MODEL_NAME="${MODEL_NAME:-model}"
STEP_NAME="${STEP_NAME:-step}"
MODEL_PATH="${MODEL_PATH:-}"
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
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4.0}"
CFG_NORM="${CFG_NORM:-global}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRID_ROWS="${GRID_ROWS:-1}"
GRID_COLS="${GRID_COLS:-1}"
OUTPUT_BASE="${OUTPUT_FOLDER:-${OUTPUT_PATH:-${REPO_ROOT}/outputs}}"

TIIFBENCH_SPLIT="${TIIFBENCH_SPLIT:-testmini}"
TIIFBENCH_PROMPT_DIR="${TIIFBENCH_PROMPT_DIR:-${TIIF_BENCH_ROOT}/data/${TIIFBENCH_SPLIT}_prompts}"
TIIFBENCH_EVAL_PROMPT_DIR="${TIIFBENCH_EVAL_PROMPT_DIR:-${TIIF_BENCH_ROOT}/data/${TIIFBENCH_SPLIT}_eval_prompts}"
TIIFBENCH_EVAL_MODEL="${TIIFBENCH_EVAL_MODEL:-gpt-4o}"
TIIFBENCH_EVAL_BASE_URL="${TIIFBENCH_EVAL_BASE_URL:-${EVAL_API_BASE_URL:-${OPENAI_BASE_URL:-https://tokenhub.sensetime.com/v1}}}"
TIIFBENCH_API_VERSION="${TIIFBENCH_API_VERSION:-2025-01-01-preview}"
TIIFBENCH_SPECIFIC_FILE="${TIIFBENCH_SPECIFIC_FILE:-}"
# API_KEY is optional: when empty, resolve_eval_api_key reads api_key.yaml:token_hub_eval.
API_KEY="${API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}}"
RESIZE_TO="${RESIZE_TO:-}"

count_visible_gpus() {
  local devices_csv="$1"
  IFS=',' read -r -a _VISIBLE_DEVICES <<< "$devices_csv"
  echo "${#_VISIBLE_DEVICES[@]}"
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
if [[ ! -d "$TIIFBENCH_PROMPT_DIR" ]]; then
  echo "Prompt dir not found: $TIIFBENCH_PROMPT_DIR (see docs/data_setup.md)"
  exit 1
fi
if [[ ! -d "$TIIFBENCH_EVAL_PROMPT_DIR" ]]; then
  echo "Eval prompt dir not found: $TIIFBENCH_EVAL_PROMPT_DIR (see docs/data_setup.md)"
  exit 1
fi

mkdir -p "$OUTPUT_BASE"
OUTPUT_ROOT="$(build_output_root)"
RESULT_ROOT="${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME}"
BENCH_NAME="TIIFBench-${TIIFBENCH_SPLIT}"
IMAGE_DIR="${RESULT_ROOT}/${BENCH_NAME}"
RESULTS_DIR="${RESULT_ROOT}/tiifbench-${TIIFBENCH_SPLIT}_results"

if (( RUN_GENERATION != 0 )); then
  if [[ -z "$MODEL_PATH" ]]; then
    echo "MODEL_PATH is required for generation."
    exit 1
  fi

  GEN_ARGS=(
    --model_path "$MODEL_PATH"
    --model_name "$MODEL_NAME"
    --output_path "$IMAGE_DIR"
    --resolution "$IMAGE_SIZE"
    --cfg_scale "$CFG_SCALE"
    --cfg_norm "$CFG_NORM"
    --timestep_shift "$TIMESTEP_SHIFT"
    --num_steps "$NUM_STEPS"
    --batch_size "$BATCH_SIZE"
    --rows "$GRID_ROWS"
    --cols "$GRID_COLS"
    --input_folder "$TIIFBENCH_PROMPT_DIR"
    --language en
    --num_shards "$NUM_NODES"
    --shard_rank "$NODE_RANK"
  )
  if [[ -n "$TIIFBENCH_SPECIFIC_FILE" ]]; then
    GEN_ARGS+=(--specific_file "$TIIFBENCH_SPECIFIC_FILE")
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$GPUS" "${BENCH_DIR}/inference.py" "${GEN_ARGS[@]}"
fi

if [[ -n "$RESIZE_TO" ]]; then
  bash "${REPO_ROOT}/gen/common/resize_outputs.sh" "$IMAGE_DIR" "$RESIZE_TO"
fi

if (( RUN_EVAL == 0 )); then
  exit 0
fi
if (( NUM_NODES > 1 )) && (( NODE_RANK != 0 )); then
  echo "Skipping TIIFBench eval on node ${NODE_RANK}; eval runs only on node 0."
  exit 0
fi

# API_KEY is optional: the patched scorer resolves it from env API_KEY or
# api_key.yaml:token_hub_eval (via gen/common/eval_api_key.py) when not passed.
umm_conda_activate "${TIIFBENCH_EVAL_ENV:-${GEN_ENV}}"
cd "$TIIF_BENCH_ROOT"

mkdir -p "$RESULTS_DIR"
EVAL_JSON_DIR="${RESULTS_DIR}/eval_json"
mkdir -p "$EVAL_JSON_DIR"

# Route the judge through the tokenhub OpenAI-compatible gateway via the patched scorer
# (upstream eval_with_vlm_mp.py is AzureOpenAI-only). Run by absolute path so cwd doesn't matter.
python "${BENCH_DIR}/eval/eval_with_vlm_mp_tokenhub.py" \
  --jsonl_dir "$TIIFBENCH_EVAL_PROMPT_DIR" \
  --image_dir "$IMAGE_DIR" \
  --eval_model "$MODEL_NAME" \
  --output_dir "$EVAL_JSON_DIR" \
  --model "$TIIFBENCH_EVAL_MODEL" \
  --api_key "$API_KEY" \
  --base_url "$TIIFBENCH_EVAL_BASE_URL"

python eval/summary_results.py --input_dir "$EVAL_JSON_DIR"
python eval/summary_dimension_results.py \
  --input_excel "${EVAL_JSON_DIR}/result_summary.xlsx" \
  --output_txt "${RESULTS_DIR}/result_summary_dimension.txt"
