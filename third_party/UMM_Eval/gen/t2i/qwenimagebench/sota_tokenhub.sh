#!/usr/bin/env bash
# Qwen-Image-Bench SOTA/baseline generation via TokenHub (e.g. gpt-image-2) — GEN ONLY.
# Pure API (CPU) work: generate en+zh images into the standard t2i output layout so the
# local Q-Judger eval path (srun_eval.sh qwenimagebench RUN_GENERATION=0 RUN_EVAL=1
# MODEL_NAME=<model> STEP_NAME=<step>) can score them afterwards on a GPU node.
# Multi-process API requests; resumable.
set -eo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../../.." && pwd)"

source activate bagel
export PATH=/opt/conda/envs/bagel/bin:$PATH
export PYTHONPATH="${REPO_ROOT}/gen/common:${PYTHONPATH:-}"

MODEL_NAME="${SOTA_MODEL_NAME:-gpt-image-2}"
STEP_NAME="${SOTA_STEP_NAME:-api}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs}"
SIZE="${SOTA_SIZE:-1024x1024}"
GEN_WORKERS="${SOTA_GEN_WORKERS:-48}"
MODES="${SOTA_MODES:-en zh}"
TARGET_PROMPT_IDS="${TARGET_PROMPT_IDS:-}"

TARGET_ARGS=()
if [[ -n "$TARGET_PROMPT_IDS" ]]; then
  TARGET_ARGS=(--target_prompt_ids "$TARGET_PROMPT_IDS")
fi

OUTPUT_ROOT="$(python -c "import output_paths;print(output_paths.derive_t2i_root(1024,4.0,3.0,'global'))")"

echo "=== SOTA gen: model=${MODEL_NAME} size=${SIZE} modes='${MODES}' gen_workers=${GEN_WORKERS} -> ${OUTPUT_BASE}/${OUTPUT_ROOT}/${MODEL_NAME}/${STEP_NAME} ==="

for MODE in ${MODES}; do
  python "${BENCH_DIR}/eval/sota_tokenhub_gen.py" \
    --mode "$MODE" --model_name "$MODEL_NAME" --step_name "$STEP_NAME" \
    --output_base "$OUTPUT_BASE" --size "$SIZE" --workers "$GEN_WORKERS" \
    "${TARGET_ARGS[@]}"
done

echo "=== SOTA gen DONE. Score next with:"
echo "    RUN_GENERATION=0 RUN_EVAL=1 MODEL_NAME=${MODEL_NAME} STEP_NAME=${STEP_NAME} bash srun_eval.sh <partition> qwenimagebench ==="
