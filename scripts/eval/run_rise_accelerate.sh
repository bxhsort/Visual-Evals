#!/usr/bin/env bash
# Run BAGEL RISE generation and the official UMM_Eval judge.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UMM_ROOT="${UMM_ROOT:-${BAGEL_ROOT}/third_party/UMM_Eval}"

MODEL_PATH="${MODEL_PATH:-${model_path:-}}"
MODEL_NAME="${MODEL_NAME:-bagel}"
STEP_NAME="${STEP_NAME:-$(basename "${MODEL_PATH:-step}")}"
OUTPUT_BASE="${OUTPUT_BASE:-${BAGEL_ROOT}/outputs}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
THINK_MODE="${THINK_MODE:-0}"

TXT_CFG="${TXT_CFG:-4.0}"
IMG_CFG="${IMG_CFG:-2.0}"
CFG_RENORM="${CFG_RENORM:-text_channel}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
MIN_IMAGE_SIZE="${MIN_IMAGE_SIZE:-512}"
NUM_STEPS="${NUM_STEPS:-50}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}"
MAX_MEMORY="${MAX_MEMORY:-80GiB}"
BAGEL_ENV="${BAGEL_ENV:-${BAGEL_ROOT}/.venv}"

RISE_DATA_DIR="${RISE_DATA_DIR:-${UMM_ROOT}/gen/it2i/rise/eval/data}"
RISE_DATA_JSON="${RISE_DATA_JSON:-${RISE_DATA_DIR}/datav2_total_w_subtask.json}"
RISE_INPUT_ROOT="${RISE_INPUT_ROOT:-${RISE_DATA_DIR}}"
API_KEY="${API_KEY:-${openai_api_key:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}}}"

if [[ -z "$MODEL_PATH" && "$RUN_GENERATION" != "0" ]]; then
  echo "MODEL_PATH/model_path is required when RUN_GENERATION=1"
  exit 1
fi
if [[ -n "$MODEL_PATH" && ! -e "$MODEL_PATH" ]]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH"
  exit 1
fi
if [[ ! -f "$RISE_DATA_JSON" ]]; then
  echo "RISE metadata does not exist: $RISE_DATA_JSON"
  exit 1
fi

case "${THINK_MODE,,}" in
  1|true|yes|on)
    MODE_SUFFIX="think"
    THINK_ARGS=(--think)
    ;;
  *)
    MODE_SUFFIX="raw"
    THINK_ARGS=()
    ;;
esac
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-rise_${MODE_SUFFIX}}"

OUTPUT_FOLDER="$(PYTHONPATH="${UMM_ROOT}/gen/common:${PYTHONPATH:-}" python - "$TXT_CFG" "$IMG_CFG" "$CFG_RENORM" "$TIMESTEP_SHIFT" "$IMAGE_SIZE" <<'PY'
import sys
import output_paths
print(output_paths.derive_it2i_folder(*sys.argv[1:]))
PY
)"
RESULT_DIR="${OUTPUT_BASE}/${OUTPUT_FOLDER}/${MODEL_NAME}/${STEP_NAME}/${OUTPUT_SUBDIR}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/logs}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
N_GPU="${#GPUS[@]}"
if (( N_GPU <= 0 )); then
  echo "No GPU was provided."
  exit 1
fi

export PYTHONPATH="${BAGEL_ROOT}:${UMM_ROOT}/gen/common:${PYTHONPATH:-}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "[bagel-rise] model=${MODEL_PATH}"
echo "[bagel-rise] result_dir=${RESULT_DIR}"
echo "[bagel-rise] gpu_list=${GPU_LIST} think_mode=${THINK_MODE}"

if (( RUN_GENERATION != 0 )); then
  if [[ -f "${BAGEL_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${BAGEL_ENV}/bin/activate"
  fi
  pids=()
  for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    log="${LOG_DIR}/gen_worker_$(printf '%02d' "$idx")_gpu_${gpu}.log"
    echo "[bagel-rise] worker ${idx}/${N_GPU} gpu=${gpu} log=${log}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python "${BAGEL_ROOT}/eval/gen/gen_images_mp_it2i.py" \
        --benchmark rise \
        --metadata_file "$RISE_DATA_JSON" \
        --image_root "$RISE_INPUT_ROOT" \
        --output_dir "$RESULT_DIR" \
        --shard_id "$idx" \
        --total_shards "$N_GPU" \
        --device 0 \
        --cfg_text_scale "$TXT_CFG" \
        --cfg_img_scale "$IMG_CFG" \
        --cfg_renorm_type "$CFG_RENORM" \
        --timestep_shift "$TIMESTEP_SHIFT" \
        --num_timesteps "$NUM_STEPS" \
        --max_image_size "$IMAGE_SIZE" \
        --min_image_size "$MIN_IMAGE_SIZE" \
        --max_latent_size "$MAX_LATENT_SIZE" \
        --model-path "$MODEL_PATH" \
        --accelerate_device_map \
        --max_memory "$MAX_MEMORY" \
        --offload_folder "${OFFLOAD_FOLDER:-${BAGEL_ROOT}/offload_rise}/worker_${idx}" \
        ${TARGET_INDICES:+--target_indices "$TARGET_INDICES"} \
        "${THINK_ARGS[@]}"
    ) >"$log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[bagel-rise] worker ${idx} failed; tailing log:"
      tail -80 "${LOG_DIR}/gen_worker_$(printf '%02d' "$idx")_gpu_${GPUS[$idx]}.log"
      failed=1
    fi
  done
  if (( failed != 0 )); then
    exit 1
  fi
fi

if (( RUN_EVAL != 0 )); then
  RUN_GENERATION=0 RUN_EVAL=1 \
  MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" OUTPUT_BASE="$OUTPUT_BASE" \
  OUTPUT_FOLDER="$OUTPUT_FOLDER" OUTPUT_SUBDIR="$OUTPUT_SUBDIR" \
  TXT_CFG="$TXT_CFG" IMG_CFG="$IMG_CFG" CFG_RENORM="$CFG_RENORM" \
  TIMESTEP_SHIFT="$TIMESTEP_SHIFT" IMAGE_SIZE="$IMAGE_SIZE" THINK_MODE="$THINK_MODE" \
  RISE_DATA_DIR="$RISE_DATA_DIR" RISE_DATA_JSON="$RISE_DATA_JSON" RISE_INPUT_ROOT="$RISE_INPUT_ROOT" \
  API_KEY="$API_KEY" CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  bash "${UMM_ROOT}/gen/it2i/rise/launch.sh"
fi

echo "[bagel-rise] done"
