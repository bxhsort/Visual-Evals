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
for legacy_output in RESULT_DIR RESULT_ROOT_DIR OUTPUT_PATH output_path OUTPUT_DIR output_dir OUTPUT_ROOT OUTPUT_SUBDIR OUTPUT_FOLDER RISE_OUTPUT_BASE; do
  if [[ -n "${!legacy_output:-}" ]]; then
    echo "${legacy_output} bypasses the experiment layout; use OUTPUT_BASE and EXPERIMENT_NAME instead." >&2
    exit 1
  fi
done
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
THINK_MODE="${THINK_MODE:-0}"
TARGET_INDICES="${TARGET_INDICES:-}"

TXT_CFG="${TXT_CFG:-4.0}"
IMG_CFG="${IMG_CFG:-2.0}"
CFG_RENORM="${CFG_RENORM:-text_channel}"
TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-3.0}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
MIN_IMAGE_SIZE="${MIN_IMAGE_SIZE:-512}"
NUM_STEPS="${NUM_STEPS:-50}"
MAX_LATENT_SIZE="${MAX_LATENT_SIZE:-64}"
MAX_MEMORY="${MAX_MEMORY:-80GiB}"

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

case "$(printf '%s' "$THINK_MODE" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    MODE_SUFFIX="think"
    THINK_ARGS=(--think)
    ;;
  *)
    MODE_SUFFIX="raw"
    THINK_ARGS=()
    ;;
esac
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
N_GPU="${#GPUS[@]}"
GENERATION_SHARDS="$N_GPU"
DG2_EFFECTIVE_CONFIG_JSON="$(python - "$TXT_CFG" "$IMG_CFG" "$CFG_RENORM" "$TIMESTEP_SHIFT" \
    "$IMAGE_SIZE" "$MIN_IMAGE_SIZE" "$NUM_STEPS" "$MAX_LATENT_SIZE" "$GENERATION_SHARDS" \
    "$RISE_DATA_JSON" "$RISE_INPUT_ROOT" "${JUDGE_MODEL:-gpt-4o}" "$TARGET_INDICES" <<'PY'
import json, sys
txt, img, renorm, shift, size, minimum, steps, latent, shards, metadata, images, judge, target = sys.argv[1:]
print(json.dumps(dict(TXT_CFG=float(txt), IMG_CFG=float(img), CFG_RENORM=renorm,
    CFG_TYPE="serial_text_img", CFG_INTERVAL=[0, 1], TIMESTEP_SHIFT=float(shift),
    IMAGE_SIZE=int(size), MIN_IMAGE_SIZE=int(minimum), NUM_STEPS=int(steps),
    MAX_LATENT_SIZE=int(latent), SEED=42, SEED_STRATEGY="base_plus_stable_index",
    MAX_LENGTH=2048, TEMPERATURE=0.3, CROP_BBOX=True, GENERATOR="gen_images_mp_it2i.py",
    GENERATION_SHARDS=int(shards), RISE_DATA_JSON=metadata, IMAGE_ROOT=images,
    JUDGE_MODEL=judge, TARGET_INDICES=target)))
PY
)"
SETTING_LABEL="$(PYTHONPATH="${UMM_ROOT}/gen/common:${PYTHONPATH:-}" python - "$TXT_CFG" "$IMG_CFG" "$CFG_RENORM" "$TIMESTEP_SHIFT" "$IMAGE_SIZE" <<'PY'
import sys
import output_paths
print(output_paths.derive_it2i_folder(*sys.argv[1:]))
PY
)"
source "${SCRIPT_DIR}/experiment_layout.sh"
dg2_prepare_experiment rise "$SETTING_LABEL"
RESULT_DIR="$DG2_RESULT_DIR"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/logs}"

source "${SCRIPT_DIR}/coverage_report.sh"
dg2_write_coverage_reports() {
  dg2_run_coverage_audit --bench rise --results-root "$RESULT_DIR" --model-name "$MODEL_NAME" \
      --metadata-file "${COVERAGE_METADATA_FILE:-$RISE_DATA_JSON}" \
      --image-dir "${COVERAGE_IMAGE_DIR:-$RESULT_DIR}" --judge-root "${COVERAGE_JUDGE_ROOT:-$RESULT_DIR}"
}
dg2_install_coverage_trap

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
  source activate bagel
  export PATH=/opt/conda/envs/bagel/bin:$PATH
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
  source activate geneval
  export PATH=/opt/conda/envs/geneval/bin:$PATH
  RUN_GENERATION=0 RUN_EVAL=1 \
  MODEL_NAME="$MODEL_NAME" STEP_NAME="$STEP_NAME" OUTPUT_BASE="$OUTPUT_BASE" \
  OUTPUT_PATH="$RESULT_DIR" RESULT_ROOT_DIR="$RESULT_DIR" LOG_DIR="$LOG_DIR" \
  RISE_PREFIX="rise_${MODE_SUFFIX}" \
  JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}" \
  TXT_CFG="$TXT_CFG" IMG_CFG="$IMG_CFG" CFG_RENORM="$CFG_RENORM" \
  TIMESTEP_SHIFT="$TIMESTEP_SHIFT" IMAGE_SIZE="$IMAGE_SIZE" THINK_MODE="$THINK_MODE" \
  RISE_DATA_DIR="$RISE_DATA_DIR" RISE_DATA_JSON="$RISE_DATA_JSON" RISE_INPUT_ROOT="$RISE_INPUT_ROOT" \
  API_KEY="$API_KEY" CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  bash "${UMM_ROOT}/gen/it2i/rise/launch.sh"
fi

echo "[bagel-rise] done"
