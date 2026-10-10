#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash simple_judge_rise.sh <api_key_or_-> <result_image_dir>
  bash simple_judge_rise.sh <api_key_or_-> <judge_model_note> <result_image_dir>

Examples:
  bash simple_judge_rise.sh sk-xxx /path/to/eval_output_edit_txtcfg4_imgcfg1.5/model/step/rise_raw
  bash simple_judge_rise.sh - gpt-4o /path/to/eval_output_edit_txtcfg4_imgcfg1.5/model/step/rise_think

Notes:
  - <result_image_dir> should be the RISE result directory that contains <category>/*.png.
  - The optional <judge_model_note> is accepted for interface compatibility only.
    The current upstream RISE gpt_eval.py fixes its Azure deployment internally.
  - By default the script uses 8 slices and runs them in parallel.
  - Final outputs follow launch_rise.sh and are written to:
      <result_image_dir>/<prefix>.pkl
      <result_image_dir>/<prefix>_judge.xlsx
      <result_image_dir>/<prefix>_judge.csv
EOF
}

safe_conda_activate() {
  local had_nounset=0
  case $- in
    *u*) had_nounset=1 ;;
  esac
  set +u
  conda activate "$1"
  if (( had_nounset != 0 )); then
    set -u
  fi
}

split_rise_data() {
  local json_path="$1"
  local split_dir="$2"
  local selected_json_path="$3"
  local num_slices="$4"
  local target_indices="$5"

  python - "$json_path" "$split_dir" "$selected_json_path" "$num_slices" "$target_indices" <<'PY'
import json
import os
import shutil
import sys

json_path, split_dir, selected_json_path, num_slices_raw, target_indices_raw = sys.argv[1:6]
num_slices = int(num_slices_raw)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

targets = {
    value.strip()
    for value in target_indices_raw.split(",")
    if value.strip()
}

def matches(index):
    if not targets:
        return True
    index_str = str(index).strip()
    if index_str in targets:
        return True
    return index_str.rsplit("_", 1)[-1] in targets

data = [item for item in data if matches(item.get("index"))]
data = sorted(data, key=lambda item: str(item.get("index")))

shutil.rmtree(split_dir, ignore_errors=True)
os.makedirs(split_dir, exist_ok=True)
os.makedirs(os.path.dirname(selected_json_path), exist_ok=True)

with open(selected_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

slices = [[] for _ in range(num_slices)]
for index, item in enumerate(data):
    slices[index % num_slices].append(item)

for slice_index, shard in enumerate(slices):
    if not shard:
        continue
    path = os.path.join(split_dir, f"slice_{slice_index:02d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(shard, f, ensure_ascii=False, indent=2)
    print(path)
PY
}

merge_slice_pickles() {
  local slice_root="$1"
  local final_pkl="$2"

  python - "$slice_root" "$final_pkl" <<'PY'
import glob
import os
import pickle
import sys

slice_root, final_pkl = sys.argv[1:3]
merged = {}

for path in sorted(glob.glob(os.path.join(slice_root, "**", "*.pkl"), recursive=True)):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        merged.update(payload)

os.makedirs(os.path.dirname(final_pkl), exist_ok=True)
with open(final_pkl, "wb") as f:
    pickle.dump(merged, f)

print(f"merged_records={len(merged)}")
PY
}

if [[ $# -eq 2 ]]; then
  API_KEY_RAW="$1"
  JUDGE_MODEL_NOTE="upstream-fixed"
  RESULT_IMAGE_DIR="$2"
elif [[ $# -eq 3 ]]; then
  API_KEY_RAW="$1"
  JUDGE_MODEL_NOTE="$2"
  RESULT_IMAGE_DIR="$3"
else
  usage
  exit 1
fi

if [[ ! -d "$RESULT_IMAGE_DIR" ]]; then
  echo "Result image directory does not exist: $RESULT_IMAGE_DIR"
  exit 1
fi

RESULT_IMAGE_DIR="$(cd "$RESULT_IMAGE_DIR" && pwd)"

if ! find "$RESULT_IMAGE_DIR" -type f -name '*.png' -print -quit | grep -q .; then
  echo "No png images found under: $RESULT_IMAGE_DIR"
  echo "Expected layout: <result_dir>/<category>/*.png"
  exit 1
fi

API_KEY_FROM_ENV="${API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}}"
if [[ "$API_KEY_RAW" == "-" || "$API_KEY_RAW" == "none" || "$API_KEY_RAW" == "NONE" ]]; then
  API_KEY="$API_KEY_FROM_ENV"
else
  API_KEY="$API_KEY_RAW"
fi

if [[ -z "$API_KEY" ]]; then
  echo "API key is required for RISE evaluation."
  echo "Pass it as the first argument or set API_KEY / AZURE_OPENAI_API_KEY / OPENAI_API_KEY."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RISE_ROOT="${RISE_ROOT:-${SCRIPT_DIR}}"
RISE_EVAL_ROOT="${RISE_EVAL_ROOT:-${RISE_ROOT}}"
RISE_DATA_JSON="${RISE_DATA_JSON:-${RISE_ROOT}/data/datav2_total_w_subtask.json}"
RISE_INPUT_ROOT="${RISE_INPUT_ROOT:-${RISE_ROOT}/data}"
EVAL_ENV="${EVAL_ENV:-${RISE_EVAL_ENV:-${GEN_ENV:-}}}"
FINAL_PREFIX="${RISE_PREFIX:-$(basename "$RESULT_IMAGE_DIR")}"
NUM_SLICES="${NUM_SLICES:-8}"
TOTAL_NPROC="${RISE_NPROC:-${EVAL_NUM_PROCESSES:-96}}"
TARGET_INDICES="${TARGET_INDICES:-}"
SLICE_ROOT="${SLICE_ROOT:-${RESULT_IMAGE_DIR}/.rise_judge_slices_${FINAL_PREFIX}}"
LOG_DIR="${LOG_DIR:-${RESULT_IMAGE_DIR}/rise_judge_logs_${FINAL_PREFIX}}"

if [[ ! -d "$RISE_EVAL_ROOT" ]]; then
  echo "RISE_EVAL_ROOT does not exist: $RISE_EVAL_ROOT"
  exit 1
fi

if [[ ! -f "$RISE_DATA_JSON" ]]; then
  echo "RISE_DATA_JSON does not exist: $RISE_DATA_JSON"
  exit 1
fi

if [[ ! -d "$RISE_INPUT_ROOT" ]]; then
  echo "RISE_INPUT_ROOT does not exist: $RISE_INPUT_ROOT"
  exit 1
fi

if [[ ! -f "${RISE_EVAL_ROOT}/gpt_eval.py" ]]; then
  echo "RISE evaluator not found: ${RISE_EVAL_ROOT}/gpt_eval.py"
  exit 1
fi

if ! [[ "$NUM_SLICES" =~ ^[0-9]+$ ]] || (( NUM_SLICES <= 0 )); then
  echo "NUM_SLICES must be a positive integer, got: $NUM_SLICES"
  exit 1
fi

if ! [[ "$TOTAL_NPROC" =~ ^[0-9]+$ ]] || (( TOTAL_NPROC <= 0 )); then
  echo "RISE_NPROC must be a positive integer, got: $TOTAL_NPROC"
  exit 1
fi

if [[ -n "${NPROC_PER_SLICE:-}" ]]; then
  if ! [[ "$NPROC_PER_SLICE" =~ ^[0-9]+$ ]] || (( NPROC_PER_SLICE <= 0 )); then
    echo "NPROC_PER_SLICE must be a positive integer, got: $NPROC_PER_SLICE"
    exit 1
  fi
else
  NPROC_PER_SLICE=$(((TOTAL_NPROC + NUM_SLICES - 1) / NUM_SLICES))
  if (( NPROC_PER_SLICE <= 0 )); then
    NPROC_PER_SLICE=1
  fi
fi

mkdir -p "$SLICE_ROOT" "$LOG_DIR"

if [[ -z "${CONDA_SH:-}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
[[ -n "$EVAL_ENV" ]] || { echo "EVAL_ENV or RISE_EVAL_ENV is required"; exit 1; }
[[ -f "${CONDA_SH:-}" ]] || { echo "CONDA_SH is required"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
safe_conda_activate "$EVAL_ENV"
cd "$RISE_EVAL_ROOT"

SLICE_RUN_TAG="$(python - "$JUDGE_MODEL_NOTE" "$NUM_SLICES" "$TARGET_INDICES" "$FINAL_PREFIX" <<'PY'
import hashlib
import sys

payload = "||".join(sys.argv[1:5])
print(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12])
PY
)"

RUN_ROOT="${SLICE_ROOT}/${SLICE_RUN_TAG}"
INPUT_SPLIT_DIR="${RUN_ROOT}/inputs"
SELECTED_DATA_JSON="${RUN_ROOT}/selected.json"
ARTIFACT_SUBDIR=".rise_judge_slices_${FINAL_PREFIX}/${SLICE_RUN_TAG}"
ARTIFACT_ROOT="${RESULT_IMAGE_DIR}/${ARTIFACT_SUBDIR}"
mkdir -p "$RUN_ROOT" "$ARTIFACT_ROOT"

echo "Running RISE scoring..."
echo "result_dir=$RESULT_IMAGE_DIR"
echo "prefix=$FINAL_PREFIX"
echo "judge_model_note=$JUDGE_MODEL_NOTE"
echo "num_slices=$NUM_SLICES"
echo "nproc_per_slice=$NPROC_PER_SLICE"
if [[ $# -eq 3 ]]; then
  echo "Note: judge_model_note is for compatibility only; upstream RISE gpt_eval.py fixes the Azure deployment internally."
fi
if [[ -n "$TARGET_INDICES" ]]; then
  echo "target_indices=$TARGET_INDICES"
fi

mapfile -t slice_jsons < <(split_rise_data "$RISE_DATA_JSON" "$INPUT_SPLIT_DIR" "$SELECTED_DATA_JSON" "$NUM_SLICES" "$TARGET_INDICES")

if (( ${#slice_jsons[@]} == 0 )); then
  echo "No prompts to evaluate after filtering."
  exit 1
fi

pids=()
failed=0

for slice_json in "${slice_jsons[@]}"; do
  slice_name="$(basename "$slice_json" .json)"
  slice_prefix="${ARTIFACT_SUBDIR}/${slice_name}/${FINAL_PREFIX}"
  slice_output_dir="${RESULT_IMAGE_DIR}/$(dirname "$slice_prefix")"
  worker_log="${LOG_DIR}/${slice_name}.log"
  mkdir -p "$slice_output_dir"

  echo "Launching ${slice_name}. Log: ${worker_log}"
  (
    python gpt_eval.py \
      --data "$slice_json" \
      --output "$RESULT_IMAGE_DIR" \
      --input "$RISE_INPUT_ROOT" \
      --prefix "$slice_prefix" \
      --api_key "$API_KEY" \
      --nproc "$NPROC_PER_SLICE"
  ) >"$worker_log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "At least one RISE judge slice failed. Check logs under $LOG_DIR."
  exit 1
fi

FINAL_PKL="${RESULT_IMAGE_DIR}/${FINAL_PREFIX}.pkl"
echo "Merging slice judge caches..."
merge_slice_pickles "$ARTIFACT_ROOT" "$FINAL_PKL"

echo "Running final RISE aggregation..."
python gpt_eval.py \
  --data "$SELECTED_DATA_JSON" \
  --output "$RESULT_IMAGE_DIR" \
  --input "$RISE_INPUT_ROOT" \
  --prefix "$FINAL_PREFIX" \
  --api_key "$API_KEY" \
  --nproc 1 | tee "${LOG_DIR}/final_aggregate.log"

echo "Done."
echo "Final outputs:"
echo "  ${RESULT_IMAGE_DIR}/${FINAL_PREFIX}.pkl"
echo "  ${RESULT_IMAGE_DIR}/${FINAL_PREFIX}_judge.xlsx"
echo "  ${RESULT_IMAGE_DIR}/${FINAL_PREFIX}_judge.csv"
