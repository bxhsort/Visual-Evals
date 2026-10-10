#!/usr/bin/env bash
# Resolve one managed benchmark leaf before generation, scoring, or diagnostics.
dg2_prepare_experiment() {
  local benchmark="$1" setting_label="$2" variable resolution layout_python effective_json
  local config_args=()
  [[ -n "${EXPERIMENT_NAME:-}" ]] || {
    echo "[experiment] EXPERIMENT_NAME is required; e.g. EXPERIMENT_NAME=bagel-baseline" >&2
    return 2
  }
  layout_python="${EXPERIMENT_PYTHON:-$(command -v python || command -v python3 || true)}"
  [[ -n "$layout_python" ]] || { echo "[experiment] Python is required to resolve experiment results." >&2; return 2; }
  local core_keys="MODEL_PATH MODEL_NAME STEP_NAME THINK_MODE"
  local input_keys
  case "$benchmark" in
    qwenimagebench) input_keys="QIB_METADATA_FILE_EN QIB_METADATA_FILE_ZH" ;;
    longtextbench) input_keys="LONGTEXTBENCH_METADATA_FILE_EN LONGTEXTBENCH_METADATA_FILE_ZH" ;;
    tiifbench) input_keys="METADATA_FILE TIIFBENCH_EVAL_PROMPT_DIR" ;;
    imgedit) input_keys="METADATA_FILE IMAGE_ROOT GENERATION_IMAGE_ROOT PROMPTS_JSON" ;;
    geditbench) input_keys="GEDIT_DATASET_PATH" ;;
    rise) input_keys="RISE_DATA_JSON RISE_INPUT_ROOT" ;;
    editcompass) input_keys="DATA_ROOT BENCHMARK_ROOT" ;;
    *) input_keys="METADATA_FILE" ;;
  esac
  local sampling_keys="IMAGE_SIZE CFG_SCALE CFG_NORM NUM_STEPS NUM_TIMESTEPS NUM_IMAGES BATCH_SIZE MAX_LATENT_SIZE TIMESTEP_SHIFT TXT_CFG IMG_CFG CFG_RENORM CFG_TEXT_SCALE CFG_IMG_SCALE CFG_TYPE MAX_IMAGE_SIZE MIN_IMAGE_SIZE SEED GENERATION_SHARDS TARGET_INDICES TARGET_PROMPT_IDS TARGET_KEYS QIB_LANGS LONGTEXTBENCH_LANGS ONEIGBENCH_LANGS CVTG_SUBSETS CVTG_AREAS BIZGENEVAL_DEFAULT_ASPECT_RATIO WISE_CATEGORIES TIIFBENCH_SPLIT TIIFBENCH_SPECIFIC_FILE QIB_EVAL_BACKEND QIB_DIM_MODE QIB_REWARD_BASE_URL QIB_REWARD_BASE_URLS REWARD_BASE_URL REWARD_BASE_URLS QJUDGER_MODEL_PATH QJUDGER_BATCH_SIZE QJUDGER_MAX_NEW_TOKENS TIIFBENCH_EVAL_MODEL TIIFBENCH_EVAL_BASE_URL TIIFBENCH_API_VERSION EVAL_BACKBONE EVAL_MODEL_NAME INSTRUCTION_LANGUAGES LANGUAGE PARTS METRICS API_BASE OPENAI_BASE_URL"
  [[ "$benchmark" == "geneval" ]] && core_keys+=" GENEVAL_PROMPT_MODE"
  # With an explicit effective config, avoid recording irrelevant inherited knobs.
  if [[ -z "${DG2_EFFECTIVE_CONFIG_JSON:-}" ]]; then
    core_keys+=" $sampling_keys"
  fi
  effective_json="${DG2_EFFECTIVE_CONFIG_JSON:-}"
  [[ -n "$effective_json" ]] || effective_json='{}'
  for variable in $core_keys $input_keys; do
    [[ -n "${!variable:-}" ]] && config_args+=(--config "${variable}=${!variable}")
  done
  if ! resolution="$("$layout_python" "${BAGEL_ROOT}/third_party/UMM_Eval/gen/common/experiment_layout.py" \
      --bench "$benchmark" --output-base "${OUTPUT_BASE:-${BAGEL_ROOT}/outputs}" \
      --experiment-name "$EXPERIMENT_NAME" --setting-label "${EXPERIMENT_SETTING:-$setting_label}" \
      --config-json "$effective_json" "${config_args[@]}")"; then
    return 2
  fi
  DG2_RESULT_DIR="${resolution%%$'\n'*}"
  DG2_SETTING_NAME="${resolution#*$'\n'}"
  export DG2_RESULT_DIR DG2_SETTING_NAME EXPERIMENT_NAME
}
