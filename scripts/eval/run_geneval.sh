#!/usr/bin/env bash
# Compatibility entrypoint: use the unified experiment layout and prompt switch.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${model_path:-}}"
export OUTPUT_BASE="${OUTPUT_BASE:-${output_path:-}}"
export TASK=geneval
exec bash "${BAGEL_ROOT}/eval_t2i_go.sh"
