#!/usr/bin/env bash
# Coverage diagnostics are sidecars; scorer exit codes and scores stay unchanged.

dg2_run_coverage_audit() {
  local audit_python
  audit_python="${COVERAGE_PYTHON:-$(command -v python || command -v python3 || true)}"
  local audit_script="${BAGEL_ROOT}/third_party/UMM_Eval/gen/common/coverage_audit.py"
  if [[ -z "$audit_python" || ! -f "$audit_script" ]]; then
    echo "[coverage] WARNING: coverage reporter/interpreter unavailable; completeness is unverified." >&2
    return 0
  fi
  if ! "$audit_python" "$audit_script" "$@"; then
    echo "[coverage] WARNING: diagnostics failed; existing scores and evaluation exit status are preserved." >&2
  fi
  return 0
}

dg2_export_gedit_expected() {
  local source="$1" destination="$2" audit_python
  audit_python="${COVERAGE_PYTHON:-$(command -v python || command -v python3 || true)}"
  [[ -n "$audit_python" && -d "$source" ]] || return 1
  "$audit_python" - "$source" "$destination" <<'PY'
import json
import os
import sys
from pathlib import Path
from datasets import load_from_disk

source, destination = map(Path, sys.argv[1:])
data = load_from_disk(str(source))
if not hasattr(data, "column_names") or isinstance(data.column_names, dict):
    data = data["train"]
rows = [{key: row[key] for key in ("key", "task_type", "instruction_language")} for row in data]
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(destination.name + ".tmp." + str(os.getpid()))
temporary.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
temporary.replace(destination)
PY
}

dg2_coverage_on_exit() {
  local evaluation_status=$?
  trap - EXIT
  if [[ "${COVERAGE_REPORT:-1}" != "0" && "${RUN_EVAL:-1}" != "0" ]]; then
    if ! dg2_write_coverage_reports; then
      echo "[coverage] WARNING: could not finish diagnostics; completeness is unverified." >&2
    fi
  fi
  exit "$evaluation_status"
}

dg2_install_coverage_trap() {
  trap dg2_coverage_on_exit EXIT
}
