#!/usr/bin/env bash
# Pack DG2 code only. Absolute AFS links, environments, models and outputs are excluded.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
OUTPUT="${1:-${PARENT}/DG2-code.tar.gz}"

case "$OUTPUT" in
  "$ROOT"/*) echo "Output must be outside DG2: $OUTPUT" >&2; exit 1 ;;
  *.tar.gz|*.tgz) ;;
  *) echo "Output must end in .tar.gz or .tgz" >&2; exit 1 ;;
esac

excludes=(
  "--exclude=${NAME}/.git"
  "--exclude=${NAME}/.venv"
  "--exclude=${NAME}/.conda"
  "--exclude=${NAME}/.cache"
  "--exclude=${NAME}/models"
  "--exclude=${NAME}/outputs"
  "--exclude=${NAME}/offload_*"
  "--exclude=${NAME}/config/eval.env"
  "--exclude=${NAME}/eval_go copy.sh"
  "--exclude=${NAME}/night.sh"
  "--exclude=${NAME}/night2.sh"
  "--exclude=${NAME}/example.txt"
  "--exclude=${NAME}/third_party/UMM_Eval/config/cluster.env"
  "--exclude=*/.git"
  "--exclude=*/__pycache__"
  "--exclude=*.pyc"
)

# Absolute links point to this machine's AFS and are unusable on the target machine.
while IFS= read -r link; do
  target="$(readlink "$link")"
  if [[ "$target" == /* ]]; then
    excludes+=("--exclude=${NAME}/${link#${ROOT}/}")
  fi
done < <(find "$ROOT" -type l -print)

mkdir -p "$(dirname "$OUTPUT")"
tar -C "$PARENT" "${excludes[@]}" -czf "$OUTPUT" "$NAME"
(
  cd "$(dirname "$OUTPUT")"
  sha256sum "$(basename "$OUTPUT")" >"$(basename "$OUTPUT").sha256"
)

echo "Created $OUTPUT"
du -sh "$OUTPUT"
