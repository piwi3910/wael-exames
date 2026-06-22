#!/bin/bash
# Grade every input exam PDF (in/) through the local DGX pipeline, writing grades to out/.
# Usage: ./grade_all.sh [IN_DIR] [OUT_DIR]   (defaults: in, out)
set -euo pipefail

IN_DIR="${1:-in}"
OUT_DIR="${2:-out}"
cd "$(dirname "$0")"

declare -a PAPERS=("English paper.pdf" "Math paper.pdf" "SET paper.pdf")

for name in "${PAPERS[@]}"; do
    pdf="$IN_DIR/$name"
    if [ ! -f "$pdf" ]; then
        echo "skip: '$pdf' not found"
        continue
    fi
    subject="${name%.pdf}"  # subject = file name without extension
    echo "=== grading: $pdf ==="
    uv run python grade.py "$pdf" --subject "$subject" --out "$OUT_DIR"
done

echo
echo "Done. Reports under $OUT_DIR/ :"
ls -1 "$OUT_DIR"/*.report.md 2>/dev/null || true
