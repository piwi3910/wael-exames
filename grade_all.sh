#!/bin/bash
# Grade every sample exam PDF in this folder through the local DGX pipeline.
# Usage: ./grade_all.sh [OUT_DIR]   (OUT_DIR defaults to "out")
set -euo pipefail

OUT_DIR="${1:-out}"
cd "$(dirname "$0")"

# subject = file name without extension
declare -a PAPERS=("English paper.pdf" "Math paper.pdf" "SET paper.pdf")

for pdf in "${PAPERS[@]}"; do
    if [ ! -f "$pdf" ]; then
        echo "skip: '$pdf' not found"
        continue
    fi
    subject="${pdf%.pdf}"
    echo "=== grading: $pdf ==="
    uv run python grade.py "$pdf" --subject "$subject" --out "$OUT_DIR"
done

echo
echo "Done. Reports under $OUT_DIR/ :"
ls -1 "$OUT_DIR"/*.report.md 2>/dev/null || true
