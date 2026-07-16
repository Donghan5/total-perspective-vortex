#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${PROJECT_ROOT}/results"

cd "$PROJECT_ROOT"
mkdir -p "$RESULT_DIR"

FULL_LOG="${RESULT_DIR}/mandatory_full.log"
SUMMARY_LOG="${RESULT_DIR}/mandatory_summary.log"

{
  echo "=== CSP TRAIN: Subject 4 / Held-out Run 14 ==="
  python mybci.py 4 14 train

  echo
  echo "=== CSP PREDICTION: Subject 4 / Held-out Run 14 ==="
  python mybci.py 4 14 predict

  echo
  echo "=== FULL MANDATORY EVALUATION ==="
  python mybci.py
} 2>&1 | tee "$FULL_LOG"

grep -E \
'^===|^Experiment:|^Pipeline:|^Subject:|^Training runs:|^Cross-Validation Scores:|^Mean CV Score:|^Model saved to:|^Accuracy:|^Average Latency|^Maximum Latency|^2s latency|^Successful evaluations:|^Errored evaluations:|^Mean accuracy:|^Median accuracy:|^Standard deviation:|^Minimum accuracy:|^Maximum accuracy:|^  .*mean accuracy' \
"$FULL_LOG" > "$SUMMARY_LOG"

echo
echo "Full log saved to: $FULL_LOG"
echo "Summary saved to: $SUMMARY_LOG"