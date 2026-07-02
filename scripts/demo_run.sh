#!/usr/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "${PROJECT_ROOT:?}"

SUBJECT_ID="${1:-4}"
TEST_RUN="${2:-14}"
PIPELINE_MODE="${3:-all}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "Virtual environment not found. Please set up the virtual environment first."
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"

echo "=============================================================="
echo "Total Perspective Vortex - Demo Run"
echo "=============================================================="
echo "Subject ID: S$(printf "%03d" "$SUBJECT_ID")"
echo "Test Run: S$(printf "%03d" "$TEST_RUN")"
echo "Pipeline Mode: ${PIPELINE_MODE}"
echo "=============================================================="
echo ""

run_pipeline() {
    local pipeline_name="$1"

    local train_log="${LOG_DIR}/S$(printf "%03d" "$SUBJECT_ID")_S$(printf "%03d" "$TEST_RUN")_${pipeline_name}_train.log"
    local predict_log="${LOG_DIR}/S$(printf "%03d" "$SUBJECT_ID")_S$(printf "%03d" "$TEST_RUN")_${pipeline_name}_predict.log"

    echo ""
    echo "--------------------------------------------------------------"
    echo "Running pipeline: ${pipeline_name}"
    echo "--------------------------------------------------------------"

    echo "[1/2] Training ${pipeline_name} pipeline..."
    "${PYTHON_BIN}" mybci.py "${SUBJECT_ID}" "${TEST_RUN}" train --pipeline "${pipeline_name}" > "${train_log}" 2>&1

    echo "[2/2] Predicting held-out run with ${pipeline_name} pipeline..."
    "${PYTHON_BIN}" mybci.py "${SUBJECT_ID}" "${TEST_RUN}" predict --pipeline "${pipeline_name}" > "${predict_log}" 2>&1

    echo ""
    echo "Summary for ${pipeline_name} pipeline:"
    echo "--------------------------------------------------------------"

    grep -E "Experiment:|Pipeline:|Subject:|Training runs:|Mean CV Score:|Model saved to:" "${train_log}" || true
    grep -E "Accuracy:|Average Latency|Maximum Latency|2s latency constraint" "${predict_log}" || true

    echo ""
    echo "Full logs:"
    echo "  Train  : ${train_log}"
    echo "  Predict: ${predict_log}"
}

case "${PIPELINE_MODE}" in
    all)
        run_pipeline "csp"
        run_pipeline "wavelet"
        ;;
    csp)
        run_pipeline "csp"
        ;;
    wavelet)
        run_pipeline "wavelet"
        ;;
    *)
        echo "Unknown mode: ${PIPELINE_MODE}"
        echo "Usage:"
        echo "  ./scripts/bonus_demo.sh [subject_id] [test_run] [all|csp|wavelet]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Demo finished."
echo "============================================================"