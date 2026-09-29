#!/usr/bin/env bash

set -euo pipefail

BASE_URL="https://physionet.org/files/eegmmidb/1.0.0"
DATA_ROOT="physionet.org/files/eegmmidb/1.0.0"
BAR_WIDTH=30

FILES=()
SIZES=()
TOTAL_BYTES=0

cd "$(dirname "$0")/.."


usage() {
    echo "Usage:"
    echo "  $0"
    echo "  $0 --sample SUBJECT_ID RUN_ID"
}


human_size() {
    awk -v bytes="$1" '
        BEGIN {
            if (bytes >= 1073741824)
                printf "%.2f GB", bytes / 1073741824
            else if (bytes >= 1048576)
                printf "%.2f MB", bytes / 1048576
            else if (bytes >= 1024)
                printf "%.2f KB", bytes / 1024
            else
                printf "%d B", bytes
        }
    '
}


show_progress() {
    local current=$1
    local total=$2
    local elapsed=$3

    local percent=0

    if (( total > 0 )); then
        percent=$((current * 100 / total))
    fi

    (( percent > 100 )) && percent=100

    local filled=$((percent * BAR_WIDTH / 100))
    local empty=$((BAR_WIDTH - filled))

    local full
    local rest

    printf -v full "%*s" "$filled" ""
    printf -v rest "%*s" "$empty" ""

    full=${full// /#}
    rest=${rest// /-}

    printf "\rDownloading: [%s%s] %3d%% | %s / %s | Elapsed: %02d:%02d" \
        "$full" \
        "$rest" \
        "$percent" \
        "$(human_size "$current")" \
        "$(human_size "$total")" \
        "$((elapsed / 60))" \
        "$((elapsed % 60))"
}


remote_size() {
    local file=$1
    local size

    size=$(
        curl -fsSIL "$BASE_URL/$file" |
        awk '
            tolower($1) == "content-length:" {
                gsub("\r", "", $2)
                size = $2
            }

            END {
                if (size != "")
                    print size
            }
        '
    )

    if [[ ! "$size" =~ ^[0-9]+$ ]]; then
        echo "Failed to get remote size: $file" >&2
        exit 1
    fi

    echo "$size"
}


load_sample_files() {
    local subject_id=$1
    local run_id=$2
    local output

    output=$(
        python3 - "$subject_id" "$run_id" <<'PY'
import sys

from src.experiments import resolve_single_run_task

subject_id = int(sys.argv[1])
run_id = int(sys.argv[2])

if not 1 <= subject_id <= 109:
    raise SystemExit("SUBJECT_ID must be between 1 and 109")

_, runs = resolve_single_run_task(run_id)

subject = f"S{subject_id:03d}"

for run in runs:
    print(f"{subject}/{subject}R{run:02d}.edf")
PY
    )

    mapfile -t FILES <<< "$output"
}


load_full_files() {
    local records

    records=$(curl -fsSL "$BASE_URL/RECORDS")

    mapfile -t FILES <<< "$records"
}


prepare_download() {
    SIZES=()
    TOTAL_BYTES=0

    local count=${#FILES[@]}
    local current=0

    echo "Calculating download size..."

    for file in "${FILES[@]}"; do
        local size

        size=$(remote_size "$file")

        SIZES+=("$size")
        TOTAL_BYTES=$((TOTAL_BYTES + size))

        current=$((current + 1))

        printf "\rChecking metadata: %d/%d" \
            "$current" \
            "$count"
    done

    printf "\n"
    echo "Files: $count"
    echo "Total size: $(human_size "$TOTAL_BYTES")"
}


download_files() {
    local downloaded=0
    local start=$SECONDS

    for i in "${!FILES[@]}"; do
        local file="${FILES[$i]}"
        local expected="${SIZES[$i]}"

        local url="$BASE_URL/$file"
        local output="$DATA_ROOT/$file"
        local part="$output.part"

        mkdir -p "$(dirname "$output")"

        # Skip files that are already complete.
        if [[ -f "$output" ]]; then
            local existing
            existing=$(stat -c '%s' "$output")

            if (( existing == expected )); then
                downloaded=$((downloaded + expected))

                show_progress \
                    "$downloaded" \
                    "$TOTAL_BYTES" \
                    "$((SECONDS - start))"

                continue
            fi
        fi

        rm -f "$output" "$part"

        curl -fsSL "$url" -o "$part" &
        local pid=$!

        while kill -0 "$pid" 2>/dev/null; do
            local partial=0

            if [[ -f "$part" ]]; then
                partial=$(stat -c '%s' "$part")
            fi

            show_progress \
                "$((downloaded + partial))" \
                "$TOTAL_BYTES" \
                "$((SECONDS - start))"

            sleep 1
        done

        if ! wait "$pid"; then
            printf "\n"
            echo "Download failed: $file" >&2
            rm -f "$part"
            exit 1
        fi

        local actual
        actual=$(stat -c '%s' "$part")

        if (( actual != expected )); then
            printf "\n"
            echo "Size mismatch: $file" >&2
            echo "Expected: $expected bytes" >&2
            echo "Received: $actual bytes" >&2

            rm -f "$part"
            exit 1
        fi

        mv "$part" "$output"

        downloaded=$((downloaded + expected))

        show_progress \
            "$downloaded" \
            "$TOTAL_BYTES" \
            "$((SECONDS - start))"
    done

    show_progress \
        "$TOTAL_BYTES" \
        "$TOTAL_BYTES" \
        "$((SECONDS - start))"

    printf "\nDownload complete.\n"
}


download_sample() {
    local subject_id=$1
    local run_id=$2

    load_sample_files "$subject_id" "$run_id"

    echo "Sample files:"
    printf "  %s\n" "${FILES[@]}"

    prepare_download
    download_files
}


download_full() {
    load_full_files

    prepare_download
    download_files
}


case "${1:-}" in
    "")
        download_full
        ;;

    --sample)
        if (( $# != 3 )); then
            usage
            exit 1
        fi

        download_sample "$2" "$3"
        ;;

    -h|--help)
        usage
        ;;

    *)
        echo "Unknown option: $1" >&2
        usage
        exit 1
        ;;
esac