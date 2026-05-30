#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CSV_PATH="${CSV_PATH:-training_data/csv/filtered_columns_cxr_0.3M.csv}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/root/bionicsuite1/home/ai-user/chest_classifier/disk/chest_images}"
OUTPUT_DIR="${OUTPUT_DIR:-training_data/llava}"
VAL_RATIO="${VAL_RATIO:-0.02}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

EXTRA_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
  EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

python -m cxr_vlm.data.prepare_llava \
  --csv-path "$CSV_PATH" \
  --image-folder "$IMAGE_FOLDER" \
  --output-dir "$OUTPUT_DIR" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

echo "Wrote ${OUTPUT_DIR}/train.json and ${OUTPUT_DIR}/val.json"
