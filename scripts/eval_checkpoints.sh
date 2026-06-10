#!/usr/bin/env bash
# Evaluate phase-1 checkpoints on val.json with RadGraph F1 + NLG metrics.
set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/phase1_freeze_llm}"
VAL_JSON="${VAL_JSON:-training_data/llava/val.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/root/bionicsuite1/home/ai-user/chest_classifier/disk/chest_images}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/phase1_freeze_llm}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

EXTRA=()
if [[ -n "$MAX_SAMPLES" ]]; then
  EXTRA+=(--max-samples "$MAX_SAMPLES")
fi

python -m cxr_vlm.eval.run_eval \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --val-json "$VAL_JSON" \
  --image-folder "$IMAGE_FOLDER" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA[@]}" \
  "$@"
