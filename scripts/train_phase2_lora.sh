#!/usr/bin/env bash
# Phase 2 (LoRA variant): freeze base LLM weights, train LoRA adapters on top of phase-1 checkpoint.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PHASE1_DIR="${PHASE1_DIR:-outputs/phase1_freeze_llm}"
MODEL_NAME="${MODEL_NAME:-$PHASE1_DIR}"
DATA_PATH="${DATA_PATH:-training_data/llava/train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/root/bionicsuite1/home/ai-user/chest_classifier/disk/chest_images}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2_lora}"

NUM_DEVICES="${NUM_DEVICES:-1}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-$((256 * 16 * 16))}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-$((128 * 16 * 128 * 16))}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/zero2.json}"
if [[ "$NUM_DEVICES" -eq 1 ]]; then
  LAUNCHER=(python)
else
  LAUNCHER=(deepspeed --num_gpus "$NUM_DEVICES")
fi

"${LAUNCHER[@]}" src/train/train_sft.py \
  --use_liger_kernel True \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --model_id "$MODEL_NAME" \
  --data_path "$DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --remove_unused_columns False \
  --lora_enable True \
  --freeze_llm True \
  --freeze_vision_tower True \
  --freeze_merger True \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 True \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$BATCH_PER_DEVICE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --image_min_pixels "$IMAGE_MIN_PIXELS" \
  --image_max_pixels "$IMAGE_MAX_PIXELS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --tf32 True \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --lazy_preprocess True \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --dataloader_num_workers 4 \
  --max_seq_length 8192 \
  "$@"
