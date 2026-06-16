#!/usr/bin/env bash
# Phase 1: freeze LLM, train vision tower + merger (projector).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

# Reuse the already-downloaded HF model cache instead of re-downloading.
export HF_HOME="${HF_HOME:-/data/raja/vlm_training/.hf_cache}"

deepspeed --num_gpus 4 src/train/train_sft.py \
  --use_liger_kernel True \
  --deepspeed configs/deepspeed/zero2.json \
  --model_id Qwen/Qwen3.5-4B \
  --data_path /data/raja/vlm_training/training_data/llava/train.json \
  --image_folder /data/raja/chest_images \
  --remove_unused_columns False \
  --freeze_llm True \
  --freeze_vision_tower False \
  --freeze_merger False \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 True \
  --output_dir /data/raja/vlm_training/outputs/phase1_freeze_llm_bkp \
  --num_train_epochs 2 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 4 \
  --image_min_pixels 65536 \
  --image_max_pixels 4194304 \
  --learning_rate 1e-5 \
  --merger_lr 1e-5 \
  --vision_lr 2e-6 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --gradient_checkpointing True \
  --report_to wandb \
  --lazy_preprocess True \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 100 \
  --dataloader_num_workers 8 \
  --max_seq_length 8192 \
  "$@"
