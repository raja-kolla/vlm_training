#!/usr/bin/env bash
# Phase 2: unfreeze LLM (full SFT) starting from phase-1 checkpoint.
# Uses torchrun + DDP instead of DeepSpeed ZeRO (often better GPU utilization).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NUM_GPUS="${NUM_GPUS:-4}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

# Reuse the already-downloaded HF model cache instead of re-downloading.
export HF_HOME="${HF_HOME:-/data/raja/vlm_training/.hf_cache}"

torchrun --nproc_per_node "$NUM_GPUS" src/train/train_sft.py \
  --use_liger_kernel True \
  --model_id /data/raja/vlm_training/outputs/phase1_full_sft \
  --data_path /data/raja/vlm_training/training_data/llava/train.json \
  --image_folder /data/raja/chest_images \
  --remove_unused_columns False \
  --freeze_llm False \
  --freeze_vision_tower False \
  --freeze_merger False \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 True \
  --output_dir /data/raja/vlm_training/outputs/phase2_full_sft \
  --num_train_epochs 4 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --image_min_pixels 65536 \
  --image_max_pixels 4194304 \
  --learning_rate 5e-6 \
  --merger_lr 5e-6 \
  --vision_lr 1e-6 \
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
  --save_total_limit 5 \
  --dataloader_num_workers 16 \
  --max_seq_length 8192 \
  "$@"
