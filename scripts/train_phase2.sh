#!/usr/bin/env bash
# Phase 2: unfreeze LLM (full SFT) starting from phase-1 checkpoint.
# DeepSpeed ZeRO-2 shards optimizer states — required for full 8B SFT (DDP OOMs on 80GB).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NUM_GPUS="${NUM_GPUS:-4}"

# W&B (override via env: WANDB_PROJECT, WANDB_RUN_NAME, WANDB_RUN_GROUP)
WANDB_PROJECT="${WANDB_PROJECT:-cxr-vlm-qwen35}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-phase2-unfreeze-all-8b}"
export WANDB_PROJECT
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-phase2}"
export WANDB_NAME="$WANDB_RUN_NAME"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

# Reuse the already-downloaded HF model cache instead of re-downloading.
export HF_HOME="${HF_HOME:-/data/raja/vlm_training/.hf_cache}"

# Reduce CUDA allocator fragmentation (see PyTorch CUDA memory notes).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Step checkpoints (checkpoint-N) rotate via save_total_limit for resume.
# Permanent epoch copies for evaluation: output_dir/epoch_checkpoints/epoch-{1..N}.
deepspeed --num_gpus "$NUM_GPUS" src/train/train_sft.py \
  --use_liger_kernel True \
  --deepspeed configs/deepspeed/zero2.json \
  --model_id /data/raja/vlm_training/outputs/phase1_freeze_llm_8b \
  --data_path /data/raja/vlm_training/training_data/llava/train.json \
  --image_folder /data/raja/images/chest_images \
  --remove_unused_columns False \
  --freeze_llm False \
  --freeze_vision_tower False \
  --freeze_merger False \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 True \
  --output_dir /data/raja/vlm_training/outputs/phase2_unfreeze_all_8b \
  --num_train_epochs 4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
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
  --run_name "$WANDB_RUN_NAME" \
  --lazy_preprocess True \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 5 \
  --dataloader_num_workers 8 \
  --max_seq_length 8192 \
  "$@"
