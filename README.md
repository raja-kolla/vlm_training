# CXR Report Generation — Qwen3.5-4B Fine-Tuning

Fine-tune [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) for chest X-ray report generation using a two-phase training recipe:

| Phase | What trains | What is frozen |
|-------|-------------|----------------|
| **Phase 1** | Vision encoder + merger (projector) | LLM |
| **Phase 2** | Full model (or LoRA on LLM) | — |

Training code is adapted from [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) (Apache 2.0).

## Project layout

```
vlm_training/
├── cxr_vlm/                  # CXR-specific data prep & inference
│   └── data/
│       ├── prompts.py        # Prompt / response templates
│       └── prepare_llava.py    # CSV → LLaVA JSON converter
├── src/                      # Qwen-VL SFT trainer (vision + LLM)
├── scripts/
│   ├── prepare_data.sh
│   ├── train_phase1.sh       # freeze LLM
│   ├── train_phase2.sh       # full SFT from phase-1 ckpt
│   └── train_phase2_lora.sh  # optional LoRA variant
├── configs/
│   ├── data.yaml
│   └── deepspeed/
└── training_data/            # your CSV + generated LLaVA JSON
```

## Data format

**Input CSV** (`training_data/csv/filtered_columns_cxr_0.3M.csv`):

| Column | Use |
|--------|-----|
| `file_path` | Image filename (resolved under `IMAGE_FOLDER`) |
| `observations` | Target findings section |
| `conclusion` | Target impression section |
| `history` | Clinical history (optional; ~42% empty) |

**LLaVA JSON** (one sample):

```json
{
  "id": "cxr_0",
  "image": "2025_08_01_CF4D8A9B_00F62951_47994995.jpeg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nYou are an expert radiologist...\n\nClinical history: NO TB IN HISTORY\n\nGenerate a structured chest X-ray report..."
    },
    {
      "from": "gpt",
      "value": "<observations>\nVisualised lung fields appear normal...\n</observations>\n<conclusion>\nRadiograph chest does not reveal any significant abnormality.\n</conclusion>"
    }
  ]
}
```

The `<observations>...</observations>` and `<conclusion>...</conclusion>` tags make outputs easy to parse at inference time.

## Setup

```bash
cd vlm_training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation   # optional; phase scripts use SDPA for Qwen3.5
```

> **Note:** Qwen3.5 training scripts set `--disable_flash_attn2 True` because Flash Attention 2 can raise CUDA errors on Qwen3.5; SDPA is the stable path.

## Step 1 — Prepare LLaVA dataset

```bash
export IMAGE_FOLDER=/root/bionicsuite1/home/ai-user/chest_classifier/disk/chest_images

# Full dataset (~318k rows after filtering)
bash scripts/prepare_data.sh

# Quick dry run (first 1000 rows)
MAX_SAMPLES=1000 bash scripts/prepare_data.sh
```

Outputs:

- `training_data/llava/train.json` — training split (80%)
- `training_data/llava/val.json` — validation split (20%)
- `training_data/llava/meta.json` — stats

Or run directly:

```bash
python -m cxr_vlm.data.prepare_llava \
  --csv-path training_data/csv/filtered_columns_cxr_0.3M.csv \
  --image-folder "$IMAGE_FOLDER" \
  --output-dir training_data/llava
```

## Step 2 — Phase 1 training (freeze LLM)

Trains **vision tower + merger** while keeping the language model frozen.

```bash
export IMAGE_FOLDER=/path/to/chest_images
export NUM_DEVICES=1          # or 2, 4, 8 for multi-GPU deepspeed
export BATCH_PER_DEVICE=1
export GRAD_ACCUM_STEPS=8

bash scripts/train_phase1.sh
```

Key settings:

- Model: `Qwen/Qwen3.5-4B`
- `--freeze_llm True`
- `--freeze_vision_tower False`
- `--freeze_merger False`
- Max image side **2048** via `image_max_pixels = 128 × 16 × 128 × 16` (Qwen3.5 patch size = 16)

Checkpoint saved to `outputs/phase1_freeze_llm/`.

## Step 3 — Phase 2 training

### Option A — Full SFT (unfreeze LLM)

```bash
export PHASE1_DIR=outputs/phase1_freeze_llm
bash scripts/train_phase2.sh
```

Uses DeepSpeed ZeRO-3 offload by default (lower VRAM). Adjust `GRAD_ACCUM_STEPS` and `BATCH_PER_DEVICE` for your GPU.

### Option B — LoRA on LLM (lower VRAM)

```bash
export PHASE1_DIR=outputs/phase1_freeze_llm
bash scripts/train_phase2_lora.sh
```

## Image resolution

Qwen3.5 uses a **16×16 patch size**. To cap the longest side at 2048 px:

```bash
IMAGE_MIN_PIXELS=$((256 * 16 * 16))    # 65536
IMAGE_MAX_PIXELS=$((128 * 16 * 128 * 16))  # 4194304 == 2048²
```

These are the defaults in the training scripts. Images are resized by `qwen-vl-utils` while preserving aspect ratio.

## Inference (after training)

```bash
export PYTHONPATH=src:$PYTHONPATH
python -m cxr_vlm.inference \
  --model-path outputs/phase2_full_sft \
  --image-path /path/to/chest_images/example.jpeg \
  --history "Chest PA"
```

Parsed output includes `observations`, `conclusion`, and `raw`.

## Environment variables reference

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_FOLDER` | chest_images path | Root folder for JPEG files |
| `DATA_PATH` | `training_data/llava/train.json` | LLaVA training JSON |
| `MODEL_NAME` | `Qwen/Qwen3.5-4B` | Base model (phase 1) |
| `PHASE1_DIR` | `outputs/phase1_freeze_llm` | Checkpoint for phase 2 |
| `OUTPUT_DIR` | phase-specific | Where checkpoints are saved |
| `NUM_DEVICES` | `1` | GPUs for DeepSpeed |
| `BATCH_PER_DEVICE` | `1` | Per-GPU batch size |
| `GRAD_ACCUM_STEPS` | `8` | Gradient accumulation |

## Tips

1. **VRAM:** Phase 1 with 2048 max side is memory-heavy. Start with `BATCH_PER_DEVICE=1`, `GRAD_ACCUM_STEPS=8`, and gradient checkpointing (enabled by default).
2. **Monitor:** TensorBoard logs go to `./tf-logs` (or set `--report_to wandb`).
3. **Resume:** Training auto-resumes if checkpoints exist in `OUTPUT_DIR`.
4. **Eval:** Pass `--eval_path training_data/llava/val.json --eval_strategy steps --eval_steps 500` to any training script for validation loss.

## License

Training code adapted from [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) (Apache 2.0). Qwen3.5 model license: Apache 2.0.
