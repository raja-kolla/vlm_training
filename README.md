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
│   ├── setup_node.sh         # one-time env setup on a dedicated GPU node
│   ├── train_local.py        # main launcher: prepare_data / train_phase1 / train_phase2
│   ├── prepare_data.sh
│   ├── train_phase1.sh       # freeze LLM (bare script, no restart loop)
│   ├── train_phase2.sh       # full SFT from phase-1 ckpt
│   └── train_phase2_lora.sh  # optional LoRA variant
├── configs/
│   ├── data.yaml
│   └── deepspeed/            # zero1.json, zero2.json, zero2_offload.json, zero3.json, zero3_offload.json
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

## Training on a dedicated GPU node (4× H100) — recommended

If you have your own machine with GPUs (no Modal/serverless), use `scripts/train_local.py`. It is the local equivalent of the old Modal app: same training recipe and hyperparameters, auto-resume from checkpoints, and automatic restart on crash — but with no 24h timeout and no volume syncing.

### One-time setup on the node

```bash
git clone <repo-url> vlm_training && cd vlm_training
bash scripts/setup_node.sh        # creates .venv, installs torch cu128 + requirements
source .venv/bin/activate
wandb login                       # optional, for W&B metrics
```

### Put the data in place

```bash
# Images can live anywhere — just point IMAGE_FOLDER at them
export IMAGE_FOLDER=/data/chest_images

# CSV goes here (or pass --csv-path)
mkdir -p training_data/csv
cp /path/to/filtered_columns_cxr_0.3M.csv training_data/csv/
```

If checkpoints/data should live on a different (bigger) disk, set `CXR_DATA_ROOT=/big/disk/cxr` — outputs, logs, HF cache, and LLaVA JSON all move under it.

### Run

```bash
python scripts/train_local.py prepare_data      # CSV -> training_data/llava/{train,val}.json
python scripts/train_local.py download_model    # optional: pre-cache Qwen3.5-4B

# Phase 1 — run inside tmux/screen so it survives SSH disconnects
tmux new -s phase1
python scripts/train_local.py train_phase1

# Phase 2 — starts from outputs/phase1_freeze_llm
python scripts/train_local.py train_phase2
```

### Defaults for 4× H100 80GB

| Phase | Per-GPU batch | Grad accum | Global batch | DeepSpeed |
|-------|---------------|------------|--------------|-----------|
| Phase 1 (freeze LLM) | 4 | 8 | **128** | ZeRO-2 |
| Phase 2 (full SFT) | 2 | 4 | **32** | ZeRO-3 |

If you hit OOM, halve `--batch-size` and double `--grad-accum` (keeps the global batch constant):

```bash
python scripts/train_local.py train_phase1 --batch-size 2 --grad-accum 16
```

As a last resort for phase 2, switch to CPU offload by editing `PHASE_PROFILES` in `scripts/train_local.py` to use `configs/deepspeed/zero3_offload.json`.

### Checkpoints, resume & crash recovery

- **Auto-resume:** if `checkpoint-*` exists in the output dir, training continues from the latest one. Just re-run the same command after any interruption.
- **Auto-restart:** if the training process crashes, the launcher restarts it (up to 100 times, exponential backoff) and resumes from the latest checkpoint.
- **Fresh start:** `--fresh-start` archives the old output dir to `*.archived.<timestamp>/` (never deletes).
- **Skip resume without touching files:** `CXR_FORCE_FRESH=1 python scripts/train_local.py train_phase1`.
- Checkpoints save every 100 steps, keeping the last 3 (`save_total_limit=3`).

### Useful overrides

| Env var | Default | Description |
|---------|---------|-------------|
| `CXR_NUM_GPUS` | `4` | GPUs to use (`--num-gpus` also works) |
| `IMAGE_FOLDER` | `training_data/images` | JPEG root |
| `CXR_DATA_ROOT` | repo dir | Root for data/outputs/logs/caches |
| `CXR_OUTPUT_ROOT` | `$CXR_DATA_ROOT/outputs` | Checkpoint root |
| `CXR_MODEL_ID` | `Qwen/Qwen3.5-4B` | Base model |
| `CXR_REPORT_TO` | `wandb` | Set `tensorboard` or `none` to skip W&B |
| `WANDB_PROJECT` / `WANDB_ENTITY` | `cxr-vlm-qwen35` / — | W&B routing |

Dry-run to inspect the exact training command without launching:

```bash
python scripts/train_local.py train_phase1 --dry-run
```

## Step 1 — Prepare LLaVA dataset

```bash
export IMAGE_FOLDER=/root/bionicsuite1/home/ai-user/chest_classifier/disk/chest_images

# Full dataset
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

## Evaluate checkpoints (RadGraph F1 + NLG metrics)

Install eval dependencies once:

```bash
pip install -r requirements-eval.txt
```

Evaluate all checkpoints under `checkpoints/phase1_freeze_llm/` on `training_data/llava/val.json`:

```bash
bash scripts/eval_checkpoints.sh
```

Quick smoke test (first 50 samples):

```bash
MAX_SAMPLES=50 bash scripts/eval_checkpoints.sh
```

Outputs per checkpoint under `eval_results/phase1_freeze_llm/`:

| File | Contents |
|------|----------|
| `checkpoint-100/metrics.json` | Aggregate scores |
| `checkpoint-100/predictions.jsonl` | Per-sample reference vs prediction |
| `summary.csv` | Compare all checkpoints side-by-side |

**Metrics computed:**

| Metric | What it measures |
|--------|------------------|
| **RadGraph F1** | Clinical entity/relation correctness (primary) |
| **RadGraph entity / relation F1** | Sub-scores from RadGraph |
| **BLEU-1/2/4** | N-gram overlap with reference |
| **ROUGE-L** | Longest common subsequence (full report) |
| **ROUGE-L (observations / conclusion)** | Section-level overlap |
| **METEOR** | Synonym-aware overlap |
| **BERTScore F1** | Optional (`--bertscore`); semantic similarity |
| **tag_format_rate** | Fraction of outputs with valid `<observations>` / `<conclusion>` tags |

**Other metrics used in CXR literature** (not wired yet): CheXbert label F1, RadCliQ composite, GREEN, temporal/critical finding accuracy. RadGraph F1 is the standard for factual clinical correctness.

Single checkpoint:

```bash
python -m cxr_vlm.eval.run_eval \
  --checkpoint checkpoints/phase1_freeze_llm/checkpoint-500 \
  --val-json training_data/llava/val.json \
  --image-folder "$IMAGE_FOLDER" \
  --output-dir eval_results/checkpoint-500
```

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
2. **Monitor:** Training logs go to **W&B** (Modal) or TensorBoard locally. On Modal, open your `cxr-vlm-qwen35` project on wandb.ai.
3. **Resume:** Training auto-resumes if checkpoints exist in `OUTPUT_DIR`.
4. **Eval:** Pass `--eval_path training_data/llava/val.json --eval_strategy steps --eval_steps 500` to any training script for validation loss.

## Training on Modal (legacy — superseded by the dedicated-node workflow above)

> Kept for reference. New training runs should use `scripts/train_local.py` on the dedicated GPU node.

Modal runs training in a **GPU container** with **pip/uv** — you do not need conda on Modal.

### Volume layout (`vlm_training`)

Everything is written under `/vol` on your volume:

```
/vol/
├── images/                              # chest X-ray JPEGs (input)
├── csv/
│   └── filtered_columns_cxr_0.3M.csv    # source CSV (input)
├── llava/                               # prepare_data output
│   ├── train.json                       # 80%
│   ├── val.json                         # 20%
│   └── meta.json
├── hf_cache/                            # HuggingFace + datasets cache
├── wandb/                               # W&B local run files
├── logs/
│   ├── phase1_freeze_llm/               # trainer logs
│   └── phase2_full_sft/
└── outputs/
    ├── phase1_freeze_llm/               # checkpoints
    └── phase2_full_sft/
```

Verify contents:

```bash
modal volume ls vlm_training
modal volume ls vlm_training images
modal volume ls vlm_training csv
```

### One-time setup (local machine)

```bash
pip install modal
modal setup

# W&B — store API key + personal entity (not org) to avoid permission errors
modal secret create wandb WANDB_API_KEY=<your-key> WANDB_ENTITY=rajaphanindra

# Optional overrides
export WANDB_PROJECT=cxr-vlm-qwen35
```

### Workflow

```bash
# 1) Build LLaVA JSON from CSV + images on the volume
modal run modal/app.py --action prepare_data

# 2) Optional: cache Qwen3.5-4B weights on the volume
modal run modal/app.py --action download_model

# 3) Phase 1 — logs go to W&B + /vol/logs + checkpoints to /vol/outputs
modal run modal/app.py --action train_phase1

# 4) Phase 2
modal run modal/app.py --action train_phase2
```

Training metrics (loss, eval loss, lr, etc.) are logged to **Weights & Biases** via `--report_to wandb`. Check your project at [wandb.ai](https://wandb.ai) under project `cxr-vlm-qwen35` (override with `WANDB_PROJECT`).

If you already generated `train.json` / `val.json` locally, upload them:

```bash
modal volume put vlm_training training_data/llava/train.json /llava/train.json
modal volume put vlm_training training_data/llava/val.json /llava/val.json
```

### Why this approach

| Approach | Verdict |
|----------|---------|
| **Modal Image + pip** | Best — reproducible, no conda |
| **Conda on Modal** | Unnecessary — Modal containers are ephemeral; use `Image.pip_install` |
| **Multi-GPU DeepSpeed** | Only if you request `gpu="A100:2"` etc.; start with 1× A100-80GB |
| **Volume for data/ckpt** | Required — container filesystem is ephemeral |
| **flash-attn** | Skipped — Qwen3.5 uses `--disable_flash_attn2 True` (SDPA) |

### GPU sizing & batch size

Default: **8× H200** (`gpu="H200:8"`) with DeepSpeed ZeRO-2.

| Phase | Per-GPU batch | Grad accum | GPUs | Global batch |
|-------|---------------|------------|------|--------------|
| Phase 1 | 16 | 1 | 8 | **128** |
| Phase 2 | 4 | 1 | 8 | **32** |

**Run (default 8× H200):**

```bash
modal run modal/app.py --action train_phase1
modal run modal/app.py --action train_phase2
```

**Single GPU** (override):

```bash
CXR_NUM_GPUS=1 CXR_PHASE1_GPU=H200 modal run modal/app.py --action train_phase1
```

**Override batch size:**

```bash
modal run modal/app.py --action train_phase1 --batch-size 2 --grad-accum 2
```

**Tuning guide:** Keep `batch_size × grad_accum ≈ 16`. If you hit OOM, halve `batch_size` and double `grad_accum`. Watch W&B for stable loss before increasing batch further.

### Checkpoints & resume (Modal)

| Behavior | Detail |
|----------|--------|
| **Auto-resume** | Default: if `checkpoint-*` exists under `/vol/outputs/phase1_freeze_llm/`, training continues from the latest checkpoint. |
| **Auto-restart on crash** | Training subprocess is restarted up to **100 times** (60s backoff, max 5 min); each restart resumes from the latest checkpoint. Modal also retries the whole job up to **10×** on container failure/24h timeout. |
| **Volume sync** | Checkpoints are `volume.commit()`'d every **5 minutes** and on success/failure so they survive crashes and Modal retries. |
| **Save frequency** | Every **100 steps** (`save_steps=100`); keeps last **3** checkpoints (`save_total_limit=3`). |
| **Fresh start** | `--fresh-start` **archives** the old output dir (does not delete it) to `phase1_freeze_llm.archived.<timestamp>/`. |
| **Force no resume** | `CXR_FORCE_FRESH=1` on the training job skips resume but leaves existing files on disk. |

**Do not** use `--fresh-start` to recover from a failed run — just re-run without it:

```bash
modal run --detach modal/app.py --action train_phase1
```

DeepSpeed checkpoints (8× GPU + ZeRO) resume fully. Older single-GPU HF checkpoints load **weights only** (optimizer/step reset) but are **never renamed or deleted**.

### Training timeout (Modal limit)

Modal allows **at most 24 hours per invocation** — you cannot set a one-month timeout on a single run.

In `modal/app.py`:

```python
TRAIN_TIMEOUT_SECONDS = 24 * 60 * 60   # Modal maximum (86400s)
TRAIN_RETRIES = modal.Retries(initial_delay=0.0, max_retries=10)  # auto-restart on timeout
```

This gives up to **~11 days** (11 × 24h) per `modal run` command. Training **auto-resumes from checkpoints** in `/vol/outputs/` after each timeout/retry.

For a full month of wall time, re-run the same command when the job finishes or exhausts retries:

```bash
modal run modal/app.py --action train_phase1   # run again; resumes from latest checkpoint
```

Check progress:

```bash
modal volume ls vlm_training outputs/phase1_freeze_llm
```

### W&B on Modal

Create the secret once (name must be `wandb`, or set `CXR_WANDB_SECRET`):

```bash
modal secret create wandb WANDB_API_KEY=your_key_here WANDB_ENTITY=rajaphanindra
```

Use your **personal W&B username** as `WANDB_ENTITY`, not the org (`ai_5c`), if you see `user does not have models write access for this org`. Metrics-only logging is enabled (`WANDB_LOG_MODEL=false`); checkpoints still save to `/vol/outputs/`.

Runs are grouped as `phase1` / `phase2`. Custom run name:

```bash
modal run modal/app.py --action train_phase1 --run-name cxr-p1-run1
```

## License

Training code adapted from [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) (Apache 2.0). Qwen3.5 model license: Apache 2.0.
