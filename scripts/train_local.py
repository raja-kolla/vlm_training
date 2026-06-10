#!/usr/bin/env python3
"""
Local training launcher for a dedicated GPU node (default: 4x H100).

This replaces the Modal serverless entrypoints (modal/app.py) with a direct
launcher for a machine you own. Same training recipe, hyperparameters, and
auto-resume behavior — no Modal volume, no 24h timeout, no container retries.

Directory layout (override with env vars, see below):

  <repo>/training_data/csv/filtered_columns_cxr_0.3M.csv   source CSV
  <repo>/training_data/llava/                              train.json / val.json / meta.json
  $IMAGE_FOLDER                                            chest X-ray JPEGs
  <repo>/outputs/phase1_freeze_llm/                        phase-1 checkpoints
  <repo>/outputs/phase2_full_sft/                          phase-2 checkpoints
  <repo>/logs/                                             trainer logs per phase
  <repo>/.hf_cache/                                        HuggingFace cache

Usage:
  python scripts/train_local.py prepare_data
  python scripts/train_local.py download_model
  python scripts/train_local.py train_phase1
  python scripts/train_local.py train_phase2
  python scripts/train_local.py train_phase1 --batch-size 2 --grad-accum 16
  python scripts/train_local.py train_phase1 --fresh-start   # archives old outputs
  python scripts/train_local.py train_phase1 --dry-run       # print command only

Env overrides:
  CXR_NUM_GPUS=4            number of GPUs to use (deepspeed --num_gpus)
  IMAGE_FOLDER=/path        image root (required if images are not in training_data/images)
  CXR_DATA_ROOT=/path       move all data/outputs off the repo (e.g. to a big disk)
  CXR_OUTPUT_ROOT=/path     checkpoints root (default: $CXR_DATA_ROOT/outputs)
  HF_HOME=/path             HuggingFace cache (default: $CXR_DATA_ROOT/.hf_cache)
  WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT   W&B logging (or `wandb login` once)
  CXR_FORCE_FRESH=1         skip checkpoint resume without touching files on disk
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Paths -------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("CXR_DATA_ROOT", str(REPO)))

TRAINING_DATA = DATA_ROOT / "training_data"
DEFAULT_CSV = TRAINING_DATA / "csv" / "filtered_columns_cxr_0.3M.csv"
LLAVA_DIR = TRAINING_DATA / "llava"
IMAGE_FOLDER = Path(os.environ.get("IMAGE_FOLDER", str(TRAINING_DATA / "images")))
OUTPUT_ROOT = Path(os.environ.get("CXR_OUTPUT_ROOT", str(DATA_ROOT / "outputs")))
LOGS_ROOT = DATA_ROOT / "logs"
HF_CACHE = Path(os.environ.get("HF_HOME", str(DATA_ROOT / ".hf_cache")))
WANDB_DIR = DATA_ROOT / "wandb"

MODEL_ID = os.environ.get("CXR_MODEL_ID", "Qwen/Qwen3.5-4B")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "cxr-vlm-qwen35")

IMAGE_MIN_PIXELS = 256 * 16 * 16
IMAGE_MAX_PIXELS = 128 * 16 * 128 * 16  # max side 2048

NUM_GPUS = int(os.environ.get("CXR_NUM_GPUS", "4"))

# In-process restarts after a training crash (resume from latest checkpoint each time)
MAX_RESTARTS = int(os.environ.get("CXR_MAX_TRAIN_RESTARTS", "100"))
RESTART_DELAY_S = int(os.environ.get("CXR_TRAIN_RESTART_DELAY_S", "60"))
RESTART_DELAY_MAX_S = int(os.environ.get("CXR_TRAIN_RESTART_DELAY_MAX_S", "300"))

# Per-phase profiles tuned for 4x H100 80GB (Qwen3.5-4B, 2048px, seq 8192).
# Effective global batch = batch_size x grad_accum x NUM_GPUS.
# OOM? Halve batch_size and double grad_accum (keeps global batch constant).
PHASE_PROFILES: dict[str, dict] = {
    "phase1": {
        "batch_size": 4,
        "grad_accum": 8,            # global batch = 4 x 8 x 4 = 128
        "eval_batch_size": 2,
        "deepspeed": "configs/deepspeed/zero2.json",
        "dataloader_workers": 8,
        "learning_rate": "1e-5",
        "vision_lr": "2e-6",
        "merger_lr": "1e-5",
    },
    "phase2": {
        "batch_size": 2,
        "grad_accum": 4,            # global batch = 2 x 4 x 4 = 32
        "eval_batch_size": 1,
        "deepspeed": "configs/deepspeed/zero3.json",
        "dataloader_workers": 8,
        "learning_rate": "5e-6",
        "vision_lr": "1e-6",
        "merger_lr": "5e-6",
    },
}


# --- Helpers -----------------------------------------------------------------

def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO / 'src'}:{env['PYTHONPATH']}".rstrip(":")
    env.setdefault("HF_HOME", str(HF_CACHE))
    env.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))
    env.setdefault("WANDB_DIR", str(WANDB_DIR))
    env.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    env.setdefault("WANDB_LOG_MODEL", "false")
    env.setdefault("WANDB_WATCH", "false")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # DeepSpeed needs CUDA_HOME for JIT ops; only set if the toolkit is there.
    if "CUDA_HOME" not in env and Path("/usr/local/cuda").exists():
        env["CUDA_HOME"] = "/usr/local/cuda"
    return env


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env or _base_env())


def _list_checkpoints(output_dir: Path) -> list[Path]:
    found = []
    for path in output_dir.glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        if step.isdigit():
            found.append(path)
    return sorted(found, key=lambda p: int(p.name.split("-", 1)[1]))


def _archive_output(output_dir: Path) -> None:
    """Move prior run aside instead of deleting checkpoints (safe fresh start)."""
    if not output_dir.exists() or not any(output_dir.iterdir()):
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = output_dir.parent / f"{output_dir.name}.archived.{stamp}"
    print(f"Archiving {output_dir} -> {backup} for fresh start", flush=True)
    shutil.move(str(output_dir), str(backup))
    output_dir.mkdir(parents=True, exist_ok=True)


def _run_training_resilient(cmd: list[str], env: dict[str, str]) -> None:
    """Run training; on crash, restart (train_sft.py auto-resumes from latest checkpoint)."""
    delay_s = RESTART_DELAY_S
    for attempt in range(1, MAX_RESTARTS + 2):
        if attempt > 1:
            print(
                f"=== Restarting training (attempt {attempt - 1}/{MAX_RESTARTS}) "
                f"— resuming from latest checkpoint in output_dir ===",
                flush=True,
            )
        try:
            _run(cmd, env=env)
            return
        except subprocess.CalledProcessError as exc:
            if attempt > MAX_RESTARTS:
                print(f"Max training restarts ({MAX_RESTARTS}) exceeded.", flush=True)
                raise
            print(f"Training crashed (exit {exc.returncode}). Retrying in {delay_s}s…", flush=True)
            time.sleep(delay_s)
            delay_s = min(delay_s * 2, RESTART_DELAY_MAX_S)


def _wrap_train_cmd(cmd: list[str], deepspeed_config: str | None, num_gpus: int) -> list[str]:
    """DeepSpeed launcher for multi-GPU; plain python single-GPU without DeepSpeed."""
    if not deepspeed_config:
        if num_gpus > 1:
            return ["torchrun", "--nproc_per_node", str(num_gpus), *cmd[1:]]
        return cmd
    return ["deepspeed", "--num_gpus", str(num_gpus), *cmd[1:]]


def _train_args(
    *,
    model_id: str,
    output_dir: Path,
    logging_dir: Path,
    run_name: str,
    freeze_llm: bool,
    profile: dict,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    num_gpus: int,
    logging_steps: int,
) -> list[str]:
    cmd = [
        "python",
        "src/train/train_sft.py",
        "--use_liger_kernel",
        "True",
    ]
    if profile["deepspeed"]:
        cmd.extend(["--deepspeed", profile["deepspeed"]])
    cmd.extend([
        "--model_id",
        model_id,
        "--data_path",
        str(LLAVA_DIR / "train.json"),
        "--image_folder",
        str(IMAGE_FOLDER),
        "--cache_dir",
        str(HF_CACHE),
        "--remove_unused_columns",
        "False",
        "--freeze_llm",
        str(freeze_llm),
        "--freeze_vision_tower",
        "False",
        "--freeze_merger",
        "False",
        "--bf16",
        "True",
        "--fp16",
        "False",
        "--disable_flash_attn2",
        "True",
        "--output_dir",
        str(output_dir),
        "--logging_dir",
        str(logging_dir),
        "--run_name",
        run_name,
        "--num_train_epochs",
        str(num_epochs),
        "--per_device_train_batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--image_min_pixels",
        str(IMAGE_MIN_PIXELS),
        "--image_max_pixels",
        str(IMAGE_MAX_PIXELS),
        "--learning_rate",
        profile["learning_rate"],
        "--merger_lr",
        profile["merger_lr"],
        "--vision_lr",
        profile["vision_lr"],
        "--weight_decay",
        "0.01",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_strategy",
        "steps",
        "--logging_steps",
        str(logging_steps),
        "--logging_first_step",
        "True",
        "--include_num_input_tokens_seen",
        "True",
        "--tf32",
        "True",
        "--gradient_checkpointing",
        "False",
        "--report_to",
        os.environ.get("CXR_REPORT_TO", "wandb"),
        "--lazy_preprocess",
        "True",
        "--save_strategy",
        "steps",
        "--save_steps",
        "100",
        "--save_total_limit",
        "3",
        "--dataloader_num_workers",
        str(profile["dataloader_workers"]),
        "--max_seq_length",
        "8192",
        "--eval_path",
        str(LLAVA_DIR / "val.json"),
        "--eval_strategy",
        "steps",
        "--eval_steps",
        "500",
        "--per_device_eval_batch_size",
        str(profile["eval_batch_size"]),
        "--prediction_loss_only",
        "False",
    ])
    return _wrap_train_cmd(cmd, profile["deepspeed"], num_gpus)


def _train_phase(phase: str, args: argparse.Namespace) -> None:
    profile = PHASE_PROFILES[phase]
    batch_size = args.batch_size or profile["batch_size"]
    grad_accum = args.grad_accum or profile["grad_accum"]
    num_gpus = args.num_gpus

    if phase == "phase1":
        model_id = MODEL_ID
        freeze_llm = True
        output_dir = OUTPUT_ROOT / "phase1_freeze_llm"
        logging_dir = LOGS_ROOT / "phase1_freeze_llm"
        default_run_name = "phase1-freeze-llm"
    else:
        model_id = args.phase1_dir or str(OUTPUT_ROOT / "phase1_freeze_llm")
        freeze_llm = False
        output_dir = OUTPUT_ROOT / "phase2_full_sft"
        logging_dir = LOGS_ROOT / "phase2_full_sft"
        default_run_name = "phase2-full-sft"

    run_name = args.run_name or default_run_name
    global_batch = batch_size * grad_accum * num_gpus
    print(
        f"{phase} on {num_gpus} GPU(s): batch={batch_size}/gpu, grad_accum={grad_accum}, "
        f"global_batch={global_batch}, deepspeed={profile['deepspeed'] or 'disabled'}",
        flush=True,
    )

    if args.fresh_start:
        _archive_output(output_dir)
    for path in (output_dir, logging_dir, WANDB_DIR, HF_CACHE):
        path.mkdir(parents=True, exist_ok=True)

    checkpoints = _list_checkpoints(output_dir)
    if checkpoints and not args.fresh_start:
        print(
            f"Found {len(checkpoints)} checkpoint(s); will resume from {checkpoints[-1].name}.",
            flush=True,
        )

    cmd = _train_args(
        model_id=model_id,
        output_dir=output_dir,
        logging_dir=logging_dir,
        run_name=run_name,
        freeze_llm=freeze_llm,
        profile=profile,
        num_epochs=args.num_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        num_gpus=num_gpus,
        logging_steps=args.logging_steps,
    )

    if args.dry_run:
        print("$", " ".join(cmd), flush=True)
        return

    env = _base_env()
    env["WANDB_RUN_GROUP"] = phase
    env["WANDB_NAME"] = run_name
    _run_training_resilient(cmd, env)
    print(f"Done. Checkpoints in {output_dir}", flush=True)


# --- Actions -----------------------------------------------------------------

def prepare_data(args: argparse.Namespace) -> None:
    LLAVA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "cxr_vlm.data.prepare_llava",
        "--csv-path",
        args.csv_path,
        "--image-folder",
        str(IMAGE_FOLDER),
        "--output-dir",
        str(LLAVA_DIR),
        "--val-ratio",
        str(args.val_ratio),
    ]
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    _run(cmd)


def download_model(args: argparse.Namespace) -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        "-c",
        (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download('{args.model_id}', cache_dir='{HF_CACHE}')"
        ),
    ])


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    p_data = sub.add_parser("prepare_data", help="CSV -> LLaVA train/val JSON")
    p_data.add_argument("--csv-path", default=str(DEFAULT_CSV))
    p_data.add_argument("--val-ratio", type=float, default=0.20)
    p_data.add_argument("--max-samples", type=int, default=0)

    p_dl = sub.add_parser("download_model", help="Pre-download base model into HF cache")
    p_dl.add_argument("--model-id", default=MODEL_ID)

    for phase in ("phase1", "phase2"):
        p = sub.add_parser(f"train_{phase}", help=f"Run {phase} training")
        p.add_argument("--num-gpus", type=int, default=NUM_GPUS)
        p.add_argument("--batch-size", type=int, default=0, help="per-GPU batch (0 = profile default)")
        p.add_argument("--grad-accum", type=int, default=0, help="grad accumulation (0 = profile default)")
        p.add_argument("--num-epochs", type=int, default=1)
        p.add_argument("--run-name", default="")
        p.add_argument("--logging-steps", type=int, default=1)
        p.add_argument("--fresh-start", action="store_true", help="archive old output dir, start fresh")
        p.add_argument("--dry-run", action="store_true", help="print the training command and exit")
        if phase == "phase2":
            p.add_argument("--phase1-dir", default="", help="phase-1 checkpoint dir (default: outputs/phase1_freeze_llm)")

    args = parser.parse_args()

    if args.action == "prepare_data":
        prepare_data(args)
    elif args.action == "download_model":
        download_model(args)
    elif args.action == "train_phase1":
        _train_phase("phase1", args)
    elif args.action == "train_phase2":
        _train_phase("phase2", args)


if __name__ == "__main__":
    main()
