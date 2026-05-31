"""
Modal entrypoints for CXR VLM training.

No conda required — Modal builds a GPU container with pip/uv.

Volume: vlm_training (mounted at /vol)

  /vol/images/                         JPEG files (file_path in CSV)
  /vol/csv/filtered_columns_cxr_0.3M.csv
  /vol/llava/                          train.json, val.json, meta.json
  /vol/hf_cache/                       HuggingFace + datasets cache
  /vol/wandb/                          W&B run files (offline cache)
  /vol/logs/                           trainer logs per phase
  /vol/outputs/phase1_freeze_llm/      phase-1 checkpoints
  /vol/outputs/phase2_full_sft/        phase-2 checkpoints

Setup W&B (once, local):
  modal secret create wandb WANDB_API_KEY=<your-key> WANDB_ENTITY=rajaphanindra

Usage:
  modal run modal/app.py --action prepare_data
  modal run modal/app.py --action train_phase1
  modal run modal/app.py --action train_phase2
  modal run modal/app.py --action download_model
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

# --- Config -----------------------------------------------------------------

APP_NAME = "cxr-vlm-training"
DEFAULT_VOLUME = os.environ.get("CXR_MODAL_VOLUME", "vlm_training")
WANDB_SECRET = os.environ.get("CXR_WANDB_SECRET", "wandb")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "cxr-vlm-qwen35")
# Set in Modal secret or shell: WANDB_ENTITY=rajaphanindra (personal, not org)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")

VOL = Path("/vol")
VOL_IMAGES = VOL / "images"
VOL_CSV = VOL / "csv"
VOL_LLAVA = VOL / "llava"
VOL_OUTPUTS = VOL / "outputs"
VOL_HF_CACHE = VOL / "hf_cache"
VOL_WANDB = VOL / "wandb"
VOL_LOGS = VOL / "logs"
VOL_TORCH = VOL / "torch"
DEFAULT_CSV = VOL_CSV / "filtered_columns_cxr_0.3M.csv"

CODE = Path("/root/vlm_training")

MODEL_ID = "Qwen/Qwen3.5-4B"
IMAGE_MIN_PIXELS = 256 * 16 * 16
IMAGE_MAX_PIXELS = 128 * 16 * 128 * 16  # max side 2048

# GPU defaults — H200 has 141 GB HBM; override via CXR_PHASE1_GPU / CXR_PHASE2_GPU
PHASE1_GPU = os.environ.get("CXR_PHASE1_GPU", "H200")
PHASE2_GPU = os.environ.get("CXR_PHASE2_GPU", "H200")

# Per-GPU training profiles (batch tuned for Qwen3.5-4B + 2048px + seq 8192)
# Effective batch = per_device_train_batch_size × gradient_accumulation_steps
GPU_PROFILES: dict[str, dict[str, dict]] = {
    "H200": {
        "phase1": {
            "batch_size": 4,
            "grad_accum": 4,          # effective batch = 16
            "eval_batch_size": 2,
            "deepspeed": None,          # single H200: skip DeepSpeed (avoids MPI/mpi4py on Modal)
            "dataloader_workers": 8,
        },
        "phase2": {
            "batch_size": 2,
            "grad_accum": 8,          # effective batch = 16
            "eval_batch_size": 1,
            "deepspeed": None,
            "dataloader_workers": 8,
        },
    },
    "H100": {
        "phase1": {
            "batch_size": 2,
            "grad_accum": 8,
            "eval_batch_size": 1,
            "deepspeed": "configs/deepspeed/zero2.json",
            "dataloader_workers": 8,
        },
        "phase2": {
            "batch_size": 1,
            "grad_accum": 16,
            "eval_batch_size": 1,
            "deepspeed": "configs/deepspeed/zero3_offload.json",
            "dataloader_workers": 4,
        },
    },
    "A100-80GB": {
        "phase1": {
            "batch_size": 1,
            "grad_accum": 8,
            "eval_batch_size": 1,
            "deepspeed": "configs/deepspeed/zero2.json",
            "dataloader_workers": 4,
        },
        "phase2": {
            "batch_size": 1,
            "grad_accum": 16,
            "eval_batch_size": 1,
            "deepspeed": "configs/deepspeed/zero3_offload.json",
            "dataloader_workers": 4,
        },
    },
}


def _gpu_profile(gpu: str, phase: str) -> dict:
    """Resolve batch/deepspeed settings; fall back to A100-80GB if GPU unknown."""
    for key in (gpu, gpu.split(":")[0]):
        if key in GPU_PROFILES:
            return GPU_PROFILES[key][phase]
    return GPU_PROFILES["A100-80GB"][phase]

# --- Image (pip-only, no conda) ---------------------------------------------

def _wandb_env() -> dict[str, str]:
    """W&B settings — metrics only; checkpoints stay on the volume."""
    env = {
        "WANDB_LOG_MODEL": "false",  # avoid org "models write access" errors
        "WANDB_WATCH": "false",
    }
    entity = os.environ.get("WANDB_ENTITY") or WANDB_ENTITY
    if entity:
        env["WANDB_ENTITY"] = entity
    return env


def _volume_env() -> dict[str, str]:
    """Env vars so caches and W&B files land on the volume, not ephemeral disk."""
    return {
        "PYTHONPATH": f"{CODE}/src",
        "HF_HOME": str(VOL_HF_CACHE),
        "TRANSFORMERS_CACHE": str(VOL_HF_CACHE),
        "HF_DATASETS_CACHE": str(VOL_HF_CACHE / "datasets"),
        "TORCH_HOME": str(VOL_TORCH),
        "WANDB_DIR": str(VOL_WANDB),
        "WANDB_PROJECT": WANDB_PROJECT,
        "TOKENIZERS_PARALLELISM": "false",
    }


def _cuda_env() -> dict[str, str]:
    """DeepSpeed requires CUDA toolkit headers (CUDA_HOME) at import time."""
    return {
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
    }


def _code_mounts(image: modal.Image) -> modal.Image:
    return (
        image.add_local_dir("src", remote_path=f"{CODE}/src")
        .add_local_dir("cxr_vlm", remote_path=f"{CODE}/cxr_vlm")
        .add_local_dir("configs", remote_path=f"{CODE}/configs")
    )


# Lightweight CPU image for data prep / model download
cpu_image = _code_mounts(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas>=2.2.0", "ujson>=5.10.0", "huggingface_hub>=0.34.0")
    .env(_volume_env())
)

# GPU training image — nvidia devel base so DeepSpeed finds CUDA_HOME
training_image = _code_mounts(
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install_from_requirements("modal/requirements.txt")
    .env({**_volume_env(), **_cuda_env()})
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(DEFAULT_VOLUME, create_if_missing=True)


def _distributed_env() -> dict[str, str]:
    """Single-GPU torch.distributed env (needed if DeepSpeed launcher is used)."""
    return {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }


def _run(cmd: list[str], env: dict | None = None, *, cuda: bool = False) -> None:
    merged = os.environ.copy()
    merged.update(_volume_env())
    if cuda:
        merged.update(_cuda_env())
    if env:
        merged.update(env)
    # DeepSpeed launcher needs distributed env even on 1 GPU
    if cuda and cmd and cmd[0] == "deepspeed":
        merged.update(_distributed_env())
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(CODE), env=merged)


def _ensure_vol_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _wrap_train_cmd(cmd: list[str], deepspeed_config: str | None) -> list[str]:
    """Use DeepSpeed launcher for multi-process init; plain python for single-GPU."""
    if not deepspeed_config:
        return cmd
    if cmd[0] != "python":
        raise ValueError(f"Expected python launcher, got: {cmd[0]}")
    return ["deepspeed", "--num_gpus", "1", *cmd[1:]]


def _train_args(
    *,
    model_id: str,
    data_path: Path,
    image_folder: Path,
    output_dir: Path,
    logging_dir: Path,
    run_name: str,
    freeze_llm: bool,
    deepspeed_config: str | None,
    num_epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    eval_batch_size: int = 1,
    dataloader_workers: int = 4,
    learning_rate: str = "1e-5",
    vision_lr: str = "2e-6",
    merger_lr: str = "1e-5",
    eval_path: Path | None = None,
) -> list[str]:
    eval_path = eval_path or (VOL_LLAVA / "val.json")
    cmd = [
        "python",
        "src/train/train_sft.py",
        "--use_liger_kernel",
        "True",
    ]
    if deepspeed_config:
        cmd.extend(["--deepspeed", deepspeed_config])
    cmd.extend([
        "--model_id",
        model_id,
        "--data_path",
        str(data_path),
        "--image_folder",
        str(image_folder),
        "--cache_dir",
        str(VOL_HF_CACHE),
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
        learning_rate,
        "--merger_lr",
        merger_lr,
        "--vision_lr",
        vision_lr,
        "--weight_decay",
        "0.01",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "10",
        "--tf32",
        "True",
        "--gradient_checkpointing",
        "True",
        "--report_to",
        "wandb",
        "--lazy_preprocess",
        "True",
        "--save_strategy",
        "steps",
        "--save_steps",
        "500",
        "--save_total_limit",
        "3",
        "--dataloader_num_workers",
        str(dataloader_workers),
        "--max_seq_length",
        "8192",
        "--eval_path",
        str(eval_path),
        "--eval_strategy",
        "steps",
        "--eval_steps",
        "500",
        "--per_device_eval_batch_size",
        str(eval_batch_size),
        "--prediction_loss_only",
        "False",
    ])
    return _wrap_train_cmd(cmd, deepspeed_config)


# --- Entrypoints ------------------------------------------------------------

_TRAIN_KWARGS = dict(
    image=training_image,
    volumes={str(VOL): volume},
    secrets=[modal.Secret.from_name(WANDB_SECRET)],
)


@app.function(
    image=cpu_image,
    volumes={str(VOL): volume},
    timeout=60 * 60,
    cpu=4,
    memory=16384,
)
def prepare_data(
    csv_path: str = str(DEFAULT_CSV),
    image_folder: str = str(VOL_IMAGES),
    output_dir: str = str(VOL_LLAVA),
    val_ratio: float = 0.20,
) -> dict:
    """Convert CSV on the volume to LLaVA train/val JSON (CPU-only, cheap)."""
    _ensure_vol_dirs(Path(output_dir))
    _run(
        [
            "python",
            "-m",
            "cxr_vlm.data.prepare_llava",
            "--csv-path",
            csv_path,
            "--image-folder",
            image_folder,
            "--output-dir",
            output_dir,
            "--val-ratio",
            str(val_ratio),
        ]
    )
    volume.commit()
    meta_path = Path(output_dir) / "meta.json"
    import json

    return json.loads(meta_path.read_text())


@app.function(
    **_TRAIN_KWARGS,
    gpu=PHASE1_GPU,
    timeout=24 * 60 * 60,
    cpu=8,
    memory=65536,
)
def train_phase1(
    num_epochs: int = 1,
    batch_size: int = 0,
    grad_accum: int = 0,
    run_name: str = "phase1-freeze-llm",
) -> str:
    """Phase 1: freeze LLM, train vision encoder + merger."""
    profile = _gpu_profile(PHASE1_GPU, "phase1")
    batch_size = batch_size or profile["batch_size"]
    grad_accum = grad_accum or profile["grad_accum"]
    print(
        f"Phase 1 on {PHASE1_GPU}: batch={batch_size}, grad_accum={grad_accum}, "
        f"effective_batch={batch_size * grad_accum}, deepspeed={profile['deepspeed'] or 'disabled'}",
        flush=True,
    )

    output_dir = VOL_OUTPUTS / "phase1_freeze_llm"
    logging_dir = VOL_LOGS / "phase1_freeze_llm"
    _ensure_vol_dirs(output_dir, logging_dir, VOL_WANDB, VOL_HF_CACHE)

    _run(
        _train_args(
            model_id=MODEL_ID,
            data_path=VOL_LLAVA / "train.json",
            image_folder=VOL_IMAGES,
            output_dir=output_dir,
            logging_dir=logging_dir,
            run_name=run_name,
            freeze_llm=True,
            deepspeed_config=profile["deepspeed"],
            num_epochs=num_epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            eval_batch_size=profile["eval_batch_size"],
            dataloader_workers=profile["dataloader_workers"],
        ),
        env={"WANDB_RUN_GROUP": "phase1", "WANDB_NAME": run_name, **_wandb_env()},
        cuda=True,
    )
    volume.commit()
    return str(output_dir)


@app.function(
    **_TRAIN_KWARGS,
    gpu=PHASE2_GPU,
    timeout=24 * 60 * 60,
    cpu=8,
    memory=65536,
)
def train_phase2(
    phase1_dir: str = str(VOL_OUTPUTS / "phase1_freeze_llm"),
    num_epochs: int = 1,
    batch_size: int = 0,
    grad_accum: int = 0,
    run_name: str = "phase2-full-sft",
) -> str:
    """Phase 2: unfreeze LLM (full SFT) from phase-1 checkpoint."""
    profile = _gpu_profile(PHASE2_GPU, "phase2")
    batch_size = batch_size or profile["batch_size"]
    grad_accum = grad_accum or profile["grad_accum"]
    print(
        f"Phase 2 on {PHASE2_GPU}: batch={batch_size}, grad_accum={grad_accum}, "
        f"effective_batch={batch_size * grad_accum}, deepspeed={profile['deepspeed'] or 'disabled'}",
        flush=True,
    )

    output_dir = VOL_OUTPUTS / "phase2_full_sft"
    logging_dir = VOL_LOGS / "phase2_full_sft"
    _ensure_vol_dirs(output_dir, logging_dir, VOL_WANDB, VOL_HF_CACHE)

    _run(
        _train_args(
            model_id=phase1_dir,
            data_path=VOL_LLAVA / "train.json",
            image_folder=VOL_IMAGES,
            output_dir=output_dir,
            logging_dir=logging_dir,
            run_name=run_name,
            freeze_llm=False,
            deepspeed_config=profile["deepspeed"],
            num_epochs=num_epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            eval_batch_size=profile["eval_batch_size"],
            dataloader_workers=profile["dataloader_workers"],
            learning_rate="5e-6",
            vision_lr="1e-6",
            merger_lr="5e-6",
        ),
        env={"WANDB_RUN_GROUP": "phase2", "WANDB_NAME": run_name, **_wandb_env()},
        cuda=True,
    )
    volume.commit()
    return str(output_dir)


@app.function(
    image=cpu_image,
    volumes={str(VOL): volume},
    timeout=2 * 60 * 60,
    cpu=2,
)
def download_model(model_id: str = MODEL_ID) -> str:
    """Pre-download base model weights into the volume HF cache."""
    _ensure_vol_dirs(VOL_HF_CACHE, VOL_HF_CACHE / "datasets", VOL_TORCH)

    _run(
        [
            "python",
            "-c",
            (
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download('{model_id}', cache_dir='{VOL_HF_CACHE}')"
            ),
        ]
    )
    volume.commit()
    return str(VOL_HF_CACHE)


@app.local_entrypoint()
def main(
    action: str = "train_phase1",
    run_name: str = "",
    batch_size: int = 0,
    grad_accum: int = 0,
):
    """
    Local CLI wrapper.

    Examples:
      modal run modal/app.py --action train_phase1
      modal run modal/app.py --action train_phase2 --run-name cxr-p2-v1
      modal run modal/app.py --action train_phase1 --batch-size 6 --grad-accum 2
    """
    kwargs: dict = {}
    if run_name and action in {"train_phase1", "train_phase2"}:
        kwargs["run_name"] = run_name
    if batch_size and action in {"train_phase1", "train_phase2"}:
        kwargs["batch_size"] = batch_size
    if grad_accum and action in {"train_phase1", "train_phase2"}:
        kwargs["grad_accum"] = grad_accum

    fn = {
        "prepare_data": prepare_data,
        "train_phase1": train_phase1,
        "train_phase2": train_phase2,
        "download_model": download_model,
    }[action]
    result = fn.remote(**kwargs)
    print(result)
