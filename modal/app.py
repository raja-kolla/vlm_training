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

import copy
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
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

# Modal hard limit: 24 hours per function invocation (cannot set 1 month on a single run).
# Long jobs: max timeout + retries + checkpoint resume. Re-run the same command if needed.
TRAIN_TIMEOUT_SECONDS = 24 * 60 * 60  # 86400s — Modal maximum
TRAIN_RETRIES = modal.Retries(initial_delay=30.0, max_retries=10)  # whole-container retry on node crash/timeout
# In-process restarts after training subprocess crash (resume from latest checkpoint each time)
TRAIN_MAX_RESTARTS = int(os.environ.get("CXR_MAX_TRAIN_RESTARTS", "100"))
TRAIN_RESTART_DELAY_S = int(os.environ.get("CXR_TRAIN_RESTART_DELAY_S", "60"))
TRAIN_RESTART_DELAY_MAX_S = int(os.environ.get("CXR_TRAIN_RESTART_DELAY_MAX_S", "300"))

# GPU defaults — 8× H200 on one node (Modal max per container). Override with CXR_NUM_GPUS / CXR_PHASE1_GPU.
NUM_GPUS = int(os.environ.get("CXR_NUM_GPUS", "8"))
PHASE1_GPU = os.environ.get("CXR_PHASE1_GPU", f"H200:{NUM_GPUS}")
PHASE2_GPU = os.environ.get("CXR_PHASE2_GPU", f"H200:{NUM_GPUS}")

# Per-GPU training profiles (batch tuned for Qwen3.5-4B + 2048px + seq 8192)
# Effective batch = per_device_train_batch_size × gradient_accumulation_steps
GPU_PROFILES: dict[str, dict[str, dict]] = {
    "H200": {
        "phase1": {
            "batch_size": 4,
            "grad_accum": 4,          # effective batch = 16 (single GPU)
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


def _parse_gpu(gpu: str) -> tuple[str, int]:
    if ":" in gpu:
        name, count = gpu.rsplit(":", 1)
        return name, int(count)
    return gpu, 1


def _gpu_profile(gpu: str, phase: str) -> dict:
    """Resolve batch/deepspeed settings for GPU type and count."""
    base, num_gpus = _parse_gpu(gpu)
    for key in (gpu, base):
        if key in GPU_PROFILES:
            profile = copy.deepcopy(GPU_PROFILES[key][phase])
            break
    else:
        profile = copy.deepcopy(GPU_PROFILES["A100-80GB"][phase])

    profile["num_gpus"] = num_gpus

    # Multi-GPU H200: use DeepSpeed ZeRO-2 launcher (single-GPU skips DeepSpeed on Modal)
    if base == "H200" and num_gpus > 1:
        profile["deepspeed"] = "configs/deepspeed/zero2.json"
        # 8 workers per process is enough; do not set to global batch size
        profile["dataloader_workers"] = min(profile.get("dataloader_workers", 8), 8)
        if phase == "phase1":
            profile["batch_size"] = 16
            profile["grad_accum"] = 1  # global batch = 16 × 1 × 8 = 128
        else:
            profile["batch_size"] = 4
            profile["grad_accum"] = 1  # global batch = 4 × 1 × 8 = 32

    return profile

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


def _distributed_env(num_gpus: int) -> dict[str, str]:
    """Fallback env for single-GPU DeepSpeed only; multi-GPU launcher sets these."""
    return {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "WORLD_SIZE": str(num_gpus),
        "RANK": "0",
        "LOCAL_RANK": "0",
    }


def _reload_volume(reason: str) -> None:
    """Read durable volume state (e.g. after a prior container exited)."""
    try:
        volume.reload()
        print(f"Volume reloaded ({reason})", flush=True)
    except Exception as exc:
        print(f"Volume reload warning ({reason}): {exc}", flush=True)


def _commit_volume(reason: str) -> None:
    """Persist volume writes so checkpoints survive container crash/retry."""
    try:
        volume.commit()
        print(f"Volume committed ({reason})", flush=True)
    except Exception as exc:
        print(f"Volume commit warning ({reason}): {exc}", flush=True)


def _volume_commit_loop(stop: threading.Event, interval_s: int) -> None:
    while not stop.wait(interval_s):
        _commit_volume("periodic checkpoint sync")


def _run(
    cmd: list[str],
    env: dict | None = None,
    *,
    cuda: bool = False,
    num_gpus: int = 1,
    commit_volume: bool = False,
    commit_interval_s: int = 300,
) -> None:
    merged = os.environ.copy()
    merged.update(_volume_env())
    if cuda:
        merged.update(_cuda_env())
    if env:
        merged.update(env)
    # Only inject manual env for single-GPU DeepSpeed; multi-GPU launcher sets ranks
    if cuda and cmd and cmd[0] == "deepspeed" and num_gpus == 1:
        merged.update(_distributed_env(num_gpus))
    print("$", " ".join(cmd), flush=True)

    if not commit_volume:
        subprocess.run(cmd, check=True, cwd=str(CODE), env=merged)
        return

    stop_commit = threading.Event()
    commit_thread = threading.Thread(
        target=_volume_commit_loop,
        args=(stop_commit, commit_interval_s),
        daemon=True,
    )
    commit_thread.start()
    try:
        subprocess.run(cmd, check=True, cwd=str(CODE), env=merged)
    finally:
        stop_commit.set()
        commit_thread.join(timeout=5)
        _commit_volume("training finished or failed")


def _run_training_resilient(
    cmd: list[str],
    env: dict | None = None,
    *,
    num_gpus: int = 1,
    commit_interval_s: int = 60,
    max_restarts: int = TRAIN_MAX_RESTARTS,
) -> None:
    """
    Run training; on subprocess crash, commit volume and restart (auto-resume from checkpoint).
    Modal-level retries (_TRAIN_KWARGS) still apply if the whole container dies.
    """
    delay_s = TRAIN_RESTART_DELAY_S
    for attempt in range(1, max_restarts + 2):
        if attempt > 1:
            print(
                f"=== Restarting training (attempt {attempt - 1}/{max_restarts}) "
                f"— will resume from latest checkpoint in output_dir ===",
                flush=True,
            )
        try:
            _run(
                cmd,
                env=env,
                cuda=True,
                num_gpus=num_gpus,
                commit_volume=True,
                commit_interval_s=commit_interval_s,
            )
            return
        except subprocess.CalledProcessError as exc:
            _commit_volume(f"training crash exit={exc.returncode}")
            _reload_volume("before in-process restart")
            if attempt > max_restarts:
                print(f"Max training restarts ({max_restarts}) exceeded.", flush=True)
                raise
            print(
                f"Training crashed (exit {exc.returncode}). "
                f"Retrying in {delay_s}s…",
                flush=True,
            )
            time.sleep(delay_s)
            delay_s = min(delay_s * 2, TRAIN_RESTART_DELAY_MAX_S)


def _list_checkpoints(output_dir: Path) -> list[Path]:
    """Return sorted checkpoint-N dirs (numeric suffix only)."""
    found: list[Path] = []
    for path in output_dir.glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        if step.isdigit():
            found.append(path)
    return sorted(found, key=lambda p: int(p.name.split("-", 1)[1]))


def _ensure_vol_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _archive_training_output(output_dir: Path) -> None:
    """Move prior run aside instead of deleting checkpoints (safe fresh start)."""
    import shutil

    if not output_dir.exists() or not any(output_dir.iterdir()):
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = output_dir.parent / f"{output_dir.name}.archived.{stamp}"
    print(f"Archiving {output_dir} -> {backup} for fresh start", flush=True)
    shutil.move(str(output_dir), str(backup))
    output_dir.mkdir(parents=True, exist_ok=True)
    _commit_volume("archived prior output dir")


def _wrap_train_cmd(cmd: list[str], deepspeed_config: str | None, num_gpus: int = 1) -> list[str]:
    """DeepSpeed launcher for multi-GPU; plain python for single-GPU without DeepSpeed."""
    if not deepspeed_config:
        if num_gpus > 1:
            return ["torchrun", "--nproc_per_node", str(num_gpus), *cmd[1:]]
        return cmd
    if cmd[0] != "python":
        raise ValueError(f"Expected python launcher, got: {cmd[0]}")
    return ["deepspeed", "--num_gpus", str(num_gpus), *cmd[1:]]


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
    num_gpus: int = 1,
    learning_rate: str = "1e-5",
    vision_lr: str = "2e-6",
    merger_lr: str = "1e-5",
    eval_path: Path | None = None,
    logging_steps: int = 1,
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
        "wandb",
        "--lazy_preprocess",
        "True",
        "--save_strategy",
        "steps",
        "--save_steps",
        "100",
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
    return _wrap_train_cmd(cmd, deepspeed_config, num_gpus=num_gpus)


# --- Entrypoints ------------------------------------------------------------

_TRAIN_KWARGS = dict(
    image=training_image,
    volumes={str(VOL): volume},
    secrets=[modal.Secret.from_name(WANDB_SECRET)],
    timeout=TRAIN_TIMEOUT_SECONDS,
    retries=TRAIN_RETRIES,
    single_use_containers=True,
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
    cpu=32,
    memory=262144,
)
def train_phase1(
    num_epochs: int = 1,
    batch_size: int = 0,
    grad_accum: int = 0,
    run_name: str = "phase1-freeze-llm",
    fresh_start: bool = False,
    logging_steps: int = 1,
) -> str:
    """Phase 1: freeze LLM, train vision encoder + merger."""
    profile = _gpu_profile(PHASE1_GPU, "phase1")
    batch_size = batch_size or profile["batch_size"]
    grad_accum = grad_accum or profile["grad_accum"]
    num_gpus = profile["num_gpus"]
    global_batch = batch_size * grad_accum * num_gpus
    print(
        f"Phase 1 on {PHASE1_GPU}: batch={batch_size}/gpu, grad_accum={grad_accum}, "
        f"gpus={num_gpus}, global_batch={global_batch}, "
        f"deepspeed={profile['deepspeed'] or 'disabled'}, fresh_start={fresh_start}",
        flush=True,
    )

    output_dir = VOL_OUTPUTS / "phase1_freeze_llm"
    logging_dir = VOL_LOGS / "phase1_freeze_llm"
    _reload_volume("start of train_phase1")
    if fresh_start:
        _archive_training_output(output_dir)
    _ensure_vol_dirs(output_dir, logging_dir, VOL_WANDB, VOL_HF_CACHE)

    checkpoints = _list_checkpoints(output_dir)
    if checkpoints and not fresh_start:
        print(
            f"Found {len(checkpoints)} checkpoint(s); will resume from {checkpoints[-1].name}.",
            flush=True,
        )

    _run_training_resilient(
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
            num_gpus=num_gpus,
            logging_steps=logging_steps,
        ),
        env={"WANDB_RUN_GROUP": "phase1", "WANDB_NAME": run_name, **_wandb_env()},
        num_gpus=num_gpus,
    )
    _commit_volume("phase1 complete")
    return str(output_dir)


@app.function(
    **_TRAIN_KWARGS,
    gpu=PHASE2_GPU,
    cpu=32,
    memory=262144,
)
def train_phase2(
    phase1_dir: str = str(VOL_OUTPUTS / "phase1_freeze_llm"),
    num_epochs: int = 1,
    batch_size: int = 0,
    grad_accum: int = 0,
    run_name: str = "phase2-full-sft",
    logging_steps: int = 1,
) -> str:
    """Phase 2: unfreeze LLM (full SFT) from phase-1 checkpoint."""
    profile = _gpu_profile(PHASE2_GPU, "phase2")
    batch_size = batch_size or profile["batch_size"]
    grad_accum = grad_accum or profile["grad_accum"]
    num_gpus = profile["num_gpus"]
    global_batch = batch_size * grad_accum * num_gpus
    print(
        f"Phase 2 on {PHASE2_GPU}: batch={batch_size}/gpu, grad_accum={grad_accum}, "
        f"gpus={num_gpus}, global_batch={global_batch}, "
        f"deepspeed={profile['deepspeed'] or 'disabled'}",
        flush=True,
    )

    output_dir = VOL_OUTPUTS / "phase2_full_sft"
    logging_dir = VOL_LOGS / "phase2_full_sft"
    _reload_volume("start of train_phase2")
    _ensure_vol_dirs(output_dir, logging_dir, VOL_WANDB, VOL_HF_CACHE)

    checkpoints = _list_checkpoints(output_dir)
    if checkpoints:
        print(
            f"Found {len(checkpoints)} checkpoint(s); will resume from {checkpoints[-1].name}.",
            flush=True,
        )

    _run_training_resilient(
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
            num_gpus=num_gpus,
            learning_rate="5e-6",
            vision_lr="1e-6",
            merger_lr="5e-6",
            logging_steps=logging_steps,
        ),
        env={"WANDB_RUN_GROUP": "phase2", "WANDB_NAME": run_name, **_wandb_env()},
        num_gpus=num_gpus,
    )
    _commit_volume("phase2 complete")
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
    fresh_start: bool = False,
    logging_steps: int = 0,
):
    """
    Local CLI wrapper.

    Examples:
      modal run modal/app.py --action train_phase1
      modal run modal/app.py --action train_phase1 --fresh-start  # archives old outputs, does not delete
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
    if fresh_start and action == "train_phase1":
        kwargs["fresh_start"] = True
    if logging_steps and action in {"train_phase1", "train_phase2"}:
        kwargs["logging_steps"] = logging_steps

    fn = {
        "prepare_data": prepare_data,
        "train_phase1": train_phase1,
        "train_phase2": train_phase2,
        "download_model": download_model,
    }[action]
    result = fn.remote(**kwargs)
    print(result)
