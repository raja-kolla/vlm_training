import pathlib
import transformers
import torch
import logging


def _checkpoint_step(path: pathlib.Path) -> int:
    return int(path.name.split("-", 1)[1])


def find_latest_checkpoint(output_dir: str) -> pathlib.Path | None:
    """Return the highest-numbered checkpoint-N dir (excludes archived .hf-single-gpu dirs)."""
    checkpoints: list[pathlib.Path] = []
    for path in pathlib.Path(output_dir).glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        if step.isdigit():
            checkpoints.append(path)
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_step)


def is_deepspeed_checkpoint(checkpoint_dir: pathlib.Path) -> bool:
    """True if checkpoint contains DeepSpeed ZeRO shards (required for DS resume)."""
    if any(checkpoint_dir.glob("global_step*")):
        return True
    if any(checkpoint_dir.glob("**/bf16_zero*.pt")):
        return True
    if any(checkpoint_dir.glob("**/zero_pp*.pt")):
        return True
    if any(checkpoint_dir.glob("**/mp_rank_*_model_states.pt")):
        return True
    return False


def load_hf_checkpoint_weights(model, checkpoint_dir: str | pathlib.Path, *, log_fn=print) -> None:
    """Load model.safetensors / pytorch_model.bin from a standard HF Trainer checkpoint."""
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log_fn(
            f"Loaded weights from {safetensors_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return

    bin_path = checkpoint_dir / "pytorch_model.bin"
    if bin_path.is_file():
        state_dict = torch.load(bin_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log_fn(
            f"Loaded weights from {bin_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return

    raise FileNotFoundError(
        f"No model weights found in {checkpoint_dir} (expected model.safetensors or pytorch_model.bin)"
    )


def resolve_training_resume(
    output_dir: str,
    *,
    using_deepspeed: bool,
    log_fn=print,
) -> tuple[str | None, pathlib.Path | None]:
    """
    Decide how to resume training. Never deletes or renames existing checkpoints.

    Returns (resume_checkpoint_path, hf_weights_only_dir).
    - resume_checkpoint_path: pass to trainer.train(resume_from_checkpoint=...)
    - hf_weights_only_dir: load model weights only (optimizer/step reset); checkpoints kept on disk
    """
    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        log_fn("No checkpoint found — starting training from scratch.")
        return None, None

    if using_deepspeed and is_deepspeed_checkpoint(latest):
        log_fn(f"Resuming DeepSpeed training from {latest}")
        return str(latest), None

    if using_deepspeed and not is_deepspeed_checkpoint(latest):
        log_fn(
            f"Found HF-format checkpoint at {latest} (not DeepSpeed-sharded). "
            "Loading model weights only; optimizer/scheduler/step counter will reset. "
            "Checkpoint files are preserved on disk."
        )
        return None, latest

    log_fn(f"Resuming training from {latest}")
    return str(latest), None


class CheckpointPersistCallback(transformers.TrainerCallback):
    """Log checkpoint path on save (rank 0); Modal commits volume on a timer + on crash."""

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        latest = find_latest_checkpoint(args.output_dir)
        if latest is not None:
            print(f"Checkpoint saved: {latest} (global_step={state.global_step})", flush=True)
        return control


class StepDetailsCallback(transformers.TrainerCallback):
    """Attach per-step GPU memory stats to each W&B log entry."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return control
        logs["train/global_step"] = state.global_step
        logs["train/epoch"] = round(state.epoch, 4)
        if torch.cuda.is_available():
            logs["gpu/mem_allocated_gb"] = round(
                torch.cuda.memory_allocated() / (1024**3), 3
            )
            logs["gpu/max_mem_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            )
        return control


def maybe_zero_3(param, ignore_status=False, name=None, device=torch.device('cpu')):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if type(device) is str:
        device = torch.device(device)
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach()
    else:
        param = param.detach()
    if device == param.device:
        return param.clone()
    else:
        return param.to(device)

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)