"""Generic VLM inference: image + prompt -> raw model text."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.load_model import load_qwen_vl_generation_model


def load_model_and_processor(
    model_path: str,
    *,
    attn_implementation: str = "sdpa",
):
    model = load_qwen_vl_generation_model(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


def generate_from_loaded(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int = 1024,
    max_pixels: int = 128 * 16 * 128 * 16,
) -> str:
    return generate_batch_from_loaded(
        model,
        processor,
        [image],
        prompt,
        max_new_tokens=max_new_tokens,
        max_pixels=max_pixels,
    )[0]


def generate_batch_from_loaded(
    model,
    processor,
    images: list[Image.Image],
    prompt: str,
    *,
    max_new_tokens: int = 1024,
    max_pixels: int = 128 * 16 * 128 * 16,
) -> list[str]:
    if not images:
        return []

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "max_pixels": max_pixels},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return processor.batch_decode(
        output_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def load_image_from_path(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(path).convert("RGB")


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    import io

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
