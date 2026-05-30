"""Simple inference helper for CXR report generation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

import sys

from cxr_vlm.data.prompts import build_user_prompt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model.load_model import load_qwen_vl_generation_model


def parse_report(text: str) -> dict[str, str]:
    obs_match = re.search(r"<observations>\s*(.*?)\s*</observations>", text, re.DOTALL | re.IGNORECASE)
    concl_match = re.search(r"<conclusion>\s*(.*?)\s*</conclusion>", text, re.DOTALL | re.IGNORECASE)
    return {
        "observations": obs_match.group(1).strip() if obs_match else "",
        "conclusion": concl_match.group(1).strip() if concl_match else "",
        "raw": text.strip(),
    }


def generate_report(
    model_path: str,
    image_path: str,
    history: str | None = None,
    max_new_tokens: int = 1024,
    max_pixels: int = 128 * 16 * 128 * 16,
) -> dict[str, str]:
    model = load_qwen_vl_generation_model(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    image = Image.open(image_path).convert("RGB")
    user_text = build_user_prompt(history)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": user_text.replace("<image>\n", "")},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_report(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CXR report generation inference.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--history", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    result = generate_report(
        model_path=args.model_path,
        image_path=args.image_path,
        history=args.history,
        max_new_tokens=args.max_new_tokens,
    )
    print("=== Observations ===")
    print(result["observations"])
    print("\n=== Conclusion ===")
    print(result["conclusion"])


if __name__ == "__main__":
    main()
