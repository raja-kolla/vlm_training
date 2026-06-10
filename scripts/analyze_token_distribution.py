#!/usr/bin/env python3
"""Token distribution for LLaVA JSON using the same path as SupervisedDataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import importlib.util

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_constants = _load_module("constants", SRC / "constants.py")
_data_utils = _load_module("data_utils", SRC / "dataset" / "data_utils.py")

DEFAULT_IM_END_TOKEN = _constants.DEFAULT_IM_END_TOKEN
DEFAULT_IM_START_TOKEN = _constants.DEFAULT_IM_START_TOKEN
DEFAULT_IMAGE_TOKEN = _constants.DEFAULT_IMAGE_TOKEN
SYSTEM_MESSAGE = _constants.SYSTEM_MESSAGE
format_assistant_response = _data_utils.format_assistant_response
get_image_info = _data_utils.get_image_info
get_mm_token_type_ids = _data_utils.get_mm_token_type_ids
get_qwen_multimodal_settings = _data_utils.get_qwen_multimodal_settings
llava_to_openai = _data_utils.llava_to_openai
use_default_system_message = _data_utils.use_default_system_message


@dataclass
class SampleStats:
    total: int
    text: int
    vision: int
    prompt: int
    response: int
    image_grid_tokens: int


def count_sample(
    record: dict,
    processor,
    *,
    image_folder: str,
    image_min_pixels: int,
    image_max_pixels: int,
    image_patch_size: int,
    model_type: str,
) -> SampleStats:
    sources = copy.deepcopy(record)
    image_file = sources["image"]
    if not os.path.isabs(image_file) and not image_file.startswith("http"):
        image_file = os.path.join(image_folder, image_file)

    image_input = get_image_info(
        image_file,
        image_min_pixels,
        image_max_pixels,
        None,
        None,
        image_patch_size,
    )

    openai_sources = llava_to_openai(sources["conversations"], is_video=False)
    all_input_ids: list[torch.Tensor] = []
    all_mm: list[torch.Tensor] = []
    prompt_len = 0
    response_len = 0

    if len(SYSTEM_MESSAGE) > 0 and use_default_system_message(model_type):
        system_message = (
            f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        )
        system_ids = processor.tokenizer(
            system_message, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        all_input_ids.append(system_ids.squeeze(0))
        all_mm.append(torch.zeros_like(system_ids, dtype=torch.long).squeeze(0))

    for j in range(0, len(openai_sources), 2):
        user_input = openai_sources[j]
        gpt_response = openai_sources[j + 1]
        assistant_prefill, assistant_content = format_assistant_response(
            gpt_response["content"],
            gpt_response.get("reasoning"),
            enable_reasoning=False,
            use_reasoning_prefill=False,
            use_closed_think_prefill=False,
        )
        user_text = (
            f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}"
            f"{DEFAULT_IM_END_TOKEN}\n"
            f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{assistant_prefill}"
        )
        gpt_text = f"{assistant_content}{DEFAULT_IM_END_TOKEN}\n"

        if DEFAULT_IMAGE_TOKEN in user_text:
            inputs = processor(
                text=[user_text],
                images=[image_input],
                videos=None,
                padding=False,
                do_resize=False,
                return_tensors="pt",
            )
            prompt_ids = inputs["input_ids"]
            mm_ids = get_mm_token_type_ids(inputs, prompt_ids)
        else:
            prompt_ids = processor.tokenizer(
                user_text, add_special_tokens=False, padding=False, return_tensors="pt"
            )["input_ids"]
            mm_ids = torch.zeros_like(prompt_ids, dtype=torch.long)

        response_ids = processor.tokenizer(
            gpt_text, add_special_tokens=False, padding=False, return_tensors="pt"
        )["input_ids"]
        response_mm = torch.zeros_like(response_ids, dtype=torch.long)

        input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
        mm_token_type_ids = torch.cat([mm_ids, response_mm], dim=1).squeeze(0)

        all_input_ids.append(input_ids)
        all_mm.append(mm_token_type_ids)
        prompt_len += int(prompt_ids.shape[-1])
        response_len += int(response_ids.shape[-1])

    input_ids = torch.cat(all_input_ids, dim=0)
    mm_token_type_ids = torch.cat(all_mm, dim=0)
    vision = int((mm_token_type_ids != 0).sum().item())
    total = int(input_ids.numel())
    text = total - vision

    image_grid_tokens = 0
    if "image_grid_thw" in locals().get("inputs", {}):
        pass
    # vision pad count from grid when available
    try:
        grid = inputs["image_grid_thw"]
        image_grid_tokens = int(grid.prod(dim=-1).sum().item())
    except Exception:
        image_grid_tokens = vision

    return SampleStats(
        total=total,
        text=text,
        vision=vision,
        prompt=prompt_len,
        response=response_len,
        image_grid_tokens=image_grid_tokens,
    )


def summarize(values: np.ndarray, name: str) -> str:
    p = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    over_8k = int((values > 8192).sum())
    return (
        f"{name}: n={len(values):,}  mean={values.mean():.1f}  std={values.std():.1f}  "
        f"min={values.min()}  max={values.max()}\n"
        f"  p01={p[0]:.0f}  p05={p[1]:.0f}  p25={p[2]:.0f}  p50={p[3]:.0f}  "
        f"p75={p[4]:.0f}  p95={p[5]:.0f}  p99={p[6]:.0f}  >8192={over_8k:,} ({100*over_8k/len(values):.2f}%)"
    )


def histogram(values: np.ndarray, bins: list[int]) -> str:
    counts, edges = np.histogram(values, bins=bins)
    lines = ["  bin_range          count      pct"]
    total = len(values)
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        label = f"[{lo:5d}, {hi:5d})" if hi != edges[-1] else f"[{lo:5d}, inf)"
        pct = 100.0 * c / total if total else 0
        lines.append(f"  {label:18s} {c:8,d}  {pct:6.2f}%")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="training_data/llava/train.json")
    parser.add_argument("--image_folder", default="/root/bionicsuite1/home/ai-user/chest_classifier/disk/chest_images")
    parser.add_argument("--model_id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--image_min_pixels", type=int, default=256 * 16 * 16)
    parser.add_argument("--image_max_pixels", type=int, default=128 * 16 * 128 * 16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_stride", type=int, default=1, help="Use every Nth sample for faster estimate")
    args = parser.parse_args()

    data_path = ROOT / args.data_path
    print(f"Loading {data_path} ...")
    with open(data_path) as f:
        records = json.load(f)
    if args.sample_stride > 1:
        records = records[:: args.sample_stride]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    print(f"Processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model_type, image_patch_size, _ = get_qwen_multimodal_settings(args.model_id)
    print(f"model_type={model_type}  patch_size={image_patch_size}")
    print(f"image_min_pixels={args.image_min_pixels}  image_max_pixels={args.image_max_pixels}")
    print(f"Samples to analyze: {len(records):,}\n")

    totals, texts, visions, prompts, responses, grids = [], [], [], [], [], []
    missing_images = 0

    for i, record in enumerate(records):
        image_file = record["image"]
        path = image_file if os.path.isabs(image_file) else os.path.join(args.image_folder, image_file)
        if not os.path.exists(path):
            missing_images += 1
            continue
        try:
            stats = count_sample(
                record,
                processor,
                image_folder=args.image_folder,
                image_min_pixels=args.image_min_pixels,
                image_max_pixels=args.image_max_pixels,
                image_patch_size=image_patch_size,
                model_type=model_type,
            )
        except Exception as e:
            if missing_images < 3:
                print(f"  skip {record.get('id')}: {e}")
            missing_images += 1
            continue
        totals.append(stats.total)
        texts.append(stats.text)
        visions.append(stats.vision)
        prompts.append(stats.prompt)
        responses.append(stats.response)
        grids.append(stats.image_grid_tokens)
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1:,} / {len(records):,} ...", flush=True)

    if not totals:
        raise SystemExit("No samples processed. Check image_folder paths.")

    totals_a = np.array(totals)
    texts_a = np.array(texts)
    visions_a = np.array(visions)
    prompts_a = np.array(prompts)
    responses_a = np.array(responses)
    grids_a = np.array(grids)

    print("\n=== Token distribution (SupervisedDataset-style, Qwen3.5 processor) ===\n")
    print(summarize(totals_a, "Total sequence length"))
    print(summarize(texts_a, "Text tokens (non-vision)"))
    print(summarize(visions_a, "Vision/mm tokens (mm_token_type_ids != 0)"))
    print(summarize(prompts_a, "Prompt tokens (user turn incl. image placeholders)"))
    print(summarize(responses_a, "Response tokens (assistant, trained)"))
    print(summarize(grids_a, "Image grid product (image_grid_thw)"))

    print("\n--- Total length histogram ---")
    bins = [0, 512, 1024, 2048, 4096, 6144, 8192, 10240, 16384, 10**9]
    print(histogram(totals_a, bins))

    print("\n--- Vision token histogram ---")
    v_bins = [0, 64, 128, 192, 256, 384, 512, 768, 1024, 2048, 10**9]
    print(histogram(visions_a, v_bins))

    print("\n--- Text token histogram ---")
    t_bins = [0, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 10**9]
    print(histogram(texts_a, t_bins))

    print(f"\nSkipped/missing: {missing_images:,}")
    print(f"Effective dataset fraction analyzed: {len(totals):,} / {len(records):,}")


if __name__ == "__main__":
    main()
