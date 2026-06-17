#!/usr/bin/env python3
"""Generate predictions with in-process batched GPU inference (fast path)."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "deploy"))

from deploy.config import (
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    IMAGE_MAX_PIXELS,
)
from deploy.inference import generate_batch_from_loaded, load_image_from_path, load_model_and_processor
from prompt import user_prompt

IMAGE_DIR = Path("/root/disk/data/chest")
EVAL_CSV = "/home/raja/eval_data/evalset_5k.csv"
OUT_CSV = "/home/raja/eval_data/my_model_predictions.csv"
FIELDNAMES = ["image_path", "raw_report"]


def load_name_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_done_image_names(out_path: Path) -> set[str]:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    done = pd.read_csv(out_path, usecols=["image_path"])
    return {Path(p).name for p in done["image_path"].astype(str)}


def chunked(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched local prediction generation")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--eval-csv", default=EVAL_CSV)
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    df = pd.read_csv(args.eval_csv)
    total = len(df)
    out_path = Path(args.out_csv)
    skipped_path = out_path.with_suffix(".skipped.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_names = load_done_image_names(out_path)
    skipped_names = load_name_set(skipped_path)
    resuming = bool(done_names)
    if resuming:
        print(f"Resuming: skipping {len(done_names)} images already in {args.out_csv}")
    if skipped_names:
        print(f"Skipping {len(skipped_names)} images listed in {skipped_path}")

    pending = [
        row.image_name
        for row in df.itertuples(index=False)
        if row.image_name not in done_names and row.image_name not in skipped_names
    ]
    if not pending:
        print(f"nothing to do -> {args.out_csv} ({len(done_names)}/{total} rows)")
        return

    print(f"Loading model from {args.model_path} ...")
    model, processor = load_model_and_processor(
        str(args.model_path),
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    )
    prompt = user_prompt.strip()
    print(
        f"Predicting {len(pending)} images | batch_size={args.batch_size} "
        f"| max_new_tokens={args.max_new_tokens}"
    )

    t0 = time.perf_counter()
    completed = 0
    with out_path.open("a" if resuming else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not resuming:
            writer.writeheader()

        pbar = tqdm(total=len(pending), desc="Predicting", unit="img")
        for batch_names in chunked(pending, args.batch_size):
            batch_images = []
            batch_paths = []
            for image_name in batch_names:
                image_path = args.image_dir / image_name
                if not image_path.exists():
                    tqdm.write(f"skip: {image_name} (file not found)")
                    with skipped_path.open("a", encoding="utf-8") as sf:
                        sf.write(f"{image_name}\n")
                    skipped_names.add(image_name)
                    pbar.update(1)
                    continue
                batch_images.append(load_image_from_path(str(image_path)))
                batch_paths.append(str(image_path))

            if not batch_images:
                continue

            texts = generate_batch_from_loaded(
                model,
                processor,
                batch_images,
                prompt,
                max_new_tokens=args.max_new_tokens,
                max_pixels=IMAGE_MAX_PIXELS,
            )
            for image_path, text in zip(batch_paths, texts):
                writer.writerow({"image_path": image_path, "raw_report": text})
                done_names.add(Path(image_path).name)
                completed += 1
            f.flush()
            pbar.update(len(batch_paths))
            elapsed = time.perf_counter() - t0
            if completed:
                pbar.set_postfix_str(f"{completed / elapsed:.2f} img/s", refresh=True)
        pbar.close()

    elapsed = time.perf_counter() - t0
    rate = completed / elapsed if elapsed > 0 else 0.0
    print(f"done -> {args.out_csv} ({len(done_names)}/{total} rows, {len(skipped_names)} skipped, {rate:.2f} img/s)")


if __name__ == "__main__":
    main()
