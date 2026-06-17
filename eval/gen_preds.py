#!/usr/bin/env python3
"""Generate predictions CSV for vlm-evals-xray."""
import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, "/home/raja/vlm_training/deploy")
from prompt import user_prompt

DEFAULT_API_URL = "http://localhost:8000/v1/generate"
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


def predict_one(
    *,
    api_url: str,
    image_name: str,
    image_dir: Path,
    max_new_tokens: int,
    timeout: int,
) -> tuple[str, str | None, str | None]:
    """Returns (image_name, raw_report, skip_reason)."""
    image_path = str(image_dir / image_name)
    if not Path(image_path).exists():
        return image_name, None, "file not found"

    resp = requests.post(
        api_url,
        json={
            "prompt": user_prompt.strip(),
            "image_path": image_path,
            "max_new_tokens": max_new_tokens,
        },
        timeout=timeout,
    )
    if resp.status_code == 404:
        detail = resp.json().get("detail", resp.text)
        return image_name, None, detail

    resp.raise_for_status()
    return image_name, resp.json()["text"], None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model predictions in parallel")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", DEFAULT_API_URL))
    parser.add_argument("--eval-csv", default=EVAL_CSV)
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("NUM_REPLICAS", "5")),
        help="Concurrent API requests (match server NUM_REPLICAS)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=600)
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

    print(f"Predicting {len(pending)} images with {args.workers} workers -> {args.api_url}")

    write_lock = threading.Lock()
    with out_path.open("a" if resuming else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not resuming:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    predict_one,
                    api_url=args.api_url,
                    image_name=image_name,
                    image_dir=args.image_dir,
                    max_new_tokens=args.max_new_tokens,
                    timeout=args.timeout,
                ): image_name
                for image_name in pending
            }

            pbar = tqdm(as_completed(futures), total=len(futures), desc="Predicting", unit="img")
            for future in pbar:
                image_name, raw_report, skip_reason = future.result()
                pbar.set_postfix_str(image_name, refresh=False)

                if skip_reason is not None:
                    tqdm.write(f"skip: {image_name} ({skip_reason})")
                    with write_lock:
                        with skipped_path.open("a", encoding="utf-8") as sf:
                            sf.write(f"{image_name}\n")
                        skipped_names.add(image_name)
                    continue

                with write_lock:
                    writer.writerow({"image_path": str(args.image_dir / image_name), "raw_report": raw_report})
                    f.flush()
                    done_names.add(image_name)

    print(f"done -> {args.out_csv} ({len(done_names)}/{total} rows, {len(skipped_names)} skipped)")


if __name__ == "__main__":
    main()
