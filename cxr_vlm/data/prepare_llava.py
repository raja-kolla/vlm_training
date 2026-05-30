"""Convert CXR CSV data to LLaVA-format JSON for Qwen-VL fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from cxr_vlm.data.prompts import build_assistant_response, build_user_prompt


def _validate_row(row: pd.Series, image_folder: Path, require_image: bool) -> str | None:
    observations = str(row.get("observations", "")).strip()
    conclusion = str(row.get("conclusion", "")).strip()
    if not observations or not conclusion:
        return "missing observations or conclusion"

    image_name = str(row["file_path"]).strip()
    if not image_name:
        return "missing file_path"

    if require_image and not (image_folder / image_name).exists():
        return f"image not found: {image_name}"

    return None


def _row_to_llava(row: pd.Series, sample_id: str) -> dict:
    return {
        "id": sample_id,
        "image": str(row["file_path"]).strip(),
        "conversations": [
            {
                "from": "human",
                "value": build_user_prompt(row.get("history")),
            },
            {
                "from": "gpt",
                "value": build_assistant_response(
                    observations=row["observations"],
                    conclusion=row["conclusion"],
                ),
            },
        ],
    }


def prepare_llava_dataset(
    csv_path: str | Path,
    output_dir: str | Path,
    image_folder: str | Path,
    val_ratio: float = 0.20,
    seed: int = 42,
    max_samples: int | None = None,
    require_image: bool = True,
) -> dict[str, int]:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    image_folder = Path(image_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.head(max_samples)

    records: list[dict] = []
    skipped: dict[str, int] = {}

    for idx, row in df.iterrows():
        reason = _validate_row(row, image_folder, require_image=require_image)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        records.append(_row_to_llava(row, sample_id=f"cxr_{idx}"))

    if not records:
        raise RuntimeError("No valid samples after filtering. Check CSV paths and image folder.")

    split_df = pd.DataFrame({"idx": range(len(records))})
    split_df = split_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(round(len(split_df) * val_ratio)))
    val_indices = set(split_df.iloc[:n_val]["idx"].tolist())

    train_data = [records[i] for i in range(len(records)) if i not in val_indices]
    val_data = [records[i] for i in val_indices]

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    meta_path = output_dir / "meta.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    stats = {
        "source_rows": len(df),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "skipped_total": sum(skipped.values()),
        **{f"skipped_{k.replace(' ', '_')}": v for k, v in skipped.items()},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "csv_path": str(csv_path),
                "image_folder": str(image_folder),
                "val_ratio": val_ratio,
                "seed": seed,
                "stats": stats,
            },
            f,
            indent=2,
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LLaVA-format CXR dataset.")
    parser.add_argument(
        "--csv-path",
        default="training_data/csv/filtered_columns_cxr_0.3M.csv",
        help="Input CSV with observations, conclusion, history, file_path.",
    )
    parser.add_argument(
        "--image-folder",
        default=os.environ.get(
            "CXR_IMAGE_FOLDER",
            "/root/bionicsuite1/home/ai-user/chest_classifier/disk/chest_images",
        ),
        help="Directory containing CXR JPEG files referenced by file_path.",
    )
    parser.add_argument(
        "--output-dir",
        default="training_data/llava",
        help="Output directory for train.json and val.json.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick dry runs.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Keep rows even if image file is missing locally.",
    )
    args = parser.parse_args()

    stats = prepare_llava_dataset(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        image_folder=args.image_folder,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        require_image=not args.allow_missing_images,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
