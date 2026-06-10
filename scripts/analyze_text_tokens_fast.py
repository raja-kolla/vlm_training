#!/usr/bin/env python3
"""Fast text-only token stats (no vision expansion) for all LLaVA records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

constants = _load("constants", SRC / "constants.py")
data_utils = _load("data_utils", SRC / "dataset" / "data_utils.py")

IM_START = constants.DEFAULT_IM_START_TOKEN
IM_END = constants.DEFAULT_IM_END_TOKEN
llava_to_openai = data_utils.llava_to_openai
format_assistant_response = data_utils.format_assistant_response


def main():
    data_path = ROOT / "training_data/llava/train.json"
    model_id = "Qwen/Qwen3.5-4B"
    print(f"Loading tokenizer for {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Loading {data_path}")
    with open(data_path) as f:
        records = json.load(f)

    user_lens, asst_lens, total_text = [], [], []
    for rec in records:
        conv = llava_to_openai(rec["conversations"], is_video=False)
        user = conv[0]["content"]
        asst = conv[1]["content"]
        user_ids = tok.encode(user, add_special_tokens=False)
        asst_ids = tok.encode(asst, add_special_tokens=False)
        user_lens.append(len(user_ids))
        asst_lens.append(len(asst_ids))
        total_text.append(len(user_ids) + len(asst_ids))

    u, a, t = np.array(user_lens), np.array(asst_lens), np.array(total_text)
    for name, arr in [("User prompt text (raw, incl <image>)", u), ("Assistant text", a), ("Combined raw text", t)]:
        p = np.percentile(arr, [5, 25, 50, 75, 95])
        print(
            f"{name}: n={len(arr):,} mean={arr.mean():.1f} min={arr.min()} max={arr.max()} "
            f"p5={p[0]:.0f} p25={p[1]:.0f} p50={p[2]:.0f} p75={p[3]:.0f} p95={p[4]:.0f}"
        )


if __name__ == "__main__":
    main()
