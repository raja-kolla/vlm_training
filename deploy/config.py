"""Deployment configuration for the phase-2 CXR VLM checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", "/home/raja/qwen_ckpt/phase2_full_sft")
)
DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
DEFAULT_ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
DEFAULT_NUM_REPLICAS = int(os.environ.get("NUM_REPLICAS", "1"))
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))

# Qwen3.5 patch size is 16; cap longest side at 2048 px (same as training).
IMAGE_MAX_PIXELS = int(os.environ.get("IMAGE_MAX_PIXELS", str(128 * 16 * 128 * 16)))
