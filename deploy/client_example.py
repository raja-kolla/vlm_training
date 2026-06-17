"""Example client for the deploy API."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import requests


def generate(
    prompt: str,
    *,
    image_path: str | None = None,
    image_base64: str | None = None,
    base_url: str = "http://localhost:8000",
    max_new_tokens: int = 1024,
) -> str:
    payload: dict = {"prompt": prompt, "max_new_tokens": max_new_tokens}
    if image_path:
        payload["image_path"] = image_path
    elif image_base64:
        payload["image_base64"] = image_base64
    else:
        raise ValueError("Provide image_path or image_base64")

    response = requests.post(f"{base_url}/v1/generate", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the CXR VLM deploy API")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image-path", required=True, help="Local path (sent as image_path or encoded as base64)")
    parser.add_argument("--use-base64", action="store_true", help="Send image as base64 instead of path")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    if args.use_base64:
        b64 = base64.b64encode(Path(args.image_path).read_bytes()).decode()
        text = generate(args.prompt, image_base64=b64, base_url=args.base_url, max_new_tokens=args.max_new_tokens)
    else:
        text = generate(args.prompt, image_path=args.image_path, base_url=args.base_url, max_new_tokens=args.max_new_tokens)

    print(text)


if __name__ == "__main__":
    main()
