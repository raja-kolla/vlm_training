"""FastAPI server: image + prompt -> raw model output."""

from __future__ import annotations

import argparse
import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from deploy.config import (
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_HOST,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_NUM_REPLICAS,
    DEFAULT_PORT,
    IMAGE_MAX_PIXELS,
)
from deploy.inference import load_image_from_bytes, load_image_from_path
from deploy.model_pool import ModelPool

MODEL_POOL: ModelPool | None = None
MODEL_PATH = DEFAULT_MODEL_PATH
NUM_REPLICAS = DEFAULT_NUM_REPLICAS


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_POOL, MODEL_PATH, NUM_REPLICAS
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {MODEL_PATH}. "
            "Set MODEL_PATH to a directory containing model.safetensors."
        )
    MODEL_POOL = ModelPool(
        str(MODEL_PATH),
        NUM_REPLICAS,
        attn_implementation=DEFAULT_ATTN_IMPLEMENTATION,
    )
    yield


app = FastAPI(
    title="CXR VLM API",
    description="Image + prompt in, raw model text out",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt / instruction")
    image_path: str | None = Field(None, description="Path to image on the server")
    image_base64: str | None = Field(None, description="Base64-encoded image bytes")
    max_new_tokens: int = 8192

    @model_validator(mode="after")
    def check_image_input(self):
        has_path = bool(self.image_path and self.image_path.strip())
        has_b64 = bool(self.image_base64 and self.image_base64.strip())
        if has_path == has_b64:
            raise ValueError("Provide exactly one of image_path or image_base64")
        return self


@app.get("/health")
def health():
    stats = MODEL_POOL.stats() if MODEL_POOL is not None else {}
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_loaded": MODEL_POOL is not None,
        **stats,
    }


@app.post("/v1/generate")
def generate(body: GenerateRequest):
    if MODEL_POOL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if body.image_path:
            image = load_image_from_path(body.image_path)
        else:
            image = load_image_from_bytes(base64.b64decode(body.image_base64))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    text = MODEL_POOL.generate(
        image,
        body.prompt,
        max_new_tokens=body.max_new_tokens,
        max_pixels=IMAGE_MAX_PIXELS,
    )
    return {"text": text}


def main() -> None:
    global MODEL_PATH, NUM_REPLICAS
    parser = argparse.ArgumentParser(description="Deploy CXR VLM (phase-2 full SFT)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=DEFAULT_NUM_REPLICAS,
        help="Number of model copies to load for parallel inference",
    )
    args = parser.parse_args()

    MODEL_PATH = args.model_path.resolve()
    NUM_REPLICAS = args.num_replicas
    os.environ["MODEL_PATH"] = str(MODEL_PATH)
    os.environ["NUM_REPLICAS"] = str(NUM_REPLICAS)

    import uvicorn

    uvicorn.run(
        "deploy.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
        factory=False,
    )


if __name__ == "__main__":
    main()
