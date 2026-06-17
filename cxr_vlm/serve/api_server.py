"""FastAPI server for CXR report generation."""

from __future__ import annotations

import argparse
import base64
import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from cxr_vlm.inference import generate_report_from_loaded, load_model_and_processor

MODEL = None
PROCESSOR = None
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/raja/qwen_ckpt/phase2_1_full_sft")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, PROCESSOR, MODEL_PATH
    MODEL, PROCESSOR = load_model_and_processor(MODEL_PATH, attn_implementation="sdpa")
    yield


app = FastAPI(
    title="CXR VLM API",
    description="Chest X-ray report generation API",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded chest X-ray image")
    history: str | None = Field(None, description="Optional clinical history")
    max_new_tokens: int = Field(1024, ge=1, le=4096)


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/v1/generate")
async def generate(
    image: UploadFile = File(...),
    history: str | None = Form(None),
    max_new_tokens: int = Form(1024),
):
    if MODEL is None or PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        image_path = tmp.name

    try:
        result = generate_report_from_loaded(
            MODEL,
            PROCESSOR,
            image_path,
            history=history,
            max_new_tokens=max_new_tokens,
        )
    finally:
        Path(image_path).unlink(missing_ok=True)

    return result


@app.post("/v1/generate/json")
def generate_json(body: GenerateRequest):
    if MODEL is None or PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image_bytes = base64.b64decode(body.image_base64)
        from PIL import Image

        Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        image_path = tmp.name

    try:
        result = generate_report_from_loaded(
            MODEL,
            PROCESSOR,
            image_path,
            history=body.history,
            max_new_tokens=body.max_new_tokens,
        )
    finally:
        Path(image_path).unlink(missing_ok=True)

    return result


def main() -> None:
    global MODEL_PATH
    parser = argparse.ArgumentParser(description="Serve CXR VLM API")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    os.environ["MODEL_PATH"] = MODEL_PATH

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
