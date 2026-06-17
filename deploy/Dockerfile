FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/model \
    HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONPATH=/app/src:/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt deploy/requirements.txt ./
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install \
        torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python3.11 -m pip install -r requirements.txt -r deploy/requirements.txt

COPY src/ src/
COPY cxr_vlm/ cxr_vlm/
COPY deploy/ deploy/

EXPOSE 8000

# Mount your checkpoint at /model, e.g.:
#   docker run --gpus all -p 8000:8000 -v /home/raja/qwen_ckpt/phase2_full_sft:/model:ro cxr-vlm
CMD ["python3.11", "-m", "deploy.server", "--model-path", "/model"]
