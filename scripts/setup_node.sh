#!/usr/bin/env bash
# One-time environment setup on a dedicated GPU node (e.g. 4x H100).
# Creates .venv and installs the training stack. No conda, no Modal.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3.11}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3

echo "==> Checking GPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || {
  echo "nvidia-smi failed — are NVIDIA drivers installed?" >&2
  exit 1
}

echo "==> Creating venv (.venv) with $PYTHON"
[[ -d .venv ]] || "$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "==> Installing torch (cu128) + training requirements"
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

echo "==> Sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
import deepspeed, transformers
print("deepspeed", deepspeed.__version__, "| transformers", transformers.__version__)
PY

cat <<'EOF'

Setup complete. Next steps:

  source .venv/bin/activate
  wandb login                                # optional, for W&B metrics
  export IMAGE_FOLDER=/path/to/chest_images  # where the JPEGs live

  python scripts/train_local.py prepare_data
  python scripts/train_local.py download_model
  nohup python scripts/train_local.py train_phase1 > phase1.log 2>&1 &   # or run inside tmux
  python scripts/train_local.py train_phase2

EOF
