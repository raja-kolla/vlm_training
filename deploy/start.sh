#!/usr/bin/env bash
# Start the CXR VLM API server with the phase-2 full SFT checkpoint.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"
export MODEL_PATH="${MODEL_PATH:-/home/raja/qwen_ckpt/phase2_full_sft}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export NUM_REPLICAS="${NUM_REPLICAS:-1}"

CONDA_ENV="${CONDA_ENV:-/home/raja/conda_envs/dep}"

activate_python_env() {
  if [[ -n "${CONDA_ENV:-}" && -x "${CONDA_ENV}/bin/python" ]]; then
    # Prefer the conda env on /home (torch already installed; root disk may be full).
    # shellcheck disable=SC1091
    if command -v conda >/dev/null 2>&1; then
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV"
    else
      export PATH="${CONDA_ENV}/bin:${PATH}"
    fi
    return
  fi

  if [[ -d .venv ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    return
  fi
}

activate_python_env

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: torch is not installed in the active Python environment." >&2
  echo "Use the conda env (recommended): CONDA_ENV=/home/raja/conda_envs/dep bash deploy/start.sh" >&2
  echo "Or install deps on a disk with free space (root / is often full):" >&2
  echo "  pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128" >&2
  echo "  pip install -r requirements.txt -r deploy/requirements.txt" >&2
  exit 1
fi

if ! python -c "import fastapi" 2>/dev/null; then
  echo "Installing deploy requirements..."
  pip install -r deploy/requirements.txt
fi

exec python -m deploy.server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --num-replicas "$NUM_REPLICAS"
