#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${TENNIS_ROBOT_SIM_ENV:-tennis-robot-sim}"
PYTHON_VERSION="${TENNIS_ROBOT_SIM_PYTHON:-3.10}"

cd "$(dirname "$0")/.."

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi
  conda activate "$ENV_NAME"
else
  if [ ! -d ".venv" ]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if [ -f pyproject.toml ] || [ -f setup.py ]; then
  python -m pip install -e .
fi

python scripts/check_env.py

