#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/validation

echo "[1/4] Checking Python environment"
python scripts/check_env.py > outputs/validation/check_env.json

echo "[2/4] Running unit tests"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q

echo "[3/4] Running smoke simulation"
python -m tennis_robot_sim.run_sim --scenario default --no-hardware --output outputs/validation/smoke

echo "[4/4] Replaying saved logs"
python -m tennis_robot_sim.tools.replay --input outputs/validation/smoke --output outputs/validation/replay

echo "Validation complete. Artifacts are under outputs/validation/"
