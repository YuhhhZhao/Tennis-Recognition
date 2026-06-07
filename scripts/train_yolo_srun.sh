#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_ENV="/mnt/hwfile/zhaoyuhong/miniforge3/envs/tennis/bin/yolo"

cd "$PROJECT_ROOT"

srun \
  --partition=Omnilab \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --quotatype=reserved \
  --job-name test \
  "$PY_ENV" detect train \
    model=weights/yolov8n.pt \
    data=data/datasets/tennis_ball/tennis_ball.yaml \
    imgsz=640 \
    epochs=80 \
    batch=64 \
    device=0 \
    workers=8 \
    project="$PROJECT_ROOT/runs/detect" \
    name=tennis_ball_yolov8n_pretrained \
    exist_ok=True
