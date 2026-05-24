#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ball_tracker.config import load_config
from ball_tracker.pipeline import TrackerPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO warm-start + HSV tracker.")
    parser.add_argument("--config", default="configs/tracker.yaml")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--yolo-model", help="Override YOLO model path.")
    parser.add_argument("--no-yolo", action="store_true", help="Use full-frame HSV init.")
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument("--control", action="store_true", help="Enable control output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    cfg = load_config(config_path)
    if args.yolo_model:
        cfg.yolo.model_path = args.yolo_model
    if cfg.yolo.enabled:
        model_path = Path(cfg.yolo.model_path)
        if not model_path.is_absolute():
            cfg.yolo.model_path = str(PROJECT_ROOT / model_path)
    if args.no_yolo:
        cfg.yolo.enabled = False
    if args.headless:
        cfg.display.enabled = False
    if args.control:
        cfg.control.enabled = True
    TrackerPipeline(cfg, source=args.source).run()


if __name__ == "__main__":
    main()
