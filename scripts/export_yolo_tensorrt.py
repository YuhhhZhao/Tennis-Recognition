#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights to TensorRT.")
    parser.add_argument("--weights", required=True, help="Path to .pt weights.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Use FP16 export.")
    parser.add_argument("--int8", action="store_true", help="Use INT8 export.")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        device=0,
    )


if __name__ == "__main__":
    main()

