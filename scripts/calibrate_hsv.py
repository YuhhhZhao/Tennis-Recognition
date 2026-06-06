#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2


def parse_source(source: str) -> Any:
    return int(source) if source.isdigit() else str(Path(source))


def nothing(_: int) -> None:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive HSV calibration tool.")
    parser.add_argument("--source", default="1")
    args = parser.parse_args()

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    window = "hsv-calibration"
    cv2.namedWindow(window)
    defaults = {
        "H low": 25,
        "S low": 70,
        "V low": 70,
        "H high": 45,
        "S high": 255,
        "V high": 255,
    }
    for name, value in defaults.items():
        max_value = 179 if name.startswith("H") else 255
        cv2.createTrackbar(name, window, value, max_value, nothing)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low = [
            cv2.getTrackbarPos("H low", window),
            cv2.getTrackbarPos("S low", window),
            cv2.getTrackbarPos("V low", window),
        ]
        high = [
            cv2.getTrackbarPos("H high", window),
            cv2.getTrackbarPos("S high", window),
            cv2.getTrackbarPos("V high", window),
        ]
        mask = cv2.inRange(hsv, tuple(low), tuple(high))
        preview = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.putText(
            preview,
            f"lower={low} upper={high}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window, preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"lower: {low}")
            print(f"upper: {high}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

