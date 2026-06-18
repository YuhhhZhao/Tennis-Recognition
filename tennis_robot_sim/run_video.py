from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from tennis_robot_sim.config import load_config
from tennis_robot_sim.logging_utils import ensure_output_dir, write_csv
from tennis_robot_sim.perception import ColorBallDetector, PixelTracker
from tennis_robot_sim.visualization import draw_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Process an optional real video through the safe detector/tracker path.")
    parser.add_argument("--input", required=False, help="Video file path. No hardware movement is performed.")
    parser.add_argument("--output", default="outputs/video_test")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if not args.input:
        parser.print_help()
        return 0
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"video file does not exist: {input_path}")
        return 2
    cfg = load_config(args.config)
    detector = ColorBallDetector(cfg)
    tracker = PixelTracker()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"cannot open video: {input_path}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg["camera"]["fps"]
    output_dir = ensure_output_dir(args.output)
    rows = []
    overlay_path = output_dir / "overlay.mp4"
    writer = None
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_id / fps
        det = detector.detect(frame, frame_id, timestamp)
        track = tracker.update(det, frame_id, timestamp)
        rows.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "center_x_px": "" if det is None else det.center_px[0],
            "center_y_px": "" if det is None else det.center_px[1],
            "radius_px": "" if det is None else det.radius_px,
            "confidence": "" if det is None else det.confidence,
        })
        overlay = draw_overlay(frame, det, track, None, None)
        if writer is None:
            h, w = overlay.shape[:2]
            writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if writer.isOpened():
            writer.write(overlay)
        frame_id += 1
    cap.release()
    if writer is not None:
        writer.release()
    write_csv(output_dir / "detections.csv", rows, ["frame_id", "timestamp", "center_x_px", "center_y_px", "radius_px", "confidence"])
    print(f"Wrote detections and overlay under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
