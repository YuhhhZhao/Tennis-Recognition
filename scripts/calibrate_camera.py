#!/usr/bin/env python3
"""交互式相机标定脚本.

用法:
  python scripts/calibrate_camera.py --source 1 --pattern 9x6 --square 0.025

操作:
  - 将棋盘格置于相机视野中不同位置/角度
  - 看到彩色棋盘格 overlay 时按 空格 保存这一帧
  - 按 q 退出并用已采集的图片计算标定参数
  - 结果保存到 configs/calibration.npz

提示: 没有棋盘格? 运行:
  python -c "from tennis_tracker.prediction import generate_chessboard_png; generate_chessboard_png('chessboard.png')"
  然后用 A4 纸打印, 贴在平板上.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_tracker.prediction import calibrate_from_images, save_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive camera calibration.")
    parser.add_argument("--source", default="1", help="Camera index or video path.")
    parser.add_argument(
        "--pattern", default="9x6", help="Chessboard inner corners (cols x rows)."
    )
    parser.add_argument(
        "--square", type=float, default=0.025, help="Square size in meters."
    )
    parser.add_argument(
        "--output", default="configs/calibration.npz", help="Output .npz path."
    )
    return parser.parse_args()


def parse_source(source: str):
    return int(source) if source.isdigit() else str(Path(source))


def main() -> None:
    args = parse_args()
    cols, rows = map(int, args.pattern.split("x"))

    cap = cv2.VideoCapture(parse_source(args.source), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    saved: list[np.ndarray] = []
    window = "calibration"
    cv2.namedWindow(window)

    print(f"棋盘格模式: {cols}x{rows} 内角点, 方格 {args.square}m")
    print("空格 = 保存当前帧,  q = 完成标定")
    print()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if found:
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            cv2.putText(
                display,
                f"FOUND — saved={len(saved)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                display,
                f"no pattern — saved={len(saved)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" ") and found:
            saved.append(frame.copy())
            print(f"  [{len(saved)}] 已保存一帧")

    cap.release()
    cv2.destroyAllWindows()

    if len(saved) < 10:
        print(f"仅采集了 {len(saved)} 张, 至少需要 10 张才能标定. 退出.")
        return

    print(f"\n正在用 {len(saved)} 张图片标定...")
    calib = calibrate_from_images(saved, (cols, rows), args.square)

    if calib is None:
        print("标定失败! 请检查棋盘格模式参数是否正确.")
        return

    output_path = PROJECT_ROOT / args.output
    save_calibration(output_path, calib)
    print(f"标定成功! 保存到: {output_path}")
    print(f"  fx={calib.fx:.1f}  fy={calib.fy:.1f}  cx={calib.cx:.1f}  cy={calib.cy:.1f}")
    print(f"  畸变系数: {calib.dist_coeffs.ravel()}")


if __name__ == "__main__":
    main()
