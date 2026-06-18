from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tennis_robot_sim.data import Detection, InterceptionPlan, LandingPrediction, RobotState, TrackState
from tennis_robot_sim.sim.render import save_video


def draw_overlay(
    frame: np.ndarray,
    detection: Optional[Detection],
    track: Optional[TrackState],
    prediction: Optional[LandingPrediction],
    plan: Optional[InterceptionPlan],
) -> np.ndarray:
    out = frame.copy()
    if detection is not None:
        center = (int(round(detection.center_px[0])), int(round(detection.center_px[1])))
        cv2.circle(out, center, max(2, int(round(detection.radius_px))), (0, 0, 255), 2)
    if track is not None and track.center_px is not None:
        center = (int(round(track.center_px[0])), int(round(track.center_px[1])))
        cv2.drawMarker(out, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
    if prediction is not None:
        cv2.putText(
            out,
            f"landing=({prediction.landing_xy[0]:.2f},{prediction.landing_xy[1]:.2f}) t={prediction.time_to_land:.2f}s",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if plan is not None:
        status = "reachable" if plan.reachable else "best effort"
        cv2.putText(out, status, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_overlay_artifacts(output_dir: Path, cfg: dict, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    logging_cfg = cfg["logging"]
    cv2.imwrite(str(output_dir / logging_cfg["overlay_image"]), frames[-1])
    if cfg["sim"].get("save_overlay_video", True):
        save_video(output_dir / logging_cfg["overlay_video"], frames, int(cfg["camera"]["fps"]))


def save_top_down(
    path: str | Path,
    cfg: dict,
    ball_positions: list[tuple[float, float, float]],
    robot_states: list[RobotState],
    prediction: Optional[LandingPrediction],
    plan: Optional[InterceptionPlan],
    truth_landing_xy: Optional[tuple[float, float]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    court = cfg["court"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(court["x_min"] - 0.5, court["x_max"] + 0.5)
    ax.set_ylim(court["y_min"] - 0.5, court["y_max"] + 0.5)
    ax.set_xlabel("world x forward (m)")
    ax.set_ylabel("world y left (m)")
    ax.grid(True, alpha=0.25)
    rect_x = [court["x_min"], court["x_max"], court["x_max"], court["x_min"], court["x_min"]]
    rect_y = [court["y_min"], court["y_min"], court["y_max"], court["y_max"], court["y_min"]]
    ax.plot(rect_x, rect_y, color="white", linewidth=1.5)
    if ball_positions:
        arr = np.array(ball_positions)
        ax.plot(arr[:, 0], arr[:, 1], color="#d6d928", linewidth=2, label="ball truth")
    if robot_states:
        xs = [s.x for s in robot_states]
        ys = [s.y for s in robot_states]
        ax.plot(xs, ys, color="#1f77b4", linewidth=2, label="robot")
        ax.scatter(xs[-1], ys[-1], color="#1f77b4", s=35)
    if truth_landing_xy is not None:
        ax.scatter([truth_landing_xy[0]], [truth_landing_xy[1]], color="green", marker="x", s=70, label="truth landing")
    if prediction is not None:
        ax.scatter([prediction.landing_xy[0]], [prediction.landing_xy[1]], color="red", marker="o", s=45, label="predicted landing")
    if plan is not None:
        ax.scatter([plan.target_pose.x], [plan.target_pose.y], color="black", marker="*", s=90, label="robot target")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
