from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tennis_robot_sim.config import load_config
from tennis_robot_sim.sim import PinholeCamera, make_scenario
from tennis_robot_sim.sim.render import render_frame, save_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a synthetic tennis ball sequence.")
    parser.add_argument("--scenario", default="default", choices=["default", "high_lob", "fast_cross", "noisy_camera"])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", default="outputs/synthetic/default.mp4")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["sim"]["frames"] = args.frames
    scenario = make_scenario(args.scenario, cfg=cfg)
    camera = PinholeCamera.from_config(cfg)
    rng = np.random.default_rng(int(cfg["sim"]["seed"]))
    frames = [render_frame(camera, state.position, cfg, rng) for state in scenario.states]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = save_video(output, frames, int(cfg["camera"]["fps"]))
    if not ok and frames:
        fallback = output.with_suffix(".png")
        cv2.imwrite(str(fallback), frames[-1])
        print(f"Video writer unavailable; wrote fallback image {fallback}")
    else:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

