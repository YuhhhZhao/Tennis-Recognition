from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tennis_robot_sim.config import load_config
from tennis_robot_sim.data import LandingPrediction, RobotState
from tennis_robot_sim.visualization import save_top_down


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate a top-down plot from saved simulation logs.")
    parser.add_argument("--input", required=True, help="Directory containing trajectory.csv and robot_path.csv.")
    parser.add_argument("--output", required=True, help="Output directory for replay plot.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"input directory does not exist: {input_dir}")
        return 2
    traj_path = input_dir / cfg["logging"]["trajectory_csv"]
    robot_path = input_dir / cfg["logging"]["robot_path_csv"]
    if not traj_path.exists() or not robot_path.exists():
        print(f"missing replay logs under {input_dir}")
        return 2
    traj_rows = _read_csv(traj_path)
    robot_rows = _read_csv(robot_path)
    ball_positions = [(float(r["ball_x_m"]), float(r["ball_y_m"]), float(r["ball_z_m"])) for r in traj_rows]
    robot_states = [RobotState(float(r["truth_x_m"]), float(r["truth_y_m"]), float(r["truth_yaw_rad"])) for r in robot_rows]
    pred = None
    for row in reversed(traj_rows):
        if row.get("pred_landing_x_m"):
            pred = LandingPrediction(
                landing_xy=(float(row["pred_landing_x_m"]), float(row["pred_landing_y_m"])),
                time_to_land=float(row["pred_time_to_land_s"]),
                uncertainty=0.0,
                confidence=1.0,
            )
            break
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_top_down(output_dir / "top_down_replay.png", cfg, ball_positions, robot_states, pred, None)
    print(f"Wrote {output_dir / 'top_down_replay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

