from __future__ import annotations

import json
import math
from argparse import Namespace

from tennis_robot_sim.run_sim import run_closed_loop


def test_end_to_end_sim_outputs(tmp_path):
    out = tmp_path / "smoke"
    metrics = run_closed_loop(Namespace(scenario="default", config=None, output=str(out), seed=42, frames=70, no_hardware=True, hardware=False))
    for name in ["metrics.json", "trajectory.csv", "robot_path.csv", "top_down.png", "overlay_last.png"]:
        assert (out / name).exists(), name
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["predicted_landing_xy"] is not None
    assert all(math.isfinite(v) for v in saved["predicted_landing_xy"])
    assert saved["landing_error_m"] < 0.5
    final_distance = saved["robot_final_distance_to_target_m"]
    assert final_distance is not None
    assert final_distance < 0.8 or saved["planner_reachable"] is False
    assert metrics["hardware_enabled"] is False
