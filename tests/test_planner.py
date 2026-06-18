from __future__ import annotations

from tennis_robot_sim.config import load_config
from tennis_robot_sim.data import LandingPrediction, RobotState
from tennis_robot_sim.robot.planner import InterceptionPlanner


def test_planner_reachable_target():
    cfg = load_config()
    plan = InterceptionPlanner(cfg).plan(LandingPrediction((1.0, 0.2), 2.0, 0.1, 1.0), RobotState(0.0, 0.0, 0.0))
    assert plan.reachable
    assert plan.target_pose.x == 1.0


def test_planner_unreachable_best_effort_and_clipping():
    cfg = load_config()
    plan = InterceptionPlanner(cfg).plan(LandingPrediction((50.0, 50.0), 0.2, 0.1, 1.0), RobotState(0.0, 0.0, 0.0))
    assert not plan.reachable
    assert plan.distance > 0
    assert "best effort" in plan.reason

