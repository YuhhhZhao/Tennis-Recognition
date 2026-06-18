from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from tennis_robot_sim.data import ControlCommand, RobotState
from tennis_robot_sim.robot.sim_robot import SimRobot


class RobotInterface(ABC):
    @abstractmethod
    def send(self, command: ControlCommand, dt: float) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def emergency_stop(self) -> None:
        raise NotImplementedError


class SimRobotInterface(RobotInterface):
    def __init__(self, robot: SimRobot):
        self.robot = robot
        self.last_command = ControlCommand(0.0, 0.0)

    def send(self, command: ControlCommand, dt: float) -> RobotState:
        self.last_command = command
        return self.robot.step(command, dt)

    def emergency_stop(self) -> None:
        self.last_command = ControlCommand(0.0, 0.0)
        self.robot.step(self.last_command, 0.0)


class RealRobotInterface(RobotInterface):
    def __init__(self, cfg: dict, port: Optional[str] = None):
        safety = cfg.get("safety", {})
        if not safety.get("enable_real_hardware", False):
            raise RuntimeError("real hardware is disabled; set safety.enable_real_hardware=true and pass --hardware")
        self.cfg = cfg
        self.port = port
        self.last_command = ControlCommand(0.0, 0.0)

    def send(self, command: ControlCommand, dt: float) -> RobotState:
        del dt
        self.last_command = command
        if self.cfg.get("safety", {}).get("dry_run", True):
            print(f"[DRY-RUN hardware] v={command.v:.3f} omega={command.omega:.3f}")
            return RobotState(0.0, 0.0, 0.0)
        raise RuntimeError("real command transport is not implemented in simulation package")

    def emergency_stop(self) -> None:
        self.last_command = ControlCommand(0.0, 0.0)
        print("[DRY-RUN hardware] emergency stop")
