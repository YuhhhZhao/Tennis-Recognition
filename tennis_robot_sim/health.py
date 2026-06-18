from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _import_status(name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"available": False, "error": str(exc)}


def _command_status(cmd: list[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"returncode": None, "error": str(exc)}


def health_check() -> Dict[str, Any]:
    packages = {
        "cv2": _import_status("cv2"),
        "numpy": _import_status("numpy"),
        "yaml": _import_status("yaml"),
        "matplotlib": _import_status("matplotlib"),
        "scipy": _import_status("scipy"),
        "torch": _import_status("torch"),
        "tennis_robot_sim": {"available": True, "version": "local"},
    }
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "nvidia_smi": _command_status(["nvidia-smi"]),
    }
