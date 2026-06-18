#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def import_status(name: str, required: bool = True) -> dict:
    try:
        module = importlib.import_module(name)
        return {"name": name, "required": required, "ok": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"name": name, "required": required, "ok": False, "error": str(exc)}


def command_status(cmd: list[str]) -> dict:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
        return {"cmd": cmd, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "error": str(exc)}


def main() -> int:
    checks = [
        import_status("cv2"),
        import_status("numpy"),
        import_status("yaml"),
        import_status("matplotlib"),
        import_status("scipy", required=False),
        import_status("torch", required=False),
        import_status("tennis_tracker", required=False),
        import_status("tennis_robot_sim"),
    ]
    from tennis_robot_sim.health import health_check

    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "imports": checks,
        "conda": command_status(["conda", "--version"]),
        "pip": command_status([sys.executable, "-m", "pip", "--version"]),
        "nvidia_smi": command_status(["nvidia-smi"]),
        "health_check": health_check(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    failed_required = [item for item in checks if item["required"] and not item["ok"]]
    if failed_required:
        print("FAILED required imports: " + ", ".join(item["name"] for item in failed_required))
        return 1
    print("OK core simulation environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
