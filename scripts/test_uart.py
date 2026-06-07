#!/usr/bin/env python3
"""Jetson ↔ ESP32 UART 调试工具 — 无需相机即可测试电控

用法:
  # 交互模式 (手动发指令)
  python scripts/test_uart.py --port /dev/ttyTHS1 --mode interactive

  # 自动化测试序列
  python scripts/test_uart.py --port /dev/ttyTHS1 --mode auto

  # 遥测监视 (持续监听 ESP32 TELEM)
  python scripts/test_uart.py --port /dev/ttyTHS1 --mode monitor

  # 默认: 使用虚拟模式 (不连硬件, 模拟测试协议)
  python scripts/test_uart.py --mode virtual
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# ---- 可选 pyserial ----------------------------------------------------------
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


def parse_args():
    p = argparse.ArgumentParser(description="ESP32 UART debug tool")
    p.add_argument("--port", default="/dev/ttyTHS1", help="Serial port")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument(
        "--mode", choices=["interactive", "auto", "monitor", "virtual"],
        default="virtual",
    )
    return p.parse_args()


# =============================================================================
# Virtual ESP32 (无硬件时模拟)
# =============================================================================

import math

class VirtualESP32:
    """模拟 ESP32 响应, 用于无硬件时调试 Jetson 端代码."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.has_target = False
        self.start_t = time.monotonic()
        self.KP = 2.0
        self.TOLERANCE = 0.05
        self.MAX_SPEED = 1.5

    def handle(self, line: str) -> Optional[str]:
        line = line.strip()
        if not line:
            return None

        if line.startswith("TARGET"):
            parts = line.split()
            if len(parts) >= 3:
                self.target_x = float(parts[1])
                self.target_y = float(parts[2])
                self.has_target = True
                self.x = 0.0
                self.y = 0.0
                self.start_t = time.monotonic()
                return "OK"

        if line == "STOP":
            self.has_target = False
            return "OK"

        if line == "PING":
            dx = self.target_x - self.x if self.has_target else 0
            dy = self.target_y - self.y if self.has_target else 0
            return f"PONG x={self.x:.3f} y={self.y:.3f} err_x={dx:.3f} err_y={dy:.3f}"

        if line.startswith("KPX") or line.startswith("KPY") or \
           line.startswith("KVX") or line.startswith("KIX"):
            return "OK"

        if line == "DEBUG 1":
            return "DEBUG ON"

        if line == "DEBUG 0":
            return "DEBUG OFF"

        return f"ERR unknown: {line}"

    def step(self, dt: float) -> Optional[str]:
        """模拟小车运动, 到达目标时返回 'DONE'."""
        if not self.has_target:
            return None
        err_x = self.target_x - self.x
        err_y = self.target_y - self.y
        dist = math.hypot(err_x, err_y)
        if dist < self.TOLERANCE:
            self.has_target = False
            self.x = self.target_x
            self.y = self.target_y
            return "DONE"
        speed = min(self.MAX_SPEED, self.KP * dist)
        if dist > 0:
            self.x += (err_x / dist) * speed * dt
            self.y += (err_y / dist) * speed * dt
        return None


# =============================================================================
# Real UART
# =============================================================================

def open_serial(port: str, baud: int):
    if not HAS_SERIAL:
        print("pyserial 未安装: pip install pyserial")
        return None
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        print(f"串口 {port} 已打开, 等待 ESP32 RDY...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if "RDY" in line:
                print(f"ESP32 就绪: {line}")
                return ser
        print("未收到 RDY, 继续尝试...")
        return ser
    except (OSError, serial.SerialException) as e:
        print(f"无法打开串口 {port}: {e}")
        return None


# =============================================================================
# Command-line interface
# =============================================================================

def interactive_mode(ser=None, virt=None):
    """交互式命令 REPL."""
    print("\n=== UART 交互模式 ===")
    print("命令:")
    print("  TARGET <x> <y> [t]   — 发送落点")
    print("  STOP                 — 紧急停车")
    print("  PING                 — 查询状态")
    print("  KPX <val>            — 调位置 P 增益")
    print("  KVX <val>            — 调速度 P 增益")
    print("  DEBUG 1              — 开遥测")
    print("  Q                    — 退出")
    print()

    while True:
        try:
            cmd = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd.upper() == "Q":
            break
        if not cmd:
            continue

        # Send
        if ser:
            ser.write((cmd + "\n").encode())
            ser.flush()
            time.sleep(0.1)
            # Read response
            while ser.in_waiting:
                resp = ser.readline().decode("ascii", errors="ignore").strip()
                if resp:
                    print(f"  ← {resp}")

        if virt:
            resp = virt.handle(cmd)
            if resp:
                print(f"  ← [VIRT] {resp}")
            # 驱动虚拟小车直到到达目标 (最多 3 秒)
            for _ in range(120):
                if not virt.has_target:
                    break
                result = virt.step(0.025)
                if result == "DONE":
                    print(f"  ← [VIRT] DONE (reached target)")
                    break

    print("退出.")


def auto_test(ser=None, virt=None):
    """自动测试序列."""
    print("\n=== 自动测试序列 ===")

    tests = [
        ("TARGET 0.5 0.0 2.0", "前进 0.5m"),
        ("PING", "查询状态"),
        ("TARGET 0.0 0.3 1.5", "左移 0.3m"),
        ("PING", "查询状态"),
        ("STOP", "停车"),
    ]

    for cmd, desc in tests:
        print(f"\n[{desc}] → {cmd}")
        if ser:
            ser.write((cmd + "\n").encode())
            ser.flush()
            time.sleep(0.3)
            while ser.in_waiting:
                resp = ser.readline().decode("ascii", errors="ignore").strip()
                if resp:
                    print(f"  ← {resp}")
        if virt:
            resp = virt.handle(cmd)
            if resp:
                print(f"  ← [VIRT] {resp}")
            # 模拟小车运动
            for _ in range(60):
                if not virt.has_target:
                    break
                result = virt.step(0.025)
                if result == "DONE":
                    print(f"  ← [VIRT] DONE")
                    break

    if virt:
        vx, vy = virt.x, virt.y
        print(f"\n虚拟小车最终位置: x={vx:.3f} y={vy:.3f}")

    print("\n自动测试完成.")


def monitor_mode(ser):
    """持续监听 ESP32 遥测."""
    if not ser:
        print("需要物理串口连接")
        return
    print("\n=== 遥测监视 (Ctrl+C 退出) ===")
    ser.write(b"DEBUG 1\n")
    ser.flush()
    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if line:
                    print(f"[{time.monotonic():.1f}] {line}")
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        ser.write(b"DEBUG 0\n")
        ser.flush()
    print("\n监视结束.")


# =============================================================================
# main
# =============================================================================

def main():
    args = parse_args()

    ser = None
    virt = None

    if args.mode == "virtual":
        print("=== 虚拟模式 (无硬件) ===")
        virt = VirtualESP32()
        interactive_mode(virt=virt)

    elif args.mode == "interactive":
        ser = open_serial(args.port, args.baud)
        interactive_mode(ser=ser)

    elif args.mode == "auto":
        ser = open_serial(args.port, args.baud)
        if ser is None and not HAS_SERIAL:
            virt = VirtualESP32()
            print("(使用虚拟 ESP32)")
        auto_test(ser=ser, virt=virt)

    elif args.mode == "monitor":
        ser = open_serial(args.port, args.baud)
        monitor_mode(ser)

    if ser:
        ser.close()


if __name__ == "__main__":
    main()
