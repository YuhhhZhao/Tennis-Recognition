"""
小车 UART 运动测试脚本 (Jetson)
- 连接 /dev/ttyTHS1，等待 RDY
- 发送 VEL 指令让小车前进一小段距离
- 发送 STOP 停车
- 安全：低速、短时、Ctrl+C 立即 STOP
"""

import serial
import time
import sys

PORT = "/dev/ttyCH341USB0"
BAUD = 115200

# 安全测试参数
TEST_VX = 0.2      # 前进速度 m/s (低速)
TEST_VY = 0.0      # 横向速度
TEST_W  = 0.0      # 旋转角速度
DURATION = 1.0     # 运动时长 (秒)
INTERVAL = 0.1     # 指令发送间隔 (秒, 远小于 1000ms 看门狗)


def read_all(ser, timeout=0.3):
    """读取串口所有可用数据"""
    lines = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode("ascii", errors="replace").strip()
            if line:
                lines.append(line)
        else:
            time.sleep(0.01)
    return lines


def send_cmd(ser, cmd: str):
    """发送指令并读取应答"""
    ser.write((cmd + "\n").encode("ascii"))
    ser.flush()


def main():
    ser = None
    try:
        # 1. 打开串口
        print(f"[1/5] 打开串口 {PORT} ...")
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
        ser.dtr = False
        ser.rts = False
        print("      串口已打开")

        # 2. 等待 ESP32 RDY
        print("[2/5] 等待 ESP32 就绪 (RDY) ...")
        deadline = time.time() + 5.0
        ready = False
        while time.time() < deadline:
            if ser.in_waiting:
                line = ser.readline().decode("ascii", errors="replace").strip()
                if line:
                    print(f"      [MCU] {line}")
                if "RDY" in line:
                    ready = True
                    break
            else:
                time.sleep(0.05)

        if not ready:
            # 可能 ESP32 已经启动过了，尝试发 STAT
            print("      未收到 RDY，尝试发 STAT 检测...")
            send_cmd(ser, "STAT")
            time.sleep(0.3)
            for line in read_all(ser, 1.0):
                print(f"      [MCU] {line}")
                if "STAT" in line or "FW:" in line:
                    ready = True
                    break

        if not ready:
            print("[!] 无法检测到 ESP32，但继续尝试...")
        else:
            print("[+] ESP32 就绪")

        # 3. 发 STAT 看状态
        print("[3/5] 查询状态 (STAT) ...")
        send_cmd(ser, "STAT")
        time.sleep(0.3)
        for line in read_all(ser, 1.0):
            print(f"      [MCU] {line}")

        # 4. 持续发送 VEL 让小车前进
        print(f"\n[4/5] 发送 VEL 指令: vx={TEST_VX}, vy={TEST_VY}, w={TEST_W}")
        print(f"      持续 {DURATION}s, 间隔 {INTERVAL}s ...")
        print("      (按 Ctrl+C 紧急停止)")
        print()

        deadline = time.time() + DURATION
        cmd_count = 0
        while time.time() < deadline:
            loop_start = time.time()

            cmd = f"VEL {TEST_VX:.3f} {TEST_VY:.3f} {TEST_W:.3f}"
            send_cmd(ser, cmd)
            cmd_count += 1

            # 读取应答
            for line in read_all(ser, 0.05):
                if line:
                    print(f"      [{cmd_count}] {line}")

            # 控制发送间隔
            elapsed = time.time() - loop_start
            if elapsed < INTERVAL:
                time.sleep(INTERVAL - elapsed)

        # 5. STOP
        print(f"\n[5/5] 停止 (STOP x5) ...")
        for i in range(5):
            send_cmd(ser, "STOP")
            time.sleep(0.05)

        time.sleep(0.3)
        for line in read_all(ser, 1.0):
            print(f"      [MCU] {line}")

        print(f"\n[+] 完成。共发送 {cmd_count} 条 VEL 指令")

    except KeyboardInterrupt:
        print("\n[!] 中断！立即停车...")
        if ser and ser.is_open:
            for _ in range(10):
                send_cmd(ser, "STOP")
                time.sleep(0.05)
            time.sleep(0.3)
            for line in read_all(ser, 1.0):
                print(f"      [MCU] {line}")

    except serial.SerialException as e:
        print(f"\n[!] 串口错误: {e}")
        print(f"    请检查: 1) {PORT} 是否存在  2) 是否有权限 (sudo chmod 666 {PORT})")

    except Exception as e:
        print(f"\n[!] 异常: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if ser and ser.is_open:
            ser.close()
            print("[*] 串口已关闭")


if __name__ == "__main__":
    main()
