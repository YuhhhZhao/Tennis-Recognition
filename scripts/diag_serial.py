"""
串口诊断脚本 - 排查 ESP32 通信问题
逐项检查: 端口数据 / RDY / STAT / VEL 响应
"""
import serial
import time
import sys

PORT = "/dev/ttyTHS1"
BAUD = 115200


def read_and_print(ser, label, timeout=2.0):
    """读取串口数据并打印"""
    print(f"\n--- {label} (等待 {timeout}s) ---")
    deadline = time.time() + timeout
    found = False
    while time.time() < deadline:
        if ser.in_waiting:
            data = ser.read(ser.in_waiting)
            text = data.decode("ascii", errors="replace")
            print(f"  RAW: {repr(text)}")
            found = True
        else:
            time.sleep(0.05)
    if not found:
        print("  (无数据)")
    return found


def main():
    ser = None
    try:
        print("=" * 50)
        print("ESP32 串口诊断")
        print("=" * 50)

        # 1. 打开串口
        print(f"\n[1] 打开 {PORT} @ {BAUD}")
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        print("    串口已打开")
        print(f"    配置: {ser}")

        # 2. 看看有没有残留数据
        time.sleep(0.5)
        read_and_print(ser, "[2] 缓冲区残留数据", timeout=0.5)

        # 3. 发 STAT 看是否有回应
        print("\n[3] 发送 STAT ...")
        ser.write(b"STAT\n")
        ser.flush()
        time.sleep(0.3)
        read_and_print(ser, "STAT 回应", timeout=1.0)

        # 4. 发 PING 测试
        print("\n[4] 发送 PING ...")
        ser.write(b"PING\n")
        ser.flush()
        time.sleep(0.3)
        read_and_print(ser, "PING 回应", timeout=1.0)

        # 5. 发 VEL 指令
        print("\n[5] 发送 VEL 0.2 0.0 0.0 ...")
        ser.write(b"VEL 0.200 0.000 0.000\n")
        ser.flush()
        time.sleep(0.3)
        read_and_print(ser, "VEL 回应", timeout=1.0)

        # 6. 再发一次 STAT
        print("\n[6] 再查 STAT ...")
        ser.write(b"STAT\n")
        ser.flush()
        time.sleep(0.3)
        read_and_print(ser, "STAT 回应", timeout=1.0)

        # 7. STOP
        print("\n[7] STOP ...")
        ser.write(b"STOP\n")
        ser.flush()
        time.sleep(0.3)
        read_and_print(ser, "STOP 回应", timeout=1.0)

        print("\n" + "=" * 50)
        print("诊断完成")

    except serial.SerialException as e:
        print(f"\n串口错误: {e}")
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭")


if __name__ == "__main__":
    main()
