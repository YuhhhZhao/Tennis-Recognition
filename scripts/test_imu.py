"""
WitMotion IMU 功能验证脚本
- 读取加速度、角速度、姿态角、磁场
- 实时显示数据变化（晃动 IMU 观察）
- 按 Ctrl+C 退出
"""

import serial
import struct
import time
import sys

PORT = "/dev/ttyACM0"
BAUD = 115200

# WitMotion 0x55 协议解析
SCALE_ANGLE = 180.0 / 32768.0   # 角度缩放
SCALE_ACCEL = 16.0 / 32768.0    # 加速度缩放 (g) — 通常 ±16g 量程
SCALE_GYRO  = 2000.0 / 32768.0  # 角速度缩放 (°/s) — 通常 ±2000°/s 量程


def parse_packet(data, offset):
    """解析一帧 WitMotion 0x55 数据包 (11字节)"""
    if offset + 11 > len(data):
        return None, offset
    if data[offset] != 0x55:
        return None, offset + 1

    ptype = data[offset + 1]
    result = {"type": ptype}

    if ptype == 0x51:  # 加速度
        ax, ay, az, temp = struct.unpack_from("<hhhH", data, offset + 2)
        result["name"] = "Accel"
        result["values"] = {
            "ax_g": round(ax * SCALE_ACCEL, 3),
            "ay_g": round(ay * SCALE_ACCEL, 3),
            "az_g": round(az * SCALE_ACCEL, 3),
            "temp": temp / 100.0,
        }
    elif ptype == 0x52:  # 角速度
        gx, gy, gz, _ = struct.unpack_from("<hhhH", data, offset + 2)
        result["name"] = "Gyro"
        result["values"] = {
            "gx_dps": round(gx * SCALE_GYRO, 2),
            "gy_dps": round(gy * SCALE_GYRO, 2),
            "gz_dps": round(gz * SCALE_GYRO, 2),
        }
    elif ptype == 0x53:  # 姿态角
        roll, pitch, yaw, _ = struct.unpack_from("<hhhH", data, offset + 2)
        result["name"] = "Angle"
        result["values"] = {
            "roll_deg": round(roll * SCALE_ANGLE, 1),
            "pitch_deg": round(pitch * SCALE_ANGLE, 1),
            "yaw_deg": round(yaw * SCALE_ANGLE, 1),
        }
    elif ptype == 0x54:  # 磁场
        mx, my, mz, _ = struct.unpack_from("<hhhH", data, offset + 2)
        result["name"] = "Mag"
        result["values"] = {
            "mx": mx,
            "my": my,
            "mz": mz,
        }
    else:
        return None, offset + 1

    return result, offset + 11


def main():
    ser = None
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
        time.sleep(0.2)
        ser.reset_input_buffer()

        print("WitMotion IMU 实时数据 (Ctrl+C 退出)")
        print("-" * 60)

        stats = {"Accel": 0, "Gyro": 0, "Angle": 0, "Mag": 0}
        last_print = time.time()
        accel_g = 0.0

        while True:
            # 读取所有可用数据
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                offset = 0
                while offset < len(data):
                    pkt, offset = parse_packet(data, offset)
                    if pkt:
                        stats[pkt["name"]] += 1
                        if pkt["name"] == "Accel":
                            accel_g = (
                                pkt["values"]["ax_g"] ** 2
                                + pkt["values"]["ay_g"] ** 2
                                + pkt["values"]["az_g"] ** 2
                            ) ** 0.5
                        if pkt["name"] == "Angle":
                            self = pkt["values"]

            # 每秒打印一次
            now = time.time()
            if now - last_print >= 1.0 and stats["Angle"] > 0:
                total = sum(stats.values())
                print(
                    f"\r[Hz:{total:4d}] "
                    f"Roll={self.get('roll_deg',0):+6.1f}° "
                    f"Pitch={self.get('pitch_deg',0):+6.1f}° "
                    f"Yaw={self.get('yaw_deg',0):+6.1f}°  "
                    f"|G|={accel_g:.2f}g  "
                    f"  ",
                    end="",
                    flush=True,
                )
                stats = {"Accel": 0, "Gyro": 0, "Angle": 0, "Mag": 0}
                last_print = now

    except KeyboardInterrupt:
        print("\n\n测试完成")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
