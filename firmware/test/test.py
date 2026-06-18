"""
Mecanum 麦轮持续运动测试脚本 v2.0
- 持续发送 VEL 指令直到 Ctrl+C
- 等待固件 OK 应答，记录完整通信日志
- 支持 STAT 诊断命令
"""
import serial
import time
import sys
import threading

COM_PORT = "COM6"
BAUD_RATE = 115200

# ===================== 测试参数（可修改） =====================
TARGET_VX = 0.3      # 前进速度 m/s
TARGET_VY = 0.0      # 横向速度 m/s
TARGET_W  = 0.0      # 旋转角速度 rad/s
INTERVAL  = 0.1      # 指令发送间隔（秒）
# ==============================================================

# 全局开关：Ctrl+C 时设为 False
running = True


def read_thread_func(ser: serial.Serial):
    """后台线程：持续读取所有串口输出并打印"""
    global running
    while running:
        try:
            if ser.in_waiting:
                line = ser.readline().decode('ascii', errors='replace').strip()
                if line:
                    print(f"  [MCU] {line}")
        except Exception as e:
            if running:
                print(f"  [!] 读取异常: {e}")
            break


def main():
    global running

    ser = None
    cmd_count = 0
    ok_count = 0
    timeout_count = 0
    start_time = 0.0
    try:
        # ---- 打开串口 ----
        ser = serial.Serial()
        ser.port = COM_PORT
        ser.baudrate = BAUD_RATE
        ser.timeout = 0.5
        ser.dtr = False
        ser.rts = False
        ser.open()
        print(f"[*] 串口 {COM_PORT} 已打开 (115200 Baud)")

        # ---- 等待固件就绪 ----
        print("[*] 等待主板就绪信号...")
        is_ready = False
        wait_start = time.time()
        while time.time() - wait_start < 8:
            if ser.in_waiting:
                line = ser.readline().decode('ascii', errors='replace').strip()
                if line:
                    print(f"  [MCU] {line}")
                if "SYSTEM_BOOT_READY" in line:
                    is_ready = True
                    break
        if is_ready:
            print("[+] 主板就绪！")
        else:
            print("[!] 未收到就绪信号，3秒后强行继续...")
            time.sleep(3)

        # 清空残留缓冲区
        ser.reset_input_buffer()
        time.sleep(0.2)

        # ---- 先发送一次 STAT 查看状态 ----
        print("\n[*] 发送诊断命令 STAT...")
        ser.write(b"STAT\n")
        ser.flush()
        time.sleep(0.3)
        while ser.in_waiting:
            line = ser.readline().decode('ascii', errors='replace').strip()
            if line:
                print(f"  [MCU] {line}")

        # ---- 启动后台读取线程 ----
        reader = threading.Thread(target=read_thread_func, args=(ser,), daemon=True)
        reader.start()

        # ---- 主循环：持续发送 VEL ----
        print(f"""
╔══════════════════════════════════════════════╗
║  开始持续运动测试                             ║
║  VX={TARGET_VX:.2f}  VY={TARGET_VY:.2f}  W={TARGET_W:.2f}          ║
║  间隔={INTERVAL:.0f}ms                           ║
║  按 Ctrl+C 停止                              ║
╚══════════════════════════════════════════════╝
""")
        cmd_count = 0
        ok_count = 0
        timeout_count = 0
        start_time = time.time()

        while running:
            loop_start = time.time()

            # 发送 VEL 指令
            cmd = f"VEL {TARGET_VX:.3f} {TARGET_VY:.3f} {TARGET_W:.3f}\n"
            try:
                ser.write(cmd.encode('ascii'))
                ser.flush()
                cmd_count += 1
            except Exception as e:
                print(f"\n[!] 发送失败: {e}")
                break

            # 等待 OK 应答（最多等 INTERVAL 秒）
            ok_received = False
            wait_until = time.time() + INTERVAL * 0.8  # 留 20% 余量
            while time.time() < wait_until:
                if ser.in_waiting:
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line:
                        # 后台线程也会打印，这里只做状态追踪
                        if line.startswith("OK"):
                            ok_count += 1
                            ok_received = True
                        elif "TIMEOUT_STOP" in line:
                            timeout_count += 1
                            print(f"\n[!!!] 固件看门狗超时！车轮可能已停止！")
                        elif "ERR" in line:
                            print(f"\n[!] 固件错误: {line}")
                else:
                    break  # 没有数据就退出等待循环

            # 每 20 次命令打印一次摘要
            if cmd_count % 20 == 0:
                elapsed = time.time() - start_time
                print(f"[*] 已发送 {cmd_count} 条 | OK {ok_count} | 超时 {timeout_count} | 运行 {elapsed:.0f}s")

            # 控制发送间隔
            elapsed = time.time() - loop_start
            sleep_time = INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[*] 用户中断")

    except Exception as e:
        print(f"\n[!] 异常: {e}")
        import traceback
        traceback.print_exc()

    finally:
        running = False
        print("\n[*] 正在停止电机...")
        if ser and ser.is_open:
            for _ in range(5):
                try:
                    ser.write(b"STOP\n")
                    ser.flush()
                    time.sleep(0.1)
                except:
                    break
            # 读取最后的响应
            time.sleep(0.3)
            while ser.in_waiting:
                try:
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line:
                        print(f"  [MCU] {line}")
                except:
                    break
            ser.close()
            print("[*] 串口已关闭")

        elapsed_total = time.time() - start_time if start_time > 0 else 0
        print(f"\n测试结束 | 总发送: {cmd_count} | OK: {ok_count} | 超时: {timeout_count} | 时长: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
