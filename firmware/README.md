# ESP32 TinyBee 麦轮控制固件

上位机 Jetson 通过 UART 发落点指令，ESP32 执行 **级联 PID 控制** 驱动麦轮小车。

## 控制架构

```
Jetson TARGET (x, y)
        ↓
  ┌─ 外环: 位置 PID (50Hz) ─┐
  │   vx_cmd = PID(error_x) │
  │   vy_cmd = PID(error_y) │
  └─────────┬───────────────┘
            ↓
  ┌─ 麦轮逆运动学 ──────────┐
  │   (vx,vy) → 四轮转速    │
  └─────────┬───────────────┘
            ↓
  ┌─ 内环: 速度 PID ×4 (200Hz) ┐
  │   pwm = PID(w_cmd, w_meas) │
  └─────────┬───────────────────┘
            ↓
      电机 PWM + 编码器反馈
            ↓
        里程计 (x, y)
```

## 通信协议

| 方向 | 指令 | 说明 |
|------|------|------|
| ESP32 → Jetson | `RDY` | 上电就绪 |
| Jetson → ESP32 | `TARGET <x> <y> [t]` | 落点 (米), t 可忽略 |
| ESP32 → Jetson | `OK` / `DONE` / `ERR <msg>` | 状态响应 |
| Jetson → ESP32 | `STOP` | 紧急停车 |
| Jetson → ESP32 | `PING` | 查询里程计 |
| Jetson → ESP32 | `KPX/KPY/KIX/KVX/KVI <val>` | 运行时调 PID 增益 |
| Jetson → ESP32 | `DEBUG 1` / `DEBUG 0` | 开关遥测 |
| ESP32 → Jetson | `TELEM x y ...` | 调试遥测 (~10Hz) |

波特率: **115200 8N1**

## 调试步骤

调试分三个阶段：**无硬件 → 串口回环 → 实机联调**。

### 阶段 1: 无硬件虚拟调试

在 Jetson 上，不需要 ESP32，用虚拟模式测试协议：

```bash
python scripts/test_uart.py --mode virtual
```

这会启动一个虚拟 ESP32，模拟小车运动。你可以输入 `TARGET 0.5 0.0` 看虚拟小车如何逼近目标。

```
>> TARGET 0.5 0.0 2.0
  ← [VIRT] OK
>> PING
  ← [VIRT] PONG x=0.482 y=0.000 vx=0 vy=0
```

### 阶段 2: ESP32 串口监视

烧录固件后，用 Arduino IDE 串口监视器或任意串口工具连接 TinyBee USB：

```bash
# Linux
screen /dev/ttyUSB0 115200
# Windows — Arduino IDE 串口监视器 (115200 baud)
```

1. 通电后应收到 `RDY`
2. 手动发 `TARGET 0.3 0.0` → 应回 `OK`
3. 发 `PING` → 应回里程计
4. 发 `DEBUG 1` → 应回 `DEBUG ON` 并开始输出 `TELEM ...`

**无编码器时**：设 `ENCODERS_ENABLED = false`，里程计基于速度指令估算（开环，会漂移）。先验证电机转得对不对。

**有编码器时**：设 `ENCODERS_ENABLED = true`，里程计来自编码器反馈（闭环，精确）。接线后检查 `PING` 返回的里程计是否随小车移动正确变化。

### 阶段 3: Jetson ↔ ESP32 联调

1. 接线: Jetson J12 pin8(TX)→ESP32 RX, pin10(RX)→ESP32 TX, GND↔GND
2. Jetson 端禁用串口终端: `sudo systemctl stop nvgetty`
3. 运行调试脚本:

```bash
# 交互式 — 手动发指令
python scripts/test_uart.py --port /dev/ttyTHS1 --mode interactive

# 自动测试 — 预设序列
python scripts/test_uart.py --port /dev/ttyTHS1 --mode auto

# 遥测监视 — 持续观察 PID 行为
python scripts/test_uart.py --port /dev/ttyTHS1 --mode monitor
```

### PID 调参指南

默认增益是保守的，需要根据实际小车调整。

**1. 先调速度环 (内环)** — 让轮子能精确跟踪目标转速:

```
KVX 0.5     # 从小 P 开始
KVI 0.1     # 加一点 I 消除稳态误差
```

观察 TELEM 中的轮速跟踪 — 轮子实际转速是否跟上指令。

**2. 再调位置环 (外环)** — 让小车准确到达目标:

```
KPX 1.0     # P 增益: 越大越激进, 太大会振荡
KPY 1.0
KIX 0.05    # I 增益: 消除稳态误差, 太大会超调
KID 0.2     # D 增益: 抑制振荡, 太大会引入噪声
```

**调参经验**:
- 小车到达目标后来回摆动 → 减小 KPX/KPY, 增大 KID
- 小车停在一个固定偏差处不动 → 增大 KIX
- 小车走 S 形 → X/Y 增益不一致, 分别调整
- 轮子尖叫/抖动 → 速度环 P 太大, 调 KVX

### 串口不通信排查

1. **无 RDY**: 检查波特率 115200, 检查 ESP32 是否烧录成功, 按 RST 按钮重试
2. **TX/RX 反了**: TX 应接对端 RX, 用万用表测量交叉
3. **共地**: Jetson GND 必须连 ESP32 GND
4. **nvgetty 占用**: `sudo systemctl stop nvgetty`
5. **电平**: 两者都是 3.3V, 直连无需电平转换

## 烧录

1. Arduino IDE 安装 ESP32 支持 (附加开发板管理器: `https://espressif.github.io/arduino-esp32/package_esp32_index.json`)
2. 开发板选 `ESP32 Dev Module`
3. 端口选 TinyBee 对应端口
4. 上传 `mecanum_controller.ino`

> **注意**: 会覆盖 TinyBee 上原有 Marlin 固件。需要恢复时重新烧录 Marlin 即可。

## 接线: Jetson ↔ TinyBee

```
Jetson J12              MKS TinyBee
──────────────────────────────────
Pin  8  (UART1 TX)  →   RX
Pin 10  (UART1 RX)  →   TX
GND                 →   GND

电机驱动板 (TB6612 / L298N):
TinyBee GPIO 12,13 → M1 PWM, DIR (左前)
TinyBee GPIO 14,15 → M2 PWM, DIR (右前)
TinyBee GPIO 16,17 → M3 PWM, DIR (左后)
TinyBee GPIO 18,19 → M4 PWM, DIR (右后)

编码器 (可选, 移到 ENCODERS_ENABLED=true 后连接):
TinyBee GPIO 32,33 → M1 编码器 A, B
TinyBee GPIO 34,35 → M2 编码器 A, B
TinyBee GPIO 36,23 → M3 编码器 A, B (注意 GPIO 36 仅输入)
TinyBee GPIO 39,22 → M4 编码器 A, B (注意 GPIO 39 仅输入)
```
