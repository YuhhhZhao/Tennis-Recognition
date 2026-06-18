/**
 * MKS TinyBee 麦轮控制器 v3.0 (TARGET 协议版)
 *
 * Jetson ↔ ESP32 通信协议 (文本, 115200 8N1, \\n 结束):
 *   Jetson → ESP32:  TARGET <x_m> <y_m> <t_s>\n
 *                     VEL <vx> <vy> <w>\n   (调试用)
 *                     STOP\n
 *                     PING\n
 *                     STAT\n
 *                     DEBUG 1|0\n
 *   ESP32 → Jetson:  RDY\n
 *                     OK\n
 *                     PONG x=0 y=0\n
 *                     ERR <msg>\n
 *                     TIMEOUT_STOP\n
 *
 * TARGET 处理: vx = x/t, vy = y/t → 麦轮运动学 → 步进脉冲
 * 无里程计反馈 (纯开环), 看门狗 1000ms
 *
 * 物理映射: E1(左前), E0(右前), Z(左后), Y(右后)
 */

#include <Arduino.h>
#include <WiFi.h>

// ============================================================================
// I2S 引脚与寄存器级极速宏
// ============================================================================
#define I2S_WS   26
#define I2S_BCK  25
#define I2S_DATA 27

#define FAST_HIGH(pin) REG_WRITE(GPIO_OUT_W1TS_REG, 1UL << (pin))
#define FAST_LOW(pin)  REG_WRITE(GPIO_OUT_W1TC_REG, 1UL << (pin))

// 虚拟引脚 bit 位
#define V_Y_EN    3
#define V_Y_STEP  4
#define V_Y_DIR   5
#define V_Z_EN    6
#define V_Z_STEP  7
#define V_Z_DIR   8
#define V_E0_EN   9
#define V_E0_STEP 10
#define V_E0_DIR  11
#define V_E1_EN   12
#define V_E1_STEP 13
#define V_E1_DIR  14

struct MotorBits { int step, dir, en; };
const MotorBits MB[4] = {
  {V_E1_STEP, V_E1_DIR, V_E1_EN },  // M0: 左前 (E1)
  {V_E0_STEP, V_E0_DIR, V_E0_EN },  // M1: 右前 (E0)
  {V_Z_STEP,  V_Z_DIR,  V_Z_EN  },  // M2: 左后 (Z)
  {V_Y_STEP,  V_Y_DIR,  V_Y_EN  },  // M3: 右后 (Y)
};

// ============================================================================
// 运动学参数
// ============================================================================
const int   MICROSTEPS    = 8;
const int   STEPS_PER_REV = 200 * MICROSTEPS;       // 1600
const float WHEEL_RADIUS  = 0.023f;
const float LX            = 0.15f;
const float LY            = 0.12f;
const float L_SUM         = LX + LY;                // 0.27
const float INV_R_DIV_2PI = 1.0f / (WHEEL_RADIUS * 2.0f * PI);
                                                      // 1/(R*2π) = 6.919

const uint32_t VEL_TIMEOUT_MS = 1000;

// 安全速度上限
const float MAX_VX = 1.5f;   // 最大前进速度 m/s
const float MAX_VY = 1.0f;   // 最大横向速度 m/s
const float MIN_T   = 0.05f; // TARGET t 最小值 (避免除零导致无穷大速度)

// ============================================================================
// 定点数脉冲引擎 (20kHz ISR) —— 全程无浮点！
// ============================================================================
volatile uint32_t stream_bits = 0;
#define STREAM_HZ 20000

// 定点格式: step_rate_int = desired_steps_per_second * 256
// 例如 3321.6 steps/s → step_rate_int = 850330
volatile int32_t step_rate_int[4] = {0, 0, 0, 0};

// 步进脉冲累加器 (整数)
volatile int32_t step_acc[4] = {0, 0, 0, 0};

// ACC_SCALE = STREAM_HZ * 256 = 5,120,000
// 每个 ISR tick 累加 step_rate_int，达到 ACC_SCALE 即触发一步
const int32_t ACC_SCALE = STREAM_HZ * 256;

// 诊断计数器
volatile uint32_t isr_count = 0;
volatile uint32_t step_count[4] = {0, 0, 0, 0};

hw_timer_t *timer = nullptr;

void IRAM_ATTR isr_20khz() {
  isr_count++;

  // ---- 输出当前 stream_bits 到 I2S ----
  uint32_t bits = stream_bits;
  for (int b = 23; b >= 0; b--) {
    if ((bits >> b) & 1) FAST_HIGH(I2S_DATA);
    else                 FAST_LOW(I2S_DATA);
    FAST_LOW(I2S_BCK);
    FAST_HIGH(I2S_BCK); FAST_HIGH(I2S_BCK);
    FAST_LOW(I2S_BCK);
  }
  FAST_LOW(I2S_WS);
  FAST_HIGH(I2S_WS); FAST_HIGH(I2S_WS);
  FAST_LOW(I2S_WS);

  // ---- 清除上轮的 step bits ----
  for (int i = 0; i < 4; i++) stream_bits &= ~(1 << MB[i].step);

  // ---- 纯整数步进脉冲生成 ----
  for (int i = 0; i < 4; i++) {
    int32_t r_int = step_rate_int[i];  // 原子读 int32_t
    if (r_int == 0) continue;

    // 方向控制 (正值=正向, 负值=反向)
    if (r_int > 0) stream_bits &= ~(1 << MB[i].dir);
    else           stream_bits |=  (1 << MB[i].dir);

    // 绝对值累加 (纯整数)
    int32_t abs_r = (r_int > 0) ? r_int : -r_int;
    step_acc[i] += abs_r;

    if (step_acc[i] >= ACC_SCALE) {
      step_acc[i] -= ACC_SCALE;
      stream_bits |= (1 << MB[i].step);
      step_count[i]++;
    }
  }
}

// ============================================================================
// 全局状态与指令处理 (仅主循环使用，可以用浮点)
// ============================================================================
uint32_t last_cmd_time = 0;
bool motors_enabled = false;
bool debug_enabled = false;   // DEBUG 1/0 控制诊断输出
String serial_buf = "";
uint32_t diag_last_print = 0;
String last_cmd_type = "";    // 记录上一条指令类型

// 将浮点 step_rate 转为定点整数
static int32_t to_fixed(float steps_per_sec) {
  return (int32_t)(steps_per_sec * 256.0f);
}

// 麦轮运动学: (vx, vy, w) → 4轮转速 (rps)
// vx=前向速度, vy=左向速度, w=逆时针角速度
static void kinematics(float vx, float vy, float w, float rps[4]) {
  // X 型麦轮: 对角同向 (FL=RR:\, FR=RL:/)
  // FL vy=-, FR vy=+, RL vy=+, RR vy=-
  rps[0] = (vx - vy - w * L_SUM) * INV_R_DIV_2PI;  // M0: 左前 E1  (vy=-)
  rps[1] = (vx + vy + w * L_SUM) * INV_R_DIV_2PI;  // M1: 右前 E0  (vy=+)
  rps[2] = (vx + vy - w * L_SUM) * INV_R_DIV_2PI;  // M2: 左后 Z   (vy=+)  ← 改
  rps[3] = (vx - vy + w * L_SUM) * INV_R_DIV_2PI;  // M3: 右后 Y   (vy=-)  ← 改
}

// 将运动学解算结果写入 step_rate_int[] 并刷新看门狗
static void apply_motor_rates(const float rps[4]) {
  for (int i = 0; i < 4; i++) {
    step_rate_int[i] = to_fixed(rps[i] * STEPS_PER_REV);
  }
  last_cmd_time = millis();
  motors_enabled = true;
}

// 限制速度在安全范围内
static float clamp_f(float val, float limit) {
  if (val > limit) return limit;
  if (val < -limit) return -limit;
  return val;
}

void process_command(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // ---- TARGET: Jetson 落点指令 (生产协议) ----
  if (cmd.startsWith("TARGET")) {
    float x = 0, y = 0, t = 0;
    int n = sscanf(cmd.c_str(), "TARGET %f %f %f", &x, &y, &t);
    if (n >= 3) {
      last_cmd_type = "TARGET";
      float vx, vy, w = 0;

      if (t > MIN_T) {
        vx = clamp_f(x / t, MAX_VX);
        vy = clamp_f(y / t, MAX_VY);
      } else {
        // t 太小 → 球即将落地，全速冲刺或急停
        vx = (x > 0) ? MAX_VX : ((x < 0) ? -MAX_VX : 0);
        vy = (y > 0) ? MAX_VY : ((y < 0) ? -MAX_VY : 0);
      }

      float rps[4];
      kinematics(vx, vy, w, rps);
      apply_motor_rates(rps);

      // 简洁应答 (匹配 uart_bridge.py 的协议)
      if (debug_enabled) {
        Serial.print("OK TGT x="); Serial.print(x, 2);
        Serial.print(" y="); Serial.print(y, 2);
        Serial.print(" t="); Serial.print(t, 2);
        Serial.print(" vx="); Serial.print(vx, 2);
        Serial.print(" vy="); Serial.print(vy, 2);
      } else {
        Serial.print("OK");  // 生产模式最简应答
      }
      Serial.println();
    } else {
      Serial.print("ERR TARGET parse: "); Serial.println(cmd);
    }

  // ---- VEL: 原始速度指令 (调试用) ----
  } else if (cmd.startsWith("VEL")) {
    float vx = 0, vy = 0, w = 0;
    int n = sscanf(cmd.c_str(), "VEL %f %f %f", &vx, &vy, &w);
    if (n >= 1) {
      last_cmd_type = "VEL";
      float rps[4];
      kinematics(vx, vy, w, rps);
      apply_motor_rates(rps);

      Serial.print("OK VX="); Serial.print(vx, 3);
      Serial.print(" VY="); Serial.print(vy, 3);
      Serial.print(" W="); Serial.print(w, 3);
      Serial.print(" SR=");
      for (int i = 0; i < 4; i++) {
        Serial.print(step_rate_int[i] / 256.0f, 0);
        if (i < 3) Serial.print(",");
      }
      Serial.print(" ISR="); Serial.print(isr_count);
      Serial.println();
    } else {
      Serial.print("ERR VEL parse: "); Serial.println(cmd);
    }

  // ---- STOP: 紧急停车 ----
  } else if (cmd == "STOP") {
    last_cmd_type = "STOP";
    for (int i = 0; i < 4; i++) step_rate_int[i] = 0;
    motors_enabled = false;
    Serial.print("STOP_OK STEPS=");
    for (int i = 0; i < 4; i++) {
      Serial.print(step_count[i]);
      if (i < 3) Serial.print(",");
    }
    Serial.println();
    for (int i = 0; i < 4; i++) step_count[i] = 0;

  // ---- PING: 里程计查询 (当前无编码器, 返回 0) ----
  } else if (cmd == "PING") {
    Serial.println("PONG x=0 y=0");

  // ---- STAT: 诊断状态查询 ----
  } else if (cmd == "STAT") {
    Serial.print("STAT ENABLED="); Serial.print(motors_enabled ? 1 : 0);
    Serial.print(" ISR="); Serial.print(isr_count);
    Serial.print(" SR=");
    for (int i = 0; i < 4; i++) {
      Serial.print(step_rate_int[i] / 256.0f, 0);
      if (i < 3) Serial.print(",");
    }
    Serial.print(" SC=");
    for (int i = 0; i < 4; i++) {
      Serial.print(step_count[i]);
      if (i < 3) Serial.print(",");
    }
    unsigned long since_last = millis() - last_cmd_time;
    Serial.print(" SINCE_LAST="); Serial.print(since_last);
    Serial.print(" LAST="); Serial.print(last_cmd_type);
    Serial.print(" DBG="); Serial.print(debug_enabled ? 1 : 0);
    Serial.println();

  // ---- DEBUG: 诊断输出开关 ----
  } else if (cmd == "DEBUG 1") {
    debug_enabled = true;
    Serial.println("DEBUG ON");
  } else if (cmd == "DEBUG 0") {
    debug_enabled = false;
    Serial.println("DEBUG OFF");

  } else {
    Serial.print("ERR unknown: "); Serial.println(cmd);
  }
}

// ============================================================================
// 初始化
// ============================================================================
void setup() {
  WiFi.mode(WIFI_OFF);

  Serial.begin(115200);
  delay(100);

  pinMode(I2S_WS, OUTPUT);   digitalWrite(I2S_WS, LOW);
  pinMode(I2S_BCK, OUTPUT);  digitalWrite(I2S_BCK, LOW);
  pinMode(I2S_DATA, OUTPUT); digitalWrite(I2S_DATA, LOW);

  stream_bits = 0;
  for (int i = 0; i < 4; i++) stream_bits &= ~(1 << MB[i].en);

  isr_count = 0;
  for (int i = 0; i < 4; i++) step_count[i] = 0;

  timer = timerBegin(1000000);
  timerAttachInterrupt(timer, &isr_20khz);
  timerAlarm(timer, 50, true, 0);
  timerRestart(timer);

  Serial.println("RDY");
  Serial.print("FW:3.0-target | TIMEOUT="); Serial.print(VEL_TIMEOUT_MS);
  Serial.print("ms MAX_VX="); Serial.print(MAX_VX, 1);
  Serial.print(" MAX_VY="); Serial.print(MAX_VY, 1);
  Serial.println();
}

// ============================================================================
// 主循环
// ============================================================================
void loop() {
  // 非阻塞串口读取
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serial_buf.length() > 0) {
        process_command(serial_buf);
        serial_buf = "";
      }
    } else {
      serial_buf += c;
    }
  }

  // 安全看门狗
  if (motors_enabled && (millis() - last_cmd_time > VEL_TIMEOUT_MS)) {
    for (int i = 0; i < 4; i++) step_rate_int[i] = 0;
    motors_enabled = false;
    if (debug_enabled) {
      Serial.print("TIMEOUT_STOP SINCE_LAST=");
      Serial.print(millis() - last_cmd_time);
      Serial.print(" STEPS=");
      for (int i = 0; i < 4; i++) {
        Serial.print(step_count[i]);
        if (i < 3) Serial.print(",");
      }
      Serial.println();
    } else {
      Serial.println("TIMEOUT");
    }
  }

  // 诊断心跳：仅在 DEBUG 模式 + 电机使能时输出
  if (debug_enabled && motors_enabled && (millis() - diag_last_print > 2000)) {
    diag_last_print = millis();
    Serial.print("DIAG SR=");
    for (int i = 0; i < 4; i++) {
      Serial.print(step_rate_int[i] / 256.0f, 0);
      if (i < 3) Serial.print(",");
    }
    Serial.print(" SC=");
    for (int i = 0; i < 4; i++) {
      Serial.print(step_count[i]);
      if (i < 3) Serial.print(",");
    }
    Serial.print(" AGE="); Serial.print(millis() - last_cmd_time);
    Serial.println("ms");
  }
}
