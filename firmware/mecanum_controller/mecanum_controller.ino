/**
 * MKS TinyBee 麦轮小车控制器 v2 — 级联 PID + 编码器
 *
 * 控制架构 (级联 PID):
 *   Jetson TARGET (x,y) → [外环: 位置 PID] → (vx, vy) 指令
 *                       → [麦轮逆运动学] → 四轮目标转速
 *                       → [内环: 速度 PID ×4] → PWM 输出
 *                       → [编码器反馈] → 实际轮速 + 里程计
 *
 * 通信协议 (UART 115200 8N1):
 *   Jetson → ESP32:
 *     TARGET <x> <y> <t>\n   落点 (米, 秒)
 *     STOP\n                 紧急停车
 *     PING\n                 查询状态
 *     KPX <val>\n            调位置 P 增益 (X)
 *     KPY <val>\n            调位置 P 增益 (Y)
 *     KIX <val>\n            调位置 I 增益
 *     KVX <val>\n            调速度 P 增益
 *     DEBUG <0|1>\n          开关调试遥测
 *
 *   ESP32 → Jetson:
 *     RDY\n                  上电就绪
 *     OK\n                   指令已接收
 *     DONE\n                 到达目标
 *     ERR <msg>\n            错误
 *     PONG x=... y=... vx=... vy=...\n
 *     TELEM x y vx vy w1 w2 w3 w4 err_x err_y\n  (调试遥测, ~10Hz)
 */

#include <Arduino.h>

// ============================================================================
// 硬件引脚 — 根据实际接线修改
// ============================================================================

// --- 电机 PWM + 方向 (需接电机驱动板 TB6612 / L298N) ---
struct MotorPins {
  int pwm, dir;
};
const MotorPins MOTORS[4] = {
  {12, 13},  // M1: 左前
  {14, 15},  // M2: 右前
  {16, 17},  // M3: 左后
  {18, 19},  // M4: 右后
};

// --- 编码器 (可选, 无编码器时设 ENCODERS_ENABLED=false) ---
const bool ENCODERS_ENABLED = false;   // 有编码器接好线后改 true
const int  ENC_A[4] = {32, 34, 36, 39};  // 四路编码器 A 相
const int  ENC_B[4] = {33, 35, 23, 22};  // 四路编码器 B 相
const int  ENC_PPR = 11;                  // 编码器每转脉冲数 (电机轴)
const float GEAR_RATIO = 30.0;            // 减速比

// ============================================================================
// 运动学 + 控制参数
// ============================================================================

const float LX = 0.15f;           // 轮子距中心左右 (m)
const float LY = 0.12f;           // 轮子距中心前后 (m)
const float WHEEL_RADIUS = 0.032f;// 轮子半径 (m)
const float MAX_SPEED = 1.5f;     // 最大线速度 (m/s)
const float MAX_ACCEL = 2.0f;     // 最大线加速度 (m/s²)
const float POS_TOLERANCE = 0.05f;// 到达容差 (m)
const uint32_t CMD_TIMEOUT_MS = 2000;

// PID 默认增益 (运行时可通过串口调整)
float KP_X = 1.5f, KI_X = 0.1f, KD_X = 0.3f;   // 位置环 X
float KP_Y = 1.5f, KI_Y = 0.1f, KD_Y = 0.3f;   // 位置环 Y
float KP_V = 0.8f, KI_V = 0.3f, KD_V = 0.05f;  // 速度环 (四轮共用)

// ============================================================================
// PID 控制器
// ============================================================================

class PID {
public:
  float kp, ki, kd;
  float integral, prev_error, prev_meas;
  float out_min, out_max;
  bool use_d_on_meas;  // true = derivative on measurement (less kick)

  PID(float p=1.0, float i=0.0, float d=0.0,
      float min_out=-1.0, float max_out=1.0, bool d_on_meas=true)
    : kp(p), ki(i), kd(d), integral(0), prev_error(0), prev_meas(0),
      out_min(min_out), out_max(max_out), use_d_on_meas(d_on_meas) {}

  void reset() { integral = 0; prev_error = 0; prev_meas = 0; }

  float update(float setpoint, float measurement, float dt) {
    if (dt <= 0) dt = 0.001f;
    float error = setpoint - measurement;

    // Proportional
    float p_out = kp * error;

    // Integral (with anti-windup clamping)
    if (ki > 0) {
      integral += error * dt;
      // Clamp integral to prevent windup
      float i_max = (out_max - p_out) / (ki + 1e-6f);
      float i_min = (out_min - p_out) / (ki + 1e-6f);
      integral = constrain(integral, i_min, i_max);
    }
    float i_out = ki * integral;

    // Derivative
    float d_out = 0;
    if (kd > 0) {
      if (use_d_on_meas) {
        d_out = -kd * (measurement - prev_meas) / dt;
      } else {
        d_out = kd * (error - prev_error) / dt;
      }
    }

    float output = p_out + i_out + d_out;
    output = constrain(output, out_min, out_max);

    prev_error = error;
    prev_meas = measurement;
    return output;
  }
};

// ============================================================================
// 编码器 (中断驱动)
// ============================================================================

volatile long encoder_counts[4] = {0, 0, 0, 0};

void IRAM_ATTR enc0_isr() { encoder_counts[0] += (digitalRead(ENC_B[0]) ^ digitalRead(ENC_A[0])) ? 1 : -1; }
void IRAM_ATTR enc1_isr() { encoder_counts[1] += (digitalRead(ENC_B[1]) ^ digitalRead(ENC_A[1])) ? 1 : -1; }
void IRAM_ATTR enc2_isr() { encoder_counts[2] += (digitalRead(ENC_B[2]) ^ digitalRead(ENC_A[2])) ? 1 : -1; }
void IRAM_ATTR enc3_isr() { encoder_counts[3] += (digitalRead(ENC_B[3]) ^ digitalRead(ENC_A[3])) ? 1 : -1; }

typedef void (*EncISR)();
EncISR enc_isrs[4] = {enc0_isr, enc1_isr, enc2_isr, enc3_isr};

void init_encoders() {
  if (!ENCODERS_ENABLED) return;
  for (int i = 0; i < 4; i++) {
    pinMode(ENC_A[i], INPUT_PULLUP);
    pinMode(ENC_B[i], INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENC_A[i]), enc_isrs[i], CHANGE);
  }
}

// 读轮速 (RPM), 清零计数器
float read_wheel_rpm(int idx, float dt) {
  if (!ENCODERS_ENABLED || dt <= 0) return 0;
  long cnt = encoder_counts[idx];
  encoder_counts[idx] = 0;
  // RPM = (脉冲数 / PPR) / 减速比 * 60 / dt
  return (float)cnt / (float)ENC_PPR / GEAR_RATIO * 60.0f / dt;
}

// ============================================================================
// 麦轮逆运动学
// ============================================================================

void mecanum_ik(float vx, float vy, float omega, float w[4]) {
  float L = LX + LY;
  w[0] = (vx - vy - omega * L) / WHEEL_RADIUS;  // rad/s
  w[1] = (vx + vy + omega * L) / WHEEL_RADIUS;
  w[2] = (vx + vy - omega * L) / WHEEL_RADIUS;
  w[3] = (vx - vy + omega * L) / WHEEL_RADIUS;
}

// 正运动学: 轮速 → 机器人速度
void mecanum_fk(const float w[4], float &vx, float &vy, float &omega) {
  float L = LX + LY;
  vx = WHEEL_RADIUS * 0.25f * (w[0] + w[1] + w[2] + w[3]);
  vy = WHEEL_RADIUS * 0.25f * (-w[0] + w[1] + w[2] - w[3]);
  omega = WHEEL_RADIUS * 0.25f * (-w[0] + w[1] - w[2] + w[3]) / L;
}

// ============================================================================
// 全局状态
// ============================================================================

// PID 控制器实例
PID pid_x(KP_X, KI_X, KD_X, -MAX_SPEED, MAX_SPEED);
PID pid_y(KP_Y, KI_Y, KD_Y, -MAX_SPEED, MAX_SPEED);
PID pid_vel[4] = {
  PID(KP_V, KI_V, KD_V, -1.0, 1.0),
  PID(KP_V, KI_V, KD_V, -1.0, 1.0),
  PID(KP_V, KI_V, KD_V, -1.0, 1.0),
  PID(KP_V, KI_V, KD_V, -1.0, 1.0),
};

// 目标 + 里程计
float target_x = 0, target_y = 0;
bool has_target = false;
uint32_t last_cmd_ms = 0;

float odom_x = 0, odom_y = 0, odom_heading = 0;
float cmd_vx = 0, cmd_vy = 0;  // 当前速度指令 (用于诊断)

bool debug_telem = false;
uint32_t last_telem_ms = 0;

// 加速度限幅
float prev_vx = 0, prev_vy = 0;
uint32_t last_control_ms = 0;

// ============================================================================
// 电机驱动
// ============================================================================

void set_motor_raw(int idx, float duty) {
  // duty: -1.0 ~ +1.0
  float d = constrain(duty, -1.0f, 1.0f);
  digitalWrite(MOTORS[idx].dir, d >= 0 ? HIGH : LOW);
  analogWrite(MOTORS[idx].pwm, (int)(fabs(d) * 255));
}

void stop_all() {
  for (int i = 0; i < 4; i++) set_motor_raw(i, 0);
}

// ============================================================================
// 串口协议
// ============================================================================

String serial_buf = "";

void parse_cmd(String &cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // --- TARGET <x> <y> [t] ---
  if (cmd.startsWith("TARGET")) {
    float x, y, t_ignored;
    int n = sscanf(cmd.c_str(), "TARGET %f %f %f", &x, &y, &t_ignored);
    if (n >= 2) {
      target_x = x;
      target_y = y;
      has_target = true;
      last_cmd_ms = millis();
      // 重置里程计和 PID 积分器
      odom_x = 0; odom_y = 0; odom_heading = 0;
      pid_x.reset(); pid_y.reset();
      for (int i = 0; i < 4; i++) pid_vel[i].reset();
      prev_vx = 0; prev_vy = 0;
      Serial.println("OK");
    } else {
      Serial.println("ERR parse TARGET");
    }
    return;
  }

  // --- STOP ---
  if (cmd == "STOP") {
    has_target = false;
    stop_all();
    Serial.println("OK");
    return;
  }

  // --- PING ---
  if (cmd == "PING") {
    Serial.print("PONG x="); Serial.print(odom_x, 3);
    Serial.print(" y="); Serial.print(odom_y, 3);
    Serial.print(" vx="); Serial.print(cmd_vx, 3);
    Serial.print(" vy="); Serial.println(cmd_vy, 3);
    return;
  }

  // --- PID 调参 ---
  if (cmd.startsWith("KPX"))  { sscanf(cmd.c_str(), "KPX %f", &KP_X); pid_x.kp = KP_X; Serial.println("OK"); return; }
  if (cmd.startsWith("KPY"))  { sscanf(cmd.c_str(), "KPY %f", &KP_Y); pid_y.kp = KP_Y; Serial.println("OK"); return; }
  if (cmd.startsWith("KIX"))  { float v; sscanf(cmd.c_str(), "KIX %f", &v); pid_x.ki = pid_y.ki = v; Serial.println("OK"); return; }
  if (cmd.startsWith("KID"))  { float v; sscanf(cmd.c_str(), "KID %f", &v); pid_x.kd = pid_y.kd = v; Serial.println("OK"); return; }
  if (cmd.startsWith("KVX"))  { float v; sscanf(cmd.c_str(), "KVX %f", &v); for (int i=0;i<4;i++) pid_vel[i].kp = v; Serial.println("OK"); return; }
  if (cmd.startsWith("KVI"))  { float v; sscanf(cmd.c_str(), "KVI %f", &v); for (int i=0;i<4;i++) pid_vel[i].ki = v; Serial.println("OK"); return; }

  // --- DEBUG <0|1> ---
  if (cmd.startsWith("DEBUG")) {
    int v = cmd.endsWith("1") ? 1 : 0;
    debug_telem = (v == 1);
    Serial.println(debug_telem ? "DEBUG ON" : "DEBUG OFF");
    return;
  }

  // 未知命令
  Serial.print("ERR unknown: "); Serial.println(cmd);
}

// ============================================================================
// 初始化
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(100);

  for (int i = 0; i < 4; i++) {
    pinMode(MOTORS[i].pwm, OUTPUT);
    pinMode(MOTORS[i].dir, OUTPUT);
  }
  stop_all();

  if (ENCODERS_ENABLED) init_encoders();

  Serial.println("RDY");
  last_cmd_ms = millis();
  last_control_ms = millis();
}

// ============================================================================
// 主循环
// ============================================================================

void loop() {
  uint32_t now = millis();

  // ---- 串口接收 ----
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serial_buf.length() > 0) { parse_cmd(serial_buf); serial_buf = ""; }
    } else {
      serial_buf += c;
    }
  }

  // ---- 控制循环 (200Hz) ----
  float dt = (now - last_control_ms) / 1000.0f;
  if (dt >= 0.004f && dt < 0.1f) {  // ~200Hz, 防跳变
    last_control_ms = now;

    if (has_target) {
      // 超时保护
      if (now - last_cmd_ms > CMD_TIMEOUT_MS) {
        has_target = false; stop_all();
        Serial.println("ERR timeout");
      } else {
        // --- 外环: 位置 PID → (vx_cmd, vy_cmd) ---
        float vx_cmd = pid_x.update(target_x, odom_x, dt);
        float vy_cmd = pid_y.update(target_y, odom_y, dt);

        // 加速度限幅
        vx_cmd = constrain(vx_cmd, prev_vx - MAX_ACCEL*dt, prev_vx + MAX_ACCEL*dt);
        vy_cmd = constrain(vy_cmd, prev_vy - MAX_ACCEL*dt, prev_vy + MAX_ACCEL*dt);
        prev_vx = vx_cmd; prev_vy = vy_cmd;
        cmd_vx = vx_cmd; cmd_vy = vy_cmd;

        // --- 逆运动学: (vx, vy) → 四轮目标转速 ---
        float w_cmd[4];
        mecanum_ik(vx_cmd, vy_cmd, 0, w_cmd);

        // --- 内环: 速度 PID → PWM ---
        float motor_pwm[4];
        for (int i = 0; i < 4; i++) {
          float w_actual = read_wheel_rpm(i, dt) * (PI / 30.0f);  // RPM→rad/s
          motor_pwm[i] = pid_vel[i].update(w_cmd[i], w_actual, dt);
        }

        // 输出 PWM
        for (int i = 0; i < 4; i++) {
          set_motor_raw(i, motor_pwm[i]);
        }

        // --- 里程计更新 ---
        if (ENCODERS_ENABLED) {
          // 从编码器反推机器人速度, 积分得位置
          float w_meas[4];
          for (int i = 0; i < 4; i++) {
            w_meas[i] = read_wheel_rpm(i, dt) * (PI / 30.0f);
          }
          float meas_vx, meas_vy, meas_omega;
          mecanum_fk(w_meas, meas_vx, meas_vy, meas_omega);
          odom_x += meas_vx * dt;
          odom_y += meas_vy * dt;
          odom_heading += meas_omega * dt;
        } else {
          // 无编码器: 用速度指令估算
          odom_x += vx_cmd * dt;
          odom_y += vy_cmd * dt;
        }

        // --- 到达检查 ---
        float dist = sqrtf((target_x - odom_x)*(target_x - odom_x) +
                           (target_y - odom_y)*(target_y - odom_y));
        if (dist < POS_TOLERANCE && fabs(vx_cmd) < 0.05f && fabs(vy_cmd) < 0.05f) {
          has_target = false;
          stop_all();
          Serial.println("DONE");
        }
      }
    } else {
      stop_all();
      pid_x.reset(); pid_y.reset();
      for (int i = 0; i < 4; i++) pid_vel[i].reset();
    }

    // ---- 调试遥测 (10Hz) ----
    if (debug_telem && (now - last_telem_ms >= 100)) {
      last_telem_ms = now;
      float w_now[4] = {0,0,0,0};
      Serial.print("TELEM ");
      Serial.print(odom_x, 3); Serial.print(" ");
      Serial.print(odom_y, 3); Serial.print(" ");
      Serial.print(cmd_vx, 3); Serial.print(" ");
      Serial.print(cmd_vy, 3); Serial.print(" ");
      Serial.print(target_x, 3); Serial.print(" ");
      Serial.print(target_y, 3); Serial.print(" ");
      Serial.print(pid_x.integral, 3); Serial.print(" ");
      Serial.println(pid_y.integral, 3);
    }
  }

  delay(2);
}
