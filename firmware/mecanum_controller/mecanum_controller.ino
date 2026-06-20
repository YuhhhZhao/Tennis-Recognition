/**
 * MKS TinyBee 麦轮控制器 v3.1
 *
 * 协议 (115200 8N1):
 *   Jetson → ESP32:
 *     TARGET <x_m> <y_m> <t_s>\n    — 落点指令
 *     VEL <vx> <vy> <w>\n           — 速度指令(调试)
 *     STOP\n                        — 停车
 *     PING\n                        — 查询步数
 *     STAT\n                        — 诊断
 *     RESET_STEPS\n                 — 步数归零
 *     DEBUG 1|0\n                   — 诊断开关
 *
 *   ESP32 → Jetson:
 *     RDY\n
 *     OK\n
 *     PONG s=<s0>,<s1>,<s2>,<s3>\n  — 带符号步数(正=前进方向)
 *     STOP_OK STEPS=<s0>,...\n
 *
 * 里程计算 → Jetson(Python) 端完成
 * 物理映射: E1(左前), E0(右前), Z(左后), Y(右后)
 */

#include <Arduino.h>
#include <WiFi.h>

// ── I2S 引脚 ──
#define I2S_WS   26
#define I2S_BCK  25
#define I2S_DATA 27
#define FAST_HIGH(pin) REG_WRITE(GPIO_OUT_W1TS_REG, 1UL << (pin))
#define FAST_LOW(pin)  REG_WRITE(GPIO_OUT_W1TC_REG, 1UL << (pin))

// 虚拟引脚 bit
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

// ── 运动学常数 ──
const int   MICROSTEPS    = 8;
const int   STEPS_PER_REV = 200 * MICROSTEPS;  // 1600
const float WHEEL_RADIUS  = 0.023f;
// 物理实测: LX=0.096m, LY=0.097m, L_SUM=0.193m (轮中心距÷2)
// IMU标定: L_SUM_eff=0.173m (麦轮辊子打滑导致有效力臂缩短)
// 下面使用IMU标定值确保转弯精度
const float LX = 0.086f, LY = 0.087f, L_SUM = LX + LY;  // 0.173 (IMU校准)
const float INV_R_DIV_2PI = 1.0f / (WHEEL_RADIUS * 2.0f * PI);

const uint32_t VEL_TIMEOUT_MS = 1000;
const float MAX_VX = 1.5f, MAX_VY = 1.5f, MIN_T = 0.05f;

// ── ISR 状态 ──
#define STREAM_HZ 20000
volatile uint32_t stream_bits = 0;
volatile int32_t  step_rate_int[4] = {0, 0, 0, 0};
volatile int32_t  step_acc[4] = {0, 0, 0, 0};
const int32_t ACC_SCALE = STREAM_HZ * 256;

volatile uint32_t isr_count = 0;
volatile int32_t  step_count[4] = {0, 0, 0, 0};  // ← 带符号!
volatile int8_t   step_dir[4] = {1, 1, 1, 1};    // 当前方向 (1/-1)

hw_timer_t *timer = nullptr;

// ── 20kHz ISR ──
void IRAM_ATTR isr_20khz() {
  isr_count++;

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

  for (int i = 0; i < 4; i++) stream_bits &= ~(1 << MB[i].step);

  for (int i = 0; i < 4; i++) {
    int32_t r_int = step_rate_int[i];
    if (r_int == 0) continue;

    // 方向
    if (r_int > 0) {
      stream_bits &= ~(1 << MB[i].dir);
      step_dir[i] = 1;
    } else {
      stream_bits |=  (1 << MB[i].dir);
      step_dir[i] = -1;
    }

    int32_t abs_r = (r_int > 0) ? r_int : -r_int;
    step_acc[i] += abs_r;
    if (step_acc[i] >= ACC_SCALE) {
      step_acc[i] -= ACC_SCALE;
      stream_bits |= (1 << MB[i].step);
      step_count[i] += step_dir[i];  // ← 带符号累加
    }
  }
}

// ── 主循环状态 ──
uint32_t last_cmd_time = 0;
bool motors_enabled = false, debug_enabled = false;
String serial_buf = "";
uint32_t diag_last_print = 0;
String last_cmd_type = "";

static int32_t to_fixed(float sps) { return (int32_t)(sps * 256.0f); }

static void kinematics(float vx, float vy, float w, float rps[4]) {
  rps[0] = (vx - vy - w * L_SUM) * INV_R_DIV_2PI;  // 左前
  rps[1] = (vx + vy + w * L_SUM) * INV_R_DIV_2PI;  // 右前
  rps[2] = (vx + vy - w * L_SUM) * INV_R_DIV_2PI;  // 左后
  rps[3] = (vx - vy + w * L_SUM) * INV_R_DIV_2PI;  // 右后
}

static void apply_motor_rates(const float rps[4]) {
  for (int i = 0; i < 4; i++)
    step_rate_int[i] = to_fixed(rps[i] * STEPS_PER_REV);
  last_cmd_time = millis();
  motors_enabled = true;
}

static float clamp_f(float v, float lim) {
  if (v > lim) return lim; if (v < -lim) return -lim; return v;
}

// ── 安全读取带符号步数 (关中断防止撕裂) ──
static void read_step_counts(int32_t out[4]) {
  noInterrupts();
  for (int i = 0; i < 4; i++) out[i] = step_count[i];
  interrupts();
}

// ── 命令处理 ──
void process_command(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.startsWith("TARGET")) {
    float x=0,y=0,t=0;
    if (sscanf(cmd.c_str(), "TARGET %f %f %f", &x, &y, &t) >= 3) {
      last_cmd_type = "TARGET";
      float vx, vy;
      if (t > MIN_T) { vx = clamp_f(x/t, MAX_VX); vy = clamp_f(y/t, MAX_VY); }
      else { vx = (x>0)?MAX_VX:((x<0)?-MAX_VX:0); vy = (y>0)?MAX_VY:((y<0)?-MAX_VY:0); }
      float rps[4]; kinematics(vx, vy, 0, rps); apply_motor_rates(rps);
      Serial.println(debug_enabled ? "OK TGT" : "OK");
    } else Serial.println("ERR TARGET parse");

  } else if (cmd.startsWith("VEL")) {
    float vx=0,vy=0,w=0;
    if (sscanf(cmd.c_str(), "VEL %f %f %f", &vx, &vy, &w) >= 1) {
      last_cmd_type = "VEL";
      float rps[4]; kinematics(vx, vy, w, rps); apply_motor_rates(rps);
      Serial.println(debug_enabled ? "OK VEL" : "OK");
    } else Serial.println("ERR VEL parse");

  } else if (cmd == "STOP") {
    for (int i=0;i<4;i++) step_rate_int[i] = 0;
    motors_enabled = false; last_cmd_type = "STOP";
    int32_t sc[4]; read_step_counts(sc);
    Serial.print("STOP_OK STEPS=");
    Serial.print(sc[0]); Serial.print(",");
    Serial.print(sc[1]); Serial.print(",");
    Serial.print(sc[2]); Serial.print(",");
    Serial.println(sc[3]);

  } else if (cmd == "PING") {
    int32_t sc[4]; read_step_counts(sc);
    Serial.print("PONG s=");
    Serial.print(sc[0]); Serial.print(",");
    Serial.print(sc[1]); Serial.print(",");
    Serial.print(sc[2]); Serial.print(",");
    Serial.println(sc[3]);

  } else if (cmd == "RESET_STEPS") {
    noInterrupts();
    for (int i=0;i<4;i++) step_count[i] = 0;
    interrupts();
    Serial.println("STEPS_RESET");

  } else if (cmd == "STAT") {
    int32_t sc[4]; read_step_counts(sc);
    Serial.print("STAT ENABLED="); Serial.print(motors_enabled?1:0);
    Serial.print(" STEPS=");
    Serial.print(sc[0]); Serial.print(",");
    Serial.print(sc[1]); Serial.print(",");
    Serial.print(sc[2]); Serial.print(",");
    Serial.print(sc[3]);
    Serial.print(" SINCE="); Serial.print(millis()-last_cmd_time);
    Serial.print(" LAST="); Serial.print(last_cmd_type);
    Serial.println();

  } else if (cmd == "DEBUG 1") { debug_enabled=true; Serial.println("DEBUG ON"); }
    else if (cmd == "DEBUG 0") { debug_enabled=false; Serial.println("DEBUG OFF"); }
    else { Serial.print("ERR unknown: "); Serial.println(cmd); }
}

// ── 初始化 ──
void setup() {
  WiFi.mode(WIFI_OFF);
  Serial.begin(115200); delay(100);
  pinMode(I2S_WS,OUTPUT); digitalWrite(I2S_WS,LOW);
  pinMode(I2S_BCK,OUTPUT); digitalWrite(I2S_BCK,LOW);
  pinMode(I2S_DATA,OUTPUT); digitalWrite(I2S_DATA,LOW);
  stream_bits = 0;
  for (int i=0;i<4;i++) stream_bits &= ~(1<<MB[i].en);
  isr_count = 0;
  for (int i=0;i<4;i++) { step_count[i]=0; step_dir[i]=1; }
  timer = timerBegin(1000000);
  timerAttachInterrupt(timer, &isr_20khz);
  timerAlarm(timer, 50, true, 0);
  timerRestart(timer);
  Serial.println("RDY");
  Serial.print("FW:3.1 | VEL_TIMEOUT="); Serial.print(VEL_TIMEOUT_MS);
  Serial.println("ms");
}

// ── 主循环 ──
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c=='\n' || c=='\r') {
      if (serial_buf.length()>0) { process_command(serial_buf); serial_buf=""; }
    } else serial_buf += c;
  }
  if (motors_enabled && (millis()-last_cmd_time > VEL_TIMEOUT_MS)) {
    for (int i=0;i<4;i++) step_rate_int[i]=0;
    motors_enabled=false;
    Serial.println("TIMEOUT");
  }
}
