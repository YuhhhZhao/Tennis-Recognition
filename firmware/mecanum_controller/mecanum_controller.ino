/**
 * MKS TinyBee 麦轮速度流控制器 (Velocity Node - UDP Wireless)
 * 专为高动态视觉伺服/网球拦截任务设计
 * * 硬件状态：
 * - 驱动插槽：使用后四个槽位 (E1, E0, Z, Y)
 * - 通信方式：ESP32 自建 Wi-Fi 热点 + UDP 协议
 * - 控制协议：VEL <vx> <vy> <w> (单位: m/s, rad/s)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// ============================================================================
// I2S 引脚与寄存器级极速宏 (ESP32)
// ============================================================================
#define I2S_WS   26
#define I2S_BCK  25
#define I2S_DATA 27

#define FAST_HIGH(pin) REG_WRITE(GPIO_OUT_W1TS_REG, 1UL << (pin))
#define FAST_LOW(pin)  REG_WRITE(GPIO_OUT_W1TC_REG, 1UL << (pin))

// ============================================================================
// 虚拟引脚 bit 位 (跳过 X 轴)
// ============================================================================
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

// 按照实际物理插槽映射 (已修复槽位顺序)
const MotorBits MB[4] = {
  {V_E1_STEP, V_E1_DIR, V_E1_EN },  // M0: 左前轮 -> 绑定 E1 插槽
  {V_E0_STEP, V_E0_DIR, V_E0_EN },  // M1: 右前轮 -> 绑定 E0 插槽
  {V_Z_STEP,  V_Z_DIR,  V_Z_EN  },  // M2: 左后轮 -> 绑定 Z  插槽
  {V_Y_STEP,  V_Y_DIR,  V_Y_EN  },  // M3: 右后轮 -> 绑定 Y  插槽
};

// ============================================================================
// 物理运动学参数
// ============================================================================
const int   MICROSTEPS = 8;         // 驱动器拨码须为 8 细分
const int   STEPS_PER_REV = 200 * MICROSTEPS;
const float WHEEL_RADIUS = 0.023f;  // 轮半径 23mm
const float LX = 0.15f;             // 轴距 X (需根据实际车体微调)
const float LY = 0.12f;             // 轮距 Y (需根据实际车体微调)
const float L_SUM = LX + LY;

const uint32_t VEL_TIMEOUT_MS = 500; // 看门狗：超过500ms未收到指令则自动刹车

// ============================================================================
// Wi-Fi 与 UDP 配置
// ============================================================================
const char *AP_SSID = "TinyBee_Robot";  // ESP32 发出的热点名称
const char *AP_PASS = "12345678";       // ESP32 热点密码
const uint16_t UDP_PORT = 8888;         // 监听端口

WiFiUDP udp;
char packetBuffer[255]; // UDP 接收缓冲区

// ============================================================================
// 底层高频脉冲引擎 (20kHz ISR)
// ============================================================================
volatile uint32_t stream_bits = 0; 
#define STREAM_HZ 20000
volatile int32_t step_acc[4] = {0, 0, 0, 0};
const int32_t ACC_SCALE = STREAM_HZ * 256; 
volatile float step_rate[4]  = {0, 0, 0, 0};

hw_timer_t *timer = nullptr;

void IRAM_ATTR isr_20khz() {
  uint32_t bits = stream_bits;
  
  // 1. 24位移位数据到 3片 74HC595
  for (int b = 23; b >= 0; b--) {
    if ((bits >> b) & 1) FAST_HIGH(I2S_DATA);
    else                 FAST_LOW(I2S_DATA);
    
    // 拉长 Setup/Hold 时间
    FAST_LOW(I2S_BCK);  
    FAST_HIGH(I2S_BCK); FAST_HIGH(I2S_BCK); 
    FAST_LOW(I2S_BCK);  
  }
  
  // 锁存输出
  FAST_LOW(I2S_WS);
  FAST_HIGH(I2S_WS); FAST_HIGH(I2S_WS);
  FAST_LOW(I2S_WS);

  // 2. 清理脉冲 (拉低 STEP 引脚)
  for(int i = 0; i < 4; i++) stream_bits &= ~(1 << MB[i].step); 

  // 3. Bresenham 步进频率生成器
  for (int i = 0; i < 4; i++) {
    float r = step_rate[i];
    if (r == 0.0f) continue;
    
    // 设置方向电平
    if (r > 0) stream_bits &= ~(1 << MB[i].dir);
    else       stream_bits |=  (1 << MB[i].dir);

    float abs_r = (r > 0.0f) ? r : -r;
    step_acc[i] += (int32_t)(abs_r * 256.0f);
    
    if (step_acc[i] >= ACC_SCALE) {
      step_acc[i] -= ACC_SCALE;
      stream_bits |= (1 << MB[i].step); // 触发步进高电平
    }
  }
}

// ============================================================================
// 全局状态变量
// ============================================================================
uint32_t last_cmd_time = 0;
bool motors_enabled = false;

// ============================================================================
// Setup
// ============================================================================
void setup() {
  Serial.begin(115200); 
  delay(100);

  // 初始化 I2S 引脚
  pinMode(I2S_WS, OUTPUT);   digitalWrite(I2S_WS, LOW);
  pinMode(I2S_BCK, OUTPUT);  digitalWrite(I2S_BCK, LOW);
  pinMode(I2S_DATA, OUTPUT); digitalWrite(I2S_DATA, LOW);

  // 默认使能所有电机驱动 (低电平有效)
  stream_bits = 0;
  for (int i = 0; i < 4; i++) stream_bits &= ~(1 << MB[i].en);  

  // 启动 Wi-Fi 热点 (AP 模式)
  Serial.println("\nStarting Wi-Fi AP...");
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("AP IP Address: ");
  Serial.println(WiFi.softAPIP()); // 默认通常为 192.168.4.1

  // 监听 UDP 端口
  udp.begin(UDP_PORT);
  Serial.printf("Listening on UDP port %d\n", UDP_PORT);

  // 启动 20kHz 硬件定时器中断
  timer = timerBegin(1000000); 
  timerAttachInterrupt(timer, &isr_20khz);
  timerAlarm(timer, 50, true, 0);
  timerRestart(timer); 

  Serial.println("VELOCITY NODE RDY (WIRELESS)");
}

// ============================================================================
// Loop
// ============================================================================
void loop() {
  // 解析 UDP 数据包
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(packetBuffer, 255);
    if (len > 0) {
      packetBuffer[len] = 0; // 字符串结尾符
      String buf = String(packetBuffer);
      buf.trim();
      
      // 指令格式解析
      if (buf.startsWith("VEL")) {
        float vx=0, vy=0, w=0;
        if (sscanf(buf.c_str(), "VEL %f %f %f", &vx, &vy, &w) >= 1) {
          
          float rps[4];
          // ==================================================================
          // 逆运动学 (IK) - 已修复 O 型/错位安装导致的横移与自转反转问题
          // ==================================================================
          rps[0] = (vx - vy - w * L_SUM) / WHEEL_RADIUS / (2.0f * PI); // 左前
          rps[1] = (vx + vy + w * L_SUM) / WHEEL_RADIUS / (2.0f * PI); // 右前
          rps[2] = (vx - vy + w * L_SUM) / WHEEL_RADIUS / (2.0f * PI); // 左后 (修正)
          rps[3] = (vx + vy - w * L_SUM) / WHEEL_RADIUS / (2.0f * PI); // 右后 (修正)
          
          // 换算为内部步进频率 (Hz)
          for (int i=0; i<4; i++) {
            step_rate[i] = rps[i] * STEPS_PER_REV;
          }
          last_cmd_time = millis();
          motors_enabled = true;
        }
      } else if (buf == "STOP") {
        for (int i=0; i<4; i++) step_rate[i] = 0;
        motors_enabled = false;
      }
    }
  }

  // 安全看门狗：如果超过设定时间没有收到上位机的连续指令，强制停车
  if (motors_enabled && (millis() - last_cmd_time > VEL_TIMEOUT_MS)) {
    for (int i=0; i<4; i++) step_rate[i] = 0;
    motors_enabled = false;
    Serial.println("WARN: UDP TIMEOUT - MOTORS STOPPED");
  }
}