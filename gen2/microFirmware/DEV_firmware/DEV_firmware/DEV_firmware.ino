#include <Servo.h>
#include <Wire.h>

// Servos
Servo servos[8];   // Array to store servo objects
const int servoPins[8] = {4, 3, 0, 2, 1, 6, 5, 7};
int lastThrusterPWM[8] = {1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500};   

// Serial setup
char inputBuffer[64]; // Buffer to store incoming serial data
int bufferPosition = 0; // Position in the buffer
const int DIGITS = 8;

// Depth sensor
#include "MS5837.h"
MS5837 sensor;

// high accuracy temperature sensor
#include <Adafruit_MCP9808.h>
Adafruit_MCP9808 mcp9808_sensor = Adafruit_MCP9808();

// environmental sensor (temp/pressure/humidity)
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
Adafruit_BME280 bme280_sensor; 

// Battery sensor
const int voltagePin = 40;
const int currentPin = 41;

// ============================================================================
// BNO085 9-DOF IMU (Teyleten Robot GY-BNO085 module) -- CURRENTLY DISABLED
// ============================================================================
// Secondary IMU wired to the Teensy over I2C. This is NOT the XSENS MTi (that
// one talks to the Orin directly over USB and never touches this firmware).
// The BNO085 runs CEVA's SH-2 protocol and does sensor fusion ON-CHIP, so
// instead of raw axes it hands us fused outputs:
//   - rotation vector: absolute orientation as a unit quaternion (w,x,y,z)
//   - calibrated gyroscope: angular velocity, rad/s
//   - linear acceleration: m/s^2 with gravity already removed
//   - calibrated magnetometer: uT
//
// TO ENABLE once the sensor is physically connected:
//   1. Wire the module to the Teensy 4.1 I2C1 bus:
//        module SDA -> Teensy pin 17 (SDA1)
//        module SCL -> Teensy pin 16 (SCL1)
//        VIN -> 3.3V, GND -> GND
//      (Module has onboard pullups; if the bus is flaky add external
//      2.2k-4.7k pullups to 3.3V.)
//   2. Make sure "adafruit/Adafruit BNO08x" is in lib_deps for the
//      teensy41_dev env in platformio.ini (already added).
//   3. Flip ENABLE_BNO085 to 1 below and re-flash.
//   4. If init fails, the module may be strapped to the alternate I2C
//      address: change BNO085_I2C_ADDR from 0x4A to 0x4B.
#define ENABLE_BNO085 0

#if ENABLE_BNO085
#include <Adafruit_BNO08x.h>

const uint8_t BNO085_I2C_ADDR = 0x4A;         // default; 0x4B if ADR pin pulled high
#define BNO085_WIRE Wire1                     // Teensy 4.1 I2C1: SDA1 = pin 17, SCL1 = pin 16
const uint32_t BNO085_REPORT_INTERVAL_US = 10000; // 100 Hz per report

Adafruit_BNO08x bno085(-1);                   // -1: reset pin not wired
bool bno085Ok = false;                        // init succeeded; gates all reads

// Latest fused values. poll_bno085() refreshes these every loop pass; the
// telemetry line just reports whatever is cached here.
float bnoQuatW = 1, bnoQuatX = 0, bnoQuatY = 0, bnoQuatZ = 0; // orientation quaternion
float bnoGyroX = 0, bnoGyroY = 0, bnoGyroZ = 0;               // rad/s
float bnoAccX  = 0, bnoAccY  = 0, bnoAccZ  = 0;               // m/s^2, gravity removed
float bnoMagX  = 0, bnoMagY  = 0, bnoMagZ  = 0;               // uT
#endif

// Indicators
const int greenIndicatorLedPin = 37; // LED1
const int redIndicatorLedPin = 36; // LED2
const int killSwitchPin = 26;

// ===== Kill switch ISR state =====
volatile bool isKilled = false;               // Latched when ISR fires
constexpr int KILL_ASSERTED_LEVEL = HIGH; // killed when HIGH (flip if this changes)
volatile unsigned long killISRMicros = 0;

constexpr unsigned long KILL_ISR_DEBOUNCE_US = 3000; // ~3 ms
volatile unsigned long _lastKillIsrUs = 0;

// ===== Kill island config =====
// While the kill switch is asserted the firmware sits in a hard, blocking loop
// (enterKillIsland) that does nothing but idle every actuator and periodically
// tell the Orin the sub is killed. KILL_MSG_PERIOD_MS is how often that message
// is sent.
constexpr unsigned long KILL_MSG_PERIOD_MS = 500;

// ===== Red LED status config =====
// The red LED no longer mirrors the green (kill) LED. It now signals:
//   - flashing "... pause ..." : Orin heartbeat lost (highest priority)
//   - solid ON                 : battery below LOW_BATTERY_VOLTAGE
//   - OFF                      : nominal
constexpr float LOW_BATTERY_VOLTAGE = 14.5f;
constexpr float LOW_BATTERY_HYSTERESIS = 0.2f;   // must recover above 14.7 V to clear
constexpr unsigned long BATTERY_CHECK_PERIOD_MS = 500;
bool lowBattery = false;
unsigned long lastBatteryCheckMs = 0;

// Orin heartbeat watchdog. The Orin-side sender is not implemented yet, so the
// watchdog only arms after the FIRST "heartbeat N" message is received --
// until then the red LED stays quiet instead of flashing forever.
constexpr unsigned long ORIN_HB_TIMEOUT_MS = 3000;
bool orinHeartbeatSeen = false;
unsigned long lastOrinHeartbeatMs = 0;

// Flash pattern for a dead heartbeat: three short blinks, pause, repeat.
constexpr unsigned long HB_BLINK_MS = 150;   // per on/off segment
constexpr unsigned long HB_PAUSE_MS = 900;   // gap between blink bursts

// External lumen lights
Servo lightServo;
int lumenPin = 8;
int lightVal = 1100;
int lightCycles = 5;

// SD Card
#include <SD.h>
constexpr bool ENABLE_SD_CARD = false;   // set true to re-enable SD card logging
const int chipSelect = BUILTIN_SDCARD;
unsigned long loopIterationCounter = 0; // for perioidic SD logging
int sdLoggingFrequency = 20000; 

// dropper
Servo dropper;
const int dropperPin = 21;  // change this depending
const int dropperMinUS = 800;     // SER-201X spec
const int dropperMaxUS = 2200;    // SER-201X spec
float dropperHalfRangeDeg = 70.0f;   // default ±70° per datasheet (use 100.0f if reprogrammed)
float dropperDeg = 0;

// torpedo
Servo torpedo;
const int torpedoPin = 20;
const int torpedoMinUS = 800;     // SER-201X spec
const int torpedoMaxUS = 2200;    // SER-201X spec
float torpedoHalfRangeDeg = 70.0f;   // default ±70° per datasheet (use 100.0f if reprogrammed)
float torpedoDeg = 0;

// ======= Forward declarations =======
void applyNeutralAll();
void killISR();
void handleSerial();
void enterKillIsland();
bool killSwitchAsserted();
void setKillLeds(bool killed);
void updateRedLed(bool heartbeatMonitored);

void setup() {
    // Begin serial and I2C (via wire)
    Serial.begin(9600);
    Wire.begin();
    
    config_servos();
    config_dropper(); 
    config_torpedo();
    config_depth();
    config_battery();
    config_internal_sensors();
    config_bno085();      // no-op unless ENABLE_BNO085 is 1
    config_indicator();   // sets up interrupt
    config_lumen();
    config_sd_card();
    
    Serial.println("Initialize Complete");
}

void config_servos() {
    for (int i = 0; i < 8; i++) {
        servos[i].attach(servoPins[i]);   // Attach servo
        servos[i].writeMicroseconds(1500);  // Start at neutral
    }
}

// Map degrees (-HALF_RANGE..+HALF_RANGE) to microseconds (dropperMinUS..dropperMaxUS)
int degToUS_generic(float deg, float halfRangeDeg, int usMin, int usMax) {
  if (deg < -halfRangeDeg) deg = -halfRangeDeg;
  if (deg > +halfRangeDeg) deg = +halfRangeDeg;
  float t = (deg + halfRangeDeg) / (2.0f * halfRangeDeg); // 0..1
  return (int)(usMin + t * (usMax - usMin) + 0.5f);
}

// Specific wrappers
inline int dropperDegToUS(float d)  { return degToUS_generic(d,  dropperHalfRangeDeg,  dropperMinUS,  dropperMaxUS); }
inline int torpedoDegToUS(float d)  { return degToUS_generic(d,  torpedoHalfRangeDeg,  torpedoMinUS,  torpedoMaxUS); }


void config_dropper() {
  dropper.attach(dropperPin, dropperMinUS, dropperMaxUS);
  dropper.writeMicroseconds(dropperDegToUS(0));
}

void config_torpedo() {
  torpedo.attach(torpedoPin, torpedoMinUS, torpedoMaxUS);
  torpedo.writeMicroseconds(torpedoDegToUS(0)); // neutral/safe
}

void config_depth() {
    sensor.setModel(MS5837::MS5837_02BA); // model number for Bar02
    sensor.init();
    sensor.setFluidDensity(997); // kg/m^3 (997 freshwater, 1029 for seawater)
}

void config_battery() {
    pinMode(currentPin, INPUT);
    pinMode(voltagePin, INPUT);
}

void config_internal_sensors() {
   // Initialize the MCP9808 High-Accuracy Temperature Sensor.
  // The return value is ignored to prevent serial output.
  mcp9808_sensor.begin();

  // Initialize the BME280 Environmental Sensor.
  // Assumes the I2C address is 0x77. If you have issues, try 0x77.
  // The return value is ignored to prevent serial output.
  bme280_sensor.begin(0x77);
}

// ===================== BNO085 IMU functions =====================
#if ENABLE_BNO085

// Tell the BNO085 which fused outputs to stream and at what rate. Called at
// init and again whenever the chip reports it reset itself (reports do not
// survive a reset).
static void bno085EnableReports() {
  bno085.enableReport(SH2_ROTATION_VECTOR,           BNO085_REPORT_INTERVAL_US);
  bno085.enableReport(SH2_GYROSCOPE_CALIBRATED,      BNO085_REPORT_INTERVAL_US);
  bno085.enableReport(SH2_LINEAR_ACCELERATION,       BNO085_REPORT_INTERVAL_US);
  bno085.enableReport(SH2_MAGNETIC_FIELD_CALIBRATED, BNO085_REPORT_INTERVAL_US);
}

// One-time init on the I2C1 bus. If the sensor is missing/unresponsive we just
// leave bno085Ok false and everything else keeps working without it.
void config_bno085() {
  BNO085_WIRE.begin();
  bno085Ok = bno085.begin_I2C(BNO085_I2C_ADDR, &BNO085_WIRE);
  if (bno085Ok) {
    bno085EnableReports();
  } else {
    Serial.println("[BNO085] init FAILED (check wiring / try addr 0x4B)");
  }
}

// Drain every report the sensor has queued since the last pass and cache the
// newest value of each kind. Non-blocking; called once per loop() pass.
void poll_bno085() {
  if (!bno085Ok) return;

  // The BNO085 resets itself occasionally (brownout, internal watchdog);
  // report subscriptions are lost when that happens, so re-enable them.
  if (bno085.wasReset()) {
    bno085EnableReports();
  }

  sh2_SensorValue_t event;
  while (bno085.getSensorEvent(&event)) {
    switch (event.sensorId) {
      case SH2_ROTATION_VECTOR:            // absolute orientation quaternion
        bnoQuatW = event.un.rotationVector.real;
        bnoQuatX = event.un.rotationVector.i;
        bnoQuatY = event.un.rotationVector.j;
        bnoQuatZ = event.un.rotationVector.k;
        break;
      case SH2_GYROSCOPE_CALIBRATED:       // angular velocity, rad/s
        bnoGyroX = event.un.gyroscope.x;
        bnoGyroY = event.un.gyroscope.y;
        bnoGyroZ = event.un.gyroscope.z;
        break;
      case SH2_LINEAR_ACCELERATION:        // m/s^2, gravity removed
        bnoAccX = event.un.linearAcceleration.x;
        bnoAccY = event.un.linearAcceleration.y;
        bnoAccZ = event.un.linearAcceleration.z;
        break;
      case SH2_MAGNETIC_FIELD_CALIBRATED:  // uT
        bnoMagX = event.un.magneticField.x;
        bnoMagY = event.un.magneticField.y;
        bnoMagZ = event.un.magneticField.z;
        break;
      default:
        break;
    }
  }
}

// Extra "key: value" fields appended to the main "> ..." telemetry line. The
// Orin-side parser (arduino.py) splits that line into a key/value dict, so
// new keys are picked up without breaking the existing ones.
String bno085Telemetry() {
  if (!bno085Ok) return "";
  return " quat_w: "  + String(bnoQuatW, DIGITS) +
         " quat_x: "  + String(bnoQuatX, DIGITS) +
         " quat_y: "  + String(bnoQuatY, DIGITS) +
         " quat_z: "  + String(bnoQuatZ, DIGITS) +
         " gyro_x: "  + String(bnoGyroX, DIGITS) +
         " gyro_y: "  + String(bnoGyroY, DIGITS) +
         " gyro_z: "  + String(bnoGyroZ, DIGITS) +
         " accel_x: " + String(bnoAccX, DIGITS) +
         " accel_y: " + String(bnoAccY, DIGITS) +
         " accel_z: " + String(bnoAccZ, DIGITS) +
         " mag_x: "   + String(bnoMagX, DIGITS) +
         " mag_y: "   + String(bnoMagY, DIGITS) +
         " mag_z: "   + String(bnoMagZ, DIGITS);
}

// Human-readable dump for the "imu" debug command (like "batt" / "depth").
void print_bno085_debug() {
  if (!bno085Ok) {
    Serial.println("BNO085 enabled in firmware but init failed (check wiring / address)");
    return;
  }
  poll_bno085();  // grab the freshest data before printing
  Serial.println("Quat(wxyz): " + String(bnoQuatW, DIGITS) + " " + String(bnoQuatX, DIGITS) + " " + String(bnoQuatY, DIGITS) + " " + String(bnoQuatZ, DIGITS));
  Serial.println("Gyro rad/s: " + String(bnoGyroX, DIGITS) + " " + String(bnoGyroY, DIGITS) + " " + String(bnoGyroZ, DIGITS));
  Serial.println("LinAccel m/s^2: " + String(bnoAccX, DIGITS) + " " + String(bnoAccY, DIGITS) + " " + String(bnoAccZ, DIGITS));
  Serial.println("Mag uT: " + String(bnoMagX, DIGITS) + " " + String(bnoMagY, DIGITS) + " " + String(bnoMagZ, DIGITS));
}

#else  // ENABLE_BNO085 == 0: no-op stubs so call sites need no #if guards

void config_bno085() {}
void poll_bno085() {}
String bno085Telemetry() { return ""; }
void print_bno085_debug() {
  Serial.println("BNO085 disabled (set ENABLE_BNO085 to 1 in firmware)");
}

#endif  // ENABLE_BNO085

void config_indicator() {
  pinMode(greenIndicatorLedPin, OUTPUT);
  pinMode(redIndicatorLedPin, OUTPUT);
  pinMode(killSwitchPin, INPUT); // use INPUT_PULLUP + invert logic if needed

  // Seed state from live pin so LED matches reality at boot
  int ks = digitalRead(killSwitchPin);
  isKilled = (ks == KILL_ASSERTED_LEVEL);
  digitalWrite(greenIndicatorLedPin, isKilled ? LOW : HIGH);
  digitalWrite(redIndicatorLedPin, LOW);   // red is status-only now; starts off

  attachInterrupt(digitalPinToInterrupt(killSwitchPin), killISR, CHANGE);
}


void config_lumen() {
  lightServo.attach(lumenPin);
  lightServo.writeMicroseconds(1100); // Turn off the light
}

void config_sd_card() {
  if (!ENABLE_SD_CARD) return;
  SD.begin(chipSelect);
  write_data_sd("Configuring");
}

// ===== Interrupt Service Routine =====
void killISR() {
  unsigned long nowUs = micros();
  if (nowUs - _lastKillIsrUs < KILL_ISR_DEBOUNCE_US) return;
  _lastKillIsrUs = nowUs;

  int ks = digitalRead(killSwitchPin);
  bool asserted = (ks == KILL_ASSERTED_LEVEL);

  isKilled = asserted;
  killISRMicros = nowUs;

  #ifdef digitalWriteFast
    digitalWriteFast(greenIndicatorLedPin, asserted ? LOW : HIGH);
  #else
    digitalWrite(greenIndicatorLedPin, asserted ? LOW : HIGH);
  #endif
}

// Apply neutral to all 8 outputs and remember values
void applyNeutralAll() {
  for (int i = 0; i < 8; i++) {
    servos[i].writeMicroseconds(1500);
    lastThrusterPWM[i] = 1500;
  }
  // Also neutralize these:
  dropper.writeMicroseconds(dropperDegToUS(14));
  torpedo.writeMicroseconds(torpedoDegToUS(0));
}

void loop() {
    // The kill switch is checked FIRST, every pass. The instant it is asserted we
    // drop into the kill island and do not come back until it is re-enabled.
    if (killSwitchAsserted() || isKilled) {
        enterKillIsland();   // blocks until the kill switch is released
        return;              // restart loop() clean once released
    }

    setKillLeds(false);
    updateRedLed(true);   // battery + Orin-heartbeat status
    poll_bno085();        // keep cached IMU values fresh (no-op if disabled)

    // ===== Normal operation (not killed) =====
    // SD Card periodic logs
    loopIterationCounter++;
    if (loopIterationCounter % sdLoggingFrequency == 0) {
        logPeriodicData();
    }

    // Handle serial input
    handleSerial();
}

// True while the kill switch is physically asserted. Read straight from the pin
// (independent of the ISR) so island entry/exit can never miss an edge.
bool killSwitchAsserted() {
    return digitalRead(killSwitchPin) == KILL_ASSERTED_LEVEL;
}

// Kill indicator: green LED only. Off when killed, on when nominal.
// The red LED is driven independently by updateRedLed().
void setKillLeds(bool killed) {
    digitalWrite(greenIndicatorLedPin, killed ? LOW : HIGH);
}

// Red LED status machine. Non-blocking; call every loop pass.
//   heartbeatMonitored=false suppresses the heartbeat-dead flash (used inside
//   the kill island, where the UART is drained so heartbeats can't arrive).
// Priority: heartbeat-dead flash > low-battery solid > off.
void updateRedLed(bool heartbeatMonitored) {
    unsigned long now = millis();

    // Sample the battery on a slow tick, with hysteresis so the LED doesn't
    // flicker around the threshold.
    if (now - lastBatteryCheckMs >= BATTERY_CHECK_PERIOD_MS) {
        lastBatteryCheckMs = now;
        float current, voltage;
        handle_battery_command(current, voltage);
        if (lowBattery) {
            if (voltage > LOW_BATTERY_VOLTAGE + LOW_BATTERY_HYSTERESIS) lowBattery = false;
        } else if (voltage < LOW_BATTERY_VOLTAGE) {
            lowBattery = true;
        }
    }

    bool hbDead = heartbeatMonitored && orinHeartbeatSeen &&
                  (now - lastOrinHeartbeatMs > ORIN_HB_TIMEOUT_MS);

    if (hbDead) {
        // "... pause ...": three HB_BLINK_MS blinks (on/off), then HB_PAUSE_MS dark.
        unsigned long cycle = 6 * HB_BLINK_MS + HB_PAUSE_MS;
        unsigned long t = now % cycle;
        bool on = (t < 6 * HB_BLINK_MS) && ((t / HB_BLINK_MS) % 2 == 0);
        digitalWrite(redIndicatorLedPin, on ? HIGH : LOW);
    } else {
        digitalWrite(redIndicatorLedPin, lowBattery ? HIGH : LOW);
    }
}

// ========================= KILL ISLAND =========================
// A hard, blocking loop entered the instant the kill switch is asserted. Until the
// switch is physically released this does ONLY two things:
//   1) hold every actuator at its passive/neutral value (motors idled), and
//   2) send a periodic "sub is killed" message to the Orin.
// Nothing else runs: no commands processed, no sensors read, no SD logging. Any
// bytes the Orin sends are drained and discarded. The one and only way out is the
// kill switch being re-enabled (de-asserted).
void enterKillIsland() {
    isKilled = true;
    setKillLeds(true);
    applyNeutralAll();
    bufferPosition = 0;   // drop any half-received command

    unsigned long beat = 0;
    unsigned long lastBeat = millis();
    // Immediate first notice so the Orin knows the moment we entered.
    Serial.println(String("[KILLED] sub is killed  n=") + beat + " uptime_ms=" + millis());

    while (killSwitchAsserted()) {
        // 1) Keep every actuator idle on each pass (defends against any glitch).
        applyNeutralAll();

        // Keep the low-battery indication alive while killed. Heartbeat-dead
        // flashing is suppressed here: the island drains the UART, so a
        // missing heartbeat means nothing during a kill.
        updateRedLed(false);

        // 2) Periodically tell the Orin we are still killed.
        unsigned long now = millis();
        if (now - lastBeat >= KILL_MSG_PERIOD_MS) {
            lastBeat = now;
            beat++;
            Serial.println(String("[KILLED] sub is killed  n=") + beat + " uptime_ms=" + now);
        }

        // Discard anything the Orin sends; nothing acts while killed.
        while (Serial.available() > 0) (void)Serial.read();
    }

    // ---- Kill switch released: leave the island cleanly ----
    isKilled = false;
    setKillLeds(false);
    // Heartbeats couldn't arrive during the kill; restart the watchdog grace
    // period so the red LED doesn't flash the instant we resume.
    lastOrinHeartbeatMs = millis();
    bufferPosition = 0;
    while (Serial.available() > 0) (void)Serial.read();   // start with an empty buffer
    Serial.println(String("[READY] kill cleared. Resuming normal operation. uptime_ms=") + millis());
}

// Read and dispatch any complete newline-terminated commands from the UART.
// Used only in normal operation (the kill island drains UART on its own).
void handleSerial() {
    while (Serial.available() > 0) {
        char inChar = (char)Serial.read();
        if (inChar == '\n' || inChar == '\r') { // End of one command
            inputBuffer[bufferPosition] = '\0'; // Null-terminate the string
            process_input(inputBuffer);
            bufferPosition = 0; // Reset buffer for the next command
        } else {
            if (bufferPosition < (int)sizeof(inputBuffer) - 1) { // Prevent buffer overflow
                inputBuffer[bufferPosition++] = inChar;
            }
        }
    }
}

void process_input(char *input) {
  // ---- Heartbeat (normal operation) ----
  // Consumed SILENTLY: heartbeats arrive once a second, so acking them would
  // spam every serial monitor on the port. Set HB_ACK_REPLY to 1 to get an
  // "[HB] ack ..." reply per heartbeat when debugging the link.
  // While the kill switch is asserted the firmware is in the kill island, which
  // drains UART -- heartbeats are NOT seen during a kill; the periodic
  // "[KILLED] sub is killed" message is sent instead.
  #define HB_ACK_REPLY 0
  long hbSeq;
  if (sscanf(input, "heartbeat %ld", &hbSeq) == 1) {
    orinHeartbeatSeen = true;         // arms the red-LED heartbeat watchdog
    lastOrinHeartbeatMs = millis();
    #if HB_ACK_REPLY
    Serial.println("[HB] ack seq=" + String(hbSeq) +
                   " uptime_ms=" + String(millis()) +
                   " isKilled=" + String(isKilled ? 1 : 0));
    #endif
    return;
  }

  // Hard safety gate (defense in depth): never act on an actuator command while
  // killed. In practice the kill island already owns the UART during a kill, so
  // this is unreachable then, but keep it so no path can move an actuator mid-kill.
  if (isKilled) return;

  // ---- Quick neutralize ('q') ----
  // Type "q" to slam all 8 thrusters back to 1500 us (neutral). Emergency-stop
  // convenience for the serial monitor; does not touch dropper/torpedo/light.
  if (strcmp(input, "q") == 0) {
    for (int i = 0; i < 8; i++) {
      servos[i].writeMicroseconds(1500);
      lastThrusterPWM[i] = 1500;
    }
    Serial.println("Thrusters neutralized: 1500 1500 1500 1500 1500 1500 1500 1500");
    return;
  }

  int s0, s1, s2, s3, s4, s5, s6, s7;
  int servoNum, val;

  if (sscanf(input, "%d %d %d %d %d %d %d %d", &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7) == 8) {   // Setting servos
    // Set servos
    servos[0].writeMicroseconds(s0);
    servos[1].writeMicroseconds(s1);
    servos[2].writeMicroseconds(s2);
    servos[3].writeMicroseconds(s3);
    servos[4].writeMicroseconds(s4);
    servos[5].writeMicroseconds(s5);
    servos[6].writeMicroseconds(s6);
    servos[7].writeMicroseconds(s7);

    // Record last thruster value
    lastThrusterPWM[0] = s0; 
    lastThrusterPWM[1] = s1;
    lastThrusterPWM[2] = s2;
    lastThrusterPWM[3] = s3;
    lastThrusterPWM[4] = s4;
    lastThrusterPWM[5] = s5;
    lastThrusterPWM[6] = s6;
    lastThrusterPWM[7] = s7;

    // THIS MUST BE KEPT THE SAME IN ORDER TO WORK WITH ORIN CODE
    float pressure, temperature, depth;
    handle_depth_command(pressure, temperature, depth);

    float internalTemp1, internalTemp2, humidity;
    handle_internalSensors_command(internalTemp1, internalTemp2, humidity);

    float current, voltage;
    handle_battery_command(current, voltage);

    // Print those data for orin. bno085Telemetry() appends the IMU fields
    // when the BNO085 is enabled and alive, and is an empty string otherwise.
    Serial.println("> pressure: " + String(pressure, DIGITS) + " external_temperature: " + String(temperature, DIGITS) + " depth: " + String(depth, DIGITS) + " internal_temperature1: " + String(internalTemp1, DIGITS) + " internal_temperatur2: " + String(internalTemp2, DIGITS) + " humidity: " + String(humidity, DIGITS) + " current: " + String(current, DIGITS) + " voltage: " + String(voltage, DIGITS) + bno085Telemetry());
  
  } else if (strcmp(input, "batt") == 0) {   // Debugging battery
    Serial.println("TESTING BATTERY");
    float current, voltage;
    handle_battery_command(current, voltage);
    Serial.println("Current:" + String(current, DIGITS) + " voltage:" + String(voltage, DIGITS));

  } else if (sscanf(input, "%d %d", &servoNum, &val) == 2) {   // Turn on single servo
     Serial.println("TESTING SINGLE SERVO");
    if (val >= 1100 && val <= 1900 && servoNum >= 0 && servoNum <= 7) {
        servos[servoNum].writeMicroseconds(val);
        lastThrusterPWM[servoNum] = val;
    } else {
        Serial.println("Invalid command");
    }

  } else if (strcmp(input, "imu") == 0) {   // Debugging BNO085 IMU
    Serial.println("TESTING BNO085 IMU");
    print_bno085_debug();

  } else if (strcmp(input, "depth") == 0) {   // Debugging depth sensor
    Serial.println("TESTING DEPTH SENSOR");
    float pressure, temperature, depth;
    handle_depth_command(pressure, temperature, depth);
    Serial.println("Pressure: " + String(pressure, DIGITS) + " mbar, Temperature: " + String(temperature, DIGITS) + " °C, Depth: " + String(depth, DIGITS) + " m");

  } else if (strcmp(input, "test") == 0) {   // Debugging all servos
    Serial.println("RUNNING SERVO TEST SCRIPT");
    test_servos();

  } else if (sscanf(input, "light %d", &lightVal) == 1) {   // Turn on lumen lights
    if (lightVal >= 1100 && lightVal <= 1900) {
      lightServo.writeMicroseconds(lightVal);
      Serial.println("Light set to " + String(lightVal));
    } else {
      Serial.println("Invalid light value. Please enter a value between 1100 and 1900.");
    }

  } else if (sscanf(input, "light gradient %d", &lightCycles) == 1) {   // Graient lumen lights
    Serial.println("LIGHT GRADIENT SET");
    gradient_lumen_light(lightCycles);

  // Droppers
  } else if (strcmp(input, "pearl-harbor") == 0) {   // dropper left
    int us = dropperDegToUS(22);
    dropper.writeMicroseconds(us);
  } else if (strcmp(input, "iwo-jima") == 0) {   // dropper right
    int us = dropperDegToUS(-22);
    dropper.writeMicroseconds(us);
  } else if (sscanf(input, "dropperPosition %f", &dropperDeg) == 1) {  // dropperDeg is a float
    // Clamp for safety
    if (dropperDeg < -dropperHalfRangeDeg) dropperDeg = -dropperHalfRangeDeg;
    if (dropperDeg > +dropperHalfRangeDeg) dropperDeg = +dropperHalfRangeDeg;

    int us = dropperDegToUS(dropperDeg);
    dropper.writeMicroseconds(us);

    Serial.print("Commanded: ");
    Serial.print(dropperDeg, 1);
    Serial.print(" dropperDeg  ->  ");
    Serial.print(us);
    Serial.println(" us");
  
  // Torpedos
  } else if (strcmp(input, "hiroshima") == 0) {
    int us = torpedoDegToUS(-9);
    torpedo.writeMicroseconds(us);
  } else if (strcmp(input, "nagasaki") == 0) {
    int us = torpedoDegToUS(37);
    torpedo.writeMicroseconds(us);
  } else if (strcmp(input, "enola-gay") == 0) {
    int us = torpedoDegToUS(14);
    torpedo.writeMicroseconds(us);
  } else if (sscanf(input, "torpedoPosition %f", &torpedoDeg) == 1) {
    if (torpedoDeg < -torpedoHalfRangeDeg) torpedoDeg = -torpedoHalfRangeDeg;
    if (torpedoDeg > +torpedoHalfRangeDeg) torpedoDeg = +torpedoHalfRangeDeg;
    int us = torpedoDegToUS(torpedoDeg);
    torpedo.writeMicroseconds(us);
    Serial.print("Commanded: ");
    Serial.print(torpedoDeg, 1);
    Serial.print(" torpedoDeg  ->  ");
    Serial.print(us);
    Serial.println(" us");
  } else if (strcmp(input, "transfer") == 0) {   // SD Card Transfer
    transfer_sd_log();
  } else if (strcmp(input, "delete") == 0) {   // SD Card Delete
    delete_sd_log();
  } else {
    Serial.println(input);
  }
}

void handle_depth_command(float& pressure, float& temperature, float& depth) {
  sensor.read();
  pressure = sensor.pressure(); // mbar
  temperature = sensor.temperature(); // C
  depth = sensor.depth(); // m
}

void handle_internalSensors_command(float& internalTemp1, float& internalTemp2, float& humidity){
    internalTemp1 = mcp9808_sensor.readTempC();
    internalTemp2 = bme280_sensor.readTemperature();
    humidity = bme280_sensor.readHumidity();
}

void handle_battery_command(float& current, float& voltage) {
  current = (analogRead(currentPin) * 120.0) / 1024; // A
  voltage = (analogRead(voltagePin) * 60.0) / 1024; // V
}

void test_servos() {
  for (int i = 0; i < 8; i++) {
    // Set the current servo to 1550 microseconds
    servos[i].writeMicroseconds(1550);
    lastThrusterPWM[i] = 1550;
    delay(2000);  // wait 500 ms for the servo to move
    // Return the servo to its neutral position (1500 microseconds)
    servos[i].writeMicroseconds(1500);
    lastThrusterPWM[i] = 1500;
    delay(500);  // wait 500 ms before moving to the next servo
  }
}

void logPeriodicData() {
  float current, voltage;
  handle_battery_command(current, voltage);

  float pressure, temperature, depth;
  handle_depth_command(pressure, temperature, depth);
  
  String dataString = "current:" + String(current, DIGITS) +
                  " voltage:" + String(voltage, DIGITS) +
                  " servo:"; 
  for (int i = 0; i < 8; i++) {
    dataString += "PWM:" + String(lastThrusterPWM[i]);
    if (i < 7)
      dataString += ",";
  }
  
  dataString += " pressure:" + String(pressure, DIGITS) +
                " temperature:" + String(temperature, DIGITS) +
                " depth:" + String(depth, DIGITS);

  dataString += " killSwitchPin:" + String(digitalRead(killSwitchPin)) +
                " isKilled:" + String(isKilled ? 1 : 0);

  write_data_sd(dataString);
}

void transfer_sd_log(){
  if (!ENABLE_SD_CARD) { Serial.println("SD card disabled."); return; }
  if (SD.exists("datalog.txt")) {
    Serial.println("Transfering datalog.txt ...");
    // Open the file for reading
    File dataFile = SD.open("datalog.txt", FILE_READ);
    if (dataFile) {
      // Read the file and send its content to the Serial port
      while (dataFile.available()) {
        Serial.write(dataFile.read());
      }
      dataFile.close();
      Serial.println("\nFile transfer complete. Save the output on your local device as datalog.txt.");
    } else {
      Serial.println("Error opening datalog.txt for reading.");
    }
  } else {
    Serial.println("datalog.txt does not exist.");
  }
}

void delete_sd_log() {
  if (!ENABLE_SD_CARD) { Serial.println("SD card disabled."); return; }
  if (SD.exists("datalog.txt")) {
      // Remove the file
      if (SD.remove("datalog.txt")) {
        Serial.println("datalog.txt deleted.");
      } else {
        Serial.println("Error deleting datalog.txt.");
      }
      // Recreate the file
      File dataFile = SD.open("datalog.txt", FILE_WRITE);
      if (dataFile) {
        dataFile.close();
        Serial.println("datalog.txt recreated.");
      } else {
        Serial.println("Error recreating datalog.txt.");
      }
    } else {
      Serial.println("datalog.txt does not exist.");
    }
}

void write_data_sd(String dataString) {
  if (!ENABLE_SD_CARD) return;
  // Get the timestamp
  String timestamp = getTimestamp();
  // open the file.
  File dataFile = SD.open("datalog.txt", FILE_WRITE);
  // if the file is available, write to it
  if (dataFile) {
    dataFile.println(timestamp + " -> " + dataString);
    dataFile.close();
  } else {
    // if the file isn't open, pop up an error:
    Serial.println("error opening datalog.txt");
  }
}

void gradient_lumen_light(int cycles) {
  for (int i = 0; i < cycles; i++) {
    for (int lightVal = 1100; lightVal <=1900; lightVal++) {
      lightServo.writeMicroseconds(lightVal);
      delay(1);
    }
    for (int lightVal = 1900; lightVal >=1100; lightVal--) {
      lightServo.writeMicroseconds(lightVal);
      delay(1);
    }
  }
}

String getTimestamp() {
  unsigned long milliseconds = millis();
  unsigned long seconds = milliseconds / 1000;
  unsigned long minutes = seconds / 60;
  unsigned long hours = minutes / 60;

  milliseconds %= 1000;
  seconds %= 60;
  minutes %= 60;

  // Format the timestamp
  String timestamp = "";
  if (hours < 10) timestamp += "0"; // Add leading zero for hours
  timestamp += String(hours) + ":";
  if (minutes < 10) timestamp += "0"; // Add leading zero for minutes
  timestamp += String(minutes) + ":";
  if (seconds < 10) timestamp += "0"; // Add leading zero for seconds
  timestamp += String(seconds) + ".";
  if (milliseconds < 100) timestamp += "0"; // Add leading zeros for milliseconds
  if (milliseconds < 10) timestamp += "0";
  timestamp += String(milliseconds);

  return timestamp;
}
