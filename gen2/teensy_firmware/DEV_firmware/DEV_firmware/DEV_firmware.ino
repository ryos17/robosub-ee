// ============================================================================
// STABLE_firmware -- competition build for the Teensy 4.1 breakout board.
//
// IMPORTANT NOTES (keep in sync with the repo README):
//
// Thrusters:
//   * DO NOT RUN THRUSTERS AT FULL POWER FOR THE TIME BEING -- 20 to 30% max.
//   * DO NOT RUN FOR MORE THAN 10 SECONDS WHILE OUTSIDE OF WATER.
//   * PWM assignment: 1100 us = full reverse, 1500 us = off/neutral,
//     1900 us = full forward.
//
// Comms (Orin <-> Teensy over USB serial, 9600 baud, newline-terminated):
//   * "s0 s1 s2 s3 s4 s5 s6 s7"  -- set all 8 thruster PWMs; replies with the
//     "> ..." telemetry line the Orin parses (format is load-bearing, see
//     process_input()). Values are clamped to THRUSTER_MIN_US..THRUSTER_MAX_US.
//   * "thruster_number pwm_value" -- set a single thruster (same clamp range).
//   * "q" -- software emergency stop: all thrusters to 1500 us immediately.
//   * "heartbeat N" -- Orin liveness; arms the red-LED heartbeat watchdog.
//   * Debug commands: "batt", "depth", "test", "light <us>",
//     "light gradient <n>", dropper/torpedo commands, "transfer", "delete".
//
// Kill switch (physical, overrides everything):
//   * Fail-safe: pin HIGH (switch open / wire broken) = KILLED.
//   * While asserted the firmware blocks in the kill island: all actuators
//     held neutral, UART drained, periodic "[KILLED]" message to the Orin.
// ============================================================================

#include <Servo.h>
#include <Wire.h>

// Thrusters (driven as servos via ESCs)
Servo servos[8];
const int servoPins[8] = {4, 3, 0, 2, 1, 6, 5, 7};
int lastThrusterPWM[8] = {1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500};

// Hard PWM limits enforced on EVERY thruster command (full ESC range is
// 1100-1900 us; kept tighter on purpose while the "20-30% max power" rule is
// in effect). A corrupted or out-of-range value is clamped, never passed raw.
constexpr int THRUSTER_MIN_US = 1200;
constexpr int THRUSTER_MAX_US = 1800;

inline int clampThrusterUS(int us) {
  if (us < THRUSTER_MIN_US) return THRUSTER_MIN_US;
  if (us > THRUSTER_MAX_US) return THRUSTER_MAX_US;
  return us;
}

// Serial command buffer
char inputBuffer[64];
int bufferPosition = 0;
const int DIGITS = 8;   // decimal places in telemetry output

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

// ===== Cached sensor readings =====
// The MS5837 read blocks ~40 ms (two ADC conversions with delays inside the
// BlueRobotics lib) and the I2C sensors add more. Doing that inline for every
// thruster command capped the command rate and put a possible I2C stall in
// the control path. Instead, refresh_sensor_cache() samples everything on a
// slow tick in loop() and the telemetry reply just reads these cached values.
bool depthSensorOk = false;   // MS5837 init succeeded; gates all reads
constexpr unsigned long SENSOR_REFRESH_MS = 100;
unsigned long lastSensorRefreshMs = 0;
float cachedPressure = 0, cachedExternalTemp = 0, cachedDepth = 0;
float cachedInternalTemp1 = 0, cachedInternalTemp2 = 0, cachedHumidity = 0;

// Indicators
const int greenIndicatorLedPin = 37; // LED1
const int redIndicatorLedPin = 36; // LED2
const int killSwitchPin = 26;

// ===== Kill switch ISR state =====
volatile bool isKilled = false;               // Latched when ISR fires
constexpr int KILL_ASSERTED_LEVEL = HIGH; // killed when HIGH. Switch CLOSED shorts pin to GND (LOW) = running; switch
                                          // OPEN lets the 4.7k + internal pull-up drive the pin HIGH = dead. Any wiring
                                          // fault (open/disconnected pin) therefore fails safe to HIGH = dead.
constexpr unsigned long KILL_ISR_DEBOUNCE_US = 3000; // ~3 ms
volatile unsigned long _lastKillIsrUs = 0;

// ===== Kill-switch instability watchdog (easily disabled) =====
// A dying / intermittent kill switch does not make a clean HIGH<->LOW
// transition -- it FLICKERS the line rapidly. Rather than stutter the thrusters
// by toggling running<->killed on every edge, we treat a burst of edges as a
// fault and LATCH into the killed (dead) state -- fail-safe. The latch clears
// only after the line has read running (de-asserted), with no new edges, for
// INSTABILITY_CLEAR_STABLE_MS continuously.
//
// To DISABLE this feature entirely, set ENABLE_INSTABILITY_KILL to 0: the raw
// kill-switch level then fully controls the sub, flicker and all.
#define ENABLE_INSTABILITY_KILL 1
constexpr unsigned int  INSTABILITY_EDGE_THRESHOLD  = 10;    // edges within the window to trip
constexpr unsigned long INSTABILITY_WINDOW_MS       = 500;   // evaluation window
constexpr unsigned long INSTABILITY_CLEAR_STABLE_MS = 2000;  // line must be stably running this long to clear the latch
volatile unsigned int   killEdgeCount = 0;          // raw kill-pin edges, bumped in the ISR (pre-debounce)
volatile bool           instabilityLatched = false; // latched killed because the line was judged unreliable

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
unsigned long loopIterationCounter = 0; // for periodic SD logging
int sdLoggingFrequency = 20000;   // log every N loop passes

// dropper
Servo dropper;
const int dropperPin = 21;
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
void updateInstabilityWatchdog();
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
    config_indicator();   // sets up interrupt
    config_lumen();
    config_sd_card();
    
    Serial.println("Initialize Complete");
}

void config_servos() {
    for (int i = 0; i < 8; i++) {
        servos[i].attach(servoPins[i]);
        servos[i].writeMicroseconds(1500);  // start at neutral
    }
}

// Map degrees (-halfRangeDeg..+halfRangeDeg) to microseconds (usMin..usMax)
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
    // If init fails (sensor unplugged, I2C fault) all depth reads are skipped:
    // calling sensor.read() with a dead sensor can stall the I2C bus, and that
    // stall would sit in the control loop.
    depthSensorOk = sensor.init();
    if (!depthSensorOk) Serial.println("[DEPTH] MS5837 init FAILED -- depth telemetry will read 0");
    sensor.setFluidDensity(997); // kg/m^3 (997 freshwater, 1029 for seawater)
}

void config_battery() {
    pinMode(currentPin, INPUT);
    pinMode(voltagePin, INPUT);
}

void config_internal_sensors() {
  // Return values ignored on purpose: a missing sensor just reads garbage in
  // telemetry instead of blocking boot or spamming the Orin's serial parser.
  mcp9808_sensor.begin();
  bme280_sensor.begin(0x77);   // if init fails, try the alternate address 0x76
}

void config_indicator() {
  pinMode(greenIndicatorLedPin, OUTPUT);
  pinMode(redIndicatorLedPin, OUTPUT);
  pinMode(killSwitchPin, INPUT_PULLUP); // external 4.7k + internal pull-up to 3V3. Switch CLOSED = LOW = running;
                                        // switch OPEN (or any disconnect) floats HIGH = dead. Pull-up reinforces the
                                        // fail-safe (dead) direction and stops the pin floating to ~Vdd/2.

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

#if ENABLE_INSTABILITY_KILL
  // Count EVERY raw edge (before the debounce return below) so a flicker storm
  // is still visible even when edges arrive closer together than the debounce.
  killEdgeCount++;
#endif

  if (nowUs - _lastKillIsrUs < KILL_ISR_DEBOUNCE_US) return;
  _lastKillIsrUs = nowUs;

  int ks = digitalRead(killSwitchPin);
  bool asserted = (ks == KILL_ASSERTED_LEVEL);

  isKilled = asserted;

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
    if (killSwitchAsserted() || isKilled
#if ENABLE_INSTABILITY_KILL
        || instabilityLatched
#endif
       ) {
        enterKillIsland();   // blocks until the kill switch is released (and stable)
        return;              // restart loop() clean once released
    }

    setKillLeds(false);
    updateRedLed(true);   // battery + Orin-heartbeat status
    updateInstabilityWatchdog();  // latch to killed if the kill line is flickering (no-op if disabled)
    refresh_sensor_cache();       // slow-tick sensor sampling; telemetry replies read the cache

    // ===== Normal operation (not killed) =====
    // SD Card periodic logs
    if (ENABLE_SD_CARD) {
        loopIterationCounter++;
        if (loopIterationCounter % sdLoggingFrequency == 0) {
            logPeriodicData();
        }
    }

    // Handle serial input
    handleSerial();
}

// Sample all slow sensors (MS5837 depth, MCP9808, BME280) into the cached_*
// globals on a SENSOR_REFRESH_MS tick. This is the ONLY place the blocking
// MS5837 read happens; command handling never waits on a sensor.
void refresh_sensor_cache() {
    unsigned long now = millis();
    if (now - lastSensorRefreshMs < SENSOR_REFRESH_MS) return;
    lastSensorRefreshMs = now;

    if (depthSensorOk) {
        sensor.read();                          // blocks ~40 ms
        cachedPressure     = sensor.pressure(); // mbar
        cachedExternalTemp = sensor.temperature(); // C
        cachedDepth        = sensor.depth();    // m
    }
    cachedInternalTemp1 = mcp9808_sensor.readTempC();
    cachedInternalTemp2 = bme280_sensor.readTemperature();
    cachedHumidity      = bme280_sensor.readHumidity();
}

// True while the kill switch is physically asserted. Read straight from the pin
// (independent of the ISR) so island entry/exit can never miss an edge.
bool killSwitchAsserted() {
    return digitalRead(killSwitchPin) == KILL_ASSERTED_LEVEL;
}

// Kill-switch instability watchdog. Called once per loop pass during normal
// operation. Counts raw kill-pin edges over a tumbling INSTABILITY_WINDOW_MS
// window; if too many occur the line is judged unreliable and we latch killed.
// The latch is only cleared later, inside the kill island, once the line has
// been stably running again. No-op when ENABLE_INSTABILITY_KILL is 0.
void updateInstabilityWatchdog() {
#if ENABLE_INSTABILITY_KILL
    static unsigned long windowStartMs = 0;
    unsigned long now = millis();
    if (now - windowStartMs < INSTABILITY_WINDOW_MS) return;

    noInterrupts();
    unsigned int edges = killEdgeCount;
    killEdgeCount = 0;
    interrupts();
    windowStartMs = now;

    if (edges >= INSTABILITY_EDGE_THRESHOLD && !instabilityLatched) {
        instabilityLatched = true;
        isKilled = true;
        Serial.println(String("[KILL-INSTAB] kill line unstable (") + edges +
                       " edges in " + INSTABILITY_WINDOW_MS + " ms) -> latching KILLED");
    }
#endif
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

#if ENABLE_INSTABILITY_KILL
    // Start of the current run of "stably running" time used to clear the latch.
    unsigned long stableRunningSinceMs = millis();
    noInterrupts(); killEdgeCount = 0; interrupts();   // ignore edges from before entry
#endif

    while (killSwitchAsserted()
#if ENABLE_INSTABILITY_KILL
           || instabilityLatched
#endif
          ) {
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

#if ENABLE_INSTABILITY_KILL
        // Try to clear an instability latch. The line must read running
        // (de-asserted) AND produce no new edges for INSTABILITY_CLEAR_STABLE_MS
        // continuously; any assertion or flicker restarts the stability timer.
        if (instabilityLatched) {
            noInterrupts(); unsigned int edges = killEdgeCount; killEdgeCount = 0; interrupts();
            if (killSwitchAsserted() || edges > 0) {
                stableRunningSinceMs = now;
            } else if (now - stableRunningSinceMs >= INSTABILITY_CLEAR_STABLE_MS) {
                instabilityLatched = false;
                Serial.println(String("[KILL-INSTAB] kill line stable again -> clearing latch  uptime_ms=") + now);
            }
        }
#endif

        // Discard anything the Orin sends; nothing acts while killed.
        while (Serial.available() > 0) (void)Serial.read();
    }

    // ---- Kill switch released (and stable): leave the island cleanly ----
    isKilled = false;
    setKillLeds(false);
#if ENABLE_INSTABILITY_KILL
    noInterrupts(); killEdgeCount = 0; interrupts();   // fresh window for normal operation
#endif
    // Heartbeats couldn't arrive during the kill; restart the watchdog grace
    // period so the red LED doesn't flash the instant we resume.
    lastOrinHeartbeatMs = millis();
    bufferPosition = 0;
    while (Serial.available() > 0) (void)Serial.read();   // start with an empty buffer
    Serial.println(String("[READY] kill cleared. Resuming normal operation. uptime_ms=") + millis());
}

// Read and dispatch any complete newline-terminated commands from the UART.
// Used only in normal operation (the kill island drains UART on its own).
// NOTE: commands longer than sizeof(inputBuffer)-1 (63 chars) are silently
// truncated -- the overflow bytes are dropped and the truncated prefix is
// still parsed at the newline. Since actuator commands must now parse
// EXACTLY (see process_input), a truncated command fails parsing and gets
// echoed back instead of driving anything.
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

  int s[8];
  int servoNum, val;
  int consumed = 0;   // set by %n: how much of the input the sscanf matched

  // ---- Set all 8 thrusters ----
  // The trailing "%n + end-of-string" check makes the parse EXACT: a garbled
  // line (e.g. a corrupted digit mid-message) that only partially parses no
  // longer falls through to the 2-integer single-thruster branch below and
  // silently drives one thruster. Partial/garbled lines match nothing and get
  // echoed back by the final else.
  if (sscanf(input, "%d %d %d %d %d %d %d %d %n",
             &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7], &consumed) == 8
      && input[consumed] == '\0') {
    for (int i = 0; i < 8; i++) {
      int us = clampThrusterUS(s[i]);
      servos[i].writeMicroseconds(us);
      lastThrusterPWM[i] = us;
    }

    // Telemetry reply from the sensor cache (refreshed on a 100 ms tick in
    // loop(); no blocking sensor reads here). The key names MUST stay exactly
    // as-is (including the "internal_temperatur2" typo) -- the Orin-side
    // parser matches on them. snprintf into a static buffer instead of String
    // concatenation so the hot path does no heap allocation.
    float current, voltage;
    handle_battery_command(current, voltage);
    static char telem[320];
    snprintf(telem, sizeof(telem),
             "> pressure: %.*f external_temperature: %.*f depth: %.*f"
             " internal_temperature1: %.*f internal_temperatur2: %.*f"
             " humidity: %.*f current: %.*f voltage: %.*f",
             DIGITS, cachedPressure, DIGITS, cachedExternalTemp, DIGITS, cachedDepth,
             DIGITS, cachedInternalTemp1, DIGITS, cachedInternalTemp2,
             DIGITS, cachedHumidity, DIGITS, current, DIGITS, voltage);
    Serial.println(telem);

  } else if (strcmp(input, "batt") == 0) {   // Debugging battery
    Serial.println("TESTING BATTERY");
    float current, voltage;
    handle_battery_command(current, voltage);
    Serial.println("Current:" + String(current, DIGITS) + " voltage:" + String(voltage, DIGITS));

  } else if (sscanf(input, "%d %d %n", &servoNum, &val, &consumed) == 2
             && input[consumed] == '\0') {   // Set single thruster (exact parse, same clamp range)
    Serial.println("TESTING SINGLE SERVO");
    if (val >= THRUSTER_MIN_US && val <= THRUSTER_MAX_US && servoNum >= 0 && servoNum <= 7) {
        servos[servoNum].writeMicroseconds(val);
        lastThrusterPWM[servoNum] = val;
    } else {
        Serial.println("Invalid command");
    }

  } else if (strcmp(input, "depth") == 0) {   // Debugging depth sensor (reads the cache)
    Serial.println("TESTING DEPTH SENSOR");
    if (!depthSensorOk) Serial.println("MS5837 init failed at boot -- values are stale zeros");
    Serial.println("Pressure: " + String(cachedPressure, DIGITS) + " mbar, Temperature: " + String(cachedExternalTemp, DIGITS) + " °C, Depth: " + String(cachedDepth, DIGITS) + " m");

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

  } else if (sscanf(input, "light gradient %d", &lightCycles) == 1) {   // Sweep lumen light brightness
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
    // Unrecognized command: echo it back verbatim. This is INTENTIONAL -- the
    // echo is how the Orin (and anyone on a serial monitor) sees that a line
    // arrived but matched nothing, without a prefix the telemetry parser
    // would have to special-case.
    Serial.println(input);
  }
}

void handle_battery_command(float& current, float& voltage) {
  current = (analogRead(currentPin) * 120.0) / 1024; // A
  voltage = (analogRead(voltagePin) * 60.0) / 1024; // V
}

// Pulse each thruster in turn at 1550 us (barely above neutral) for 2 s.
void test_servos() {
  for (int i = 0; i < 8; i++) {
    servos[i].writeMicroseconds(1550);
    lastThrusterPWM[i] = 1550;
    delay(2000);
    servos[i].writeMicroseconds(1500);
    lastThrusterPWM[i] = 1500;
    delay(500);
  }
}

// Depth values come from the sensor cache (refreshed on a 100 ms tick).
void logPeriodicData() {
  float current, voltage;
  handle_battery_command(current, voltage);

  float pressure = cachedPressure, temperature = cachedExternalTemp, depth = cachedDepth;

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

// Dump datalog.txt over serial so it can be saved on the host.
void transfer_sd_log(){
  if (!ENABLE_SD_CARD) { Serial.println("SD card disabled."); return; }
  if (SD.exists("datalog.txt")) {
    Serial.println("Transfering datalog.txt ...");
    File dataFile = SD.open("datalog.txt", FILE_READ);
    if (dataFile) {
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

// Delete datalog.txt and recreate it empty.
void delete_sd_log() {
  if (!ENABLE_SD_CARD) { Serial.println("SD card disabled."); return; }
  if (SD.exists("datalog.txt")) {
      if (SD.remove("datalog.txt")) {
        Serial.println("datalog.txt deleted.");
      } else {
        Serial.println("Error deleting datalog.txt.");
      }
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
  String timestamp = getTimestamp();
  File dataFile = SD.open("datalog.txt", FILE_WRITE);
  if (dataFile) {
    dataFile.println(timestamp + " -> " + dataString);
    dataFile.close();
  } else {
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

// Uptime as zero-padded "HH:MM:SS.mmm" (from millis(); no RTC).
String getTimestamp() {
  unsigned long milliseconds = millis();
  unsigned long seconds = milliseconds / 1000;
  unsigned long minutes = seconds / 60;
  unsigned long hours = minutes / 60;

  milliseconds %= 1000;
  seconds %= 60;
  minutes %= 60;

  String timestamp = "";
  if (hours < 10) timestamp += "0";
  timestamp += String(hours) + ":";
  if (minutes < 10) timestamp += "0";
  timestamp += String(minutes) + ":";
  if (seconds < 10) timestamp += "0";
  timestamp += String(seconds) + ".";
  if (milliseconds < 100) timestamp += "0";
  if (milliseconds < 10) timestamp += "0";
  timestamp += String(milliseconds);

  return timestamp;
}
