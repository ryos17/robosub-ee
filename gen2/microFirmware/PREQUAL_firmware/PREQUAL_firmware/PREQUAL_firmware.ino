#include <Servo.h>

// Servos
Servo servos[8];   // Array to store servo objects
const int servoPins[8] = {4, 3, 0, 2, 1, 6, 5, 7};
int lastThrusterPWM[8] = {1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500};

// Serial setup
char inputBuffer[64]; // Buffer to store incoming serial data
int bufferPosition = 0; // Position in the buffer

// Indicators
const int greenIndicatorLedPin = 37; // LED1
const int killSwitchPin = 26;
bool firstIteration = false;

void setup() {
    // Begin serial
    Serial.begin(9600);

    // Initialize all servos
    config_servos();

    // Initialize indicator
    config_indicator();

    Serial.println("Initialize Complete");
}

void config_servos() {
    for (int i = 0; i < 8; i++) {
        servos[i].attach(servoPins[i]);   // Attach servo
        servos[i].writeMicroseconds(1500);  // Start at neutral
    }
}

void config_indicator() {
    pinMode(greenIndicatorLedPin, OUTPUT);
    pinMode(killSwitchPin, INPUT);
    digitalWrite(greenIndicatorLedPin, HIGH);
}

void loop() {
    // Indicator (kill switch) logic
    int killSwitch = digitalRead(killSwitchPin); // Read the value from the pin (0, loose = nominal or 1, tighten = kill)
    if (killSwitch == 0 && firstIteration) {
        // Un-killed: restore previous thruster command and resume going forward
        digitalWrite(greenIndicatorLedPin, HIGH);
        for (int i = 0; i < 8; i++) {
            servos[i].writeMicroseconds(lastThrusterPWM[i]);
        }
        firstIteration = false;
    } else if (killSwitch == 1) { // If KILLED, neutralize motors
        config_servos();
        digitalWrite(greenIndicatorLedPin, LOW);
        firstIteration = true;
        return; // Skip further processing when killed
    }

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
    int s0, s1, s2, s3, s4, s5, s6, s7;

    if (sscanf(input, "%d %d %d %d %d %d %d %d", &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7) == 8) {   // Setting thrusters
        int vals[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
        for (int i = 0; i < 8; i++) {
            servos[i].writeMicroseconds(vals[i]);
            lastThrusterPWM[i] = vals[i];
        }

        // Acknowledge command for Orin
        Serial.println("> ok");
    }
}
