# Teensy 4.1 Firmware

Firmware for the gen 2 Teensy 4.1 breakout board: drives the eight thrusters,
reads the depth / temperature / humidity / pressure sensors, handles the
physical kill switch, and streams telemetry to the Orin over USB serial.

The firmware is kept as Arduino `.ino` sketches so they can be opened and
flashed from the Arduino IDE, or built/flashed with PlatformIO. Building and
flashing (including how to pick which sketch is flashed via `src_dir`) is
described in [`../../platformio.ini`](../../platformio.ini).

## Sketches

- **`STABLE_firmware/`** — the competition build. **This is what should be
  running on the sub.** If you're flashing the vehicle to operate it, flash
  this one.
- **`DEV_firmware/`** — the actively-developed build. Work in progress; expect
  it to be ahead of `STABLE_firmware` and not yet fully validated.
- **`PREQUAL_firmware/`** — the build used for the prequalification run.
- **`misc/`** — older and one-off sketches (sensor tests, linear-rail /
  strobe experiments, software-stack scratch code). Not part of the vehicle
  build.
