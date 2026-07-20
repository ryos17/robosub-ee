# Audio Board Firmware (Electrosmith Daisy Seed)

Firmware for the two Electrosmith Daisy Seed boards that read the vehicle's
four hydrophones (96 kHz codec inputs, two per board). The **master** board
reads hydrophones 0/1 and talks to the Orin over USB serial; the **slave**
board reads hydrophones 2/3 and reports its levels to the master as analog
voltages (slave DAC → master ADC A0/A1).

**Currently deployed:** both boards are kept flashed with `stream_audio.cpp`,
which continuously streams their hydrophone audio to the Orin (for the
[`research/whisper_ivc/`](../../research/whisper_ivc/) work). The other
programs below are selected per build via `CURRENT_PROGRAM`.

## libDaisy (vendored, not committed)

The Makefile links against Electrosmith's [libDaisy](https://github.com/electro-smith/libDaisy)
core, which lives in `libDaisy/` **inside this directory**. It is deliberately
**not** committed to this repo — it is a large third-party tree with its own git
submodules — so it is git-ignored (`gen2/daisyseed_firmware/libDaisy/` in the
repo-root `.gitignore`). After a fresh clone you have to fetch and build it once:

```sh
cd gen2/daisyseed_firmware

# 1. clone libDaisy *with its submodules* into ./libDaisy
git clone --recurse-submodules https://github.com/electro-smith/libDaisy.git

# (optional, for a reproducible build) pin to the commit this firmware was last
# built against:
git -C libDaisy checkout c02245d22b38acad3916d9c2f156bcba34fa15af
git -C libDaisy submodule update --init --recursive

# 2. build the core static library -> libDaisy/build/libdaisy.a
make -C libDaisy
```

That produces `libDaisy/build/libdaisy.a`, which the firmware Makefile picks up
via `LIBDAISY_DIR = ./libDaisy/`. DaisySP is **not** used, so it does not need to
be cloned or built.

## Building and flashing

Requires `gcc-arm-none-eabi`, `dfu-util`, and `libDaisy/` built in this directory
(see [libDaisy](#libdaisy-vendored-not-committed) above; the Makefile expects it
at `./libDaisy/`).

> **Flash from a laptop, not the Orin.** Flashing over the Orin's USB has been
> very flaky (marginal hub/ports, `-71`/`-110` enumeration errors — see the
> troubleshooting notes below). Whenever possible, flash the Daisy boards from a
> laptop and only move them onto the Orin once they're confirmed running good
> firmware.

```sh
make CURRENT_PROGRAM=<name>          # build only (default: master_level)
make flash CURRENT_PROGRAM=<name>    # build + flash over USB, no buttons
make flash DAISY_PORT=/dev/serial/by-id/usb-Electrosmith_...  # if 2+ boards attached
```

With both boards attached, pass `DAISY_PORT` to pick one (serials are the
current wiring: `573433` = slave, `533433` = master). **Always wipe `build/`
first** — the Makefile reuses it across programs, so leftover objects from a
different `CURRENT_PROGRAM` produce a broken binary (symptom: the flashed board
hangs and won't enumerate on USB):

```sh
# slave.cpp -> slave board
rm -rf build && make flash CURRENT_PROGRAM=slave \
     DAISY_PORT=/dev/serial/by-id/usb-Electrosmith_Daisy_Seed_Built_In_376C36573433-if00

# master_level -> master board. The master's serial can enumerate garbled on a
# marginal link (e.g. 376636603433 instead of 376C36533433), so its hard-coded
# by-id path may not exist. Pick whatever Electrosmith by-id is NOT the slave:
rm -rf build && make flash CURRENT_PROGRAM=master_level \
     DAISY_PORT=/dev/serial/by-id/$(ls /dev/serial/by-id | grep -i electrosmith | grep -v 376C36573433)
```

Never interrupt a flash (e.g. don't run it in a tmux session that might die
mid-write) — a partial write leaves the board hung and non-enumerating. If that
happens, or a board otherwise won't enumerate, force DFU with the buttons
(hold **BOOT**, tap **RESET**, release) and program it directly instead of
`make flash` (which can't `reboot` a hung/DFU board):

```sh
rm -rf build && make CURRENT_PROGRAM=<name>
make program-dfu CURRENT_PROGRAM=<name>          # board must already be in DFU
```

On a marginal USB hub/port/cable the flash can be flaky (`-71`/`-110`
enumeration errors in `dmesg`), with two symptoms worth knowing:

- **`dfu-util: No DFU capable USB device available`** even though the board is
  running good firmware — the board rebooted but its DFU device enumerated
  *after* the Makefile's short wait. This wrote nothing (safe); just re-run
  `make flash` — the next run finds it already in DFU and programs it. A few
  retries usually gets it. If it never enumerates on one port, use a different
  port (the hub here has known-bad ports).
- **Garbled `by-id` serial** — a marginal link can corrupt the USB serial
  descriptor, so a board shows up as e.g. `376636603433` instead of its real
  `376C36533433`. The `DAISY_PORT` path above then won't match. Target it by
  whatever Electrosmith `by-id` is present that *isn't* the other board:
  `DAISY_PORT=/dev/serial/by-id/$(ls /dev/serial/by-id | grep -i electrosmith | grep -v <other-board-serial>)`.

`make flash` sends `reboot` over the Daisy's USB serial port. The firmware's
`SerialLibrary` recognizes that command and calls `System::ResetToBootloader()`,
dropping the MCU into the STM DFU bootloader so `dfu-util` can program it —
no BOOT/RESET button presses needed. Caveats:

- Works only if the board is currently running firmware built from this tree
  (anything using `SerialLibrary`). A board running older firmware, or a
  crashed board, needs the buttons once: hold **BOOT**, tap **RESET**, release,
  then `make program-dfu`.
- The slave normally has no USB data connection; plug its micro-USB into the
  Orin to flash it.
- Use the stable `/dev/serial/by-id/usb-Electrosmith_Daisy_Seed_Built_In_*`
  paths, not `ttyACM*` (the Teensy also enumerates as ttyACM).

## Main programs (one `CURRENT_PROGRAM` at a time)

- **`master_level.cpp`** — deployed master firmware. FFTs hydrophones 0/1
  (96 kHz, 64-point FFT) for the magnitude at a target frequency (currently
  25 kHz, ±1%), reads the slave's two levels on ADC pins A0/A1, and prints all
  four normalized levels (`hydrophone_log: Mic0..Mic3`) over USB serial every
  1 ms. Calibration constants (`hydrophone_*_max`, gain, target frequency) at
  the top of the file; commented "Competition" block holds the pool settings.
- **`slave.cpp`** — deployed slave firmware. Same FFT level detection for
  hydrophones 2/3, but outputs the two normalized levels as 0–3.3 V on the two
  DAC pins (wired to the master's A0/A1). Runs standalone: serial is
  initialized non-blocking, so it boots without a USB host but still accepts
  `reboot` if one is attached.
- **`master_ping.cpp`** — ping-direction detector. On the serial command
  `ping`, watches all four channels for a 10 s window for a burst at the
  target frequency (14 080 Hz) and prints `hydrophone:front` / `back` /
  `inconclusive` based on which hydrophones detected it first (TDOA of the
  earliest two detections).
- **`master_tdoa.cpp`** — continuous TDOA logger at 1 760 Hz using 1024-point
  FFTs. Prints per-hydrophone threshold-crossing timestamps (µs) and their
  differences over USB serial, for acoustic localization experiments.
- **`stream_audio.cpp`** — raw audio streamer for the Whisper intervehicle-
  communication work (`research/whisper_ivc/`). Decimates this board's two
  codec channels 96 kHz → 16 kHz and streams them as framed s16 PCM over USB
  CDC (132-byte frames, `0xDA 0x7A` magic). Board-agnostic: flash on master
  for hydrophones 0/1, on the slave for 2/3. Host side:
  `research/whisper_ivc/stream_transcribe.py`.

## Test programs

- **`frequency_level_test.cpp`** — frequency-level detection test at 35 kHz
  (64-point FFT). Prints instantaneous and 4-sample-averaged magnitudes every
  100 ms and flags a detection when 2 consecutive averages exceed a 0.1
  threshold.
- **`pitch_track_test.cpp`** — pitch tracker (2048-point FFT, parabolic peak
  interpolation). Prints RMS amplitude and detected fundamental frequency
  every 100 ms; useful for verifying the codec inputs and FFT chain.
- **`test_air.cpp`** — in-air, 2-microphone TDOA test. On the serial command
  `start`, measures absolute detection times (µs) of a 1 760 Hz tone on both
  mics and prints which detected first plus the time difference (handles timer
  overflow).

## Library (`library/`)

- **`fft_library.h/.cpp`** — recursive radix-2 Cooley–Tukey FFT with Hanning
  windowing. `getFrequencyMagnitude()` sums bin magnitudes within a tolerance
  band around a target frequency (used by all level/TDOA programs);
  `detectPitch()` returns the fundamental via peak + parabolic interpolation.
- **`serial_library.h/.cpp`** — USB CDC command handling. `Init(bool wait)`
  starts logging (optionally without blocking for a host), `CheckCommand()`
  matches newline-terminated commands, `Poll()` processes input when no
  command is expected. Any received `reboot` line resets the board into the
  DFU bootloader (this is what `make flash` uses).

## Other

- **`Makefile`** — selects the program via `CURRENT_PROGRAM`, pulls in the
  libDaisy core build, and adds the button-free `flash` target described
  above.
- **`build/`** — compiler output (git-ignored).
