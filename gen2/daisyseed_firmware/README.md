# daisyseed_firmware

Firmware for the two Electrosmith Daisy Seed boards that read the vehicle's
four hydrophones (96 kHz codec inputs, two per board). The **master** board
reads hydrophones 0/1 and talks to the Orin over USB serial; the **slave**
board reads hydrophones 2/3 and reports its levels to the master as analog
voltages (slave DAC → master ADC A0/A1).

## Building and flashing

Requires `gcc-arm-none-eabi`, `dfu-util`, and `libDaisy/` + `DaisySP/` built at
the repo root (the Makefile expects them at `../../`).

```sh
make CURRENT_PROGRAM=<name>          # build only (default: master_level)
make flash CURRENT_PROGRAM=<name>    # build + flash over USB, no buttons
make flash DAISY_PORT=/dev/serial/by-id/usb-Electrosmith_...  # if 2+ boards attached
```

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
- **`plot/plot_hydrophones.py`** — plots 4-channel hydrophone levels from
  saved serial logs (`HH:MM:SS:ms -> Mic0..3` lines); supports `--start/--end`
  cropping and `--save` to PNG. Example datasets in `plot/data/*.txt` (various
  FFT sizes/frequencies from pool and TDOA tests).
- **`build/`** — compiler output (git-ignored).
