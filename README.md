# Stanford RoboSub — Electrical Repo

Electrical subsystem for the Stanford RoboSub AUV. It includes PCB designs (KiCad), firmware code, Python tools, and research endeavors.

🌐 [stanfordrobosub.org](https://www.stanfordrobosub.org/)

## Repository layout

- **`platformio.ini`** — PlatformIO config for building/flashing the Teensy firmware. `src_dir` points at one sketch folder; change it to pick which firmware gets flashed (see comments in the file).
- **`gen1/`** — First hardware generation: includes PCBs only.
- **`gen2/`** — Second (current) generation:
  - **`PCBs/`** — KiCad projects: MLU Breakout, Power Management, Power Distribution, Battery Hot Swap boards.
  - **`teensy_firmware/`** — Teensy 4.1 breakout firmware (Arduino sketches):
    - `STABLE_firmware/` — competition build, the one that flies
    - `DEV_firmware/` — work in progress
    - `PREQUAL_firmware/` — prequalification run build
    - `misc/` — one-off hardware tests (depth sensor, neopixels, …)
  - **`daisyseed_firmware/`** — Electrosmith DaisySeed hydrophone/audio firmware (TDOA, ping detection, level tests; built with the included Makefile).
  - **`software/`** — Python tools: `test_dvl.py`, `teensy_analyze.py`, `StarOddi.py`, hydrophone plotting.
- **`gen3/`** — Third generation (in progress): `PCBs/` with Micro Breakout, Battery Boards, Power Distribution.
- **`imported_parts/`** — Shared KiCad libraries: symbols, footprints, 3D models, the `jlc_to_robosub.py` import script, and the JLC parts order log.
- **`misc/`** — Older and experimental material.
- **`research/`** — Research endeavors (MBARI temperature profiles).
