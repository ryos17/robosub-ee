# Stanford RoboSub — Electrical Repo

Electrical subsystem for the Stanford RoboSub AUV. It includes PCB designs (KiCad), firmware, Python tools, and research.

🌐 [stanfordrobosub.org](https://www.stanfordrobosub.org/)

## Repository layout

- **`platformio.ini`** — PlatformIO config for building/flashing the Teensy firmware. `src_dir` points at one sketch folder; change it to pick which firmware gets flashed (see comments in the file).
- **`gen1/`** — First hardware generation: PCBs only. Deprecated; the sub no longer exists.
- **`gen2/`** — Second (current) generation: PCBs and firmware for the audio and Teensy boards.
- **`gen3/`** — Third generation (in progress): ongoing designs of PCBs with Micro Breakout, Battery Boards, Power Distribution.
- **`imported_parts/`** — Shared KiCad libraries: symbols, footprints, 3D models, import script, and the JLC parts order log.
- **`misc/`** — Older and experimental material.
- **`research/`** — Research work (whisper-based intervehicle communication).
