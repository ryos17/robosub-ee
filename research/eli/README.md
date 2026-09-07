# research/eli — underwater acoustics for OCEANS 2027 Aberdeen

Research toward an OCEANS 2027 Aberdeen paper: underwater acoustic algorithm
development running on the sub's Jetson AGX Orin, using the 4-hydrophone
array + Ping360, validated in the pool.

- `notes/00_directions.md` — literature-driven survey of candidate directions
  (start here).
- `papers/` — text extracts of key papers (arnumber-prefixed).

## Target venue

OCEANS 2027 Aberdeen, 21–24 June 2027, P&J Live. IEEE OES + MTS.
Abstract deadline tentatively **7 Dec 2026** (verify on site; user recalls
15 Jan 2027 — treat 7 Dec as the safe date). Full paper due later (typically
~Mar/Apr). Student Poster Competition: ~20 selected from 100+, travel +
registration funded, student must be lead author + presenter, full paper goes
to IEEE Xplore. Strong option for us.

Relevant tracks:
- 01 Underwater Acoustics & Acoustical Oceanography (sound propagation,
  calibration, signal coherence)
- 02 Sonar Signal/Image Processing & Communication (array processing,
  classification, acoustic telemetry, bio-inspired processing)
- 10 Ocean Vehicles (AUV navigation)
- Local track: Digital Innovation in Oceans, Underwater Security

## What we have

| Resource | Detail |
|---|---|
| Compute | NVIDIA Jetson AGX Orin, on-vehicle, CUDA/torch working (`whisper_ivc`) |
| Hydrophones | 4× Aquarian H2a/AS-1 class, audio-band; 96 kHz / 24-bit via two Daisy Seed boards (2 ch each). **No shared sample clock between boards** — cross-board TDOA needs sync/calibration. |
| Sonar | Blue Robotics Ping360, 750 kHz mechanically scanning single beam |
| Source | cheap underwater speaker, plays SD-card files (audio band, uncalibrated) |
| Pool | ~25 m, highly reverberant (concrete walls, shallow) |
| People | 3 at pool typically, more "data monkeys" on request |
| Existing sw | hydrophone → Orin streaming, Whisper transcription IVC demo |
