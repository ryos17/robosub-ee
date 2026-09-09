# research/eli — underwater acoustics for OCEANS 2027 Aberdeen

Research toward an OCEANS 2027 Aberdeen paper: underwater acoustic algorithm
development running on the sub's Jetson AGX Orin, using the 4-hydrophone
array + Ping360, validated in the pool.

- `notes/04_research_gaps.md` — **start here**: the gaps we should study, in
  plain terms, ranked, with evidence and the closest competitor group.
- `notes/00_directions.md` — literature-driven survey of candidate directions.
- `notes/01_critique.md` — adversarial critique; P1 narrowed to the headline
  paper (start here for the current plan).
- `notes/02_hardware_updates.md` — Zoom H8 solves clock sync; Ping360 stack.
- `src/` — differentiable pool channel model, fit, matched-field localizer,
  probe WAV generator. Synthetic tests pass; see `src/README.md`.
- `papers/` — text extracts of key papers (arnumber-prefixed).

## Status (2026-09-09)

- Direction chosen: G1+G2 headline (walls as virtual array + self-calibrated
  differentiable pool model + on-Orin matched-field localizer), with G3
  ablation, G4 sonar validation, G5 Orin timing. G6 (self-localization from
  own echoes) is the follow-on.
- Code: synthetic pipeline passes (`src/README.md`). No real data yet.
- Hardware decisions: record on the Zoom H8 (one clock), hydrophones + speaker
  on a jig, Ping360 sweep per pose. Open: hydrophone variant, pool dimensions,
  speaker model.
- Abstract deadline ~7 Dec 2026. Pool session 1 target: first week of October.

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
