# research/eli — underwater acoustics on the sub's GPU, for OCEANS 2027 Aberdeen

## The idea in one breath

Cheap underwater speaker + 4 hydrophones + Jetson Orin → a fitted 3D acoustic
model of the pool **and** the 3D position of the speaker.

- The speaker plays chirps from a pole at surveyed spots. The hydrophones
  hear the direct sound and the echoes off every wall, floor and the surface.
- Usually those echoes are treated as noise. We use them: each wall is a
  mirror that places a virtual copy of the tiny array metres away, which is
  what gives *distance*, not just direction.
- A differentiable image-source model of the pool is fit to the recordings
  by gradient descent (the DAFx "differentiable reverb" trick, underwater).
  The fit also calibrates hydrophone positions, sound speed and wall
  reflectivity, so nobody needs to survey those to the millimetre.
- The fitted model is then the "template" a matched-field localizer on the
  Orin uses to find where a sound came from, in real time.
- Ground truth for the speaker position is a physical survey (tape measure,
  marked line, pole depth stop). **The Ping360 is not the ground truth**; it
  is a second opinion on where the walls are, to confirm the model inferred
  them correctly, and a rough jig-to-wall range cross-check.

Headline result we are after: 3D position error of a sound source in a
reverberant 25 m pool, from a 6 cm array, with and without the fitted
model, plus milliseconds and watts on the vehicle GPU.

## Reading order

| # | File | What it is | Read when |
|---|---|---|---|
| 1 | this file | idea, status, venue, hardware, plan, open questions | always |
| 2 | [notes/04_research_gaps.md](notes/04_research_gaps.md) | the six gaps in plain terms, ranked, with evidence and competitors | deciding what we claim |
| 3 | [notes/05_experiment_plan.md](notes/05_experiment_plan.md) | pool protocol, rig, probe signal, sessions, metrics, week-by-week | doing the work |
| 4 | [src/README.md](src/README.md) | the code and what the synthetic tests show | touching code |
| 5 | [notes/03_method_notes.md](notes/03_method_notes.md) | what to copy from the key papers; ranked code changes | improving the model |
| 6 | [notes/02_hardware_updates.md](notes/02_hardware_updates.md) | Zoom H8 clock fix; Ping360 software stack and options | hardware questions |
| 7 | [notes/01_critique.md](notes/01_critique.md) | adversarial critique of five directions; how P1 was narrowed | understanding why |
| 8 | [notes/00_directions.md](notes/00_directions.md) | original 390-paper survey and eight candidate directions | history / other ideas |
| 9 | [papers/README.md](papers/README.md) | index of full-text extracts with why each matters | citing |
| 10 | [notes/06_reconciliation_with_ryota.md](notes/06_reconciliation_with_ryota.md) | what Ryota's `research/ryota` found, corrections to these notes, overlap map, proposed merge | coordinating |

Later notes supersede earlier ones where they disagree (06 > 05 > 04 > 03 > 02 > 01 > 00).

## Status (2026-09-09)

- Direction fixed: the idea above (gaps G1+G2 in note 04), with three
  supporting sections: sim-to-real ablation (G3), Ping360 wall check (G4),
  Orin latency/power (G5). Follow-on paper: sub localizing itself from its
  own echoes (G6).
- Code: synthetic pipeline passes end to end (renderer matches
  pyroomacoustics; self-calibration recovers 1 cm hydrophone errors to 0.4 cm;
  held-out pose found to ~0 cm vs 71 cm with an unfitted model). No real data.
- Hardware decisions: record on the Zoom H8 (one clock, 4 XLR, 96 kHz);
  hydrophones and speaker on a jig; one Ping360 sweep per pose.
- Literature: ~400 IEEE records screened, 22 full texts read, gaps
  cross-checked on IEEE Xplore with a fresh session on 2026-09-09.
- Not done: any pool recording, Orin timing, hydrophone/pool/speaker facts
  below.
- 2026-09-09: Ryota's parallel plan (`../ryota`, branch `ryota_dev`) reviewed.
  Same core (differentiable pool model, echoes as sensors, Ping360 check,
  Orin table) plus learned element compensation for the DIY piezo discs and a
  JANUS comm track. Corrections adopted: deadline is 15 Jan 2027, hydrophones
  are CCRMA piezo discs. Merge proposal in note 06, not yet agreed.

## Open questions for Eli

1. Hydrophones are four CCRMA DIY piezo discs (per Ryota). What buffer or
   preamp sits between a disc and the Daisy codec today, and can the same
   front-end feed the Zoom H8? Ryota plans one Aquarian AS-1 reference.
2. Pool: length, width, depth profile, wall/floor material, gutter and lane
   rope layout, booking dates for early October and early November.
3. Speaker make/model; can it play WAV from SD (not Bluetooth) for the
   probe file?
4. Can the H8 be tested on the Orin as a USB interface (Linux class
   compliance unconfirmed)?

## Target venue

OCEANS 2027 Aberdeen, 21–24 June 2027, P&J Live. IEEE OES + MTS.
Abstract deadline **15 Jan 2027** (official important-dates page, verified by
Ryota): 2-page extended abstract, 500–1000 words, preliminary results
expected. Submission opens 1 Oct 2026. Notification March 2027. **Full paper
24 May 2027.** Only the abstract is peer-reviewed. (Tethys lists a tentative
7 Dec 2026; keep a December-ready draft but plan for January.) Student
Poster Competition: ~20 selected from 100+, travel and registration funded,
student must be lead author and presenter, full paper goes to IEEE Xplore.

Tracks that fit: 01 Underwater Acoustics; 02 Sonar Signal/Image Processing
and Communication; 10 Ocean Vehicles. Aberdeen local tracks that make
reviewers care: Offshore Infrastructure, Underwater Security and Protection
(confined-space inspection: harbours, ship hulls, nuclear ponds, tanks).

## What we have

| Resource | Detail |
|---|---|
| Compute | NVIDIA Jetson AGX Orin on the vehicle, CUDA/torch working (`../whisper_ivc`) |
| Hydrophones | 4× CCRMA DIY piezo-disc hydrophones: resonant, uncalibrated, mismatched (per Ryota's survey). On the sub: two Daisy Seed boards with no shared clock (front = ch 0 & 3, back = 1 & 2). For research: Zoom H8 (one clock, 96 kHz/24-bit) if a preamp path exists, else Ryota's Daisy sync fix. Reference hydrophone (Aquarian AS-1 + PA4) planned. |
| Recorder | Zoom H8: 6 XLR inputs, multitrack to SD, 12-in USB interface mode at 96 kHz (Linux support to be tested) |
| Sonar | Blue Robotics Ping360, 750 kHz mechanically scanning, open-source drivers |
| Source | cheap underwater speaker, plays SD-card audio, ≤ ~20 kHz, uncalibrated, own clock |
| Pool | ~25 m, concrete, highly reverberant. Published decay constant for a 25×12.5 m sport pool ≈ 0.125 s at 50–80 kHz (Misiurewicz 2022); 20–50 kHz unknown. |
| People | 3 at the pool typically; more helpers on request |
| Existing software | hydrophone → Orin streaming, Whisper transcription IVC demo |

## Plan at a glance

| When | What |
|---|---|
| Sep | Bench: speaker characterization, H8 on Orin, jig build, Ping360 driver test. Code: SVF wall filters, source FIR, fit protocol. |
| early Oct | Pool session 1: ~30 surveyed poses, thrusters off; 8 thruster-on poses; Ping360 sweep per pose. Fit model, classical baselines. |
| Nov | Learned localizer + ablation; Orin timing; pool session 2 (held-out poses, rotated jig). |
| early Dec | Numbers frozen for a December-ready draft (in case the tentative 7 Dec date is enforced). |
| 15 Jan 2027 | 2-page extended abstract (Track 01 or 02; Student Poster optional). |
| Jan–May | Self-echo follow-on; dataset packaging; full paper by 24 May 2027. |

Details in [notes/05_experiment_plan.md](notes/05_experiment_plan.md).

## Rules for this directory

- No secrets, cookies, PDFs of paywalled papers, or personal paths. Text
  extracts of papers are for our reading; do not push new PDFs.
- Work on branch `eli_dev`; keep the KiCad tree untouched.
- Everything decided lives in a file here, not in chat.
