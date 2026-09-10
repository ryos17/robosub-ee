# Reconciliation with Ryota's plan (branch `ryota_dev`, 2026-09-09)

Ryota's `research/ryota/` (survey 2026-09-06, plan 2026-09-07, 91 DAFx26 paper
cards, three landscape surveys, adversarial judge verdicts) reached the same
territory from the DAFx side. This note lists what he found that corrects my
notes, where the two plans overlap, and a proposed merge. Nothing here has
been agreed with Ryota yet.

## 0. The two ideas in plain words

**Eli.** Cheap speaker + 4 hydrophones + Orin → a fitted 3D echo model of the
pool and the 3D position of the speaker. The walls act as mirrors that place
virtual copies of the tiny array metres away, which is what gives distance,
not just direction. Ground truth is a tape-measure survey; the Ping360 only
confirms where the walls are.

**Ryota.** The 4 cheap piezo-disc hydrophones are bad (they ring and don't
match each other), so teach a small network to un-ring and rebalance each one,
then find the RoboSub pinger's direction from the corrected signals on the
Orin. To train it, build a pool echo model fitted to real recordings and
measure the discs against a proper reference hydrophone.

**Same:** fit a pool echo model from recordings; use echoes instead of
fighting them; check walls with the Ping360; report Orin speed and power.

**Different:** he starts from "fix the bad microphones", I start from "use the
walls to get distance". His target is the real 25–40 kHz pinger; mine used
the speaker's 2–18 kHz. Each needs the other half: my model needs his
per-element compensation to work on ringing discs; his pinger localizer
needs my fitted model and wall echoes to get range.

## 1. Facts from Ryota's work that change my notes

| Item | My notes said | Ryota verified | Fixed where |
|---|---|---|---|
| Abstract deadline | 7 Dec 2026 (Tethys, tentative) | **15 Jan 2027**, 2-page extended abstract with preliminary results, on the official important-dates page; submission opens 1 Oct; notification March; **full paper 24 May 2027**. Only the abstract is peer-reviewed. | README, 05 plan |
| Hydrophones | Aquarian H2a / AS-1 class (my grep matched binary junk) | **Four CCRMA DIY piezo-disc hydrophones**: resonant, uncalibrated, mismatched, ringing smears onsets. Ryota plans to buy one Aquarian AS-1 + PA4 as a reference. | README, 02, 05 |
| Pinger task | audio-band speaker chirps 2–18 kHz | RoboSub pingers are 25–40 kHz CW bursts; his plan buys a 20–50 kHz projector + 192 kHz interface (~$1–2k) for pinger-band CIRs; cheap speaker is the sub-20 kHz fallback | 05 plan |
| Misiurewicz 2022 (Sensors, 25×12.5 m sport pool) | "treats reflections as a problem to reject" | Uses reflections via multi-hypothesis TDOA (H0 direct, H1 one mirror-image reflection per sensor), 50–80 kHz LFM, four hydrophones **mounted mid-wall**, 130k hypotheses/s on a Cortex-A7, no fitted model, no ground-truth accuracy reported. Reports measured pool decay constant ≈ 0.125 s and reflections often stronger than direct. | 04 gaps |
| Pool acoustics | RT60 "of order seconds" (guess) | decay constant ≈ 0.125 s at 50–80 kHz (Misiurewicz); T60 at 20–50 kHz not published anywhere; ISM enumeration explodes (millions of images), so early ISM + FDN/stochastic late tail | 05 plan |
| Ping360 + RGU | none | Host university RGU published the only Ping360 + deep-learning papers (Hasan 2025, Jetson Orin). Real local hook. | 04 gaps |
| Curtin speech-over-modem | cited as Rong 2025 | WER 2.4 %, 4.3 s latency, RPi5 LoRA Whisper → text over 24–32 kHz modem. Reviewers will hold whisper_ivc against it. | — |

His judge's feasibility objection to "echoes as sensors" (his D2 ≈ my G1):
on ringing discs at 0.5–1 m depth the surface echo is unresolvable from the
direct path, and reflections-as-virtual-sensors is old in air (Ribeiro 2010)
and exists underwater (Misiurewicz 2022; Wang 2018, 8604304). His fix: side
walls and floor only (delays > ~1 ms), CLEAN-style arrival picker, learned
refiner only as an ablated correction, surveyed ground truth, GPU
multi-hypothesis solver as the Orin-load-bearing section. That matches my
narrowed P1 (early window, side walls, envelope-first) and adds the ringing
constraint I did not have.

## 2. Overlap map

| Ryota | Eli | Relationship |
|---|---|---|
| D7 PoolVerb: differentiable ISM + FDN fitted to measured pool CIRs, T30/EDC numbers, dataset, sim-to-real study | P1 model (G1/G2/G3): differentiable ISM fit, self-calibration, fitted-vs-generic ablation | Same core. Mine has working synthetic code (`src/poolsim`); his adds the FDN tail, 20–50 kHz band, decay metrics, dataset release. |
| D2 Echoes-as-Sensors: arrival set + GPU image-source association | G1 walls as virtual array + matched-field localizer | Same idea; his judge descoped it to a section; my code does it as MFP on the fitted model rather than arrival picking + association. Both should be run; MFP avoids echo labeling. |
| D1 SelfCal-NGCC: learned per-element EQ + conv front-end + learned frequency weighting for pinger TDOA | (none; my model has per-channel gain only) | Complementary and needed: disc ringing is a per-channel response my model must absorb. Becomes a learnable per-channel FIR/IIR in the fit. |
| D8 Know-Thy-Disc: disc characterization vs reference, compensator, blind self-cal | G2 joint self-calibration | Complementary; G2 calibrates geometry, D8 calibrates response; one fit can do both. |
| D3 sonar-vs-tape-vs-none geometry experiment | G4 Ping360 wall cross-check | Same experiment. |
| Orin table (TensorRT, p50/p99, watts, Whisper co-load) | G5 | Same; his is more specific. |
| D5 Neural JANUS-Lite, D6 ASR voice link (Abstract B) | dropped (comms taken) | His comm track is JANUS-specific and unjudged; I found the general chirp/speaker link taken (UWLoRa+, DeWater). JANUS-over-DIY is still open. His call. |
| — | G6 self-localization from own echoes | Not in his plan. Beaconless confined-space navigation; keep as follow-on. |

## 3. Proposed merge

One pool campaign, one capture format, one model, two abstracts at most.

- **Abstract A (headline, joint).** Ryota's title stands: learned element
  compensation and multipath-aware localization for low-cost resonant
  hydrophone arrays in reverberant pools, on an in-hull GPU. Inside it, my
  fitted-model matched-field localizer is the "multipath-aware" method
  alongside his GPU hypothesis solver; my self-calibration is the geometry
  half of his D8; my ablation (free-field / generic box / fitted) is his D7
  transfer study. Ground truth = survey; Ping360 = geometry cross-check (D3).
- **Abstract B.** His JANUS receiver, if the team has bandwidth. Otherwise
  fold my G6 (self-echo localization) into the full paper as a section.
- **Code.** My `research/eli/src/poolsim` (renderer, fit, MFP localizer,
  tests) maps onto his planned `uwa/poolverb` + `uwa/tdoa`. Propose moving it
  into his package layout once he confirms, rather than two copies.
- **Hardware.** Keep his projector purchase for 25–40 kHz; keep my H8
  route for a one-clock capture that needs no firmware work, but the discs
  need a buffer/preamp in front of the H8 (piezo discs are high impedance;
  phantom power does not help). His Daisy sync fix (shared MCLK or GPIO
  sync counter) remains the on-vehicle path. Decide in the week-1 sync spike.
- **Timeline.** His 18-week plan to 15 Jan replaces my 7-Dec plan; my
  weeks 1–4 checklist (05) merges into his weeks 1–4.

## 4. What I should read from his corpus

- `notes/dafx26_paper_cards.md`: paper_07 (DiffRIR), paper_10 (differentiable
  RIR/FDN fitting), paper_11 (pressure-release sign flip), paper_12 (fast
  lattice ISM), paper_48/15/30 (differentiable EQ), paper_41/03 (ERA, SSMs),
  challenge_87/89 (arrival estimation).
- `notes/landscape_array_channel_sonar_edge.md` §on pool acoustics and
  Daisy sync.
- `notes/judge_verdicts_D1-D4.json` for the exact prior art strings.
