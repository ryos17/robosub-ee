# Experiment plan (consolidated 2026-09-09)

Single place for how the data gets collected and what gets measured. Pulled
together from the critique (01, section 3–4), the hardware notes (02) and the
method notes (03). Update this file, not those, when the plan changes.

## 1. Rig

**Receiver jig.** Rigid frame carrying the 4 hydrophones in a square or
tetrahedron, 4–8 cm spacing (c/2f ≈ 4.1 cm at 18 kHz keeps the top band
unambiguous; a two-scale layout is an option). Measure element positions to
2 mm and record them; the fit refines them. Do not copy the sub's
front/back parallel-pair layout. Cables to the Zoom H8 on deck.

**Recorder.** Zoom H8, inputs 1–4, phantom on if H2a-XLR, 96 kHz / 24-bit
multitrack WAV to SD. One ADC clock, so channels are sample-synchronous.
Optionally USB interface mode into the Orin at 96 kHz for live capture
(test Linux class compliance in week 1). Log water temperature each session.

**Source.** The SD-card underwater speaker on a second rigid pole with a
depth stop and bubble level, facing down, fixed orientation. Plays the probe
WAV (`src/probe.py`). Characterize once in a bucket: WAV vs codec playback,
0.5–24 kHz response, THD at three volumes, directivity at 0/45/90°, DAC ppm
from chirp time-scaling, any AGC/fade.

**Sonar.** Ping360 on the receiver jig. One 360° sweep per pose (~15 s),
gain 0, short transmit, 1200 samples. Driver: `pip install bluerobotics-ping`.

**Ground truth.** Taut marked line along a lane rope for along-pool
distance, tape measure across, pole depth stop for depth, jig on a taped
floor mark aligned to a lane line. Budget: ±10 cm horizontal, ±3 cm depth,
±3° heading. Report every position error against this budget. Sonar
range-to-two-walls is a cross-check, not the truth.

## 2. Probe signal

Per pose block (~90 s), from `src/probe.py`:
1. 1 s sync sweep 500 Hz → 20 kHz (alignment; clock-ratio estimate from the
   drift of arrivals across the block: 120 ppm × 90 s = 10.8 ms).
2. 20 × 50 ms linear chirps 2–18 kHz at 2 s spacing (RT60 ~3 s, so echoes
   die between chirps).
3. One 2 s exponential sweep 500 Hz → 20 kHz (full-band CIR).
4. One 1 s pink burst.
Optional: one chirp-spread-spectrum packet for the comms paragraph.

Design rules learned: never assume the strongest peak is the direct path
(UWLoRa+ pool CIRs show later paths exceeding the first); estimate the
speaker clock ratio from inter-chirp drift, not from a single chirp.

## 3. Sessions

**Bench, weeks 1–2.** Speaker characterization. H8 on Orin. Jig built and
element positions measured. Ping360 sweep in a bucket, raw profile access
and step timing confirmed. Recorder validated for 30 min at 4 × 96 kHz with
no gaps. Renderer sanity check vs pyroomacoustics (done).

**Pool session 1, early October, ~3 h, fit set.** Jig at one spot (deep
end floor or mid-depth on a pole), heading aligned to a lane line. Speaker
at 25–30 poses: ranges 1–12 m, two depths, both sides of the jig. Each pose:
one probe block, one Ping360 sweep, pose logged on the sheet. Then ~8 poses
with the sub's thrusters running at two PWM levels nearby (5 s bursts) for
the robustness subsection. Record a 1 m speaker-response clip first.

**Pool session 2, early November, ~3 h, evaluation set.** 25–30 new poses
including a held-out range band; jig moved and rotated 90° (transfer test);
one slow continuous speaker walk for tracking; same thruster-on subset.

**Dry rehearsal** of the pose-logging sheet and rig assembly before
session 1. Three people: one on the speaker pole, one on the tape/line and
sheet, one on the recorder and sonar.

## 4. Processing pipeline

1. Alignment: find sync sweep per channel, estimate clock ratio from drift,
   cut chirp windows, deconvolve by the probe → early CIR per channel (first
   10–30 ms).
2. Fit (`src/poolsim`): image-source model, order 1 laterally, order ~3
   vertically; learnable hydrophone positions, pool size and depth, sound
   speed, per-wall reflection filters, per-channel gain, source response;
   Gaussian priors on the survey; coarse-to-fine band schedule (<5, <10,
   full); several random inits, pick by held-out pose loss.
3. Sonar geometry: Hough on the polar sweep → wall lines; compare with fitted
   wall offsets; use as init and as a third ablation arm.
4. Localization: coarse-to-fine matched-field grid (envelope score at
   0.25 m, coherent at 5 cm and 1 cm), then continuous refine.
5. Learned localizer: one small CNN/ConvLSTM on cross-spectra, trained on
   (a) free-field, (b) generic shoebox, (c) fitted model; evaluated on the
   same real held-out poses, with and without N-pose fine-tuning.
6. Orin: same code with `.to("cuda")`; ms/frame and W at MAXN and 15 W via
   jtop.

## 5. Baselines and metrics

Baselines: GCC-PHAT least-squares bearing; SRP-PHAT on the same grid;
incoherent MUSIC; single-hydrophone surface-echo range (Li 2016, 7485369);
sparse first-arrival bearing (Kasetkasem 2017, 8096292); unfitted generic
shoebox model.

Metrics: azimuth MAE and 90th percentile; range error in 0.5 m bins by
range band (0–5, 5–10, >10 m); 3D error against the stated ground-truth
budget; miss rate; per-estimator ms/frame and W; fitted-vs-sonar wall
offset agreement in cm; sim-to-real ablation table (three training sets ×
with/without fine-tune).

## 6. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Walls not planar (gutters, steps) | early window only, learnable wall offsets, sonar-measured geometry, report which walls are used |
| Speaker distortion / codec | SD playback only, characterize first, treat response as learnable |
| H8 not usable on Orin | record to SD, process offline; real-time claims apply to the estimator only |
| Elevation unobservable in 2 m depth | report depth as a limitation; headline is azimuth + range |
| Fit unstable on real data | envelope-first schedule, more inits, freeze walls after sonar init |
| Too few poses for the ablation | fold session 2 into training/eval split, report intervals |

## 7. Week 1–4 checklist

**Week 1.** Identify hydrophone variant; order adapters/preamp if needed.
Speaker bucket test. Measure and record jig element positions. Get pool
dimensions, materials and bookings. H8 on Orin test. `pip install
bluerobotics-ping`, bucket sweep.

**Week 2.** 30 min drop-free recording test. Probe WAV rendered and played
through the speaker in the bucket, recorded, aligned. Code: SVF wall
filters, source FIR, multi-init fit protocol.

**Week 3.** Ground-truth rig built; dry rehearsal; classical baselines on
synthetic data; Ping360 Hough wall extraction on the bucket/tank sweep.

**Week 4.** Full chain soak test in a tank or shallow pool corner; Orin
timing on synthetic data; abstract skeleton drafted with the citation list
from notes 03–04.
