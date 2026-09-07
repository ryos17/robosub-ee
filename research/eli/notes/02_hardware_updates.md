# Hardware updates after the critique (2026-09-06)

## Clock sync: solved by the Zoom H8

The critique's biggest risk (two Daisy boards, no shared sample clock,
silent sample drops) is removed by recording the four hydrophones on the
team's Zoom H8 handy recorder mounted on a jig, instead of the Daisy boards.

H8 facts (vendor specs):
- 6 XLR/TRS inputs (1–4 plus combo A/B), +24/+48 V phantom switchable per input.
- Multitrack WAV to SD at 44.1/48/96 kHz, 16/24-bit, up to 12 tracks. One
  ADC clock, so all four hydrophone tracks are sample-synchronous.
- 12-in/2-out USB audio interface mode at up to 96 kHz/24-bit (Mac/Win/iOS).
  Linux/ALSA support not confirmed by vendor; iOS support suggests USB Audio
  Class compliance. Test on the Orin in week 1. If it works, live capture on
  the Orin at 96 kHz is possible; if not, record to SD and process offline,
  which is fine for the paper (real-time claims then apply to the estimator,
  not the capture chain).

Hydrophone cabling depends on the variant we own (repo does not say):
- H2a-XLR: plug straight into H8 XLR, enable phantom.
- H2a 3.5 mm (plug-in power): needs a 2.5–5 V bias adapter or the existing
  preamp; H8 combo inputs do not supply plug-in power.
- AS-1: needs Aquarian PA4/PA6 preamp into TRS/XLR.
Action: check which variant, order adapters if needed.

Consequences for the paper:
- Full 4-channel sample-synchronous capture: 2D/3D localization claims are
  back on the table. The "cross-board clock residual" contribution is dropped
  from P1 (it was a problem we no longer have), replaced by a stronger
  localization result.
- Jig geometry is ours to choose. Do NOT keep the front/back parallel-pair
  layout from the sub; use a square or tetrahedral 4-element layout sized
  for the 2–18 kHz band (spacing ~4–8 cm avoids aliasing at 18 kHz,
  c/2f ≈ 4.1 cm; larger spacing improves bearing resolution but aliases at
  high frequency; a 2-scale layout is an option).
- "Hull-mounted on an AUV" framing weakens; reframe as "compact 4-element
  array deployable on a small vehicle", with the sub-mounted Daisy capture as
  a secondary real-time demo if time allows.
- Recording sessions get simpler: jig on a pole or on the pool floor, H8 on
  deck, speaker on a second pole.

Still measure once: H8 vs speaker DAC clock offset (ppm) from chirp
time-scaling, because the speaker plays from its own clock. The fitted model
absorbs it as a learnable time-scale parameter.

## Ping360: mature stack, use it inside P1, not as its own paper

Software (all open source):
- Protocol: per-ping control of angle (0–399 gradians, 0.9° steps), gain
  (3 levels), transmit duration (1–1000 µs), sample period, transmit
  frequency (500–1000 kHz), 200–1200 samples. Reply = up to 1200 × 8-bit
  intensity envelope. No phase/IQ, no device timestamps. `auto_transmit`
  does sector scans with start/stop angle. https://docs.bluerobotics.com/ping-protocol/pingmessage-ping360/
- ping-python (`pip install bluerobotics-ping`): `transmitAngle(angle)`,
  `control_auto_transmit(...)`, serial or UDP. https://github.com/bluerobotics/ping-python
- ROS2 driver publishing raw echo + LaserScan + image: https://github.com/CentraleNantesRobotics/ping360_sonar
- Hansen & Birk ICRA 2024 SLAM code (ROS2 C++, GTSAM, runs on RPi4 so Orin is
  fine): https://github.com/constructor-robotics/underwaterSLAM_2D_lowCost.
  Needs DVL + IMU odometry for motion compensation; static jig use bypasses it.
- Ping Viewer records raw `.bin` logs for offline work.
- "Zero Ping360 code in repo" is true but costs ~1 day, not a reason to drop.

Timing at pool ranges (15–30 m): ~20–40 ms per step, ~10–17 s per full 360°,
60° sector ≈ 1.5–3 s. 1200 samples at 30 m = 2.5 cm range bins. 25° vertical
beam smears surface/floor into the profile at short range.

Already published with Ping360 (so taken): Hansen & Birk motion-compensated
scan formation + SLAM (2023, 2024); Hasan 2025 pool artifact study + GPU U-Net
segmentation with public dataset (https://github.com/junayed/Sonar-Scan-Dataset-with-Ping-360);
AquaScan MobiCom 2025 pool activity recognition with physics-model denoising;
SwimSonar IPSN 2022 drowning detection; Huynh & Robinette 2025 HoloOcean
sim2real detection. Not found anywhere: learned processing of raw 1-D
profiles, Ping360 + hydrophone fusion, Ping360 as geometry source for an
acoustic channel model, Ping360 EMI/stepper noise on hydrophones.

Three options, ranked:

**A. Sonar-surveyed geometry as independent truth for P1's wall model. DO IT,
as a section of P1.** One 360° sweep per pose (~15 s) on the jig. Hough on
the polar scan gives the 4 walls + corners to ~2–5 cm. Use it (i) to
validate the wall offsets P1 fits from hydrophone echoes, (ii) as
initialization/prior for the fit, (iii) as a third ablation arm
("sonar-measured shoebox" vs "CIR-fitted" vs "generic"), and (iv) as extra
pose ground truth (jig range to two walls), tightening the ±10–15 cm GT
budget the critique worried about. Active sonar validating a passively
inferred channel model in a real pool is unpublished. Zero extra sessions.
Risk: tiled corners saturate at gain 2, so use gain 0 and short transmit.

**B. Passive-cued sector scan (D6 revisited). Poster fallback only.** Now
demonstrably programmable: slew to GCC-PHAT bearing in ~50 ms, or
`auto_transmit` a 60° sector in 1.5–3 s vs 10–17 s full sweep. Honest new
numbers would be time-to-first-contact and false-contact rate vs gate width,
plus measured Ping360 stepper/EMI leakage into the hydrophone band, which
nobody reports and which we need anyway for co-mounting. ~2 h of pool data
with beacon at 6–8 bearings. Do only if P1 finishes early.

**C. Learned raw-scanline front-end on the Orin (D8). Drop.** Hasan 2025 and
AquaScan already own GPU denoising/segmentation of Ping360 pool scans;
Hansen owns motion compensation. Remaining gap (1-D profiles with P1's
fitted CIR as ghost-echo prior) needs hundreds of annotated scans and moving
targets. Not feasible alongside P1 with 3 people.

Concrete week-1 add: `pip install bluerobotics-ping`, log one 360° sweep in
a bucket/tank, confirm raw profile access and step timing on the Orin.
