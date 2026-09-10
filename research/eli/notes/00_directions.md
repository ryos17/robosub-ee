# Research directions: underwater acoustics on the sub's Orin, for OCEANS 2027 Aberdeen

> **Superseded (2026-09-09).** This is the original survey. The directions below were
> critiqued in `01_critique.md`, narrowed there, and restated as plain-language gaps in
> `04_research_gaps.md`. Start at `../README.md`. Kept for the literature map and the
> dropped ideas.

Literature survey, 2026-09-06. Source: 46 IEEE Xplore searches (2016+), ~390
unique records, abstracts reviewed in four theme clusters (localization/DOA,
comms, edge/sonar, DAFx-style DSP/ML). Paper IDs below are IEEE Xplore
"arnumber"s: `https://ieeexplore.ieee.org/document/<arnumber>`.

See [`../README.md`](../README.md) for venue, deadlines, and hardware.

---

## TL;DR

**The opening is real and specific:** across ~390 papers, nobody combines
(a) a small hydrophone array, (b) ML/DSP inference on an embedded GPU on the
vehicle, and (c) a reverberant bounded environment (pool/tank) with ground
truth. Jetson-underwater papers are 100% camera vision. Pool/tank acoustics
papers are ~5 and none do DOA/ML. The DAFx toolbox (differentiable DSP,
neural codecs, self-supervised audio, Noise2Noise, source separation) has
essentially not been ported to hydrophone arrays.

**Recommended primary direction (P1):** *Differentiable pool channel + learned
pinger/beacon localization on the Orin.* Fit a DDSP-style image-source /
ray model of the pool to 4-channel recordings, use it as (i) a physically
grounded simulator for training, (ii) a dereverberation prior, and (iii) a
likelihood for localization; deploy the resulting DOA/position estimator on
the Orin in real time and report accuracy + latency + power vs GCC-PHAT /
MUSIC baselines. Multipath becomes the feature, not the bug.

**Recommended secondary/stretch (P2):** *Audio-band chirp-spread-spectrum
(LoRa-style) inter-vehicle link from a consumer speaker, GPU-demodulated on the
Orin, whose packets double as localization beacons (comm + ranging).* Shares
the dataset and channel model with P1.

**Insurance (P0):** *Open 4-channel pool dataset with labelled source poses.*
Falls out of P1 data collection. Dataset papers are welcome at OCEANS and no
AUV-array-with-ground-truth dataset exists.

---

## 1. What the literature says (per theme)

### 1.1 Pinger / TDOA / DOA localization
- Mature classical stack: cross-correlation → TDOA → hyperbolic LS; phase-difference
  USBL with ambiguity resolution (10056767 Wang 2023); hybrid TDOA+bearing closed
  form (9296227 Zhang 2021); sensor-position-uncertainty-robust TDOA (11617090
  Zhou 2026); multipath outlier rejection in USBL cross-corr (8962723 Takada 2019).
- DL-DOA: CNN/ConvLSTM on covariance/wavelet/raw pulse. Beam-network DNN
  (10124137 Nie 2023, most cited), saliency ConvLSTM on 3-element array
  (11104616 Song 2025, closest analogue to us), PulseDoaNet (11621547 Chen 2026),
  cross-domain MI loss (10918686 Li 2025). Trained on Bellhop sims; none report
  edge latency; none in reverberant tanks.
- Multipath-as-feature: single-hydrophone range/depth from direct/surface/bottom
  delay structure (7485369 Li 2016; 11085563 Li 2025; 8604304 Wang 2018),
  2-hydrophone 3D via frequency diversity (9662075 Cao 2022). All assume two
  reflectors (surface+bottom). A pool has six. Nobody has exploited that.
- Pool/tank work is thin: tank multipath model for comms (9708257 More 2021),
  bounded-reservoir source localization model (9604320 Narang 2021), pool
  EKF/UKF INS+acoustic nav (8604773 Wang 2018), close hydrophone pairs in
  multipath (8084699 Sim 2017). Query "multipath pool" returned zero hits.

### 1.2 Communications
- DL receivers replace chan-est+EQ+detect with one net; recurring pain is
  sim→real mismatch because there is no accurate channel model (9564511 Zhao
  2021; multi-task rx). Online-trained rx with no offline training (8672093
  Zhang 2019, 55 cites) and channel-replay training (10244334 He 2023) are
  the practical recipes.
- Chirp spread spectrum / "underwater LoRa" is hot: Steinmetz 9905674 (2022,
  sea trials vs JANUS), Petroni 10283587 / 11644049 (sync + Doppler),
  Jia 9816086. **DeWater (11571741 Wu, INFOCOM 2026)**: learning-enhanced
  demod explicitly targeting commodity hardware. Closest competitor to a
  speaker-based link.
- Low-cost modems: Padua SDM line (10070400 Campagnaro 2023), MODA swarm modem
  (9775308), phone-as-receiver (8084730 Sherlock, OCEANS Aberdeen 2017). Nobody
  transmits from a consumer speaker.
- Voice: Rong 11104740 (OCEANS 2025) does STT→text-over-modem→TTS. Our
  existing Whisper-over-raw-audio IVC is the natural comparison.
- Neural *audio* codecs underwater: zero hits (only image compression for
  acoustic uplink: 10445326, 9934922).
- GPU/FPGA real-time receiver on vehicle: zero hits.

### 1.3 Edge GPU / sonar
- Jetson-underwater = camera vision only (YOLO + enhancement on Nano/NX/Orin).
  One sonar-on-edge paper (10809254 Zhang 2024, FLS YOLO). No hydrophone DSP
  on Jetson anywhere. Edge-benchmark writing template: 10459973 Mohammadi 2023.
- GPU acoustic arrays exist only topside: 160-element whale array (10704599
  Mohebbi-Kalkhoran 2024), SAS beamforming (9389388 Gerg 2020).
- Ping360: Hansen/Birk own it — synthetic scan formation with motion comp
  (10239384, 2023), open-source BlueROV2+Ping360+DVL mapping (10609976 ICRA
  2024). Sim2real Ping360 detection via HoloOcean (11244929 Huynh 2025).
  No ML on raw polar intensity profiles, no learned distortion correction, no
  hydrophone+Ping360 fusion.
- Active-echo classification with little data: BiSAD anomaly detection
  (9976054 Kim 2022), ABCL attention (10549883 Hwang 2024). Navy datasets.

### 1.4 DAFx-style techniques underwater
- Differentiable DSP: all music/speech. Differentiable Artificial Reverberation
  (9837411 Lee 2022, TASLP), differentiable biquads (9413996 Nercessian 2021),
  DDSP-vocoder speech enhancement on edge (11433840 Guimarães 2025),
  differentiable loudspeaker lumped models (11143572 Massi 2025). Nearest
  underwater relative: **data-aided differentiable ray model (10224658 Li &
  Chitre 2023, JOE, 33 cites)** — but fit on ocean data, not a closed domain.
- Self-supervised: all single-channel ship-noise classification (10414073 Feng
  2024; 10288071 Xu 2024). No multichannel/spatial pretext, no SSL for DOA.
- Denoising: WaveN2N = Noise2Noise across array channels (9389338 Koh 2020,
  Global OCEANS). No diffusion enhancement, no dereverb of tank multipath, no
  self-noise cancellation conditioned on thruster telemetry.
- Source separation: two papers, both single-channel PAM (10289193 Mancusi 2023).
- Datasets: DeepShip, ShipsEar, WATERMARK (channel replay). No AUV-mounted
  array + known source + ground-truth pose dataset.

---

## 2. Candidate directions, ranked

Scoring: novelty (N), fit to OCEANS tracks (F), feasibility in 9 months with
pool + 3 people (E), risk (R). 1–5, higher better (R: higher = lower risk).

| # | Direction | N | F | E | R | Sum |
|---|---|---|---|---|---|---|
| P1 | Differentiable pool channel + learned beacon localization on Orin | 5 | 5 | 4 | 3 | 17 |
| P2 | Speaker-TX chirp-SS link, GPU demod, packets double as TDOA beacons | 4 | 5 | 3 | 3 | 15 |
| P0 | Open 4-ch pool dataset + sim2real gap study for DL-DOA | 3 | 4 | 5 | 5 | 17 |
| D4 | Thruster-conditioned multichannel Noise2Noise self-noise denoiser | 4 | 4 | 4 | 3 | 15 |
| D5 | Self-supervised multichannel pretraining, spatial pretext → DOA | 4 | 3 | 3 | 2 | 12 |
| D6 | Hydrophone-bearing steers Ping360 scan sector (active sensing fusion) | 4 | 4 | 3 | 2 | 13 |
| D7 | Neural speech codec over acoustic link (Whisper-IVC upgrade) | 4 | 3 | 3 | 2 | 12 |
| D8 | Ping360 learned scan-distortion correction / raw-profile ML | 3 | 4 | 3 | 3 | 13 |

P0 scores equal to P1 but is a byproduct, not a headline. Do P1, publish P0
data alongside (or as student-poster entry).

### P1. Differentiable pool channel + learned beacon localization on the Orin
**Claim.** First physically-structured, end-to-end-differentiable multipath
model of a bounded underwater environment fit from a 4-hydrophone AUV array,
used to train and run a real-time GPU localizer that *exploits* wall echoes.

**What it is.**
1. Model: image-source method (6 walls, order 2–3) with learnable per-wall
   reflection filters (differentiable biquads, 9413996), learnable hydrophone
   positions/gains/delays (self-calibration; cf. 11617090), learnable speaker
   response. Rendered RIR convolved with known probe signal; loss = multi-res
   STFT (DAFx staple) vs recorded 4-ch audio. This is Differentiable Artificial
   Reverberation (9837411) with a physical room prior, meeting Chitre's
   data-aided ray model (10224658).
2. Uses of the fitted model:
   - Simulator for unlimited labelled training data across poses.
   - Dereverb / matched filter: whiten direct path against known echoes.
   - Likelihood for MAP localization: 3D position from 4 channels + echoes,
     which plain TDOA can't do (only 3 TDOAs).
3. Deployed estimator: small CNN/ConvLSTM on cross-spectra (11104616 template)
   or direct MAP grid search on GPU; benchmark vs GCC-PHAT, SRP-PHAT, MUSIC.
   Report on Orin: latency, power (jtop), accuracy vs pose, vs SNR.

**Why feasible.** Pool geometry known; speaker plays chirps/pings from SD
card at gridded poses (tape measure / weighted line / AprilTag on a pole);
sub static on bottom or held. ~2–3 pool sessions for fit + eval set. All
software on existing whisper_ivc streaming stack.

**Known hazards.**
- Two Daisy boards ≠ shared clock: cross-board TDOA drifts. Fix options:
  common sync pulse into one channel of each board, or estimate offset as a
  learnable parameter (nice: becomes a contribution). Must solve early.
- Speaker is band-limited, nonlinear, maybe codec-compressed. Characterize
  first (11245687 Marin 2025 does this for cheap transducers). Use it as a
  learnable component, not ground truth.
- Reviewers: "pool ≠ ocean." Frame as harbour/tank/structure-inspection
  (Aberdeen local track "Offshore Infrastructure", "Underwater Security")
  and as a sim2real methodology. Cite 9604320 Narang 2021 bounded reservoirs.

**Builds on.** 10224658, 9837411, 9413996, 11104616, 8962723, 7485369,
9708257, 11617090, 10459973 (how to write the edge eval).

### P2. Audio-band chirp-spread-spectrum IVC from a consumer speaker, GPU demod, comm+ranging
**Claim.** First LoRa-style CSS physical layer through an unmodified consumer
underwater speaker, demodulated in real time on the vehicle GPU, whose
preambles simultaneously give bearing/range on the 4-hydrophone array
(integrated sensing and comm, ISAC-lite).

**What it is.** CSS frames in 2–18 kHz, SF sweep, dechirp via batched CUDA
FFTs; sync/Doppler per Petroni 11644049; 4-channel diversity combining;
compare against learned demod (DeWater 11571741, online-trained 8672093);
preamble TDOA → bearing of transmitting teammate. Report bps, BER vs range
and vs thruster noise in pool; WER for Whisper-over-raw-audio vs text-over-CSS
(the Rong 11104740 comparison).

**Why feasible.** TX is SD-card playback of pre-rendered waveforms: no
hardware. Everything else is offline-then-realtime software. Uses P1's
channel model for training the learned demod.

**Hazards.** One-way only (speaker can't receive); frame it as broadcast
beacon/telemetry. Speaker Bluetooth/codec may mangle chirps: verify with SD
playback, not Bluetooth.

**Builds on.** 9905674, 11644049, 10283587, 11571741, 8672093, 10244334,
8084730, 9706010 (DCSS comm+loc), 11104740.

### P0. Open dataset: 4-channel 96 kHz AUV-array pool recordings with labelled source pose
**Claim.** First public multichannel hydrophone-array dataset from a small AUV
with ground-truth source positions, plus a sim2real study showing how far
Bellhop/image-source-trained DL-DOA degrades in a real reverberant tank and
what closes the gap.

Contents: chirps, pings at RoboSub pinger-like bands (limited by speaker;
note ≤20 kHz), CSS frames, speech, DeepShip noise replayed via speaker,
thruster-on/off, ~50–100 poses. Release with loader + baselines. Builds on
9564511, 10244334, 10608448 (UACC-GAN as sim baseline), 11226529 (augmentation).

### D4. Thruster-telemetry-conditioned multichannel Noise2Noise self-noise denoiser
WaveN2N (9389338) trick: 4 channels = free noisy pairs. Add conditioning on
thruster PWM/IMU (nobody uses vehicle telemetry as side-info). DDSP-vocoder
style light model for Orin (11433840). Improves everything downstream (P1, P2,
Whisper IVC). Good as a section of P1 or a second short paper.

### D5. Self-supervised multichannel pretraining (spatial pretext) → DOA
Masked/contrastive SSL (10414073, 10288071) but on raw 4-ch waveforms with
cross-channel prediction as pretext; fine-tune with few labelled poses.
Higher risk; data-hungry; better as a follow-up journal extension.

### D6. Passive-bearing-steered Ping360
Hydrophone DOA gates the slow mechanical scan sector; active-sensing policy.
Fun, unique to our hardware, but needs two subsystems working. Builds on
11104638 Cecchi 2025, 10609976 Hansen 2024. Candidate for a second paper.

### D7. Neural speech codec over the acoustic link
EnCodec/DAC-class ~1–3 kbps speech tokens over CSS from P2; MOS/WER vs bitrate
vs range. Zero prior work underwater. Depends on P2 existing.

### D8. Ping360 learned front-end / distortion correction
Learned denoise + motion-compensation of polar intensity profiles, GPU
real-time, vs Hansen's geometric synthetic-scan. Sonar track fit, but Hansen's
group is strong and moving; harder to out-novel.

---

## 3. Suggested plan to 7 Dec 2026 abstract / early-2027 paper

| When | What |
|---|---|
| Sep | Read must-reads (§4). Characterize speaker (freq response, SD vs BT, THD). Solve Daisy clock sync. Write image-source pool sim + differentiable fit code (torch, run on laptop first). |
| Oct | Pool session 1: chirps at ~30 poses, thrusters off/on. Fit model. Classical baselines (GCC-PHAT/SRP/MUSIC). |
| Nov | Learned localizer trained on fitted sim, fine-tuned on real. Port to Orin (TensorRT), measure latency/power. Pool session 2: eval set. Draft abstract. |
| 7 Dec | Abstract submitted (Track 01 or 02; also Student Poster Competition). |
| Dec–Feb | P2 CSS link + comm/ranging; pool session 3. Dataset packaging (P0). |
| Mar–Apr | Full paper. |

---

## 4. Must-read list (fetch text extracts in `../papers/` as they arrive)

Localization / channel
1. 10224658 Li & Chitre 2023, Data-aided underwater acoustic ray propagation modeling (JOE)
2. 9837411 Lee 2022, Differentiable artificial reverberation (TASLP)
3. 11104616 Song 2025, Robust UWA pulse DOA via saliency-aware ConvLSTM
4. 8962723 Takada 2019, Multipath error reduction for UWA DOA
5. 7485369 Li 2016, Single-hydrophone multipath range localization
6. 9708257 More 2021, Multipath modeling under tank environment
7. 11617090 Zhou 2026, TDOA with sensor-position uncertainty

Comms
8. 11571741 Wu 2026, DeWater (INFOCOM) learned demod on commodity hardware
9. 9905674 Steinmetz 2022, Taking LoRa for a dive
10. 11644049 Petroni 2026, LoRa sync/detection under Doppler (TCOM)
11. 8672093 Zhang 2019, Online-trained DL receiver
12. 11104740 Rong 2025, STT→text→TTS diver comm (OCEANS Brest)

Edge / sonar
13. 10609976 Hansen 2024, Open-source Ping360 mapping (ICRA)
14. 10459973 Mohammadi 2023, Edge-board benchmark methodology
15. 10704599 Mohebbi-Kalkhoran 2024, Real-time array detection/bearing pipeline

DAFx-port
16. 9389338 Koh 2020, WaveN2N multichannel Noise2Noise
17. 10414073 Feng 2024, Masked-token SSL for UATR (TASLP)
18. 11433840 Guimarães 2025, DDSP-vocoder speech enhancement on edge
19. 10608448 Liu 2024, UACC-GAN channel simulator
20. 10716649 Müller 2024, Survey of underwater acoustic ML

Also: 8096292 Kasetkasem 2017 (sparse pinger DOA, already in `../papers/`).

---

## 5. Open questions for Eli
- Pinger band: RoboSub pingers are 25–40 kHz; speaker tops out ~20 kHz. P1 is
  validated with audio-band beacons. Do we also want a real 25–40 kHz pinger
  for a second eval set? (Would strengthen the RoboSub tie-in.)
- Can a second sub / a second speaker be at the pool for the IVC experiments?
- Is the Daisy master/slave link hackable for a shared sync pulse?
- Pool dimensions, depth profile, wall material (needed for image-source prior).
