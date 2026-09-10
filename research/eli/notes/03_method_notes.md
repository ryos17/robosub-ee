# Method notes mined from the key papers (2026-09-08)

> Ranked code changes at the end are the to-do list for `../src/poolsim`. Items done so far:
> concrete reflection init, axial z-boosting, soft time gate, geometry priors, envelope coarse score.

What to copy, what to differ from, and measured tank/pool facts, for the P1 channel model and localizer in `../src/poolsim`. Produced by a full-text read of the eight papers listed; ranked code changes at the end.

**Report: 8 papers vs src/poolsim (P1 fit + MFP localizer)**

**DiffRIR (Wang 2024, CVPR; IEEE Xplore version exists)**
- Copy: image-source per-path response = directivity D(d) x prod R_s x 1/rho x air-abs^t, magnitude-only per surface on F octave-band knots, `sigmoid` -> energy coeff -> `sqrt` -> amplitude, linear interp in freq; **minimum-phase IFFT** of magnitude (not our first-order lowpass phase). Source = learned **time-domain FIR** convolved with sum. Directivity = per-octave spherical heatmaps, 128 Fibonacci points, spherical-Gaussian interp. **Spatially-uniform residual r** blended by 16-knot temporal spline gamma (late field). **Axial boosting**: parallel-wall pairs traced to order ~50 while general order N=5. Source pos from first-peak TOA before fitting. Loss = multi-scale log-spectral L1 (DDSP/DiffImpact). 12 RIRs train; 6 points beat all baselines trained on 100.
- Numbers: rooms 7.1x7.9x2.7 RT60 0.69 s; hallway 1.5x18.1x2.8 RT60 1.41 s (closest analog to long pool). Ablation: residual matters most in hallway (Mag 9.13 -> 15.49 without). Learned wall amplitude ~0.6-0.9 mid-band, carpet lower, foam ~0 above 2 kHz.
- Differ: air/room, single omni mic per RIR, no clock/array self-cal.

**Chitre/Li data-aided ray model (10224658)**
- Copy: ISM with **learnable error terms** e_theta, e_psi, e_d on pre-computed nominal image directions/distances, L2 penalties; **higher-order images get smaller penalty** (angular errors amplify with bounces). L1 on ray amplitudes for sparsity. Angle-dependent reflection = tiny NN (1-in, 2-out mag+phase, 1 hidden layer) shared across rays; energy-conservation penalty. Loss = **MAE not RMS** (outlier robust); **weighted MAE** by per-point std over 40 s repeats. **Two-stage**: fit model on surveyed positions, freeze, then fit position errors with L2 prior. 20 random inits, keep best validation, early stopping, Adam. c known (measured).
- Numbers: near-field RMS err 1.9 dB clean -> 3.4/4.7/5.4/6.2 dB at 0.1/0.2/0.3/0.4 m x sqrt3 position error. **Reflection phase far more sensitive to position error than magnitude**; more data partially mitigates.
- Tank facts: 2.5x1.2x0.8 m fiberglass, 10 kHz CW, c=1505 m/s (CTD), 500 points via fishing line + reel/slider; dimension mismatch 1-2 cm, source 2 cm, receivers 2-4 cm; recovered position errors mostly <4 cm; MATE 0.376 dB, Spearman 0.961. Quote: tank "strongly reverberant and surprisingly complicated to model... multiple reflections lead to strong sensitivity to minor geometrical irregularities."

**Lee DAR (9837411)**
- Copy: **SVF parameterization** of biquads: cutoff f = tan(pi*sigmoid(x)/2), resonance R = softplus scaled (centered 1), mixing gains m_LP/BP/HP init 1,2,1 (near-flat start, no vanishing grads). Init cutoffs log-spaced 40 Hz-12 kHz. **Frequency-sampling** IIR -> FIR on N-point grid (we already do freq-domain, so SVF is a drop-in). Loss: multi-scale STFT L1 on magnitudes, FFT 256-4096, hop 25%, Hann, **log-frequency axis**. Regularizer on pole radius (tail/head IR energy ratio, n0=N/8) to limit time aliasing. Adam lr 1e-4, grad clip +/-10, decay 10^-0.2 per 50k after 250k.
- Numbers: SVF beats raw biquad coefficients "by large margin"; plain FIR worse than SVF; shelving needs R > sqrt2/2 to avoid >1 spike. 200k shoebox-simulated RIRs (pyroomacoustics) for training, real only for val/test; sim-to-real gap named as main limitation (supports our ablation framing).

**Li 2016 single-hydrophone D-SR (7485369)**
- Method: Hilbert envelope of autocorrelation, pick D-SR delay, match to model delay-number curve vs range; source depth assumed known. 700-1500 Hz LFM 1.5 s / 2 s; error 0.2 km at 8.6 km; valid <13 km. Range resolution parabolic: better at short range, deeper source, wider BW, higher fs; single shot unreliable, needs track. Our baseline recipe: surface echo only, 96 kHz + 1-18 kHz chirp gives ~cm-scale delay resolution at 1-12 m.

**More 2021 tank SWARA (9708257)**
- Tank 3.85x2.4x2.0 m **concrete**, filled 1.8 m, 26 C, c assumed 1500. 25-35 kHz chirp 100 ms, fs 240 kHz, 0.3 m deep, 2 m apart. Impedances: concrete 4.08e6, water 1.48e6, air 353 -> normal-incidence |R| ~0.47 amplitude (prior for our floor/wall r0; our 0.9 init is high). Delays by Klauder-wavelet peaks of cross-corr; 48 paths; RMSE 0.11 V waveform. Weak; cite as "ad-hoc tank model, no fitting."

**Zhou 2026 TDOA w/ sensor position uncertainty (11617090)**
- Sensor position error makes pseudo-linear TDOA an EIV problem; WTLS + Tikhonov (lambda=1e-8) + geometric-consistency correction, **safeguard: accept corrected solution only if nonlinear residual decreases**. 2D sim, 8 sensors r=500 m; N=4 sensors degrades sharply. Use: motivates joint hydrophone self-cal; copy safeguard in localizer refine.

**N2N AUV self-noise (10103424)**
- UNet on log-power spectrogram, self-supervised: noise-only recordings + random sinusoids as input/target pair, predicts noise spectrum, then improved spectral subtraction. Gains: 3.9 dB avg / 8.4 dB max (16-el), 1.73 dB (4-el, 32 kHz, Discovery-100). Needs only thruster-on no-chirp clips; alt to PWM-indexed Wiener.

**UWLoRa+ pool CSS (10623112)**
- Indoor pool 4.6x3.0x1.5 m, tx/rx 0.5 m deep, >=3 m, 22-28 kHz. Measured CIR: many paths over 0-40 ms, **later paths often exceed first-path amplitude** (pitfall for first-arrival detectors). Quadratic chirp -> energy scattering, per-path demod windows, multipath combining; 2x SER, 3x BER gain. Use: probe design + intra-board preamble bearing.

**Ranked changes to src/poolsim**
1. `WallFilter` -> K serial SVF sections with Lee activations/init + min-phase from magnitude (DiffRIR); init |R|~0.5 (concrete) not 0.9; keep surface -1.
2. Chitre error terms: per-wall/depth offsets already learnable; add L2 priors to survey (walls cm, hyd cm, src +-10 cm), penalty weight decreasing with image order; two-stage fit (model, then per-pose source-position errors, frozen model).
3. `losses.py`: L1/MAE spectral terms, log-frequency binning, sizes 256-4096 hop 25%; per-pose weights from repeat-chirp std; keep envelope-then-phase schedule.
4. Axial boosting: order 1 in x,y but z-images to order ~10 (2 m surface-floor spacing => many arrivals inside early window).
5. Add spatially-uniform learned residual + soft time taper instead of hard `max_delay` gate (gate non-differentiable, biases fit near window edge).
6. Source: time-domain FIR + simple cosine-power directivity per octave band (speaker facing down).
7. Fit protocol: 10-20 random inits, pick by held-out pose loss, early stop.
8. `localize.py`: add continuous refine (Adam on position) after grid, Zhou-style accept-only-if-score-improves safeguard.
9. Probe option: quadratic chirp; first-arrival picker must not assume max peak = direct.

**Framing sentences**
- Wang 2024 (DiffRIR): recovers a room's acoustic field from ~12 RIRs and a planar geometry; 6 training points beat neural baselines trained on 100 -- sparse-pose analysis-by-synthesis is the right regime for a 30-pose pool session.
- Li & Chitre 10224658: physics-structured ray/ISM networks trained on hundreds of points in a 2.5 m tank recovered field with 0.38 dB MATE, but tank multipath showed "strong sensitivity to minor geometrical irregularities" and reflection phase is the most position-error-sensitive quantity -- hence learnable geometry error terms and envelope-first fitting.
- Lee 9837411 + Zhou 11617090: SVF-parameterized frequency-sampled filters train far better than raw biquads/FIRs, and sensor-position uncertainty is a "non-negligible" EIV error source for TDOA -- both absorbed into one jointly self-calibrated differentiable channel model.

Files: `src/poolsim/{render,losses,localize}.py`; papers in `papers/`. DiffRIR loss Eq. 6 (Appendix E) not in extract.
