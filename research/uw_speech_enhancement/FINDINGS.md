# Deep-dive findings (2026-10-27)

Raw research notes behind RESEARCH_PLAN.md: five parallel deep-dives (physics, full texts, SE methodology, hardware, venue). Unconfirmed items are marked (unverified).

## FINDINGS: physics
# Simulator physics for the speech band (100 Hz to 8 kHz) in a pool or tank: findings and numbers for `uwsim`

All numbers marked "computed" come from my own Python check of the formulas shown. I did not open every primary source. Anything taken only from a search snippet, or recalled from memory, is marked **(unverified)**.

## TL;DR: what the simulator plan has to change

1. **Absorption is negligible, so drop it or keep it as a no-op.** Over 50 m at 8 kHz it is under 0.03 dB in seawater and under 0.001 dB in freshwater (computed). Reverberation and the boundaries carry all of the physics.
2. **pyroomacoustics cannot model a pressure-release surface.** I read its source. In `libroom_src/wall.cpp`, `energy_reflection = 1 - absorption; transmission = sqrt(energy_reflection)` is real and never negative, so R = -1 (a sign flip) cannot be represented ([wall.cpp](https://raw.githubusercontent.com/LCAV/pyroomacoustics/master/pyroomacoustics/libroom_src/wall.cpp)). Changing `c` with `pra.constants.set("c", 1480)` works ([docs](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.parameters.html)). The README line "pyroomacoustics ShoeBox, R ≈ -1 surface" is therefore wrong as written.
   - **Fix:** write a small custom shoebox image-source model in numpy or torch. In the Allen–Berkley formula each wall's reflection coefficient is raised to an integer power, so a signed coefficient such as R_surface = -1 works directly.
   - Habets' rir_generator may accept a negative β **(unverified; check its input validation)**.
3. **The direct path dominates only within about 0.7-2 m of the source** (the critical radius). At the planned 3, 10 and 20 m positions the direct-to-reverberant ratio is negative, so the benchmark is mainly a **dereverberation** problem, not just denoising.
4. **Below the Schroeder frequency the image-source model is the wrong tool.** That frequency is roughly 300-800 Hz in a 25 m pool and roughly 2-3 kHz in a small tank. Below it, use a hybrid: the image-source model for early reflections, plus a stochastic late tail fitted to the measured energy decay curve per octave, or convolve with the measured IRs directly.
5. **Very little propagates below the cutoff c/(4H).** That is 185 Hz in a 2 m pool. A projector near the surface also loses 10-20 dB below about 500 Hz (Lloyd's mirror).
6. **Doppler doesn't matter for the spectrum but does matter for alignment.** At 1 m/s the time-scaling is 676 ppm, larger than typical clock drift. Add ±0.1% resampling augmentation.
7. **No calibrated public T200 spectrum exists. Measure it yourself.**

---

## (a) Absorption: negligible over 1-50 m

**Francois–Garrison (1982) equations,** as implemented in echopype ([uwa.py](https://raw.githubusercontent.com/OSOceanAcoustics/echopype/main/echopype/utils/uwa.py)). Units: f in kHz, α in dB/km, z = depth in m.

```
c  = 1412 + 3.21T + 1.19S + 0.0167z
A1 = 8.86/c · 10^(0.78·pH − 5);   f1 = 2.8·sqrt(S/35)·10^(4 − 1245/(T+273))      [boric acid]
A2 = 21.44·S/c·(1+0.025T);        P2 = 1 − 1.37e-4 z + 6.2e-9 z²
f2 = 8.17·10^(8 − 1990/(T+273)) / (1 + 0.0018(S−35))                               [MgSO4]
A3 = 3.964e-4 − 1.146e-5T + 1.45e-7T² − 6.5e-10T³  (T ≥ 20 °C)                      [pure water]
P3 = 1 − 3.83e-5 z + 4.9e-10 z²
α  = A1·f1·f²/(f²+f1²) + A2·P2·f2·f²/(f²+f2²) + A3·P3·f²
```

The model is valid from 200 Hz to 1 MHz ([F&G 1982 PDF](https://data-ww3.ifremer.fr/BIB/Francois_Garrison_JASA1982.pdf)). Ainslie & McColm (JASA 103(3):1671, 1998, doi:10.1121/1.421258) is a simplified fit of it, checked from 100 Hz to 1 MHz ([marineinfo](https://marineinfo.org/doc/publication/254927)).

**Computed values** at T = 27 °C (pool), pH 8, z = 2 m. Boric acid f1 = 1.98 kHz, MgSO4 f2 = 190 kHz.

| f (kHz) | Seawater S=35 (dB/km) | Over 50 m | Freshwater S=0 (dB/km) | Over 50 m |
|---|---|---|---|---|
| 0.5 | 0.013 | 0.0006 dB | 4e-5 | ~0 |
| 1 | 0.045 | 0.002 dB | 1.8e-4 | ~0 |
| 4 | 0.23 | 0.012 dB | 0.003 | 0.0001 dB |
| 8 | 0.47 | 0.024 dB | 0.012 | 0.0006 dB |
| 40 (pinger) | 7.1 | 0.35 dB | 0.29 | 0.014 dB |

**Conclusion:** omit absorption in the pool simulator. Keep it only in the Bellhop open-water path, for ranges of a kilometre or more.

## (b) Reverberation in pools and tanks

### Formulas, rescaled for c = 1480 m/s

The in-air constants (2000, 0.161) do not carry over to water.

- **Sabine:** T60 = 55.26·V/(c·A) = **0.0373·V/A** s, where A = Σ Sᵢαᵢ.
- **Schroeder frequency:** f_s = sqrt(c³·T60/(4·ln10·V)) = **18,760·sqrt(T60/V)** Hz. In air the constant is about 2,093, so a water space with the same T60/V has an f_s about 9× higher. This general form is the one used by Trinh et al. (below).
- **Critical radius:** R_c = sqrt(A/(16π)) = sqrt(55.26·V/(16π·c·T60)).
- **Modal density:** n(f) = 4πVf²/c³ modes/Hz.
  - 625 m³ pool: 0.6 at 500 Hz, 2.4 at 1 kHz.
  - 2 m³ tank: 0.008 at 1 kHz, 0.12 at 4 kHz. Such a tank is strongly modal across most of the speech band.

### Measured anchor: DST Maritime Division tank (Trinh, Chen & Forrest, Acoustics 2018)

Source: [AAS2018 p51](https://www.acoustics.asn.au/conference_proceedings/AAS2018/papers/p51.pdf). I read the PDF.

- **Tank:** mild steel, 10.9×10 m at the top, 9.9×9 m at the bottom, 6 m deep, about 600 m³. The steel walls have air behind them.
- **Measured:** T60 in one-third-octave bands up to 16 kHz. The Schroeder (cut-off) frequency is **830 Hz** in the highest band and about 800 Hz wideband. The **maximum critical radius is 0.75 m**.
- **Method:** swept-sine and white-noise excitation, a J9 projector, four TC4013 hydrophones kept at least 1 m from the walls, with λ/4 near-field exclusion at the boundaries.
- **Accuracy:** source levels derived from the reverberant field matched the known source to within 0.4-1.5 dB (2-4 kHz).
- **Back-calculated T60 (my computation):** about **0.8-1.2 s**.

### Other measured data

- **NTU towing tank** (Chao, MS thesis 2019, [NTU](https://tdr.lib.ntu.edu.tw/handle/123456789/73554?locale=en)): EDT, T20 and T30 all under 0.5 s. Side-wall absorption is about 0.97 below 4 kHz and 0.6-0.7 at other frequencies. EDT was more reliable than T20 or T30, and T30 was unsuitable.
- **BYU tank** (Vongsawad, MS 2021, [ETD 9818](https://scholarsarchive.byu.edu/etd/9818)): swept sine, deconvolution, Schroeder backward integration, then α(f) per band. It covers 5-500 kHz only, which is above our band. Numbers are not in the abstract.
- **Penn State ARL tank,** 6.9×8.7×5.5 m: f_s ≈ 630 Hz, a rigid-wall estimate ([JSV 2012 snippet](https://www.sciencedirect.com/science/article/abs/pii/S0022460X12002830)) **(unverified)**.
- **Non-diffuse fields:** the Southampton work (Way et al., Ocean Engineering, and Way's thesis) finds that a towing tank's field is not diffuse, so Sabine and Eyring don't apply directly. They extended the image-source model with spherical-wave reflection coefficients and fitted the concrete's compressional and shear speeds to measured IRs ([thesis record](https://eprints.soton.ac.uk/467700), [paper PDF](https://eprints.soton.ac.uk/444998/1/Ocean_Engineering_Revised_to_submit2.pdf)). Both returned 403, so the numbers are **(unverified)**.
- **No published in-water T60 for a swimming pool** turned up. Our pool measurement would be a small contribution in its own right.

### Predicted values for a 25×12.5×2 m competition pool

Computed with Sabine and Eyring. The surface is lossless; walls and floor total 462.5 m².

| Wall/floor α | T60 | f_s | R_c |
|---|---|---|---|
| 0.05 | 1.0 s | 750 Hz | 0.7 m |
| 0.1 | 0.5 s | 530 Hz | 1.0 m |
| 0.2 | 0.25 s | 380 Hz | 1.4 m |
| 0.3 | 0.17 s | 310 Hz | 1.7 m |
| 0.5 | 0.1 s | 240 Hz | 2.1 m |

Concrete reflects totally beyond its critical angles, so the effective α is probably low (0.05-0.2). That puts T60 at roughly 0.25-1 s (an estimate, to be measured).

### Waveguide low-frequency cutoff

With a pressure-release top and a rigid bottom, f_c = c/(4H). The lowest closed-pool mode is f = (c/2)·sqrt((l/Lx)² + (m/Ly)² + ((2n+1)/(2H))²), whose minimum is c/(4H). So there is no propagating field below c/(4H). (Jensen et al. via [uwacan docs](https://underwater-acoustic-analysis.readthedocs.io/en/latest/reference/generated/uwacan.propagation.cutoff_frequency.html).) A soft bottom raises the cutoff, and measured ponds deviate from the ideal ([bioacoustics.info](https://bioacoustics.info/article/sound-propagation-shallow-water-implications-acoustic-communication-aquatic-animals)).

Computed:

| Water depth H | c/(4H) |
|---|---|
| 1.0 m | 370 Hz |
| 1.2 m | 308 Hz |
| 2 m | 185 Hz |
| 3 m | 123 Hz |
| 4.3 m | 86 Hz |
| 5 m | 74 Hz |

Male F0 sits around 85-180 Hz. In a 2 m pool, voice fundamentals are therefore weak or evanescent and only the harmonics carry. Model this as a depth-dependent high-pass, or rely on measured IRs.

### Reflection coefficients

Normal-incidence R = (Z₂ − Z₁)/(Z₂ + Z₁), with Z_water = 1.48 MRayl. Material constants are textbook values recalled from memory **(unverified)**; the R values are computed.

| Boundary | Z (MRayl) | R normal | Compressional critical angle |
|---|---|---|---|
| Air (surface) | 0.0004 | **−0.999** | — |
| Concrete (ρ 2300, c 3700) | 8.5 | 0.70 | 23.6° |
| Ceramic tile (ρ 2400, c ~5500) | 13 | 0.80 | 15.6° |
| Steel, thick | 46 | 0.94 | 14.5° |
| Fiberglass (GRP) | 5.3 | 0.57 | 32° |
| Acrylic | 3.2 | 0.37 | 33° |
| HDPE | 2.3 | 0.21 | 38° |

- **Concrete is more reflective than its normal-incidence R suggests.** Beyond the critical angles (compressional, and shear at about 42° if c_s ≈ 2200 m/s) |R| approaches 1, apart from mode-conversion loss.
- **Thin-walled tanks backed by air behave like a pressure release, not a rigid wall.** Mass-law plate: Z_plate = jωm.
  - Computed for 3 mm steel, 10 mm acrylic and 6 mm GRP: |R| ≈ 0.999 from 200 Hz to 8 kHz, with phase going from about 179° to 103-138° at 8 kHz.
  - Stiffeners, frames and soil backing change this (computed for an ideal plate only).
  - **For tank simulation, use complex, frequency-dependent wall R or fit it to measured IRs. Don't assume R ≈ +0.7.**
- **Surface roughness** (Rayleigh parameter 2kσ·sinθ): at 8 kHz with σ = 1 cm, coherent loss is small except at steep angles. Swimmers and ripples mainly add time variation.

## (c) Is the image-source model valid in tanks?

- **Li & Chitre, IEEE JOE 2023** (doi:10.1109/JOE.2023.3292417, [arXiv html](https://arxiv.org/html/2205.06066v2)).
  - Setup: a 2.5×1.2×0.8 m fiberglass tank at 10 kHz CW with TC4013 transducers. The model is ray and image-source with the surface fixed at R = −1 and wall and bottom R learned from data.
  - Results: about 0.38 dB mean error with the learned R. The authors stress strong sensitivity to small geometric irregularities, inhomogeneous fiberglass reflection, and only rough knowledge of positions. A plain image-source model with guessed R would not match.
- **Way et al. (Southampton):** image-source model plus spherical-wave reflection coefficients plus volume absorption, validated against MARIN basin transfer functions **(numbers unverified)**.
- **Randomized image method:** pyroomacoustics implements it. Negative reflection coefficients also suppress the non-physical sweeping-echo tail of a rigid shoebox ([FA2025](https://dael.euracoustics.org/confs/fa2025/data/articles/000843.pdf)), which supports a signed-R custom image-source model.
- **Validity limits:**
  - The image-source model is a high-frequency, geometric approximation. Below f_s and near c/(4H) the modal field dominates.
  - For a 625 m³ pool, the model is reasonable above about 0.5-1 kHz. For tanks of 10 m³ or less, it is poor across most of the speech band.
  - The Trinh results show that the diffuse-field assumption holds only above about 800 Hz even in a 600 m³ tank.
- **Recommendation:**
  - Use (i) a signed-R image-source model for the first ~20-50 ms, plus (ii) a late tail of Gaussian noise with octave-band T60 and EDT fitted to the measured IRs.
  - Report a sim-vs-measured comparison per octave: EDT, T20, C50/DRR, and spectral coloration.
  - In ablation, compare with "convolve with measured IRs" as a ceiling.

## (d) Measuring underwater IRs

- **Use an exponential sine sweep (ESS), not MLS.**
  - ESS gives better SNR than MLS underwater (Armelloni, Farina & Burgalassi, UAM 2005, [unipr](https://air.unipr.it/handle/11381/1445059)).
  - ESS also separates the harmonic-distortion IRs, which directly provides data for the Hammerstein projector model ([MDPI Proc. 4(1):53](https://www.mdpi.com/2504-3900/4/1/53); [Farina AES122](https://angelofarina.it/Public/papers/226-AES122.pdf)).
- **Pitfalls** (Farina AES122): low-frequency pre-ringing, impulsive noise, **clock mismatch between playback and record**, and pre-equalization smearing.
  - Clock mismatch applies to us because the Daisy clock differs from the playback DAC clock. Drive playback and capture from the same Daisy, or resample using the end chirp.
- **Suggested protocol:**
  - 96 kHz, ESS from 50 Hz to 20 kHz, 5-10 s long, averaged 4-8 times. Average synchronously only if the clocks are locked.
  - SNR gain is about 10·log10(T·B).
  - Record at least 3 × T60 of silence after each sweep (≥3 s).
  - Compute T20 and EDT per octave with Schroeder backward integration and noise-floor truncation (Lundeby).
- **Time-gating for free-field projector and hydrophone FIRs.** The usable gate is Δt = (first-reflection path − direct path)/c, and the reliable lower frequency is about 2-3/Δt. Buogo et al. ([1999](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-a1f1a8c4-327f-42eb-9e4f-e51a784d1aef/c/Buogo_i_in._HYDROACOUSTICS-nr_2_1999.pdf)) state that the gate is set by geometry only. Computed examples, 1 m range:
  - Both at 1 m depth in a 2 m pool: Δ ≈ 1.24 m, about 0.84 ms, so valid above about 2.5-3.5 kHz.
  - Both at 2 m depth in a 4.3 m well: Δ ≈ 3.1 m, about 2.1 ms, so valid above about 1-1.5 kHz.
  - IEC 60565-1:2020 puts the free-field calibration lower limit at "200 Hz to 5 kHz depending mainly on facility dimensions" and excludes continuous-signal reverberant-tank calibration ([SCC](https://scc-ccn.ca/standardsdb/standards/2045981)).
  - **So below about 1-2 kHz, the projector response must come from near-field or close-range measurement (keep the hydrophone at least λ/4 from boundaries) or from a reverberant-field method (Trinh 2018).**
- **Hydrophone placement:** keep hydrophones at least λ/4 at the lowest frequency of interest from walls. That is 0.45 m at 830 Hz (Trinh 2018), and 1 m was used in practice.

## (e) Near field and directivity of small projectors and hydrophones

Computed for a piston of radius a:

| Radius a | ka at 4 kHz | ka at 8 kHz | Rayleigh distance πa²/λ at 8 kHz |
|---|---|---|---|
| 2.5 cm | 0.42 | 0.85 | 1.1 cm |
| 5 cm | 0.85 | 1.7 | 4.2 cm |
| 10 cm | 1.7 | 3.4 | 17 cm |

- The geometric near field is negligible beyond about 0.2 m.
- **Directivity matters above about 4 kHz** for sources with a ≥ 5 cm (ka > 1). Randomize the source orientation, or apply measured directivity per octave.
- The hydrophones (a few cm) are effectively omnidirectional over 0.1-8 kHz.
- **Lloyd's mirror (surface image dipole):** the gain relative to a free-field monopole is |2·sin(k·d·sinθ)|. Computed at 10° grazing:

| Source depth d | 200 Hz | 500 Hz | 1 kHz | 2 kHz |
|---|---|---|---|---|
| 0.3 m | −21 dB | −13 dB | −7 dB | −1 dB |
| 0.5 m | −17 dB | −9 dB | −3 dB | +3 dB |
| 1.0 m | −11 dB | −3 dB | +3 dB | +6 dB |

  - Mount the projector at mid-depth or deeper. A signed-R image-source model captures this effect automatically; pyroomacoustics cannot. The dipole behaviour is described in [MIT 2.682 notes](https://ocw.mit.edu/courses/2-682-acoustical-oceanography-spring-2012/c46cef6df0457b15a773c0c5f99c2cec_MIT2_682S12_bglec03.pdf).

## (f) Doppler at AUV speeds

Computed for v = 1 m/s: v/c = 6.8×10⁻⁴.

- Pitch shift is 1.17 cents, which is inaudible and irrelevant to ASR. At 8 kHz the shift is 5.4 Hz.
- A 3 s utterance is compressed by 2.0 ms.
- **The effect is spectrally negligible.** But 676 ppm is larger than typical crystal-clock drift (tens of ppm), so the chirp-based drift correction in `bench` must allow for motion. Moving positions also give time-varying multipath with a different Doppler per path.
- **Simulator:** add a ±0.1% resampling augmentation plus a slowly time-varying IR (crossfade between image-source RIRs along a trajectory). Don't build a full Doppler-spread channel for the pool.
- In the open ocean, surface-wave Doppler spread dominates; that belongs in the Bellhop or replay path.

## (g) Thruster, bubble and ESC noise

- **No calibrated public T200 spectrum in dB re 1 µPa @ 1 m was found.**
  - **Davis 2023 (MIT/WHOI S.M., [DSpace 1721.1/152736](https://dspace.mit.edu/handle/1721.1/152736))** tested a single T200 with trapezoidal vs field-oriented control (FOC), and standard vs toroidal propellers. **Motor noise dominated, and FOC was quieter.** The spectra are in the full text; the fetch was rate-limited (429).
  - **RoboVaaS ([arXiv 2206.11082](https://arxiv.org/pdf/2206.11082)):** 4×T200 with the hydrophone about 1.1 m away. In the 50-75 kHz band, noise rose from about −34.5 to −18.2 dB/Hz (system units), with a thruster peak near 72 kHz. That band is outside speech.
  - **Forums:** "drowns out everything while armed" (BlueROV2); a spectrogram exists only as an image ([BR forum](https://discuss.bluerobotics.com/t/t200-motor-noise-level-and-frequency/19800), [hydrophone on ROV](https://discuss.bluerobotics.com/t/use-of-a-hydrophone-on-the-rov/16700)).
  - **Loose AUV reference:** Autosub, about 124 dB re 1 µPa @ 1 m integrated over 100 Hz-5 kHz ([NORA](https://nora.nerc.ac.uk/id/eprint/100256/)) **(snippet only)**.
- **Tonal structure to model:** **(formulas standard; T200 constants unverified)**
  - Shaft rate f_r = RPM/60.
  - Blade-pass frequency = 3·f_r for the 3-blade propeller. If maximum RPM is about 3500 (unverified), that is about 175 Hz.
  - Motor electrical frequency = pole pairs × f_r, plus harmonics.
  - Commutation and PWM switching tones: BLHeli-family ESCs offer 24, 48 or 96 kHz drive PWM ([Bluejay](https://github.com/aocodarc/bluejay)). This is distinct from the 50-400 Hz control-signal rate ([BR forum](https://discuss.bluerobotics.com/t/blheli-32-pwm-frequency/9206/2)).
  - Switching tones sit above the speech band. They matter only through aliasing, intermodulation, or conducted EMI on the hydrophone lines. The Daisy codec's anti-alias filter covers aliasing, so check for conducted EMI by recording with the thruster in air and the hydrophone in water.
- **Bubbles:** Minnaert f₀ = (1/2πa)·sqrt(3γP₀/ρ) ≈ 3.29/a Hz (a in m). That gives **6.6 kHz at 0.5 mm, 3.3 kHz at 1 mm, 1.1 kHz at 3 mm** (computed), squarely in the speech band.
  - Propeller ventilation near the surface and diver exhaust bubbles therefore add in-band noise.
  - Smartphone-in-water work also measured bubble and flow noise at 1-4 kHz and 10-20 dB multipath fading within a few kHz ([arXiv 2209.01780](https://arxiv.org/pdf/2209.01780), [UW messaging](https://underwatermessaging.cs.washington.edu/underwater_comms.pdf)).
- **Action:**
  - Record our own thruster noise per throttle step (10% increments) at 0.5, 1 and 2 m in the pool, with the hydrophone calibrated against the B&K or Daisy chain.
  - Record it in-vehicle (self-noise) and with trapezoidal vs FOC drive if the ESC allows.
  - Release it as part of the noise bank. Do not rely on DeepShip, which is ship noise below about 1 kHz-dominated **(characterization unverified)**.

## Parameter sheet for `uwsim` v0

| Parameter | Value / range |
|---|---|
| c | 1480 ± 10 m/s, freshwater at 25-28 °C (jitter ±0.5%) |
| Absorption | Off. Optional Francois–Garrison for open water. |
| Surface R | −1 (range −0.9 to −1.0); time-varying for ripple |
| Wall/floor R (concrete/tile) | +0.6 to +0.95, angle-dependent if possible; fit to measured EDC |
| Wall R (thin tank) | Complex, \|R\| ≈ 1, phase 100-180°; fit |
| T60 (pool) | Randomize 0.15-1.2 s; center on measured value |
| Image-source order / tail | Image source to about 30-50 ms, then stochastic tail per octave |
| High-pass | Depth-dependent at about c/(4H) |
| Source depth | ≥ 0.5 m (Lloyd's mirror is included automatically with signed R) |
| Directivity | Measured per octave above 4 kHz |
| Doppler | Resample ±0.1%; slowly time-varying IR |
| Noise | Measured T200 per throttle; bubble bursts at 1-7 kHz; SNR randomized −5 to +20 dB |
| Benchmark distances | Include 0.5-1 m (direct-dominated) as well as 3-20 m (reverberant-dominated), since R_c ≈ 0.7-2 m |

Open items to confirm:
- The pool's actual dimensions and depth (dive well vs lap pool). I did not verify the Avery Aquatic Center figures.
- Whether rir_generator accepts a negative β.
- The T200 pole count and maximum RPM, from Davis 2023 or the Blue Robotics specs.

## FINDINGS: fulltexts
# Deep-dive findings: underwater speech enhancement prior art (2026-10-27)

## 0. What changes for the proposal

1. **There is a new close competitor that the first sweep missed: the Kiel DSS group (Gerhard Schmidt).**
   - DAGA 2025 (Röhrdanz, Wisch, … Schmidt), "Data Generation for Training of Neural Networks Dealing with Underwater Speech Transmission" ([PDF](https://pub.dega-akustik.de/DAS-DAGA_2025/files/upload/paper/134.pdf)).
   - DAGA 2026, "Neural-network Based Speech Dereverberation for Underwater Applications" ([doi 10.71568/daga2026.144](https://doi.org/10.71568/daga2026.144), [PDF](https://pub.dega-akustik.de/DAGA_2026/files/upload/paper/144.pdf)).
   - Together they are a physics simulator plus LibriSpeech pairs plus a trained neural SE model for underwater speech, i.e. a closed version of our C1 with one bespoke model.
   - The C1 claim must change: we can no longer say "first simulator that makes paired underwater speech data". Say instead "first **open**, **measurement-validated**, pool-calibrated simulator".
   - C2 (real benchmark) and C3 (cross-architecture sim-to-real) still hold. Kiel has no objective metrics, no ASR, no real benchmark released and only one architecture.
2. **Jiang 2024 already uses an off-the-shelf SE model (MP-SENet) and swaps in BSRNN.** It concludes that "the denoising network's contribution to the system is limited". Cite it accurately: it is a digital semantic link, and its SE runs at the transmitter against diver breathing and bubble noise, not against the water channel.
3. **Rong 2025's 2.4% WER is in-mask, speaker-dependent ASR.** Speech never crosses the water. The model was fine-tuned on 100 sentences from the same diver and tested on 41 sentences, with 4.3 s latency on a Pi 5. This is the material for the rebuttal.
4. **The RealMAN claim is now confirmed with numbers.** The sim-to-real gap depends on the architecture. This is the strongest supporting citation for C3.
5. **The carrier claim needs correcting.** NATO STANAG 1074 analog underwater telephone band-passes 300 Hz to 3 kHz and uses SSB to 8087.5 Hz (source: Kiel DAGA 2025). The ~32 kHz figure is Casio Logosease, a consumer product, not "commercial diver comms" in general. 8 kHz SSB sits far below the RoboSub 25-40 kHz pinger band.

---

## 1. Full-text deep dives

### 1.1 Jiang, Liu, Zhang, Zhu, Yuan — 语义引导的水下带噪语音通信 (SUNSC)

**Citation and access**
- Acta Acustica (声学学报) 49(6):1324-1336, Nov 2024. Xiamen University, MOE Key Lab of Underwater Acoustic Communication. Funded by NSFC 62371404.
- Full PDF read via [jac.ac.cn preview](https://www.jac.ac.cn/cn/article/pdf/preview/10.12395/0371-0025.2024187.pdf). The site rate-limits; a retry worked.
- Correction to PRIOR_ART: the page range is **1324-1336**.

**Pipeline**
- Noisy speech → **MP-SENet** denoiser (off-the-shelf, Lu et al. Interspeech 2023; pretrained 100 epochs).
- Content branch: HuBERT-Base plus K-means (content tokens).
- Pitch branch: YAAPT F0, then a VQ-VAE (pitch tokens). Quantization-loss weight λ = 0.2.
- Bits → **QPSK-OFDM** → channel → semantic correction module → HiFi-GAN-style generator with "MRF-SE" blocks plus MPD/MSD discriminators. Loss weights λ = 45 (mel), μ = 2 (feature matching).
- Timbre is a shared, untransmitted embedding.

**Channel model**
- r̂(t) = h ∗ r(t) + AWGN, with h = Σ A_l(t) δ(t − τ_l(t)).
- h comes from **measured CIRs**, not Bellhop: a "multi-scenario validation dataset" of tens of thousands of CIRs from **NCS** (Norway continental shelf), **NOF** (Oslofjord) and **CWR** (China Wanlu Reservoir). It is cited from Zhao et al., IEEE JSTSP 16(3):474-486, 2022 (the federated meta-learning paper).
- During training, a "degradation-informed model" injects random bit errors with BER ∈ [0.001, 0.1].

**OFDM parameters**

| Parameter | Value |
|---|---|
| Bandwidth | 6.4 kHz |
| Active subcarriers | 128 |
| Symbol time | 40 ms |
| Cyclic prefix / guard | 10 ms / 10 ms |
| Pilot | 40 ms |
| Symbols per frame | 8 |
| Bit rate | **4.7 kbit/s** |
| Sync / equalizer | Chirp sync, frequency-domain MMSE |

SUNSC itself needs only **208 symbols/s**.

**Data**
- Speech: **LJSpeech**, a single female speaker, 13,100 clips, about 24 h, 22.05 kHz, split 9:0.5:0.5.
- Noise: about **15 min of diver breathing and bubble noise recorded in a swimming pool** (Tu et al., JMSE 8(1):44, 2020, [doi](https://doi.org/10.3390/jmse8010044)), resampled to 16 kHz and split 6:2:2.
- Training SNR: random 0-18 dB. Test SNR: fixed {0, 3, …, 18} dB.

**Metrics**
- STOI, cepstral distance (CD), WER using **SenseVoice** ASR, and BLEU.

**Results at SNRs = 0 dB with no channel (Table 4)**

| Method | SR (ksps) | STOI | CD | WER | BLEU |
|---|---|---|---|---|---|
| Clean reference | – | – | – | 5.00% | – |
| HuBERT | 0.208 | 0.790 | 5.033 | 30.34% | 0.708 |
| EnCodec | 0.750 | 0.752 | 8.153 | 31.80% | 0.693 |
| MELP + ½LDPC | 1.200 | 0.443 | 6.207 | 90.48% | 0.565 |
| SEStream | 1.500 | 0.708 | 4.939 | 30.15% | 0.724 |
| DeepSC-S | 327.68 | 0.920 | 4.107 | 13.54% | 0.869 |
| **SUNSC** | 0.208 | 0.790 | 5.007 | **28.83%** | 0.724 |

**Other results**
- NCS channel at SNRc 12 dB: SUNSC STOI 0.803 vs HuBERT 0.735.
- Channel BER at SNRc = 0 / 12 / 24 dB: about 2.3e-1 / 5.7e-2 / 5.0e-3.
- Ablation, STOI at SNRc = 0 / 12 / 24 dB:
  - SUNSC: 0.444 / 0.743 / 0.786
  - w/ BSRNN (denoiser swapped): 0.442 / 0.695 / 0.717
  - w/o SDN (no denoiser): 0.368 / 0.647 / 0.691
- Complexity: 0.52 s per utterance on an RTX 3090Ti; 7.86e7 parameters.

**Real water:** none. The paper says "仿真实验", i.e. simulation experiments only.

**Gaps relative to us**
- The link is digital, not acoustic speech.
- SE fights source noise, not channel reverberation.
- One speaker.
- No projector or hydrophone.

### 1.2 Jin, Zhao, Zhang — Underwater Embedded Offline Speech Recognition System with Ocean Noise Augmentation

**Citation:** IEEE USYS 2024, Xi'an, 18-20 Oct 2024, pp. 1-5. Northwestern Polytechnical University (School of Marine Science and Technology) and China Yantai Salvage Technology Center.

**Access:** the abstract was obtained via [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/USYS62456.2024.11116558?fields=abstract) and the reference list via Crossref. The paper is closed access; the full text was not available.

**From the abstract**
- AISHELL-1 (Mandarin) augmented with an ocean-noise dataset.
- A Transducer end-to-end model, quantized and deployed on a **Raspberry Pi 4B**.
- Evaluated with CER and RTF.

**Inferred from the 14 references (unverified):**
- The noise set is **DeepShip** (ref: ESWA 2021, 115270).
- The ASR is **Zipformer with stateless and pruned RNN-T in k2/icefall/Lhotse**. The references include Zipformer, "RNN-Transducer with Stateless Prediction Network", Lhotse and Interspeech 2022-10340.

**Verdict:** additive-noise augmentation only. There is no channel, no SE and no water propagation. The CER numbers were not retrieved. It is not a threat to novelty; cite it as "ASR-side noise augmentation".

### 1.3 Rong, Chen, Nan — Underwater Diver Communication System Based on Speech/Text Conversion

**Citation:** OCEANS 2025 Brest, [doi 10.1109/OCEANS58557.2025.11104740](https://doi.org/10.1109/OCEANS58557.2025.11104740). Curtin University and UNSW. Full 5-page PDF read from [ddfe.curtin.edu.au/yurong/Oceans25.pdf](https://ddfe.curtin.edu.au/yurong/Oceans25.pdf).

**Rationale given:** 140 words/min × 5.5 chars × 8 bit / 60 ≈ **103 bps** for text. Neural codecs (Lyra, Opus) need "several thousand bps".

**Hardware**
- Raspberry Pi 5 (8 GB) in a watertight canister.
- Interspiro Divator AGA mask microphone.
- Push-to-talk button.
- **Succorfish Delphis** modems:

| Delphis spec | Value |
|---|---|
| Band | 24-32 kHz |
| Source level | 168 dB re 1 µPa @ 1 m |
| Raw data rate | 640 bps |
| Physical layer | Spread-spectrum BPSK with FEC |
| Range | 2 km sea / 3.5 km fresh water |
| Current | 300 mA when transmitting |

- The receiver runs **PicoTTS**.
- Note: Delphis' 24-32 kHz band **overlaps the RoboSub pinger band** (25-40 kHz).

**Fine-tuning**
- Data: **100 sentences from the same diver** "under different underwater acoustic conditions".
- Wav2vec2-base-960h: CNN frozen, Adam, lr 1e-4, 150 epochs.
- Whisper: LoRA r = 8, α = 32, dropout 0.05 on q_proj/v_proj, i.e. 885k trainable parameters (0.36%). lr 1e-3, batch 8, 16 epochs, trained on an RTX A6000.

**Test set:** 41 sentences of "common diver dialogue", each under 4 s, recorded during the trial at Success Boat Harbour, Fremantle, WA. Wind and boat noise were present.

**WER and latency (Table III)**

| Model | Params | WER original | WER fine-tuned | Time on Pi 5 |
|---|---|---|---|---|
| Whisper-small | 244M | 5.7% | 1.7% | 12.9 s |
| **Whisper-base** | 74M | 14.2% | **2.4%** | **4.3 s** |
| Whisper-tiny | 39M | 17.9% | 7.1% | 2.2 s |
| Wav2vec2 | 95M | 41.9% | 8.4% | 1.3 s |

Mean Whisper WER reduction is 71.2%.

**Not reported:** range, distance or modem packet error rate.

**Rebuttal material**
- The WER is measured on **in-mask audio before the water**, so the channel plays no part in it.
- It is speaker-dependent (train and test on the same diver) on 41 utterances.
- 4.3 s ASR latency plus modem and TTS time.
- The listener loses speaker identity and prosody.
- Out-of-vocabulary or unclear speech fails silently.

### 1.4 Luo, Zhou, Qiao — Underwater short-utterance speaker identification via Active Identity Embedding and parameter-efficient adaptation

**Citation:** JASA 159(6):5053-5068, June 2026, [doi](https://doi.org/10.1121/10.0044106). Harbin Engineering University, National Key Lab of Underwater Acoustic Technology. Methods and most results were read on [pubs.aip.org](https://pubs.aip.org/jasa/article/159/6/5053/3393814/Underwater-short-utterance-speaker-identification).

**Method**
- A 16-bit identity watermark embedded into 1-s speech via **HiFi-Codec-16k-320d** (50 frames/s).
- Decoder uses a multi-dilation gated convolutional unit.
- The fine-tuning "attack layer" is baseband **OFDM**:

| OFDM parameter | Value |
|---|---|
| FFT size | 8192 |
| Subcarriers | 4096 |
| Comb pilots | 1024 |
| Bandwidth | 8 kHz |
| Sampling rate | 16 kHz |
| Cyclic prefix | 2048 |
| Modulation | QPSK |
| Equalizer | MMSE |

**Channel:** measured **WATERMARK NOF1 and NCS1** CIRs, chosen as the two sites with the most data, split 8:1:1. AWGN at 10-20 dB for training and 5-25 dB for testing.

**Data:** LibriTTS resampled from 24 to 16 kHz.

**Training**
- Pretraining: 40 epochs, lr 2e-4.
- Fine-tuning: 20 epochs, LoRA r = 4, α = 16 on encoder and decoder (lr 2e-4), full-parameter fine-tuning (FPFT) elsewhere (lr 5e-5), discriminators frozen.
- Trainable parameters: 12.48M, i.e. 18.7% of FPFT's 66.89M.

**Results**
- NOF1 at 15 dB: PESQ-NB 2.52, STOI 0.88, AUC 0.9997, minDCF 0.0215.
- NCS1 at 15 dB: PESQ-NB 2.48, STOI 0.88, AUC 0.9999, minDCF 0.0196.
- BER: under 1% at 15 dB and about 5% at 5 dB.
- ABX listening test (10 listeners × 100 trials): about chance at 15 dB or higher, 61-63% at 5 dB.

**Real water:** none.

**Relevance:** a precedent for **LoRA on a speech model under underwater channels** (in JASA). The task is digital and is not SE.

### 1.5 Fulton, Sattar, Absar — SIREN: Underwater Robot-to-Human Communication Using Audio

**Citation:** IEEE RA-L 8(10), Oct 2023, about pp. 6139-6146. Full text read via [NSF PAR](https://par.nsf.gov/servlets/purl/10481555).

**Projector**
- **Dayton Audio DAEX25W-8**, a 25 mm 10 W 8 Ω IP67 exciter bonded to the LoCO AUV hull so the hull acts as the radiator.
- Driven by a small dual 10 W Parts Express amplifier.
- **Total cost $35.**
- Spec-sheet band: "most capable between 1-10 kHz". The authors say the in-water response differs and did not measure it. Immersed to 2 m for extended periods.

**Study**
- Pool, 12 participants.
- Each soneme played at **1 m, then 15 m or 20 m**, with ambient noise from other pool activity.
- Two signal types: TTS sonemes and tonal sonemes.
- No hydrophone recordings and no objective metric; human identification only.

**Results**
- Accuracy at least 50% overall and at least 70% at 1 m. Operational accuracy at least 80%. Time to answer 5-12 s.
- TTS is better at 1 m (H(1) = 11.57, p < 0.01, η² = 0.049).
- Tonal is far better at 15-20 m (H(1) = 41.55, η² = 0.190).
- Distance effect: H(2) = 107.75, η² = 0.247.
- Longer sonemes are more accurate at distance (r(34) = 0.662).

**Relevance**
- A cheap projector option for EE project #3.
- An independent pool precedent that **speech intelligibility collapses by 15-20 m**, which motivates SE.
- The listening-test design can be reused.

---

## 2. Verification of items marked (unverified)

| Item | Status | Details |
|---|---|---|
| RealMAN arch-dependent sim-to-real gap ([arXiv 2406.19959](https://arxiv.org/abs/2406.19959), NeurIPS 2024 D&B) | **Confirmed, from the full text** | Real-recorded 32-channel test set, two models.<br>**FaSNet-TAC**, sim/real vs real/real: WB-PESQ 1.47 vs 1.46, CER 20.5% vs 21.9%. Authors: "speech simulation does not have the simulation-to-real problem for FaSNet-TAC."<br>**SpatialNet**: PESQ 1.13 vs 2.16, CER **44.2% vs 14.5%**. Moving speaker: CER 49.0% vs 18.6%.<br>CER from a WenetSpeech ASR. The abstract alone does not state it, so cite the results section. |
| RemixIT "any model" ([arXiv 2202.08862](https://arxiv.org/abs/2202.08862)) | **Confirmed** | The abstract says it "can be combined with any separation model". Tzinis et al., IEEE JSTSP 2022. |
| AnyEnhance ([arXiv 2501.15417](https://arxiv.org/abs/2501.15417)) | **Confirmed** | Zhang et al., accepted to IEEE TASLP 2025. Masked generative model with prompt guidance and self-critic. Code is at github.com/viewfinder-annn/anyenhance-v1-ccf-aatc; whether weights are released is (unverified). |
| TRAMBA ([arXiv 2405.01242](https://arxiv.org/abs/2405.01242)) | **Confirmed (arXiv); venue corrected to unconfirmed** | Sui et al. Pretrained on air speech, fine-tuned on about 15 min of real bone-conduction data per user. Up to +7.3% PESQ and +1.8% STOI vs GANs. The paper uses an ACM IMWUT template, but the peer-reviewed venue is (unverified). |
| ESPnet-SE++ ([arXiv 2207.09514](https://arxiv.org/abs/2207.09514)) | **Confirmed** | Lu et al., Interspeech 2022. SE front-ends combined with ASR, speech translation and spoken language understanding; CHiME-4 and WSJ0-2mix benchmarks. |
| SpeechBrain ([arXiv 2407.00463](https://arxiv.org/abs/2407.00463)) | **Confirmed** | "Open-Source Conversational AI with SpeechBrain 1.0", Ravanelli et al., JMLR MLOSS track. Over 200 recipes and over 100 Hugging Face models. |
| ClearerVoice-Studio ([arXiv 2506.19398](https://arxiv.org/abs/2506.19398)) | **Confirmed** | Zhao, Pan, Ma, Interspeech 2025. FRCRN, MossFormer and SpeechScore. |
| Less is More ([arXiv 2506.23859](https://arxiv.org/abs/2506.23859)) | **Confirmed** | Li et al., **ASRU 2025**. A curated 700 h subset beats the full 2,500 h set. |
| MSRANet, DSP 2025 (PII S1051200425002155) | **Paper exists; Bellhop claim still unverified** | Crossref maps the PII to [10.1016/j.dsp.2025.105193](https://doi.org/10.1016/j.dsp.2025.105193), "Adaptive underwater acoustic target recognition based on multi-scale residual and attention mechanism", DSP vol 163, Aug 2025 (Wang, Qian, Chen, Zhou, Li, Xiao). The abstract was not retrievable (ScienceDirect returned 403). Drop the Bellhop claim or keep it marked. |
| Federated meta-learning receiver ([arXiv 2105.13296](https://arxiv.org/abs/2105.13296)) | **Partly resolved** | Published as IEEE JSTSP 16(3):474-486, 2022 (doi 10.1109/JSTSP.2022.3144020). Jiang 2024 cites it as the source of **measured** CIRs from NCS, NOF and CWR. The arXiv abstract does not say which channels were used. Public availability is (unverified). |
| ClariCodec | **Update** | Reissued as [arXiv 2605.19541](https://arxiv.org/abs/2605.19541) (300 bps): 3.55% WER on LibriSpeech test-clean with RL. It mentions underwater channels only as motivation, with no water test. Cite this version, not 2604.14654. |

---

## 3. New 2025-2026 underwater-speech items missed by the first sweep

| Work | What it is | Threat |
|---|---|---|
| **Röhrdanz, Wisch, …, Schmidt (Kiel). DAGA 2025, pp. 951-954** ([PDF](https://pub.dega-akustik.de/DAS-DAGA_2025/files/upload/paper/134.pdf)) | Simulator for training underwater SE networks:<br>• y = s∗h_tx,rx + n + Σ nᵢ∗h_ni,rx (active noise sources convolved with their own IRs).<br>• Random polygon environments with image sources, 180° phase flip per reflection.<br>• Francois-Garrison absorption plus spherical/cylindrical spreading.<br>• LibriSpeech speech.<br>• TX/RX protocols modelled: **STANAG** (300 Hz-3 kHz band-pass, SSB to 8087.5 Hz) and **MAD** (whitened, power-normalized analog path plus codebook index on a digital path).<br>• Example scene: WTD 71 harbour, Kiel, with ship noise.<br>• No network results. Outlook: "tested and validated on both artificially generated as well as recorded signals". | **High for C1** |
| **Röhrdanz et al. (Kiel). DAGA 2026, pp. 814-817** ([doi 10.71568/daga2026.144](https://doi.org/10.71568/daga2026.144)) | U-Net ideal-ratio-mask dereverberation:<br>• 4 encoder/decoder levels, 5×3 convolutions, 32→512 channels.<br>• Input: 128×256 log-magnitude; fs 6 kHz, FFT 512, hop 32.<br>• Training: 84,000 examples, Adam lr 1e-4, 100 epochs, batch 32.<br>• **500 synthetic harbour IRs**: 6-10 interfaces, rooms 200×200×10 m to 800×800×30 m. Cites a Baltic T60 over 2 s (Abshagen & Haak, DAGA 2024).<br>• **Spectrogram figures only: no PESQ, STOI or WER.** The abstract claims evaluation on measurement data, but the paper shows none.<br>• TX/RX processing (MAD) not yet included. Plans to enlarge the datasets and add real recordings. | **High: direct competitor; expect a full paper in 2026-27** |
| Kiel thesis, Dyrßen 2024, "Speech Quality Improvement of Speech Signals after Underwater Transmission Based on Neural Network Approaches" ([list](https://dss-kiel.de/index.php/teaching/theses)) | Finished. Same line of work. Kiel also has a measuring tank and a maritime test facility. | Context |
| Kumar et al., SYMPOL 2025, "Doppler-Robust Zero-Crossing Deep Autoencoder Vocoder…" ([doi](https://doi.org/10.1109/SYMPOL68153.2025.11395914)) | 800 bps digital vocoder on simulated underwater speech. STOI vs LPC-10: 0.81→0.85 clean, 0.65→0.75 at 3 dB, 0.58→0.64 with Doppler 1.05. | Low (digital codec) |
| Dmitriev & Aronov, SUMMA 2025, modified Khurgin-Yakovlev algorithm ([doi](https://doi.org/10.1109/SUMMA68668.2025.11302344)) | Classical analog voice-link processing, simulation only. Band 1-6.3 kHz, carrier 3.3-21 kHz, claims about 75 km range gain. | Low |
| Deeb et al., Archives of Acoustics 2026, KWS-DD ([doi](https://doi.org/10.24423/archacoust.2026.4434)) | HuBERT keyword-spotting model repurposed for diver-breathing detection. 94.4% accuracy from 120 samples. | None (speech model transferred to underwater, not SE) |
| Wisch, Kaak, Namenas, Schmidt, DAGA 2018, "Spracherkennung in stark gestörten Unterwasserumgebungen" (listed on [dss-kiel.de](https://dss-kiel.de/index.php/research/research-topics/underwater-signal-processing/underwater-telephone-systems)) | Underwater ASR from the same group; method not read. | Low (unverified) |

**Chinese-language pass** (声学学报, 应用声学, 信号处理, 通信学报, 电信科学; terms 水下语音增强, 水声语音; plus arXiv and OpenAlex since June 2024): no other deep-learning underwater speech enhancement paper was found besides Jiang 2024. CNKI was not searched directly, so a 2025-26 CNKI-only paper cannot be ruled out.

---

## 4. What kind of research this has to be

The bar is now set by Kiel: simulator, then one U-Net, then spectrograms. To clear it, the work should be **an empirical benchmark and measurement study, not a methods paper**.

1. **Measured channel ground truth.**
   - Exponential sine-sweep IRs at at least 3 source × 3 receiver positions in the pool.
   - Report T60 and early decay time per band.
   - Show simulated vs measured IRs. Kiel and Jiang have no validated simulator.
2. **Real projector-to-hydrophone speech corpus with aligned references.**
   - Use a LibriSpeech test-clean subset, multiple distances (SIREN suggests 1, 5, 10, 15-20 m), and thrusters on/off.
   - Use chirp sync for alignment.
   - This is what nobody has: Kiel promised it, and Jiang, Luo and Rong are digital or in-mask.
3. **Cross-architecture zero-shot vs sim fine-tune vs sim plus real.**
   - At least 3 families (e.g. MetricGAN+ or DeepFilterNet, Demucs or SepFormer, MP-SENet, the last because Jiang used it).
   - Use the RealMAN result as the hypothesis that the gap depends on the architecture.
4. **Objective plus ASR metrics with confidence intervals.**
   - Whisper WER plus one other ASR (SenseVoice, following Jiang), STOI and PESQ.
   - Kiel reports none of these.
5. **Transmission modes as an ablation.**
   - Baseband 300 Hz-3 kHz vs **STANAG-style SSB at 8.0875 kHz**, which is outside the 25-40 kHz pinger band.
   - This aligns with the military standard and with Kiel's setup.
6. **Positioning against Rong.**
   - An end-to-end WER versus distance curve for speech that actually crosses the water.
   - Latency under 1 s vs 4.3 s.
   - Note that their WER does not include the channel at all.
7. **Optional**
   - LoRA vs full fine-tuning across architectures (Luo 2026 shows LoRA on a speech codec under Watermark channels; nobody has done it for SE).
   - A RemixIT-style pass on unlabeled pool audio.

Low-cost hardware options from this pass: the Dayton DAEX25W-8 exciter ($35, SIREN) and the pool breathing/bubble noise set of Tu et al. 2020 (whether it can be obtained is unverified).

---

## Sources

- Jiang 2024 PDF: https://www.jac.ac.cn/cn/article/pdf/preview/10.12395/0371-0025.2024187.pdf ; page: https://www.jac.ac.cn/en/article/doi/10.12395/0371-0025.2024187
- Jin 2024: https://doi.org/10.1109/USYS62456.2024.11116558 ; abstract via https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/USYS62456.2024.11116558?fields=abstract ; references via https://api.crossref.org/works/10.1109/USYS62456.2024.11116558
- Rong 2025: https://ddfe.curtin.edu.au/yurong/Oceans25.pdf
- Luo 2026: https://pubs.aip.org/jasa/article/159/6/5053/3393814/Underwater-short-utterance-speaker-identification
- SIREN: https://par.nsf.gov/servlets/purl/10481555 ; https://irvlab.cs.umn.edu/projects/human-robot-collaboration/older-projects/siren-underwater-robot-human-communication-audio
- Kiel DAGA 2025: https://pub.dega-akustik.de/DAS-DAGA_2025/konferenz-1869.html?article=130 ; DAGA 2026: https://pub.dega-akustik.de/DAGA_2026/konferenz-1909.html ; theses: https://dss-kiel.de/index.php/teaching/theses
- RealMAN: https://arxiv.org/html/2406.19959v2 ; RemixIT: https://arxiv.org/abs/2202.08862 ; AnyEnhance: https://arxiv.org/abs/2501.15417 ; TRAMBA: https://arxiv.org/abs/2405.01242 ; ESPnet-SE++: https://arxiv.org/abs/2207.09514 ; SpeechBrain 1.0: https://arxiv.org/abs/2407.00463 ; ClearerVoice: https://arxiv.org/abs/2506.19398 ; Less is More: https://arxiv.org/abs/2506.23859 ; ClariCodec 300 bps: https://arxiv.org/abs/2605.19541
- MSRANet DOI: https://doi.org/10.1016/j.dsp.2025.105193 ; federated meta-learning: https://arxiv.org/abs/2105.13296
- SYMPOL 2025: https://doi.org/10.1109/SYMPOL68153.2025.11395914 ; SUMMA 2025: https://doi.org/10.1109/SUMMA68668.2025.11302344 ; KWS-DD: https://doi.org/10.24423/archacoust.2026.4434 ; Tu 2020: https://doi.org/10.3390/jmse8010044

Local copies of the full texts:
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\jiang\jiang.pdf
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\siren\siren.pdf
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\daga\134.pdf
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\daga\144_2026.pdf

## FINDINGS: se-methodology
# Experimental design for underwater sim-to-real speech enhancement: what reviewers expect, plus minimal and full designs (2026-10-27)

**Bottom line**
- **The README needs one correction.** It says no LoRA/PEFT work exists for speech enhancement (SE). That is wrong: there are at least three 2026 papers (section 2a). The gap that is still open is narrower: comparing LoRA with full fine-tuning across several architectures under a physical channel shift, measured on real data.
- **Few papers in this area use statistics.** SonicSim, URGENT 2024/25 and RealMAN report no significance tests; RealMAN uses single runs and no error bars. Only Milo et al. reports paired bootstrap 95% CIs. Cluster bootstrap CIs plus paired tests would therefore make this study stand out, and they cost little.
- **No published method exists for computing PESQ/STOI on loudspeaker re-recordings.** The two closest precedents are RealMAN's direct-path reference and the 512-tap filter tolerance in BSS-Eval / CI-SDR (section 2b).

---

## 1. How each reference study is designed

| Study | Training data | Models compared | Metrics | Real test set | Statistics | Ablations / baselines |
|---|---|---|---|---|---|---|
| **SonicSim / SonicSet**, ICLR 2025 ([arXiv 2410.01481](https://arxiv.org/abs/2410.01481)) | About 952 h (960 h in Table 9). 57,102 training trajectories of 60 s at 16 kHz. 1,001 LibriSpeech speakers split 921/40/40. 90 Matterport3D scenes split 62/19/9. Noise from FSD50K plus FMA music with vocals removed. | **20 models.** Separation (11): Conv-TasNet, DPRNN, DPTNet, SuDoRM-RF, A-FRCNN, SKiM, TDANet, BSRNN, TF-GridNet, MossFormer, MossFormer2. Enhancement (9): DCCRN, Fullband, FullSubNet, Fast-FullSubNet, FullSubNet+, TaylorSENet, GaGNet, G2Net, Inter-SubNet. | Intrusive: SI-SNR, SDR, STOI, PESQ. Non-intrusive: NISQA, SigMOS, and DNSMOS (enhancement only). ASR: WER for separation, CER for enhancement (a Chinese ESPnet model, because RealMAN is Mandarin). | RealSEP: 5 h, 30 samples, 10 scenes, a laptop speaker played while walking, one Blue Yeti mic. HumanSEP: 16 live speakers, 6 scenarios. Enhancement was tested on RealMAN. HumanENH size was not read (appendix truncated). | **None.** | Baselines: the same models trained on Libri2Mix / LRS2-2Mix and on VoiceBank-DEMAND / DNS (about 760 h), then tested on real data. Ablation on 300 test-clean files: obstacles, room shape, materials. Removing materials raised MossFormer WER from 24.31% to 31.10%. |
| **Milo et al.**, IWAENC 2026 ([arXiv 2608.20971](https://arxiv.org/abs/2608.20971)) | Hours not stated. Reduced (0.2×, about 30 GB) and full DNS4 speech. Image-source (ISM) set: 60k RIRs from 600 rooms (DNS4/SLR28). Hybrid wave/geometric set (Treble): 324 furnished rooms, 29,144 line-of-sight RIRs, rooms matched on RT60. 70/15/15 split by speaker and room. | **1** (DeepFilterNet3, unchanged), 4 training configurations. | PESQ, SI-SDRi, STOI, SRMR. WER from NVIDIA NeMo 1.23 (checkpoint and normalizer not stated). | 827 VCTK utterances convolved with **284 measured RIRs** (ACE plus MIT), SNR 0-40 dB. This is semi-real: real RIRs, simulated mixing. | Medians per configuration, plus **paired Hybrid−ISM differences with 95% bootstrap CIs**. Example: PESQ +0.166 [0.149, 0.184], STOI +0.013 [0.010, 0.015]. | Small vs full data, 30 vs 120 epochs, decay augmentation on/off. WER: clean 2.24%, noisy 16.71%, Hybrid best 12.74%, ISM best 14.52%. **Gains are not attributed to simulator components** (the authors say so). |
| **URGENT 2024** ([arXiv 2406.04660](https://arxiv.org/abs/2406.04660)) | Curated pool of about 1,300 h speech and 250 h noise. Preliminary split about 400/30/15 h. Sample rates 8 to 48 kHz. Distortions: noise, reverberation, clipping, bandwidth limitation. SNR −5 to 20 dB. | Baselines: OM-LSA, VoiceFixer, Conv-TasNet, BSRNN, TF-GridNet. | **12 metrics.** Intrusive: POLQA, PESQ, ESTOI, SDR, MCD, LSD. Non-intrusive: DNSMOS, NISQA. Task-independent: PhnSim, SpeechBERTScore. Downstream: speaker similarity (RawNet3), word accuracy (OWSM v3.1). | The non-blind test includes real RIRs. Real-recording counts not stated. | None. | Ranking averages ranks per category, then across categories. |
| **URGENT 2025**, Interspeech 2025 ([arXiv 2505.23212](https://arxiv.org/abs/2505.23212)) | Track 1 about 2.5k h, Track 2 about 60k h. Five languages. Seven distortion types plus wind; up to 5 per sample. | TF-GridNet baseline (8.5M parameters). 21+ teams. | **14 metrics**: the 2024 set plus UTMOS, LPS, character accuracy (OWSM v3.1), and P.808 ACR MOS on crowdsourced listeners. | Blind test: 900 samples (150 × 6 languages), **50% real recordings**. | None (Friedman-style rank averaging). | Track 2's extra data did not beat the best Track 1 system. A generative system hallucinated (especially on unseen Japanese at low SNR) while **DNSMOS stayed high**. |
| **URGENT 2024 Lessons**, follow-up ([arXiv 2506.01611](https://arxiv.org/abs/2506.01611)) | n/a | 21 teams | Correlation of each metric with MOS (P.808, 300 samples, 8 vetted listeners each). | 50% real. **Intrusive and downstream metrics were computed only on the simulated half**, because the real half has no reference. | Kendall rank correlation (KRCC) and linear correlation (LCC) with MOS. Overall rank vs MOS rank: KRCC 0.73. | UTMOS and SCOREQ correlate best with MOS. DNSMOS has poor rank correlation. NISQA and DNSMOS Pro are consistent in both rank and value. WADA-SNR is worst. Word accuracy and speaker similarity have high KRCC but low LCC, meaning a non-linear relation to quality. |
| **RealMAN**, NeurIPS 2024 D&B ([arXiv 2406.19959](https://arxiv.org/abs/2406.19959)) | 83.7 h of real speech (48.3 h static, 35.4 h moving) and 144.5 h of noise. 32 speech scenes. 55 Mandarin speakers split 43/6/6. 32-mic array; benchmarks use 9 channels. Distances 0.5-5 m. | **2 SE models** (FaSNet-TAC, a small SpatialNet) and 2 localization models. | SI-SDR, WB-PESQ, DNSMOS (SIG/BAK/OVR), CER (WenetSpeech ASR). | Test set: 21 scenes and 6 speakers, all of them unseen. | **None.** Single runs, no error bars. | Training-data ablation over every combination of simulated/real speech and simulated/real noise. **The sim-to-real gap depends on the architecture.** SpatialNet static sim/sim reached CER 17.9% vs real/real 14.5%. Sim speech with real noise: SpatialNet CER 44.2%, while FaSNet-TAC improved to 20.5%. Real noise always beats simulated noise. |
| **CHiME-7 UDASE** ([arXiv 2307.03533](https://arxiv.org/abs/2307.03533)) | Labeled out-of-domain data: LibriMix. Unlabeled in-domain data: CHiME-5, 83.4 h (27,517 segments, mean 10.9 s). | Out-of-domain teacher (Sudo rm -rf), RemixIT, RemixIT-VAD. | SI-SDR on a synthetic in-domain proxy (reverberant LibriCHiME-5). DNSMOS on real CHiME-5 eval. **P.835 listening test**: 32 listeners in 4 panels of 8, 128 samples, 1,024 votes per condition. | CHiME-5 eval: 4,775 segments, plus a 241-segment listening-test subset (19 min). | Not described in the text read. | RemixIT-VAD beat the teacher by +2.3 dB SI-SDR on eval, yet **DNSMOS OVRL barely moved (2.82-2.88)**. DNSMOS is insensitive to the adaptation gains. |
| **Heitkaemper et al.**, ICASSP 2025 ([Google](https://research.google/pubs/bone-conducted-signal-guided-speech-enhancement-for-voice-assistant-on-earbuds/), [IEEE](https://ieeexplore.ieee.org/document/10889416)) | Pretrained on simulated speech with simulated bone-conduction signal (hours not stated), then fine-tuned on **under 7 h of real earbud data** (IEEE abstract: about 6 h). | 1 model plus 1 multichannel state-of-the-art baseline. | **WER only.** | "Realistic test sets" (count and size not in the abstract) (unverified). | Not in the abstract. | Ablation on the bone-conduction signal bandwidth: cutting it to 500 Hz costs little WER. Full text was not read. |

**Conventions you can infer across these papers**
1. **What gets compared.** Compare the same architecture trained on different data (Milo, SonicSim, RealMAN) and/or several architectures on the same data (SonicSim: 20; RealMAN: 2; URGENT: 5 baselines).
2. **Real tests always include unseen speakers and unseen scenes or positions.**
3. **ASR metric.** Every paper reports WER/CER/word accuracy. The ASR is never Whisper in these papers: NeMo, OWSM, ESPnet and WenetSpeech are used.
4. **Intrusive metrics on real audio** appear only when a constructed reference exists (RealMAN). Otherwise the paper falls back to non-intrusive metrics plus ASR (URGENT, CHiME-7).
5. **A listening test** (P.808 or P.835) is standard for challenges. It is uncommon in single-lab papers.
6. **Statistics.** Bootstrap CIs (Milo) is the only practice seen. The rest give none.

---

## 2. Specific questions

### (a) LoRA / PEFT for SE: prior work exists (fix the README and PRIOR_ART)
- **Cheng & Liu**, "Towards Lightweight Adaptation of SE Models in Real-World Environments", ICASSP 2026 ([arXiv 2603.07471](https://arxiv.org/abs/2603.07471)). Low-rank adapters on a frozen backbone, trained self-supervised with no clean references. Under 1% of parameters. Two backbones, 111 environments, 37 noise types. Average **+1.51 dB SI-SDR after 20 updates per scene**.
- **Bereuter, Plumbley & Sontacchi**, IWAENC 2026 ([arXiv 2607.11630](https://arxiv.org/abs/2607.11630)). Adapts SE models (one discriminative, one generative) to singing voice separation.
  - Full fine-tuning gives the best target-task performance but causes **catastrophic forgetting** of SE.
  - LoRA with 6-12% extra parameters is competitive and keeps SE performance.
- **DiT-Flow** (Cao et al., arXiv preprint [2603.21608](https://arxiv.org/abs/2603.21608)). Mixture-of-LoRA experts, 4.9% trainable parameters, 5 unseen distortions, trained on a SonicSet derivative.
- **Gap that is still open (none found in this search):**
  - LoRA vs full fine-tuning compared **across 3 or more SE architecture families** (masking, time-domain, generative);
  - adaptation to a **physical channel** rather than noise types;
  - forgetting measured on the original in-air domain;
  - results on real data.

  Frame C5 as "LoRA vs full FT vs RemixIT across N families: transfer vs forgetting". Do not frame it as "first LoRA for SE".

### (b) PESQ/STOI on real recordings with a played-back source

There is no standard protocol. Precedents:
- **RealMAN.** The target is the source filtered by an estimated direct-path filter (delay plus gain only), fitted by cross-correlation with a network-guided search (AlignSpatialNet). It was validated by 8 listeners (no audible difference) and identical ASR output.
- **BSS-Eval SDR.** It allows a 512-tap distortion filter. **CI-SDR** (Boeddeker et al., [arXiv 2011.15003](https://arxiv.org/abs/2011.15003)) is a convolutive-transfer-function-invariant version with a PyTorch package (`ci_sdr`). Use it as the "coloration-invariant" fidelity metric.
- **DAPS** (Mysore, IEEE SPL 2015; [page](https://ccrma.stanford.edu/~gautham/Site/daps.html)). Studio speech played through a high-quality loudspeaker and re-recorded on devices, so device versions can be time-aligned to the ground truth (confirmed via the [auditory.org post](https://auditory.org/postings/2015/9.html)).
- **URGENT 2024.** It simply skipped intrusive metrics on the real half.

Recommended recipe:
1. **Align.** Play LFM sync chirps at the start and end of each clip. Cross-correlate each chirp to get the start offset. Estimate clock drift from the start-to-end lag difference and resample to correct it. Then do a fine per-utterance cross-correlation lag search at 16 kHz.
2. **Measure the channel.** At each position, measure h with an exponential sine sweep (ESS).
3. **Build two references and report both:**
   - **R0 = s.** The end-to-end target, if the model is trained to output the clean source. It penalizes projector band-limiting, which the model cannot undo below about 1 kHz.
   - **R1 = s ∗ h_dp.** h_dp is the measured direct-path segment of h: the first few ms, covering projector + hydrophone coloration without pool reverb. This is "reference = source through measured transfer function". **The training target in uwsim must match whichever reference is primary.** If you train toward s ∗ h_proj, R1 is the correct one.
4. **Restrict to the usable band.** Band-limit both references and outputs to the projector's usable band before scoring, and state the band.
5. **Add CI-SDR (512 taps).** It is insensitive to the residual linear-filter choice.
6. **Score at 16 kHz.** Compute PESQ-WB, STOI and ESTOI at 16 kHz; DNSMOS, UTMOS and Whisper also expect 16 kHz input. Keep the 96 kHz raw files for release.

**Non-intrusive metrics on out-of-domain audio: secondary only.**
- **DNSMOS.** Poor rank correlation with MOS (URGENT Lessons). It stayed high under hallucination (URGENT 2025). It did not register a +2.3 dB adaptation gain (CHiME-7).
- **UTMOS** ([arXiv 2204.02152](https://arxiv.org/abs/2204.02152), trained on TTS/VC data) and **NISQA** have the best MOS correlation in URGENT Lessons. UTMOS also has SCOREQ alongside it.
- **MOS-Bench** ([arXiv 2411.03715](https://arxiv.org/abs/2411.03715)) argues that learned MOS predictors often fail out of domain.
- **SQUIM** ([arXiv 2304.01448](https://arxiv.org/abs/2304.01448), ICASSP 2023) predicts STOI, PESQ and SI-SDR without a reference. It is also trained in air. Use it only as a sanity check against true intrusive scores on the subset where references exist; this check is itself a small finding.
- **Recommendation:** report UTMOS and DNSMOS P.835 together with WER. Ideally validate them with a small listening test, for example MUSHRA or P.808 with about 15-20 listeners. CHiME-7 used 32 listeners and 1,024 votes per condition.

### (c) Whisper WER conventions
- **Normalizer.** Use Whisper's `EnglishTextNormalizer` (`whisper.normalizers`) on both reference and hypothesis, as in the Whisper paper ([arXiv 2212.04356](https://arxiv.org/abs/2212.04356)). Compute **corpus-level** WER with jiwer (total errors / total reference words), not the mean of per-utterance WER. Report substitutions, deletions and insertions separately.
- **Decoding.** `large-v3`, `language="en"`, `task="transcribe"`, temperature 0 or beam size 5, `condition_on_previous_text=False`, no prompt. Each utterance is under 30 s, so there are no long-form effects. Disclose whether the default temperature fallback (compression ratio 2.4, logprob −1.0, no-speech 0.6) was on. These are openai-whisper defaults, taken from the code from memory (unverified).
- **Hallucinations.** Barański et al., ICASSP 2025 ([arXiv 2501.11378](https://arxiv.org/abs/2501.11378)): **40.3% of non-speech inferences hallucinated** (121,378 of 301,317), 9.1% of those looped. They propose filtering with a "bag of hallucinations". Careless Whisper (Koenecke et al. 2024) found about 1% of transcripts contained fabricated sentences.
  - Report the insertion rate.
  - Flag utterances where hypothesis length / reference length > 2 or compression ratio > 2.4, and report WER with and without the flagged ones.
  - Run Whisper on **thruster-only and silence segments** and report the hallucination rate. This shows whether SE suppresses or induces hallucinations, a cheap and novel analysis.
- **Second ASR.** Add one non-Whisper ASR (NeMo, as in Milo, or OWSM v3.1, as in URGENT), because URGENT and Milo do not use Whisper.
- **Floor.** Whisper on the clean source played directly (no water) is the floor. Expect about 2% on test-clean (unverified figure).

### Statistics to use
- **Cluster bootstrap.** Resample positions and speakers, not utterances; 10k resamples. Give 95% CIs on WER and on paired metric differences (Milo-style).
- **WER significance.** Paired per-utterance tests: matched-pairs (MAPSSWE) or bootstrap (Bisani & Ney, ICASSP 2004 (unverified DOI)).
- **Continuous metrics.** Wilcoxon signed-rank with Holm correction across the model × condition grid.
- **Seeds.** 3 seeds for every fine-tuning run (in the full design).
- **Rough power estimate.**
  - STOI: per-utterance SD of the paired difference about 0.05 and a 0.02 effect gives n ≈ (2.8 × 0.05 / 0.02)² ≈ **49 utterances**.
  - WER: SD about 15 points and a 5-point effect gives n ≈ **71 utterances**.
  - Position clustering inflates both by roughly 1 + (m−1)·ICC, so aim for **at least 200 utterances per condition**.

---

## 3. Recommended designs

### Minimal design: OCEANS abstract, data by about mid-December
- **Simulated training data.** LibriSpeech train-clean-100 (100.6 h, 251 speakers) through uwsim v0, using the image-source pool, measured projector/hydrophone FIR and our thruster noise. Paired 16 kHz output; about 100 h, or 50 h if compute is tight. Validation on 2 h of held-out speakers.
- **Models (3 families).**
  - MetricGAN+ (SpeechBrain), masking;
  - DeepFilterNet3, deep filtering;
  - Demucs DNS48 or SepFormer-WHAMR, time-domain.
- **Conditions.** Unprocessed; zero-shot; sim full fine-tune. That is 7 rows, single seed. If time permits, add sim + LoRA (rank 8) for DeepFilterNet3 only.
- **Real test set.**
  - 200 LibriSpeech test-clean utterances (5 each from all 40 speakers, about 25 min at an average 7.4 s).
  - 4 positions: 1, 3, 6 and 10 m (or whatever the pool allows), at one depth.
  - Thrusters off and on, so 8 sessions: about 3.3 h of audio plus about 2 h of overhead, i.e. **1-2 pool days**.
  - Score hydrophone channel 0 for the single-channel models. Keep all 4 channels.
  - Record an ESS sweep at each position (for R1 and for simulator validation).
  - Record 2 min of thruster-only audio (for the hallucination probe).
- **Metrics.** Whisper large-v3 WER (primary), ESTOI vs R0 and R1, PESQ-WB, CI-SDR, UTMOS / DNSMOS (secondary). Cluster bootstrap 95% CIs.
- **Figure.** Simulated vs measured pool IR / energy decay curve (EDC) at 2 positions.

### Full design: final paper (24 May 2027) and journal / Interspeech
- **Models.** 6-8 from 4 families:
  - masking: MetricGAN+, FRCRN;
  - deep filtering: DeepFilterNet3;
  - time-domain: Demucs, SepFormer;
  - metric-GAN / conformer: MP-SENet, CMGAN;
  - generative: SGMSE+;
  - URGENT baseline: TF-GridNet.
- **Conditions per model.**
  - zero-shot;
  - sim full fine-tune;
  - sim LoRA (ranks 4/8/16);
  - sim + real fine-tune (0.5, 1 and 2 h of real data, mirroring Heitkaemper's under-7 h);
  - real-only;
  - RemixIT on unlabeled pool audio (CHiME-7 style).

  Also measure **forgetting**: the in-air VoiceBank-DEMAND or DNS score before and after each adaptation (Bereuter-style). Run 3 seeds.
- **Simulator ablations (fixes Milo's admitted gap).** Remove one component at a time:
  - projector FIR;
  - nonlinearity;
  - hydrophone response;
  - reflections (anechoic);
  - surface sign (−1 vs +1);
  - thruster noise vs DeepShip vs white noise;
  - domain randomization off;
  - image-source sim vs measured-IR replay (the Milo analogue).

  Also vary the amount of simulated data: 10, 50 and 200 h. RealMAN's simulated-vs-real noise grid maps to 4 combinations of simulated/real channel × simulated/real noise.
- **Real benchmark v1.**
  - At least 1,000 utterances: test-clean, all 40 speakers, plus at least 6 live talkers. About 3-5 h of speech per channel.
  - At least 8 positions × 2 depths × 2 orientations; thrusters off/on.
  - At least 2 days, to capture session variability.
  - Splits by position and speaker for real training data; leave-one-position-out analysis.
  - Optionally a second site.
  - Release raw 96 kHz 4-channel files plus R0/R1 references, the IRs and alignment metadata.
- **Evaluation.**
  - WER from Whisper large-v3 and NeMo/OWSM, both normalized, with the hallucination analysis;
  - ESTOI, PESQ, CI-SDR vs R0 and R1;
  - UTMOS, DNSMOS, NISQA, SQUIM, with their correlation to the intrusive scores reported;
  - a P.808 or MUSHRA test with 20 listeners × about 24 items per condition on 4-5 systems;
  - optionally multichannel models (for example FaSNet-TAC, as in RealMAN) as a 4-channel extension.
- **Statistics.** Cluster bootstrap CIs, paired tests with Holm correction, mean ± SD over seeds, and a sim-to-real gap (sim score − real score) per architecture. That gap is the headline C3 figure, motivated by RealMAN's architecture-dependent gap.

---

## 4. Not verified, check before citing
- Exact Whisper large-v3 WER on LibriSpeech test-clean (about 2%).
- The Bisani & Ney 2004 DOI.
- The SonicSim HumanENH test size (appendix not read).
- Heitkaemper's test-set details (full text not read).
- SQUIM training data and correlations (only the abstract was read).
- The DNSMOS-C paper (2606.26903) was seen only through secondary pages and is not used above.
- The VoiceMOS 2026 SE track ([arXiv 2609.13792](https://arxiv.org/abs/2609.13792)) was found but not read. It may give a direct out-of-domain reliability comparison of these metrics on SE output.

## FINDINGS: hardware-practical
# Hardware and protocol findings for the real through-water speech benchmark (Stanford RoboSub, OCEANS 2027 abstract)

Checked 2026-10-27. Anything a source did not confirm is marked **(unverified)**. Text marked "Calc" is my own arithmetic, not a sourced figure.

## TL;DR
1. **Projector.** No commercial projector under $2k publishes a full speech-band source-level curve. The Lubell LL916C is the best documented: 180 dB re 1 µPa @ 1 m at 1 kHz, 200 Hz–23 kHz, about $1.5–1.85k plus a $200 transformer box. The LL1424HP costs $13,000 and is sold to government and labs only, so it is out. The Electro-Voice UW30 (100 Hz–10 kHz, about $500) and the Clark AQ339 ($1,350) are the cheaper options. A DAEX25W-8 exciter ($29) is a pilot-only option that SIREN has already used.
2. **The hydrophone front end is a bigger risk than the projector.** The team's hydrophones are custom piezo elements (RS26 TDR). The Daisy input is a 30 kΩ, AC-coupled line input with 0 dBFS at 1 Vrms. A bare piezo of 1–10 nF into 30 kΩ gives a high-pass corner of **5.3 kHz to 530 Hz** (Calc), which removes most of the speech band. Each channel needs a high-impedance buffer or charge-amp preamp with 20–40 dB gain, clamped to ±1.8 V.
3. **Calibration.** A $230–600 Aquarian reference hydrophone plus a side-by-side comparison in the pool is the realistic student method. Full reciprocity is too laborious; use it only as a cross-check above a few kHz.
4. **RoboSub rule.** The pinger band, 25–40 kHz, is "always off limits" for inter-vehicle comms, and the method and frequencies must be posted on Discord. Baseband speech (≤8 kHz) complies. A 32 kHz ultrasonic carrier like Logosease does **not**.
5. **Prior setups.** None of the prior setups recorded speech that crossed the water on a hydrophone with aligned references. Woodward & Sari and Curtin recorded the in-mask microphone. SIREN used human listeners and no recording. This strengthens the benchmark claim (C2).

---

## (a) Projectors, speech band 200 Hz–8 kHz

| Model | Band | Source level | Drive / impedance | Price | Status | Source |
|---|---|---|---|---|---|---|
| **Lubell LL916C / LL916H** | "200 Hz – 23 kHz (500 Hz – 21 kHz ±10 dB)" | max **180 dB re 1 µPa/m @ 1 kHz**; the SPL curve is a separate plot (not read) | 20 Vrms / 3 A at 100% duty; **needs a Lubell transformer box** (AC202C 50 W/20 Vrms, AC205C 78 W/25 Vrms, AC102C 200 W/40 Vrms); recommended amp "78 W @ 8 Ω (25 Vrms) w/AC205C" | Not listed by Lubell. Inyo Pools sells the LL916-200 (speaker only, 200 ft cord) at **$1,539.99**. An eBay new-open-box unit was $1,850. AC205B $209.99, AC202 $203. | Current | [Lubell LL916C/H](https://www.lubell.com/products/lubell-ll916c-ll916h-portable-underwater-speakers-for-lap-synchro-swimming/), [Inyo Pools](https://www.inyopools.com/Products/19400029046141.htm), [eBay.de](https://www.ebay.de/itm/175661931824) |
| Lubell LL9816 | Same class as the LL916 (retailer lists them together) | 180 dB re 1 µPa/m @ 1 kHz (retailer) | Also a 20 V model, must use a Lubell transformer box | n/a | No Lubell datasheet found **(unverified)** | [Inyo Pools](https://www.inyopools.com/Products/19400029046144.htm) |
| Lubell LL1424HP | "useful frequency range of 200 Hz – 9 kHz" | max **196 dB @ 600 Hz** (resonance); 188.75 dB @ 600 Hz on one Crown XLi800 channel | 70 Vrms or 10 A; 4 Ω nominal with the AC1424HP box; depth 1.83–12.19 m | **$13,000**, "government and lab sales only" | Out of budget | [Lubell LL1424HP](https://www.lubell.com/products/ll1424hp-2/) |
| **Electro-Voice UW30** | **100 Hz – 10 kHz** | Not published | 30 W (500–5,000 Hz pink noise), 8 Ω, 50 ft cable, operating depth ≤3.0 m, recommended depth 1.2 m, 1.8 kg; datasheet suggests a 25 Ω/20 W resistor in parallel plus a 1.5 A fast fuse | About **£441 / €389** (UK/DE retail; about $500 Calc) | The Telex-branded listing is "no longer available", but EV-branded listings still exist. **Not confirmed discontinued.** | [UW30 datasheet PDF](https://goknight.com/content/documentation/UW30%20EDS.pdf), [Bax-shop](https://bax-shop.co.uk/installation-speakers/electro-voice-uw30-underwater-announcement-speaker) |
| **Clark Synthesis AQ339 Diluvio** | 20 Hz – 17 kHz (in-air/marketing figure; no underwater SPL given) | Not published | 4 Ω, 135 W continuous, amp 125–150 W @ 4 Ω; 100 ft 16-2 cable | **$1,349.95** at Parts Express | Current | [Parts Express](https://www.parts-express.com/Clark-Synthesis-AQ339-Aquasonic-Underwater-Speaker-300-868) |
| Clark TST239 (tactile transducer, DIY in a housing) | 15 Hz – 17 kHz (in air) | n/a | 4 Ω, 100 W | $239.95 at Parts Express | Not waterproof; would need a sealed plate housing | [Audiophonics](https://www.audiophonics.fr/en/exciters/clark-synthesis-tst239-silver-speaker-driver-exciter-bass-shaker-100w-4-ohm-p-16605.html) |
| **Dayton DAEX25W-8 exciter** | Fs 1,276 Hz; SIREN says it is "most capable between 1k–10 kHz" (in air, from the spec sheet) | Not published | 10 W RMS, 8 Ω, IP67 (1 m for 1 h); SIREN kept it at 2 m for long periods | **$28.98** at Parts Express (MSRP $37.99) | Current | [Dayton](https://daytonaudio.com/product/1181/daex25w-8-waterproof-25mm-exciter-10w-8-ohm), [SIREN paper](https://par.nsf.gov/servlets/purl/10481555) |
| Piezo disks, rings or spheres | Resonant well above the speech band; weak below 1 kHz | n/a | n/a | Cheap | One patent notes "submersible quartz loudspeakers do not reproduce … below 400 Hz". No DIY speech-band data was found. | [US8181735](https://patents.google.com/patent/US8181735B2/en) |

**Below 1 kHz.** No source gives measured source level versus frequency for the LL916, UW30 or AQ339 in the 200–1,000 Hz range. The LL916 is only ±10 dB from 500 Hz. Lubell's transformer box "limits current and flattens the frequency response" ([Inyo Pools](https://www.inyopools.com/Products/19400029046025.htm)), so the measured response depends on which box and amplifier are used. Measure the whole chain (amplifier + box + projector) and report the usable band.

**Distortion.** No vendor publishes THD. Measure it yourself with an exponential sine sweep (ESS), which separates the harmonics from the linear IR ([Farina, AES122](https://angelofarina.it/Public/papers/226-AES122.pdf)). Otero & Felis applied Farina's method to a piezo transducer in water, cutting the signal before the first reflection ([MDPI Proc. 4(1):53](https://www.mdpi.com/2504-3900/4/1/53)).

**Lead to check: ask the pool before buying.** Lubell portable speakers are standard for synchro and artistic swimming ([USA Synchro partnership](https://www.sportsdestinations.com/management/marketing-sponsorships/usa-synchro-announces-partnership-lubell-labs-10448)). If Stanford Aquatics already owns an LL916 or a CAWSS system, borrowing it removes the biggest cost **(unverified)**.

## (b) Amplifiers

- **Lubell.** The amplifier must suit the transformer box and must not be bridged. The AC102 listing says "Amplifier must have unbalanced output — no bridged amplifiers!" Lubell asks for a UL-listed amplifier kept in a dry area and run from a GFCI-protected outlet ([Inyo Pools](https://www.inyopools.com/Products/19400029046030.htm)).
- **Crown XLi800.** Lubell's own LL1424HP research spec uses it. It delivers 2×300 W @ 4 Ω and 2×200 W @ 8 Ω, at **$349** from Full Compass ([Full Compass](https://www.fullcompass.com/prod/223773-crown-xli800-2-channel-300w-at-4-ohm-power-amplifier)). It also suits the AQ339, one channel at 4 Ω.
- **SIREN.** It used a "small dual 10 W amplifier from Parts Express", and the exciter plus amplifier cost $35 in total ([SIREN](https://par.nsf.gov/servlets/purl/10481555)).
- **Power needed (Calc).** At 160 dB re 1 µPa @ 1 m, the LL916 is 20 dB below its 180 dB maximum, which is about 2 Vrms instead of 20 Vrms. A few watts is enough, so a large amplifier is about headroom, not level.

## (c) Hydrophone calibration students can afford

1. **Comparison against a reference hydrophone (recommended).** IEC 60565-1:2020 covers three-transducer reciprocity and relative calibration against a reference. Its range is 200 Hz–1 MHz, and "the lower usable limit depends on tank dimensions … typically 200 Hz to 5 kHz" ([standard.no summary](https://online.standard.no/en/iec-60565-1-2020-ed1)). Gating limit (Calc): with source and receiver 1 m apart at mid-depth in 3 m of water, the surface reflection arrives about 1.46 ms after the direct path. Five or more cycles in that window only works above about 3.4 kHz. Below that, use **side-by-side substitution**: the device under test (DUT) and the reference sit within a few cm of each other, far less than λ (1.5 m at 1 kHz), so both see the same pressure even in the reverberant field. Drive an ESS and take the ratio of the two transfer functions. This is my method reasoning, not a quoted standard.
2. **Reference hydrophones** (Aquarian prices from its store catalogue, `aquarianaudio.com/products.json`):
   - H2d: XLR, phantom-powered; $232 (3 m) to $280 (15 m).
   - S1 (S1e): $496–592.
   - AS-1: $460–560, now deprecated in favour of the S1n. One AS-1 datasheet says "typical AS-1s with 9 m cable … −206 dBV re 1 µPa, ±3 dB" ([reseller PDF](https://vitglobal.odoo.com/shop/as1-hydrophone-aquarian-as1-92/document/6)).
   - H2d sensitivity is about −165 dB re 1 V/µPa, ±4 dB over 20 Hz–4 kHz. The listing prints the unit as "1V/Pa", which is probably a typo **(unverified)** ([audiosense](https://audiosense.be/en/aquarian-audio-h2d-xlr-3-aqu-h2dx3-1)).
   - None of these comes with an individual calibration certificate **(unverified)**.
3. **In-air coupler calibration.** The Aquarian **S1 calibration coupler costs $99**. It fits 1-inch microphone calibrators and "for best results, use at 250 Hz" with GRAS 42AG, Norsonic NOR1256 or ROGA 4010 calibrators (Aquarian catalogue). This gives one absolute anchor point for the reference hydrophone. Calibrator prices were not found. The IOA low-cost pistonphone paper uses a loudspeaker source for 10 Hz–1 kHz ([IOA](https://www.ioa.org.uk/system/files/proceedings/fl_de_magalhaes_oj_ribeiro_afonso_acoustic_aspects_of_a_non-conventional_pistonphone_for_calibrating_hyd.pdf)).
4. **Three-transducer reciprocity.** It is absolute and needs only voltages, currents and distances, but it is "extremely laborious" and transient-limited in small tanks ([Atkins & Zhang, IOA](https://www.ioa.org.uk/system/files/publications/P%20ATKINS%20JX%20ZHANG%20THE%20AUTOMATION%20OF%20THE%20THREE-TRANSDUCER%20RECIPROCITY%20CALIBRATION%20METHOD.pdf)). The team's custom piezos are reversible, so it is possible as a cross-check above about 5 kHz. Do not rely on it for the abstract.
5. **Lead to check.** Borrow a factory-calibrated hydrophone, such as a Reson or B&K unit, from Hopkins Marine Station or CCRMA, which already advised on the hydrophone build **(unverified availability)**.

## (d) Pool safety and RoboSub rules

**RoboSub 2026 Team Handbook** ([PDF, 2026-07-11](https://robonation.org/app/uploads/sites/4/2026/07/RoboSub-2026_Team-Handbook_20260711-compressed.pdf)), p. 98–99. The 2025 handbook has the same wording ([PDF](https://robonation.org/app/uploads/sites/4/2025/07/2025-RoboSub_Team-Handbook-07_21.pdf)):
> "Inter-Vehicle Communication: Teams wishing to have communication between each vehicle must post their method and frequencies on the RoboSub Discord's technical-forum channel. Acoustic frequencies that are the frequency range used by the pingers, are always off limits for inter-sub communication. Inter-vehicle communication and cueing of one vehicle by another is an advanced behavior that merits special points. If such behavior is executed, teams are required to present post-run supporting documentation (e.g., vehicle logs) to the judges."

- **§5.4.3 Pingers.**
  - Pinger: Benthos ALP-365, "user selectable from 25 to 40 kHz in 0.5 kHz increments". The full range of 25–40 kHz and pulse rates of 0.5–2 Hz can be used.
  - Spacing: active pingers are at least 2 kHz apart.
  - Sections: A = 30 kHz, B = 40 kHz, C = 25 kHz, D = 35 kHz.
  - Supply: "These pingers are no longer in production … RoboNation is working to replace the pinger system in future competition years." **The 2027 bands may change.**
- **IVC scoring.** IVC is worth 1,000 points in both 2025 and 2026, graded Simple (1–33%), Intermediate (34–66%) or Complex (67–100%). "Indicator lights … would help with the confirmation."
- **What this means for the project.**
  - Baseband speech up to 8 kHz is compliant.
  - A Logosease-style 32 kHz SSB/AM carrier is **not allowed at RoboSub**. Keep it as an offline ablation only.
  - Projector harmonics of 8 kHz content (3rd harmonic = 24 kHz, 4th = 32 kHz) can leak into 25–40 kHz. Low-pass the drive signal and check the 25–40 kHz spectrum on the 96 kHz raw captures (Calc).
- **Electrical safety.**
  - Lubell requires its transformer box, which provides isolation, plus an amplifier in a dry area on a GFCI outlet (Inyo, above).
  - NEC 680.27(A), on underwater audio, applies to permanent installs with forming shells. The secondary-source summary is in [EC&M](https://www.ecmag.com/magazine/articles/article-detail/a-backyard-oasis-underwater-pool-sound-and-decorative-gas-lighting-systems); the code text was not read.
  - For portable temporary use: battery-powered or GFCI-protected supply, isolation transformer, and no mains gear within reach of the water.
- **Exposure.** Sources found:
  - US Navy: 160 dB for no more than 100 s at 50% duty (160–320 Hz), for trained divers only ([sonicfield summary](https://sonicfield.org/library/evolution-of-navy-diver-exposure-standards-for-deterministic-underwater-sound-in)).
  - A Navy diver study: 2% "very severe" aversion at 148 dB.
  - About 145 dB "safe for sport divers": NJ forum post, **(unverified)** ([tapatalk](https://www.tapatalk.com/groups/njdive/to-dive-boat-captains-dive-shops-and-the-diving-co-t1908.html)).
  - No NATO 154 dB figure was found.
  - **Practice:** record with no swimmers in the water, or keep source levels at or below about 150 dB @ 1 m when people are in the water. Get written approval from Stanford Aquatics; its specific rules were not found.

## (e) How other groups recorded speech or audio through water

| Study | Setup | Distances | Volume of material | Through water on a hydrophone? |
|---|---|---|---|---|
| **SIREN** (Fulton, Sattar, Absar, RA-L 8(10), 2023) | DAEX25W-8 on the LoCO AUV hull plus a dual 10 W Parts Express amp, $35 total; university pool with ambient activity and a water-inlet noise source | **1 m, then 15 or 20 m** | 16 "sonemes" (TTS and tonal); 14 participants, 12 analysed | No. Human listeners only; ≥50% overall accuracy, >70% at 1 m ([PDF](https://par.nsf.gov/servlets/purl/10481555)) |
| **Woodward & Sari**, JASA-EL 121(4), 2007 | Lab tank **9 m × 5.5 m × 2 m**; Knowles CF-2949 electret **inside the mask**; Nagra III tape, 3.4 kHz LPF, 8 kHz/8-bit | Not reported | 1 diver, 4 masks (Wet Mask, Aga, Exo-26, Aqua Lung), "a list of words"; subjective evaluation | No. In-mask microphone ([JASA](https://pubs.aip.org/asa/jasa/article/121/4/EL156/538528)) |
| **Rong, Chen, Nan**, OCEANS 2025 (Curtin) | Success Boat Harbour, Fremantle; AGA (Interspiro Divator) mask mic, Raspberry Pi 5, Delphis modems (24–32 kHz, SL 168 dB, 640 bps), laptop on a jetty | Not reported | **41 test sentences** (4 s clips) from 1 diver; fine-tuned on 100 sentences from the same diver; Whisper-base WER 14.2% → 2.4% after fine-tuning | No. The speech went over the modem as text ([PDF](https://ddfe.curtin.edu.au/yurong/Oceans25.pdf)) |

**Takeaways.**
- Benchmark v0 should beat 41 utterances by a wide margin. For example, 200 LibriSpeech test-clean utterances × 4 positions × thrusters on/off × 4 channels.
- 15–20 m is proven feasible in a pool, even with a $35 exciter.

## (f) Daisy Seed input limits and the preamp

- **Datasheet v1.0.5** ([PDF](https://www.electrokit.com/upload/product/41020/41020659/Daisy_Seed_datasheet_v1.0.5.pdf)):
  - Absolute maximum on the audio inputs: **−1.8 V to +1.8 V**.
  - "Audio inputs are AC coupled and 3.6 Vpp, or approx. 1 Vrms"; **0 dBFS @ 1 Vrms**.
  - **Input impedance 30 kΩ (typ.)**.
- **Codec by board revision** (Electrosmith forum):
  - **AK4556** (Rev 4) is discontinued. Its ADC full scale is 0.7 × VA.
  - **Rev 5 uses the WM8731**: SNR 95–100 dB versus 98–106 dB for the AK4556, and users report about 13 dB more noise ([forum](https://forum.electro-smith.com/t/rev5-seed-producing-13db-higher-noise-floor/3963)).
  - **Rev 7 uses the PCM3060** in hardware mode, at 3 Vpp full scale and 99 dB ADC SNR ([TI](https://ti.com/lit/gpn/PCM3060)).
  - Rev 7 is auto-detected as Rev 4 ([forum](https://forum.electro-smith.com/t/rev-7-seed-is-detected-as-rev-4/4876)).
  - **Check which revision the two boards are.** It sets the noise floor, and it must be in the paper.
- **The piezo loading problem (Calc).** A piezo element looks like a capacitor. Into 30 kΩ, the corner frequency is 1/(2π·R·C): **5.3 kHz for 1 nF and 530 Hz for 10 nF**. A direct connection therefore removes the speech band. Use a JFET buffer or charge amplifier with an input impedance of 10 MΩ or more.
  - Example from the literature: a 7 nF element run flat from 10 Hz to 100 kHz, with about 26 dB of charge gain plus 44 dB of voltage gain ([allaboutcircuits thread](https://forum.allaboutcircuits.com/threads/low-noise-hydrophone-preamplifier.205669/); [Linear Systems FET guide](https://www.linearsystems.com/_files/ugd/7e8069_34cb1d96117b407c913dac4e7560b1e4.pdf)).
  - Off the shelf: the **Aquarian hydrophone preamplifier board, $70**, has 6–56 dB jumper-selectable gain, phantom or 6–12 V supply, and **300 kΩ input** (store catalogue). 300 kΩ still gives a 530 Hz corner at 1 nF (Calc), so it is OK only for elements of 10 nF or more. The **IEPE BNC preamp costs $80**.
- **Level budget (Calc).** Assume source level 160 dB @ 1 m, receiver at 10 m with spherical spreading (140 dB, about 10 Pa), and a piezo sensitivity of −200 dB re 1 V/µPa.
  - That gives about 1 mV, or −60 dBFS. Against a codec noise floor of about −99 dBFS, that is only about 39 dB of SNR.
  - At 1 m the same signal is −40 dBFS.
  - **Use a switchable 20/40 dB gain**, with a rail-to-rail op-amp on a 3.3–5 V supply or clamp diodes, so nothing ever exceeds ±1.8 V.

---

## Recommended setup and budget

| Item | Pick | Cost |
|---|---|---|
| Projector, main option | **Lubell LL916C + AC205C**. Ask Aquatics first whether one can be borrowed. | about $1,750–2,050 |
| Projector, budget option | **EV UW30** (100 Hz–10 kHz) or **Clark AQ339** | about $500 / $1,350 |
| Pilot projector (week 1–2 IR and THD shakedown) | DAEX25W-8 on a sealed plate or hull, plus a small class-D amp | about $60 |
| Amplifier | Crown XLi800, unbridged, on GFCI | $349 |
| Preamp | 4× JFET or charge amp, ≥10 MΩ input, 20/40 dB switchable, ±1.8 V clamp (custom PCB); or Aquarian boards at $70 each if the elements are ≥10 nF | about $50–280 |
| Reference hydrophone | Aquarian **H2d** ($232–280), or **S1e + $99 coupler** for an absolute anchor at 250 Hz | $280–690 |
| Loopback | Amplifier output → attenuator → spare Daisy channel or a USB interface, for alignment and drift ([loopback precedent, FA2026](https://dael.euracoustics.org/confs/fa2026/data/articles/000077.pdf)) | about $0–100 |

The Lubell path comes to about $2.4–3.4k, above the $2k target. The UW30 path comes to about $1.2–1.5k (Calc).

**Protocol.**
1. Confirm the Daisy board revision and the piezo element capacitance.
2. Calibrate the reference hydrophone in air (coupler) and against the projector field. Then calibrate each array hydrophone side by side against the reference with an ESS from 100 Hz to 10 kHz.
3. Measure the projector's transmit response (TVR) and THD with ESS at three drive levels, holding the reference 1 m away. Gate above 3–4 kHz and use substitution below that.
4. Measure pool IRs at 1, 3, 10 and 20 m.
5. Play LibriSpeech with start and end chirps. Low-pass the drive at 8 kHz, keep the source level at or below 150–160 dB, and run with no swimmers in the water.
6. Before any IVC demo, post the method and frequencies on the RoboSub Discord and confirm there is no energy in 25–40 kHz.

## Open and unverified items
- Lubell LL916 source level below 1 kHz: the SPL plot was not read.
- UW30 discontinuation status.
- NEC 680.27 exact text.
- Stanford pool rules.
- Calibrator prices.
- LL9816 datasheet.
- RoboSub 2027 pinger system: replacement is announced, bands unknown.

Local copies, kept so the quotes can be checked:
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\dl\hb26.txt (RoboSub 2026 handbook)
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\dl\siren.txt (SIREN paper)
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\dl\oceans25.txt (Curtin OCEANS 2025 paper)
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\dl\daisy.txt (Daisy Seed datasheet)
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\dl\uw30.txt (UW30 datasheet)
- C:\Users\ryota\AppData\Local\Temp\claude\C--Users-ryota-Documents-repos-robosub-ee\0d8e8feb-985f-482b-a15c-b11c5b5eb071\scratchpad\dl\tdr.txt (Stanford RS26 TDR)

## FINDINGS: venue-craft
# OCEANS 2027 Aberdeen venue research: how to win, deadlines, overlap rules and where to publish the dataset

*Researched 2026-10-27. Anything marked (unverified) or (inferred) needs checking before you act on it.*

## (a) OCEANS 2027 Aberdeen: abstract rules

**Dates.** Source: [important dates](https://aberdeen27.oceansconference.org/important-dates/).

| Item | Date |
|---|---|
| Portal opened | 2026-10-01 |
| **Abstract deadline** | **2027-01-15** (11 weeks from today) |
| Author notification | March 2027 |
| **Final paper due** | **2027-05-24** |
| Conference (P&J Live, Aberdeen) | 2027-06-21 to 24 |

**What the Aberdeen pages do not say.** The [call for abstracts](https://aberdeen27.oceansconference.org/2027-call-for-abstracts/) gives no word limit, figure rule or template. The rules sit behind the login of the [EventsAir portal](https://abdn.eventsair.com/PresentationPortal/Account/Login?ReturnUrl=%2FPresentationPortal%2Fieee-oes-oceans-2027%2Foceans-abstract-submission-portal). Create an account now and read them.

**Rules carried over from the last three editions.** Assume these until the portal says otherwise; they are not confirmed for Aberdeen.
- The abstract is a **PDF of 500 to 1000 words, 2 pages at most**. The 2 pages include text, graphics and references.
- **Figures are allowed** and count toward the 2 pages. No template; put author name, affiliation and title at the top.
- Sources: [Monterey 2026](https://monterey26.oceansconference.org/instructions-for-abstracts-papers-posters-spc/), [Brest 2025](https://brest25.oceansconference.org/submit-an-abstract/), [Great Lakes 2025](https://greatlakes25.oceansconference.org/experience/abstracts-papers-instructions-for-authors-2/).
- Monterey 2026 also had a portal summary of at most 2250 characters (unverified for Aberdeen).
- **Full paper:** 4 to 6 pages plus references at Monterey 2026; 4 to 10 pages at Brest and Great Lakes 2025. IEEE format, published in IEEE Xplore.
- **Abstract review:** at Brest 2025, abstracts were checked for "suitability for the conference and lack of plagiarism". Regular-track acceptance is permissive. The real filter is the Student Poster Competition (SPC).
- Papers are screened for plagiarism against the authors' own work submitted elsewhere ([Great Lakes](https://greatlakes25.oceansconference.org/experience/abstracts-papers-instructions-for-authors-2/)). This matters for the overlap rules in section (e).

**Track to pick.** From the [Aberdeen technical tracks](https://aberdeen27.oceansconference.org/technical-tracks-2027/):
- **Primary:** Track 02, Sonar Signal/Image Processing and Communication. It explicitly covers "acoustic telemetry and communication".
- **Alternative:** Local Track 3, Digital Innovation, which covers "AI in ocean science".
- Track 01 (Underwater Acoustics) fits the simulator-validation angle.
- There is no dedicated AI/ML track.

## (b) Student Poster Competition (SPC): odds, judging and past finalists

**Aberdeen 2027 rules.** Source: [SPC 2027 page](https://aberdeen27.oceansconference.org/student-poster-competition-2027/).
- Applicant must be a **full-time student** in an accredited programme.
- The student must be listed as **lead author and presenter**, and must tick student status in the portal.
- About **20 are selected from 100+ submissions**.
- Selected students get travel and registration support (amount not stated) and must write a full paper plus a poster.
- Monterey 2026 also required a **5-minute oral presentation**, and previous first-place winners cannot re-enter ([Monterey SPC](https://monterey26.oceansconference.org/student-poster-competition/)).

**Judging criteria:** "technical merit, originality, clarity and potential impact within the OCEANS community." Judges expect the problem, method, results and relevance to be explained clearly ([Monterey SPC](https://monterey26.oceansconference.org/student-poster-competition/)). Judging is an interview at the poster: 5 judges at Halifax 2024, 9 at Hampton Roads 2022. The SPC is funded by ONR (Office of Naval Research), per the [Great Lakes plenary page](https://greatlakes25.oceansconference.org/plenary-iii/).

**Prizes:** 1st is the Norman Miller Award, $3,000. 2nd is $2,000 and 3rd is $1,000. Registration is waived and travel and accommodation are supported.

**Selectivity in past years:**

| Edition | Submissions | Shortlisted | Rate |
|---|---|---|---|
| 2022 Hampton Roads | 84 | 27 | 32% |
| 2023 Gulf Coast | 81 | 22 | 27% |
| 2024 Halifax | 110+ | 21 | about 19% |

Sources: [2022](https://beacon.ieeeoes.org/?p=14324), [2023](https://beacon.ieeeoes.org/?p=16762), [2024](https://beacon.ieeeoes.org/?p=19252).

**Winners:**
- **2022:** 1st Veatch (Rutgers), HF radar and Lagrangian coherent structures for prey. 2nd Smith (USNA), bio-inspired lateral-line sensor for UUVs. 3rd Benham (BYU), 3D reef reconstruction from an autonomous surface vessel.
- **2023:** 1st Saran (USM), nested ROMS/COAWST ocean models for oyster reefs. 2nd Bergey (Grove City College, an undergraduate school), underwater time-of-flight camera for an ROV. **3rd Chenyang Zhang (UNSW), a low-complexity underwater acoustic channel emulator.** This is the closest analogue to our simulator contribution.
- **2024:** 1st de Loe (Waterloo), fusing ice surface temperature with the AI4Arctic dataset for sea-ice estimation. 2nd Jacob (URI), sonar-driven AUV path planner for following iceberg walls. 3rd Noel (URI), fiber-optic microtethers.
- **2025 (Brest and Great Lakes):** I could not find the winners (unverified). The [OES December 2025 newsletter PDF](https://ieeeoes.org/wp-content/uploads/2025/12/OES_Dec2025-web.pdf) reportedly has them, but it was too large to fetch.
- **2026:** I could not find the winners.

**Acoustics and ML finalists who did not place:**
- 2022: Hassan (Curtin), DNN OFDM receiver; Kurdila (Penn State), sonar signal decomposition; Panisilvam (Melbourne), VAE for sonar image simulation; Erstorp (KTH), underwater network power control.
- 2024: Eccher (Trento), physical-layer authentication by crafting the underwater acoustic multipath channel; Zhou (Tokyo), acoustic-camera debris counting; Ayyagari (Dalhousie), deep learning on underwater cameras.

**What the winners have in common:**
1. Real, field-collected data or real hardware. None of the podium entries was a pure simulation paper, though Zhang's 3rd-place emulator is partly an exception.
2. A clear application story tied to the OES community: AUVs, ice, reefs.
3. Hardware that can be shown at the poster.
4. Most podium entries came from groups with an ocean focus. An ML-only paper is not favoured. Speech underwater would be **novel within the SPC**: no speech topic appears in the 2022 to 2024 finalist lists.

**How we should pitch it.**
- Lead with the **real pool benchmark** and the AUV/diver voice link.
- Bring the hydrophone array and Daisy Seed rig, plus before/after audio on headphones, to the poster interview.
- Put an undergraduate as the sole student lead author.
- Have the WER table ready for the 5-minute talk.

## (c) Accepted OCEANS papers on underwater acoustic communication with ML, 2022 to 2026

All of these were confirmed through Crossref. The page counts show what reviewers accept.

| Paper | Venue | Pages | Notes |
|---|---|---|---|
| Chen et al., "OFDM UWA Comm Receiver Based on Deep Learning" ([doi](https://doi.org/10.1109/oceans51537.2024.10682211)) | OCEANS 2024 Singapore | 7 | Transformer vs FC-DNN vs CNN, BER only. Abstract shows 3 models and no field trial (whether any field data exists is unverified). |
| Al-Shawabka, Demirors, Melodia, "Real-Time Modulation Classification in UWA Channels Using DL" ([doi](https://doi.org/10.23919/oceans59106.2025.11245148)) | OCEANS 2025 Great Lakes | 7 | CNN/RNN classifier for automatic modulation classification. |
| Fauziya et al., "ML Solutions for Improved Channel Estimation in UWA Comm" ([doi](https://doi.org/10.1109/oceans58557.2025.11104667)) | OCEANS 2025 Brest | 4 | |
| Gahlot et al., "Redefining Channel Estimation in UWA OFDM with DNN" ([doi](https://doi.org/10.1109/oceans58557.2025.11104655)) | OCEANS 2025 Brest | 7 | |
| Hassan et al., "Doppler Compensation Using LSTM in UWA Comm" ([doi](https://doi.org/10.1109/oceanslimerick52467.2023.10244541)) | OCEANS 2023 Limerick | 7 | |
| Khan et al., "Multi-Threaded Autoencoder Hydrophone Array Receiver for LPD Covert UWA Comm" ([doi](https://doi.org/10.1109/oceans66983.2026.11616817)) | OCEANS 2026 Sanya | 8 | Multi-hydrophone fusion; the architecture is close to our 4-channel array. |
| Tian et al., "DL Multi-Step Channel Prediction for Adaptive UWA OFDM" ([doi](https://doi.org/10.1109/oceans66983.2026.11616845)) | OCEANS 2026 Sanya | 5 | |
| Pan et al., "Channel Sounding for Long-Range UWA Comm: Analysis and **Dataset**…" ([doi](https://doi.org/10.1109/oceans58557.2025.11104776)) | OCEANS 2025 Brest | 6 | Shows a dataset-plus-analysis paper is accepted at OCEANS. |
| Blom et al., "Towards UWA Comm for Swimmer and Diver Wearables" ([doi](https://doi.org/10.1109/oceans51537.2024.10682276)) | OCEANS 2024 Singapore | 7 | Real swimming experiments plus "initial conclusions". An exploratory paper with real data is acceptable. |
| Rong, Chen, Nan, "Underwater Diver Comm System Based on Speech/Text Conversion" ([doi](https://doi.org/10.1109/oceans58557.2025.11104740)) | OCEANS 2025 Brest | 5 | The competing design (speech to text over a modem). |

**Typical depth:** 4 to 8 pages, one task, 2 to 4 compared models, 1 or 2 metrics (BER or accuracy), often simulation plus one trial. **We can beat that** with real pool data plus several SE architectures plus Whisper WER, PESQ and STOI.

**Suggested abstract layout (about 800 words, 2 pages):**
1. Motivation: diver/AUV voice, and why speech-to-text over a modem (Rong 2025) loses prosody and needs a modem at both ends. About 100 words.
2. Setup: projector, pool, 4-channel 96 kHz array, simulator. Include one figure of the rig and the pipeline.
3. **Preliminary real numbers.** At least zero-shot WER of Whisper on raw pool audio vs after MetricGAN+/SepFormer, plus the WER after one sim-fine-tuned model. Use one small table.
4. Planned full-paper contributions.

You need pilot pool recordings before 2027-01-15.

## (d) Extending to IEEE JOE (Journal of Oceanic Engineering)

- **Allowed with disclosure.** "Prior publication of an abbreviated or preliminary form of the material in conference proceedings or abstracts" does not stop JOE publication, provided it is disclosed at submission ([JOE submitting guide](https://beacon.ieeeoes.org/publications/ieee-journal-of-oceanic-engineering/submitting-a-paper-to-joe/)).
- **Concurrent submission** "is viewed as a serious breach of ethics".
- **Sanctions** apply for more than 25% crossover with another *journal* manuscript.
- No page limit. Decision in about 3 to 4 months. Optional open access costs $2,645; there are no page charges.
- **Overlap guideline from a JOE special issue.** The UT2025 call says: "No more than about 30% overlap is permitted between the journal article and the symposium paper." The conference paper must be cited, uploaded as supplementary material, and the cover letter must explain the extension ([UT2025 CfP](https://ieeeoes.org/wp-content/uploads/2025/05/UT2025_CfP_FinalV3.pdf)). This is a special-issue rule, but use **at most 30% overlap** as the working target.
- JOE encourages sharing code and data ([editorial](https://ieeexplore.ieee.org/document/8410389)).
- **Route in reverse:** an accepted JOE paper can be presented at OCEANS without a new paper ("JOE paper transfer"; the Monterey 2026 deadline was 2026-07-15) ([Monterey JOE page](https://monterey26.oceansconference.org/town-halls-ieee-joe-papers/)).
- No JOE special issue for OCEANS 2027 has been announced. Watch [JOE special issues](https://ieeeoes.org/publications/ieee-journal-of-oceanic-engineering/joe-special-issues).

## (e) Interspeech 2027 and ICASSP 2028: dates and overlap rules

**Interspeech 2027, São Paulo, 2027-08-29 to 09-02** ([home](https://interspeech2027.org/), [call for papers](https://interspeech2027.org/calls/call_for_papers/)):

| Item | Date or rule |
|---|---|
| Portal opens | 2027-01-19 |
| **Paper deadline** | **2027-02-09** |
| Update deadline | 02-16 |
| Rebuttal | 04-13 to 20 |
| Notification | 06-02 |
| Camera-ready | 06-09 |
| Regular paper | 4 pages + 2 for references |
| **Long Paper track** | 8 + 2 pages, target acceptance below 30% |
| First-author limit | 2 papers per first author |
| Desk rejection | Papers that cite references that do not exist |

- Relevant areas: "Speech Coding and Enhancement" and "Resources and Evaluation".
- Proposals for a challenge were due 2026-11-03; that is too soon for us.
- **Overlap rule:**
  - Submissions must not be "currently submitted to any other conference, workshop, or journal".
  - The 2026 paper kit says cross-checks against other conferences are carried out (unverified for 2027).
  - A "version" includes expanded or summarised papers with a different title ([2024 policy](https://interspeech2024.org/submission-policy/)).
  - **Anonymity period:** no non-anonymised preprint from 1 month before the deadline (about 2027-01-09) until decisions. That is the 2024 rule; the 2027 policy is not yet posted (unverified).
- **Conflict:** our OCEANS abstract is under review from 01-15 to March, and the Interspeech deadline is 02-09. **The same results in both would count as a concurrent submission.**

**ICASSP 2028, Tokyo, 2028-05-28 to 06-03** ([SPS listing](https://signalprocessingsociety.org/events/2028-ieee-international-conference-acoustics-speech-and-signal-processing-icassp)):
- The deadline is not yet announced. ICASSP 2027 was due 2026-09-16 or 09-23; the sources conflict ([ICASSP 2027 call](https://2027.ieeeicassp.org/call-for-papers/)). So **expect mid to late September 2027** (inferred).
- ICASSP requires "original, unpublished technical papers".
- arXiv preprints "do[es] not count as a prior publication" ([ICASSP 2025 policies](https://2025.ieeeicassp.org/conference-policies/)).
- All submissions get an iThenticate self-plagiarism check ([ICASSP 2023 editorial policies](https://2023.ieeeicassp.org/editorial-policies/)).
- The OJSP (Open Journal of Signal Processing) track allows 8+1 pages, published as a journal paper.
- The OCEANS paper will be in IEEE Xplore by then, so the ICASSP paper needs a distinct contribution.
- IEEE allows conference material to grow into journal papers if there is "sufficient new material" and it is disclosed ([ComSoc summary of IEEE PSPB policy](https://comsoc.org/publications/journals/conference-paper-versus-journal-paper)).

**Recommended split:**
1. **OCEANS 2027 (SPC):** benchmark plus rig, simulator validated against measured pool impulse responses and noise, zero-shot vs sim-fine-tuned results for 2 or 3 models, Whisper WER, and the AUV/diver framing.
2. **Interspeech 2027 (02-09):** submit **only** if the ML contribution is clearly separate, for example LoRA/PEFT or RemixIT adaptation across N architectures, with the benchmark cited as "under review". This is risky because there are 15 weeks left and the data overlap could look like double-dipping. The safer option is to skip it.
3. **NeurIPS 2027 Evaluations & Datasets, or ICASSP 2028 / OJSP (about September 2027):** the cross-architecture sim-to-real study, citing the OCEANS paper.
4. **JOE (late 2027):** the full system with at most 30% overlap and disclosure.

## (f) Where to release the dataset

- **NeurIPS 2027 Evaluations & Datasets** (the track was renamed from Datasets & Benchmarks in 2026):
  - 2026 deadlines were abstract 05-04 and paper 05-06. Expect about early May 2027 (inferred).
  - 9 pages. The data must be hosted and accessible at submission, on Dataverse, Kaggle, Hugging Face or OpenML.
  - A Croissant metadata file with Responsible AI fields is required. Over 4 GB, include a reviewer sample.
  - Single-blind is allowed if the data cannot be anonymised.
  - Sources: [E&D call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets), [handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook).
  - Dual submission: other archival papers are treated as prior work, and "if publishing one would render the other too incremental, both may be rejected." The OCEANS paper must therefore be the smaller one.
- **Scientific Data (Data Descriptor):**
  - Open access fee of $2,690 / £2,150 / €2,390, with a waiver policy ([open access page](https://www.nature.com/sdata/open-access), [DOAJ](https://doaj.org/toc/2052-4463)).
  - No deadline and any field. It suits a release after OCEANS, as long as the descriptor adds full documentation beyond the OCEANS paper.
- **Interspeech "Resources and Evaluation" area:** 2027-02-09, with the same overlap caveat as section (e).
- **OCEANS itself:** dataset papers are accepted (Pan et al. 2025).
- **Zero-cost minimum:** Zenodo DOI plus a Hugging Face mirror at OCEANS camera-ready (2027-05-24). The README notes the benchmark claim only holds if the data is released.

## Key action items

1. Create the EventsAir account now and record the real word limit, figure rule and template.
2. Get pilot pool recordings and a zero-shot WER baseline in December 2026.
3. Submit by 2027-01-15 with the student lead author ticked as SPC.
4. Decide on Interspeech by early January, before the anonymity window.
5. Plan the JOE and NeurIPS E&D extensions with at most 30% overlap.