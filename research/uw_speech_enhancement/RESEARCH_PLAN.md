# Research Plan: Underwater Speech Enhancement, Sim-to-Real (rev. 2)

Rev. 3 (2026-10-27): edits from RESEARCH_UPDATE_3.md applied.

*Plan written 2026-10-27 (W1) and revised the same day after two internal reviews: an underwater-acoustics review and a speech/ML review. Lead: Ryota. Companion files: [`README.md`](README.md) (design), [`PRIOR_ART.md`](PRIOR_ART.md) (positioning), [`../whisper_ivc/README.md`](../whisper_ivc/README.md) (capture rig: Daisy Seed, 4 ch, 96 kHz / 24-bit, Whisper on the Jetson Orin).*

**How to read this plan**

- This plan scripts the research that still has to be done. **It contains no measured results.** Every number about our own system is one of three things: a target, a threshold fixed before the data exist, or a `[placeholder]`.
- "Computed" marks first-principles arithmetic from the formulas given. The calculations are collected in Appendix A.
- **(unverified)** marks anything that came from a search snippet or from memory, or that has not been checked against a primary source. Check every one before it goes into a paper. Interspeech desk-rejects papers that cite references that do not exist, and every reviewer will catch a wrong number.
- **What kind of research this is.** This is an *empirical measurement and benchmark study*, not a methods paper. The contribution is a measured, characterized channel, a released real benchmark built on a frozen protocol, and adaptation results that survive strong baselines, pre-registration and test–retest noise floors. The ML recipe itself is not new. Simulate, fine-tune and test sim-to-real is established practice in air (SonicSim, RealMAN, URGENT, Milo et al.).

---

## 0. Corrections to README.md and PRIOR_ART.md adopted by this plan

Fold these into the README and PRIOR_ART before anything is submitted.

| # | Current text | Correction | Source |
|---|---|---|---|
| K1 | "pyroomacoustics ShoeBox, R ≈ −1 surface" | Not possible. pyroomacoustics stores `energy_reflection = 1 − absorption` and `transmission = sqrt(energy_reflection)`. Both are real and non-negative, so a pressure-release surface (R = −1) cannot be represented. Use a custom image-source model (ISM) with signed, angle-dependent reflection coefficients (§4). pyroomacoustics stays in the plan as the *competitor baseline* (an in-air-style simulator with c set to the water value). | [wall.cpp](https://raw.githubusercontent.com/LCAV/pyroomacoustics/master/pyroomacoustics/libroom_src/wall.cpp) |
| K2 | `c = 1480 m/s` | **Use a fresh-water sound-speed equation, not the Francois–Garrison absorption formula.** The expression c = 1412 + 3.21T + … is the rough sound-speed term *inside* Francois–Garrison and was never meant as a sound-speed equation. The Marczak (1997) pure-water polynomial gives 1482.4, 1496.7, 1501.9 and 1504.4 m/s at 20, 25, 27 and 28 °C (computed; coefficients confirmed against the NPL calculator; Marczak, JASA 102(5):2776–2779, 1997, doi:10.1121/1.420332). Del Grosso or Bilaniuk–Wong are cross-checks. **Ask Aquatics whether the pool is salt-chlorinated.** About 3 g/kg salinity adds about +3.3 m/s (UNESCO slope 1.09–1.11 m/s per g/kg at 25–27 °C; NPL). Moving from 1480 to 1497–1502 m/s is about a 1% change. **It is a minor correction, not a material one.** Measure c anyway (§4.3) and log temperature at every session. | Reviewer correction; [echopype uwa.py](https://raw.githubusercontent.com/OSOceanAcoustics/echopype/main/echopype/utils/uwa.py) shows the F&G expression is used inside the absorption formula |
| K3 | C1, "first open, physics-based underwater speech channel simulator that produces paired data" | The Kiel DSS group (Röhrdanz, Wisch, …, Schmidt) has already published an underwater speech simulator ([DAGA 2025 PDF](https://pub.dega-akustik.de/DAS-DAGA_2025/files/upload/paper/134.pdf)) and a U-Net dereverberator trained on it ([DAGA 2026, doi 10.71568/daga2026.144](https://doi.org/10.71568/daga2026.144)). The simulator is no longer our lead contribution. Lead with the **released benchmark and measured channel** (C2). The simulator becomes "open and **measurement-validated**". | Kiel DAGA 2025/2026 |
| K4 | C5, "LoRA/PEFT for SE: no prior work found" | Wrong. See Cheng & Liu, ICASSP 2026 ([arXiv 2603.07471](https://arxiv.org/abs/2603.07471)), Bereuter et al., IWAENC 2026 ([arXiv 2607.11630](https://arxiv.org/abs/2607.11630)), and DiT-Flow ([arXiv 2603.21608](https://arxiv.org/abs/2603.21608)). The open gap is narrower: LoRA vs full FT across several architecture families under a physical channel shift, with forgetting measured and results on real data. | se-methodology findings |
| K5 | "Commercial diver comms use an ultrasonic carrier of about 32 kHz" | The earlier correction ("only Casio Logosease") went too far. Professional through-water diver units use ultrasonic SSB. Retailer spec sheets for the OTS Buddy Phone list channel A at **32.768 kHz USB** and channel B at **31.250 kHz** ([scuba.com surface station](https://www.scuba.com/p-otssbp/ots-buddy-phone-2-channel-surface-station)). An OTS brochure lists 25.000 kHz USB on another channel of a different model ([OTS brochure](https://ndiver-commercial.com/storage/uploads/documents/2019/5/ots-brochure-with-nd-logo.pdf)); channel plans vary by model. NATO STANAG 1074 analog telephony band-passes 300 Hz–3 kHz and uses SSB at 8087.5 Hz (as described in Kiel DAGA 2025). **Any carrier in 25–40 kHz is forbidden at RoboSub.** The pinger band is "always off limits" for inter-vehicle communication ([RoboSub 2026 handbook, pp. 98–99](https://robonation.org/app/uploads/sites/4/2026/07/RoboSub-2026_Team-Handbook_20260711-compressed.pdf)). | Retailer listings (not manufacturer datasheets) |
| K6 | RealMAN, RemixIT, AnyEnhance, ESPnet-SE++, SpeechBrain, ClearerVoice and *Less is More* marked (unverified) | Confirmed. RealMAN numbers: SpatialNet CER 44.2% with simulated speech vs 14.5% real/real; FaSNet-TAC 20.5% vs 21.9% ([arXiv 2406.19959](https://arxiv.org/abs/2406.19959)). | fulltexts §2 |
| K7 | Jiang 2024 page range; "simulated underwater channels" | Pages are 1324–1336. The channels are **measured** CIRs (NCS, NOF, CWR). The SE stage is an off-the-shelf MP-SENet at the transmitter, working against breathing and bubble noise. | [Jiang 2024 PDF](https://www.jac.ac.cn/cn/article/pdf/preview/10.12395/0371-0025.2024187.pdf) |
| K8 | Benchmark distances 1, 3, 10, 20 m; "critical radius ≈ 1 m" | R_c = sqrt(A/16π) ≈ 0.7–1 m comes from a **Sabine diffuse-field estimate with guessed α**. Stanford's Avery Competition pool (37 × 20 m, 3.35–4.27 m deep; gostanford.com) is a **shallow waveguide** with a length-to-depth ratio of about 10:1. The Maas dive well (25 × 17 m, 18 ft per an archived Stanford Athletics page) is the deepest option. The pool is chosen at G1. Most bounces at long range are beyond the critical angle, so the late field is probably non-diffuse and possibly double-slope. Never state R_c as fact. **Report the measured DRR, C50 and EDC curvature per position.** Add 0.5–1 m positions. | Physics findings; reviewer |
| K9 | "Lloyd's-mirror loss of −3 to −11 dB at 200 Hz–1 kHz for 1 m depth at 10°" | Two separate errors. First, the findings table gives **+3 dB at 1 kHz** for 1 m depth at 10° grazing. Second, 10° grazing with both ends at 1 m depth corresponds to about 11 m range. Computed two-point values (surface image only, R = −1, both ends at 1 m depth): **200 Hz: −1.3, −6.2, −11.4, −15.6 dB** and **500 Hz: +2.9, +0.8, −3.7, −7.8 dB** at 1, 3, 6 and 10 m. In a closed pool the full multipath supersedes these numbers; treat them as free-field intuition only. | Appendix A |
| K10 | SIREN "15–20 m shown feasible even with a $35 exciter" | Overstated. SIREN measured **human identification of 16 sonemes**, with no hydrophone recording. TTS accuracy *fell* at 15–20 m. It shows audibility, not a recordable speech-band SNR. | [SIREN](https://par.nsf.gov/servlets/purl/10481555) |
| K11 | "Kiel reports no objective metrics or real-water results" | Write: "the DAGA 2026 paper **presents** no objective metrics or measured-data results". Its abstract claims evaluation on measurement data, and Kiel owns a measuring tank. | [DAGA 2026](https://doi.org/10.71568/daga2026.144) |
| K12 | Iwamoto et al., "venue acceptance unverified" | Confirmed: Interspeech 2022, pp. 5418–5422. Finding: SE *artifacts*, not residual noise, drive ASR degradation. | [ISCA archive](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.html) |
| K13 | "UWA Channel Library (arXiv 2609.03207)" was unsourced | Verified to exist: *Underwater Acoustic Channel Library*, Li, Chitre, Cuji, Preisig, Singer, Stojanovic, van Walree. It provides at-sea channel models, noise models and replay software. Which bands it covers has not been checked. | [arXiv 2609.03207](https://arxiv.org/abs/2609.03207) |
| K14 | Watermark replay of baseband speech | Watermark bands are **4–8, 10–18 and 32.5–37.5 kHz** ([FFI, JOE 2017 abstract](https://publications.ffi.no/en/item/the-watermark-benchmark-for-underwater-acoustic-modulation-schemes)). NOF1 and NCS1 are 10–18 kHz ([EURASIP table](https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-021-01995-3/tables/2)). The KAU band is unverified. Replaying 0.1–8 kHz baseband speech through these channels would need frequency translation, which is not physical. E19 is reframed accordingly (§9). | Reviewer; search |
| K15 | "First … through-water speech benchmark" | Hedge it: "**to our knowledge, the first openly released** through-water speech benchmark with aligned references". Our search did not cover CNKI or the older underwater-telephone literature. | fulltexts §3 |
| K16 | Pool "25 × 12.5 × 2 m (dimensions unverified)" | False for Stanford. Avery Competition pool: 37 × 20 m, 3.35–4.27 m deep. Maas dive well: 25 × 17 m, 18 ft (5.49 m) deep (archived page; the current facilities page gives no depth). The waveguide cutoff c/(4H) is about 88–112 Hz (Avery) or about 68 Hz (Maas), not 187 Hz (computed). Every depth-dependent number in Appendix A is recomputed once the pool is chosen at G1 (D17). | [gostanford.com Avery](https://gostanford.com/facilities/avery-aquatic-center); [archived Maas page](https://gostanford.com/sports/2013/4/17/208433735.aspx) |
| K17 | "The LL916 is only ±10 dB from 500 Hz"; f_lo expected about 300–500 Hz | NUWC SPL curve (20 Vrms, dB re 1 µPa @ 1 m): about 160 dB at 500 Hz vs 179 dB at 1 kHz, so the unequalized −10 dB point is about 560–660 Hz (computed from plot readings). A deep notch of about 148 dB sits near 6.2 kHz, inside the speech band. Rated operating depth 1–18 m, so the source sits at ≥ 1 m. | [LL916C SPL](https://lubell.com/wp-content/uploads/2024/04/LL916C_SPL.pdf); [Lubell](https://www.lubell.com/products/lubell-ll916c-ll916h-portable-underwater-speakers-for-lap-synchro-swimming/) |
| K18 | "Baltic harbour T60 > 2 s" (Abshagen & Haak, as cited by Kiel) | Abshagen & Haak, DAGA 2024, pp. 903–905: bistatic 1–2 kHz reverberation off Schönhagen, Kiel Bay, 22 m deep, with a 20 dB fall within about 2 s. **No T60, and not a harbour.** Use only as a qualitative long-decay example (RQ9). | [DAGA 2024 PDF](https://pub.dega-akustik.de/DAGA_2024/files/upload/paper/467.pdf) |
| K19 | VB-DEMAND as DFN3's "in-domain" forgetting set | DFN3 was trained on DNS4. Its VB-DEMAND result (PESQ 3.17, STOI 0.944) is on an unseen test set, so VB-DEMAND is a *general benchmark* for DFN3, **not in-domain** (RQ6, §7.2). | [arXiv 2305.08227](https://arxiv.org/abs/2305.08227) |
| K20 | Models and data assumed usable for a CC BY 4.0 release | Licence blockers: Demucs (facebookresearch/denoiser) is CC-BY-NC 4.0 and archived, so it is dropped from phase 2; WHAM!/WHAMR! mixtures need licensed WSJ0 (LDC); DEMAND (and so VB-DEMAND) is listed as both CC BY-SA 3.0 and CC BY 4.0; DeepShip and ShipsEar state no licence. These are evaluation or internal-ablation only, never redistributed. | RESEARCH_UPDATE_3 §5; [WHAM!](http://wham.whisper.ai/); [DEMAND](https://zenodo.org/records/1227121) |
| K21 | Implied novelty of recording and enhancing real through-water speech | Real through-water speech has been recorded and enhanced before, without release: IOA-CAS multichannel enhancement of sea-trial SSB voice (Cao, Zhu, Wu 2017), blind deconvolution of frogman SSB voice (Zhang & Liu 2021), Kiel in-pool speech (DAGA 2018), and WTD 71 sea-trial telephone speech with listening tests (2014–2017). The novelty claim is "**openly released**" plus "**pretrained cross-architecture**" adaptation (D19). | RESEARCH_UPDATE_3 §2 |
| K22 | Chirp alignment and measured-channel equalization presented as our idea | Direct precedent: Nissen & Kuhnt-Matthé (WTD 71, DAGA 2017) used in-band OFDM pre- and postambles around STANAG 1475 speech for Doppler correction, segmentation, channel estimation, equalization and ASR. Cite it in §6.5; do not claim the alignment idea. | [DAGA 2017 PDF](https://pub.dega-akustik.de/DAGA_2017/data/articles/000223.pdf) |

---

## 1. Thesis and title candidates

**Thesis.** Speech carried through water from a projector to a hydrophone suffers three kinds of degradation:
- a band-limited, nonlinear projector;
- a shallow-waveguide multipath channel bounded by a pressure-release surface. Its direct path is only resolvable at short range, and its late field is probably not diffuse;
- vehicle and pool noise.

We **measure** this channel in a pool and characterize its uncertainty. We release a real projector-to-hydrophone speech benchmark with a frozen protocol and aligned references. Then we test whether fine-tuning off-the-shelf in-air speech enhancement (SE) models on an open, measurement-validated simulator lowers Whisper WER on real through-water speech. The fine-tuned models are compared against four kinds of baseline:
- unprocessed audio;
- strong non-simulation baselines (WPE dereverberation, reverb-trained SE);
- an in-air reverberation control;
- adapting the recognizer instead of the enhancer.

To make the pool results transferable, we fit a **performance model**: WER as a function of measured SNR, DRR or C50, and T60. It is validated on held-out positions and a second site. Every claim is scoped to *enclosed reverberant water* unless the open-water session (E23) succeeds.

**Headline claim, to be tested.** *Fine-tuning on a pool-calibrated simulator makes in-air SE useful on real through-water speech: it lowers Whisper WER relative to unprocessed hydrophone audio by more than the test–retest noise floor, where zero-shot SE does not.*

**Pre-registered primary endpoint** (filed on OSF before v0-test is scored; §8.5). The endpoint is the paired corpus ΔWER of **DeepFilterNet3 sim-FT (C2) − unprocessed (C0)**. It uses Whisper large-v3 on the v0-test set, pooled over all v0 positions and thruster states, with a cluster bootstrap. Everything else is secondary or exploratory and is labelled as such.

**Title candidates**

1. *Speech Through Water: A Measured Pool Channel and Open Benchmark for Adapting Speech Enhancement to Projector-to-Hydrophone Voice Links* (OCEANS / SPC)
2. *Does Simulated Water Transfer? Sim-to-Real Speech Enhancement and ASR on Real Through-Water Speech* (journal or speech-community extension)
3. *From Measured Pool Impulse Responses to Whisper WER: A Performance Model for Underwater Voice Links* (if RQ9 turns out to be the strongest result)

---

## 2. Research questions

**Tiering.**
- **Confirmatory for the OCEANS paper: RQ1, RQ2 and RQ9.** Hypotheses and thresholds are pre-registered.
- **Exploratory or extension (JOE / ICASSP 2028 / NeurIPS E&D): RQ3–RQ8.**

**Conventions.**
- "CI" means the 95% cluster-bootstrap interval from §8.5.
- δ_WER and δ_ESTOI are equivalence margins fixed in the pre-registration: δ_WER = max(3 points absolute, the test–retest 95% half-width) and δ_ESTOI = max(0.02, the test–retest half-width).
- **Non-significance is never read as support.** Null predictions use TOST equivalence or non-inferiority tests.
- Every RQ has three possible outcomes: supported, refuted, or inconclusive. All three are reported.

### RQ1: What is the pool channel, and does a calibrated simulator reproduce what it was not fitted to? (confirmatory)

- **H1a (channel physics).** The pool's late field is not diffuse. Support requires **both** of the following:
  - a two-slope decay model beats a single-slope model in ≥ 2 octave bands at ≥ half of the measured positions (decision rule pre-registered: two-slope EDC RMS error lower by ≥ 1 dB and BIC lower);
  - the measured DRR at ≥ 3 m differs from the Sabine-diffuse prediction 20·log10(R_c/r) by > 3 dB, with R_c taken from the measured T20.
  - **Refuted** if single-slope fits suffice and the diffuse prediction holds within 3 dB. That outcome would justify a diffuse tail.
  - **Either way,** the result decides the tail model (§4.2 #7).
- **H1b (simulator fidelity).** After calibration, uwsim reproduces held-out positions, meaning source and receiver positions never used in fitting, within the repeatability-based thresholds of §4.4, on quantities that were *not fitted*.
  - Those quantities are per-arrival timing and amplitude, the spatial variation of DRR and C50, coloration, cutoff behaviour and EDC curvature.
  - The **downstream criterion** must also hold: unprocessed sim-twin WER within δ_sim of the real unprocessed WER at each held-out benchmark position.
  - Calibrated uwsim must also **beat the in-air-style competitor baseline** (pyroomacoustics with c set to the water value and absorption-only walls fitted to measured T20) on the downstream criterion.
- **Experiments.** E3 (measure), E5 (fit and validate), E21 (repeatability).
- **One pass rule, used for H1b and gate G3 alike.**
  - **Pass** if the acoustic criteria are met in ≥ 75% of held-out (position × octave) cells **and** the downstream criterion is met at ≥ 3 of 4 held-out benchmark positions.
  - **Fail** otherwise. There is no grey zone.
  - **On fail:** fallback F3 (train on measured-IR replay), and the simulator errors are reported as a finding.
- **Sub-checks** (reported descriptively, not pass/fail):
  - c from the TOF-vs-range slope vs c(T) from Marczak (§4.3);
  - the polarity of the resolved surface arrival in the *measured* IR (a physics observation);
  - band energy below the measured cutoff.
  - The R_s = +1 polarity "test" is dropped as a validation criterion, because it fails by construction. R_s = +1 stays only as training ablation A2.

### RQ2: Does calibrated-simulation fine-tuning help on real data, against strong baselines? (confirmatory; primary endpoint here)

- **H2-primary.** DFN3 sim-FT (C2) lowers corpus WER relative to C0 (unprocessed) on v0-test.
  - **Support:** the CI upper bound of ΔWER < 0 for each of 3 seeds, **and** the seed-mean |ΔWER| > the test–retest floor.
  - **Refuted:** the 90% CI lies within ±δ_WER (equivalence), or ΔWER > 0 significantly.
  - **Otherwise:** inconclusive.
- **H2a (zero-shot, secondary).** Zero-shot in-air SE gives no WER benefit larger than δ_WER at ≥ 3 m. This is consistent with SE artifacts hurting ASR ([Iwamoto et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.html)).
  - Tested **per model** as non-superiority: support if the 90% CI lower bound of ΔWER(C1 − C0) > −δ_WER.
  - **Refuted** for a model if the CI upper bound < 0 and the point estimate < −δ_WER.
- **H2b (is it the *water* physics?).** C2 beats the in-air reverberation control C7 (same recipe on in-air RIRs and the same noise): the CI upper bound of ΔWER(C2 − C7) < 0. This is the evidence that the calibrated water simulator matters, as opposed to "any reverberation fine-tuning".
- **H2c (simulator vs measured IRs).** C2 is non-inferior to measured-IR replay training (C9) with margin δ_WER. This matters because a simulator randomizes beyond the measured positions, while replay does not.
- **H2d (where to adapt; exploratory in the abstract).** Compare SE sim-FT + frozen Whisper with frozen-SE Whisper-LoRA trained on uwsim audio (C8). Report the paired ΔWER with its CI; no directional prediction.
- **Strong baselines that every comparison is reported against:**
  - WPE dereverberation, single-channel and 4-channel;
  - a reverb-trained SE model zero-shot (SepFormer-WHAMR! enhancement; URGENT TF-GridNet (`kohei0209/tfgridnet_urgent25`));
  - Kiel-recipe U-Net trained on uwsim (C10);
  - the C7 control;
  - the C9 replay.
- **Experiments.** E7 (baselines), E8 (fine-tuning).
- **Metrics.** Corpus WER (Whisper large-v3, normalized), paired ΔWER, ESTOI against R1 (§8.2).

### RQ3: Is the sim-to-real gap architecture-dependent? (exploratory or extension)

This is framed as a **replication of RealMAN's in-air finding under a non-room channel** ([arXiv 2406.19959](https://arxiv.org/abs/2406.19959)). It is not presented as a new discovery.

- **H3.** G_m = score_m(sim twin) − score_m(real) differs between architectures.
- **Test.** Paired bootstrap CI of G_a − G_b for each pair, with Holm correction. CI overlap is *not* used as the test.
- **Rank agreement.** Kendall τ is reported only once ≥ 6 models are available, with a bootstrap CI and no threshold. With 3 models τ can only be ±1/3 or ±1, so it is meaningless.
- **Seeds.** 3 per model. Seed SD is reported next to every G_m. A gap difference smaller than 2 × the seed SD is not interpreted.
- **Caveat.** Within-site twin means the sim is calibrated on the same pool it is tested in. The real transfer claim needs the second site (E23).

### RQ4: Which simulator components matter? (extension)

- **H4.** Removing any one of the following increases real-test WER for ≥ 2 architectures, with the CI of ΔWER excluding 0:
  - (a) reflections (anechoic channel);
  - (b) the surface sign (R = +1);
  - (c) the projector model;
  - (d) measured thruster noise (replaced by DeepShip or white noise).
- **Predicted nulls,** tested with TOST at δ_WER: (e) Doppler and time variation, (f) the hydrophone FIR.
- **Experiments.** E10, E11. These address the gap Milo et al. admit, that their gains were not attributed to simulator components ([arXiv 2608.20971](https://arxiv.org/abs/2608.20971)).

### RQ5: How much real data is needed? (extension)

- **H5.** Sim-FT plus 0.5 h of real data beats real-only fine-tuning on 2 h, on WER at unseen v1 positions, for ≥ 2 architectures (paired CI). This mirrors Heitkaemper et al.'s use of under 7 h of real data ([ICASSP 2025](https://ieeexplore.ieee.org/document/10889416)).
- **Experiment.** E12.

### RQ6: Parameter-efficient adaptation vs forgetting (extension)

- **H6.** Under LoRA:
  - real WER is non-inferior to full FT within 10% relative;
  - ≤ 10% of parameters are trained;
  - forgetting on **each model's own in-domain test set** is ≤ 0.1 PESQ-WB, vs ≥ 0.3 for full FT.

  The in-domain test sets are VoiceBank-DEMAND for MetricGAN+. VB-DEMAND for DFN3 is a *general benchmark*, not in-domain (DFN3 was trained on DNS4; it reports PESQ 3.17 on the unseen VB-DEMAND test). WHAM! for SepFormer-wham16k only if the team holds a WSJ0 licence; otherwise use LibriSpeech test-clean mixed with WHAM! noise (CC BY-NC, evaluation only) and label it as not in-domain.
- **Experiment.** E13, with a per-model LoRA insertion plan (§7.2).
- **Scope.** The claim covers only the families actually tested.

### RQ7: Hallucination (extension)

- **H7 (two-sided).** SE changes Whisper's insertion rate on noise-only and silence segments, relative to unprocessed audio. **The direction is not predicted.** Barański et al. report hallucination on 40.3% of non-speech inputs (121,378 of 301,317 inferences, Whisper large-v3, greedy; 9.1% of those looped) ([arXiv 2501.11378](https://arxiv.org/abs/2501.11378)), so near-silent SE output may *raise* hallucination.
- **Experiment.** E14, using the fixed level normalization of §8.3.

### RQ8: Metric validity on underwater audio (partly in the full paper, mostly extension)

- **H8.**
  - Non-intrusive predictors (DNSMOS P.835 OVRL, UTMOS, SQUIM-STOI) agree weakly with **human word-recognition accuracy**: |Kendall τ| < 0.3.
  - ESTOI(R1) and Whisper WER agree more strongly: |τ| ≥ 0.5.
  - τ is reported with a bootstrap CI clustered by system.
- **Anchor.** Human transcription (E22) is the anchor, not another proxy.
- **In-air precedent.** DNSMOS correlates poorly with MOS ([URGENT Lessons](https://arxiv.org/abs/2506.01611)) and stays high under hallucination ([URGENT 2025](https://arxiv.org/abs/2505.23212)).

### RQ9: A transferable performance model and usable region (confirmatory for the full paper; descriptive in the abstract)

- **H9a (performance model).** A Poisson or negative-binomial GLMM on per-utterance error counts with offset log N_ref (insertions can make errors exceed N_ref) predicts WER at held-out positions and sessions within ±δ_pred, with δ_pred = max(5 points, the test–retest half-width). The model is
  `log μ ~ offset(log N_ref) + SNR_speechband + C50 + T20 + system + system:SNR + system:C50 + (1|speaker) + (1|utterance) + (1|position:session)`. exp(β) is a WER ratio.
  - It is validated by leave-one-position-out and, in the full paper, on the second site (E23).
  - **Refuted** if held-out prediction errors exceed δ_pred at ≥ half of the held-out positions.
- **H9b (usable region).** Define the usable region as the set of (SNR, C50) points where predicted WER ≤ 15%. The best sim-FT system extends it: at the median measured C50, the SNR at which WER reaches 15% falls by ≥ 3 dB relative to C0, and the CI excludes 0.
  - This replaces the old "d*" distance metric. Distance in the pool does not map to range in the ocean, because beyond R_c the pool's reverberant level is roughly constant.
- **Transfer.** Use the fitted model to *predict* performance for published channel statistics. One example is Abshagen & Haak (DAGA 2024, pp. 903–905): open water off Schönhagen, Kiel Bay, 22 m deep, bistatic 1–2 kHz sonar reverberation, a 20 dB fall within about 2 s. It reports no T60 and is not a harbour; Kiel DAGA 2026 cites it as "T60 > 2 s". Use it only as a qualitative long-decay example. Predictions outside the measured covariate range are labelled as extrapolation.
- **Latency on the Orin (E20)** is reported descriptively. It is a system property, not a hypothesis. It is not compared with Rong's 4.3 s on a Pi 5, because those are different hardware and pipelines.

---

## 3. Scope

| Dimension | In | Out (or extension only) | Justification |
|---|---|---|---|
| Site | One pool (ours; geometry measured in W1). The **full paper** adds one more site: an open-water session (dock, lake or Hopkins; 2 positions) if access allows, otherwise a second tank or dive well (≥ 20 utterances × 2 positions). | Open ocean at range | If no second site is recorded, all claims are scoped to "**enclosed reverberant water**" and to this pool. Transfer to the ocean goes through the performance model (RQ9), not through raw distance curves. |
| Use case | **Diver-to-diver and diver-to-surface** voice links (primary); robot-to-diver audio as in SIREN (secondary); inter-vehicle (optional) | In-mask ASR (Rong) | The diver application is the real one. The discussion states what does not transfer: regulator and breathing noise, mask acoustics, and the much lower source level of a diver-borne transducer. LibriSpeech read speech at 150–160 dB from a Lubell is a best case. |
| Band | Baseband speech. Drive low-pass at 8 kHz (§6.2 on harmonics). **Scoring band [f_lo, 8 kHz]**, with f_lo = max(projector −10 dB point, 1.5 × the measured cutoff), expected about **500–650 Hz** unless E2 shows usable pre-emphasis. Scored at 16 kHz. | 25–40 kHz carriers (forbidden at RoboSub); in-mask audio | c/(4H) ≈ 88–112 Hz in the Avery Competition pool and ≈ 68 Hz in an 18 ft dive well (computed, c = 1500). The projector sets f_lo. The NUWC LL916 SPL curve gives 160 dB at 500 Hz vs 179 dB at 1 kHz, so the unequalized −10 dB point is about 560–660 Hz (computed from plot readings). A 100 Hz lower edge is not realistic. **STANAG-style SSB (8.39–11.09 kHz)** is offline extension E17 and needs the chain characterized up to about 12 kHz. |
| Channels | 4 channels recorded and released. Single-channel SE models; **4-channel WPE as a baseline in phase 1**. The scored channel is chosen on v0-dev. If RS26 low-frequency sensitivity is poor (E0), the reference hydrophone is the scored channel **by default**. | Multichannel neural SE | The SE ecosystem is single-channel. Multichannel WPE is cheap and is the obvious reviewer request. |
| Speech material | LibriSpeech playback (CC BY 4.0). v1 adds **newly written prompts read by an open TTS voice** (contamination control) and live talkers after IRB approval. | Spontaneous dialogue, non-English | LibriSpeech provides transcripts, speaker-disjoint splits and permission to redistribute. Whisper may have seen LibriSpeech, hence the new-prompt subset. |
| Geometry and motion | Static source and receiver at logged positions, including asymmetric depths for IR validation and a near-surface receiver (0.5 m depth) in v1 | Moving sources | At 1 m/s, v/c = 6.8×10⁻⁴ (computed). Spectrally negligible; handled by resampling augmentation. |
| Noise | Measured T200 noise at ≥ 2 throttle levels from **two physical thruster units** (one held out for test); pool ambient (pumps on and off); bubbles; vehicle self-noise (v1) | Ship noise as a primary source | No calibrated public T200 spectrum exists ([Davis 2023](https://dspace.mit.edu/handle/1721.1/152736); spectra unread). Self-noise claims are scoped to the T200 family. |
| Task | Joint dereverberation and denoising toward a **position-independent chain reference** R1 (§8.2). The surface-image dipole counts as part of the *channel*, so the model is expected to remove it. | Bandwidth extension below f_lo | The model cannot recover content the projector never emitted. |
| OCEANS paper content | RQ1, RQ2, RQ9 | RQ3–RQ8 | A 4–6 page paper cannot carry 9 RQs (reviewer point adopted). |

---

## 4. Simulator specification (`uwsim` v0 → v1)

### 4.1 Signal chain

```
s (clean, 48 kHz internal)
 → drive LPF (8 kHz)                                                 [same as the real playback chain]
 → projector: Hammerstein  y = Σ_{k=1..3} h_k ∗ (x^k)                [h_1 = measured chain FIR; h_2, h_3 from ESS harmonic IRs]
 → directivity G_b(θ) per octave, MEASURED (0°, 45°, 90°; interpolated)
 → channel h_ch(t) =
      ISM_early(t; R_s, R_wall(θ,f), R_floor(θ,f), tilted floor, c)  for t ≤ t_e   (f ≥ f_x)
    + tail_late(t; single- or two-slope per octave, from H1a)        for t > t_e   (f ≥ f_x)
    + low band (f < f_x): measured-IR splice (calibrated pool) | modal sum (generic pools)
 → time variation: resample by (1+ε), |ε| ≤ 1e−3; per-sweep IR variance injected from the measured sweep-to-sweep statistics
 → + noise: Σ_j n_j ∗ h_{j→rx} (thruster IRs from its position) + ambient + bubbles; SNR set from the MEASURED speech/noise level distribution
 → hydrophone FIR h_hyd,ch → preamp HPF (measured) → gain → ADC (measured floor, 24-bit, clip)
 → resample to the model rate (16 kHz; 48 kHz for DFN3)
target: R1_sim = s ∗ h_chain (identical definition to the real R1, §8.2); R0 = s, band-limited
```

The overall structure matches Kiel's y = s∗h_tx,rx + n + Σ nᵢ∗h_ni,rx ([DAGA 2025](https://pub.dega-akustik.de/DAS-DAGA_2025/files/upload/paper/134.pdf)). We differ in that the components are measured, the simulator is validated against held-out positions, and everything is released.

### 4.2 Components, equations and parameter ranges

| # | Component | Equation / model | v0 value | Randomization for training | Source / note |
|---|---|---|---|---|---|
| 1 | Sound speed | c from the TOF-vs-range slope (§4.3), cross-checked against Marczak c(T) | measured; expected about 1497–1502 m/s at 25–27 °C | ±0.5% | K2 |
| 2 | Absorption | Francois–Garrison α(f) | **Off**: < 0.001 dB over 50 m at 8 kHz in fresh water (computed) | Off | [F&G 1982](https://data-ww3.ifremer.fr/BIB/Francois_Garrison_JASA1982.pdf) |
| 3 | Geometry | Shoebox L × W plus a **tilted floor**. First- and second-order floor images are computed exactly against the tilted plane. Higher orders use the mean depth along the source–receiver path (an approximation; its error vs the exact second-order images is reported). | Pool drawings plus measured depth profile (every 2 m) | L, W, H ±10%; slope ±50%; plus 20% generic pools (15–50 × 8–25 × 1.2–5 m) | Reviewer: the slope changes the cutoff and the modes |
| 4 | Surface | R_s real and signed, raised to the power n per image | −1.0 | U(−1.0, −0.9); ripple jitter ±0.02 (v1) | R = −0.999 from the air impedance (computed) |
| 5 | Walls and floor | **Angle-dependent R(θ, f) in v0.** Plane-wave reflection from a fluid–elastic half-space (textbook form, e.g. Brekhovskikh / Jensen et al.; citation page unverified). Parameters per surface: c_p, c_s, ρ and a loss factor η. Evaluated at each image's incidence angle, which the ISM already knows. | 4 params × 2 surfaces fitted (§4.3); priors: concrete or tile c_p 3500–5500 m/s, c_s 2000–3000 m/s, ρ 2200–2500 kg/m³ (unverified textbook values) | fitted ± posterior SD | Way et al. used a similar approach with spherical-wave R (numbers unverified, [thesis](https://eprints.soton.ac.uk/467700)). Grazing paths beyond the critical angle (|R| → 1) dominate at 3–10 m. |
| 6 | Early reflections | ISM with fractional-delay windowed sinc (64 taps at 48 kHz); image order set so that delays ≤ t_e | t_e = 40 ms | t_e ∈ [30, 50] ms | physics §c |
| 7 | Late tail | **Model chosen by H1a.** Single slope: Gaussian noise × exp(−6.91t/T60_b) per octave. Two slopes: a sum of two decays with fitted levels and T60s. Energy is matched to the ISM EDC at t_e over a 5 ms crossfade. | fitted from **calibration positions only** | T60_b × U(0.7, 1.3) | Not a validation metric (§4.4) |
| 8 | Low-frequency band f < f_x | f_x = max(measured f_s, 500 Hz). **Calibrated pool:** splice in the low-passed measured IR from the nearest measured position (Linkwitz–Riley crossover at f_x). **Generic pools:** a modal sum for a shoebox with a pressure-release top and rigid walls and floor, with eigenfrequencies f = (c/2)·sqrt((l/L)² + (m/W)² + ((2n+1)/(2H))²) and modal damping taken from the fitted 250/500 Hz decays. | measured splice | splice source position jittered among the measured IRs | Replaces the old row 9, which used a diffuse tail below f_s while arguing the field is not diffuse there |
| 9 | Cutoff | Emerges from #8; no separate high-pass filter | — | — | [uwacan cutoff](https://underwater-acoustic-analysis.readthedocs.io/en/latest/reference/generated/uwacan.propagation.cutoff_frequency.html) |
| 10 | Projector linear response | Measured **chain FIR h_chain** (projector + box + amplifier → reference hydrophone), from E2 | measured at each source depth used | ±3 dB random 1/3-octave tilt; ±20% resonance shift | §6.4 |
| 11 | Projector nonlinearity | Hammerstein model, k ≤ 3, from ESS harmonic IRs at 3 drive levels | measured | drive level U(−10, 0) dB | [Farina AES122](https://angelofarina.it/Public/papers/226-AES122.pdf) |
| 12 | Directivity | **Measured** at 0°, 45° and 90° per octave (E2). Expected to matter from about 2 kHz. The LL916H is 9.25" (234.95 mm) in diameter per Lubell ([Inyo's LL916C listing](https://www.inyopools.com/Products/19400029046138.htm) says 9.0"), so a ≈ 11.4–11.7 cm and ka = 1 at about 2.03–2.09 kHz (computed). Lubell's polar plots show near-omni behaviour at 1–2 kHz and multiple lobes from 5 kHz. | measured | orientation U(0, 360°) | Reviewer correction (the old row used a = 2.5 cm) |
| 13 | Positions | Source depth ≥ 1 m for the LL916 (rated operating depth 1–18 m); ≥ 0.5 m for other projectors; UW30 ≤ 3 m (1.2 m recommended); receivers anywhere in the water volume. The λ/4 boundary rule applies **only to calibration measurements** (§6.4), not to benchmark validity. | logged | uniform in the volume; ranges 0.3–25 m | Reviewer correction |
| 14 | Time variation | Resampling by (1 + ε); per-band IR perturbation with the variance measured sweep-to-sweep (E3) | ε = 0 | as left | physics §f; E3 |
| 15 | Thruster noise | Measured T200 per throttle, unit A (training) only, convolved with a thruster→receiver IR. Flow-noise segments (jet pointed at the array) flagged and kept separate. | measured | throttle U{levels recorded}; source 0.3–1.5 m from the array | Unit B is held out for test (§5.2) |
| 16 | Bubbles | Minnaert bursts, f₀ ≈ 3.29/a Hz: 1.1–6.6 kHz for a = 0.5–3 mm (computed) | off in v0 | rate U(0, 20)/s | physics §g |
| 17 | Ambient | Pool pump and filter recordings, pumps on and off | measured | random segment | §5.2 |
| 18 | SNR | Speech-band SNR at the hydrophone, sampled from the **measured** per-take SNR distribution (E6) widened by ±10 dB | — | measured range ± 10 dB, clipped to [−10, 25] dB | Replaces the guessed U(−5, 20) |
| 19 | Hydrophone response | Per-channel FIR (§6.4); measured mount directivity at 4–8 kHz | measured | ±2 dB tilt | E1 |
| 20 | ADC | Measured codec floor; hard clip at 0 dBFS | measured | ±3 dB | Daisy revision identified in E0 |

**Implementation**
- NumPy ISM with signed, angle-dependent R, plus a PyTorch on-the-fly mixer.
- pyroomacoustics is used only for the in-air control C7 and for the competitor baseline in §4.4.
- Check Habets' `rir_generator` with negative β as a cross-check of the ISM (confirmed: only shape and type are checked and image amplitudes use `pow(beta, |n|)`, so the sign is preserved; pass a 6-element β, because a scalar is read as T60).
- Negative reflection coefficients suppress the sweeping-echo artifact of a rigid shoebox ([FA2025](https://dael.euracoustics.org/confs/fa2025/data/articles/000843.pdf)).

### 4.3 Calibration procedure

1. **Inputs (E3, session S1).**
   - ESS IRs on a **calibration set**: sources S1 and S2 × receivers R1–R3, giving 6 pairs.
   - A **validation set at wholly new positions**: source S3 × receivers R4 and R5, plus the 3 benchmark positions recorded in S2. None of these source or receiver locations appear in calibration.
   - **Depths are asymmetric** for IR validation (e.g. source 0.7 m, receiver 1.2 m in 2 m of water). With both ends at mid-depth, the surface and floor images have identical path lengths and partly cancel: R_s ≈ −1 against R_b ≈ +0.7…1. At r = 1 m with the asymmetric depths, the surface arrival lags the direct path by 0.69 ms and the floor arrival lags the surface by 0.12 ms (computed), so the two can be resolved with ≥ 10 kHz of bandwidth.
2. **Preprocess.**
   - Loopback-referenced drift correction *before* deconvolution (§6.4).
   - Compensate the chain FIR.
   - Octave-band filter from 250 Hz to 8 kHz.
   - Schroeder integration with Lundeby truncation.
   - Fit single- and two-slope decays (for H1a).
3. **Sound speed, without circularity.**
   - TOF is referenced to the loopback.
   - Fit TOF = r/c + τ₀ over ≥ 4 tape ranges (1, 3, 6, 10 m). The **slope** gives c; the **intercept** τ₀ absorbs the electronic, transducer and group delay.
   - With ±3 cm range error, the relative standard error of c is about 0.44% (computed for ranges {1, 3, 6, 10} m; 0.74% at ±5 cm).
   - Report c_slope ± SE next to Marczak c(T). The two should agree within 2 × SE.
   - The old claim "range verified acoustically" is dropped.
4. **Fitted parameters, on calibration pairs only.**
   - R(θ, f) half-space parameters for walls and floor (8 parameters) and |R_s| ∈ [0.9, 1].
   - Objective: RMS EDC error over 0–t_e plus the amplitude error of the first resolved arrivals.
   - Optimizer: CMA-ES or bounded least squares. Report the posterior or bootstrap spread of each parameter.
   - The tail decay parameters are fitted on calibration pairs only.
5. **Validate** on the validation positions with the §4.4 metrics. Report the same metrics for three comparators:
   - **Competitor baseline:** pyroomacoustics ShoeBox, c set to the measured value, absorption-only walls fitted to calibration T20 via Sabine, no sign flip. This is what a competitor would realistically use.
   - **Uncalibrated signed ISM:** textbook half-space parameters.
   - **Ceiling:** measured-IR replay from the nearest measured position.
6. **Freeze** uwsim v0 at G3 (11-25), using S1 data. Cross-session repeatability is confirmed with S2 data at G4.

### 4.4 Validation metrics and acceptance thresholds

**Principles.**
- Validate only quantities that were **not fitted**.
- **Thresholds come from measured repeatability.** σ_rep is the session-to-session SD at the same position: positions measured in both S1 and S2, plus ≥ 4 sweeps per position within each session.
- ISO 3382 just-noticeable differences describe in-air *human perception* in halls. They are cited only as context, not as tolerances.
- **All thresholds are pre-registered before fitting.** The threshold for metric m is max(2·σ_rep,m, floor_m).

| Metric (per octave 500 Hz–4 kHz unless noted) | Fitted? | Threshold | Notes |
|---|---|---|---|
| Arrival time of the first N = 5 resolved reflections | no | max(2σ_rep, 0.05 ms) | Only at positions where the arrivals are resolvable (0.5–1 m, asymmetric depths) |
| Arrival amplitude of those reflections (dB re direct) | partly (the first arrivals enter the objective at *calibration* positions only) | max(2σ_rep, 1.5 dB) | Tests R(θ) at angles not seen in fitting |
| DRR and C50 at each validation position, and their **variation across positions** (slope vs range) | no | max(2σ_rep, 1 dB); slope within its bootstrap CI | Measured DRR replaces the Sabine R_c |
| EDC curvature: single vs two-slope classification and knee time | no | same model class; knee within 20% | Tests the H1a structure |
| 1/3-octave magnitude spectrum of the IR (coloration), 1–8 kHz | no | RMS distance ≤ max(2σ_rep, 2 dB) | |
| Low-band energy (< 1.5 × cutoff) relative to 500 Hz–1 kHz | no | within max(2σ_rep, 3 dB) | Tests #8 |
| T20 and EDT per octave | the tail is fitted | **descriptive only** | Reported for comparability. Not used for pass/fail, because T20 is nearly position-independent in a single pool and would pass automatically. |
| **Downstream:** unprocessed sim-twin WER vs unprocessed real WER, per held-out benchmark position | no | δ_sim = max(5 points, retest half-width) | Primary criterion for the use the simulator is built for |
| **Sensitivity analysis** (no threshold): ΔWER of the unprocessed sim twin per unit error in T20, DRR and C50 | — | — | Shows which acoustic errors matter for ASR |

Li & Chitre needed reflection coefficients learned from data to reach about 0.38 dB error in a small fiberglass tank ([arXiv 2205.06066](https://arxiv.org/html/2205.06066v2)). That is why the plan fits a physical R(θ) rather than guessing R.

---

## 5. Data

### 5.1 Clean speech

| Corpus | Use | Size | Licence |
|---|---|---|---|
| LibriSpeech train-clean-100 | sim training (phase 1) | 100.6 h, 251 speakers | CC BY 4.0 ([OpenSLR 12](https://www.openslr.org/12/)) |
| LibriSpeech train-clean-360 | sim scaling (A10) | 363.6 h, 921 speakers | CC BY 4.0 |
| LibriSpeech dev-clean | sim validation (2 h); **v0-dev playback** (40 utterances per condition); v1 real-train playback | 5.4 h, 40 speakers | CC BY 4.0 |
| LibriSpeech test-clean | **v0-test** (200 utterances: 5 per speaker × 40) and **v1-test** (≥ 1,000 *different* utterances), plus their sim twins | 2,620 utterances, mean about 7.4 s | CC BY 4.0 |
| **New prompts (v1)** | Contamination control: 100 newly written, previously unpublished sentences, voiced by an open-licence TTS with ≥ 4 voices (TTS choice and licence are decision D13) | about 12 min | CC BY 4.0 (ours) |
| VoiceBank-DEMAND test; WHAM! test (16 kHz) | in-domain forgetting checks (E13) | 824 utterances; WHAM! tt 3,000 mixtures (9.0 h) | VB-DEMAND record CC BY 4.0, but it embeds DEMAND noise (licence listed both as CC BY-SA 3.0 and CC BY 4.0): evaluation only, never redistribute outputs. WHAM! noise CC BY-NC 4.0; the mixtures need licensed WSJ0 (LDC). |

### 5.2 Noise bank (recorded by us, released)

| Recording | Protocol | Duration | Split |
|---|---|---|---|
| T200 **unit A**, in water | throttle 10–100% in 10% steps × hydrophone distances 0.5, 1, 2 m × forward/reverse; 60 s each | 60 min | train |
| T200 **unit B** (different physical unit) | throttle 20, 50, 80% × 1 m | 6 min | test only (held-out noise unit) |
| EMI check (every session) | thruster running, **one array channel terminated by a dummy capacitive load equal to the element capacitance**, plus the thruster in air with the hydrophone in water; 3 throttle steps | 5 min per session | diagnostic |
| Flow noise | jet axis pointed at the array vs at 90° | 5 min | flagged; diagnostic |
| Vehicle self-noise (v1) | array on the sub; 3 throttle patterns | 15 min | test only |
| Pool ambient | pumps on and off; empty vs other users. Reported as a **spectrum in dB re 1 µPa²/Hz**, alongside the chain self-noise in the same units | 30 min | by session |
| Bubbles | air stone or regulator exhaust at 0.5 and 1 m | 10 min | by session |
| Noise-only segments inside the benchmark | 2 min per position × thruster state | about 12 min (v0) | test (E14) |
| DeepShip (public) | ablation only | matched | — |

### 5.3 Generated data

- **RIR bank.** 20,000 RIRs: 10,000 from the calibrated pool (jittered within the posterior) and 10,000 from generic pools. Noise and SNR are drawn on the fly.
- **Frozen sets:**
  - sim-val-2h (dev-clean, fixed seeds);
  - **sim-twin-v0/v1:** each real test utterance simulated at its logged position, with noise matched to its *measured* SNR;
  - sim-test-generic-2h.
- **Measured-IR replay set (C9).** Measured IRs from the calibration and validation positions only, **never the benchmark positions**. Perturbations: time-stretch ±10%, per-band gain ±2 dB, tail resynthesis.
- **Storage.** About 11.5 GB per 100 h frozen copy at 16 kHz and 16 bit (computed).

### 5.4 Splits (strict)

- **Speakers.**
  - Sim-train: train-clean speakers.
  - Sim-val, v0-dev and real-train: dev-clean speakers.
  - Real-test: test-clean speakers.
  - These sets are disjoint.
- **v0** (recorded in December) has two parts:
  - **v0-dev:** 40 dev-clean utterances per condition. Used for every choice: scored channel, band limits [f_lo, f_hi], VAD settings, alignment QC thresholds, WPE taps, decoding settings (A14), and the LoRA rank for any December run.
  - **v0-test:** 200 test-clean utterances per condition. Scored **once**, after the OSF pre-registration is filed and all choices are frozen on v0-dev.
- **v1** (January–February) adds **new positions, new test-clean utterances, new sessions and the new-prompt subset.** Once v1 exists, v0-test is demoted to dev. **Final full-paper numbers come from v1-test**, which no decision has touched.
- **Real fine-tuning** (F6, E12) uses only dev-clean recordings at positions disjoint from the test positions in that analysis.
- **Noise.** Training uses unit A; testing uses unit B and other sessions.
- **No leakage into the simulator.** Calibration uses only calibration pairs. Benchmark-position IRs are used only for validation, never for fitting.

---

## 6. Real benchmark protocol

### 6.1 Hardware list

**Rule: purchases stay off the critical path.** The MVR runs on whatever is in hand by G1.

| Item | Part | Qty | Est. cost | Notes / source |
|---|---|---|---|---|
| Projector, preferred | **Lubell LL916C + AC205C**: 200 Hz–23 kHz, 500 Hz–21 kHz ±10 dB, max 180 dB re 1 µPa @ 1 m at 1 kHz | 1 | **Borrow** from Aquatics if they have one (unverified); otherwise about $1,610 ([Inyo LL916C](https://www.inyopools.com/Products/19400029046138.htm)) + AC205C | [Lubell](https://www.lubell.com/products/lubell-ll916c-ll916h-portable-underwater-speakers-for-lap-synchro-swimming/). Dimensions confirmed (9.25"); operating depth 1–18 m; [SPL curve](https://lubell.com/wp-content/uploads/2024/04/LL916C_SPL.pdf) and [polar plots](https://lubell.com/wp-content/uploads/2024/04/LL916C_polar_plot.pdf). The old …046141 URL is now the AC202 box. |
| Projector, budget | Electro-Voice **UW30**: 100 Hz–10 kHz, 30 W, 8 Ω, max depth 3 m | 1 | about $500 | [datasheet](https://goknight.com/content/documentation/UW30%20EDS.pdf); ≤ 3.0 m depth, 1.2 m recommended; 25 Ω/20 W parallel resistor + 1.5 A fuse (datasheet confirmed); discontinuation still unverified (the EV product page redirects to the homepage). |
| Projector, pilot / MVR fallback | Dayton **DAEX25W-8** on a sealed plate (Fs 1,276 Hz) | 1 | about $30 + plate | [Dayton](https://daytonaudio.com/product/1181/daex25w-8-waterproof-25mm-exciter-10w-8-ohm). Usable band in water unknown; likely ≥ 1 kHz. |
| Amplifier | **Battery-powered by default.** A source level of 160 dB needs only about 2 Vrms on the LL916 (computed from 180 dB at 20 Vrms). Lubell requires an **unbalanced, non-bridged** output; many small class-D boards are bridge-tied-load and do not qualify (check each model; unverified). Crown XLi800 ($349, [Full Compass](https://www.fullcompass.com/prod/223773-crown-xli800-2-channel-300w-at-4-ohm-power-amplifier)) only with GFCI and EH&S sign-off. | 1 | $50–349 | Keeping mains gear away from the pool removes a likely approval blocker |
| Capture | 2 × Daisy Seed (`376636603433` → ch0/1, `376C36573433` → ch2/3), `stream_audio.cpp`, 96 kHz / 24-bit, `whisper_ivc/daisy_stream.py`, laptop on battery | 2 | owned | **Identify the board revision and codec** (AK4556 / WM8731 / PCM3060; [datasheet](https://www.electrokit.com/upload/product/41020/41020659/Daisy_Seed_datasheet_v1.0.5.pdf)) |
| Hydrophones | Team RS26 piezo elements | 4 | owned | Built for the 25–40 kHz pinger band. **Measure capacitance, sensitivity below 1 kHz and any built-in band-pass in E0** before scoring them. |
| Preamp | **Perfboard JFET buffer** (≥ 10 MΩ input, 20/40 dB, ±1.8 V clamp), or Aquarian $70 boards (300 kΩ input) if the elements are ≥ 10 nF | 4 | $50–280 | A 1–10 nF element into 30 kΩ gives a high-pass corner of 5.3 kHz–530 Hz (computed). No custom PCB on the critical path. |
| Reference hydrophone | **G1 requirement: request a calibrated reference covering ≥ 8 kHz** (Reson TC4032 or TC4013, or B&K 8103) from Hopkins Marine Station or CCRMA (availability unverified). Minimum in hand: Aquarian **H2d**, about −165 dB re 1 V/µPa ±4 dB over **20 Hz–4 kHz only** | 1 | $232–280 (H2d) | Without a borrowed reference, every absolute level above 4 kHz is reported as **uncalibrated (relative only)**. See §6.9 for which results need absolute calibration. |
| Playback and loopback | USB interface (2-in/2-out). in1 = amplifier output through a 40 dB attenuator (loopback); in2 = **monitor hydrophone at ≥ 0.5 m, 90° off-axis**. At 8 kHz the LL916 Rayleigh distance is about 0.22 m if a ≈ 11 cm (computed), so 0.3 m on-axis would be too close and would scatter into the beam. | 1 | $100–200 | [Loopback precedent, FA2026](https://dael.euracoustics.org/confs/fa2026/data/articles/000077.pdf) |
| Thruster rig | T200 units A and B with ESC on a frame **mechanically isolated from the array mount**; E-stop; battery supply | 2 | owned | throttle logged |
| Mechanics | **No air-filled tubing near the hydrophones.** Use flooded or perforated PVC, or thin rope with weights. Depth-marked poles; tape; laser meter for deck positions; thermometer (±0.2 °C); salinity/conductivity strip if salt-chlorinated | — | about $150 | Air-filled frames are full-strength pressure-release scatterers next to the sensors (reviewer) |

### 6.2 Safety, approvals and rules

1. **Approvals.**
   - Written approval from Stanford Aquatics.
   - Ask Aquatics in W1 whether **EH&S sign-off** is needed for powered equipment at the pool and for running thrusters (unverified).
   - The default battery-only setup is designed to minimize this.
2. **Exposure: a received-level rule.**
   - In a reverberant pool the diffuse level is about SL + 10·log10(16π/A). For the Avery Competition pool (about 1,174 m² of wall and floor, α = 0.05–0.1 guessed, so A ≈ 59–117 m²) that is **−3.7 to −0.7 dB re the 1 m level beyond R_c ≈ 1.1–1.5 m** (computed; Sabine estimate; K8 says the field is probably not diffuse). Every swimmer is therefore exposed to roughly the source level, wherever they are.
   - Rules:
     - **No swimmers in the water while transmitting.**
     - If anyone must be in the water, the **received level measured by the reference hydrophone at their location** stays ≤ 150 dB re 1 µPa (RMS over speech-active frames). The secondary source for a Navy 160 dB limit now returns HTTP 410; cite no figure until a primary Navy source is found. Our ≤ 150 dB received-level rule and "no swimmers while transmitting" stand as our own conservative limits. The 145 dB figure for sport divers is (unverified).
   - Source level is always reported as **speech-active RMS**, with peak level as well. Speech crest factor is about 12–15 dB.
3. **Pinger-band leakage.**
   - The drive LPF at 8 kHz cannot remove harmonics generated *by the projector*. The 4th harmonic of 6.25–8 kHz content and the 3rd harmonic of anything above 8.33 kHz land in 25–40 kHz.
   - Controls:
     - keep the drive level low;
     - check the 25–40 kHz spectrum on every raw 96 kHz capture;
     - if leakage persists, lower the LPF to 6 kHz *and* reduce drive. The 5th harmonic of 5–6 kHz still reaches 25–30 kHz, so drive level is the real control.
   - E17 (SSB at 8.39–11.09 kHz) puts its 3rd harmonic at 25–33 kHz. It is **offline only and never run at RoboSub.**
4. **Thruster.** Guarded frame, E-stop, nobody within 2 m.

### 6.3 Pool geometry and environment logging

- **Obtain from Aquatics or Facilities in W1:**
  - engineering drawings;
  - bathymetry or depth profile;
  - wall and floor construction (tile on concrete? backfill?);
  - whether the pool is salt-chlorinated.
- **Measure:**
  - depth every 2 m along the centreline and two parallel lines;
  - gutters, lane lines (remove if allowed), inlets and their locations (entrained bubbles).
- **Every session:**
  - water temperature at 0.5 and 1.5 m, start and end;
  - pump state;
  - pool users;
  - operator positions (keep bodies ≥ 2 m from the array during takes);
  - photos.
- **Positions:**
  - (x, y) by laser on deck, depth by marked pole; target ±3 cm;
  - c and τ₀ from the TOF slope (§4.3).

### 6.4 Calibration steps (session S1)

1. **Daisy chain (E0, bench before S1):**
   - noise floor with inputs shorted;
   - 1 kHz sine at a known Vrms → dBFS/V per channel and gain;
   - chain self-noise referred to dB re 1 µPa²/Hz once the sensitivity is known.
2. **RS26 element check (E0, bench or bucket).** Relative sensitivity vs the reference hydrophone from 300 Hz to 2 kHz, side by side.
   - **Rule:** if an element is > 15 dB below the reference-relative expectation at 500 Hz, or its speech-band SNR at the 1 m speech level is < 20 dB, the reference hydrophone becomes the scored channel by default (F5).
3. **Array vs reference (E1).**
   - **Substitution only up to about 2 kHz.** Elements within a few cm, with λ = 0.75 m at 2 kHz (computed); diffuse-field coherence is about λ/2.
   - **Gated free-field comparison above about 3 kHz.** Use the asymmetric-depth geometry and a gate of ≥ 0.7 ms.
   - **Check consistency in the 2–3 kHz overlap.**
   - **Mount directivity:** rotate each element in its mount by 0°, 90° and 180° at 4–8 kHz.
4. **Projector chain (E2).**
   - ESS at 3 drive levels and at **each source depth** used. Below about 400 Hz the radiation impedance near the surface depends on depth (λ/4 ≈ 1 m at 375 Hz, computed).
   - Orientation sweep at **0°, 45° and 90°**.
   - The chain response **h_chain** (projector + box + amplifier → reference) is measured at close range under the same gated/substitution split as step 3. This defines R1 (§8.2).
   - Report the usable band at −10 dB.
5. **Pool IR grid (E3).**
   - **ESS from the projector's lower limit** (about 150–200 Hz for the LL916, about 500 Hz for the DAEX) to 20 kHz, with **level pre-shaping** at low frequency so the projector is not over-driven. 10 s sweeps, **≥ 4 s of silence** after each sweep (3 × T60 must be ≥ 3.6 s if T60 reaches 1.2 s).
   - **Loopback-based drift correction before averaging.** At 50 ppm a 10 s sweep skews by 0.5 ms (computed).
   - Report **sweep-to-sweep coherence and IR variance**. These are the channel-variability baseline that feeds §4.4 and §4.2 #14. If coherence is low at high frequency, use single-sweep IRs instead of averages.
   - The 10·log10(T·B) ≈ 53 dB figure is **processing gain only**. The achieved IR SNR is measured per band.
   - **Extra measurements:** pumps on vs off at one position; a **rig-only test** (frame present vs absent) at one position; repeats at 2 positions within the session.
6. **Gate G2 targets** (IR SNR per band, alignment) are set from the S1 pilot. The old fixed target ("≥ 40 dB at 250 Hz–8 kHz") is dropped.

### 6.5 Playback material and synchronization

- **Blocks** of about 60 s (about 8 utterances):
  `[0.5 s silence][LFM chirp 0.25 s, f_lo–8 kHz][0.5 s][utt1][1.0 s]…[uttN][0.5 s][LFM chirp][2.0 s silence]`. The overhead is about 16% (computed).
- **Thruster in alternating blocks.** Within each position, thruster-off and thruster-on blocks alternate in **ABBA order**. The thruster spins up during the 2 s gap. This removes the confound between thruster state and time of day.
- **Drift anchor.** The first block of each session is repeated at the end.
- **Levels.**
  - Utterances are RMS-normalized to −26 dBFS before the LPF.
  - Gain is fixed per session.
  - Estimated speech-active SL is 150–160 dB re 1 µPa @ 1 m, with no swimmers in the water.
- **Alignment** (`eval/align.py`):
  1. Matched-filter the chirps on every stream.
  2. Estimate drift and resample to the loopback clock.
  3. Fine per-utterance lag search against R1 at 16 kHz, ±20 ms.
  4. Flag any utterance with peak-to-sidelobe ratio < 6 dB.
  - QC thresholds are frozen on v0-dev.
- **Acceptance.** Residual misalignment ≤ 1 sample at 16 kHz at 1 m (checked against the loopback); drift consistent within 2 ppm per board.

### 6.6 Positions grid and conditions

| Set | Session | Positions (range to array centre) | Depths | Thruster | Material | Speech audio per channel |
|---|---|---|---|---|---|---|
| **S1 (calibration)** | W3 | calibration 2 × 3 pairs; validation S3 × R4/R5 (new positions) | asymmetric (e.g. 0.7 / 1.2 m) | noise bank only | ESS; 20-utterance pilot; SNR pilot (thruster received level vs throttle at 3 m) | — |
| **v0 core** | S2, W6 | **1, 3, 6 m** | projector 1 m, array 1 m (mid-depth) | off and on, **alternating ABBA blocks**; throttle T_hi chosen from the S1 pilot so that the speech-band SNR at 3 m is about [0] dB (target fixed after the pilot) | 200 test-clean + 40 dev-clean per condition | test 2.47 h; total playback 3.43 h (computed); about 4.5–5 h with moves and ESS |
| **S3 (backup first)** | W7 | Repair any failed v0 conditions first. Only if none failed: (i) **test–retest** of 50 utterances × 2 positions (1 and 6 m), repeated from S2; (ii) **SNR sweep** at 3 m with throttle T_lo (target SNR about [+10] dB), 200 test-clean; (iii) **0.5 m** point, 100 utterances (R1-dp, DRR) | as v0 | as listed | as listed | about 1 h of playback (computed: 14 + 29 + 14 min) |
| **v1 (full paper)** | ≥ 2 sessions, Jan–Feb | P_test: 0.5, 1, 3, 6, 10, 15 m; P_train: 2, 4.5, 8, 12 m | 2 projector depths; receivers at mid-depth and **near-surface (0.5 m)** | off; ≥ 2 throttles; unit B; vehicle self-noise (one session) | ≥ 1,000 new test-clean + 100 new prompts (TTS) + live talkers if IRB clears; dev-clean P_train 0.5 / 1 / 2 h | test about 3–5 h |
| **Second site** | Feb–Mar | open water (2 positions) or tank (≥ 2 positions) | site-specific | off (+ ambient) | ≥ 20 utterances × 2 positions + ESS | about 0.1 h |

If the pool slot for S2 is shorter than 5 h, the thruster-on blocks of the 6 m position move to S3, and test–retest moves to v1.

### 6.7 Live talkers and listening tests (IRB)

- **One IRB submission in W2** covers both live talkers (v1) and the crowd transcription test (E22, a human-subjects study). It may come back as an exempt determination.
- **The consent form covers:**
  - public release under CC BY 4.0;
  - opt-out from release;
  - speaker IDs only, no names.
- **Live procedure.** Talkers record in a quiet room; the recording is R0. It is then played back through the projector, which keeps the reference exact.

### 6.8 Metadata schema (`bench/metadata.jsonl`, one row per utterance × channel)

```json
{
  "utt_id": "v0_s2_p3_b07_ch0_1089-134686-0000",
  "librispeech_id": "1089-134686-0000", "speaker_id": "1089", "transcript": "...",
  "material": "librispeech|newprompt_tts|live", "corpus_split": "test-clean|dev-clean",
  "bench_split": "v0-dev|v0-test|v1-test|v1-train|site2", "prereg_id": "osf:...",
  "session_id": "2026-12-02_s2", "session_order_idx": 0, "block_id": 7, "block_order": "ABBA", "anchor_repeat": false,
  "pool": {"name": "...", "L_m": 0.0, "W_m": 0.0, "H_profile_m": [], "construction": "tile-on-concrete", "salinity_gkg": 0.0},
  "water_temp_C": [26.8, 26.9], "c_marczak_mps": 1501.0, "c_slope_mps": 1500.2, "c_slope_se_mps": 6.6, "tau0_ms": 0.0,
  "position_id": "p3", "src_xyz_m": [0, 0, 0], "src_orient_deg": 0, "rx_xyz_m": {"ch0": [0, 0, 0]},
  "range_tape_m": 6.00,
  "thruster": {"state": "off|on", "unit": "A|B", "throttle_pct": 0, "xyz_m": [], "jet_axis_deg": 90, "drive": "trapezoidal|foc", "flow_noise_flag": false},
  "pool_activity": "empty|swimmers", "pumps": "on|off", "emi_check_id": "emi_2026-12-02",
  "projector": "LL916C+AC205C", "amp": "...", "playback_gain_dB": -12.0,
  "SL_speechactive_rms_dB_re1uPa_1m": 155.0, "SL_peak_dB": 169.0, "SL_abs_calibrated": "le4kHz|full|relative",
  "daisy_board": "376636603433", "daisy_rev": "rev7/PCM3060", "preamp_gain_dB": 40, "channel": 0, "scored_channel": true,
  "fs_raw": 96000, "bits": 24, "raw_file": "raw/ch0/12.wav", "start_sample_raw": 0, "end_sample_raw": 0,
  "drift_ppm": 0.0, "fine_lag_samples_16k": 0, "chirp_psr_dB": 0.0, "ir_id": "ess_p3_s2_a",
  "acoustic": {"snr_speechband_dB": 0.0, "snr_octave_dB": {"500": 0.0, "1000": 0.0, "2000": 0.0, "4000": 0.0},
               "drr_dB": null, "c50_dB": 0.0, "edt_s": {}, "t20_s": {}, "two_slope": false},
  "flags": ["clip", "dropout", "align_low_psr", "pinger_band_energy"], "notes": ""
}
```

DRR is `null` where the direct path cannot be gated (§8.2). The Croissant and Responsible-AI metadata are generated from this file ([E&D call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)).

### 6.9 Uncertainty budget and traceability (Table I of the paper)

| Quantity | Source of uncertainty | Expected magnitude | How obtained |
|---|---|---|---|
| Reference sensitivity ≤ 4 kHz | H2d specification | ±4 dB (spec), or the borrowed reference's certificate | E1 |
| Reference sensitivity 4–8 kHz | unknown with the H2d alone | **uncalibrated** unless borrowed | G1 request |
| Array FIRs | substitution and gate consistency in the 2–3 kHz overlap | [placeholder] dB per band | E1 |
| Chain response h_chain / TVR | gate length, near field, depth dependence | [placeholder] dB per band | E2 |
| Source level | sensitivity + TVR + drive | combined [placeholder] dB | E2 |
| Positions | tape, laser, pole | ±3 cm | §6.3 |
| c | slope SE; temperature ±0.2 °C → ±0.5 m/s (computed, dc/dT ≈ 2.6 m/s/°C at 26 °C) | about 0.44% (slope SE) | §4.3 |
| IR metrics | sweep-to-sweep and session-to-session | σ_rep per metric | E3, E21 |

**Results that need absolute calibration:** source level (safety and reporting) and the absolute noise-bank spectra in dB re 1 µPa²/Hz.

**Results that need only relative responses:**
- SE training and testing;
- R1;
- simulator SNR, provided speech and noise are measured through the same chain;
- all WER and ESTOI results.

Absolute calibration is needed only to transfer levels to other hardware.

### 6.10 Release

- **Contents:**
  - raw 96 kHz 4-channel WAVs;
  - aligned 16 kHz cuts;
  - R0, R1 and R1-dp references;
  - every ESS (raw and deconvolved) with sweep-to-sweep statistics;
  - h_chain, the THD data and the hydrophone FIRs;
  - the noise bank;
  - the uncertainty budget;
  - metadata, alignment code, uwsim code and configs, adapters, evaluation scripts;
  - the pre-registration.
- **Licences.** Audio and IRs under CC BY 4.0; code under Apache-2.0.
- **Hosting.** Zenodo DOI plus a Hugging Face mirror at camera-ready (2027-05-24).

---

## 7. Models

### 7.1 Model and system list

| Phase | Role | System | Checkpoint / toolkit | fs |
|---|---|---|---|---|
| 1 (FT, primary) | deep filtering | **DeepFilterNet3** | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | 48 kHz |
| 1 (FT) | masking (metric-GAN) | **MetricGAN+** | `speechbrain/metricgan-plus-voicebank` (in `whisper_ivc`) | 16 kHz |
| 1 (zero-shot only) | time-domain transformer | SepFormer-WHAM16k | `speechbrain/sepformer-wham16k-enhancement` (in `whisper_ivc`) | 16 kHz |
| 1 (zero-shot, **reverb-trained**) | denoising + dereverberation | **SepFormer-WHAMR! enhancement** | [`speechbrain/sepformer-whamr-enhancement`](https://huggingface.co/speechbrain/sepformer-whamr-enhancement): **8 kHz model** (WHAMR! SI-SNR 10.59 dB, PESQ 2.84 per the model card), so its output is limited to ≤ 4 kHz. Scored with a band caveat and also compared within 0–4 kHz. | 8 kHz |
| 1 (zero-shot) | universal | URGENT TF-GridNet baseline (8.5M parameters, [arXiv 2505.23212](https://arxiv.org/abs/2505.23212)) | `kohei0209/tfgridnet_urgent25` (CC BY 4.0; ESPnet `SeparateSpeech.from_pretrained`); include in C1 | 16 kHz+ |
| 1 (classical) | dereverberation | **WPE**, 1-channel and 4-channel | `nara_wpe` (fgnt, MIT, `pip install nara_wpe`; use the NumPy implementation). Starting point: STFT 512/128 at 16 kHz, delay 3, taps 20–40 (prediction length 160–320 ms, scaled toward a pool T60 of 0.25–1 s), 3 iterations. Taps tuned on v0-dev. | 16 kHz |
| 1 (ASR side) | adapt the recognizer | **Whisper large-v3 + LoRA on uwsim audio** | `openai-whisper` / HF `transformers` + `peft` | 16 kHz |
| 2 | complex masking | FRCRN | ClearerVoice `FRCRN_SE_16K` ([arXiv 2506.19398](https://arxiv.org/abs/2506.19398)) | 16 kHz |
| 2 | metric-GAN | MP-SENet (used by Jiang 2024); CMGAN | official VB-DEMAND checkpoints | 16 kHz |
| 2 | generative | SGMSE+ | sp-uhh/sgmse; VB-DEMAND checkpoint and a WSJ0-REVERB dereverberation checkpoint (both confirmed on the README; WSJ0-REVERB recommends `--N 50 --snr 0.33`; prefer the VB-DMD checkpoint for anything released, because of WSJ0/EARS lineage) | 16 kHz |
| 2 | time-domain | fine-tuned SepFormer replaces Demucs (facebookresearch/denoiser is CC-BY-NC 4.0 and archived; at most an NC-labelled zero-shot baseline) | SpeechBrain | 16 kHz |
| ext | multichannel neural | FaSNet-TAC, MVDR + best single-channel | ESPnet-SE++ | 16 kHz |

**Claim wording.** "N architectures from K families". Never "any SE model".

### 7.2 Per-model fine-tuning recipe

| Model | Loss (native where possible) | Trainable in full FT | LoRA insertion | Inference cost | Forgetting test (in-domain) |
|---|---|---|---|---|---|
| DFN3 | Its own training losses (multi-resolution spectral + DF/ERB terms as in the repo; exact terms to be read from the repo) against R1_sim | all | `peft` LoRA on Linear layers. The GRUs get a custom low-rank wrapper on the input and hidden matrices, or are left frozen (both variants logged). | low | VB-DEMAND (reported in the DFN3 paper; not in-domain) |
| MetricGAN+ | **Native MetricGAN+ loop**: the discriminator learns normalized PESQ(output, R1_sim) and the generator maximizes the predicted score. PESQ per step runs on the CPU; cost is measured in the W4 smoke test. | generator + discriminator | LoRA on the generator's Linear layers; BLSTM frozen | low | VB-DEMAND |
| Whisper large-v3 (C8) | cross-entropy on LibriSpeech transcripts of uwsim audio | — | LoRA r = 8, α = 32, dropout 0.05 on q_proj/v_proj (Rong's setting, [PDF](https://ddfe.curtin.edu.au/yurong/Oceans25.pdf)) | ASR only | clean test-clean WER |
| SepFormer (phase 2) | SI-SNR (native) vs R1_sim | all | `peft` on the transformer Linear layers (q, k, v, o, FFN) | moderate (26M (Subakan et al., arXiv 2010.13154)) | WHAM! 16k test |
| SGMSE+ (phase 2) | denoising score matching (native) | all | Linear layers of the score network | **high**: reverse sampling with N steps (default N = 30); report RTF | VB-DEMAND |
| FRCRN, MP-SENet, CMGAN (phase 2) | native losses | all | Linear/attention via `peft`; conv layers via a custom wrapper (peft supports Conv1d/2d/3d; GRU needs a custom wrapper) | moderate | VB-DEMAND |

### 7.3 Conditions

| ID | Condition | Data | Purpose | Tier |
|---|---|---|---|---|
| C0 | Unprocessed (scored channel) | — | floor | MVR |
| C1 | Zero-shot: MetricGAN+, SepFormer-wham16k, DFN3, SepFormer-WHAMR!, WPE-1ch/4ch, (TF-GridNet) | — | RQ2a; strong baselines | MVR |
| C2 | Sim full FT | uwsim v0 (frozen at G3) | RQ2 primary | MVR (DFN3 × 3 seeds, MetricGAN+ × 1) |
| C3 | Sim LoRA r ∈ {4, 8, 16} | as C2 | RQ6 | extension |
| C4 | Sim → real FT | C2 + dev-clean at P_train {0.5, 1, 2} h | RQ5 | extension |
| C5 | Real-only FT | P_train {0.5, 1, 2} h | RQ5 | extension |
| C6 | Sim + RemixIT | C2 + unlabeled pool audio | RQ6 | extension |
| C7 | **In-air control** | same recipe, pyroomacoustics in-air RIRs (c = 343) with T60 matched to the pool distribution, same noise | RQ2b | **MVR (DFN3 × 1)** |
| C8 | **Whisper-LoRA on uwsim**, no SE | uwsim 50 h | RQ2d | **MVR (× 1)** |
| C9 | **Measured-IR replay** FT | §5.3 replay set | RQ2c; F3 insurance | **MVR (DFN3 × 1)**, started in W4 |
| C10 | **Kiel-recipe U-Net on uwsim** (Kiel DAGA 2026: 4-level U-Net, IRM target, 6 kHz, log-magnitude, published hyperparameters), trained from scratch | uwsim v0 (frozen at G3) | RQ2 strong baseline; direct row for the closest prior work | MVR (× 1) if compute allows, else full paper |
| Oracle | Whisper on R1 (chain coloration only) and on R0 | — | headroom per distance | MVR |

### 7.4 Hyperparameter starting points and compute

| Setting | Full FT | LoRA | Real FT |
|---|---|---|---|
| Optimizer | AdamW (0.9, 0.999), wd 1e-4 | AdamW, wd 0 | AdamW |
| Peak LR | 1e-4 (DFN3, MetricGAN+ generator, FRCRN); 3e-5 (SepFormer, TF-GridNet); 5e-5 (SGMSE+) | 1e-3 (r = 8; Rong's Whisper precedent); Luo used r = 4, lr 2e-4 | 0.3 × the C2 LR |
| Schedule | 1k warm-up, cosine | same | 200 warm-up, cosine |
| Steps | ≤ 50k at 64 s of audio per step (about 890 h seen, computed); early stop on sim-val, patience 5 × 2k | ≤ 30k | ≤ 1k–3k; early stop on a real-dev split carved from P_train |
| Gradient clip | 5.0 | 5.0 | 5.0 |
| Seeds | DFN3: 3 in the MVR; others: 1 in the MVR, 3 in the paper | 3 | 3 |

- **Compute.** The earlier "6–12 GPU-h per 50k steps" is withdrawn as optimistic. **Throughput is measured in the W4 smoke test** and steps are capped to fit the budget.
- **Planning estimate for the MVR** (24 GB-class GPU, not measured):
  - DFN3: 5 runs × 10–25 GPU-h;
  - MetricGAN+: 10–30 GPU-h;
  - Whisper-LoRA: 15–30 GPU-h;
  - inference: about 10 GPU-h.
  - **Total about 85–195 GPU-h. Secure ≥ 200 GPU-h by 11-06 (D6).**

---

## 8. Evaluation

### 8.1 Metrics

| Tier | Metric | On | Notes |
|---|---|---|---|
| **Primary** | Whisper large-v3 corpus WER (S, D and I reported) | v0-test (abstract), v1-test (paper) | §8.3 |
| **Primary (intelligibility)** | ESTOI vs R1, band-limited to [f_lo, 8 kHz] | same | |
| **Anchor (full paper)** | **Human word-recognition accuracy** (E22) | 60 utterances × 4 systems × 2 conditions | anchors RQ8 |
| Secondary | Parakeet-TDT-0.6B-v2 WER (model card: 1.69% on test-clean, [HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)) | same | URGENT and Milo **did not use** Whisper |
| Secondary | PESQ-WB vs R1 and R0 (flagged as unreliable on band-limited, coloured and generative output); STOI; CI-SDR (512 taps = 32 ms, absorbs coloration but **not** the 0.25–1 s tail; [arXiv 2011.15003](https://arxiv.org/abs/2011.15003)) | same | |
| Secondary | UTMOS, DNSMOS P.835, NISQA, SQUIM | same | never reported alone ([MOS-Bench](https://arxiv.org/abs/2411.03715)) |
| Covariates | per-take measured SNR (speech band and octaves), C50, DRR (where gateable), T20, EDT | all takes | RQ9 |
| Diagnostic | insertion rate; hallucination rate on noise-only segments | E14 | |
| System | RTF and latency on the Orin | E20 | descriptive |

### 8.2 References for intrusive metrics on real data

- **R1 (primary): position-independent chain reference.** R1 = s ∗ h_chain,ch.
  - h_chain,ch is the projector → (that channel's hydrophone) transfer function, measured **once** at close range per source depth under the gated/substitution protocol (§6.4), and band-limited to [f_lo, 8 kHz].
  - **Per utterance, only a broadband delay and gain are fitted** (RealMAN-style, [arXiv 2406.19959](https://arxiv.org/abs/2406.19959)).
  - It contains projector and hydrophone coloration and **no pool path**. The **surface-image dipole (Lloyd's mirror) and all multipath count as channel** and are to be removed.
  - **The uwsim training target is defined identically** (R1_sim = s ∗ h_chain).
  - The abstract says "chain reference", **not** "measured direct-path response".
- **Why not a gated direct path at range.** With both ends at 1 m depth in 2 m of water, the first image arrives this much after the direct path (computed):
  - 1.04 ms at 0.5 m, so a gate is valid above about 2–3 kHz;
  - 0.82 ms at 1 m (above about 2.4–3.6 kHz);
  - 0.40 ms at 3 m (above about 5–7 kHz);
  - 0.22 ms at 6 m;
  - 0.13 ms at 10 m (above about 15–23 kHz).

  From 3 m outward there is no gateable direct path in the speech band.
- **R1-dp (diagnostic).** The position-specific gated direct path, **only at 0.5–1 m and only above about 2.5 kHz**. DRR and C50 are scored only where it exists.
- **R0 = s**, band-limited. This is the end-to-end target and is reported alongside R1.
- **R_mon.** The monitor hydrophone (≥ 0.5 m, 90° off-axis), used only to quantify projector nonlinearity.
- **Scoring rate.** Everything is scored at 16 kHz.

### 8.3 Whisper WER protocol

- **Model.** `openai-whisper` large-v3, pinned version and weights hash, fp16.
- **Decoding.** `language="en"`, `task="transcribe"`, `temperature=0`, `beam_size=5`, `condition_on_previous_text=False`, no prompt. Temperature fallback disabled and disclosed; the default thresholds (compression ratio 2.4, logprob −1.0, no-speech 0.6) are confirmed in `transcribe.py`; the CLI default is beam_size = best_of = 5 and the Python API default is greedy. Decoding ablation A14 is decided on v0-dev.
- **Level normalization (fixes the peak-normalization bug).**
  - Every cut is **RMS-normalized to −26 dBFS over the speech-active region**, where speech activity is determined by VAD on the *aligned R0*, not on the system output.
  - Noise-only segments receive **the same per-system gain as the median of that system's speech utterances at the same position and condition.**
  - Silence and residual noise are never amplified to full scale.
  - Padding is 0.25 s either side. The same cut feeds every system.
- **Text.** `whisper.normalizers.EnglishTextNormalizer` on both sides ([Whisper paper](https://arxiv.org/abs/2212.04356)). Corpus WER via jiwer.
- **Hallucination flags.** Hypothesis/reference length > 2 or compression ratio > 2.4. WER is reported with and without flagged utterances ([Barański et al.](https://arxiv.org/abs/2501.11378)).
- **Floors and oracles.**
  - Clean digital source (Whisper paper: 2.7% for large/large-v2, greedy; no primary large-v3 figure found; measured ourselves).
  - Whisper on R1 (chain coloration only).
  - The 0.5 m real recording (S3/v1).
- **Contamination.** Whisper may have seen LibriSpeech, and its language-model prior can mask acoustic degradation. Subtracting the clean floor does not fix this. **v1 reports WER separately on the new-prompt subset.**

### 8.4 Human anchor (E22, full paper; IRB required)

- Crowd transcription (e.g. Prolific) of 60 utterances × 4 systems (C0, best C1, DFN3-C2, C8) × 2 conditions (1 m off, 6 m on).
- Each listener hears each sentence at most once (Latin square).
- 3 transcriptions per clip; score human WER with the same normalizer.
- Cost estimate $200–400 (reviewer estimate, unverified). The pool should be headphone-screened.
- If IRB approval is late, E22 moves to the extension and RQ8 is reported as proxy-only, labelled as such.

### 8.5 Statistics

- **Pre-registration (OSF), filed by 12-06, before v0-test is scored.** It contains:
  - the primary endpoint (§1);
  - the confirmatory hypotheses H1a, H1b, H2-primary, H2a–c and H9a/b with their decision rules;
  - δ_WER and δ_ESTOI (the retest-based parts are filled in by formula once E21 exists);
  - the §4.4 thresholds;
  - the analysis code hash.
  - **Everything else is labelled exploratory.**
- **Unit of analysis.**
  - **v0:** one core session and 3 positions. A cluster bootstrap over **speakers within position × thruster strata** (10,000 replicates) gives CIs **valid only for these positions in this session**. The paper says so.
  - **Session and position variance** is quantified by test–retest (E21: ≥ 50 utterances × 2 positions in a later session). The test–retest WER and ESTOI spread is the **noise floor**: no claimed difference smaller than it is interpreted.
  - **v1** (≥ 6 test positions, ≥ 2 sessions): two-level bootstrap over position:session × speaker, plus the GLMM of RQ9 with random effects.
- **Tests.**
  - **Paired differences:** paired cluster bootstrap with the same resampled clusters for both systems.
  - **Nulls:** TOST equivalence or non-inferiority with the pre-registered margins.
  - **RQ3:** CI of G_a − G_b (overlap is never used as a test).
  - **Multiplicity:** Holm across the model × condition grid of exploratory tests.
  - **WER significance:** matched-pairs (MAPSSWE) as a cross-check (Bisani & Ney, ICASSP 2004, doi:10.1109/ICASSP.2004.1326009; speaker clustering justified by Liu & Peng, arXiv 1912.09508).
- **Seeds.** DFN3 runs 3 seeds in the MVR. The primary endpoint must hold for each seed; the seed SD is reported. **No architecture-dependence claim is made from single-seed runs.**
- **Correlations (RQ8).** Kendall τ with a bootstrap CI clustered by system; no τ thresholds below 6 systems.
- **Power.**
  - The old SDs (ESTOI 0.05, WER 15 points) were guesses. **Re-estimate from v0-dev** (40 utterances × 6 conditions) and the S1 pilot. The 20-utterance pilot alone is too small.
  - Inflation for position clustering: 1 + (m − 1)·ICC.
  - If the re-estimated n per condition exceeds 200, the abstract pools over positions for the primary endpoint (already pre-registered) and reports per-position results as descriptive.

---

## 9. Experiments

Compute figures are estimates, not measurements.

| E | Purpose | Conditions | Outputs | RQ | Paper item | Effort / compute |
|---|---|---|---|---|---|---|
| E0 | Rig bring-up | Daisy revision; element capacitance; **RS26 sensitivity below 1 kHz**; preamp; chain self-noise (dB re 1 µPa²/Hz once calibrated) | dBFS/V, noise floor, HPF corner, scored-channel decision | — | Table I | bench, 2–3 days |
| E1 | Hydrophone calibration | substitution ≤ 2 kHz + gated ≥ 3 kHz + overlap check; mount directivity | h_hyd,ch, uncertainty per band | RQ1 input | Table I | 1 pool hour |
| E2 | Projector chain | ESS × 3 levels × depths × 0/45/90°; THD | h_chain (R1), usable band, directivity, harmonic IRs | RQ1 input | Fig. 2 (inset) | 1.5 pool hours |
| E3 | Pool channel | calibration and validation grids (asymmetric depths); repeats; pumps on/off; rig-only; TOF-slope c | EDC, single vs two-slope fits, DRR/C50 vs range, sweep coherence, σ_rep | RQ1 (H1a) | Fig. 2, Table I | 2.5 pool hours; 3 days of analysis |
| E4 | Noise bank | units A and B × throttles × distances; EMI dummy; flow noise; ambient spectra | released bank; spectra in dB re 1 µPa²/Hz | input | Fig. 2 (panel) | 1.5 pool hours |
| E5 | Simulator calibration and validation | fit on calibration positions; validate on new positions; vs pyroomacoustics competitor, uncalibrated ISM, replay ceiling | §4.4 table; pass/fail; sensitivity curve | RQ1 (H1b) | Fig. 3, Table III | CPU, < 1 day |
| E6 | Benchmark v0 | §6.6 S2 (+ S3) | aligned v0-dev and v0-test + metadata | all | Table I | 1–2 pool sessions |
| E7 | Baselines | C0; C1 (all zero-shot incl. WHAMR!, WPE 1ch/4ch, TF-GridNet `kohei0209/tfgridnet_urgent25`, Kiel-recipe U-Net C10); oracles | WER and ESTOI per condition | RQ2a | Table II | about 5 GPU-h |
| E8 | Fine-tuning | C2 (DFN3 × 3 seeds, MetricGAN+ × 1), C7 (DFN3), C8 (Whisper-LoRA), C9 (DFN3 replay), **C10** (Kiel-recipe U-Net; 1 seed in the MVR if compute allows, else full paper; about 1 day of implementation, training cheap (estimate)) | primary endpoint; H2b–d | RQ2 | Table II | about 85–195 GPU-h (§7.4) |
| E9 | Sim-to-real gap | sim twin vs real, ≥ 6 models × 3 seeds | G_a − G_b CIs | RQ3 | ext. | inference + phase-2 FT |
| E10 | Simulator ablations | A1–A7 | ΔWER (TOST for nulls) | RQ4 | ext. | 150–450 GPU-h |
| E11 | Channel-fidelity ladder | competitor (pyroomacoustics) / ISM / ISM + R(θ) / + tail / replay / C7 | ΔWER | RQ4 | ext. (C7 and C9 already in E8) | about 60 GPU-h |
| E12 | Real-data efficiency | C4, C5 × {0.5, 1, 2} h | WER vs real hours | RQ5 | ext. | 60–120 GPU-h |
| E13 | PEFT vs full FT vs RemixIT | C2, C3, C6; in-domain forgetting | transfer vs forgetting | RQ6 | ext. | about 150 GPU-h |
| E14 | Hallucination | C0, C1, C2 on noise-only and silence; fixed normalization | insertion and hallucination rates | RQ7 | ext. | inference |
| E15 | Metric validity | all systems × conditions vs human WER | τ with CIs | RQ8 | ext. (partly paper) | inference |
| E16 | **Performance model and usable region** | GLMM on per-utterance errors; LOPO; second site | predicted vs observed WER; usable (SNR, C50) region per system | **RQ9** | **Fig. 4** | analysis |
| E17 | Transmission modes | baseband vs STANAG-style SSB (8.39–11.09 kHz); chain characterized to about 12 kHz; offline only | WER and ESTOI per mode | ext. | journal | 1 pool session |
| E18 | Multichannel neural | FaSNet-TAC, MVDR | WER vs 1-channel and WPE-4ch | ext. | journal | about 50 GPU-h |
| E19 | Ocean transfer | (i) apply the E16 model to published channel statistics, labelled as extrapolation; (ii) replay through the UWA Channel Library ([arXiv 2609.03207](https://arxiv.org/abs/2609.03207)) **only if a channel covers ≤ 8 kHz**. Watermark 10–18 kHz channels would need frequency translation and are excluded (K14). | predicted WER | RQ9 ext. | journal | analysis |
| E20 | Latency on the Orin | SE × Whisper base/small/large-v3 | RTF and latency per 4 s utterance | descriptive | one sentence or small table | 1 day |
| E21 | **Test–retest** | same positions and conditions repeated in a later session (S3; v1) | WER, ESTOI and IR-metric spread → noise floor and σ_rep | RQ1, RQ2 | Table III, §VII | 15 min of pool time + analysis |
| E22 | **Human anchor** | §8.4 | human WER | RQ8 | full paper (one row/figure) | about $200–400 (unverified) |
| E23 | **Second site** | open water (preferred) or tank | E16 transfer test; channel descriptors | RQ9 | Fig. 4 | 1 session |
| E24 | **Benchmark v1** | §6.6 v1 | final test numbers; release | RQ2, RQ9 | Table II (final) | ≥ 2 sessions |

---

## 10. Ablations (extension unless marked)

| ID | Ablation | Tests |
|---|---|---|
| A1 | Anechoic channel | value of reflections |
| A2 | Surface R_s = +1 vs −1 | pressure-release physics (training ablation only) |
| A3 | No projector FIR / no nonlinearity / both | transducer modelling |
| A4 | No hydrophone FIR | expected null (TOST) |
| A5 | Thruster unit A → DeepShip → white noise | value of measured self-noise |
| A6 | No time variation | expected null (TOST) |
| A7 | Domain randomization off vs on | robustness to an unseen pool (second site) |
| A8 | Competitor (pyroomacoustics) vs ISM + real R vs ISM + R(θ) vs + tail vs replay vs in-air (C7) | fidelity ladder. **C7 and replay are in the MVR.** |
| A9 | Target R1 vs R0 | effect of the target definition |
| A10 | Sim data 10 / 50 / 200 h | scaling ([*Less is More*](https://arxiv.org/abs/2506.23859)) |
| A11 | LoRA rank and insertion points | PEFT design |
| A12 | Scored channel: ch0 vs reference vs 4-channel average | **decided on v0-dev** |
| A13 | DFN3 at 16 vs 48 kHz | resampling |
| A14 | Whisper beam vs greedy; fallback on vs off | **decided on v0-dev**; robustness reported |
| A15 | Single-slope vs two-slope tail; measured low-band splice vs modal sum | effect of the H1a-driven modelling on WER |
| A16 | Angle-dependent R(θ) vs per-octave real R | the reviewer's waveguide argument, tested downstream |

---

## 11. Deliverables by milestone

| Milestone | Date | Must contain | Experiments |
|---|---|---|---|
| **MVR: OCEANS abstract** | **2027-01-15** (submit 01-14) | Rig figure; measured pool channel (EDC, single vs two-slope, DRR/C50 vs range) and sim-vs-measured validation on held-out positions with repeatability-based thresholds; **one table**: C0, zero-shot baselines (incl. WPE and WHAMR!), DFN3-C2 (3 seeds), MetricGAN+-C2, C7, C8, C9, oracle, on v0-test with CIs; descriptive WER vs measured SNR/C50; the release commitment. **No architecture-gap claim.** | E0–E8, E16 (descriptive), E21 (if S3 is not used as backup) |
| MVR fallback (sim-FT not ready or H2 inconclusive at G5) | 2027-01-15 | Measured channel physics (H1a), simulator validation, benchmark v0 statistics, baselines incl. WPE and Whisper on R1 vs distance and measured SNR, WER vs SNR/C50. Fine-tuning stated as planned work; no invented numbers. | E0–E7, E16 |
| **Full OCEANS paper** | **2027-05-24** | RQ1, RQ2 and RQ9 only. Final numbers on **v1-test**; test–retest floor; performance model validated by LOPO and **second site**; human anchor (if IRB); uncertainty budget; release of v0 + v1. Latency in one sentence. | E5 (cross-session), E8 (on v1), E16, E20, E21, E22, E23, E24 |
| **JOE / ICASSP 2028 / NeurIPS E&D extension** | ICASSP about Sep 2027 (inferred); JOE rolling | RQ3–RQ8: phase-2 models, gap per architecture, simulator ablations, real-data efficiency, PEFT/forgetting, hallucination, metric validity, multichannel, transmission modes, ocean transfer | E9–E15, E17–E19 (+ A1–A16). Keep overlap with OCEANS ≤ 30% ([UT2025 JOE CfP](https://ieeeoes.org/wp-content/uploads/2025/05/UT2025_CfP_FinalV3.pdf), used as a working target); disclose and cite. |

**Interspeech 2027** (deadline 02-09, [CfP](https://interspeech2027.org/calls/call_for_papers/)): recommendation is **skip**. The same results would be under OCEANS review, which counts as concurrent submission, and there is no time for a distinct ML contribution. Final call by 01-03 (D4).

---

## 12. Week-by-week schedule (from 2026-10-27)

**Roles**

| Role | Owner | Responsibilities |
|---|---|---|
| Lead | Ryota | ML, evaluation, statistics, pre-registration, writing |
| SIM | Kaito, Keiya | uwsim, ESS tooling, IR analysis |
| SPK | Keiya, Kaito | projector, amplifier, safety, chain measurements |
| HW | TBD | preamp, Daisy, calibration |
| OPS | TBD | pool booking, logging, QC |
| ML2 | TBD | adapters, training runs |

**HW, OPS and ML2 must be named by 11-01.** Otherwise F13 applies.

**Calendar** (approximate, unverified): Thanksgiving 11-26; finals early to mid December; winter closure about 12-19 → 01-03. IRB office closed 12-21 → 01-01; IRB submission deadline for December review 11-03. **All pool work must finish by 12-18.**

| Wk | Dates | Lead | SIM | SPK | HW / OPS / ML2 | Checkpoint |
|---|---|---|---|---|---|---|
| W1 | 10-27 → 11-01 | Scope, credit, author order; **name the TBD owners**; EventsAir account; compute request (≥ 200 GPU-h); apply K1–K22; draft the OSF pre-registration | ESS generator with loopback drift correction; Schroeder/Lundeby plus two-slope fit | Aquatics: 3 sessions (S1 W3, S2 W6, S3 W7), approval, **EH&S question**, borrow an LL916, **pool drawings and salinity**; ask Hopkins/CCRMA for a calibrated reference | HW: Daisy revision, element capacitance; OPS: logging sheet | — |
| W2 | 11-02 → 11-08 | Eval pipeline (alignment, RMS-active normalization, Whisper, jiwer, ESTOI, CI-SDR, cluster bootstrap, TOST); Whisper floor; **IRB submission** (live talkers + crowd transcription) | uwsim v0: signed ISM + **R(θ) half-space** + tilted floor + low-band splice/modal sum; analytic unit tests | Projector in hand (borrowed or owned); battery amplifier; **flooded or rope frames**; thruster mount isolated from the array | HW: perfboard preamp or Aquarian boards; **E0 incl. RS26 low-frequency sensitivity**; ML2: adapters + §7.2 recipes; WPE wrapper | **G1 (11-06):** a projector in hand (any of D1); ≥ 2 sessions booked; Daisy revision known; preamp corner < 100 Hz; scored-channel provisional decision; calibrated-reference request answered (yes/no). *No-go → F1/F2/F16.* |
| W3 | 11-09 → 11-15 | Zero-shot runs on the pilot | **S1:** E1, E2 (depths, orientations), E3 (calibration + validation grids, repeats, pumps, rig-only), E4 (units A/B, EMI dummy, flow), 20-utterance pilot, **SNR pilot** | Run S1; safety checklist | OPS: logging | — |
| W4 | 11-16 → 11-22 | Pilot alignment end-to-end; choose T_hi/T_lo from the SNR pilot; fix v0 playlists | IR analysis; H1a fits; σ_rep (within session); uwsim fit | Usable-band report; THD | ML2: **smoke tests on the uncalibrated sim → measured throughput, compute budget**; **start C9 replay training** (F3 insurance) | **G2 (11-20):** per-band IR SNR targets (set from the pilot) met; pilot aligned ≤ 1 sample; usable band known. *No-go → repeat ESS with single-sweep IRs or adjusted levels; F4.* |
| W5 | 11-23 → 11-29 | **Launch C2 (DFN3 × 3, MetricGAN+), C7, C8** after G3; launch C10 alongside | Validation on new positions (§4.4); freeze uwsim v0; generate sim twins at the *planned* v0 positions | — | OPS: metadata and QC tooling | **G3 (11-25):** H1b pass rule on S1 validation positions. *No-go → F3: C9 becomes the training condition; report the simulator errors.* (Thanksgiving) |
| W6 | 11-30 → 12-06 | Score **v0-dev only**; freeze all choices (channel, band, VAD, WPE taps, decoding); **file the OSF pre-registration by 12-06** | **S2:** ESS at the benchmark positions + repeats of S1 positions (cross-session σ_rep) | **S2: v0 core** (3 positions × off/on ABBA, test + dev, drift anchor, EMI check) | OPS: on-site logging; HW: channel health | — |
| W7 | 12-07 → 12-13 | Align and QC v0; E7 on dev | Sim-twin regeneration at the logged positions | **S3: backup first**, else E21 retest + T_lo SNR sweep + 0.5 m | ML2: finish the C2/C7/C8/C9 runs | (finals, approximate) |
| W8 | 12-14 → 12-20 | **Score v0-test once (12-15)**; primary endpoint; H2a–d; E16 descriptive; retest floor | Fig. 3 final; cross-session σ_rep update | — | — | **G4 (12-18):** ≥ 85% of v0 playbacks usable; primary endpoint computed; all pool work done. *No-go → MVR fallback.* |
| W9 | 12-21 → 12-27 | Buffer: reruns, figures | — | — | — | holiday |
| W10 | 12-28 → 01-03 | Abstract draft v1 (§14.2) | Physics text review | Rig text review | — | **D4 (01-03):** Interspeech go/no-go |
| W11 | 01-04 → 01-10 | Internal review | — | — | — | **G5 (01-08):** numbers frozen; MVR or fallback framing chosen |
| W12 | 01-11 → 01-15 | Final edit; **submit 01-14**, regular track + SPC (student lead author) | — | — | — | **Submit** |

**After the abstract**

| Window | Work |
|---|---|
| 01-18 → 02-28 | **Benchmark v1** (≥ 2 sessions: new positions and utterances, new-prompt TTS subset, near-surface receiver, ≥ 2 throttles, unit B, vehicle self-noise); live talkers if IRB clears; **E22 human test** (if IRB); v1 sim twins |
| 03-01 → 03-31 | Notification. **E23 second site** (open water preferred). E16 LOPO and second-site validation; E8 on v1-test. |
| 04-01 → 04-30 | Extension experiments start (E9–E13, kept out of the OCEANS paper); release packaging. **OCEANS experiments frozen 04-30.** |
| 05-01 → 05-24 | Write the full paper; release v0 + v1; camera-ready **05-24** |
| 06-21 → 06-24 | OCEANS Aberdeen; SPC poster with the rig and before/after audio |
| 07 → 09 | JOE / ICASSP 2028 (about Sep 2027, inferred) / NeurIPS E&D (D5) |

---

## 13. Risks and fallbacks

| # | Risk | Trigger | Fallback |
|---|---|---|---|
| F1 | Projector not available | Nothing in hand at G1 (11-06) | Use the DAEX25W-8 for S1/v0 and state its usable band (likely ≥ 1 kHz). f_lo rises, and the scoring band shrinks accordingly. Upgrade for v1. |
| F2 | Pool access short | < 2 sessions confirmed at G1, or S2 cancelled | S3 becomes the v0 core. Cut v0 to 2 positions (1 and 6 m) × off/on with 200 + 40 utterances. Tank or dive well as a last resort, with claims scoped to that site. |
| F3 | Simulator fails H1b | G3 fail | C9 (measured-IR replay, already training since W4) becomes the training condition. The ISM errors and the sensitivity analysis become RQ1's finding. |
| F4 | Alignment or drift fails | G2 | Shorter blocks (20 s), mid-block chirps, Daisy-driven playback if firmware allows; WER-only for affected takes. |
| F5 | RS26 front end inadequate | E0 rule (§6.4 step 2), or noise floor > −90 dBFS | **Reference hydrophone is the scored channel by default**; array channels kept as secondary data. |
| F6 | H2-primary refuted or inconclusive | v0-test result | Publish the negative or inconclusive result with the benchmark and the strong baselines; it is still a contribution. Real FT for the full paper uses **only dev-clean recordings**, evaluated on v1-test positions. Never fine-tune on v0-test. |
| F7 | Compute shortfall | < 200 GPU-h by 11-06, or smoke-test throughput too slow | DFN3 C2 (3 seeds) + C9 only; MetricGAN+ and C8 move to the full paper; cap steps at 20k. |
| F8 | Kiel publishes real data first | before submission; or the DAGA 2027 programme (March 2027) lists a Kiel underwater-speech paper with measured data | Reposition as independent validation; lead with the release, strong baselines, ASR and the performance model. |
| F9 | Whisper hallucinations dominate | > 10% of utterances flagged on dev | WER with and without flags; Parakeet as co-primary; VAD gating as a dev-chosen condition. |
| F10 | Thrusters forbidden in the pool | no approval by W3 | Digitally add unit-B recordings to real off-takes ("semi-real noise", labelled); RQ9 then uses SNR from digital mixing. |
| F11 | Pinger-band leakage | energy in 25–40 kHz above ambient | Reduce drive level first; then LPF at 6 kHz *and* reduce drive. Mandatory before any IVC demo. |
| F12 | IRB delays | no approval by 02-15 | v1 without live talkers; E22 to the extension; RQ8 labelled proxy-only. |
| F13 | TBD roles uncovered | HW/OPS/ML2 unnamed by 11-01, or any role uncovered > 1 week | The Lead takes ML2 (DFN3 only); SPK takes HW (Aquarian boards, no custom build); SIM takes OPS logging. MVR scope drops to F7 + F2-reduced. |
| F14 | Author/credit disagreement | unresolved by 11-01 | Pause external submission plans. |
| F15 | EH&S / electrical approval | required and not granted by G1 | Battery-only setup (DAEX or UW30 on a battery amplifier). The Lubell is used only if a qualifying non-bridged battery amplifier is found. |
| F16 | No calibrated reference above 4 kHz | G1 answer is no | Absolute levels above 4 kHz are reported as uncalibrated. Training and testing are unaffected (relative responses). Safety uses the ≤ 4 kHz calibrated band plus a stated margin. |
| F17 | Channel time variance ruins averaging | low sweep-to-sweep coherence at high frequency | Single-sweep IRs; report the variance; inject it in uwsim (§4.2 #14). |
| F18 | Thruster SNR too low (everything fails) | S1 pilot speech-band SNR at 3 m < −5 dB at the lowest useful throttle | Increase source level within safety limits, or move the thruster to 2 m. Choose T_hi/T_lo to span about 0 to +10 dB at 3 m. |
| F19 | Second site unavailable | none booked by 03-15 | Scope every claim to "this pool / enclosed reverberant water"; E16 transfer is reported as LOPO only. |

---

## 14. Paper skeleton and abstract template

### 14.1 OCEANS full paper (4–6 pages, IEEE; length per the Aberdeen portal, unverified)

| Section | Content | Figures / tables |
|---|---|---|
| I. Introduction | Diver and robot-to-diver voice links; audio through water vs text over a modem (Rong 2025: WER measured on in-mask audio before the water; speaker-dependent); why the pool channel differs from a room (pressure-release surface, waveguide geometry, measured DRR); headline claim and primary endpoint; contributions: **C2 benchmark + measured channel first**, then the validated simulator, adaptation results and the performance model | Fig. 1: rig + pipeline |
| II. Related work | Kiel DAGA 2025/26 (simulator + U-Net; the paper presents no objective metrics or measured-data results); Jiang 2024; Luo 2026; SIREN (human audibility); in-air sim-to-real (Milo, SonicSim, URGENT, RealMAN) | — |
| III. Measurement rig and channel | Chain, calibration, **uncertainty budget**; c from the TOF slope; measured EDC (single vs two-slope), DRR/C50 vs range, low-band cutoff, time variance, test–retest | Table I: rig, benchmark statistics, uncertainty; Fig. 2: measured channel |
| IV. Simulator | Signed ISM with R(θ), tilted floor, tail and low band; calibration; validation at **new positions** against repeatability thresholds and the downstream criterion; competitor baseline | Fig. 3: sim vs measured at held-out positions with repeatability bands; Table III: validation errors vs thresholds |
| V. Benchmark | Protocol, R1 chain reference, dev/test splits, pre-registration, release | (Table I) |
| VI. Setup | Systems (zero-shot, WPE, WHAMR!, C2, C7, C8, C9, oracle), recipes, metrics, statistics | — |
| VII. Results | Primary endpoint; strong baselines; test–retest floor; human anchor; performance model and usable region; second site | Table II: main results with CIs; Fig. 4: WER vs measured SNR and C50 per system (GLMM fit + held-out points) |
| VIII. Discussion | Limits: one pool (+ second site), LibriSpeech playback, Lubell source level as a best case vs a diver transducer, regulator noise; what transfers to harbours (performance model); RoboSub compliance | — |
| IX. Conclusion and release | Zenodo DOI, code | — |

Everything else (RQ3–RQ8, ablations, multichannel, transmission modes) goes to the extension.

### 14.2 OCEANS abstract template

The target is 2 pages and 500–1,000 words, with figures counted (Monterey 2026 rules, unverified for Aberdeen). **Every [bracket] must come from our own measured results. Never fill one from expectation.**

> **[Title, §1]**
> [Student lead author], [co-authors], Stanford University
>
> **Motivation.** Voice links for divers and underwater robots either send speech as audio through water or convert it to text over an acoustic modem. The text approach reports low word error rates (WER) only on audio captured inside the diver's mask, before the water [Rong 2025]. Audio through water keeps speaker identity and prosody. However, an enclosed water body is a shallow, pressure-release-bounded waveguide rather than a room, and vehicle noise adds to it. Speech enhancement (SE) models trained in air were not designed for this channel. To our knowledge, no openly released through-water speech benchmark with aligned references exists. Recent simulator work [Kiel DAGA 2025/26] presents no objective metrics or measured-data results.
>
> **Measured channel.** We play LibriSpeech through a [projector] in a [L × W × H m] pool and record it on a 4-hydrophone array (Daisy Seed, 96 kHz / 24-bit). The calibration uncertainty is [±x dB] up to [f] kHz. Sound speed from the time-of-flight slope is [c ± SE] m/s. Across [N_IR] sine-sweep impulse responses, the energy decay is [single-/double-slope] in [k] of [K] octave bands. The direct-to-reverberant ratio falls from [DRR_0.5] dB at 0.5 m to [DRR_1] dB at 1 m. Beyond [r] m no direct path can be separated in the speech band. Repeat measurements across sessions vary by [σ_rep].
>
> **Simulator.** Our open simulator, uwsim, combines a signed, angle-dependent image-source model, a measured low-frequency band, the measured transducer chain and measured thruster noise. At source and receiver positions not used in fitting, it reproduces arrival structure, DRR and C50 within [x] of measurement repeatability. Unprocessed WER on its digital twin is within [y] points of the real recording, versus [z] points for an in-air-style simulator.
>
> **Benchmark v0.** [N_utt] test utterances (plus [N_dev] development utterances) from [N_spk] speakers at [d1, d2, d3] m. Thruster on and off are interleaved in alternating blocks. The measured speech-band SNR spans [a, b] dB.
>
> **Preliminary results** (primary endpoint pre-registered at [OSF link]). On unprocessed hydrophone audio, Whisper large-v3 WER is [W_C0] % (95% CI [..]), against [W_R1] % for the chain-only reference. Zero-shot in-air SE changes WER by [Δ_zs] points; WPE dereverberation by [Δ_wpe]. Fine-tuning DeepFilterNet3 on uwsim changes WER by [Δ_ft] points (CI [..], 3 seeds), compared with [Δ_C7] for the same recipe on in-air reverberation, [Δ_C9] for training on measured impulse responses, and [Δ_C8] for adapting Whisper itself. The test–retest noise floor is [f] points. WER follows measured SNR and C50 as shown in Fig. [k].
>
> **Full paper.** A new held-out test set (v1), a second site, a human transcription anchor, a performance model validated on held-out positions, and the release of the benchmark, impulse responses, thruster noise bank and simulator (CC BY 4.0 / Apache-2.0).
>
> [Fig. 1: rig + pipeline] [Fig. 2: measured vs simulated channel at held-out positions] [Table 1: WER/ESTOI by system]
> References: [5–8 key references]

**Fallback abstract (MVR fallback).** Keep Motivation, Measured channel, Simulator and Benchmark. Replace Preliminary results with: unprocessed WER, zero-shot and WPE baselines and the chain-only oracle, WER vs measured SNR and C50, and "fine-tuning results are planned for the full paper". No placeholder may imply a fine-tuning result.

---

## 15. Open decisions for the lead

| ID | Decision | Options | Recommendation / deadline |
|---|---|---|---|
| D1 | Projector | Borrow an LL916 / buy LL916C + AC205C (about $1.75–2.05k) / UW30 (about $500) / DAEX only | Whatever is in hand at G1; purchases off the critical path. **11-06.** |
| D2 | Primary reference | R1 chain reference (position-independent) vs R0 | R1 primary, R0 reported, R1-dp diagnostic at ≤ 1 m. Fix in the pre-registration. |
| D3 | Fine-tuned models in the MVR | DFN3 (×3) + MetricGAN+ vs adding SepFormer | DFN3 + MetricGAN+; SepFormer zero-shot only; add C10 (Kiel-recipe U-Net) if compute allows. |
| D4 | Interspeech 2027 | skip / distinct ML paper | **Skip**; decide by 01-03. |
| D5 | Dataset venue | NeurIPS E&D / Scientific Data / Zenodo only | Zenodo/HF at camera-ready; NeurIPS E&D only if v1 ≫ the OCEANS paper. |
| D6 | Compute | FarmShare (no sponsor, L40S) now; Sherlock and Marlowe (5,000 free GPU-h for a new PI) need a faculty sponsor / lab / cloud | ≥ 200 GPU-h by 11-06 for the MVR; about 1,000 GPU-h for the extension (estimate). |
| D7 | Author order, SPC lead | — | W1 with Kaito and Keiya. |
| D8 | Code licence | Apache-2.0 / MIT | Apache-2.0. |
| D9 | Live talkers | yes (IRB) / no | Yes if IRB clears by 02-15. |
| D10 | Thrusters live vs digital | — | Live with alternating blocks if approved; otherwise F10. |
| D11 | Vehicle demo | IVC at RoboSub 2027 / benchmark only | Benchmark only for OCEANS. |
| D12 | Second ASR | Parakeet-TDT-0.6B-v2 / OWSM v3.1 | Parakeet. |
| D13 | New-prompt TTS voice | open TTS with redistributable output (licence to check) | Decide by 01-18. |
| D14 | Second site | Hopkins / dock / lake / tank | Ask in W2; book by 03-15. |
| D15 | Pre-registration scope | primary endpoint only / all confirmatory RQs | All confirmatory (RQ1, RQ2, RQ9); file by 12-06. |
| D16 | Faculty sponsor | — | **Name one by 10-30.** Needed for the IRB Academic Sponsor role, Sherlock and Marlowe: a single point of failure for three resources. |
| D17 | Pool | Avery Competition (3.35–4.27 m) / Maas dive well (18 ft) | Avery Competition by default; the Maas well gives the longest gates and the lowest cutoff. Geometry drives gates, cutoff and Appendix A. **Decide at G1 (11-06).** |
| D18 | Kiel contact | before / after the abstract | After abstract submission, 01-18 → 01-31; small ask; no co-authorship request up front (RESEARCH_UPDATE_3 §3.3). |
| D19 | Claim wording | — | Lead claim: "to our knowledge, the first openly released through-water speech benchmark with aligned references, and the first study adapting pretrained SE architectures to it". Explicitly cite prior real-data classical work (Cao 2017, Zhang & Liu 2021, WTD 71, Kiel 2018) and the frogman DNN paper (ICICSP 2023). |

---

## Appendix A. Numbers derived in this plan (computed)

- **Marczak pure-water c** (coefficients confirmed, NPL): 1482.4, 1496.7, 1501.9 and 1504.4 m/s at 20, 25, 27 and 28 °C. dc/dT ≈ 2.6 m/s/°C at 26 °C.
- **c from the TOF slope:** relative SE ≈ 0.44% for ranges {1, 3, 6, 10} m at ±3 cm; 0.74% at ±5 cm.
- **First-image delay behind the direct path,** both ends at 1 m depth, H = 2 m (surface and floor images coincide at mid-depth):

  | Range | Delay | Valid above (2–3/Δt) |
  |---|---|---|
  | 0.5 m | 1.04 ms | about 1.9–2.9 kHz |
  | 1 m | 0.82 ms | about 2.4–3.6 kHz |
  | 3 m | 0.40 ms | about 5.0–7.4 kHz |
  | 6 m | 0.22 ms | about 9–14 kHz |
  | 10 m | 0.13 ms | about 15–23 kHz |

  These delays are set by the surface image and hold for any depth ≥ 2 m with both ends at 1 m. Placing both ends at mid-depth in a deeper pool lengthens the gate (surface-image limited, c = 1500):

  | Pool / placement | Range | First-image delay | Valid above (2–3/Δt) |
  |---|---|---|---|
  | Avery Competition, both ends at mid-depth 1.9 m | 1 m | 1.95 ms | about 1.0–1.5 kHz |
  | Avery Competition, both ends at mid-depth 1.9 m | 0.5 m | 2.22 ms | about 0.9–1.35 kHz |
  | Maas well, both ends at 2.75 m | 1 m | 3.05 ms | about 0.66–0.98 kHz |

- **Lloyd's mirror, two-point** (surface only, R = −1, both ends at 1 m depth, dB re direct):

  | Range | 200 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
  |---|---|---|---|---|---|
  | 1 m | −1.3 | +2.9 | −1.0 | +2.4 | +1.6 |
  | 3 m | −6.2 | +0.8 | +4.9 | +0.4 | +4.7 |
  | 6 m | −11.4 | −3.7 | +1.8 | +5.6 | −1.9 |
  | 10 m | −15.6 | −7.8 | −2.0 | +3.3 | +5.9 |

  10° grazing at these depths corresponds to about 11 m range.
- **Asymmetric depths** (source 0.7 m, receiver 1.2 m, r = 1 m, H = 2 m): surface image 0.69 ms after the direct path; floor image 0.12 ms after the surface image.
- **Diffuse level re the 1 m level (Avery Competition pool):** −0.7 dB (A ≈ 59 m²) to −3.7 dB (A ≈ 117 m²), beyond R_c ≈ 1.1–1.5 m. A from Sabine with α = 0.05–0.1 (guessed) over about 1,174 m² of wall and floor (estimate; the field is probably not diffuse, K8).
- **LL916 directivity** (a = 11.4–11.7 cm from Lubell / Inyo): ka = 0.48, 0.96, 1.92 and 3.83 at 1, 2, 4 and 8 kHz; ka = 1 at about 2.03–2.09 kHz. Rayleigh distance at 8 kHz is about 0.22 m (λ = 18.75 cm).
- **Gate and coherence lengths:** λ at 2 kHz = 0.75 m; λ/4 at 375 Hz = 1 m (source depth effect on radiation impedance).
- **ESS:** processing gain 10·log10(10 s × 20 kHz) = 53 dB (processing gain only). Clock skew at 50 ppm over 10 s = 0.5 ms.
- **STANAG-style USB on 8,087.5 Hz** with a 300–3,000 Hz baseband occupies 8,387.5–11,087.5 Hz; its 3rd harmonic is 25.2–33.3 kHz.
- **v0 core:**
  - 3 positions × 2 thruster states × (200 + 40) utterances × 7.4 s × 1.16 = 3.43 h of playback;
  - test speech 2.47 h per channel;
  - S3 extras: retest 50 × 2 = 14.3 min; T_lo sweep of 200 = 28.6 min; 0.5 m point of 100 = 14.3 min.
- **Training throughput:** 64 s of audio per step → 50k steps ≈ 890 h seen.

---

## Reviewer notes

**Adopted** (from both reviews unless noted):

1. **Circular simulator validation.**
   - Validate only non-fitted quantities (arrivals, DRR/C50 spatial variation, coloration, cutoff, EDC curvature); T20 is descriptive only.
   - Hold out whole positions.
   - Thresholds come from measured repeatability, and a downstream sim-twin WER criterion is added.
   - One pass rule for H1b and G3.
   - The R_s = +1 polarity criterion is dropped.
   - Stronger competitor baseline: pyroomacoustics with the water c.
   - ISO 3382 JNDs are no longer used as thresholds.
2. **R1 undefined beyond about 1 m.** R1 is now the position-independent chain reference with per-utterance delay and gain; the Lloyd's-mirror dipole counts as channel; the uwsim target is identical; R1-dp only at 0.5–1 m above about 2.5 kHz. The arithmetic is reproduced in Appendix A.
3. **Waveguide, not diffuse room.**
   - R(θ) half-space model and a tilted floor in v0.
   - Two-slope test (H1a); measured DRR/C50 replace the Sabine R_c, which is labelled an estimate everywhere.
   - Low band from a measured-IR splice (calibrated pool) or a modal sum (generic pools).
4. **No link to the ocean.**
   - The RQ9 performance model replaces d*; second-site or open-water session in the full paper.
   - Claims scoped to enclosed reverberant water otherwise; diver framing first.
   - E19 Watermark baseband replay dropped (bands confirmed as 4–8, 10–18 and 32.5–37.5 kHz).
5. **Calibration traceability.** Substitution ≤ 2 kHz, gated ≥ 3 kHz with an overlap check; calibrated-reference request at G1; uncertainty budget; explicit absolute-vs-relative split.
6. **c circularity and transducer delay.** TOF slope with an intercept; loopback timing; Marczak instead of the F&G term; salinity question; the 1% change presented as minor (reviewer 2).
7. **ESS protocol.**
   - Loopback drift correction before averaging; sweep coherence and IR variance reported.
   - ESS starts at the projector's lower limit with pre-shaping; ≥ 4 s of silence.
   - G2 targets set from the pilot; 53 dB described as processing gain only.
8. **Thruster and noise design.**
   - ABBA alternating blocks remove the time confound; drift anchor; two throttle levels via the S3 SNR sweep and v1.
   - Per-take measured SNR/C50/DRR in the metadata.
   - Flow-noise flag; held-out thruster unit B; scope on self-noise.
9. **Projector directivity and depth.** Orientation sweep, chain re-measured per depth, monitor moved to ≥ 0.5 m off-axis. Dimensions confirmed from Lubell (9.25").
10. **Rig contamination.** Flooded or rope frames, isolated thruster mount, rig-only ESS, RS26 low-frequency check with the reference hydrophone as default scored channel if the elements are poor.
11. **Straw-man baselines (reviewer 2).**
    - Added: WPE 1ch/4ch, SepFormer-WHAMR! (confirmed on HF; an 8 kHz model, flagged), C7 in-air control and C9 measured-IR replay in the MVR, Whisper-LoRA on uwsim (C8), and an oracle.
    - TF-GridNet only if a checkpoint is found. (Rev. 3: found, `kohei0209/tfgridnet_urgent25`, CC BY 4.0.)
12. **Novelty framing.** Lead with C2; RQ3 presented as a RealMAN replication; within-site caveat; second site.
13. **Selection on the test set.** v0-dev split; OSF pre-registration before scoring v0-test; v1-test as the final untouched set; F6 no longer fine-tunes on test positions.
14. **Statistics.**
    - Test–retest floor; TOST and non-inferiority for nulls; paired CIs of gap differences.
    - Kendall τ dropped below 6 systems; clustered CIs for H8.
    - 3 seeds for the primary model and no architecture claims in the abstract.
    - Power re-estimated from v0-dev.
15. **The "any model" claim.** Dropped; per-model recipe table (native losses; LoRA insertion; SGMSE+ cost); per-model in-domain forgetting sets.
16. **Metrics unanchored.** Human transcription anchor (E22); new-prompt TTS subset against Whisper contamination; CI-SDR's limits stated.
17. **Schedule.**
    - Calibration and validation both happen in S1, so G3 moves to 11-25 and real C2 launches only after G3.
    - Replay training starts in W4 as insurance; S3 is a true backup.
    - Purchases off the critical path; perfboard or Aquarian preamp.
    - Compute re-estimated and measured in W4; TBD owners due 11-01; EH&S question in W1.
18. **Scope.** OCEANS paper limited to RQ1, RQ2 and RQ9, with a smaller figure and table set.
19. **Peak-normalization bug.** Replaced by speech-active RMS normalization defined on R0; H7 made two-sided.
20. **Factual fixes:** K5 (OTS carriers), K9 (Lloyd numbers), λ/4 rule, received-level safety rule with crest factor, f_lo of 300–500 Hz (Rev. 3: now about 500–650 Hz, see K17), E17 needs the chain to 12 kHz plus a harmonic warning, the F11 fix, SIREN wording, "did not use Whisper", Kiel wording, the "to our knowledge" hedge, ≥ 4 s silence, the pool geometry flagged unverified, and the Orin vs Pi 5 comparison dropped.

**Adopted with modification:**

- **"Make a borrowed calibrated reference a G1 requirement."** It is a G1 *request* with a defined fallback (F16), not a blocker. SE training and testing, R1 and WER need only relative chain responses. Only absolute levels need the reference.
- **"Dummy EMI channel on every thruster take."** Replaced by a per-session EMI check with identical cabling, plus ABBA blocks. Sacrificing a hydrophone on every benchmark take would cost a released channel.
- **"Two-level bootstrap and random position effects in v0."** v0 has only 3 positions and 1 core session, so v0 CIs are explicitly scoped, and session variance comes from test–retest. The full random-effects analysis starts with v1.
- **K5.** The reviewer's "31.5–32.768 kHz" was checked. Retailer listings give 32.768 kHz USB and 31.250 kHz, not 31.5 kHz. Manufacturer datasheets were not read.
- **Modal sum below 500 Hz.** Used for generic pools only. For the calibrated pool, the measured low-band splice is more faithful and cheaper.

**Rejected or not acted on:**

- **"AUVs do not speak, so the AUV framing is weak."** Mostly adopted (diver-first framing), but robot-to-diver audio is kept as a secondary use case, because SIREN shows it is real.
- **"ISO 3382-1 Annex A does not list T20/T30; D50's JND is 0.05; resolve D50."** Not verified in this pass. The search confirmed only the 5% reverberation-metric JND and the 1 dB C80 JND via secondary sources. The point is moot, because no ISO JND is used as a threshold any more. The Acta Acustica DOI (10.3813/AAA.918574) is a decay-range paper, not a JND source, and is no longer cited for JNDs.
- **"Use a 20-utterance W3 pilot to re-estimate power."** Too small. Power is re-estimated from v0-dev (40 utterances × 6 conditions) plus the pilot.
- **The $200–400 Prolific cost** is kept as an unverified estimate, not a quote.