# OCEANS 2027 abstract: working draft (2026-11-02)

**Status: draft built from the sim-only E7 dry run.** It is not submittable as is.

- RESULTS_E7_SIM_DRYRUN.md says its numbers are a pipeline check, "not to quote". Every simulator parameter is an uncalibrated placeholder, there is no recorded noise bank, and the pool geometry is nominal.
- The pre-registered endpoint (RESEARCH_PLAN.md §2) is WER on the real v0-test pool recordings.
- This draft exists so the framing, figures and statistics are ready before then. When real data arrive, replace every number marked [SIM] and recheck every claim against the real results.
- If the deadline (2027-01-15) arrives before real data, the honest version is a simulation-and-methods abstract. Its wording would be "in simulation" throughout, and it would claim no field result.

Abstract field length: OCEANS abstracts are typically ~500 words, but confirm the limit in the 2027 call. The draft below is about 300.

---

## Title (candidate, from RESEARCH_PLAN.md §1)

Speech Through Water: A Measured Pool Channel and Open Benchmark for Adapting Speech Enhancement to Projector-to-Hydrophone Voice Links

## Abstract (draft)

Voice links between divers, surface teams and underwater vehicles pass speech through an underwater projector, a reverberant and noisy water channel, and a hydrophone. Off-the-shelf speech enhancement (SE) and automatic speech recognition (ASR) are trained on in-air audio and have not been evaluated on this chain.

We build `uwsim`, a simulator of the projector-to-hydrophone chain in a pool. It combines image-source early reflections over a sloped floor, a stochastic late tail, a modal low band, transducer responses with drive-level nonlinearity, and impulsive bubble and broadband noise. We use it to compare two places to adapt: enhancing the audio before a frozen recognizer, or adapting the recognizer itself.

On simulated pool recordings of 200 LibriSpeech test-clean utterances at 1, 3 and 6 m, every enhancement front end tested failed to lower Whisper large-v3 word error rate (WER) relative to unprocessed audio beyond the confidence interval [SIM]. This held for zero-shot MetricGAN+, SepFormer and DeepFilterNet3, and for DeepFilterNet3 and a U-Net trained on the simulator. The enhancers raised ESTOI substantially, but their artefacts cost more recognition accuracy than the noise and reverberation they removed, increasingly so with range.

In contrast, a rank-8 LoRA adaptation of Whisper on simulated audio, with no enhancement, lowered WER at 3 m and 6 m in every noisy condition, and under stationary noise at every range, in two independent training runs. Under stationary noise at 0 dB SNR, WER fell from 20.3 / 48.3 / 56.4% to 17.1–18.1 / 42.1–43.2 / 48.8–50.1% at 1 / 3 / 6 m. The paired bootstrap 95% CIs exclude zero, and that noise type was absent from training [SIM].

The gain in quiet conditions did not replicate across training seeds. Validation cross-entropy ranked the two runs opposite to their WER.

We release the simulator, the evaluation harness, and [the measured pool channel and benchmark].

---

## Results section (draft, sim-only)

### Setup

- **Test material.** 200 LibriSpeech test-clean utterances (5 per speaker × 40 speakers), rendered through `uwsim` v0. Nominal competition pool, projector and hydrophone at 1 m depth, on-axis, at 1, 3 and 6 m, one fixed impulse response per position.
- **Conditions.**
  - quiet: 30 dB SNR.
  - noise0: 0 dB SNR, stationary −3 dB/oct Gaussian noise.
  - bubbles0: 0 dB SNR, synthetic Minnaert bubble trains. This is the noise the trained systems saw.
- **Recognizer.** Whisper large-v3 with fixed decoding: beam 5, temperature 0 with no fallback, no previous-text conditioning.
- **Statistics.** Corpus WER with speaker-cluster bootstrap 95% CIs. Paired utterance-level bootstrap for ΔWER (2000 resamples).

### Table: WER (%) of unprocessed audio (C0) vs Whisper-LoRA (C8), two training seeds [SIM]

| condition | range | C0 | C8 seed 0 | C8 seed 1 | ΔWER seed 0 [95% CI] | ΔWER seed 1 [95% CI] |
|---|---|---|---|---|---|---|
| noise0 | 1 m | 20.3 | 17.1 | 18.1 | −3.2 [−4.9, −1.7] | −2.2 [−3.8, −0.7] |
| noise0 | 3 m | 48.3 | 42.1 | 43.2 | −6.2 [−8.7, −3.5] | −5.1 [−7.7, −2.5] |
| noise0 | 6 m | 56.4 | 48.8 | 50.1 | −7.6 [−10.4, −4.7] | −6.3 [−9.2, −3.6] |
| bubbles0 | 1 m | 2.4 | 7.9 | 2.0 | +5.5 [−0.9, +19.0] | −0.4 [−0.9, 0.0] |
| bubbles0 | 3 m | 5.1 | 3.0 | 3.4 | −2.1 [−3.0, −1.4] | −1.7 [−2.5, −1.0] |
| bubbles0 | 6 m | 5.7 | 4.4 | 4.5 | −1.4 [−2.2, −0.5] | −1.2 [−2.1, −0.3] |
| quiet | 1 m | 3.1 | 2.0 | 2.7 | −1.1 [−1.9, −0.3] | −0.4 [−1.1, +0.3] |
| quiet | 3 m | 3.7 | 2.3 | 3.8 | −1.4 [−2.1, −0.8] | +0.1 [−0.6, +0.8] |
| quiet | 6 m | 4.3 | 3.7 | 4.8 | −0.7 [−1.5, +0.2] | +0.5 [−0.5, +1.5] |

The best enhancement front end in each noise0 cell, for comparison: 1 m 21.1% (WPE), 3 m 48.2% (WPE), 6 m 62.4% (WPE). Every trained or zero-shot neural enhancer was worse, between 32% and 124%.

### Findings, in the order the paper would state them

1. **The enhancement front ends do not help the recognizer.** No front end lowered WER below C0 beyond the CI in any condition.
   - Trained enhancers (DeepFilterNet3 fine-tuned, U-Net) raised ESTOI against the chain reference from 0.36 to 0.56–0.61 at quiet 6 m. They still raised WER from 4.3% to 8.0–9.2%.
   - This extends the in-air finding that SE artefacts hurt ASR (Iwamoto et al., 2022) to the underwater chain, and the effect grows with range.
2. **Adapting the recognizer helps in noise, robustly.**
   - The two C8 seeds beat C0 in 11 of the 12 noisy seed × cell pairs, and the CI excludes zero in 10 of 12 seed × cell pairs. Both exceptions are at bubbles0 1 m.
     - Seed 0 there had a single hallucination loop: one 9-word utterance produced 219 inserted words.
     - Seed 1's upper bound there is 0.0.
   - The noise0 gain grows with range, from about 3 to about 7 points. C8 never saw stationary noise in training, so the gain comes from adapting to the reverberant chain, not to the noise.
3. **Limitation: the quiet-condition gain is seed-dependent.**
   - Seed 0 improved quiet WER significantly at 1 m and 3 m; seed 1 did not, and its CIs straddle zero.
   - Seed 1 had the lower validation cross-entropy (0.097 vs 0.102) yet the higher WER in 8 of 9 cells. Checkpoint and seed selection for recognizer adaptation therefore needs a WER-based development set, not validation loss.
4. **Hallucination risk.** LoRA adaptation produced one runaway insertion loop in 1800 seed-0 decodes, and none in seed 1. Both seeds had 2 flagged utterances, against 1 for C0. Report the flag rate on the real data.

### What must change before submission

- Replace every [SIM] number with the real v0-test result. That data is the pre-registered H2 endpoint. H2d (SE vs recognizer adaptation) is exploratory in the abstract (RESEARCH_PLAN.md §2), so phrase finding 2 as exploratory.
- The simulator is uncalibrated. Do not describe it as "calibrated" until E0–E4 are done.
- The noise0 test noise is synthetic. If C8 is retrained with stationary synthetic noise (pending sign-off), the noise0 result is no longer out of distribution. Use a held-out noise or the recorded bank for that claim.
- One fixed scene per range. Real data add position variety.
- The H2-primary comparison (C2 vs C0) currently goes the wrong way in sim. Decide with the lead how the abstract frames that if it holds on real data.
