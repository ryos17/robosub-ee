# OCEANS 2027 abstract: working draft (2026-11-03)

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

In contrast, a rank-8 LoRA adaptation of Whisper on simulated audio, with no enhancement, lowered WER in every condition and at every range. The adapter that also saw stationary noise in training did best. Under stationary noise at 0 dB SNR, whose spectral slope was held out of training, WER fell from 20.3 / 48.4 / 56.4% to 16.0 / 37.2 / 44.5% at 1 / 3 / 6 m. In quiet conditions it fell from 3.1 / 3.7 / 4.3% to 1.7 / 2.6 / 3.2% [SIM]. The paired bootstrap 95% CIs exclude zero in all nine cells.

Adapted Whisper can transcribe correctly and then fail to stop, appending invented text. A decode-length cap tied to audio duration removed every such loop without changing the unadapted baseline.

We release the simulator, the evaluation harness, and [the measured pool channel and benchmark].

---

## Results section (draft, sim-only)

### Setup

- **Test material.** 200 LibriSpeech test-clean utterances (5 per speaker × 40 speakers), rendered through `uwsim` v0. Nominal competition pool, projector and hydrophone at 1 m depth, on-axis, at 1, 3 and 6 m, one fixed impulse response per position.
- **Conditions.**
  - quiet: 30 dB SNR.
  - noise0: 0 dB SNR, stationary −3 dB/oct Gaussian noise.
  - bubbles0: 0 dB SNR, synthetic Minnaert bubble trains. This is the noise the trained systems saw.
- **Recognizer.** Whisper large-v3, guarded decoding for every system:
  - beam 5 at temperature 0, with Whisper's temperature fallback (compression ratio > 2.4 or mean logprob < −1.0)
  - a decode-length cap of ceil(8 tokens/s × duration) + 16
  - no previous-text conditioning.

  The earlier fixed policy (temperature 0, no cap) is kept as a sensitivity check. Every system, front ends included, is reported under the guarded policy.
- **Statistics.** Corpus WER with speaker-cluster bootstrap 95% CIs. Paired utterance-level bootstrap for ΔWER (2000 resamples).

### Table: WER (%) of unprocessed audio (C0) vs Whisper-LoRA (C8), guarded decoding [SIM]

C8 is shown for seeds 0 and 1 trained on bubble noise only, and for one seed with stationary noise added to training ("stat"). ΔWER is the paired speaker-cluster bootstrap with a 95% CI.

| condition | range | C0 | C8 s0 | C8 s1 | C8 stat | ΔWER stat − C0 [95% CI] |
|---|---|---|---|---|---|---|
| noise0 | 1 m | 20.3 | 17.1 | 18.1 | 16.0 | −4.3 [−6.5, −2.4] |
| noise0 | 3 m | 48.4 | 42.1 | 43.2 | 37.2 | −11.3 [−13.7, −9.0] |
| noise0 | 6 m | 56.4 | 48.8 | 50.1 | 44.5 | −11.9 [−14.8, −8.8] |
| bubbles0 | 1 m | 2.4 | 2.0 | 2.0 | 1.7 | −0.8 [−1.3, −0.3] |
| bubbles0 | 3 m | 5.1 | 3.0 | 3.4 | 3.2 | −2.0 [−2.9, −1.2] |
| bubbles0 | 6 m | 5.7 | 4.4 | 4.5 | 4.1 | −1.7 [−2.6, −0.9] |
| quiet | 1 m | 3.1 | 2.0 | 2.7 | 1.7 | −1.4 [−1.9, −0.9] |
| quiet | 3 m | 3.7 | 2.3 | 3.8 | 2.6 | −1.1 [−1.8, −0.5] |
| quiet | 6 m | 4.3 | 3.7 | 4.8 | 3.2 | −1.2 [−1.9, −0.4] |

Under the fixed policy, stat had runaway loops in 46 of 1200 quiet and bubble decodes, and seed 0 in 1, the bubbles0 1 m outlier at 7.9%. The guard removes all of them (RESULTS_E7_SIM_DRYRUN.md, "Guarded decoding").

The best enhancement front end in each noise0 cell, for comparison, guarded decoding: 1 m 21.1% (WPE), 3 m 48.3% (WPE), 6 m 56.9% (WPE). Every trained or zero-shot neural enhancer was worse, between 32% and 107%.

### Findings, in the order the paper would state them

1. **The enhancement front ends do not help the recognizer.** No front end lowered WER below C0 beyond the CI in any condition.
   - Trained enhancers (DeepFilterNet3 fine-tuned, U-Net) raised ESTOI against the chain reference from 0.36 to 0.56–0.61 at quiet 6 m. They still raised WER from 4.3% to 8.0–9.2%.
   - This extends the in-air finding that SE artefacts hurt ASR (Iwamoto et al., 2022) to the underwater chain, and the effect grows with range.
2. **Adapting the recognizer helps everywhere.**
   - The bubble-only seeds beat C0 in all 12 noisy seed × cell pairs, and the CI excludes zero in 10 of them; both exceptions are at bubbles0 1 m. Their noise0 gain grows with range, from about 3 to about 7 points, although they never saw stationary noise. So part of the gain comes from adapting to the reverberant chain, not the noise.
   - Adding stationary noise to training, with the test slope band held out, beats C0 in all 9 cells (every CI excludes zero). It beats bubble-only seed 0 by a further 4–5 points at noise0 3 m and 6 m. It is one seed, so replicate before quoting the margin over seed 0.
3. **Limitation: the quiet-condition gain of the bubble-only adapter is seed-dependent.**
   - Seed 0 improved quiet WER significantly at 1 m and 3 m; seed 1 did not, and its CIs straddle zero.
   - Seed 1 had the lower validation cross-entropy (0.097 vs 0.102) yet the higher WER in 8 of 9 cells. Checkpoint and seed selection for recognizer adaptation therefore needs a WER-based development set, not validation loss.
4. **Hallucination risk, and a fix.**
   - Under the fixed decode, adapted Whisper sometimes transcribed an utterance correctly and then failed to emit end-of-text, adding about 200 invented words: 1 of 1800 decodes for seed 0, 46 of 1200 quiet and bubble decodes for stat.
   - A length cap tied to audio duration removes every loop, and C0 WER does not move by more than 0.1 points. Pre-register the guarded decode for the real-data evaluation.

### What must change before submission

- Replace every [SIM] number with the real v0-test result. That data is the pre-registered H2 endpoint. H2d (SE vs recognizer adaptation) is exploratory in the abstract (RESEARCH_PLAN.md §2), so phrase finding 2 as exploratory.
- The simulator is uncalibrated. Do not describe it as "calibrated" until E0–E4 are done.
- The noise0 test noise is synthetic. The stationary-noise retrain of C8 (signed off 2026-11-02) trains on the same Gaussian generator, but only at slopes [−6, −4] ∪ [−2, 0] dB/oct. The (−4, −2) band around the −3 dB/oct test noise is held out.
  - State this in the paper's Methods: "Training noise slopes exclude a band around the test noise, so the test noise is unseen in spectral tilt, though from the same generator family."
  - The stronger claim (unseen noise type) still needs the recorded bank.
- One fixed scene per range. Real data add position variety.
- The H2-primary comparison (C2 vs C0) currently goes the wrong way in sim. Decide with the lead how the abstract frames that if it holds on real data.
