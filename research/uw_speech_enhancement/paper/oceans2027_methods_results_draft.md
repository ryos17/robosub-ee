# OCEANS 2027: Methods and Results sections, working draft (2026-11-06)

**Status: sim-only draft.** Every number marked [SIM] comes from the E7 dry run on the uncalibrated `uwsim` v0 (RESULTS_E7_SIM_DRYRUN.md), not from pool recordings. The evaluation protocol in §III-F is frozen (signed off by the lead 2026-11-05) and will be applied unchanged to the real v0-test recordings. When those arrive, every [SIM] number is replaced and every claim rechecked. Text in *[brackets and italics]* is a placeholder for a measured quantity.

Companion files: `oceans2027_abstract_draft.md` (abstract and findings list), RESEARCH_PLAN.md §§4–8 (full specification), RESULTS_E7_SIM_DRYRUN.md (all sim results, including sensitivity checks).

**Open point for the lead.** The abstract draft headlines C8 stat seed 0 (16.0 / 37.2 / 44.5% on noise0). The frozen protocol carries forward seed 1 at its dev-WER checkpoint (16.4 / 38.2 / 43.8%), because seed 0's checkpoint was chosen by validation CE. This draft reports seed 1 as the primary C8 row and seed 0 as the replication. The abstract should be aligned one way or the other before submission.

---

## III. Methods

### A. Signal chain and simulator (`uwsim` v0)

We model the path from a clean utterance to a hydrophone recording as

y = HPF_pre( h_hyd ∗ ( Σ_k h_k ∗ (LPF_8k(s))^k ∗ h_ch ) + n ),

with the following components (RESEARCH_PLAN.md §4.2):

- **Drive.** The clean signal s is low-passed at 8 kHz, as in the real playback chain.
- **Projector.** A Hammerstein model of order 3, whose linear term is the projector-to-reference chain response h_chain. In v0 these are placeholders; the measured responses replace them after calibration (E2).
- **Channel h_ch.** An image-source model of a shoebox pool with a tilted floor, signed pressure-release surface reflections (R_s ≈ −1), and angle-dependent wall and floor reflection from a fluid–elastic half-space, for delays up to t_e = 40 ms. A stochastic late tail, energy-matched to the image-source decay at t_e, follows. Below the waveguide cutoff a modal sum replaces the image-source field.
- **Noise.** Minnaert bubble bursts (f₀ ≈ 3.29/a Hz, radius a = 0.5–3 mm, rate up to 20/s) and, in the "stat" training configuration, stationary coloured Gaussian noise (below).
- **Receiver.** Hydrophone response, preamplifier high-pass, and a 24-bit ADC with clipping, then resampling to 16 kHz.

The training target for enhancement models is the chain reference R1_sim = s ∗ h_chain, defined identically to the real-data reference R1 (§III-E): projector and hydrophone coloration with no pool path.

*All v0 parameters are uncalibrated placeholders.* Pool dimensions are the nominal competition pool; reflection parameters, tail decay and transducer responses are priors, not fits. No recorded noise bank exists yet, so thruster and ambient noise are absent from v0.

### B. Simulated test set (E7 dry run)

- **Utterances.** The v0-test selection: 200 LibriSpeech test-clean utterances, 5 from each of 40 speakers.
- **Geometry.** Projector and hydrophone at 1 m depth, on-axis, at ranges 1, 3 and 6 m, with one fixed impulse response per range.
- **Conditions.**
  - *quiet*: 30 dB speech-band SNR.
  - *noise0*: 0 dB SNR, stationary Gaussian noise with a −3 dB/oct slope.
  - *bubbles0*: 0 dB SNR, synthetic Minnaert bubble trains, the noise type the trained systems saw.

This gives 9 cells (3 conditions × 3 ranges) of 200 utterances each.

### C. Systems

- **C0.** Unprocessed hydrophone audio, decoded by the frozen recognizer.
- **C1, zero-shot front ends.** Single-channel WPE (`nara_wpe`), MetricGAN+, SepFormer (WHAM! 16 kHz and WHAMR! 8 kHz), and DeepFilterNet3, all with their published weights.
- **C2.** DeepFilterNet3 fully fine-tuned on `uwsim` toward R1_sim (50k steps, about 3.2M audio-seconds).
- **C10.** The Kiel-recipe U-Net (4-level, ideal-ratio-mask target), trained from scratch on `uwsim`.
- **C8, recognizer adaptation.** Whisper large-v3 with LoRA adapters (rank 8, α = 32, dropout 0.05) on the attention query and value projections, trained with cross-entropy on the LibriSpeech transcripts of simulated audio. No enhancement front end.
  - Data: train-clean-100 utterances of ≤ 30 s, rendered through `uwsim` on the fly. About 64 s of audio per optimizer step, AdamW, peak learning rate 1e-4 with 1k warm-up steps, at most 30k steps.
  - *Bubble-only* (seeds 0 and 1): the v0 noise mix, bubble trains only.
  - *Stat* (seeds 0 and 1): half of the training scenes also add stationary coloured Gaussian noise at −10 to 0 dB relative to the bubble component, with slopes drawn from [−6, −4] ∪ [−2, 0] dB/oct. The band (−4, −2) dB/oct around the noise0 test slope is held out, so the test noise is unseen in spectral tilt, though from the same generator family.
- **Oracles.** Whisper on R1 (chain coloration only) and on the clean digital source.

Front-end outputs are passed to the same frozen recognizer as C0. C8 decodes the unprocessed audio.

### D. Checkpoint selection for recognizer adaptation

Validation cross-entropy does not rank adapters by WER in our runs (§IV-C), so adapter checkpoints are selected on WER on a disjoint simulated development set (`scripts/dev_wer_select.py`):

- 120 LibriSpeech dev-clean utterances, so no speaker or utterance overlaps the test set.
- Different scene geometry from E7: ranges 1.5, 3.5 and 5.5 m, 1.5 m depth, receiver at 0.35 of the pool width, generator seed 20261104.
- Four conditions: quiet, bubbles at 0 dB, and stationary noise at 0 dB with slopes −5 and −1 dB/oct. Both slopes lie inside the training bands, so the held-out test band never influences selection.
- Criterion: the lowest mean of the four per-condition corpus WERs, under the decoding policy of §III-F. Every 4k-step checkpoint and the final checkpoint are candidates.

The adapter carried forward to real data is stat seed 1 at step 30k (SHA-256 `5063f02f…f52f90`), the only adapter whose checkpoint was chosen by this rule. The other adapters are reported as simulation references.

### E. Real-data benchmark (planned; protocol fixed)

*This subsection describes the v0 pool session, which has not yet taken place.*

- **Playback material.** About 60 s blocks of 8 utterances each, framed by linear-FM sync chirps (0.25 s, *[f_lo]*–8 kHz) with fixed silences, RMS-normalized to −26 dBFS, at one amplifier gain per session. Within each range, thruster-off and thruster-on blocks alternate in ABBA order, and the session's first block is repeated at its end as a drift anchor.
- **Positions.** Ranges 1, 3 and 6 m, projector and hydrophone at 1 m depth.
- **Recording.** *[Hydrophone(s), preamplifier, recorder]* at 96 kHz / 24-bit, with the amplifier output recorded through an attenuator as a loopback channel where possible.
- **Alignment.** Each take is aligned to its playback schedule by matched-filtering the sync chirps (`scripts/import_recording.py`):
  1. A coarse offset from the correlation of the matched-filter envelope with the nominal chirp pattern.
  2. A per-block chirp search around the predicted onsets, with a peak-to-sidelobe ratio (PSR) check at 6 dB.
  3. Clock drift per block from its two chirps, or from a take-level least-squares clock fit when a chirp fails the PSR check (the block is flagged).
  4. Drift correction by resampling to the playback clock, and per-utterance cuts with 0.25 s padding.

  On synthetic takes with drift of up to ±150 ppm, the offset is recovered within 2 samples at 48 kHz and the cuts within 2–3 samples.
- **References.** R1 = s ∗ h_chain, with h_chain measured once at close range per source depth, and only a broadband delay and gain fitted per utterance. R0 = s, band-limited.

### F. Evaluation protocol (frozen 2026-11-05)

Fixed before any real recording is scored. Because it was chosen after the simulation results, it is pre-registered, and any change after real data arrive will be reported as a deviation.

- **Recognizer.** `openai-whisper` large-v3 (pinned weights hash), fp16, English, transcription task, no prompt and no previous-text conditioning.
- **Level.** Each cut is RMS-normalized to −26 dBFS over its speech-active region, determined by voice activity detection on the aligned R0, and the same cut feeds every system.
- **Guarded decoding** for every system:
  - Beam search (width 5) at temperature 0.
  - Whisper's temperature fallback (0.2, 0.4, …, 1.0, best-of-5 sampling), triggered when the compression ratio exceeds 2.4 or the mean log-probability falls below −1.0. Sampling seed fixed at 0.
  - A decode-length cap of min(⌈8 tokens/s × duration⌉ + 16, 224) tokens. Correct transcripts of these clips run at a median 3.7 and a maximum 6.2 tokens/s.
- **Scoring.** Whisper's English text normalizer on both sides; corpus WER = (S + D + I) / N_ref.
- **Statistics.** Corpus WER with 95% confidence intervals from a speaker-cluster bootstrap. System differences are paired: ΔWER with a paired speaker-cluster bootstrap (2000 resamples), the same resampled speakers for both systems. A difference is called significant when its 95% CI excludes zero.
- **Runaway-loop count.** An utterance is counted as a loop if its insertions are ≥ max(10, 2 × reference length).

---

## IV. Results [SIM]

All numbers in this section are from simulation (`uwsim` v0, uncalibrated) and use the guarded decoding of §III-F.

### A. Enhancement front ends do not help the recognizer

Table II gives WER for every system. No front end lowers WER below C0 beyond the CI in any cell.

**Table II. Whisper large-v3 WER (%) by system, guarded decoding [SIM].** C8 = stat seed 1 at its dev-WER checkpoint (the protocol-selected adapter).

| cell | C0 | WPE | DFN3 0-shot | DFN3 C2 | U-Net C10 | MetricGAN+ | SepF wham16k | SepF whamr | C8 |
|---|---|---|---|---|---|---|---|---|---|
| quiet 1 m | 3.1 | 3.0 | 2.4 | 2.3 | 2.6 | 3.5 | 6.0 | 3.6 | **2.3** |
| quiet 3 m | 3.7 | 3.4 | 5.8 | 6.0 | 5.5 | 38.0 | 84.7 | 20.7 | **2.7** |
| quiet 6 m | 4.3 | 4.4 | 13.3 | 9.2 | 8.0 | 43.3 | 105.9 | 43.0 | **3.4** |
| noise0 1 m | 20.3 | 21.1 | 32.0 | 41.6 | 71.0 | 72.7 | 50.3 | 42.2 | **16.4** |
| noise0 3 m | 48.4 | 48.3 | 74.2 | 81.6 | 94.0 | 105.4 | 100.3 | 85.0 | **38.2** |
| noise0 6 m | 56.4 | 56.9 | 86.1 | 89.9 | 96.7 | 106.5 | 101.6 | 96.0 | **43.8** |
| bubbles0 1 m | 2.4 | 2.4 | 3.5 | 2.6 | 2.6 | – | – | – | **2.2** |
| bubbles0 3 m | 5.1 | 4.2 | 27.7 | 7.0 | 5.9 | – | – | – | **3.1** |
| bubbles0 6 m | 5.7 | 5.3 | 49.5 | 9.9 | 8.6 | – | – | – | **4.0** |

MetricGAN+ and the SepFormers were not run on bubbles0. R1 is 2.1–2.2% and the clean source 2.4% in every cell.

- **WPE is the only front end that does no harm.** It is level with C0 in every cell.
- **The neural front ends raise WER with range**, in quiet as well as in noise. The two models trained on the simulator toward R1 (C2, C10) roughly double C0's WER at quiet 6 m (9.2% and 8.0% vs 4.3%).
- **Intelligibility and recognition disagree.** C2 and C10 raise ESTOI against R1 at quiet 6 m from 0.36 to 0.61 and 0.56, while raising WER. At noise0 1 m, zero-shot DFN3 raises ESTOI from 0.31 to 0.50 while raising WER from 20.3% to 32.0%; it is worse than C0 on 126 of 200 utterances and better on 34. This extends the in-air finding that enhancement artefacts hurt a robust recognizer (Iwamoto et al., 2022) to the projector-to-hydrophone chain.

### B. Adapting the recognizer helps in every condition

**Table III. C8 vs C0, paired ΔWER (C8 − C0, points) [95% CI], guarded decoding [SIM].** Bold: CI excludes zero.

| cell | C0 WER | C8 stat s1 (protocol) | ΔWER s1 | C8 stat s0 (replication) | ΔWER s0 |
|---|---|---|---|---|---|
| quiet 1 m | 3.1 | 2.3 | **−0.8** [−1.4, −0.3] | 1.7 | **−1.4** [−1.9, −0.9] |
| quiet 3 m | 3.7 | 2.7 | **−0.9** [−1.6, −0.3] | 2.6 | **−1.1** [−1.8, −0.5] |
| quiet 6 m | 4.3 | 3.4 | **−1.0** [−1.7, −0.2] | 3.2 | **−1.2** [−1.9, −0.4] |
| noise0 1 m | 20.3 | 16.4 | **−4.0** [−5.7, −2.3] | 16.0 | **−4.3** [−6.5, −2.4] |
| noise0 3 m | 48.4 | 38.2 | **−10.2** [−12.8, −7.9] | 37.2 | **−11.3** [−13.7, −9.0] |
| noise0 6 m | 56.4 | 43.8 | **−12.6** [−15.5, −9.6] | 44.5 | **−11.9** [−14.8, −8.8] |
| bubbles0 1 m | 2.4 | 2.2 | −0.2 [−0.7, +0.2] | 1.7 | **−0.8** [−1.3, −0.3] |
| bubbles0 3 m | 5.1 | 3.1 | **−2.0** [−3.0, −1.1] | 3.2 | **−2.0** [−2.9, −1.2] |
| bubbles0 6 m | 5.7 | 4.0 | **−1.7** [−2.6, −0.9] | 4.1 | **−1.7** [−2.6, −0.9] |

- **The protocol-selected adapter beats C0 in 8 of 9 cells**; the exception is bubbles0 at 1 m, where C0 is already near the R1 floor. The second seed beats C0 in all 9.
- **The gain grows with range under the held-out stationary noise**, from about 4 points at 1 m to 12–13 points at 6 m, so the adapter is not only learning the noise: it is adapting to the reverberant chain.
- **The two stat seeds agree.** On noise0 their paired differences all include zero, and both beat the bubble-only adapter at noise0 3 m and 6 m (seed 1 by 3.9 and 5.0 points, seed 0 by 4.9 and 4.3). The two bubble-only seeds also beat C0 in every noisy cell they were scored in, by 2–8 points on noise0, although they never saw stationary noise.
- **C8 has the lowest WER of any non-oracle system in every cell.** It ties fine-tuned DFN3 at quiet 1 m (2.3%) and beats every front end elsewhere.

### C. Validation loss is a poor checkpoint selector

- Among the bubble-only seeds, seed 1 had the lower validation cross-entropy (0.097 vs 0.102) but the higher WER in 8 of 9 cells.
- On stat seed 1, the dev-WER pick (step 30k) beat the CE-best checkpoint (step 20k) on the test set in 3 of 9 cells beyond the CI (quiet 1 m −1.0, noise0 6 m −1.4, bubbles0 6 m −0.8 points) and was never worse. The CE-best checkpoint beats C0 in only 6 of 9 cells. The development set flagged its quiet-condition weakness (dev quiet WER 4.3% vs 3.1–3.5% for its neighbours), and the test set confirmed it.

### D. Runaway decoding in adapted Whisper, and a guard

- **Failure mode.** Under plain beam search at temperature 0 with no length limit, adapted Whisper sometimes transcribes an utterance correctly and then fails to emit end-of-text, appending 188–222 words of fluent invented text. This occurred in 1 of 1800 decodes for bubble-only seed 0 and in 46 of 1200 quiet and bubbles0 decodes for stat seed 0. Loop utterances are longer than average (median 28 vs 17 reference words).
- **Guard.** The decode-length cap of §III-F removes every loop, in all 36 system × cell pairs of the C0/C8 comparison. The temperature fallback fired on only 3 of those 7200 decodes, so the cap does the work. C0 moves by at most 0.1 points.
- **Front ends.** Under the same guard, front-end WERs that were already above 50% fall by 4–18 points, as long hallucinated decodes of badly damaged audio are cut. No conclusion changes: no front end beats C0.
- **Under the frozen protocol**, the selected C8 adapter produces no loops on the test set, and the fallback fires on 2 of 1800 decodes.

### E. Sensitivity

- **Fixed decoding policy.** Under plain beam search (temperature 0, no cap), the front-end conclusions are unchanged and C8 bubble-only seeds keep their noisy-cell gains; the stat adapter's quiet and bubbles0 cells are dominated by loops (RESULTS_E7_SIM_DRYRUN.md, "C8 with stationary training noise").
- **Seed-averaged adapter (exploratory).**¹

¹ *Exploratory, outside the frozen protocol.* We averaged the two stat seeds' adapters exactly, by stacking their low-rank factors (rank 16, α = 64) so that the merged weight update is the mean of the two seeds' updates, at matched steps 28k and 30k, and selected step 28k by dev WER (17.69% vs 17.64% for the protocol adapter). On the simulated test set the average beat C0 in all 9 cells but was level with the protocol adapter in every cell (all paired CIs include zero; noise0 point estimates −0.5, −1.2 and −0.6 points), with no loops. Weight averaging gave no measurable gain, and the protocol adapter is unchanged (RESULTS_E7_SIM_DRYRUN.md, "EXPLORATORY: C8 stat seed-average adapter").

### F. Limitations of the simulation results

- **Uncalibrated simulator.** Every `uwsim` v0 parameter is a placeholder; reverberation alone barely hurts Whisper in v0 (C0 3.1% → 4.3% from 1 m to 6 m in quiet), which may reflect over-absorptive placeholder boundaries.
- **Synthetic noise.** noise0 is synthetic Gaussian noise; its slope is held out of C8 training, but its generator is not. The claim of robustness to an unseen noise type needs the recorded noise bank.
- **One scene per range** and one test session; the real benchmark adds position variety and a test–retest noise floor.
- **The H2-primary comparison (C2 vs C0) goes the wrong way in simulation.** How the paper frames this if it holds on real data is an open decision for the lead.

---

## What changes when real data arrive

- Replace Tables II and III with v0-test results under §III-F, unchanged; report any deviation explicitly.
- Fill §III-E placeholders from the session log (hardware, *[f_lo]*, gains, measured ranges) and §III-A with the calibrated parameters.
- Add the real-data covariates (measured SNR, C50, DRR) and the test–retest floor; no difference smaller than that floor is interpreted.
- Keep the seed-average result as a footnote at most.
