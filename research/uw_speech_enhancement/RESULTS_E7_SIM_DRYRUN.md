# E7 sim-only dry run: zero-shot baselines on uwsim twins of v0-test (2026-10-29)

**Status: pipeline check, not a paper result.** Every simulator parameter is an
uncalibrated placeholder (CODE.md §6), no noise bank exists yet (the noisy
condition uses synthetic −3 dB/oct noise), and the pool geometry is nominal.
Use these numbers to debug the pipeline and to sharpen hypotheses, not to
quote. Nothing here changes the pre-registered endpoint (RESEARCH_PLAN.md §2).

Produced by `scripts/sim_baseline.py` (commit 799f2c5) into
`runs/sim_baseline/` (git-ignored): `summary.md`, `summary.json`,
`per_utt.jsonl`, `transcripts.jsonl`, `scenes.json`, `asr_info.json`.

## Setup

- **Utterances:** v0-test selection, 200 test-clean utterances (5 × 40 speakers).
- **Positions:** nominal Avery pool, projector and receiver at 1 m depth,
  on-axis, ranges 1 / 3 / 6 m. One fixed RIR per position.
- **Conditions:** `quiet` (SNR 30 dB) and `noise0` (SNR 0 dB, synthetic noise).
- **Systems:**
  - C0 = `noisy` (unprocessed);
  - C1 = `wpe1`, `metricganplus`, `sepformer_wham16k`, `sepformer_whamr`, `dfn3`, all zero-shot;
  - oracles `R1` (chain reference) and `clean` (digital LibriSpeech);
  - TF-GridNet not run (ESPnet not installed).
- **ASR:** Whisper large-v3 (openai-whisper 20250625, sha256 e5b1a55b…),
  fp16, beam 5, temperature 0, no conditioning. Input is RMS-active level
  normalized with the VAD of R0 (§8.3).
- **Statistics:** corpus WER with cluster-bootstrap 95% CIs over speakers
  (10 000 resamples); ESTOI and CI-SDR against R1.

## Results

WER % [95% CI]. Bold marks the best non-oracle system per row; R1 is about 2.1–2.2% and clean 2.4% in every row.

| condition | C0 noisy | wpe1 | dfn3 | sepformer_whamr | metricganplus | sepformer_wham16k |
|---|---|---|---|---|---|---|
| quiet 1 m | 3.1 [2.5, 3.8] | 3.0 [2.4, 3.8] | **2.4** [1.7, 3.2] | 3.6 [2.8, 4.5] | 3.5 [2.7, 4.5] | 6.0 [4.5, 7.8] |
| quiet 3 m | 3.7 [2.9, 4.6] | **3.4** [2.7, 4.3] | 5.8 [4.6, 7.1] | 20.7 [17.2, 24.6] | 38.0 [33.4, 42.8] | 91.5 [80.7, 107.1] |
| quiet 6 m | **4.3** [3.4, 5.4] | 4.4 [3.4, 5.5] | 13.3 [10.1, 16.8] | 43.0 [36.5, 49.9] | 43.3 [38.1, 48.8] | 110.3 [105.5, 115.9] |
| noise0 1 m | **20.3** [16.6, 24.6] | 21.1 [17.1, 25.6] | 32.0 [27.6, 37.2] | 42.2 [36.8, 48.4] | 72.7 [65.5, 80.3] | 54.7 [45.3, 66.6] |
| noise0 3 m | 48.3 [42.9, 54.0] | **48.2** [42.2, 54.7] | 81.9 [70.9, 96.1] | 97.7 [84.3, 114.4] | 120.0 [108.4, 134.7] | 106.3 [99.4, 116.5] |
| noise0 6 m | **56.4** [50.7, 62.4] | 62.4 [52.6, 75.9] | 90.8 [82.4, 102.3] | 96.0 [93.2, 98.7] | 124.2 [109.5, 142.3] | 102.0 [100.2, 104.3] |

ESTOI against R1:

| condition | C0 noisy | wpe1 | dfn3 | sepformer_whamr | metricganplus | sepformer_wham16k |
|---|---|---|---|---|---|---|
| quiet 1 m | 0.798 | 0.849 | 0.836 | 0.832 | 0.770 | 0.754 |
| quiet 3 m | 0.508 | 0.554 | 0.578 | 0.610 | 0.430 | 0.324 |
| quiet 6 m | 0.364 | 0.404 | 0.413 | 0.463 | 0.307 | 0.138 |
| noise0 1 m | 0.310 | 0.310 | 0.497 | 0.496 | 0.330 | 0.459 |
| noise0 3 m | 0.230 | 0.233 | 0.316 | 0.310 | 0.192 | 0.151 |
| noise0 6 m | 0.168 | 0.172 | 0.228 | 0.175 | 0.125 | 0.090 |

The full table, with S/D/I counts, hallucination flags and CI-SDR, is in
`runs/sim_baseline/summary.md`.

## What the dry run shows (all conditional on the placeholder simulator)

1. **Zero-shot SE never beats C0 on Whisper WER beyond the CI.** The only
   point estimates below C0 are dfn3 at quiet 1 m (2.4 vs 3.1) and wpe1 at
   quiet 3 m (3.4 vs 3.7) and noise0 3 m (48.2 vs 48.3); all overlap C0.
   Every neural system has a worse point estimate than C0 at 3 m and 6 m. This matches the
   in-air finding that front-end SE hurts a robust ASR model, and it is
   exactly the gap H2 (sim fine-tuning) has to close.
2. **ESTOI and WER disagree.**
   - At noise0 1 m, dfn3 raises ESTOI from 0.310 to 0.497 but raises WER from 20.3% to 32.0%.
   - Per utterance, dfn3 is worse than C0 on 126 of 200 utterances and better on 34.
   - So intelligibility metrics against R1 cannot stand in for the ASR endpoint. That supports reporting both and keeping ΔWER primary.
3. **WPE-1ch is neutral.** It is within the CI of C0 everywhere and raises
   ESTOI slightly. It is the only C1 system that does no harm. That makes it
   the honest "classical" baseline row.
4. **Reverberation alone barely hurts Whisper in this sim.** Quiet C0 goes
   from 3.1% to 4.3% between 1 m and 6 m, while ESTOI falls from 0.80 to 0.36.
   Either the placeholder boundaries are too absorptive, or Whisper is very
   robust to this kind of reverb. The measured pool RIRs (E0–E4) decide which.
   If the real pool is this benign in quiet conditions, the paper's story moves
   to noise (thrusters) rather than reverberation, and H9b's usable-region
   framing becomes the main figure.
5. **Hallucinations are mostly fluent substitutions, not loops the §8.3 flags catch.**
   - sepformer_wham16k at quiet 6 m has WER 110% but only 14 of 200 flagged.
   - Typical outputs are long, fluent, unrelated sentences, e.g. "So I'm going to take a picture of you…". There are also YouTube-style phrases, e.g. "Satsang with Mooji…".
   - The length-ratio and compression-ratio flags miss the substitution-dominated cases.
   - E14 (RQ7) needs an additional flag, e.g. low word overlap with the reference plus a high avg_logprob. The current flag count should not be read as a hallucination rate.
6. **SepFormer-wham16k and MetricGAN+ break down off-distribution.** Both are
   catastrophic by 3 m even at 30 dB SNR. They are useful as "what in-air SE
   does underwater" rows, but should not anchor any comparison.

## Sanity checks and known artefacts

- **CI-SDR of R1 is about 173 dB** because R1 is scored against itself; ignore that cell.
- **`clean` ESTOI below 1 and CI-SDR below 0 dB** are expected. R1 includes the chain response (band-limiting and EQ), so the digital clean signal is not identical to it.
- **WER above 100%** comes from insertions (corpus WER = (S+D+I)/N_ref).
- **R1 vs clean (2.1–2.2 vs 2.4%)** is inside the CI. The chain alone does not hurt Whisper.
- **Repeat transcriptions match exactly.** The 2243 transcripts from the first, crashed run were re-decoded after the fp16-weight change and matched exactly (8/8 checked).
- **Synthetic noise at 0 dB SNR is harsh and spectrally unrealistic.** Absolute noise0 WERs mean little until the recorded noise bank (thrusters, pumps, bubbles) replaces it.

## Addendum (2026-10-29): C10, Kiel-recipe U-Net trained on uwsim

- **Model.** `runs/c10_kiel_unet_s0/best.pt`, trained from scratch on uwsim v0 with `configs/c10_kiel_unet.yaml` and `data.noise_root=null`:
  - The RIR bank is `data/rirs/uwsim_v0` (20k RIRs, uncalibrated), with synthetic noise in place of a recorded bank.
  - Training SNR was drawn from −10 to 25 dB, with thrusters on in 70% of items.
  - Early stopping hit at step 24k. The best validation loss was 0.0912 at step 14k (1.5M audio-seconds seen).
  - Training was data-bound: 74 audio-s per wall-second, about 89% of it spent in data loading.
- **Evaluation.** The model enhanced the cached 16 kHz `noisy` wavs (`scripts/sim_baseline.py --stages enhance`), then was scored with the same harness.

WER and ESTOI against R1:

| condition | C0 noisy WER % | C10 kiel_unet WER % | C10 better / same / worse (utts) | C0 ESTOI | C10 ESTOI |
|---|---|---|---|---|---|
| quiet 1 m | 3.1 [2.5, 3.8] | 2.6 [1.9, 3.4] | 30 / 151 / 19 | 0.798 | 0.868 |
| quiet 3 m | 3.7 [2.9, 4.6] | 5.5 [4.5, 6.6] | 25 / 117 / 58 | 0.508 | 0.701 |
| quiet 6 m | 4.3 [3.4, 5.4] | 8.0 [6.6, 9.6] | 14 / 102 / 84 | 0.364 | 0.560 |
| noise0 1 m | 20.3 [16.6, 24.6] | 76.1 [67.6, 86.9] | 3 / 11 / 186 | 0.310 | 0.311 |
| noise0 3 m | 48.3 [42.9, 54.0] | 97.9 [91.9, 105.6] | 12 / 10 / 178 | 0.230 | 0.239 |
| noise0 6 m | 56.4 [50.7, 62.4] | 100.7 [96.8, 105.2] | 11 / 27 / 162 | 0.168 | 0.200 |

Reading:

- **Dereverberation works on the metric it was trained for.** In quiet conditions, ESTOI rises a lot (0.36 → 0.56 at 6 m), and this is the largest ESTOI gain of any system in the table.
- **WER still gets worse with range.** Even in quiet, C10 is worse than C0 at 3 m and 6 m. Finding 2 above applies even to a model trained in-domain: an IRM-masking network optimized toward R1 does not, on its own, give Whisper anything it can use.
- **The 0 dB condition is a failure.**
  - ESTOI is flat, so the model barely denoises.
  - WER triples at 1 m. Most of the increase is substitutions (1950 S vs 540).
  - Two explanations remain open:
    - (a) the training noise mix (mostly thruster, plus bubbles) does not cover the test's −3 dB/oct synthetic noise;
    - (b) masking artefacts at low SNR hurt Whisper more than residual noise does.
  - A matched-noise re-score separates (a) from (b), and needs no retraining.
- **This is one seed of a from-scratch recipe** whose optimizer settings are our §7.4 defaults, not Kiel's (see the config header). Early stopping on the noisy validation loss (patience 5 × 2k steps) cut it at about half the planned 50k steps.

## Next

- **C2 fine-tuning.** Run kiel_unet (C10) first, as the cheap pipeline check, then dfn3 C2 on uwsim. Re-score on this same harness to get the first sim ΔWER(C2 − C0).
- **Hallucination flag.** Add a substitution-hallucination flag to `evaluation.asr` before E14.
- **Calibration.** Replace the placeholders with measured pool values (E0–E4) and the recorded noise bank, then regenerate this table. The current table is retired at that point.
