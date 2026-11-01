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
  - Training SNR was drawn from −10 to 25 dB. Thrusters are on in 70% of scenes, but with no noise bank the simulator skips thruster and ambient noise, so the training noise is almost entirely synthetic Minnaert bubble trains (see the C2 addendum below).
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
    - (a) the training noise (bubble trains only, see below) does not cover the test's stationary −3 dB/oct synthetic noise;
    - (b) masking artefacts at low SNR hurt Whisper more than residual noise does.
  - A matched-noise re-score separates (a) from (b), and needs no retraining.
- **This is one seed of a from-scratch recipe** whose optimizer settings are our §7.4 defaults, not Kiel's (see the config header). Early stopping on the noisy validation loss (patience 5 × 2k steps) cut it at about half the planned 50k steps.

## Addendum (2026-10-30): C2, DeepFilterNet3 fully fine-tuned on uwsim

- **Model.** `runs/c2_dfn3_s0/best.pt`: dfn3 fully fine-tuned with `configs/c2_dfn3.yaml` and `data.noise_root=null`, seed 0.
  - It ran all 50k steps (3.2M audio-seconds). The best validation loss was 1.2245 at step 48k.
  - Validation was flat from 42k (1.2249) onward, so the run has converged. More steps are not needed.
- **Evaluation.** Labelled `dfn3_c2` (`--adapter dfn3_c2=dfn3 --ckpt dfn3_c2=...`).
  - It is fed the 48 kHz simulator output, re-simulated per item with the same RNG, exactly as zero-shot `dfn3` was.
  - Spot check on 6 items: the re-simulated noisy matched the cache within 4e-7 of peak, and zero-shot dfn3 run through the new path matched its cached output within 5e-4 (CPU vs GPU).

WER % [95% CI] and ESTOI against R1:

| condition | C0 noisy | dfn3 zero-shot | dfn3_c2 | C2 vs C0 better / same / worse (utts) | ESTOI C0 / dfn3 / C2 |
|---|---|---|---|---|---|
| quiet 1 m | 3.1 [2.5, 3.8] | 2.4 [1.7, 3.2] | 2.3 [1.7, 3.1] | 27 / 159 / 14 | 0.798 / 0.836 / 0.864 |
| quiet 3 m | 3.7 [2.9, 4.6] | 5.8 [4.6, 7.1] | 6.0 [4.9, 7.2] | 16 / 115 / 69 | 0.508 / 0.578 / 0.698 |
| quiet 6 m | 4.3 [3.4, 5.4] | 13.3 [10.1, 16.8] | 9.2 [7.7, 11.0] | 15 / 86 / 99 | 0.364 / 0.413 / 0.606 |
| noise0 1 m | 20.3 [16.6, 24.6] | 32.0 [27.6, 37.2] | 41.5 [36.3, 47.3] | 22 / 26 / 152 | 0.310 / 0.497 / 0.468 |
| noise0 3 m | 48.3 [42.9, 54.0] | 81.9 [70.9, 96.1] | 87.3 [78.4, 99.1] | 19 / 15 / 166 | 0.230 / 0.316 / 0.367 |
| noise0 6 m | 56.4 [50.7, 62.4] | 90.8 [82.4, 102.3] | 93.6 [86.2, 103.3] | 13 / 22 / 165 | 0.168 / 0.228 / 0.306 |

Reading:

- **Fine-tuning helps dfn3 in quiet conditions, but not enough to beat C0 at range.** At quiet 6 m, WER drops from 13.3% (zero-shot) to 9.2% and ESTOI rises from 0.41 to 0.61. That is still about twice C0's 4.3%. At 1 m, C2 ties C0 within the CI.
- **The first sim ΔWER(C2 − C0) is positive (worse) in 5 of 6 cells.** Only quiet 1 m is a point-estimate gain, and it overlaps C0.
  - The same pattern holds for C10. Two different architectures, both trained toward R1, raise ESTOI a lot and still lose on Whisper.
  - This is the in-air "SE hurts robust ASR" finding reproduced in-domain. It argues for C8 (Whisper fine-tuning) and joint or ASR-aware SE objectives over SE-only training.
- **The noise0 rows are probably a train/test noise mismatch, not a property of the method.** With `noise_root=null`, the simulator skips thruster and ambient noise (both need a bank). The training noise is therefore Minnaert bubble trains with rate U(0, 20)/s. The stationary −3 dB/oct fallback only appears when an item draws no bubbles. The noise0 test uses that stationary fallback. Both C10 and C2 were trained almost without stationary broadband noise.
  - A `bubbles0` test condition (a matched-noise re-score) separates this from masking artefacts. It is the next run.
  - Until the recorded bank exists, the training config should also mix in stationary synthetic noise. That is a design change and needs sign-off.
- **One seed each.** The plan calls for three seeds for C2.

## Addendum (2026-10-30): matched-noise check (`bubbles0`)

- **Condition.** `bubbles0` = 0 dB speech-band SNR of Minnaert bubble trains (rate 10/s, radii 0.5–3 mm). This is the noise C10 and C2 trained on, since no bank exists.
- **Scope.** Same scenes, utterances and harness as above. Systems: C0, wpe1, zero-shot dfn3, dfn3_c2 and kiel_unet. The SepFormer and MetricGAN+ rows were not run.

WER % [95% CI], with better / same / worse vs C0 in utterances, and ESTOI against R1:

| system | bubbles0 1 m | bubbles0 3 m | bubbles0 6 m | ESTOI 1 / 3 / 6 m |
|---|---|---|---|---|
| C0 noisy | **2.4** [1.8, 3.2] | 5.1 [4.1, 6.4] | 5.7 [4.5, 7.1] | 0.583 / 0.392 / 0.284 |
| wpe1 | **2.4** [1.8, 3.2] (3/195/2) | **4.2** [3.2, 5.3] (27/167/6) | **5.3** [4.2, 6.6] (10/186/4) | 0.586 / 0.397 / 0.288 |
| dfn3 zero-shot | 3.5 [2.6, 4.6] (14/148/38) | 27.7 [22.5, 33.1] (12/33/155) | 50.7 [42.0, 60.2] (3/14/183) | 0.725 / 0.448 / 0.302 |
| dfn3_c2 | 2.6 [1.9, 3.5] (8/181/11) | 7.0 [5.8, 8.2] (32/108/60) | 9.9 [8.3, 11.6] (25/93/82) | 0.833 / 0.670 / 0.575 |
| kiel_unet (C10) | 2.6 [1.9, 3.6] (10/178/12) | 5.9 [4.8, 7.1] (37/112/51) | 8.6 [7.0, 10.4] (23/105/72) | 0.843 / 0.684 / 0.558 |

R1 is 2.1–2.2% and clean 2.4% in every row.

Reading:

- **The noise0 collapse was a noise mismatch (explanation (a)).**
  - In matched noise, C2 and C10 no longer collapse. At 1 m both are within the CI of C0 (2.6 vs 2.4%), whereas at noise0 1 m they were at 41.5% and 76.1%.
  - Both models learned to remove bubble trains, never stationary broadband noise, and both mangle speech when given the latter.
- **Matched noise still gives no WER gain.** At 3 m and 6 m, C2 and C10 are worse than C0 by about the same margin as in quiet (C2 6 m: 9.9 vs 5.7 here, 9.2 vs 4.3 in quiet). What remains is the range-dependent penalty from dereverberation and masking artefacts, not a noise effect.
  - So (b) is real but range-driven. It shows up at 3 m and 6 m in every noise condition.
- **Bubble noise at 0 dB barely hurts Whisper.** C0 is 2.4 / 5.1 / 5.7%, close to quiet (3.1 / 3.7 / 4.3%). Sparse impulsive noise at a given speech-band SNR is far easier than stationary noise.
  - This makes the current no-bank training set a weak denoising task. It is a second reason not to train further without stationary noise or the recorded bank.
- **Zero-shot dfn3 is far worse on bubbles than fine-tuned dfn3** (27.7 → 7.0% at 3 m). Fine-tuning does fix the in-domain noise. It just cannot beat Whisper's own robustness.
- **WPE stays the only system that never hurts.** It is now nominally best at 3 m and 6 m, though inside C0's CI.

## Addendum (2026-11-01): C8, Whisper large-v3 LoRA trained on uwsim

- **Training.** `configs/c8_whisper_lora.yaml` with LoRA r 8, alpha 32, on q/v projections.
  - Data: train-clean-100 clips of 30 s or less, through uwsim with no noise bank, so the noise is bubble trains as for C2 and C10.
  - LR 1e-4. The config default of 1e-3 diverged near step 900 in run 1; the change still needs sign-off.
  - Run dir `runs/c8_whisper_lora_lr1e-4_s0`. Early stop at step 16k. Best val CE 0.1018 at step 6k; val stayed between 0.102 and 0.111 throughout.
- **Scoring.** The best LoRA is merged into openai-whisper large-v3 in fp32 before the fp16 cast. Decoding is the same fixed policy as C0: beam 5, temperature 0 with no fallback.
  - C8 decodes the unprocessed noisy audio. Its ESTOI and SI-SDR therefore equal C0's and are omitted.
  - Same 200 utterances and scenes as above.

WER %, C0 → C8, with the paired bootstrap ΔWER (C8 − C0) and its 95% CI (2000 resamples over utterances):

| condition | 1 m | 3 m | 6 m |
|---|---|---|---|
| quiet | 3.1 → **2.0**, −1.1 [−1.9, −0.3] | 3.7 → **2.3**, −1.4 [−2.1, −0.8] | 4.3 → **3.7**, −0.7 [−1.5, +0.2] |
| noise0 | 20.3 → **17.1**, −3.2 [−4.9, −1.7] | 48.3 → **42.1**, −6.2 [−8.7, −3.5] | 56.4 → **48.8**, −7.6 [−10.4, −4.7] |
| bubbles0 | 2.4 → 7.9, +5.5 [−0.9, +19.0] | 5.1 → **3.0**, −2.1 [−3.0, −1.4] | 5.7 → **4.4**, −1.4 [−2.2, −0.5] |

Reading:

- **C8 is the first system that beats C0, and it does so in 8 of 9 cells.** The CI excludes zero in 7 of them; quiet 6 m is the exception, at −0.7 [−1.5, +0.2].
  - It is also best overall in every cell except bubbles0 1 m. Every SE system above loses to C0 at 3 m and 6 m.
  - At 3 m in quiet it reaches 2.3%, which is the R1 level (2.2%).
- **The gain grows with difficulty in noise0**, from −3.2 points at 1 m to −7.6 at 6 m. This holds even though C8 never trained on stationary noise.
  - Adapting the recognizer to reverberation and range transfers to a noise type it did not see.
  - SE trained on the same data instead collapsed on that noise (C2 and C10).
- **The one regression is a single hallucination loop.** Utterance 5639-40744-0011 in bubbles0 1 m is 9 words long.
  - C8 transcribed it correctly, then kept generating: "...locked outsideSuspects were arrested on the day of the murder of a young man who had been arrested...". That adds 219 insertions, and the length/compression flag catches it.
  - Without that one utterance, C8 is at 2.0% in that cell.
  - C8 has 2 flagged utterances out of 1800 scored, against 1 for C0.
  - LoRA fine-tuning can weaken end-of-text prediction. Because the decoding policy has no temperature fallback, such a loop goes straight into the WER. Watch the flag count on the real v0 test.
- **H2 reading under the placeholder simulator.** For Whisper at range, adapting the recognizer beats every front end tried. The caveats above still apply: uncalibrated RIRs, no recorded noise, and one seed.

## Next

- **Training noise (needs sign-off).** Until the recorded bank exists, mix stationary synthetic noise (and coloured variants) into training. Today it only appears when an item draws no bubbles. Without it, any SE trained on uwsim fails on broadband noise.
- **C8 follow-ups.**
  - Get sign-off on LR 1e-4.
  - Train more seeds.
  - Retrain with stationary noise once that is approved, since noise0 is still 17–49% WER.
  - Test the hallucination rate with temperature fallback against the fixed policy.
  - Use `train.val_num_workers=0` on the 16 GB laptop: validation workers spawn on every eval.
- **Hallucination flag.** Add a substitution-hallucination flag to `evaluation.asr` before E14.
- **Calibration.** Replace the placeholders with measured pool values (E0–E4) and the recorded noise bank, then regenerate this table. The current table is retired at that point.
