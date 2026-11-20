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
  - LR 1e-4. The config default of 1e-3 diverged near step 900 in run 1. 1e-4 was signed off by the lead on 2026-11-05.
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

### C8 seed 1 (2026-11-02)

- **Run.** Same config as seed 0 except `val_num_workers=0`. Run dir `runs/c8_whisper_lora_lr1e-4_s1`.
  - Early stop at step 22k. Best val CE 0.0973 at step 12k (seed 0: 0.1018 at 6k).
- **Scoring.** Same harness as seed 0.

WER %, C0 / seed 0 / seed 1, then the seed-1 paired ΔWER vs C0 and vs seed 0 [95% CI]:

| cell | C0 | s0 | s1 | s1 − C0 | s1 − s0 |
|---|---|---|---|---|---|
| quiet 1 m | 3.1 | 2.0 | 2.7 | −0.4 [−1.1, +0.3] | +0.7 [0.0, +1.4] |
| quiet 3 m | 3.7 | 2.3 | 3.8 | +0.1 [−0.6, +0.8] | +1.5 [+0.9, +2.1] |
| quiet 6 m | 4.3 | 3.7 | 4.8 | +0.5 [−0.5, +1.5] | +1.2 [+0.3, +2.1] |
| noise0 1 m | 20.3 | 17.1 | 18.1 | **−2.2** [−3.8, −0.7] | +1.0 [−0.3, +2.3] |
| noise0 3 m | 48.3 | 42.1 | 43.2 | **−5.1** [−7.7, −2.5] | +1.1 [−1.2, +3.4] |
| noise0 6 m | 56.4 | 48.8 | 50.1 | **−6.3** [−9.2, −3.6] | +1.3 [−0.3, +3.0] |
| bubbles0 1 m | 2.4 | 7.9 | 2.0 | −0.4 [−0.9, 0.0] | −5.9 [−18.7, +0.4] |
| bubbles0 3 m | 5.1 | 3.0 | 3.4 | **−1.7** [−2.5, −1.0] | +0.4 [0.0, +0.9] |
| bubbles0 6 m | 5.7 | 4.4 | 4.5 | **−1.2** [−2.1, −0.3] | +0.2 [−0.4, +0.8] |

Reading:

- **The noisy-cell gain replicates.**
  - Seed 1 beats C0 in all 6 noise0 and bubbles0 cells. The CI excludes zero in 5 of them; bubbles0 1 m reaches 0.0.
  - The noise0 gain still grows with range, from −2.2 to −6.3 points.
  - Seed 1 is about 1 point behind seed 0 in noise0, inside the seed-vs-seed CI.
- **The quiet-cell gain does not replicate.**
  - Seed 1 is level with C0 in quiet; every CI includes zero.
  - It is significantly worse than seed 0 at 3 m and 6 m.
  - So the 8-of-9 claim for seed 0 should be read as 6 of 6 noisy cells (5 of 6 significant per seed), plus a quiet-cell gain that depends on the seed.
- **The hallucination loop does not recur.** Seed 1 has no runaway insertions; its maximum is 13 words in one utterance. It has 2 flagged utterances of 1800, both in noise0, the same count as seed 0.
  - The seed-0 bubbles0 1 m regression is therefore one stochastic event, not a systematic failure. Its rate still needs watching on real data.
- **Val CE does not pick the better recognizer.**
  - Seed 1 has the lower val CE (0.097 vs 0.102) but the higher WER in 8 of 9 cells. Only bubbles0 1 m, the cell with the seed-0 loop, goes the other way.
  - Its best checkpoint is also later (step 12k vs 6k), and the quiet-cell loss may be drift away from the base model's clean-speech behaviour as training goes on.
  - If seeds or checkpoints are compared, select on a small WER dev set, not CE.

### C8 with stationary training noise (2026-11-03)

- **Run.** `configs/c8_whisper_lora_stat.yaml`, run dir `runs/c8_whisper_lora_stat_s0`, seed 0, LR 1e-4. Half of the training scenes add stationary noise with the (−4, −2) dB/oct slope band held out (see Next).
  - Ran all 30k steps without early stopping. Best val CE 0.2066 at step 28k.
  - That CE is not comparable to the 0.10 of the earlier runs, because the val mix now contains stationary noise too.
- **Scoring.** Same harness and cached sim audio as seeds 0 and 1.

WER %. "excl. loops" drops utterances with a runaway insertion loop, defined as insertions ≥ max(10, 2 × reference length).

| cell | C0 | s0 | s1 | stat | stat, excl. loops | loop utts (of 200) |
|---|---|---|---|---|---|---|
| quiet 1 m | 3.1 | 2.0 | 2.7 | 92.7 | 1.9 | 16 |
| quiet 3 m | 3.7 | 2.3 | 3.8 | 59.7 | 2.7 | 10 |
| quiet 6 m | 4.3 | 3.7 | 4.8 | 48.1 | 3.2 | 8 |
| noise0 1 m | 20.3 | 17.1 | 18.1 | **16.0** | 16.0 | 0 |
| noise0 3 m | 48.3 | 42.1 | 43.2 | **37.2** | 37.2 | 0 |
| noise0 6 m | 56.4 | 48.8 | 50.1 | **44.4** | 44.4 | 0 |
| bubbles0 1 m | 2.4 | 7.9 | 2.0 | 24.7 | 1.7 | 4 |
| bubbles0 3 m | 5.1 | 3.0 | 3.4 | 14.2 | 3.2 | 2 |
| bubbles0 6 m | 5.7 | 4.4 | 4.5 | 40.0 | 4.1 | 6 |

Reading:

- **Stationary training noise helps on the held-out slope.** noise0 WER is the best of any system in all three cells, with no loops.
  - The gain over C0 is −4.3, −11.1 and −12.0 points at 1, 3 and 6 m, against −3.2 to −7.6 for seed 0.
  - The test slope (−3 dB/oct) is outside the training slopes, but the generator is the same family, so this is weaker evidence than an unseen noise type.
- **It fails to stop in quiet and bubble audio.** 46 of 1200 quiet and bubbles0 decodes have a runaway loop, against 1 of 2400 for the two earlier seeds.
  - In every case inspected, the model transcribes the utterance correctly and then fails to emit end-of-text. It appends unrelated invented text, e.g. "...flour fattened sauceYou're going to have to go to the doctor's...", 188 to 222 inserted words.
  - Loop utterances are longer than average: median 28 reference words vs 17.
  - Substitutions and deletions are level with the other C8 runs, so the acoustic gain is intact. Excluding loop utterances, stat is the best or level-best system in 8 of 9 cells.
- **Probable cause, not yet tested.**
  - This run trained 2.5 to 5 times longer than seeds 0 and 1 (best step 28k vs 6k and 12k) and never stopped early, so drift from the base model's end-of-text behaviour is plausible.
  - A second cause could be that the end of the audio in training clips, where stationary noise is usually present, is unlike the near-silent clip ends of quiet audio.
  - Only `best.pt` (28k) and `last.pt` (30k) were kept, so an earlier checkpoint cannot be scored without retraining.
- **As it stands, this checkpoint is not usable.** Fix the loop before taking stat forward. Options:
  1. Retrain with intermediate checkpoints kept, choosing on WER plus loop rate on a dev set.
  2. Cap decode length by audio duration, or add a repetition/no-repeat-ngram guard.
  3. Train with fewer steps or an end-of-text-weighted loss.

### Guarded decoding (2026-11-03)

The lead chose option 2, a decoding guard with no retraining, applied identically to every compared system.

- **Policy.** `DecodeConfig.guarded()` (`--decode guard`):
  - Whisper's own temperature fallback: 0 → 0.2 → … → 1.0, beam 5 at 0, best-of-5 sampling above. A retry fires when the compression ratio is > 2.4 or the mean logprob is < −1.0. Sampling is seeded.
  - A decode-length cap of ceil(8 tokens/s × duration) + 16 tokens, at most 224. Correct transcripts of these clips run at a median 3.7 and a maximum 6.2 tokens/s.
  - Transcripts are cached under their own key (`+guard`), and scores go to `summary_guard.*` and `per_utt_guard.jsonl`. The fixed-policy (§8.3) results above are unchanged.
- **Scope.** C0 (`noisy`), the three C8 adapters, and every enhancement front end (see "Guarded decoding: enhancement front ends" below). The SE vs recognizer-adaptation comparison therefore uses one policy throughout.

WER %, guarded decoding:

| cell | C0 | s0 | s1 | stat | loops (all systems) |
|---|---|---|---|---|---|
| quiet 1 m | 3.1 | 2.0 | 2.7 | **1.7** | 0 |
| quiet 3 m | 3.7 | 2.3 | 3.8 | 2.6 | 0 |
| quiet 6 m | 4.3 | 3.7 | 4.8 | **3.2** | 0 |
| noise0 1 m | 20.3 | 17.1 | 18.1 | **16.0** | 0 |
| noise0 3 m | 48.4 | 42.1 | 43.2 | **37.2** | 0 |
| noise0 6 m | 56.4 | 48.8 | 50.1 | **44.5** | 0 |
| bubbles0 1 m | 2.4 | 2.0 | 2.0 | **1.7** | 0 |
| bubbles0 3 m | 5.1 | 3.0 | 3.4 | 3.2 | 0 |
| bubbles0 6 m | 5.7 | 4.4 | 4.5 | **4.1** | 0 |

Paired ΔWER [95% CI], speaker-cluster bootstrap with 2000 resamples (`tmp/dbg_guard_paired.py`):

| cell | stat − C0 | stat − s0 | stat − s1 |
|---|---|---|---|
| quiet 1 m | **−1.4** [−1.9, −0.9] | −0.3 [−0.9, +0.3] | **−1.0** [−1.6, −0.5] |
| quiet 3 m | **−1.1** [−1.8, −0.5] | +0.3 [−0.0, +0.7] | **−1.2** [−1.9, −0.5] |
| quiet 6 m | **−1.2** [−1.9, −0.4] | −0.5 [−1.2, +0.2] | **−1.6** [−2.5, −0.8] |
| noise0 1 m | **−4.3** [−6.5, −2.4] | −1.1 [−2.6, +0.3] | **−2.1** [−3.8, −0.4] |
| noise0 3 m | **−11.3** [−13.7, −9.0] | **−4.9** [−7.0, −2.9] | **−6.0** [−8.0, −3.8] |
| noise0 6 m | **−11.9** [−14.8, −8.8] | **−4.3** [−6.4, −2.1] | **−5.7** [−8.1, −2.8] |
| bubbles0 1 m | **−0.8** [−1.3, −0.3] | −0.3 [−0.8, +0.0] | **−0.3** [−0.7, −0.1] |
| bubbles0 3 m | **−2.0** [−2.9, −1.2] | +0.2 [−0.2, +0.6] | −0.2 [−0.7, +0.2] |
| bubbles0 6 m | **−1.7** [−2.6, −0.9] | −0.3 [−1.0, +0.4] | −0.5 [−1.2, +0.2] |

Reading:

- **The guard removes every loop and costs C0 nothing.**
  - Loops fell from 46 (stat) and 1 (seed 0) to 0 in all 36 system × cell pairs.
  - The temperature fallback fired on only 3 of 7200 decodes, so the length cap did the work. A looping decode is cut shortly after the real transcript ends.
  - C0 moved by at most 0.1 points (noise0 3 m: 48.3 → 48.4). Seed 0's bubbles0 1 m regression (7.9%) becomes 2.0%. No other seed-0 or seed-1 cell changes.
- **C8 with stationary training noise is the best system.**
  - It beats C0 in all 9 cells, and every CI excludes zero.
  - It beats seed 0 by about 4–5 points at noise0 3 m and 6 m, and is level with it elsewhere.
  - Against seed 1 it is better in 7 of 9 cells, including all of quiet.
- **Caveats.**
  - This is one seed. Seed 0 vs seed 1 showed a quiet-cell difference of about 1.5 points, about the same size as stat's quiet advantage over seed 1.
  - The test noise slope is held out, but it comes from the same generator (see Next).
  - The best checkpoint is from step 28k; val CE does not select checkpoints well here (see seed 1).
- **Policy for later evaluations.** Use the guarded decode for every ASR result from now on, including real-data E7. Pre-register it before the real data arrive, because it was chosen after seeing sim results.

### Guarded decoding: enhancement front ends (2026-11-04)

The same guarded policy applied to every front end (`e7_guard_se`): 10,800 decodes, 18,000 guarded decodes in all.

WER %, guarded decoding. The change from the fixed policy is in brackets.

| cell | C0 | WPE | DFN3 0-shot | DFN3 C2 | U-Net C10 | MetricGAN+ | SepF wham16k | SepF whamr | C8 stat |
|---|---|---|---|---|---|---|---|---|---|
| quiet 1 m | 3.1 | 3.0 | 2.4 | 2.3 | 2.6 | 3.5 | 6.0 | 3.6 | **1.7** |
| quiet 3 m | 3.7 | 3.4 | 5.8 | 6.0 | 5.5 | 38.0 | 84.7 (−6.8) | 20.7 | **2.6** |
| quiet 6 m | 4.3 | 4.4 | 13.3 | 9.2 | 8.0 | 43.3 | 105.9 (−4.4) | 43.0 | **3.2** |
| noise0 1 m | 20.3 | 21.1 | 32.0 | 41.6 | 71.0 (−5.1) | 72.7 | 50.3 (−4.4) | 42.2 | **16.0** |
| noise0 3 m | 48.4 | 48.3 | 74.2 (−7.7) | 81.6 (−5.7) | 94.0 (−4.0) | 105.4 (−14.7) | 100.3 (−6.0) | 85.0 (−12.7) | **37.2** |
| noise0 6 m | 56.4 | 56.9 (−5.5) | 86.1 (−4.6) | 89.9 (−3.7) | 96.7 (−4.0) | 106.5 (−17.7) | 101.6 | 96.0 | **44.5** |
| bubbles0 1 m | 2.4 | 2.4 | 3.5 | 2.6 | 2.6 | – | – | – | **1.7** |
| bubbles0 3 m | 5.1 | 4.2 | 27.7 | 7.0 | 5.9 | – | – | – | **3.2** |
| bubbles0 6 m | 5.7 | 5.3 | 49.5 (−1.2) | 9.9 | 8.6 | – | – | – | **4.1** |

Unmarked cells moved by less than 0.5 points. MetricGAN+ and the SepFormers were not run on bubbles0.

Reading:

- **The guard helps the front ends where their output is garbled, and changes nothing else.**
  - The large drops (4–18 points) are all in cells where a front end's WER was already 50–125%. Long hallucinated decodes of badly damaged audio are now cut by the length cap.
  - The fallback fired on 263 of 18,000 decodes. 260 were on front-end output (201 noise0, 55 quiet, 4 bubbles0) and 3 on C0/C8 in noise0.
  - 6 loop-sized insertions remain, all in front-end noise0 outputs at 3 m and 6 m. They are capped there rather than removed: the audio itself carries little speech.
- **The conclusion is unchanged: no front end beats C0, and C8 stat beats every front end in every cell.**
  - WPE is level with C0 everywhere (best front end in noise0: 21.1 / 48.3 / 56.9 at 1 / 3 / 6 m).
  - At 6 m, WPE's noise0 regression under the fixed policy (62.4%) was itself a decoding artefact; it is 56.9% under the guard.
  - The neural front ends still raise WER in every noise0 cell (32–107%) and at quiet 3–6 m.

### C8 stat seed 1 with dev-WER checkpoint selection (2026-11-05)

- **Run.** Same config as stat seed 0 with seed 1 and `keep_every: 4000` (`runs/c8_whisper_lora_stat_s1`).
  - Stopped early at step 30k (patience 5); best val CE 0.2076 at step 20k.
- **Selection.** `scripts/dev_wer_select.py`, guarded decode, on a uwsim dev set that shares no utterances and no scene geometry with E7.
  - 120 dev-clean utterances, ranges 1.5 / 3.5 / 5.5 m, depth 1.5 m, receiver at 0.35 of the pool width, seed 20261104.
  - Four conditions: quiet, bubbles0, and stationary noise at 0 dB SNR with slopes −5 and −1 dB/oct. Both slopes are in the training bands, so the held-out test band is never used for selection.
  - Criterion: the lowest mean of the per-condition corpus WERs.

Dev WER %, guarded decoding (`runs/c8_whisper_lora_stat_s1/dev_wer/summary.md`). There were no loops at any checkpoint.

| checkpoint | quiet | bubbles0 | stat −5 | stat −1 | mean |
|---|---|---|---|---|---|
| base (C0) | 3.4 | 3.5 | 50.3 | 33.3 | 22.65 |
| step 4k | 3.3 | 3.6 | 40.8 | 25.9 | 18.39 |
| step 8k | 3.2 | 3.1 | 39.6 | 25.9 | 17.95 |
| step 12k | 3.1 | 3.1 | 39.7 | 25.8 | 17.93 |
| step 16k | 3.3 | 3.1 | 41.6 | 25.9 | 18.46 |
| step 20k (CE-best) | 4.3 | 3.7 | 40.0 | 26.8 | 18.69 |
| step 24k | 3.5 | 3.1 | 38.7 | 25.3 | 17.67 |
| step 28k | 3.3 | 3.2 | 39.0 | 25.2 | 17.69 |
| **step 30k (selected)** | 3.3 | 3.2 | 38.6 | 25.5 | **17.64** |

- Dev-WER picked step 30k. The CE-best checkpoint (step 20k) has the worst dev WER after step 4k, mainly from quiet (4.3 vs 3.1–3.5).
- From step 8k on, the mean varies by only about 1 point. Steps 24k, 28k and 30k are within 0.05 points of each other, so the exact pick among them is noise.

E7 WER %, guarded decoding. stat1 = seed 1 at the dev-WER pick, stat1-CE = seed 1 at the CE-best checkpoint.

| cell | C0 | stat (s0) | stat1 | stat1-CE |
|---|---|---|---|---|
| quiet 1 m | 3.1 | 1.7 | 2.3 | 3.3 |
| quiet 3 m | 3.7 | 2.6 | 2.7 | 3.0 |
| quiet 6 m | 4.3 | 3.2 | 3.4 | 3.5 |
| noise0 1 m | 20.3 | 16.0 | 16.4 | 16.5 |
| noise0 3 m | 48.4 | 37.2 | 38.2 | 38.3 |
| noise0 6 m | 56.4 | 44.5 | 43.8 | 45.2 |
| bubbles0 1 m | 2.4 | 1.7 | 2.2 | 2.4 |
| bubbles0 3 m | 5.1 | 3.2 | 3.1 | 3.3 |
| bubbles0 6 m | 5.7 | 4.1 | 4.0 | 4.8 |

There were no loops in either seed-1 checkpoint. The fallback fired on 2 of 3600 seed-1 decodes, both stat1 at bubbles0 6 m.

Paired ΔWER [95% CI], speaker-cluster bootstrap with 2000 resamples (`tmp/dbg_guard_paired_s1.py`). Bold CIs exclude zero.

| cell | stat1 − C0 | stat1-CE − C0 | stat1 − stat1-CE | stat1 − stat (s0) | stat1 − bubble s0 |
|---|---|---|---|---|---|
| quiet 1 m | **−0.8** [−1.4, −0.3] | +0.2 [−0.6, +1.0] | **−1.0** [−1.5, −0.6] | **+0.6** [+0.3, +0.9] | +0.3 [−0.4, +0.9] |
| quiet 3 m | **−0.9** [−1.6, −0.3] | −0.7 [−1.5, +0.1] | −0.2 [−0.6, +0.2] | +0.2 [−0.3, +0.6] | +0.5 [+0.0, +0.9] |
| quiet 6 m | **−1.0** [−1.7, −0.2] | **−0.8** [−1.6, −0.0] | −0.2 [−0.5, +0.1] | +0.2 [−0.3, +0.7] | −0.3 [−1.0, +0.3] |
| noise0 1 m | **−4.0** [−5.7, −2.3] | **−3.8** [−5.6, −2.3] | −0.1 [−0.9, +0.6] | +0.3 [−0.6, +1.5] | −0.7 [−1.9, +0.4] |
| noise0 3 m | **−10.2** [−12.8, −7.9] | **−10.2** [−12.9, −7.6] | −0.1 [−1.0, +1.0] | +1.0 [−0.9, +2.9] | **−3.9** [−6.1, −1.8] |
| noise0 6 m | **−12.6** [−15.5, −9.6] | **−11.2** [−14.0, −8.3] | **−1.4** [−2.5, −0.5] | −0.7 [−2.5, +1.1] | **−5.0** [−6.9, −3.2] |
| bubbles0 1 m | −0.2 [−0.7, +0.2] | −0.1 [−0.5, +0.4] | −0.2 [−0.4, +0.1] | **+0.6** [+0.2, +0.9] | +0.2 [−0.3, +0.7] |
| bubbles0 3 m | **−2.0** [−3.0, −1.1] | **−1.9** [−2.9, −0.9] | −0.1 [−0.4, +0.2] | −0.1 [−0.5, +0.4] | +0.1 [−0.3, +0.6] |
| bubbles0 6 m | **−1.7** [−2.6, −0.9] | **−0.9** [−1.8, −0.1] | **−0.8** [−1.3, −0.3] | −0.0 [−0.7, +0.6] | −0.3 [−0.8, +0.2] |

Reading:

- **The stationary-noise gain replicates.**
  - On noise0, seed 1 is level with seed 0 in all three cells (every CI includes zero) and beats C0 by 4.0 / 10.2 / 12.6 points at 1 / 3 / 6 m.
  - Its margin over the bubble-only seed 0 at 3 m and 6 m (−3.9 and −5.0) matches seed 0's (−4.9 and −4.3). The margin over bubble-only training is therefore not a single-seed effect.
  - At 1 m, neither stat seed beats bubble-only seed 0 beyond the CI.
- **Seed 1 is slightly worse than seed 0 near the source in clean audio.** It is 0.6 points worse at quiet 1 m and bubbles0 1 m, so its gain over C0 at bubbles0 1 m is not significant. stat1 beats C0 in 8 of 9 cells; seed 0 did in 9 of 9.
- **Dev-WER selection beat CE selection on the test set, and was never worse.**
  - stat1 is better than stat1-CE in 3 cells beyond the CI (quiet 1 m −1.0, noise0 6 m −1.4, bubbles0 6 m −0.8) and level in the other 6.
  - stat1-CE beats C0 in only 6 of 9 cells. The dev set flagged its quiet weakness (dev quiet 4.3), and E7 quiet 1 m confirms it (3.3 vs C0 3.1).
  - This supports finding 3 of the draft: validation CE is a poor selector for recognizer adaptation.
- **No loops.** The guarded decode keeps both seed-1 checkpoints loop-free, as it did for seed 0.

### EXPLORATORY: C8 stat seed-average adapter (2026-11-06)

Exploratory only. It does not change the frozen protocol below: the carried-forward adapter stays seed 1 step 30k.

Method:

- The two stat seeds' adapters were averaged exactly at matched steps 28k and 30k (`scripts/average_lora.py`, output `runs/c8_whisper_lora_stat_avg`).
  - Rank stacking (r 8 → 16, alpha 32 → 64) makes the merged weight delta the mean of the two seeds' deltas.
  - Naive A/B averaging would not do this, because the seeds start from different random A matrices.
- The average was selected by dev WER under the frozen rule, then scored on E7 with the guarded decode as `c8statavg`.

Dev WER (%, guarded decode, same dev set as seed 1):

| checkpoint | quiet | bubbles0 | stat_m5 | stat_m1 | mean |
|---|---|---|---|---|---|
| base | 3.4 | 3.5 | 50.3 | 33.3 | 22.65 |
| avg step 28k (selected) | 3.3 | 3.0 | 39.3 | 25.1 | 17.69 |
| avg step 30k | 3.3 | 3.0 | 39.2 | 25.2 | 17.71 |
| seed 1 step 30k (frozen) | | | | | 17.64 |

E7 corpus WER (%, guarded decode, 1 / 3 / 6 m):

| cond | C0 (noisy) | stat s0 | stat s1 (frozen) | stat avg |
|---|---|---|---|---|
| quiet | 3.1 / 3.7 / 4.3 | 1.7 / 2.6 / 3.2 | 2.3 / 2.7 / 3.4 | 2.4 / 2.8 / 3.4 |
| noise0 | 20.3 / 48.4 / 56.4 | 16.0 / 37.2 / 44.5 | 16.4 / 38.2 / 43.8 | 15.9 / 37.0 / 43.2 |
| bubbles0 | 2.4 / 5.1 / 5.7 | 1.7 / 3.2 / 4.1 | 2.2 / 3.1 / 4.0 | 1.9 / 3.2 / 3.8 |

Paired ΔWER, avg minus the other system (points, speaker-cluster bootstrap 95% CI, 2000 resamples; * = CI excludes 0):

| cell | vs C0 | vs stat s1 (frozen) | vs stat s0 | vs bubble-only s0 |
|---|---|---|---|---|
| quiet 1 m | −0.7 [−1.3, −0.1]* | +0.2 [−0.3, +0.7] | +0.7 [+0.2, +1.3]* | +0.4 [−0.4, +1.2] |
| quiet 3 m | −0.9 [−1.5, −0.3]* | +0.1 [−0.4, +0.6] | +0.2 [−0.2, +0.8] | +0.5 [+0.1, +1.0]* |
| quiet 6 m | −1.0 [−1.7, −0.2]* | +0.0 [−0.5, +0.5] | +0.2 [−0.3, +0.7] | −0.3 [−1.0, +0.5] |
| noise0 1 m | −4.4 [−6.2, −2.7]* | −0.5 [−1.1, +0.2] | −0.1 [−1.1, +0.9] | −1.2 [−2.4, +0.0] |
| noise0 3 m | −11.4 [−13.7, −9.1]* | −1.2 [−2.8, +0.5] | −0.1 [−1.5, +1.4] | −5.1 [−7.5, −2.7]* |
| noise0 6 m | −13.2 [−16.2, −10.2]* | −0.6 [−1.9, +0.6] | −1.3 [−3.0, +0.2] | −5.6 [−7.6, −3.7]* |
| bubbles0 1 m | −0.5 [−1.0, −0.1]* | −0.3 [−0.6, +0.0] | +0.2 [+0.0, +0.5]* | −0.1 [−0.4, +0.2] |
| bubbles0 3 m | −1.9 [−2.9, −1.1]* | +0.1 [−0.3, +0.5] | +0.0 [−0.4, +0.5] | +0.2 [−0.3, +0.7] |
| bubbles0 6 m | −1.9 [−2.8, −1.2]* | −0.2 [−0.6, +0.1] | −0.3 [−0.9, +0.3] | −0.6 [−1.1, −0.1]* |

Reading:

- **The average beats C0 in all 9 cells**, one more than seed 1 (it also wins bubbles0 1 m).
- **It is level with the frozen seed 1 in every cell.** All nine CIs include zero. The point estimates lean its way on noise0 (−0.5 / −1.2 / −0.6), but that is not significant.
- **Against stat seed 0 it is level or slightly worse.** It is worse by +0.7 at quiet 1 m and by +0.2 at bubbles0 1 m. Both are cells where seed 0 was the stronger seed.
- **Its margin over bubble-only training holds**: −5.1 and −5.6 at noise0 3 m and 6 m.
- **Loops: 0** across the 9 cells. The temperature fallback fired once.
- **No case for changing the frozen adapter.** On dev WER the average is 17.69 against seed 1's 17.64, and on test it is level. Weight averaging gives no measurable gain here, so seed 1 step 30k stays as the carried-forward adapter.

### EXPLORATORY: failure analysis of the noise0 errors (2026-11-07)

Exploratory and CPU-only: it reuses the guarded E7 decodes (`runs/sim_baseline/per_utt_guard.jsonl`) and changes nothing in the protocol. Script: `tmp/dbg_noise0_failures.py`, full tables in `tmp/noise0_failures.md`.

| cell | system | WER | S / D / I % | utts ≥ 70% WER | err. function / content words | err. by utterance third |
|---|---|---|---|---|---|---|
| noise0 3 m | C0 | 48.4 | 34.8 / 7.6 / 6.0 | 30% | 36.0 / 48.5 | 32.2 / 44.7 / 51.4 |
| noise0 3 m | C8 (frozen) | 38.2 | 28.3 / 3.3 / 6.6 | 17% | 25.4 / 37.4 | 24.6 / 32.8 / 38.2 |
| noise0 6 m | C0 | 56.4 | 42.2 / 7.3 / 6.9 | 40% | 42.4 / 56.2 | 42.5 / 50.3 / 56.6 |
| noise0 6 m | C8 (frozen) | 43.8 | 33.5 / 3.8 / 6.6 | 24% | 30.2 / 43.8 | 30.3 / 37.8 / 44.4 |

Findings:

- **The errors are fluent substitutions, not dropouts or loops.** Substitutions are about three quarters of the errors for both systems. No hypothesis is empty, and the hallucination flag fires on at most 1 of 200 utterances per cell. Whisper writes a plausible sentence that shares only a few words with the reference, for example "in all the novel had a difficult gestation" → "it all was not as bad as it was going to be at the station". The existing flag (insertion loops) cannot see these. A substitution-hallucination flag is already on the list before E14, and these utterances are its test cases.
- **C8 mostly removes deletions and moves utterances out of total failure.** Deletions fall from 7.6% to 3.3% at 3 m and from 7.3% to 3.8% at 6 m, and the share of utterances at ≥ 70% WER falls from 30% to 17% and from 40% to 24%. Insertions do not fall. About 90% of C8's error reduction comes from the two harder C0 terciles. Per utterance, C8 is better on 124 and 133 of 200 at 3 m and 6 m, and worse on 44 and 38.
- **Errors rise towards the end of each utterance.** The content-word error rate by utterance third is 28.6 / 38.5 / 44.8% for C8 at 3 m (C0: 37.4 / 50.2 / 57.7%). There is no such trend by absolute word index (30.7 / 32.8 / 31.6% for words 0–7, 8–15, 16+), so the trend follows sentence position, not decode length. The active speech level falls by about 3 dB from the first to the last third of an utterance (R0, noise0 3 m: +1.5 / 0.0 / −1.6 dB around the utterance mean). The noise is stationary and mixed at the utterance-average SNR, so sentence-final words sit at a lower local SNR. Real pool noise mixed by the same rule would do the same.
- **Female speakers are much worse under stationary noise, and C8 does not close the gap.** v0-test has 20 female and 20 male speakers. The F − M WER gap (speaker bootstrap, 2000 resamples) is:

  | cell | C0 | C8 (frozen) |
  |---|---|---|
  | quiet 6 m | +0.2 [−1.8, +2.3] | +0.2 [−1.0, +1.4] |
  | bubbles0 6 m | +1.5 [−1.0, +4.1] | +0.2 [−1.4, +1.8] |
  | noise0 1 m | +7.5 [+0.3, +14.9] | +4.8 [−1.8, +11.6] |
  | noise0 3 m | +15.9 [+6.0, +26.5] | +13.7 [+3.7, +24.4] |
  | noise0 6 m | +14.1 [+2.5, +24.7] | +12.2 [+1.0, +23.3] |

  There is no gap in quiet or bubbles. The cause is not established. A candidate is that the SNR is set over 500–8000 Hz while the noise falls at −3 dB/oct, so the two sexes' speech meets different per-band SNRs. The real-data analysis should report WER by speaker sex, and the 20/20 balance of v0-test should be kept in every subset that is compared.

### EXPLORATORY: held-out mismatch stress suite (design fixed 2026-11-07, before any result)

Exploratory. It does not change the frozen protocol or the carried-forward adapter. Script: `scripts/stress_eval.py`, output `runs/stress_v0`.

- **Question.** Does C8's E7 gain over C0 survive mismatches that no system trained on, of the kind the first real data may bring?
- **Design.** Each condition perturbs one factor of an E7 cell: the same 200 v0-test utterances, the same 1 / 3 / 6 m scenes and the same per-item noise seed (20261028). Bases: `q_` = quiet (SNR 30 dB), `n0_` = noise0 (SNR 0 dB, −3 dB/oct).

  | condition | factor | training range |
  |---|---|---|
  | `q_` / `n0_tiltm6`, `tiltp6` | hydrophone tilt to −6 / +6 dB at 8 kHz (+6 / −6 dB at 125 Hz), speech and noise | ±2 dB |
  | `q_` / `n0_ripple9` | projector 1/3-octave ripple U(−9, 9) dB, one fixed draw per position, speech only | ±3 dB |
  | `q_` / `n0_t60x2` | every octave T60 doubled, 2 s RIR | scale 0.7–1.3 |
  | `q_` / `n0_clip` | speech-active level −6 dBFS, peaks hard-clip | −35 to −15 dBFS |
  | `n0_slopem9`, `n0_slopep3` | stationary noise at −9 / +3 dB/oct, SNR 0 | [−6, 0] dB/oct |
  | `n0_tonal` | harmonic hum, f0 120 Hz to 4 kHz, −3 dB/oct, slow wobble, −15 dB white floor, SNR 0 | not in training |
  | `n5` | noise0 noise at SNR −5 dB | in range (harder point, not a held-out factor) |

- **Systems.** C0 (`noisy`) and the frozen C8 adapter (`c8`, SHA-256 checked) on every condition. DFN3-C2 (`dfn3_c2`) on the quiet-base conditions only, since it is already at 80–90% WER on E7 noise0. Guarded decode throughout.
- **Reading rule.**
  - Paired speaker-cluster bootstrap ΔWER vs C0 in each cell, 2000 resamples.
  - C8's advantage **holds** under a condition if the CI is below zero at all three ranges, is **lost** if it includes zero at any range, and **reverses** if it is above zero at any range.
  - Each cell is read next to the matching E7 cell's ΔWER.
- **Cost.** About 20k guarded decodes, roughly 10–14 GPU-hours on the laptop.
- **Results (2026-11-07).** All 14 conditions ran (19,800 guarded decodes). Full table: `runs/stress_v0/stress_paired.md`.

  Paired ΔWER vs C0, points [95% CI]; * = CI excludes 0. The matching E7 cell is quiet: −0.8* / −0.9* / −1.0* at 1 / 3 / 6 m, and noise0: −4.0* / −10.2* / −12.6*.

  | condition | C0 WER 1 / 3 / 6 m | C8 − C0, 1 m | 3 m | 6 m | C8 verdict |
  |---|---|---|---|---|---|
  | `q_tiltm6` | 2.9 / 4.0 / 4.6 | −0.2 [−1.0, +0.6] | −1.1 [−1.8, −0.5]* | −1.2 [−2.1, −0.3]* | lost |
  | `q_tiltp6` | 3.3 / 3.6 / 4.4 | −1.2 [−1.7, −0.7]* | −0.9 [−1.7, −0.2]* | −1.1 [−1.8, −0.4]* | holds |
  | `q_ripple9` | 3.1 / 4.0 / 4.4 | −0.7 [−1.4, −0.1]* | −1.0 [−1.8, −0.3]* | −1.1 [−2.0, −0.2]* | holds |
  | `q_t60x2` | 3.1 / 8.6 / 10.6 | −0.6 [−1.4, +0.2] | −1.4 [−2.6, −0.3]* | −0.8 [−1.9, +0.2] | lost |
  | `q_clip` | 3.6 / 4.8 / 5.3 | −0.8 [−1.4, −0.1]* | −1.3 [−2.0, −0.5]* | −1.2 [−2.1, −0.4]* | holds |
  | `n0_tiltm6` | 21.2 / 48.7 / 57.9 | −4.6 [−6.9, −2.4]* | −10.7 [−13.5, −8.0]* | −13.9 [−16.7, −11.0]* | holds |
  | `n0_tiltp6` | 20.6 / 48.6 / 58.0 | −4.8 [−6.7, −3.0]* | −9.9 [−12.6, −7.2]* | −13.8 [−16.9, −10.6]* | holds |
  | `n0_ripple9` | 30.4 / 58.6 / 53.3 | −6.4 [−9.3, −3.7]* | −13.5 [−17.3, −10.2]* | −11.5 [−14.3, −8.6]* | holds |
  | `n0_t60x2` | 26.1 / 77.9 / 84.4 | −5.9 [−8.5, −3.7]* | −12.7 [−16.0, −9.5]* | −14.5 [−17.3, −11.6]* | holds |
  | `n0_clip` | 24.4 / 52.6 / 60.2 | −5.8 [−7.6, −4.0]* | −10.3 [−12.9, −7.9]* | −14.4 [−17.6, −10.8]* | holds |
  | `n0_slopem9` | 20.8 / 40.7 / 48.4 | −3.7 [−5.4, −2.0]* | −7.9 [−10.4, −5.5]* | −8.3 [−10.7, −6.1]* | holds |
  | `n0_slopep3` | 5.1 / 20.0 / 22.7 | −0.6 [−1.2, −0.1]* | −5.3 [−7.6, −3.2]* | −5.6 [−7.8, −3.5]* | holds |
  | `n0_tonal` | 10.6 / 30.7 / 39.9 | −2.6 [−4.1, −1.4]* | −6.8 [−9.2, −4.4]* | −8.7 [−11.2, −6.1]* | holds |
  | `n5` | 61.0 / 92.9 / 96.0 | −11.5 [−14.2, −8.9]* | −15.5 [−20.1, −10.9]* | −9.0 [−12.7, −5.2]* | holds |

  DFN3-C2 vs C0 on the quiet-base conditions, 1 / 3 / 6 m (E7 quiet: −0.8* / +2.3* / +4.9*):

  | condition | DFN3-C2 − C0, 1 m | 3 m | 6 m | verdict |
  |---|---|---|---|---|
  | `q_tiltm6` | −0.3 [−0.8, +0.1] | +2.0 [+1.3, +2.8]* | +3.8 [+2.7, +4.9]* | reverses |
  | `q_tiltp6` | −0.8 [−1.4, −0.3]* | +2.8 [+1.9, +3.7]* | +5.6 [+4.2, +7.1]* | reverses |
  | `q_ripple9` | −0.6 [−1.1, −0.1]* | +3.0 [+2.0, +4.0]* | +4.7 [+3.5, +5.9]* | reverses |
  | `q_t60x2` | −0.3 [−0.8, +0.3] | +14.3 [+12.0, +16.8]* | +19.5 [+16.8, +22.6]* | reverses |
  | `q_clip` | −0.3 [−0.9, +0.3] | +4.5 [+3.3, +6.0]* | +7.9 [+5.8, +10.1]* | reverses |

  Reading, under the pre-registered rule:

  - **C8 never reverses.** Its advantage holds under 12 of 14 conditions and is lost under 2: `q_tiltm6` (the 1 m CI includes zero) and `q_t60x2` (the 1 m and 6 m CIs include zero). Both losses are in the quiet base, where the E7 gain is itself only about 1 point, so a small loss of effect or power is enough to cross zero. They are not ruled out as real losses.
  - **Under noise, every held-out factor leaves C8's gain intact.** The relative WER reduction stays at 16–25% in almost every cell (E7 noise0: 20–22%). The exceptions are `n5` at 6 m (9%; both systems are near floor at 96% and 87%) and `n0_slopep3` at 1 m (12%; C0 is already at 5%). The held-out noise family (`n0_tonal`) gives 22–25%, and the unseen slopes −9 and +3 dB/oct give 17–19% and 25–27%.
  - **Doubled reverberation is the hardest held-out factor for every system.** With noise it pushes C0 to 78% and 84% at 3 m and 6 m, against 48% and 56% in E7. Doubled reverberation also gives DFN3-C2 its largest harm: +14.3 and +19.5 points at 3 m and 6 m, four to six times its E7 quiet harm.
  - **DFN3-C2 reverses under every quiet-base factor,** as it does in E7 quiet, and clipping about doubles its harm at 3 m (+4.5 against +2.3). Its only gains are about 1 point at 1 m.
  - **Caveat.** These are single-factor perturbations of the same placeholder simulator, so they say nothing about mismatches the simulator cannot express.

### EXPLORATORY: calibrating a substitution-hallucination flag (2026-11-07)

Exploratory and CPU-only: it reuses Whisper's own decode outputs, which the transcript cache already stores for every guarded decode. Nothing in the protocol changes. Script: `tmp/dbg_subst_flag.py`, output `tmp/subst_flag.md`.

- **Data.** 25,200 guarded decodes of C0, frozen C8 and DFN3-C2: the E7 cells plus all stress conditions.
- **Target.** A "fluent hallucination" is an utterance where:
  - the hypothesis is not a dropout (n_hyp ≥ 0.5 n_ref);
  - it shares few words with the reference (hits / n_ref < 0.3);
  - the §8.3 flag does not fire.
- **Prevalence.** 2,201 decodes (8.7%) meet the target, against 27 for the §8.3 flag. Almost all of them are in the noise base:

  | base | C0 | C8 | DFN3-C2 |
  |---|---|---|---|
  | quiet | 0.2% | 0.1% | 0.8% |
  | bubbles0 | 0.3% | 0.0% | 0.5% |
  | noise0 and the noise stress conditions | 20.1% | 11.6% | 43.0% |

Findings:

- **Of the reference-free signals, only Whisper's confidence works.** AUC for the target on the noise base (C0 / C8 / DFN3-C2):

  | signal | AUC |
  |---|---|
  | mean segment `avg_logprob` | 0.91 / 0.92 / 0.82 |
  | `no_speech_prob` | 0.81 / 0.88 / 0.73 |
  | hypothesis words per second | about 0.6 |
  | fallback fired, compression ratio | about 0.5 or below |

  A threshold on `avg_logprob` trades precision against recall: −0.6 gives precision 0.53 and recall 0.81, and −0.7 gives 0.64 and 0.60. At −0.6 it fires on 1.2% of quiet decodes.
- **The confidence signal is biased between systems.** At the same accuracy, C8 is 0.06–0.11 nats more confident than C0. The mean `avg_logprob` by hits / n_ref bin is:

  | hits / n_ref | C0 | C8 |
  |---|---|---|
  | 0.3–0.6 | −0.57 | −0.51 |
  | 0.6–0.9 | −0.38 | −0.29 |
  | ≥ 0.9 | −0.23 | −0.12 |

  A fixed threshold therefore flags more partly-correct C0 utterances than C8 ones. At −0.6 it flags 19.0% of C0 decodes and 9.4% of C8 decodes. Dropping each system's own flagged utterances would remove more of C0's hard material and shrink C8's apparent gain. So a per-system "WER without flagged utterances" is not a fair comparison under a confidence flag. The adapter's change to the decoder's confidence is a confound in its own right.
- **No logprob for some decodes.** For 44 decodes (35 of them C0) Whisper returned a hypothesis but `avg_logprob` is NaN. All 44 are targets.
- **Options put to the lead.** The lead chose A on 2026-11-08; see the frozen protocol below.
  - **A.** Use the reference-based target rule above as the flag on any data with references. It needs no calibration and treats every system the same way. Report its rate per system, and report WER without flagged utterances on the paired subset that excludes an utterance flagged for any system.
  - **B.** As A, plus `avg_logprob` < −0.6 as a reference-free diagnostic for unlabelled audio only. It would be reported as a rate and never used to filter a comparison.
  - **C.** Keep only the §8.3 flag. The rate of fluent hallucinations then goes unreported.

### EXPLORATORY: is the noise0 female − male gap a per-band SNR effect? (2026-11-08)

Requested by the lead on 2026-11-08. It changes nothing in the protocol. CPU part: `tmp/dbg_sex_gap.py`, with full tables in `tmp/sex_gap.md`. GPU part: `scripts/sexgap_eval.py`, output `runs/sexgap_v0`.

**CPU evidence (already run).** It re-simulates the received speech at the point where noise is mixed in, using the E7 seed and scenes, and uses the existing decodes.

- **Male speech carries more energy below the band where SNR is set.** The mix SNR is set over 500–8000 Hz. Male speech has about 5 dB more energy below 354 Hz, relative to 500–8000 Hz, than female speech (−1.1 against −6.1 dB at 3 m). Under the −3 dB/oct noise, males therefore get much higher per-band SNR below 500 Hz:
  - 125 Hz: about +13 to +16 dB above females;
  - 250 Hz: about +4 dB;
  - 500 Hz: about +1.5 dB.

  Above 1 kHz the two sexes are within about 1.7 dB, and males are higher at 4 kHz.
- **The gap tracks the noise slope.** The C0 F − M gap at 1 / 3 / 6 m is:

  | noise slope | 1 m | 3 m | 6 m |
  |---|---|---|---|
  | −9 dB/oct (low bands buried for both sexes) | +0.1 | +7.2 | +6.8 |
  | −3 dB/oct (E7 noise0) | +7.5 | +15.9 | +14.1 |
  | +3 dB/oct (low bands clean for both sexes) | +1.9 | +9.1 | +12.0 |

  At −9 dB/oct every CI includes zero. In quiet the gap is +0.2, so Whisper has no inherent sex gap on these talkers.
- **An intelligibility index explains about half the gap.** With an SII-style index (ANSI S3.5 octave weights) as a covariate, the per-utterance sex term at noise0 falls as follows:

  | system | 3 m | 6 m |
  |---|---|---|
  | C0 | +17.8 → +12.3 | +13.6 → +7.1 |
  | C8 | +14.2 → +7.0 | +12.4 → +5.6 |

  The CIs overlap, so this is correlational.

**Causal test (design fixed before any GPU result).**

- **Conditions.** Two single-factor variants of noise0, each with the same utterances, scenes, per-item seed and 0 dB SNR:
  - `n0_ssn`: noise with each item's own received-speech spectrum (smoothed over 1/3 octave), so the per-band SNR is flat and identical for every talker;
  - `n0_fullband`: the noise0 noise, with the SNR set over 50–8000 Hz, so the male low-band energy no longer comes free.
- **Systems.** C0 and frozen C8, guarded decode: 2,400 decodes, about 1.5–2 GPU-hours.
- **Reading rule.** F − M corpus-WER gap per cell, with speakers resampled within sex (2000 resamples), compared with the E7 noise0 gap.
  - **Explained** at a range if the point estimate is at most half the E7 gap and the CI includes zero.
  - **Partly explained** if it falls by at least a third but the CI excludes zero.
  - **Not explained** otherwise.
  - The 1 m cells are reported but not used for the verdict, because the E7 gap there is already marginal.
- **Secondary check.** Overall WER moves with the noise, and the gap grows with the WER level, so the F / M WER ratio is reported alongside.
- **Prediction.** `n0_ssn` explained for C0 at 3 m and 6 m. `n0_fullband` at least partly explained.
- **Results (2026-11-08).** All 2,400 decodes ran. Tables: `runs/sexgap_v0/stress_paired.md` (WER and C8 − C0) and `tmp/sexgap_read.md` (the gap reading, from `tmp/dbg_sexgap_read.py`).

  F − M corpus-WER gap, points [95% CI]. The value in brackets after each E7 noise0 gap is the F / M WER ratio.

  | cell | E7 noise0 gap | `n0_ssn` gap (F / M ratio) | verdict | `n0_fullband` gap (F / M ratio) | verdict |
  |---|---|---|---|---|---|
  | C0 1 m | +7.5 (1.44) | +1.4 [−4.2, +6.8] (1.10) | not used | −0.3 [−8.9, +8.5] (0.99) | not used |
  | C0 3 m | +15.9 (1.39) | +5.8 [−3.1, +14.9] (1.15) | explained | −0.2 [−11.4, +10.4] (1.00) | explained |
  | C0 6 m | +14.1 (1.28) | +9.5 [−0.2, +19.9] (1.22) | not explained | +6.3 [−3.6, +15.9] (1.11) | explained |
  | C8 1 m | +4.8 (1.34) | −0.3 [−4.2, +3.7] (0.97) | not used | −2.1 [−8.5, +4.8] (0.91) | not used |
  | C8 3 m | +13.7 (1.43) | +2.3 [−5.2, +9.7] (1.07) | explained | +3.4 [−7.7, +14.2] (1.07) | explained |
  | C8 6 m | +12.2 (1.32) | +6.1 [−3.9, +16.3] (1.18) | not explained | +6.5 [−4.6, +18.3] (1.14) | not explained |

  Overall WER, C0 / C8 at 1 / 3 / 6 m:

  | condition | 1 m | 3 m | 6 m |
  |---|---|---|---|
  | E7 noise0 | 20.3 / 16.4 | 48.4 / 38.2 | 56.4 / 43.8 |
  | `n0_ssn` | 14.7 / 11.1 | 40.2 / 32.3 | 46.6 / 36.7 |
  | `n0_fullband` | 28.3 / 22.4 | 66.6 / 52.2 | 60.9 / 50.0 |

  C8 − C0 holds in every cell of both conditions.

  Reading:

  - **At 3 m the gap is explained by either change, for both systems.** Flat per-band SNR leaves +5.8 (C0) and +2.3 (C8). Counting the speech below 500 Hz towards the SNR leaves −0.2 and +3.4.
  - **At 6 m only `n0_fullband` for C0 meets the rule.** The other three 6 m cells fall by 33–50% but miss the "at most half" threshold, with CIs that include zero (C8 `n0_ssn` sits just above half). The prediction "`n0_ssn` explained for C0 at 3 m and 6 m" is therefore met at 3 m only. "`n0_fullband` at least partly explained" is met.
  - **The shrinking gap is not a side effect of WER level.** `n0_fullband` makes every cell harder, by +8 to +18 points for C0, yet the F / M ratio still falls to 1.00–1.11. A larger WER would normally widen the absolute gap.
  - **Interpretation.** Most of the noise0 F − M gap comes from the way the simulator sets the SNR, not from Whisper or C8 handling female talkers worse. Setting the SNR over 500–8000 Hz gives male talkers their energy below 500 Hz for free. A residual of about 6–9 points at 6 m is not resolved by this test.
  - **Implications for real data.**
    - Pool noise is low-frequency heavy, so the same effect is expected there. WER by speaker sex should be read next to each talker's per-band SNR, not the broadband SNR.
    - Any SNR quoted for recorded data should state its band.
    - The "0 dB" of noise0 is optimistic measured over the full band: at 50–8000 Hz, C0 is 8–18 points worse.


## Frozen evaluation protocol for real data (signed off by the lead 2026-11-05)

Fixed before any real v0 recording is scored. It was chosen after seeing sim results, so it is pre-registered here, and any change after real data arrive must be reported as a deviation.

- **Decoding.** Every ASR result, for every system, uses `DecodeConfig.guarded()` (`--decode guard`) on Whisper large-v3 with loudness normalisation (`normalize_cut`). The settings:
  - Beam 5 at temperature 0.
  - Fallback temperatures 0.2, 0.4, 0.6, 0.8, 1.0 with best-of-5 sampling. A retry fires when the compression ratio is > 2.4 or the mean logprob is < −1.0.
  - Length cap of min(ceil(8 tokens/s × duration) + 16, 224) tokens.
  - No previous-text conditioning. Sampling seed 0, set on every call.
  - The fixed policy (temperature 0, no cap) may be reported only as a sensitivity check.
- **Checkpoint selection for recognizer adaptation.** Choose by the lowest mean of per-condition corpus WERs on the uwsim dev set, using `scripts/dev_wer_select.py` with its committed constants:
  - The dev set, scene geometry and seed 20261104 are as described in "C8 stat seed 1".
  - The decode is the same guarded decode.
  - Validation cross-entropy is not used for selection.
- **Recognizer-adaptation system carried forward (C8).** `runs/c8_whisper_lora_stat_s1/ckpt_step030000.pt`, also copied to `best_wer.pt`. SHA-256 `5063f02fa14cea47064721878c2e21cd8cd8594c642769cc69ae288029f52f90`.
  - Config: `configs/c8_whisper_lora_stat.yaml`, seed 1, LR 1e-4 (signed off 2026-11-05).
  - It is the only adapter whose checkpoint was chosen by the protocol above. Stat seed 0 (CE-best at step 28k), the bubble-only seeds and the seed-1 CE-best checkpoint are sim-only references. They are not compared on real data unless reported as exploratory.
- **Addition (signed off by the lead 2026-11-08, before any real data): substitution-hallucination flag.** It is option A of "calibrating a substitution-hallucination flag" above, implemented as `evaluation.asr.substitution_flags` and `paired_unflagged_mask`.
  - **Rule.** On any data with references, an utterance is flagged when n_hyp ≥ 0.5 n_ref and hits / n_ref < 0.3, with both counts taken after normalisation. The §8.3 flag stays as well.
  - **Reporting.** Report the flag rate per system. "WER without flagged" is computed on one paired subset that drops an utterance flagged for any compared system, never on each system's own unflagged utterances.
  - **Not adopted.** Reference-free confidence flags are not used to filter any comparison: `avg_logprob` is biased between C0 and C8.
  - **Primary endpoint unchanged.** The primary endpoint remains WER on all utterances.
- **Addition (signed off by the lead 2026-11-08, before any real data; reporting only): two-band SNR and WER by speaker sex.** It follows the sex-gap test above, where most of the noise0 F − M gap came from the SNR being set over 500–8000 Hz.
  - **SNR.** Real-data results report speech-band SNR over both 50–8000 Hz and 500–8000 Hz. The definition is the same as the simulator's `mix_at_snr`: speech-active samples from an energy VAD on the speech, zero-phase band-pass, with speech and noise measured through the same chain.
  - **WER by sex.** WER by speaker sex is reported and read against the median SNR of each sex in both bands. The F − M gap is reported with a 95% CI from resampling speakers within sex.
  - **Implementation.** `scripts/score.py` writes the median SNR in both bands to the summary, and writes `by_sex.csv` and `by_sex.md`.
  - **Nothing else changes.** No endpoint, comparison, filter or hypothesis changes.
- **Addition (signed off by the lead 2026-11-08, before any real data; measurement method): per-utterance SNR of pool takes.** The lead chose option (a). It is implemented in `uwse.bench.snr` and `scripts/measure_snr.py`.
  - **Speech power.** Taken from the utterance's own thruster-off cut at the same session, position and channel. ABBA plays every content block in both states, adjacent in time. The power is the band power over speech-active samples, using the energy VAD on that cut, as in `mix_at_snr`.
  - **Noise power.** Taken from the position's noise-only take in the row's thruster state: `<pos>_noise` (thruster off) or `<pos>_noise_on` (thruster on at the speech-take throttle). It is the band power over the whole take, minus 1 s at each end.
    - **Amendment (signed off by the lead 2026-11-08, before any real data).** The band power is taken only over the 0.1 s frames whose 50–8000 Hz level is within 6 dB below the take's median frame level (`NOISE_GATE_DB`, the default of `measure_snr.py --noise-gate-db`). This drops recorder pre-roll and thruster spin-up inside a thruster-on take; see "S1 dry run" below.
    - **Ops rule.** The thruster is started and at steady speed before the `<pos>_noise_on` recorders start, and is stopped only after they stop. This is in the run sheet, the `run_session take` prompt and the session checklist.
    - The session plan had no thruster-on noise take, so `build_plan` now adds `<pos>_noise_on` (60 s) at every position that has thruster-on blocks.
  - **Bands.**
    - `acoustic.snr_speechband_dB`: 500–8000 Hz.
    - `acoustic.snr_50_8000_dB`: 50–8000 Hz, a new optional schema field.
    - `acoustic.snr_octave_dB`: octaves 500–4000 Hz (fc/√2 to fc·√2).
    - `snr_db_500_8000` and `snr_db_50_8000`: the same two values under the keys `score.py` reads (it takes them through `--snr`).
  - **Speech-level stability.** The drift anchor repeats the session's first block at the end. The session's level drift is the median, over the anchor's utterances, of the 500–8000 Hz speech-active level of the anchor cut minus the original cut, using the same state and the original's VAD mask.
    - If |drift| > 1.0 dB, every cut of that session gets the schema flag `snr_level_drift`.
    - The 1.0 dB threshold was fixed before any data.
    - A session with no anchor cuts is reported as "not checked".
  - **Assumptions.** Speech and noise are recorded through the same chain and recorder layout, and the noise is read on the cut's file channel. Thruster noise is taken to be stationary between the noise-only take and the speech blocks; the drift check does not test this.
  - **Reporting.** The flag is reported, and is not used to filter any comparison.
- **Amendment: S1 scope and primary endpoint (signed off by the lead 2026-11-09, before any pool recording exists).**
  - *Session S1* (2 h of pool time) records one position only.
    - `p3`: 3 m, projector and hydrophone at 1 m depth.
    - Thruster-off and thruster-on blocks in ABBA order, plus the drift anchor.
    - The full v0-test selection: 200 test-clean utterances from 40 speakers, plus the dev utterances.
    - Per position: the noise, `noise_on` and sweeps takes.
    - No extra noise bank, and no E1–E4 takes.
  - The bench is built into `data/bench/v0_s1`: `make_playlists.py --out data/bench/v0_s1 --positions p3`, then `make_session_audio.py`, both with default settings. Its file hashes are recorded below. `data/bench/v0` is kept for the full 3-position S2 schedule.
  - *Frozen bench (built 2026-11-09 00:09).*
    - `data/bench/v0_s1/SHA256SUMS.txt` lists all 68 files and hashes to `6e6f6d02…c5296357` (`6e6f6d021c086482f6ceedf5ab14385c133bc667af527c170e5d6626c5296357`).
    - `schedule.json` is `c150c540…6dc971` and `session/p3_speech.json` is `58d7e4cc…038009`.
    - The plan has 5 takes and 71 min of recording; `p3_speech` alone is 66.5 min, with 56 blocks and 480 utterances.
  - *Primary endpoint for S1:* paired ΔWER of C8 (the frozen stat seed-1 adapter, `best_wer.pt`) minus C0 on the S1 v0-test cuts. It uses the frozen decode (`whisper-large-v3+lnorm+guard`) and the frozen statistics above, on the scored hydrophone channel. It is reported overall and by thruster state.
  - C2 and the other front ends minus C0 are secondary. They are scored once a front-end inference script for real cuts exists.
  - The single position means range cannot be tested on S1. Any result is scoped to one pool, one position and one session.
- **Amendment: S1 noise bank, scoring runs and endpoint cuts (signed off by the lead 2026-11-09 to 11-10, before any pool recording exists).**
  - *Noise bank added (2026-11-09).* S1 now also records the default noise bank after the anchor: `noise_quiet_pool`, `noise_thruster`, `noise_bubbles_splashing` and `noise_pumps`, 90 s each (`run_session.py plan` with the default `--extra-noise`). The plan has 9 takes and 77 min of recording. The speech and anchor takes, their utterance lists and the frozen bench `data/bench/v0_s1` are unchanged. This replaces "No extra noise bank" above. The noise-bank takes are characterisation only and do not enter any WER endpoint.
  - *Bootstrap counts (2026-11-10).* `scripts/score.py` takes one `--n-boot` for every bootstrap, so S1 is scored in two runs: `--n-boot 10000` for the corpus WER CIs and `--n-boot 2000` for paired ΔWER, matching the counts below. No code change.
  - *Endpoint cuts (2026-11-10).* The primary endpoint uses only the v0-test cuts of `p3_speech` on the scored channel (200 utterances × 2 thruster states). The v0-dev cuts and the `p3_anchor` repeats are excluded; the anchor serves only the level-drift check. The scoring run drops v0-dev rows explicitly, because `make_score_manifest.py` does not filter on `bench_split`.
- **Amendment: S1 rig without a loopback, and C2 input level (signed off by the lead 2026-11-15, before any pool recording exists).**
  - *No loopback.* The 40 dB loopback divider is dropped, so no wiring can put amplifier voltage into the interface. S1 uses `configs/session_channels_s1.json`: the v0 map without the loopback, with interface in1 unused and in2 the monitor hydrophone. `configs/session_channels_v0.json` is unchanged and stays the default for a later session that adds the divider.
    - Every recorder aligns on the chirps its own hydrophone hears, as the Daisy boards already did. The cuts carry a few ms of acoustic delay, well inside the 0.25 s padding. The WER endpoint, the cuts and the scoring are unchanged.
    - Lost: the immediate loopback QC of each take, so `run_session.py check` on the hydrophone files is the first QC; `levels --live`, replaced by a 1 kHz tone at the speech file's active RMS read on a multimeter (checklist §3); and an exact electrical time zero, which only time-of-flight and the E2/E3 calibrations need, none of which are in S1.
    - The S1 plan was regenerated with the new map (`run_session.py plan --bench data/bench/v0_s1 --session 1 --channel-map configs/session_channels_s1.json --session-dir sessions/2026-11-09_s1`). The 9 takes, the take plan and both utterance lists are identical to the 11-09 plan, and `data/bench/v0_s1` is untouched (`SHA256SUMS.txt` still hashes to `6e6f6d02…`). The 11-09 files are kept in `sessions/2026-11-09_s1/superseded_2026-11-09/`.
  - *C2 input level.* The secondary C2 − C0 comparison on S1 cuts runs `enhance_cuts.py --input-level sim`. Each cut is scaled by the gain `score.py`'s level normalization gives its C0 cut (R0 VAD, −26 dBFS active speech; noise rows take the median speech gain) before C2, and the output is scaled back. C2 then sees the input level it was trained on in simulation, and its output stays on the recorder's scale and comparable with C0. The flag is set once per `enhance_cuts.py` run, so any other front end enhanced in the same run gets the same level.
- **Amendment: exploratory C8g comparison on S1 (signed off by the lead 2026-11-18, before any pool recording exists).**
  - *Why.* C8g, C8 plus a no-speech gate (`uwse/evaluation/nospeech_gate.py`), removes Whisper's invented words on simulated noise at no simulated WER cost (FINDINGS.md, "No-speech gate"). It has never been run on underwater audio. S1 is its first test.
  - *System.* C8g as committed with this amendment: the frozen C8 checkpoint (sha256 `5063f02f…f52f90`, checked before decoding), `DecodeConfig.guarded()`, `nospeech_gate.TAU = 0.05`, and `nospeech_gate.gate_text`. p_ns is `nospeech_gate.first_window_p_ns` on exactly the 16 kHz array that `score.py` decodes for the C8 row (the cut times its level-normalization gain, same fp16 setting). Every S1 input is at most 30 s (the longest v0-test cut is under 16 s with padding, and noise segments are 10 s), so the gate reduces to one window: C8g text is empty if p_ns ≥ 0.05, else the C8 hypothesis from the primary run's transcript cache. No second decode is run.
  - *No tuning on S1.* The threshold, the window rule and the system are not re-selected on S1. Any other threshold or rule computed on S1 is labelled post hoc and kept out of the C8g row.
  - *Comparison 1, speech kept.* Computed in a separate run and output directory, never inside the primary `score.py` manifest, so no primary output changes. The C8g rows are the C8 rows of the primary `per_utt.jsonl` with the hypothesis replaced as above, re-counted with the same WER code. Over the 400 primary-endpoint cuts (200 v0-test utterances × 2 thruster states, `p3_speech`, scored channel), report:
    - paired ΔWER of C8g minus C8, with the `score.py` paired speaker-cluster bootstrap (2000 resamples, seed 0, strata position and thruster), overall and per thruster state;
    - total deletions and the deletion rate for each system, and their difference;
    - the number of gated cuts (p_ns ≥ 0.05), and how many of them had a non-empty C8 hypothesis (the cuts whose words C8g removed);
    - the median, 95th percentile and maximum of p_ns over the speech cuts, per thruster state.
  - *Comparison 2, invented words on underwater noise.* The six noise-only takes: `p3_noise` and `p3_noise_on` (60 s planned), and `noise_quiet_pool`, `noise_thruster`, `noise_bubbles_splashing` and `noise_pumps` (90 s planned).
    - Audio: `s1_<take>_daisyA.wav`, file channel 0 (channel_id 0, the scored channel), resampled to 16 kHz as `score.py` does.
    - Segments: segment k covers [1 + 10k, 11 + 10k) s from the file's first sample, for k = 0 … floor((D − 2)/10) − 1, where D is the planned take length in `session_plan.json`. That gives 5 segments for each 60 s take and 8 for each 90 s take, 42 per system. A file too short for its last segment is reported as missing that segment, never padded.
    - Gain: each segment is a noise row labelled position `p3`, thruster `off` for `p3_noise`, `noise_quiet_pool`, `noise_bubbles_splashing` and `noise_pumps`, and `on` for `p3_noise_on` and `noise_thruster`. It takes the median of the `score.py` speech gains of the 200 v0-test `p3_speech` cuts of that thruster state (no dev cuts, no anchor) for C8. C8 and C8g share these gains.
    - Readout: a segment has output when its hypothesis has at least one word after the Whisper English normalizer. Report per take and pooled the segments with output and total output words, for C8 and C8g, with Wilson 95% intervals. Segments of one take are correlated, so the intervals are descriptive only.
    - A take noted on the run sheet as recorded while people were talking nearby is still scored and reported, with the note. No segment is excluded after its transcript has been seen.
  - *Harm, defined now.* C8g is reported as removing real underwater speech if the ΔWER CI excludes 0 in either thruster state, or if any gated cut had a non-empty C8 hypothesis. Each such cut is listed with its reference and C8 text.
  - *Status.* Both comparisons are exploratory. They are not the primary or a secondary endpoint, and the primary analysis is unchanged. This is an exploratory use of the noise-bank takes, which still enter no WER endpoint. Both are reported once, with this amendment cited.
- **Unchanged from RESEARCH_PLAN.md.** The endpoints, the comparisons (C0 and the SE front ends vs C8), corpus WER with speaker-cluster bootstrap 95% CIs, and paired speaker-cluster bootstrap ΔWER with 2000 resamples.

## Next

- **Training noise (signed off by the lead 2026-11-02; first run done, see above).** Until the recorded bank exists, half of the training scenes add stationary coloured Gaussian noise (`configs/uwsim_v0_stat.yaml`).
  - Level: −10 to 0 dB re the bubble component.
  - Slope: drawn from [−6, −4] ∪ [−2, 0] dB/oct. The band (−4, −2) is held out because the noise0 test noise is the same generator at −3 dB/oct. Noise0 therefore stays out of distribution in slope, though not in generator family.
  - First run: C8 (`configs/c8_whisper_lora_stat.yaml`, `runs/c8_whisper_lora_stat_s0`).
- **C8 follow-ups.**
  - LR 1e-4: signed off 2026-11-05.
  - Seed 1 is done (see above).
  - WER-based checkpoint selection on a held-out uwsim dev set: done (`scripts/dev_wer_select.py`). Use it for every future C8 run.
  - Retrain with stationary noise: done. Its loops are fixed by guarded decoding, and it is now the best system (see above).
  - Guarded decode vs the fixed policy: done for every system (see above).
  - Replicate stat with a second seed: done. The noise0 gain and the margin over bubble-only training replicate (see "C8 stat seed 1").
  - Use `train.val_num_workers=0` on the 16 GB laptop: validation workers spawn on every eval.
- **Hallucination flag.** Done: `evaluation.asr.substitution_flags`, signed off 2026-11-08 (see the frozen protocol).
- **Real-data scorer brought into line with the frozen protocol (2026-11-08).** Before this change `scripts/score.py` used the fixed §8.3 decode, could not decode C8, and computed "WER unflagged" on each system's own unflagged utterances. It now has:
  - `--decode guard` as the default, with the cache key format `whisper-<model>+lnorm+guard` taken from `sim_baseline.py`;
  - `--asr-lora SYSTEM=PATH` for C8, which is the unprocessed cut decoded by the LoRA-merged Whisper;
  - per-row §8.3 and substitution flags;
  - "WER unflagged" in the summary and in the paired ΔWER computed on the one paired subset that no system flagged.

  The sim scripts still report only the §8.3 flag. That is fine for sim-only reports, but `score.py` is the entry point for real data.
- **S1 dry run (2026-11-08, exploratory, CPU only).** The full real-data chain ran end to end on a synthetic session (`tmp/dbg_s1_dryrun.py`, not committed). The chain was `make_playlists`, then `make_session_audio`, then `run_session plan`, then synthetic files for every take on all three recorders, then `run_session take --no-audio` and `check --run-import`, then `measure_snr`, then `score.py --no-asr --snr`. The synthetic session had 2 positions × 2 content blocks with ABBA and the anchor. The daisy recorders ran at 96 kHz with +40 and −25 ppm clock error. The reverb was exponential. Thruster noise ran on the cued spans.
  - **Passed.**
    - Chirp QC passed 8/8 and 2/2 on every file.
    - Drift was recovered exactly (−40.00 and +25.00 ppm).
    - 112 cuts were imported and measured.
    - The anchor drift came out at 0.00 dB.
    - `score.py` produced the two-band SNR columns.
  - **SNR check.** Ground truth is the thruster-on cut minus the off cut, after a fractional-lag and gain fit. Measured minus truth was +0.64 dB (sd 0.05), the same at both positions and in both bands. All of it came from the noise take: the synthetic thruster was off during the recorder's pre-roll and post-roll, which is 3 of the 23 s left after trimming. With the new `--noise-gate-db 6`, measured minus truth was +0.04 dB.
  - **Fixed.**
    - The run sheet listed `<pos>_noise_on` as "(nothing; record only)" with no thruster cue. It now says THRUSTER ON and lists its cues.
    - `measure_snr --recorder` read every recorder's channels from one recorder's noise take. The recorder is now taken from each cut's `raw_file`.
    - `noise_powers(trim_s=0)` returned an empty signal.
  - **Decided (2026-11-08).** At the pool, a thruster-on noise take would also contain pre-roll and spin-up, which biases SNR high by about 10·log10(T_total / T_on). The lead adopted both remedies: the 6 dB level gate is now the method default (amendment above), and the ops rule (thruster steady before recording) is in the run sheet, the take prompt and the checklist.
  - **Gaps (closed 2026-11-08).** `scripts/make_score_manifest.py` now builds the `score.py` manifest from `align.jsonl` and writes aligned R0 cuts, so `--level-normalize` (the frozen `+lnorm+guard` decode) can run on real data.
  - **Dry-run ASR check: passed (2026-11-08).** `score.py --level-normalize --asr-lora C8=... --baseline C0` ran on `tmp/s1dry/score_in/manifest.jsonl` (`tmp/run_s1dry_asr.ps1`, under 2 min of GPU).
    - All 48 decodes (24 cuts × C0, C8) used the frozen cache key `whisper-large-v3+lnorm+guard`.
    - Every row had speaker sex and both SNR bands. `by_sex.md` and the SNR summary columns were filled.
    - WER was 0.6% for C0 (2 insertions) and 0% for C8. That is expected on the mild synthetic channel; the check is that the chain runs end to end, not the WER.
    - It ran after Whisper was changed to load straight onto the GPU (`asr._load_whisper_on_device`). The change was verified bit-identical to the frozen cache on 40 utterances each for C0 and C8, and peak RSS was 4.65 GB.
- **S1 prep (decided by the lead 2026-11-08).**
  - The 11-09 S1 session records the speech pool only. At each position that means noise, `noise_on`, sweeps, speech and anchor, followed by the noise bank. There are no E1–E4 takes.
  - `run_session.py plan` gained optional E2/E3/E4 take groups (`--e2`, `--e3-points`, `--e4`), off by default, for a later session.
  - The v0 bench build and freeze, the rig channel map, and the sweep and noise-bank analysis scripts are not done. Ask the lead before building v0.
- **Calibration.** Replace the placeholders with measured pool values (E0–E4) and the recorded noise bank, then regenerate this table. The current table is retired at that point.

## Amendment 2026-11-19: no-thruster pool plan, signed off by the lead before any recording

Signed off by the lead on 2026-11-19. No pool recording of any kind exists at this date. This amendment replaces the S1 plan above; the text above is left unchanged as the record. Where this amendment and an earlier S1 amendment disagree, this amendment wins. The current protocol in one page is `PROTOCOL.md`. `RESEARCH_PLAN.md` and `RESEARCH_UPDATE_3.md` moved to `docs/history/` on the same day; references to `RESEARCH_PLAN.md` above mean `docs/history/RESEARCH_PLAN.md`.

- **S1 retired unused.** Session S1 (`sessions/2026-11-09_s1`, bench `data/bench/v0_s1`) was never recorded. It is retired. `data/bench/v0_s1` and `data/bench/v0_pilot` stay frozen and untouched (`data/bench/v0_s1/SHA256SUMS.txt` still hashes to `6e6f6d02…`). The S1-specific amendments lapse with S1, except where restated below: the scope and primary endpoint of 2026-11-09, the noise bank and endpoint cuts of 2026-11-09 to 11-10, the rig without a loopback of 2026-11-15, and the exploratory C8g comparison of 2026-11-18. A C8g comparison on pool-plan data would need its own amendment.
- **Rig.** The lead's own private pool. Its size and depth are measured and logged on the day.
  - Projector: Lubell LL916C-050-CXA (caged, 50 ft cable) through a Lubell AC205C transformer.
  - Amplifier: Amazon B07VHG2LR9 (likely an AIYIMA unit with a TI TPA3255), with bridged (BTL) outputs, a mains DC brick and an unmarked volume knob.
  - Playback: Windows PC and USB interface, out1 to the amplifier. There is no loopback, and the interface records nothing.
  - Capture: two Daisy Seed boards (A: hydrophones ch0 and ch1; B: ch2 and ch3) at 96 kHz / 24-bit, recorded on a Linux machine (laptop or the Orin) with `scripts/record_daisy.py`. Files are copied to the Windows PC for checking, import and scoring. The scored channel is ch0. Every recorder aligns on the chirps its own hydrophones hear.
  - No thruster. The rig is not part of the submarine. Nothing in this plan has a thruster state: every block is played once (`block_order` `"none"`).
- **Takes, in order** (`scripts/run_session.py plan`; take ids in brackets).
  0. Desk dry run with the projector submerged in a bucket, never driven in air, knob just above minimum (`dryrun`): the anchor file through the full chain. QC must pass. The files go to `<session>/dryrun/`, never the session raw directory, and are never imported.
  1. Chain reference at 0.5–1 m (`ref_sweeps`, `ref_speech`): sweeps and the ~20 pilot sentences, on axis. They give the measured transducer-chain reference (h_chain, the R1 target) and the real projector response.
  2. Pilot at 3 m (`pilot_speech`): the 20 pilot sentences, scored on site with C0 (see the decision rule below).
  3. For each distance d (default 1, 3 and 6 m; chosen on the day with `--distances`):
     - noise only, 1 min (`p<d>_noise`);
     - sweeps, 1 min (`p<d>_sweeps`);
     - speech: all 200 v0-test and 40 v0-dev utterances once, in chirp-framed blocks (`p<d>_speech`, 27.1 min);
     - drift anchor: the first block repeated (`p<d>_anchor`).
  4. Bubble pass at one distance (default 3 m), with an air stone running near the speaker: bubble noise only, 2 min (`p3bub_noise`), and a deterministic subset of the test sentences, about 10 min (`p3bub_speech`, 9.4 min as built).
  - **Redos.** A failed take is redone as attempt k ≥ 2, with files named `..._r<k>_<rec>.wav`. Nothing is overwritten, and every attempt stays on disk and in `takes.jsonl`. The import uses the latest attempt that passes QC. A take is never redone because of its WER.
- **Pilot sentences.** 20 dev-clean utterances, one per speaker, none of them in v0-dev or in the 200-utterance test set. The choice is deterministic and documented in the bench (`selection/pilot_rule.json`, bench split `v0-pilot`). They are used only for the decision rule and the chain-reference WER.
- **Decision rule (pre-registered; uses only C0 on the pilot sentences).** The input is the pilot corpus WER of C0 on `pilot_speech`, on the scored channel, computed with `scripts/pilot_wer.py`. C0 is Whisper large-v3 with the frozen guarded decode and no LoRA.
  - Pilot C0 WER ≥ 10% (exactly 10% included): the quiet recordings are primary and added noise is secondary.
  - Pilot C0 WER < 10%: added real pool noise at 0 dB SNR is primary and the quiet recordings are secondary.
  - The rule is applied once. Its json is kept and not overwritten, and nothing is re-decided after the test takes are scored.
  - The rule changes only which condition is primary. The same takes are recorded either way.
- **Primary comparison.** C8 against C0, by corpus WER.
  - C8 is Whisper large-v3 with the frozen LoRA `runs/c8_whisper_lora_stat_s1/best_wer.pt` (sha256 `5063f02f…f52f90`, checked before decoding) and `DecodeConfig.guarded()`.
  - Data: the v0-test cuts of the `p<d>_speech` takes, on the scored channel, in the primary condition.
  - Reporting: per distance and pooled over distances. Corpus WER 95% CIs come from the speaker-cluster bootstrap (10000 resamples), and the paired ΔWER (C8 − C0) from the paired speaker-cluster bootstrap (2000 resamples, seed 0).
  - The machinery is `scripts/score.py`, driven by `scripts/score_session.py`.
  - Not in the primary comparison: v0-dev cuts, anchor repeats, the pilot, the chain reference and the bubble pass.
- **Secondary systems and conditions.**
  - Systems: the enhancers C1 (WPE), C2 (DeepFilterNet3 fine-tuned on uwsim) and C10 (Kiel-recipe U-Net), each against C0, as in the frozen protocol. C2 runs with `enhance_cuts.py --input-level sim`.
  - Conditions: the secondary condition (quiet or pool noise at 0 dB, whichever is not primary), pool noise at 5 dB, added bubble noise at 0 and 5 dB, and the live bubble pass.
- **Added-noise method.** Real noise is mixed into the real recorded speech cuts at 0 and 5 dB SNR with `scripts/mix_real_noise.py`.
  - Noise sources: pool noise from the `p<d>_noise` take at the cut's own distance, and bubble noise from `p3bub_noise`.
  - SNR: set over 500–8000 Hz on the speech-active samples of the aligned R0 cut, using `uwsim.noise.mix_at_snr`.
  - Offsets: deterministic, seeded by the sha256 of the seed, noise kind and utt_id (seed 0). The same offset is used at both SNRs.
  - The water path of the speech stays real. Every such result is labelled added noise, never recorded noise.
- **Chain reference.**
  - The `ref_sweeps` take measures the transducer chain: amplifier, AC205C, LL916, water at short range and hydrophone.
  - `ref_speech` gives the chain-reference WER of C0 on the pilot sentences (`pilot_wer.py --role chainref`).
  - The LL916's 5 kHz notch is **not** pre-compensated in the played audio. Any intrusive metric uses the measured chain reference as R1. Until R1 cuts exist, the intrusive metrics are skipped and WER is the only endpoint.
- **New bench `data/bench/v1`.**
  - Build: `scripts/make_playlists.py --plan pool --librispeech-root data/librispeech --out data/bench/v1`, then `scripts/make_session_audio.py --bench data/bench/v1`, both with default settings.
  - Content: the same 200 v0-test and 40 v0-dev utterances as `v0_s1`, plus the 20 pilot sentences and the bubble subset.
  - Programs, checked on 2026-11-19 with `--no-render`: pilot 2.6 min; speech 27.1 min (23 blocks, one file, under the 35 min split limit); anchor 1.2 min; bubble 9.4 min.
  - Its `SHA256SUMS.txt` hash is recorded here once built. The bench is never rebuilt after the first take.
- **Unchanged.**
  - The guarded decode with loudness normalisation (`whisper-large-v3+lnorm+guard`).
  - The C8 checkpoint and its selection rule.
  - The substitution-hallucination flag and its paired reporting.
  - The two-band SNR and WER-by-sex reporting.
  - The speaker-cluster bootstrap statistics.
  - The per-utterance SNR method, with every cut in the thruster-off state and the noise taken from the cut's own `p<d>_noise` take.
  - Any change after the first pool take is scored is reported as a deviation.
