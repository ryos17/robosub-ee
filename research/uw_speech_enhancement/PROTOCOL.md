# Current protocol (pool plan, 2026-11-19)

This page is a short summary of what is frozen for the real-data study. It contains no results. The binding text is the frozen record [`RESULTS_E7_SIM_DRYRUN.md`](RESULTS_E7_SIM_DRYRUN.md): see its section "Frozen evaluation protocol for real data" (signed off 2026-11-05) and every amendment after it. The amendment that defines the current plan is **"Amendment 2026-11-19: no-thruster pool plan"** at the end of that file. If this page and the frozen record disagree, the frozen record wins.

The original plan is in [`docs/history/RESEARCH_PLAN.md`](docs/history/RESEARCH_PLAN.md). Its pool protocol (thruster ABBA, Avery pool, E0–E4) is superseded. The operator steps are in [`docs/RECORDING_SESSION_CHECKLIST.md`](docs/RECORDING_SESSION_CHECKLIST.md).

## Research question

Does a recognizer adapted only on simulated underwater speech (C8: Whisper large-v3 with a LoRA trained on `uwsim`) transcribe real projector-to-hydrophone speech better than the unadapted recognizer (C0)? A secondary question is whether off-the-shelf or sim-fine-tuned speech enhancement front ends help.

## What is frozen

| Item | Value |
|---|---|
| Decode | Whisper large-v3, `DecodeConfig.guarded()` (beam 5, temperature 0 with fallback, length cap, no previous-text conditioning, seed 0), with loudness normalisation to −26 dBFS active speech on the R0 VAD. Cache key `whisper-large-v3+lnorm+guard`. |
| C8 checkpoint | `runs/c8_whisper_lora_stat_s1/best_wer.pt`, sha256 `5063f02fa14cea47064721878c2e21cd8cd8594c642769cc69ae288029f52f90`. It was chosen by dev-set WER (`scripts/dev_wer_select.py`). `score_session.py` refuses any other checkpoint. |
| Enhancers | C1 `wpe1`; C2 `dfn3` (`runs/c2_dfn3_s0/best.pt`, run with `--input-level sim`); C10 `kiel_unet` (`runs/c10_kiel_unet_s0/best.pt`). |
| Statistics | Corpus WER with speaker-cluster bootstrap 95% CIs (10000 resamples). Paired speaker-cluster bootstrap ΔWER (2000 resamples, seed 0). |
| Flags | §8.3 hallucination flag and the substitution flag (n_hyp ≥ 0.5 n_ref and hits/n_ref < 0.3). Both are reported per system. "WER without flagged" is computed on one paired subset. Flags never filter the primary endpoint. |
| Reporting extras | Speech-band SNR over 50–8000 and 500–8000 Hz. WER by speaker sex. The anchor level-drift flag (\|drift\| > 1.0 dB). |
| Bench | `data/bench/v1` (`make_playlists.py --plan pool`, default settings). It holds the same 200 v0-test (test-clean, 40 speakers) and 40 v0-dev utterances as v0_s1, plus 20 pilot sentences (dev-clean, outside v0-dev and the test set) and a bubble subset. It is never rebuilt after the first take. |
| Retired | S1 (`data/bench/v0_s1`, `sessions/2026-11-09_s1`) was never recorded. It stays frozen and unused, as does `data/bench/v0_pilot`. |

## Session (one pool day)

The lead's own pool. Projector: Lubell LL916C through an AC205C, driven by a bridged class-D amplifier. Playback from a Windows PC with no loopback. Two Daisy Seed boards (4 hydrophones, 96 kHz / 24-bit) record on a Linux machine. ch0 is scored. There is no thruster.

1. Desk dry run (`dryrun`): QC must pass. Never imported.
2. Chain reference at 0.5–1 m (`ref_sweeps`, `ref_speech`).
3. Pilot at 3 m (`pilot_speech`), scored with C0 on site.
4. For each distance (default 1, 3 and 6 m): `p<d>_noise` (1 min), `p<d>_sweeps` (1 min), `p<d>_speech` (all 240 sentences, 27 min), `p<d>_anchor`.
5. Bubble pass at 3 m: `p3bub_noise` (2 min), `p3bub_speech` (about 10 min).

A failed take is redone as `--attempt 2`, 3, and so on. Takes are never overwritten, and a take is never redone because of its WER.

## Decision rule (pre-registered)

The pilot C0 corpus WER on `pilot_speech` decides which condition is primary. It is computed with `scripts/pilot_wer.py`, which writes its json once.

- **≥ 10%** (10.0% included): the **quiet** recordings are primary and added noise is secondary.
- **< 10%**: **added real pool noise at 0 dB SNR** is primary and the quiet recordings are secondary.

Nothing else depends on the pilot.

## Endpoint

**Primary:** corpus WER of C8 vs C0 in the primary condition, on the v0-test cuts of the `p<d>_speech` takes, scored channel ch0. It is reported per distance and pooled, with the paired ΔWER (C8 − C0) and its 95% CI. Excluded: v0-dev cuts, anchor repeats, pilot, chain reference and the bubble pass.

**Secondary:**
- C1, C2 and C10 vs C0.
- The other of quiet / pool 0 dB, plus pool 5 dB, added bubble noise at 0 and 5 dB, and the live bubble pass.
- The chain-reference WER.

## Added noise

Real noise is mixed into the real speech cuts (`scripts/mix_real_noise.py`):

- **Sources:** pool noise from the cut's own `p<d>_noise` take, and bubble noise from `p3bub_noise`.
- **Level:** 0 and 5 dB SNR over 500–8000 Hz on the R0 speech-active samples.
- **Offsets:** deterministic (seed 0), and the same offset at both SNRs.

The water path of the speech stays real. These results are always labelled as added noise.

## Chain reference

`ref_sweeps` measures the transducer chain (h_chain, the R1 target) and the real projector response. `ref_speech` gives the C0 WER near the projector. The 5 kHz notch of the LL916 is not pre-compensated. Intrusive metrics (ESTOI, PESQ) use the measured chain reference as R1, and are skipped until R1 cuts exist.

## Scoring pipeline

Each step's output is the next step's input:

1. `import_recording.py` makes the cuts and `align.jsonl`.
2. `make_score_manifest.py` adds the R0 cuts and the C0/C8 rows.
3. `pilot_wer.py` scores the pilot and the chain reference.
4. `mix_real_noise.py` makes the added-noise conditions.
5. `score_session.py` runs `score.py` with the frozen settings, two bootstrap counts and enhancers per condition.
6. `make_paper_numbers.py` writes `paper/results_real.tex` (macros) and `paper/fig_real.pdf`.

The exact commands are in the checklist, section 6.

## Deviations

Any change after the first pool take is scored is reported as a deviation in the frozen record. A change before that is a dated amendment, signed off by the lead.
