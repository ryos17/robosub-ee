# Underwater Speech Enhancement: Sim-to-Real

Research project (lead and sole author: Ryota Sato). The project asks whether speech that has travelled through water from a projector to a hydrophone is better served by **speech enhancement** (an SE front end before an off-the-shelf recognizer) or by **recognizer adaptation** (Whisper large-v3 with a LoRA trained only on simulated underwater speech). Both are trained on a physics-based underwater channel simulator (`uwsim`) and tested on real pool recordings.

The working title of the paper is *Speech Enhancement versus Recognizer Adaptation for Underwater Speech Recognition over a Projector-to-Hydrophone Link*. It targets OCEANS 2027 Aberdeen.

It builds on these team efforts:

- **Pool acoustic simulator** (EE project #12, team project owners Kaito Kanazawa and Keiya Wada). Its channel model is the basis of `uwsim`.
- **Loud underwater speaker** (EE project #3, team project owners Keiya Wada and Kaito Kanazawa). This is the projector line of work. The rig used here is a Lubell LL916C.
- **[`whisper_ivc/`](../whisper_ivc/):** hydrophone capture (Daisy Seed, 4 ch, 96 kHz / 24-bit) and Whisper on the Orin. `scripts/record_daisy.py` uses its `daisy_stream.py`.

## Status (2026-11-19)

- **Simulation: done.** `uwsim` v0, the adapters, the training of C2, C8 and C10, and the E7 sim-only dry run are complete. The frozen evaluation protocol is signed off. The C8 checkpoint is frozen (sha256 `5063f02f…`). Every simulation number is a pipeline check on an uncalibrated simulator and is not for quoting as a real result; see [`RESULTS_E7_SIM_DRYRUN.md`](RESULTS_E7_SIM_DRYRUN.md).
- **Real data: next.** One session in the lead's own pool, under the no-thruster pool plan of 2026-11-19 ([`PROTOCOL.md`](PROTOCOL.md)). The session tooling, the analysis scripts and the paper-number macros are written and tested. No recording exists yet.
- **After the session.** Score, generate `paper/results_real.tex` and `paper/fig_real.pdf`, build the abstract and submit it (due 2027-01-15).
- **Retired.** The thruster S1 plan (`sessions/2026-11-09_s1`, bench `data/bench/v0_s1`) was never recorded. Its bench stays frozen and unused.

## Documents

| Doc | What |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | **The current protocol on one page:** what is frozen, the endpoint, the pilot decision rule, the systems and the scoring. |
| [`docs/RECORDING_SESSION_CHECKLIST.md`](docs/RECORDING_SESSION_CHECKLIST.md) | **Operator checklist:** bench build, dry run, gear, rigging, safety and amplifier procedure, take loop, pilot decision, backup, import, scoring and the abstract build. |
| [`RESULTS_E7_SIM_DRYRUN.md`](RESULTS_E7_SIM_DRYRUN.md) | The frozen record: simulation results, the frozen evaluation protocol (2026-11-05) and every dated amendment, ending with the 2026-11-19 pool plan. Append-only. |
| [`CODE.md`](CODE.md) | Install, package map, tests, smoke test, training and placeholders. |
| [`FINDINGS.md`](FINDINGS.md) | Research notes: pool physics, close papers, SE methodology, hardware, venue, and later sim-only analyses. |
| [`PRIOR_ART.md`](PRIOR_ART.md) | Prior-art search and positioning (2026-10-26, partly superseded by FINDINGS). |
| [`docs/history/RESEARCH_PLAN.md`](docs/history/RESEARCH_PLAN.md) | The original research plan (rev. 3, 2026-10-27). The frozen protocol cites it for the §-numbered definitions. Its pool protocol and schedule are superseded. |
| [`docs/history/RESEARCH_UPDATE_3.md`](docs/history/RESEARCH_UPDATE_3.md) | Third research pass (2026-10-27): multilingual, thesis and patent prior art, the Kiel lineage, licences. |
| `docs/history/oceans2027_*_draft.md` | Superseded sim-only Markdown drafts of the abstract and methods (2026-11-03/06, thruster plan). History only; not the abstract source. |
| [`paper/`](paper/) | The OCEANS 2027 abstract (`oceans2027_abstract.tex`, IEEEtran conference, at most 2 pages, 500–1000 words) and the full-paper draft (`oceans2027_full.tex`). |

## Layout

```
uw_speech_enhancement/
  README.md  PROTOCOL.md  CODE.md  RESULTS_E7_SIM_DRYRUN.md  FINDINGS.md  PRIOR_ART.md
  docs/
    RECORDING_SESSION_CHECKLIST.md   pool-session operator checklist
    history/                         superseded plans (RESEARCH_PLAN.md, RESEARCH_UPDATE_3.md)
  environment.yml                    conda env "uwse" (Python 3.10)
  uwse/
    uwsim/        channel simulator: signed ISM + R(theta), tail, low band, transducer chain, noise
    irtools/      ESS sweeps, drift correction, EDC/T20/two-slope, DRR/C50, arrivals, sound speed
    evaluation/   alignment, level normalization, Whisper WER, metrics, bootstrap, no-speech gate
    adapters/     SEAdapter interface: identity (C0), WPE (C1), DFN3 (C2), Kiel U-Net (C10), LoRA, ...
    train/        model-agnostic loop, sim/in-air/replay datasets, Whisper-LoRA (C8)
    bench/        metadata schema, LibriSpeech selection, splits, playlists, session plan, ingest, SNR
  configs/        uwsim_v0*.yaml, one YAML per training condition, session channel maps (v1 = pool plan)
  scripts/        bench build, session tooling, import, scoring, paper numbers, training (see CODE.md)
  sessions/       one directory per pool session (plan, run sheet, takes.jsonl, raw Daisy files)
  paper/          abstract and full paper (LaTeX), figure scripts, macro placeholders
  tests/          pytest suite
  data/           corpora, RIR bank, benches, imported recordings (git-ignored)
  runs/           checkpoints and score outputs (git-ignored)
```

## Quick start

Run everything from this directory in the `uwse` env. The PC is the Windows machine, and Linux is the Daisy capture machine. The checklist has every flag and the safety steps; this section is only the outline.

**Record (one session, number 2):**

```
# desk, once
python scripts/make_playlists.py --plan pool --librispeech-root data/librispeech --out data/bench/v1
python scripts/make_session_audio.py --bench data/bench/v1
python scripts/run_session.py plan --bench data/bench/v1 --session 2 --channel-map configs/session_channels_v1.json \
  --session-dir sessions/<date>_s2 --distances 1 3 6 --pool-length-m L --pool-width-m W --pool-depth-m D --date <date>
python scripts/run_session.py levels --session-dir sessions/<date>_s2 --target-sl-db 160 --brick-v <V>

# at the pool, once: set the knob with the 1 kHz tone on a floating DMM, tape it, check
python scripts/run_session.py tone   --session-dir sessions/<date>_s2
python scripts/run_session.py levels --session-dir sessions/<date>_s2 --tone-vrms <DMM> --brick-v <V>

# each take, in plan order
# Linux: type the take's two record_daisy.py lines from RUN_SHEET.md / daisy_cmds.txt (start both boards)
python scripts/run_session.py take       --session-dir sessions/<date>_s2 --take-id <take> ... # PC, within 20 s
python scripts/run_session.py verify     --session-dir sessions/<date>_s2 --take-id <take>     # PC, after copying the files
python scripts/run_session.py check      --session-dir sessions/<date>_s2 --take-id <take>     # PC, QC + import lines
```

**Score:**

```
import_recording.py (lines from import_cmds.txt, or check --run-import; --out routed by take: main, pilot/, chainref/, bubble/)
make_score_manifest.py -> pilot_wer.py (pilot decision on site; chain reference later)
mix_real_noise.py      -> score_session.py -> make_paper_numbers.py -> paper/results_real.tex, paper/fig_real.pdf
```

The exact commands are in the checklist, sections 4.3, 5 and 6.

## Systems

| Id | System | Role |
|---|---|---|
| C0 | Whisper large-v3, guarded decode, no adaptation | baseline; the pilot rule |
| C8 | C0 plus a LoRA trained on `uwsim` with stationary noise (`runs/c8_whisper_lora_stat_s1/best_wer.pt`) | **primary** comparison, C8 vs C0 |
| C1 | WPE dereverberation into C0 | secondary |
| C2 | DeepFilterNet3 fully fine-tuned on `uwsim`, into C0 | secondary |
| C10 | Kiel-recipe U-Net trained on `uwsim`, into C0 | secondary |

## Positioning (has it been done?)

As a whole, no. Prior-art passes (2026-10-26 and 10-27) found no paper that combines all three of:

1. simulated underwater speech from a physics-based channel model;
2. training SE models and an ASR adapter on it;
3. testing on real projector-to-hydrophone speech.

The closest work:

- **Kiel DSS group** (DAGA 2025/2026). An underwater speech channel simulator and a U-Net dereverberator, with no objective real-data results. The simulator is therefore not the lead contribution; C10 is a reconstruction of their recipe.
- **Earlier real through-water speech**, classical and unreleased: IOA-CAS sea trials (Cao et al. 2017), WTD 71 / NATO telephone trials (Fricke & Nissen 2015; Nissen & Kuhnt-Matthé 2017), Kiel in-pool speech (DAGA 2018).
- **Jiang et al., Acta Acustica 2024.** A learned denoiser in a digital semantic-codec link over measured channels. The speech is never sent acoustically.
- **Rong et al., OCEANS 2025.** Speech-to-text with a fine-tuned Whisper, then text over an acoustic modem. This is the competing design to argue against.
- **In-air twins** (SonicSim, URGENT, Milo et al. 2026). The simulate, train and test-sim-to-real recipe is established in air. The novelty here is the underwater domain, the real measured benchmark, and the comparison of enhancement against recognizer adaptation.

**Claims to avoid:**
- "First underwater speech enhancement" (Ju et al. 2007 and others exist).
- "First deep learning for underwater speech" (Jiang 2024, Luo 2026).
- "First real through-water speech recordings."
- "First to align with sync markers" (Nissen & Kuhnt-Matthé 2017).
- "Works for any model."
- Any claim beyond one pool, one session and one projector.

The details are in [`PRIOR_ART.md`](PRIOR_ART.md) and [`FINDINGS.md`](FINDINGS.md).

## Venue: OCEANS 2027 Aberdeen

- **Event:** IEEE/MTS OCEANS 2027, 21–24 June 2027, P&J Live, Aberdeen (<https://aberdeen27.oceansconference.org/>).
- **Deadlines:** abstract due **15 January 2027** (EventsAir portal); notifications in March 2027; final paper due **24 May 2027**.
- **Abstract format:** abstract-only review; IEEEtran conference, at most 2 pages, 500–1000 words including figures and references, as a PDF. Check the portal for the final Aberdeen limits.
- **Tracks and competition:** Track 02 (Sonar Signal/Image Processing and Communication) is the main fit. Also submit to the Student Poster Competition.
- **What the abstract needs:**
  - real pool numbers: C0 vs C8 WER with CIs, by distance;
  - the pilot decision and the chain-reference WER;
  - the oceans framing (AUV and diver voice comms);
  - explicit positioning against Rong 2025 and Jiang 2024;
  - a commitment to release the simulator, the benchmark and the scripts.

## Decided

- **Authorship:** sole author, Ryota Sato (decided 2026-11-19).
- **Projector:** Lubell LL916C with the AC205C, bought, driven by an off-the-shelf class-D amplifier with bridged outputs. The safety procedure is in the checklist, §3.
- **Pool:** the lead's own pool, available any time.
- **Speech transport:** baseband. The pool plan plays the speech files directly, with no ultrasonic carrier.

_Last updated: 2026-11-19_
