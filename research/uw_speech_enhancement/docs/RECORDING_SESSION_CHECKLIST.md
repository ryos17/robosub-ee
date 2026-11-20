# Pool recording session checklist (pool plan v1, 2026-11-19)

This checklist covers one session, from desk preparation to the numbers in the abstract. The protocol behind it is [`PROTOCOL.md`](../PROTOCOL.md). The binding version of that protocol is the amendment "Amendment 2026-11-19: no-thruster pool plan" in [`RESULTS_E7_SIM_DRYRUN.md`](../RESULTS_E7_SIM_DRYRUN.md). Anything not decided there is not decided, so ask the lead.

The thruster S1 checklist that this replaces is in git history. S1 was never recorded. `data/bench/v0_s1` and `data/bench/v0_pilot` are frozen and must not be touched.

**Machines.**
- **PC:** the Windows PC. It plays the audio and does every check, import and scoring step. It runs in the `uwse` conda env, from the project root `research/uw_speech_enhancement`.
- **Linux:** the Daisy capture machine (laptop or the Orin). It runs only `record_daisy.py`. It has no session dir and no bench (both are git-ignored), so `run_session.py` never runs there: the `record_daisy.py` lines come from the PC (`RUN_SHEET.md`, `daisy_cmds.txt`, or `run_session.py daisy-cmds` run on the PC) and are typed or pasted on Linux.

**Names used below.**
- Session number `2`.
- `S=sessions/<date>_s2`, the session dir.
- `B=data/bench/v1`, the bench.
- `R=data/real/v1_s2`, the import dir.

The session number is a suggestion; change it everywhere at once if needed.

**Long jobs.** The bench build, imports, enhancement and scoring each take minutes to hours. Run them on the PC in a detached window and tee them to `$env:TEMP\<name>.log`, as in `CODE.md` §5. For example:

```powershell
Start-Process powershell -ArgumentList @(
    '-NoExit','-Command',
    "Set-Location 'C:\Users\ryota\Documents\repos\robosub-ee-research\research\uw_speech_enhancement'; & 'C:\Users\ryota\anaconda3\shell\condabin\conda-hook.ps1'; conda activate uwse; <command> *>&1 | Tee-Object -FilePath `"$env:TEMP\<name>.log`""
) -WindowStyle Minimized
```

## 1. Days before (desk)

### Bench (PC, CPU)

- [ ] Build the bench once:
  ```
  python scripts/make_playlists.py --plan pool --librispeech-root data/librispeech --out data/bench/v1
  python scripts/make_session_audio.py --bench data/bench/v1
  ```
  - Use the default settings. The optional flags `--pilot-n 20 --bubble-min 10 --target-speech-s 60` and `--max-take-min 35` already hold the frozen defaults.
  - `make_playlists.py` refuses to write into `v0_s1` or `v0_pilot`.
  - `make_session_audio.py` writes the program files (`pilot`, `speech`, `anchor`, `bubble`), `sweeps.wav`, `tone_1k.wav` and `programs.json`. A program longer than 35 min is split into `speech_1.wav`, `speech_2.wav`, …. The speech program is 27.1 min, so it is one file.
  - Last, it writes `data/bench/v1/SHA256SUMS.txt` (one `<sha256>  <path>` line per bench file, sorted posix paths, UTF-8 without BOM, LF) and prints the sha256 of that manifest. `python scripts/make_session_audio.py --bench data/bench/v1 --sums-only` rewrites only the manifest.
- [ ] Freeze the bench.
  - Record the git hash of the build (`git rev-parse HEAD`) and the sha256 of the manifest in the session notes. The manifest hash is the one `make_session_audio.py` printed; to recompute it, run `sha256sum data/bench/v1/SHA256SUMS.txt` (Git Bash) or `(Get-FileHash data/bench/v1/SHA256SUMS.txt -Algorithm SHA256).Hash` (PowerShell).
  - To check the bench files later: `cd data/bench/v1 && sha256sum -c --quiet SHA256SUMS.txt` (Git Bash; silent when all match).
  - Send both hashes to the lead to add to the frozen record.
  - Never rebuild the bench after a take: the import reads the same block and session JSONs that made the played files.
- [ ] Check the selection: `data/bench/v1/selection/pilot_rule.json` must list the 20 pilot sentences (dev-clean, one per speaker, none of them in v0-dev or the test set).

### Plan and run sheet (PC)

- [ ] Channel map: `configs/session_channels_v1.json`.
  - Daisy A (`376636603433`) records ch0 and ch1. Daisy B (`376C36573433`) records ch2 and ch3. The interface plays only. ch0 is scored.
  - Edit the map only if the rig differs. Each hydrophone keeps one `channel_id` for the whole session.
- [ ] Plan:
  ```
  python scripts/run_session.py plan --bench data/bench/v1 --session 2 \
    --channel-map configs/session_channels_v1.json --session-dir sessions/<date>_s2 \
    --distances 1 3 6 --ref-m 1.0 --pilot-m 3.0 --bubble-m 3.0 \
    --pool-length-m L --pool-width-m W --pool-depth-m D --date <date>
  ```
  - Choose the distances to fit the pool. The longest should leave at least 1 m to the far wall.
  - Optional flags: `--no-bubbles`, `--no-dry-run`, `--noise-s 60`, `--bubble-noise-s 120`.
  - To re-plan an existing session dir, pass `--replan`. Recordings are kept.
  - The default import dir is `data/real/v1_s2`.
  - Print `$S/RUN_SHEET.md`. It holds the safety and level steps, the take order, the Daisy command lines and the log columns. `$S/daisy_cmds.txt` has every Daisy line, and `$S/utterances_<take>.tsv` lists the utterances of each played file.
- [ ] Expected levels:
  ```
  python scripts/run_session.py levels --session-dir sessions/<date>_s2 --target-sl-db 160 --brick-v <V>
  ```
  - The output is the 1 kHz tone Vrms to dial in for the target source level: about 2.0 Vrms for 160 dB re 1 µPa @ 1 m speech-active, or 0.63 Vrms for 150 dB.
  - It also prints, per file, the drive **predicted** at that tone setting (active Vrms, peak V) and the largest peak drive.
  - `<V>` is the voltage printed on the amplifier's DC brick.
  - If the command warns that the peak drive exceeds 0.9 × brick voltage, lower `--target-sl-db` and re-run until it does not. The speech files have a crest factor of about 23 dB.

### Linux capture machine

- [ ] Put the repository (branch `ryota_research`) on the Linux machine. `record_daisy.py` imports `../whisper_ivc/daisy_stream.py`, so keep the `research/` layout.
- [ ] Install Python 3 with `numpy`, `soundfile` and `pyserial` (all that `record_daisy.py` needs; nothing else of the project runs on Linux).
- [ ] Give the user access to the serial port (for example, add it to the `dialout` group).
- [ ] With both boards plugged in, run `python scripts/record_daisy.py --board 376636603433 --rate 96000 --seconds 10 --out /tmp/testA.wav`, and the same for `376C36573433`.
  - Each file must have 2 channels at 96 kHz.
  - Each sidecar `.json` must say `dropped: 0`.
- [ ] Check the clock and free disk space. A full session (three distances) is about 2 h of 2 × 2-channel 96 kHz / 24-bit audio, roughly 8–9 GB.

### Playback on the PC (desk, before the dry run)

- [ ] The `uwse` env needs `sounddevice` for `tone` and every played `take` (it is in `environment.yml`, but the env built before 2026-11-19 lacks it). The lead installs it: `conda activate uwse; pip install sounddevice`. Without it these commands stop with an install hint.
- [ ] With the USB interface plugged in, find its output device:
  ```
  python -c "import sounddevice; print(sounddevice.query_devices())"
  ```
  Note the interface's index or name. If it is not the default output, pass it as `--device <index or name>` to every `tone` and `take` (and `--out-channel` if out1 is not the first output).
- [ ] Play the tone once with the amplifier **off**, to prove the playback path: `python scripts/run_session.py tone --session-dir sessions/<date>_s2 --repeat 1 [--device …]`.

### Dry run (desk, before the pool day)

- [ ] Wire the full chain with the projector **submerged in a bucket of water** (never driven in air):
  - PC and USB interface out1 to the amplifier;
  - one amplifier channel's + and − to the AC205C primary;
  - AC205C to the LL916;
  - hydrophones in the same bucket, into both Daisy boards on the Linux machine.

  Follow the safety rules in §3 even at the desk. Set the knob just above minimum: with `tone` playing, raise it only until the DMM reads about 0.1 Vrms. At the literal minimum the projector can be silent and the chirp QC cannot pass; if QC fails for a weak chirp, raise the knob a little (to about 0.3 Vrms at most) and redo the dry run as attempt 2.
- [ ] Run the `dryrun` take exactly like a pool take (§4). Its files go to `$S/dryrun/raw/daisyA|daisyB/`, never to `$S/raw/`.
- [ ] Check it:
  ```
  python scripts/run_session.py check --session-dir sessions/<date>_s2 --take-id dryrun
  ```
  It must print `dry run: PASS`. The dry run is QC only and is never imported.
- [ ] On the PC, run the pilot scoring commands of §4.3 once on a non-session test if time allows, so Whisper large-v3 and the bench are cached on the PC before the pool day.

### Pack list (the real gear)

- [ ] **Projector and transformer:** Lubell LL916C-050-CXA (caged, 50 ft cable) and the Lubell AC205C transformer.
- [ ] **Amplifier:** Amazon B07VHG2LR9 amplifier (TPA3255, bridged outputs) and its mains DC brick. Note the brick voltage.
- [ ] **Amplifier wiring:** speaker wire from one amplifier channel to the AC205C.
- [ ] **Playback:** RCA (or 3.5 mm to RCA) cable from interface out1 to the amplifier input; USB audio interface and its cable.
- [ ] **Mains safety:** a tested GFCI (outlet or portable GFCI adapter) and an extension long enough to keep the brick and amplifier on dry ground well back from the water.
- [ ] **Meter:** a battery DMM with AC volts, floating (not connected to anything else).
- [ ] **Knob:** tape for the amplifier knob.
- [ ] **PC:** the Windows PC, charged, with its charger.
- [ ] **Capture:** the Linux machine (laptop or Orin, with its power); both Daisy Seed boards and their USB cables; the 4 hydrophones and their cables; spare cables.
- [ ] **Rigging:** rope with depth marks, weights and flooded or perforated mounts. No air-filled tubing near the hydrophones. A tape measure and a laser meter.
- [ ] **Bubble pass:** air pump, air stone and tubing.
- [ ] **Water:** a thermometer.
- [ ] **Records:** a phone or camera for rig photos.
- [ ] **Backups:** two backup drives (USB SSD or stick, at least 32 GB each).
- [ ] **Paper:** the printed run sheet and this checklist.

## 2. At the pool: rigging and metadata

- [ ] **Pool.** Log the pool length, width and depth (they should match the `plan` values; otherwise re-plan with `--replan` before the first take), the water temperature, and photos of the rig.
- [ ] **Projector depth.** The projector goes at least 1 m below the surface (the Lubell range is 1–18 m).
- [ ] **Geometry.**
  - The speaker and the hydrophone are at the same depth.
  - The hydrophone is on the speaker's horizontal axis.
  - The projector rotation is fixed. Mark it and log it as `--orientation`, because the LL916 has a 5 kHz notch and a clover-leaf polar pattern near 5 kHz.
  - Keep the same depth and rotation at every distance.
- [ ] **No pre-compensation.** Do **not** pre-compensate the 5 kHz notch. Scoring uses the measured chain reference.
- [ ] **Channel check.** Wire the hydrophones exactly as the channel map. Tap each hydrophone and watch the right Daisy channel move.

## 3. At the pool: safety and amplifier procedure (once, before any sound)

The amplifier output is **bridged**: both speaker terminals are live, and neither is ground.

- [ ] **Never ground the output side.** Never tie a speaker terminal, the AC205C primary or the cable shield to any ground: not the interface, the laptop, an RCA shield or a meter's ground.
- [ ] **One channel only.** Use one amplifier channel's + and − to the AC205C.
- [ ] **Measure floating.** Measure the drive across the two amplifier output terminals, with the floating battery DMM on AC volts.
- [ ] **Mains placement.** The mains brick and the amplifier go on a tested GFCI, on dry ground well back from the water.
- [ ] **Nobody in the water while playing.** If someone must be in the water, keep the source level at or below about 150 dB re 1 µPa @ 1 m.
- [ ] **Power up.** Set the amplifier knob to minimum. Power the interface first, then the amplifier. To power down, reverse the order: amplifier off first, then the interface.
- [ ] **Set the level with the tone:**
  ```
  python scripts/run_session.py tone --session-dir sessions/<date>_s2
  ```
  - While the tone plays, raise the knob only until the DMM reads the target Vrms from the desk `levels` run: about 2.0 Vrms for 160 dB, or 0.63 Vrms for 150 dB.
  - **Tape the knob.** Never touch it again during the session.
  - The hard cap is **20 Vrms** on the LL916 (Lubell maximum).
- [ ] **Check the set level:**
  ```
  python scripts/run_session.py levels --session-dir sessions/<date>_s2 --tone-vrms <DMM reading> --brick-v <V>
  ```
  - It prints the active Vrms, the peak drive V, the crest factor and the estimated source level for each file.
  - It warns when a speech file is above 160 dB, when any drive is above 20 Vrms, or when the peak drive exceeds 0.9 × brick voltage. On a warning, lower the knob, re-read the tone and re-run `levels`.
  - `sweeps.wav` is listed on its own line. Its sine sweeps sit about 3 dB above the speech active RMS, so its SL is above 160 dB at the 160 dB setting; that is expected and is not a reason to turn down. It is still held to the 20 Vrms cap and the brick check.
  - With the knob set exactly at the printed target, the speech files read 160 dB and nothing warns. A DMM reading a little above the target gives a small ceiling warning: trim the knob down.
  - Write the DMM reading and the brick voltage on the run sheet. They go into every `take` as `--amp-vrms` and `--brick-v`.
- [ ] **Recorder gains.** Set the recorder gains, then write them down for `--rec-gain`.

## 4. Take loop (in plan order)

The order is:

1. `dryrun` (already done at the desk; skip it if it passed there);
2. `ref_sweeps`, `ref_speech`;
3. `pilot_speech`;
4. for each distance, `p<d>_noise`, `p<d>_sweeps`, `p<d>_speech`, `p<d>_anchor`;
5. `p3bub_noise`, `p3bub_speech`.

The rig only moves between distances. Each setup line on the run sheet says what the rig must be: chain reference on axis at `--ref-m`, pilot at 3 m, air stone on for the bubble takes.

### 4.1 Each take

1. **Linux:** start the two Daisy recorders. Type or paste the take's lines from the printed `RUN_SHEET.md` (or `$S/daisy_cmds.txt`), from the project root on the Linux machine:
   - They are both `record_daisy.py --board … --rate 96000 --seconds … --out … &` lines plus `wait`.
   - They stop on their own after the take length, plus 20 s of lead and 10 s of tail.
   - For a redo, use the attempt-2 lines at the end of `daisy_cmds.txt`. For attempt k ≥ 3, write `_r<k>` in place of `_r2` in both `--out` paths, or print the lines on the **PC** with `python scripts/run_session.py daisy-cmds --session-dir sessions/<date>_s2 --take-id <take> --attempt <k> [--linux-sdir <path on Linux>]` (it reads the session dir and bench, so it runs on the PC only).
2. **PC, within 20 s:** play the take and log its metadata:
   ```
   python scripts/run_session.py take --session-dir sessions/<date>_s2 --take-id <take> [--attempt N] \
     --distance-m <measured> --speaker-depth-m <m> --hyd-depth-m <m> --orientation <mark> \
     --water-temp-c <T> --amp-vrms <DMM> --brick-v <V> --rec-gain <A=..,B=..> --pump on|off \
     --qc-note "<remark>" --notes "<free text>"
   ```
   - For noise takes, nothing plays; stay quiet.
   - For the bubble takes, run the air stone near the speaker and pass `--pump on`.
3. **Copy** both Daisy files (and their `.json` sidecars) from Linux to the PC: `sessions/<date>_s2/raw/daisyA/` and `raw/daisyB/`. The dry run goes to `sessions/<date>_s2/dryrun/raw/...` instead.
4. **PC:** check that the files are complete:
   ```
   python scripts/run_session.py verify --session-dir sessions/<date>_s2 --take-id <take>
   ```
   - It exits 1 if a file is missing or short, has the wrong rate or has too few channels.
   - Each sidecar must say `dropped: 0`.
5. **PC, every take:** QC the alignment on the hydrophone files:
   ```
   python scripts/run_session.py check --session-dir sessions/<date>_s2 --take-id <take> [--run-import]
   ```
   - Speech, anchor and dry-run takes: every file must print `PASS` (all chirps found, no flagged block, no clipping). Noise and sweeps takes print `OK … present` (no chirp QC; `verify` checked them).
   - On `FAIL`, read the reason and redo the take before moving the rig.
   - `check` writes the take's import lines into `$S/import_cmds.txt` with `--out` already set by take (table in §5). It keeps one line per take × recorder: the latest attempt that **passed** QC. A failing attempt appears only as a `# QC FAIL` comment line and never replaces a passing one. Re-running `check` rewrites the file; it does not append.
   - `--run-import` also imports the take at once into that dir (slow: minutes per channel). It is safe on every take, because the routing keeps the pilot, chain-reference and bubble cuts out of the main dir.

### 4.2 Redo

A failed take is redone as attempt 2 (then 3, …): the Daisy lines with `_r2` in `--out` (§4.1 step 1) and `take --attempt 2`. The files are then named `..._r2_<rec>.wav`. Nothing is overwritten: `record_daisy.py` refuses an existing file, and `take` refuses to log the same attempt twice (also with `--no-audio`). `verify` and `check` use the latest attempt on disk; pass `--attempt N` to check an older one. Redo only for QC reasons (missed chirps, clipping, drops, a wrong setup), never because of a transcript.

### 4.3 Pilot scoring and the decision (on site, after `pilot_speech` passes QC)

Before the distance takes, run these on the PC. `check --run-import` imports the pilot into `data/real/v1_s2/pilot` (the `--out` is set by take; nothing to edit):

```
python scripts/run_session.py check --session-dir sessions/<date>_s2 --take-id pilot_speech --run-import
python scripts/make_score_manifest.py --align data/real/v1_s2/pilot/align.jsonl --bench data/bench/v1 --out data/real/v1_s2/pilot/score
python scripts/pilot_wer.py --manifest data/real/v1_s2/pilot/score/manifest.jsonl --out data/real/v1_s2/pilot/pilot_wer.json --device cuda
```

- It prints the C0 WER with its CI and the decision:
  - **≥ 10%:** QUIET primary;
  - **< 10%:** ADDED-NOISE primary (pool noise at 0 dB).
- Write both on the run sheet.
- `pilot_wer.json` is written once. Never re-run it with `--overwrite` to change the decision.
- The decision changes nothing on site: record every remaining take as planned.

## 5. End of session

- [ ] Log the water temperature at the end. Take photos of the final rig.
- [ ] Power down: amplifier off first, then the interface.
- [ ] Every take's Daisy files must be on the PC under `$S/raw/`, and the final `verify` must pass for every take:
  ```
  python scripts/run_session.py verify --session-dir sessions/<date>_s2
  ```
- [ ] **Two backup copies before leaving the pool.** Copy the whole of `$S` (raw files, sidecars, `takes.jsonl`, `RUN_SHEET.md`, `session.json`) to both backup drives. Keep the originals on the Linux machine until the lead has the copies.
- [ ] Run `check` on every speech and anchor take that has not passed it yet.
- [ ] **Import** (PC, slow: drift correction takes minutes per channel for a 27 min take, so run it detached). Peak memory is about 2–3 GB per import (only the cut channels are loaded, one at a time, as float32); run one import at a time.
  - Run the lines in `$S/import_cmds.txt` as they stand (lines starting with `#` are QC failures and are skipped), or `check --take-id <take> --run-import` per take. The `--out` of each line is already set by take:

    | Take | `--out` |
    |---|---|
    | `p<d>_speech`, `p<d>_anchor` | `data/real/v1_s2` (the main dir: the quiet condition) |
    | `pilot_speech` | `data/real/v1_s2/pilot` (already done in §4.3) |
    | `ref_speech` | `data/real/v1_s2/chainref` |
    | `p3bub_speech` | `data/real/v1_s2/bubble` |

  - Only the `p<d>_speech` and `p<d>_anchor` cuts belong in `data/real/v1_s2/align.jsonl`. `make_score_manifest.py` and `score_session.py` take every row of an `align.jsonl`, so mixed rows would enter the quiet condition; the routing above prevents that.
  - Re-importing a take replaces its rows in `align.jsonl`; it does not duplicate them.
  - Then review `blocks_<take>.json` for each take: take drift, minimum PSR and flagged blocks.

## 6. After the session (desk, PC)

- [ ] Run the sweep and noise analysis (CPU, same day):
  ```
  python scripts/analyze_session.py all --session-dir sessions/<date>_s2
  ```
  - It writes `$S/analysis/`: IRs, EDC, T20/T30, DRR and drift per sweep take (including `ref_sweeps`, the chain reference), and band levels and spectra per noise take.
  - The levels are uncalibrated, so compare them only within a channel.
- [ ] Build the score manifests (R0 cuts):
  ```
  python scripts/make_score_manifest.py --align data/real/v1_s2/chainref/align.jsonl --bench data/bench/v1 --out data/real/v1_s2/chainref/score
  python scripts/make_score_manifest.py --align data/real/v1_s2/align.jsonl          --bench data/bench/v1 --out data/real/v1_s2/score --c8 --speaker-sex data/librispeech/LibriSpeech/SPEAKERS.TXT
  python scripts/make_score_manifest.py --align data/real/v1_s2/bubble/align.jsonl   --bench data/bench/v1 --out data/real/v1_s2/bubble/score --c8
  ```
- [ ] Score the chain-reference WER (C0 only, no decision):
  ```
  python scripts/pilot_wer.py --manifest data/real/v1_s2/chainref/score/manifest.jsonl --out data/real/v1_s2/chainref/chainref_wer.json --role chainref --device cuda
  ```
- [ ] Make the added real noise, at 0 and 5 dB. The noise files are the daisyA files of the noise-only takes:
  ```
  python scripts/mix_real_noise.py --manifest data/real/v1_s2/score/manifest.jsonl \
    --noise pool@p1=sessions/<date>_s2/raw/daisyA/s2_p1_noise_daisyA.wav \
    --noise pool@p3=sessions/<date>_s2/raw/daisyA/s2_p3_noise_daisyA.wav \
    --noise pool@p6=sessions/<date>_s2/raw/daisyA/s2_p6_noise_daisyA.wav \
    --noise bubble=sessions/<date>_s2/raw/daisyA/s2_p3bub_noise_daisyA.wav \
    --snr 0,5 --out data/real/v1_s2/added
  ```
  - Add one `pool@p<d>=` per distance actually recorded.
  - If a noise take was redone, use its latest `_r<k>` file.
- [ ] Score everything (GPU, hours: run it detached):
  ```
  python scripts/score_session.py --out runs/score/v1_s2 \
    --condition quiet=data/real/v1_s2/score/manifest.jsonl \
    --condition bubblelive=data/real/v1_s2/bubble/score/manifest.jsonl \
    --added-noise-dir data/real/v1_s2/added --enhance C1,C2,C10 \
    --align data/real/v1_s2/align.jsonl --align data/real/v1_s2/bubble/align.jsonl --splits v0-test \
    --speaker-sex data/librispeech/LibriSpeech/SPEAKERS.TXT \
    --pilot data/real/v1_s2/pilot/pilot_wer.json --chainref data/real/v1_s2/chainref/chainref_wer.json --device cuda
  ```
  - `--splits v0-test` drops the v0-dev rows, as the endpoint requires.
  - The script refuses to decode unless the C8 checkpoint's sha256 starts with `5063f02f`.
  - Corpus WER CIs use 10000 resamples (`<cond>/all`). Paired deltas use 2000 (`<cond>/paired_{ALL,<pos>}`).
  - The intrusive metrics are skipped until R1 cuts from the chain reference exist.
- [ ] Make the paper numbers and the figure:
  ```
  python scripts/make_paper_numbers.py --results runs/score/v1_s2
  C:/Users/ryota/anaconda3/python.exe scripts/make_paper_numbers.py --results runs/score/v1_s2 --fig-only
  ```
  - The first command writes `paper/results_real.tex`, then exits 3 in `uwse`, which has no matplotlib. The second writes `paper/fig_real.pdf` from base anaconda.
  - A number that is not available is typeset as a bold `??`. There must be none in the final abstract.
- [ ] Build the abstract. The build steps are in the header of `paper/oceans2027_abstract.tex`; follow that header if it differs from this list.
  - `paper/oceans2027_abstract.tex` reads `results_real.tex` and `fig_real.pdf` when they exist; otherwise it uses `results_placeholder.tex` and the simulation figure.
  - It picks the results paragraph (quiet primary or added-noise primary) from `\PrimCondName`, which comes from the pilot rule.
  - The variant cannot be forced: only the pilot rule decides (a `paper_variant.tex` is no longer read).
  - If the distances chosen on the day are not 1, 3 and 6 m, edit the `\FarDist` / `\BubDist` lines as the header says.
  - Build:
    ```
    cd paper && latexmk -pdf -interaction=nonstopmode -outdir=build oceans2027_abstract.tex
    ```
  - Check that `build/oceans2027_abstract.pdf` has 2 pages, that the log has no "undefined" and no "paper Warning", and that the word count is at most 1000.
  - The abstract is due on 15 January 2027, through the OCEANS 2027 EventsAir portal.

## Notes

- **Sync.** There is no loopback, so every recorder is aligned on the chirps its own hydrophones hear. The cuts carry a few ms of acoustic delay, well inside the 0.25 s padding.
- **Separate clocks.** Each Daisy board has its own clock and writes its own file. The import corrects drift per block from that file's chirps, so the boards need no common clock.
- **Frame drops.** A frame drop removes 32 or 64 samples (0.3–0.7 ms at 96 kHz).
  - A drop between blocks is absorbed.
  - A drop inside a block shortens it by at most that much, which is small against the padding.
  - QC will not catch a single drop. Rely on the sidecar `dropped` count, and redo a take with drops if many of its blocks disagree on `drift_ppm` in `blocks_<take>.json`.
- **RAM.** Playback and session files are streamed block by block. Programs are split at 35 min, so no take holds a whole program in memory. QC loads one channel (float32); an import loads only the channels it cuts, one at a time, as float32 (about 2–3 GB peak for a 35 min take).
