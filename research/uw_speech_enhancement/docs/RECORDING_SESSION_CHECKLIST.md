# Pool recording session checklist (v0)

This checklist takes a session from desk preparation to imported cuts, so the first real take goes straight through `scripts/import_recording.py`. The protocol behind it is RESEARCH_PLAN.md §6.2 (safety), §6.3 (logging), §6.4 (calibration), §6.5 (playback and sync), §6.6 (positions) and §6.8 (metadata). The tools:

- `scripts/make_playlists.py` and `scripts/make_session_audio.py` build the playback files.
- `scripts/run_session.py` produces the plan and run sheet, checks levels, plays and records each take, runs alignment QC, and generates the import commands.
- `scripts/record_daisy.py` records one Daisy board (two hydrophones) per take (Linux).
- `configs/session_channels_v0.json` is the channel map: which recorder file channel is the loopback, the monitor, or hydrophone `ch<N>`.

Commands below use `S=sessions/<date>_s<N>` as the session directory and `B=data/bench/v0` as the bench.

## 1. Days before (desk)

- [ ] Approvals in hand: written Aquatics approval, and the EH&S answer on powered equipment and thrusters (§6.2).
- [ ] Bench built and frozen:
  ```
  python scripts/make_playlists.py --out $B
  python scripts/make_session_audio.py --bench $B
  ```
  Commit the git hash of the bench build to the session notes. Never rebuild the bench after a take: the import reads the same `schedule.json`, `blocks/*.json` and `session/*_speech.json` that produced the played files.
- [ ] Chirp start frequency `f_lo` set from the projector's usable band (`make_playlists.py --f-lo`, default 500 Hz).
- [ ] Channel map edited to the rig: copy `configs/session_channels_v0.json`, set each recorder's rate, roles and hydrophone `channel_id`s, and the scored channel. Each hydrophone keeps one `channel_id` for the whole benchmark.
- [ ] Plan and run sheet:
  ```
  python scripts/run_session.py plan --bench $B --session <N> --channel-map <map.json> --session-dir $S
  ```
  Print `$S/RUN_SHEET.md` (take order, file names per recorder, thruster cue times, blank log). `$S/utterances_<take>.tsv` lists every utterance of each played file with its time in the file.
- [ ] Expected levels at the planned amplifier setting:
  ```
  python scripts/run_session.py levels --session-dir $S --vrms-at-0dbfs <interface Vrms at 0 dBFS> --amp-gain-db <dB>
  ```
  The nominal source level must sit at 150–160 dB re 1 µPa @ 1 m, speech-active RMS (§6.5). It is an uncalibrated estimate from the projector's nominal sensitivity.
- [ ] Dry run at the desk:
  - Interface out1 goes to a dummy load or a small speaker, and out1 also loops through the attenuator into in1.
  - Run `run_session.py take --take-id p1_anchor`. It must print `PASS` for the loopback.
  - Record the same take on both Daisy boards with `record_daisy.py` and a hydrophone in a bucket next to the speaker or projector.
  - Run `run_session.py check --take-id p1_anchor --rec daisyA=... --rec daisyB=... --run-import` and confirm cuts appear in the import directory with the right `ch<N>` ids.
  - Then delete the dry-run session directory and import directory.
- [ ] `pip install sounddevice` on the playback laptop. Confirm the interface's device name (`python -m sounddevice`).

## 2. Pack list

- [ ] Projector, cable and box; amplifier (battery), charged; attenuator for the loopback (40 dB).
- [ ] USB playback interface; playback laptop (battery), charged; Daisy capture laptop (Linux), charged; both Daisy boards and cables.
- [ ] Hydrophones, preamps, monitor hydrophone; spare cables.
- [ ] Depth-marked poles, rope and weights; flooded or perforated mounts only (no air-filled tubing near the hydrophones); tape measure and laser meter.
- [ ] Thermometer, salinity strip, phone or camera for photos.
- [ ] Thruster rig: units, ESC, E-stop, guard frame, battery; isolated from the array mount.
- [ ] Printed run sheet and this checklist.

## 3. At the pool, before the first take

- [ ] Pool clear of swimmers. Nobody in the water while transmitting (§6.2).
- [ ] Log on the run sheet: pool, water temperature at 0.5 m and 1.5 m, pump state, pool activity, operators, photos of the rig.
- [ ] Rig the projector at 1.0 m depth (≥ 1 m for the LL916) and the hydrophones at 1.0 m depth, on-axis. Keep the monitor hydrophone ≥ 0.5 m from the projector, 90° off-axis.
- [ ] Wire the channels exactly as the run sheet's channel map. Check each input by tapping its hydrophone and watching the right channel move.
- [ ] Set the amplifier gain once, at the value used for `levels`, and write it on the run sheet. **Never touch it again during the session.** Set the recorder gains and write them down.
- [ ] Live level check (plays 15 s of the first speech file and measures the loopback):
  ```
  python scripts/run_session.py levels --session-dir $S --vrms-at-0dbfs <V> --amp-gain-db <dB> --live --in-dbfs-at-1vrms <in1 dBFS at 1 Vrms> --loopback-atten-db 40 --device "<name>"
  ```
  Check:
  - The measured drive is within about 1 dB of the estimate.
  - No input clips.
  - The hydrophone peaks sit well below 0 dBFS, at least 6 dB of headroom, at the 1 m position.
- [ ] Pinger band: check that a raw 96 kHz hydrophone capture shows no energy in 25–40 kHz (§6.2). If it does, lower the drive.

## 4. Each position (in run-sheet order)

- [ ] Measure and log the range (projector to hydrophone), both depths, and the positions on deck.
- [ ] For each take of the position (noise, sweeps, speech):
  1. Start both Daisy recorders, with the file names from the run sheet:
     ```
     python scripts/record_daisy.py --board 376636603433 --out $S/raw/daisyA/s<N>_<take>_daisyA.wav
     python scripts/record_daisy.py --board 376C36573433 --out $S/raw/daisyB/s<N>_<take>_daisyB.wav
     ```
  2. Play and record the interface. This prints the thruster cues live, runs chirp QC on the loopback, and logs the metadata:
     ```
     python scripts/run_session.py take --session-dir $S --take-id <take> --device "<name>" --range-m <r> --proj-depth-m 1.0 --hyd-depth-m 1.0 --water-temp-c <T> --amp-gain-db <dB> --pumps <on|off> --operator <initials>
     ```
     For thruster blocks, add `--thruster-unit A --throttle-pct <p>`. Switch the thruster exactly at the printed `THRUSTER ON` and `THRUSTER OFF` cues. Each cue falls right after a block's end chirp, which leaves about 3.5 s of spin-up before the next block.
  3. Stop both Daisy recorders (Ctrl-C) only after the file has ended. Check that each sidecar `.json` says `dropped: 0`.
  4. For speech takes, run QC on the hydrophone files:
     ```
     python scripts/run_session.py check --session-dir $S --take-id <take> --rec daisyA=$S/raw/daisyA/s<N>_<take>_daisyA.wav --rec daisyB=$S/raw/daisyB/s<N>_<take>_daisyB.wav
     ```
     Every file must print `PASS`: all chirps found, no flagged block, no clipping. On `FAIL`, read the reason (missed chirps, clipping, wrong channel count) and redo the take before moving the rig.
- [ ] Move the rig only after the speech take of this position has passed QC on every recorder.

## 5. End of session

- [ ] Back at the first position: play the anchor (`--take-id p1_anchor`) and check it as above.
- [ ] Extra noise-bank takes (nothing playing), 1–2 min each: quiet pool, thruster, bubbles or splashing, pumps.
- [ ] Log the water temperature at the end.
- [ ] Copy `$S` to two places before leaving the pool. It holds the raw files, `takes.jsonl` and the run sheet.

## 6. After the session (desk)

- [ ] Import every speech and anchor take. Either run `check ... --run-import` per take, or run the commands collected in `$S/import_cmds.txt`. Cuts and `align.jsonl` go to `data/real/<release>_s<N>`.
- [ ] Review `blocks_<take>.json` for each take: take drift, minimum PSR and flagged blocks.
- [ ] Merge the session log (`takes.jsonl` and the run sheet) into the §6.8 metadata: geometry, gains, temperatures, thruster.
- [ ] Score with the frozen evaluation protocol (RESULTS_E7_SIM_DRYRUN.md, "Frozen evaluation protocol"). Report any deviation as a deviation.

## Notes

- **Sync.** A recorder that records the loopback is aligned on it, which is exact. A recorder without one is aligned on its own hydrophone's chirps, which carry the acoustic delay. Both paths are tested on synthetic takes with clock drift up to ±150 ppm (`tests/test_ingest.py`, `tests/test_session_run.py`).
- **Separate clocks.** Each Daisy board is a separate clock and a separate file. The import corrects drift per block from that file's own chirps, so the boards need no common clock.
- **Frame drops.** A frame drop removes 32 or 64 samples (0.3–0.7 ms at 96 kHz) from a Daisy file.
  - A drop between blocks is absorbed, because each block is located from its own chirps.
  - A drop inside a block shortens it. The per-block drift fit spreads the error over the block, so the cuts are off by at most the dropped length. That is small against the 0.25 s padding, but it does affect fine alignment against R1.
  - QC will not catch a single drop: 64 samples in a 60 s block looks like about 11 ppm of extra drift. Rely on the sidecar `dropped` count instead. If a take has drops, compare each block's `drift_ppm` in `blocks_<take>.json` with the take drift, and redo the take if many blocks disagree.
