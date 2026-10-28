# uwse code guide

This document describes the `uwse` package: the code behind the underwater speech enhancement sim-to-real study (OCEANS 2027). `RESEARCH_PLAN.md` is the specification. This file covers installation, how the code maps onto the plan, how to run the tests, the smoke test and training, and which numbers are still placeholders until the pool measurements exist.

## 1. Installation

The project uses a conda environment called `uwse`, defined in `environment.yml`. It contains Python, the CUDA 12.4 build of PyTorch, NumPy below 2.1, SciPy, soundfile, tqdm, PyYAML, pytest, jiwer, pystoi, pesq, openai-whisper, SpeechBrain, nara_wpe, peft, transformers, DeepFilterNet and pyroomacoustics.

```powershell
conda env create -f environment.yml          # creates the "uwse" env
conda activate uwse
pip install -e .                             # optional; the tests and scripts also work from the project root
```

On this machine the interpreter is `C:\Users\ryota\anaconda3\envs\uwse\python.exe`. `environment.yml` pins Python 3.10, and the env as built reports Python 3.10. The code is compatible with 3.10 and later.

Optional packages that are not in the env:

- ESPnet and espnet_model_zoo, needed by the `tfgridnet_urgent25` adapter. Without them the adapter raises an ImportError that includes the pip command.
- NeMo, needed by `ParakeetASR`.

Pretrained checkpoints download on first use. SpeechBrain models go to `pretrained_models/`; DeepFilterNet weights go to its own cache in `%LOCALAPPDATA%\DeepFilterNet`; Whisper models go to the Whisper cache. `runs/`, `pretrained_models/`, `tmp/`, `data/*`, audio files and `*.pt` are git-ignored.

Never write anything under `/datasets/`. The scripts that write data (`make_rir_bank.py`, `train.py`, `fetch_librispeech.py`, `smoke_e2e.py`) refuse to write there.

## 2. Package map

| Module | What it does | RESEARCH_PLAN section |
|---|---|---|
| `uwse/audio.py` | Load, save, resample, RMS/dB, fades | shared |
| `uwse/uwsim/reflection.py` | Plane-wave reflection coefficients: fluid–solid half-space (Brekhovskikh), fluid–fluid, pressure-release surface; `Boundary` presets | §4.2 rows 4–5 |
| `uwse/uwsim/geometry.py` | `Pool` with a linearly sloped floor; Avery and MAAS presets; waveguide cutoff c/(4H) | §4.2 rows 3, 9, 13; §6.3 |
| `uwse/uwsim/ism.py` | Signed image-source model; exact tilted-floor images up to order 2; frequency-domain or sinc rendering | §4.1, §4.2 rows 3–6 |
| `uwse/uwsim/physics.py` | Marczak sound speed, salinity correction, Francois–Garrison absorption, Minnaert frequency, octave bands | §4.2 rows 1, 2, 9, 16 |
| `uwse/uwsim/tail.py` | Octave filterbank, stochastic late tail and its splice onto the ISM, time scaling (1+ε), per-band IR perturbation | §4.2 rows 7, 14 |
| `uwse/uwsim/lowband.py` | LR4 crossover, low-band splice, shoebox modal sum | §4.2 rows 8, 9 |
| `uwse/uwsim/chain.py` | Transducer chain: drive LPF, Hammerstein projector, directivity, hydrophone FIR, preamp HPF, ADC | §4.2 rows 10–12, 19, 20 |
| `uwse/uwsim/noise.py` | Noise bank (manifest and split rules), bubble noise, VAD, band-limited SNR mixing | §4.2 rows 15–18; §5.2 |
| `uwse/uwsim/config.py` | `SimConfig` tree, which mirrors `configs/uwsim_v0.yaml` | §4.2 |
| `uwse/uwsim/simulate.py` | Scene sampling, RIR rendering, full chain to (noisy, R1, R0) | §4.1, §4.2, §8.2 |
| `uwse/uwsim/dataset.py` | `RIRBank` (npz shards) and `UWSimPairs` (on-the-fly training pairs) | §5.3 |
| `uwse/irtools/ess.py` | Exponential sine sweep, deconvolution, harmonic IRs, Hammerstein kernels, sweep drift estimation and correction, sweep averaging | §6.4 steps 4–5 |
| `uwse/irtools/analysis.py` | Lundeby, Schroeder EDC, EDT/T20/T30, two-slope fit, DRR, C50, arrivals, 1/3-octave distance | §4.3 step 2, §4.4 |
| `uwse/irtools/soundspeed.py` | Time-of-flight fit of c and comparison with Marczak | §4.3 step 3, §6.9 |
| `uwse/evaluation/align.py` | Canonical sync chirp, chirp detection, drift, fine lag, utterance cuts, `layout_from_timing` | §6.5 |
| `uwse/evaluation/normalize.py` | R0-VAD speech-active RMS normalization to −26 dBFS | §8.3 |
| `uwse/evaluation/asr.py` | Whisper protocol, text normalization, corpus WER (S/D/I), hallucination flags, transcript cache | §8.1, §8.3 |
| `uwse/evaluation/metrics.py` | Band-limited ESTOI/STOI, PESQ-WB, SI-SDR, CI-SDR at 16 kHz | §8.1, §8.2 |
| `uwse/evaluation/stats.py` | Cluster bootstrap, paired CIs, TOST, Holm, MAPSSWE, Kendall τ | §8.5 |
| `uwse/adapters/` | `SEAdapter` registry; identity (C0), wpe1/wpe4, metricganplus, sepformer_wham16k/whamr, dfn3, tfgridnet_urgent25, kiel_unet (C10); `lora.py` (peft LoRA plus LoRA-GRU) | §7.1–7.3 |
| `uwse/train/` | Model-agnostic loop (`fit`/`train`) and datasets for C2/C7/C9 (`SimPairsDataset`, `InAirPairs`, `ReplayPairs`); Whisper-LoRA (C8) | §7.2–7.4 |
| `uwse/bench/` | Metadata schema, LibriSpeech manifests and selection, split guards, playlists, ABBA schedule, playback budget | §5.1, §5.4, §6.5, §6.6, §6.8 |
| `scripts/make_rir_bank.py` | Pre-renders the RIR bank | §5.3 |
| `scripts/make_playlists.py`, `scripts/fetch_librispeech.py` | Builds benchmark playlists; downloads LibriSpeech | §5.1, §6.5 |
| `scripts/train.py` | Training CLI driven by `configs/*.yaml` | §7.3, §7.4 |
| `scripts/score.py` | Scoring CLI: WER, metrics, bootstrap summary and paired comparisons | §8 |
| `scripts/smoke_e2e.py` | End-to-end smoke test | — |

Conventions that cross module boundaries:

- **Sync chirp.** The sync chirp is defined once, in `uwse.evaluation.align.lfm_chirp`: linear FM from f_lo to 8 kHz over 0.25 s, a continuous Tukey(0.1) window, and −20 dBFS peak. `uwse.bench.playlist` imports it, so the chirp that is played and the matched-filter template are the same function. Use `align.layout_from_timing(timing_json)` to turn a bench timing JSON into the alignment layout.
- **Target alignment.** R1 and R0 targets are aligned to the geometric direct arrival, range/c, in both uwsim and the C7 in-air pairs. Peak picking is used only for measured IRs (C9 replay).
- **Amplitude convention.** The ISM and the modal low band both write image amplitudes as R/r, pressure re 1 m.
- **Adapter checkpoints.** For LoRA runs, call `adapter.trainable_params("lora")` before `load_checkpoint`. The base `load_checkpoint` raises if a checkpoint contains LoRA keys that the model lacks, and warns about any other key mismatch.

## 3. Tests

Run the tests from the project root:

```powershell
& "C:\Users\ryota\anaconda3\envs\uwse\python.exe" -m pytest -q                    # default: fast, CPU, offline
& "C:\Users\ryota\anaconda3\envs\uwse\python.exe" -m pytest -q -m slow            # long-running checks
& "C:\Users\ryota\anaconda3\envs\uwse\python.exe" -m pytest -q -m network         # downloads checkpoints (run one file at a time; RAM)
& "C:\Users\ryota\anaconda3\envs\uwse\python.exe" -m pytest -q -m gpu             # needs CUDA
```

The default run deselects anything marked `slow`, `network` or `gpu`; this is set in `pyproject.toml`. Each module has a builder test file and a reviewer test file (`test_*_review.py`). `tests/test_integration.py` covers the cross-module contracts listed above.

## 4. Smoke test

```powershell
& "C:\Users\ryota\anaconda3\envs\uwse\python.exe" scripts/smoke_e2e.py --out runs/smoke
```

The script does the following:

1. Synthesizes a 4 s speech-like signal: a harmonic source with moving formants and pauses.
2. Samples one Avery-pool scene from `configs/uwsim_v0.yaml`.
3. Simulates the noisy signal and the R1/R0 targets at both 16 kHz and 48 kHz.
4. Runs the identity, wpe1, metricganplus and dfn3 adapters. An adapter whose checkpoint cannot be downloaded is reported and skipped.
5. Scores ESTOI, SI-SDR, CI-SDR and PESQ-WB against R1 at 16 kHz and prints a table.
6. Writes WAVs and `results.json` to `runs/smoke/`.

The input is not speech, so the numbers only show that the pipeline runs. With a near-surface source at long range, the direct path and the surface image nearly cancel (Lloyd's mirror). The noisy signal is then almost orthogonal to the direct-path-aligned R1, so SI-SDR of the noisy input is around −45 dB. That is expected; CI-SDR (512-tap filter) is the meaningful number there.

The training smoke test (the W4 throughput measurement) runs about 20 steps on synthetic pairs and writes its throughput and peak GPU memory to `runs/<run_name>_smoke/summary.json`:

```powershell
& "C:\Users\ryota\anaconda3\envs\uwse\python.exe" scripts/train.py --config configs/c2_dfn3.yaml --smoke
& "C:\Users\ryota\anaconda3\envs\uwse\python.exe" scripts/train.py --config configs/c10_kiel_unet.yaml --smoke
```

The first measurement was made on 2026-10-28 on the RTX 4070 Laptop GPU, with synthetic pairs, 20 steps, 64 s of audio per step and fp32. It does not include the cost of uwsim data generation.

| Config | Audio-s per wall-s | Audio-s per compute-s (steady) | Peak GPU memory | Time for 50k steps |
|---|---|---|---|---|
| c10_kiel_unet (16 kHz, batch 16 × 4 s) | 206 | 1318 | 3.25 GB | about 4.3 h |
| c2_dfn3 (48 kHz, 4 × 4 s, accum 4) | 47 | 71 | 0.95 GB | about 19 h |

## 5. Training

Each condition has a YAML file in `configs/`: `c2_dfn3`, `c2_metricganplus`, `c7_inair_dfn3`, `c8_whisper_lora`, `c9_replay_dfn3` and `c10_kiel_unet`. C3 (LoRA) is `c2_dfn3` with overrides; the exact command is in that file's header. Settings can be overridden with `--set key=value`, which is repeatable and parsed as YAML. Use `--dry-run` to print the resolved config.

Long jobs follow the project convention: start them in a detached, minimized PowerShell window and tee the output to `$env:TEMP\<name>.log`.

```powershell
Start-Process powershell -ArgumentList @(
    '-NoExit','-Command',
    "Set-Location 'C:\Users\ryota\Documents\repos\robosub-ee\research\uw_speech_enhancement'; & 'C:\Users\ryota\anaconda3\shell\condabin\conda-hook.ps1'; conda activate uwse; python scripts/train.py --config configs/c2_dfn3.yaml *>&1 | Tee-Object -FilePath `"$env:TEMP\c2_dfn3.log`""
) -WindowStyle Minimized
```

To follow the log, run `Get-Content $env:TEMP\c2_dfn3.log -Wait`. To stop the job, close the window or use `Stop-Process -Id <pid>`.

Runs write to `runs/<run_name>/`:

- `last.pt` and `last_state.pt`, used for exact resume. Relaunching the same command resumes the run, and a run that has already early-stopped stays stopped.
- `best.pt`
- `train_log.jsonl`
- `config.json`
- `summary.json`

Generating the RIR bank takes about 0.3–3 s per RIR on CPU, single process. Launch it the same way:

```powershell
python scripts/make_rir_bank.py --n 20000 --out data/rirs/uwsim_v0 --config configs/uwsim_v0.yaml --resume
```

RAM on the laptop is tight, so run only one heavy process at a time: training, RIR generation, or a network test.

## 6. Uncalibrated placeholders (replace from E0–E4, then the §4.3 fit)

`configs/uwsim_v0.yaml` sets `calibrated: false`. Every value below is a textbook prior or a guess until the named experiment produces data.

**E0 (rig bring-up):**

- ADC/codec noise floor: `chain.adc_floor_dbfs = -110`, with ±3 dB jitter.
- Preamp high-pass corner and order: `chain.preamp_hpf_hz = 100`, order 2.
- Received level range: `chain.level_dbfs = [-35, -15]` dBFS. Check the `meta["clipped"]` rate.
- The scored channel. The adapters default to channel 0.

**E1 (hydrophone calibration):**

- Hydrophone FIR per channel: `chain.hydrophone_fir_path = null`, which means a delta.
- Hydrophone tilt randomization: ±2 dB.
- Mount directivity at 4–8 kHz: not modelled.

**E2 (projector chain):**

- Linear chain response h_chain, the R1 target: `chain.h1_path = null`, a delta.
- Hammerstein kernels: `h2_path` and `h3_path` are null, which means zero.
- Projector directivity: `directivity_path = null`. The default is a piston with `piston_radius_m = 0.117`, applied once at the direct-path angle.
- Drive range: `drive_level_db = [-10, 0]`.
- The usable band, f_lo:
  - `target.f_lo = 500` Hz
  - `noise.band` lower edge 500 Hz
  - the `--f-lo` default in `make_playlists.py`
  - `DEFAULT_F_LO` in `evaluation.align` and `evaluation.metrics`
- The chirp peak level of −20 dBFS.
- The C7 and C9 `chain_fir`. Until it is set, these conditions train toward R0 rather than R1.

**E3 (pool channel):**

- Per-octave T60 and its scale: `acoustics.t60_bands` (1.0 s at 125 Hz down to 0.4 s at 8 kHz), `t60_scale = [0.7, 1.3]`, `t60_two_slope = null`.
- Wall and floor boundary priors (`pool.walls` and `pool.floor`):
  - `cp` 3500–5500 m/s
  - `cs` 2000–3000 m/s
  - `rho` 2200–2500 kg/m³
  - `eta` 0.01–0.05
  - the `CONCRETE_UNCALIBRATED` and `TILE_UNCALIBRATED` presets
- Surface reflection coefficient `surface_R = [-1, -0.9]`.
- The Avery depth profile: which end is deep, and the 3.35–4.27 m values.
- `dim_jitter` and `slope_jitter`.
- Sound speed: `temperature_c = 26` and `c_jitter`. The TOF-slope fit replaces these.
- Per-band IR perturbation SD `ir_perturb_db = 0` (σ_rep, sweep to sweep) and time-scale bound `eps_max = 1e-3`.
- Modal low-band damping, the `modal_rir` default T60 of 1.0 s and 0.8 s at 250 and 500 Hz. Measured low-band IRs replace these for calibrated scenes.
- Early/late split `t_e = [0.03, 0.05]` s and crossover floor `f_x_min = 500` Hz.
- C7 in-air T60 range of 0.25–1.0 s; this should come from measured pool T60s.

**E4 (noise bank) and the E6 SNR pilot:**

- Measured SNR range: `noise.snr_measured = [0, 15]` dB, widened by ±10 dB and clipped to [−10, 25].
- Thruster and ambient probabilities and relative levels: `thruster_prob 0.7`, `ambient_rel_db [-20, -5]`.
- Bubble rate, radius and level.
- `recorded_bubbles_prob = 0`.
- Until a bank exists, `simulate` falls back to synthetic coloured noise, which is for smoke tests only.

**Frozen on v0-dev (§5.4), not from E0–E4:**

- Evaluation VAD settings.
- Fine-lag peak-to-sidelobe ratio (PSR) threshold of 6 dB and the 1 ms main lobe.
- Chirp PSR QC threshold.
- Hallucination thresholds.

## 7. Known limitations

**Simulator:**

- Reflection coefficients are frequency-independent plane-wave values, with no spherical-wave or head-wave correction.
- One boundary material is shared by all four walls.
- The surface R is a constant real scalar, with no ripple.
- Exact tilted-floor images go up to order 2. Higher orders use the mean depth and get no validity check.
- The modal low band is a flat box at the mean depth.
- There is a single receive channel; the 4-element array is not simulated.
- Directivity is applied at the direct-path angle only.
- Recorded bubble noise is not used yet.
- Rendering is about 0.3–3 s per RIR, so use `RIRBank` for training.

**Target definition (needs the lead's sign-off):**

- The target gain defaults to `"rms"`; the literal §8.2 least-squares fit is available as `"ls"`.
- t_e is measured from the direct arrival.
- Targets include the (1+ε) time scaling and the chain's bulk delay.

**Evaluation:**

- Band-limited ESTOI also drops the 1/3-octave bands below f_lo.
- pystoi works at 10 kHz internally, so (E)STOI only covers up to about 4.3 kHz.
- Holm correction is applied across systems on the pooled WER difference only.
- MAPSSWE is a reimplementation that has not been checked against NIST `sc_stats`.
- `ParakeetASR` and Whisper large-v3 have not been run end to end; only the tiny Whisper model has.
- `uwse.evaluation.align` and `uwse.irtools.ess` both define `estimate_drift_ppm` and `correct_drift`, with different signatures: chirp onsets in align, sweep cross-correlation in irtools. Both use the convention that a positive value means the recording clock is fast. Import them from the submodule, not by bare name.

**Adapters:**

- TF-GridNet is untested because ESPnet is not installed.
- SepFormer LoRA covers only the feed-forward Linear layers. The fused attention weights bypass peft.
- The MetricGAN+ discriminator starts from random weights. Call `trainer.warmup()` before training. Its validation loss tracks a moving discriminator, so early stopping should use a fixed metric. Keep `amp: false` for MetricGAN+.
- SepFormer and MetricGAN+ outputs are not level-preserving, so evaluation must level-normalize them.
- The Kiel U-Net is a reconstruction of an unpublished recipe; the choices are documented in `kiel_unet.py`. Its output is band-limited to about 5.8 kHz.

**Training:**

- C7 and C9 differ from C2 in noise composition and level rule, as well as in the room response.
- C8 truncates audio longer than 30 s but keeps the full transcript, so filter the file lists.
- Frozen validation sets are reused by item count only, so change `val_dir` when the config changes.
- Exact resume on GPU needs `train.deterministic: true`.

**Bench:**

- The playback budget in the plan underestimates the total: the correct multiplier is 1.186, giving 3.51 h, not 3.43 h.
- The OpenSLR MD5 values were not verified online.
- The ABBA first state is not counterbalanced across positions.
- Only v0 and v1-test playlists are generated.
