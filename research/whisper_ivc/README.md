# Whisper Intervehicle Communication Research

Talk to the sub: hydrophone audio is streamed from a Daisy Seed to the Orin
and transcribed live on the GPU with Whisper.

## Pipeline

```
hydrophones -> Daisy Seed (stream_audio.cpp firmware, 96 kHz codec, host-set
rate 16-96 kHz and 16/24-bit, PCM over USB CDC) -> daisy_stream.py ->
stream_transcribe.py (CLI or --web)
```

Two boards stream four hydrophones: board `376636603433` carries ch0/ch1,
board `376C36573433` carries ch2/ch3. The boards boot at 96 kHz / 24-bit; the
host changes it live with the serial commands `rate
96000|48000|32000|24000|16000` and `bits 16|24`.

The firmware lives in `gen2/daisyseed_firmware/stream_audio.cpp` (flash with
`make flash CURRENT_PROGRAM=stream_audio`, or `make flash-master` /
`flash-slave` per board). ch0 is the primary voice hydrophone.

### What a session saves — `data/<year_date_time>/`

Per transcribed window `N` (manual mode has one window, `N=0`; continuous mode
has one per `--window` seconds):

- `ch<C>_<N>.wav` + `ch<C>_<N>.txt` — the 16 kHz mono audio whisper actually
  heard (selected channel `C`, or `mix_<N>.wav` for the channel mix) and its
  transcription. Byte-identical to whisper's input.
- `raw/ch<C>/<N>.wav` — every hydrophone's untouched audio at the NATIVE rate
  and bit depth (up to 96 kHz / 24-bit), one mono file per channel per window.
  Full-fidelity, pinger band (25-40 kHz) intact. Continuous-mode fragments
  `0.wav, 1.wav, ...` concatenate back into the whole take. Kept full-rate so
  the pinger band survives for offline analysis.

## Setup

```sh
conda env create -f environment.yaml
conda activate whisper_ivc
```

Read the comments in `environment.yaml`: the Jetson torch wheels are pinned by
URL, and recreating the env needs the `activate.d/jetson_cuda_libs.sh`
LD_LIBRARY_PATH script (cudss/cusparselt only — never add pip's cublas).

GPU status: use `jtop`, not nvidia-smi (NVML can't read Tegra iGPUs).

## `stream_transcribe.py` — record + transcribe

Runs a purely-terminal CLI by default; `--web` serves the browser UI instead.
Both share one UI-agnostic engine, so they behave identically.

```sh
# terminal CLI (default) — prints transcripts + a per-channel spectrum
# sparkline ('*' = the channel whisper is listening to); Ctrl-C to stop
python stream_transcribe.py --model large-v3 --channel 0
python stream_transcribe.py --continuous --window 10 --channel mix
python stream_transcribe.py --channel 2 --rate 96000 --bits 24

# browser UI — 4-row per-channel spectrogram (box on the listened channel,
# all four on 'mix'), live controls, transcript + log
python stream_transcribe.py --web        # then open http://<orin>:7860
```

Two recording modes (both save the files above):

- **manual** (default) — records until you stop (Ctrl-C in the CLI, the Stop
  button in the web UI), then transcribes the whole take.
- **continuous** (`--continuous`) — cuts the stream into `--window`-second
  chunks and transcribes each one live as it completes.

Per window the selected channel is normalized, optionally denoised
(`--denoise off|metricgan|sepformer`), resampled to 16 kHz, and transcribed
(whisper `--model`, default large-v3). CLI flags: `--model --no-model --channel
{0,1,2,3,mix} --denoise --continuous --window --rate --bits --min-db --max-db
--status-every`; `--web --port` for the UI. `--help` lists them all.

- **`daisy_stream.py`** — shared reader: finds each board by USB serial
  (`BOARDS`), decodes the 0xDA7A PCM frames, and sends `rate`/`bits`/`reboot`
  serial commands. Reading is delegated to a `cat` subprocess (own session, so
  a terminal Ctrl-C doesn't kill it) feeding a 1 MB pipe, which keeps the
  GIL-bound host from overflowing the kernel tty buffer.

Only one process can hold the boards at a time.
