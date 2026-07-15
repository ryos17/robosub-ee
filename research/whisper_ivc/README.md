# Whisper Intervehicle Communication Research

Talk to the sub: hydrophone audio is streamed from a Daisy Seed to the Orin
and transcribed live on the GPU with Whisper.

## Pipeline

```
hydrophones -> Daisy Seed (stream_audio.cpp firmware, 96 kHz codec, host-set
rate 16-96 kHz and 16/24-bit, PCM over USB CDC) -> daisy_stream.py ->
record / meter / transcribe scripts

The boards boot at 96 kHz / 16-bit; the host changes it live with the serial
commands `rate 96000|48000|32000|24000|16000` and `bits 16|24`. While
recording, stream_transcribe.py also saves each board's untouched stereo
stream as data/<session>/raw_ch01.wav / raw_ch23.wav at the native rate and
depth — full-fidelity, pinger band (25-40 kHz) included. The TDOA ports in
../tdoa consume those WAVs (or the live stream) directly.
```

The firmware lives in `gen2/daisyseed_firmware/stream_audio.cpp` (flash with
`make flash CURRENT_PROGRAM=stream_audio`). Each Daisy streams its own two
codec channels; ch0 is the primary voice hydrophone.

## Setup

```sh
conda env create -f environment.yaml
conda activate whisper_ivc
```

Read the comments in `environment.yaml`: the Jetson torch wheels are pinned by
URL, and recreating the env needs the `activate.d/jetson_cuda_libs.sh`
LD_LIBRARY_PATH script (cudss/cusparselt only — never add pip's cublas).

GPU status: use `jtop`, not nvidia-smi (NVML can't read Tegra iGPUs).

## Scripts (all auto-detect the streaming Daisy port; `--port` to override)

- **`stream_transcribe.py`** — live transcription AND recording. The stream is
  cut into consecutive `--window` second chunks (default 10 s); per chunk the
  selected channel (`--channel 0|1|mix`, default 0) is normalized, optionally
  denoised (`--denoise off|metricgan|sepformer`, default off), recorded to
  `data/<date_time>/<N>.wav`, and transcribed (whisper large-v3 default).
  The saved wav is byte-identical to what whisper hears; `--no-record`
  disables saving.
- **`daisy_stream.py`** — shared library: port auto-detection (probes for the
  0xDA7A frame magic, so it picks the streaming board even with two Daisies
  attached) and frame parsing.

Only one script can hold the serial port at a time — stop the transcriber
before recording or metering.

## Known physics

Speech in air couples very weakly into these piezo hydrophones (they're
matched to water): talk loudly within a few cm of the element and expect
window RMS ≥ 0.005 when it's working. In-water coupling is far better and is
the real use case.
