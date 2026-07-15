# TDOA / pinger detection ports

Host-side Python ports of the Daisy Seed detection firmwares, algorithm- and
settings-exact, so the pinger pipeline can run on the Orin against the 96 kHz
hydrophone stream (or its recorded WAVs) at the same time as the Whisper IVC
demo (`../whisper_ivc`).

`master.py` is one entry point with three modes, each a line-faithful port of
the matching `master_*.cpp`, run with the firmware's **competition** settings
(the values that ship commented-out as "Competition Configuration"; the boards
default to the looser "Testing" values):

| mode | ported from | FFT | target | output |
|---|---|---|---|---|
| `level` | `master_level.cpp` | 64-pt | 35 kHz | level log every ms |
| `ping` | `master_ping.cpp` | 64-pt | 25 kHz | front/back verdict |
| `tdoa` | `master_tdoa.cpp` | 1024-pt | 1.76 kHz | edge timestamps (µs) |

Competition vs. testing, per mode: `level` targets 35 kHz (not 25 kHz);
`ping` targets 25 kHz with threshold 0.04 and a 3 ms arrival window (not
14.08 kHz / 0.02 / 1 s); `tdoa` already shipped on its competition block
(1.76 kHz, maxes 9/12). Hydrophones 2/3 use `slave.cpp`'s competition config
(maxes 3.5/3.5, 25 kHz) in every mode, since the slave board runs one fixed
program regardless of the master.

`fft_library.py` is a line-faithful port of `library/fft_library.cpp`:
recursive radix-2 Cooley-Tukey (not numpy's FFT), the same Hanning window,
the same bin-window magnitude sum, float32 throughout. It is parity-checked
against a literal scalar transliteration of the C++ by `tmp/test_fft_parity.py`
(a dev test, kept out of this package).

## Input

Both sources come from `audio_source.py`:

```sh
# offline: the raw hi-fi per-channel WAVs stream_transcribe.py records
python master.py ping --raw-dir ../whisper_ivc/data/<session>

# live: record N seconds from the connected stream_audio boards, then analyze
python master.py ping --live 10
```

`--raw-dir` accepts a session directory or its `raw/` subdirectory and reads
every `raw/ch<C>/0.wav` (one gapless mono file per hydrophone, at the native
rate and bit depth). Run inside the `whisper_ivc` conda env (numpy / tqdm /
pyserial).

## Faithfulness notes (differences that were forced, all documented in code)

- **Gain**: `stream_audio` bakes `kGain = 100` into every sample — the same
  ×100 the detection firmwares apply in their audio callbacks — so the ports
  do not multiply again.
- **Hydrophones 2/3**: on the vehicle these arrive as analog levels via the
  slave's DAC → master's ADC. The ports compute them with `slave.cpp`'s exact
  pipeline (64-pt FFT, 25 kHz target) directly from the recorded channels.
  Because slave.cpp is fixed, in `level`/`tdoa` the slave detects 25 kHz while
  the master detects its own (different) target — exactly as the boards do.
- **Clock**: timestamps derive from the sample clock (`sample_index/rate`),
  not `System::GetUs()` polling — sample-accurate rather than loop-jittered.
- **Blocks**: the ports process consecutive non-overlapping FFT blocks; the
  firmware skips whatever arrives while it is busy processing.
