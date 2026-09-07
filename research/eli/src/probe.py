#!/usr/bin/env python3
"""Generate the probe WAV the underwater speaker plays from its SD card.

Layout per "pose block" (repeat as many as needed):
  1.0 s  sync sweep 500 Hz -> 20 kHz (for coarse alignment / clock-rate estimate)
  N x    50 ms linear chirps 2-18 kHz, spaced `gap` s (pool RT60 ~3 s -> gap 2 s)
  1x     2 s exponential sweep 500 Hz -> 20 kHz (full-band CIR)
  1x     1 s pink noise burst

Usage: probe.py out.wav [--fs 48000] [--chirps 20] [--gap 2.0] [--blocks 1]
The speaker is band-limited to ~20 kHz and plays from its own clock; the fitted
model absorbs the clock ratio as a learnable time-scale. Keep 48 kHz so the file
is small and any consumer DAC plays it.
"""
import argparse
import numpy as np
import soundfile as sf


def lin_chirp(fs, dur, f0, f1, amp=0.8):
    t = np.arange(int(fs * dur)) / fs
    k = (f1 - f0) / dur
    x = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    w = np.hanning(int(0.005 * fs) * 2)
    n = w.size // 2
    x[:n] *= w[:n]; x[-n:] *= w[n:]
    return amp * x


def exp_sweep(fs, dur, f0, f1, amp=0.8):
    t = np.arange(int(fs * dur)) / fs
    R = np.log(f1 / f0)
    x = np.sin(2 * np.pi * f0 * dur / R * (np.exp(t * R / dur) - 1))
    w = np.hanning(int(0.01 * fs) * 2); n = w.size // 2
    x[:n] *= w[:n]; x[-n:] *= w[n:]
    return amp * x


def pink(fs, dur, amp=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs); f[0] = f[1]
    x = np.fft.irfft(X / np.sqrt(f), n)
    return amp * x / np.abs(x).max()


def block(fs, chirps, gap):
    parts = [exp_sweep(fs, 1.0, 500, 20000), np.zeros(int(fs * gap))]
    for _ in range(chirps):
        parts += [lin_chirp(fs, 0.05, 2000, 18000), np.zeros(int(fs * (gap - 0.05)))]
    parts += [exp_sweep(fs, 2.0, 500, 20000), np.zeros(int(fs * gap)),
              pink(fs, 1.0), np.zeros(int(fs * gap))]
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--chirps", type=int, default=20); ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--blocks", type=int, default=1)
    a = ap.parse_args()
    x = np.concatenate([block(a.fs, a.chirps, a.gap) for _ in range(a.blocks)])
    sf.write(a.out, x.astype(np.float32), a.fs, subtype="PCM_16")
    print(f"wrote {a.out}: {x.size / a.fs:.1f} s at {a.fs} Hz")


if __name__ == "__main__":
    main()
