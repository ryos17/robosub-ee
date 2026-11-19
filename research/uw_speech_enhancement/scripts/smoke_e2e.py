"""End-to-end smoke test of the uwse pipeline (no corpus download needed).

Chain exercised (RESEARCH_PLAN.md §4.1-4.2 simulator, §7.1 adapters, §8.1-8.2 metrics):

1. synthesize a 4 s speech-like test signal at 48 kHz (harmonic source with a
   gliding F0, formant resonances that move between vowel targets, syllable-rate
   envelope and pauses);
2. sample one calibrated-pool scene (Avery preset, ``configs/uwsim_v0.yaml``),
   render its RIR once and simulate noisy + R1/R0 targets at 16 kHz and 48 kHz
   (same RIR and noise seed for both rates);
3. run the adapters identity, wpe1 and, if their checkpoints load, metricganplus
   and dfn3, each at its own sample rate;
4. score ESTOI / SI-SDR / CI-SDR (and PESQ-WB) of every output against R1 at
   16 kHz, print a table and write WAVs plus ``results.json`` to ``runs/smoke/``.

The signal is not speech, so the numbers only show that the pipeline runs and
behaves sensibly; they are not results. Usage::

    python scripts/smoke_e2e.py [--out runs/smoke] [--seed 0] [--device auto]
                                [--adapters identity,wpe1,metricganplus,dfn3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uwse import audio  # noqa: E402

FS = 48000


def speech_like(dur_s: float = 4.0, fs: int = FS, seed: int = 0) -> np.ndarray:
    """Harmonic + formant test signal, peak-normalised to -6 dBFS, 48 kHz."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    # F0 contour: declining 180 -> 110 Hz with a slow vibrato-like wobble
    f0 = 180.0 - 70.0 * t / dur_s + 8.0 * np.sin(2 * np.pi * 3.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    k = np.arange(1, int(8000 / 110) + 1)[:, None]
    harm = np.sin(k * phase[None, :]) / k  # -6 dB/oct glottal-like tilt
    harm[(k * f0[None, :]) > 0.45 * fs] = 0.0
    src = harm.sum(0) + 0.05 * rng.standard_normal(n)  # a little breath noise
    # vowel targets (F1, F2, F3) per syllable; formants filtered frame-wise (10 ms)
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480), (300, 870, 2240),
              (660, 1720, 2410), (490, 1350, 1690)]
    hop = fs // 100
    y = np.zeros(n)
    zi = [np.zeros(2) for _ in range(3)]
    syl = 0.25  # s per syllable
    for i0 in range(0, n, hop):
        tt = i0 / fs
        a, b = vowels[int(tt / syl) % len(vowels)], vowels[(int(tt / syl) + 1) % len(vowels)]
        w = (tt % syl) / syl
        seg = src[i0 : i0 + hop]
        for j, (fa, fb) in enumerate(zip(a, b)):
            fc = (1 - w) * fa + w * fb
            bw = 60.0 + 0.05 * fc
            r = np.exp(-np.pi * bw / fs)
            den = [1.0, -2 * r * np.cos(2 * np.pi * fc / fs), r * r]
            seg, zi[j] = lfilter([1.0 - r], den, seg, zi=zi[j])
        y[i0 : i0 + hop] = seg
    # syllable envelope with two pauses (VAD and level normalisation see silence)
    env = 0.5 - 0.5 * np.cos(2 * np.pi * (t % syl) / syl)
    env = env ** 0.5
    env[(t > 1.45) & (t < 1.85)] = 0.0
    env[t > 3.6] = 0.0
    env = np.convolve(env, np.hanning(int(0.01 * fs)) / (0.005 * fs), mode="same")
    y = np.diff(y * env, prepend=0.0)  # lip radiation (+6 dB/oct)
    y = audio.fade(y, fs, 0.01, 0.01)
    return 0.5 * y / np.max(np.abs(y))


def run_adapter(name: str, noisy: np.ndarray, device: str) -> np.ndarray:
    import torch

    from uwse.adapters import build, import_errors

    if name in import_errors:
        raise RuntimeError(f"adapter module failed to import: {import_errors}")
    se = build(name, device=device)
    se.load_pretrained()
    with torch.no_grad():
        y = se.enhance(torch.as_tensor(noisy, dtype=torch.float32)[None].to(device))
    out = y.detach().float().cpu().numpy().reshape(-1)
    del se
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / "smoke"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "uwsim_v0.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--adapters", default="identity,wpe1,metricganplus,dfn3")
    args = ap.parse_args(argv)
    if Path(args.out).resolve().as_posix().split(":", 1)[-1].startswith("/datasets"):
        raise SystemExit("refusing to write under /datasets")

    import torch

    from uwse.evaluation.metrics import compute_metrics
    from uwse.uwsim.config import load_config
    from uwse.uwsim.simulate import render_rir, sample_scene, simulate

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clean = speech_like(seed=args.seed)
    sf.write(out / "clean_48k.wav", clean.astype(np.float32), FS)

    cfg = load_config(args.config)
    scene = sample_scene(cfg, args.seed, generic=False)
    t0 = time.perf_counter()
    rir = render_rir(scene)
    t_rir = time.perf_counter() - t0
    sims = {}
    for fs_model in (16000, 48000):
        sims[fs_model] = simulate(clean, scene, None, np.random.default_rng(args.seed + 1),
                                  fs_model=fs_model, rir=rir)
        for k in ("noisy", "target_R1", "target_R0"):
            sf.write(out / f"{k}_{fs_model // 1000}k.wav", sims[fs_model][k], fs_model)
    meta = sims[16000]["meta"]
    print(f"scene: pool L={scene.pool['L']:.1f} W={scene.pool['W']:.1f} "
          f"H0={scene.pool['H0']:.2f} H1={scene.pool['H1']:.2f} m, range {scene.range_m:.2f} m, "
          f"c={scene.c:.1f} m/s, SNR {scene.snr_db:.1f} dB, f_x {scene.f_x:.0f} Hz, "
          f"thruster {'on' if scene.thruster_on else 'off'}; RIR {rir.size / FS:.2f} s "
          f"rendered in {t_rir:.1f} s; noise {[d.get('kind') for d in meta['noise']]}")

    r1 = sims[16000]["target_R1"]
    r0 = sims[16000]["target_R0"]
    rows: list[dict] = []

    def score(label: str, est16: np.ndarray, fs_in: int, secs: float) -> None:
        m = compute_metrics(est16, r1, 16000, ref_r0=r0)
        rows.append({"system": label, "fs": fs_in, "time_s": round(secs, 2), **m})

    score("noisy (16k sim)", sims[16000]["noisy"], 16000, 0.0)
    n48to16 = audio.resample(sims[48000]["noisy"].astype(np.float64), 48000, 16000)[: r1.size]
    score("noisy (48k sim -> 16k)", n48to16, 48000, 0.0)
    score("R1 (oracle)", r1, 16000, 0.0)

    from uwse.adapters import build  # noqa: F401  (registers adapters)
    from uwse.adapters.base import _REGISTRY  # type: ignore[attr-defined]

    for name in [a for a in args.adapters.split(",") if a]:
        try:
            fs_a = _REGISTRY[name].sample_rate
            noisy = sims[fs_a]["noisy"] if fs_a in sims else audio.resample(
                sims[48000]["noisy"].astype(np.float64), 48000, fs_a)
            t0 = time.perf_counter()
            y = run_adapter(name, noisy, device if name not in ("identity", "wpe1") else "cpu")
            dt = time.perf_counter() - t0
            sf.write(out / f"enh_{name}.wav", y.astype(np.float32), fs_a)
            y16 = audio.resample(y.astype(np.float64), fs_a, 16000) if fs_a != 16000 else y
            y16 = np.pad(y16, (0, max(0, r1.size - y16.size)))[: r1.size]
            score(name, y16, fs_a, dt)
        except Exception as e:  # noqa: BLE001 - report and continue (e.g. no network)
            print(f"[skip] {name}: {e!r}")
            traceback.print_exc(limit=2)
            rows.append({"system": name, "error": repr(e)})

    hdr = f"{'system':24s} {'fs':>6s} {'ESTOI':>7s} {'SI-SDR':>8s} {'CI-SDR':>8s} {'PESQ-WB':>8s} {'ESTOI_R0':>9s} {'time s':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['system']:24s} error: {r['error'][:80]}")
            continue
        print(f"{r['system']:24s} {r['fs']:6d} {r['estoi']:7.3f} {r['si_sdr']:8.2f} "
              f"{r['ci_sdr']:8.2f} {r['pesq_wb_r1']:8.2f} {r['estoi_r0']:9.3f} {r['time_s']:7.2f}")
    (out / "results.json").write_text(json.dumps(
        {"scene": scene.to_dict(), "rows": rows, "device": device}, indent=1, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
