"""Stress-suite pieces: tonal noise, opt-in scene fields, per-condition scenes."""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np

from uwse.uwsim.noise import tonal_noise
from uwse.uwsim.simulate import Scene

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tonal_noise_harmonics():
    fs, n = 48000, 48000 * 2
    y = tonal_noise(n, fs, np.random.default_rng(0), f0=120.0, f_max=4000.0, wobble=0.0, floor_rel_db=None)
    assert abs(np.std(y) - 1.0) < 1e-6
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    on = sum(P[np.argmin(np.abs(f - k * 120.0))] for k in range(1, 34))
    assert on / P.sum() > 0.99  # all power on the harmonics
    assert P[f > 4100].sum() / P.sum() < 1e-6
    yf = tonal_noise(n, fs, np.random.default_rng(0))
    assert abs(np.std(yf) - 1.0) < 1e-6


def test_scene_tonal_fields_default_off():
    names = {f.name: f for f in dataclasses.fields(Scene)}
    assert names["tonal_on"].default is False
    # scenes serialized before the field existed still load
    d = {k: (f.default if f.default is not dataclasses.MISSING else None) for k, f in names.items()}
    d.pop("tonal_on")
    assert Scene.from_dict(d).tonal_on is False


def test_condition_scene_and_conditions():
    sb = _script("sim_baseline")
    st = _script("stress_eval")
    base = Scene.from_dict({f.name: None for f in dataclasses.fields(Scene)})
    base = dataclasses.replace(base, range_m=3.0, snr_db=30.0, t60_bands={"125.0": 1.0, "1000.0": 0.7},
                               rir_length_s=1.0, level_dbfs=-26.0)
    s, own = sb.condition_scene(base, 0.0, st.CONDITIONS["n0_t60x2"][1])
    assert own and s.snr_db == 0.0 and s.t60_bands["1000.0"] == 1.4 and s.rir_length_s == 2.0
    s, own = sb.condition_scene(base, 30.0, st.CONDITIONS["q_clip"][1])
    assert not own and s.level_dbfs == -6.0
    s1, _ = sb.condition_scene(base, 0.0, st.CONDITIONS["n0_ripple9"][1])
    s2, _ = sb.condition_scene(base, 30.0, st.CONDITIONS["q_ripple9"][1])
    g = np.asarray(s1.proj_eq[1])
    assert s1.proj_eq == s2.proj_eq and np.abs(g).max() <= 9.0 and np.abs(g).max() > 3.0
    g = np.asarray(st.CONDITIONS["q_tiltm6"][1]["hyd_eq"][1])
    assert abs(g[0] - 6.0) < 1e-9 and abs(g[-1] + 6.0) < 1e-9
    assert len(st.CONDITIONS) == 14 and all(c.split("_")[0] in st.BASE_OF for c in st.CONDITIONS)


def test_speech_shaped_noise_flat_band_snr():
    from uwse.uwsim.noise import speech_shaped_noise

    fs, rng = 48000, np.random.default_rng(1)
    t = np.arange(fs * 2) / fs
    # pink-ish "speech": a 150 Hz harmonic series falling 6 dB/oct, so its spectrum is far from white
    s = sum(np.sin(2 * np.pi * 150 * k * t + k) / k for k in range(1, 40))
    y = speech_shaped_noise(s, rng)
    assert abs(np.std(y) - 1.0) < 1e-6
    f = np.fft.rfftfreq(s.size, 1 / fs)
    S, Y = np.abs(np.fft.rfft(s)) ** 2, np.abs(np.fft.rfft(y)) ** 2
    snr = [10 * np.log10(S[(f >= lo) & (f < 2 * lo)].sum() / Y[(f >= lo) & (f < 2 * lo)].sum())
           for lo in (250, 500, 1000, 2000)]
    assert max(snr) - min(snr) < 3.0  # per-octave SNR flat within 3 dB
    assert dataclasses.fields(Scene)[[f.name for f in dataclasses.fields(Scene)].index("ssn_on")].default is False
