"""Post-session analysis: sweep IRs and metrics, noise spectra, the analyze_session CLI on synthetic takes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

from test_session_run import CMAP, _script, _thruster_bench
from uwse import audio
from uwse.bench import playlist as pl
from uwse.bench import postsession as PS
from uwse.irtools import analysis as an
from uwse.irtools import ess as es

ROOT = Path(__file__).resolve().parents[1]


def _sweep_file(fs, n_sweeps=3, T=1.0, silence_s=1.0, f1=150.0, f2=None):
    """sweeps.wav + sweeps.json as make_session_audio.py writes them (short sweeps)."""
    f2 = f2 or 0.45 * fs
    e = es.generate_ess(f1, f2, T, fs, amplitude=0.1, silence_s=silence_s, lf_shelf_hz=300.0, lf_shelf_db=12.0)
    lay = pl.BlockLayout()
    lead, ch, post = np.zeros(int(round(lay.lead_silence_s * fs))), pl.lfm_chirp(fs, lay.chirp_s, f1=min(8000.0, 0.45 * fs)), np.zeros(fs)
    x = np.concatenate([lead, ch, post] + [e.signal] * n_sweeps)
    s0 = len(lead) + len(ch) + len(post)
    sw = {"fs": fs, "f1": e.f1, "f2": e.f2, "T": e.T, "L": e.L, "peak_dBFS": -20.0, "lf_shelf_hz": 300.0,
          "lf_shelf_db": 12.0, "synchronized": True, "chirp_start_sample": len(lead),
          "sweep_start_samples": [s0 + k * len(e.signal) for k in range(n_sweeps)],
          "sweep_n_samples": len(e.excitation), "silence_s": silence_s}
    return x, sw


def _room(fs, t60=0.25, d_s=0.010, seed=0):
    """Direct path at ``d_s``, one reflection 3 ms later, exponential diffuse tail."""
    rng = np.random.default_rng(seed)
    h = np.zeros(int(0.8 * fs))
    d = int(round(d_s * fs))
    h[d], h[d + int(0.003 * fs)] = 1.0, -0.5
    n0 = d + int(0.005 * fs)
    t = np.arange(len(h) - n0) / fs
    h[n0:] += 0.08 * rng.standard_normal(t.size) * np.exp(-3 * np.log(10) * t / t60)
    return h


def _record(played, h, fs, ppm, lead_s, noise=1e-5, seed=1):
    """(loopback, hydrophone) on a recorder whose clock runs ``ppm`` fast, started ``lead_s`` early."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([fftconvolve(played, h), [0.0]])
    lb = np.concatenate([played, np.zeros(len(h))])
    r = 1.0 + ppm * 1e-6
    out = []
    for z in (lb, y):
        z = es.correct_drift(z, (1.0 / r - 1.0) * 1e6)  # event at s -> s (1 + ppm 1e-6)
        out.append(np.concatenate([np.zeros(int(lead_s * fs)), z, np.zeros(fs // 2)]))
    x = np.stack(out)
    return x + noise * rng.standard_normal(x.shape)


def test_excitation_matches_bench_sweeps():
    """The regenerated excitation is the one in the frozen S1 bench (skipped without the bench)."""
    b = ROOT / "data" / "bench" / "v0_s1" / "session"
    if not (b / "sweeps.json").exists():
        pytest.skip("no data/bench/v0_s1")
    sw = json.loads((b / "sweeps.json").read_text())
    e = PS.excitation_from_meta(sw)
    x, fs = audio.load(str(b / "sweeps.wav"))
    s = sw["sweep_start_samples"][1]
    assert fs == sw["fs"] and np.max(np.abs(x[s:s + len(e.excitation)] - e.excitation)) < 1e-6
    with pytest.raises(ValueError, match="generator or metadata"):
        PS.excitation_from_meta({**sw, "sweep_n_samples": sw["sweep_n_samples"] + 1})


@pytest.mark.parametrize("sync", [0, 1])  # loopback, or the hydrophone itself (Daisy recorders)
def test_sweep_irs_recover_room(sync):
    fs = 16000
    played, sw = _sweep_file(fs)
    h = _room(fs)
    x = _record(played, h, fs, ppm=40.0, lead_s=0.37)
    res = PS.sweep_irs(x, fs, sw, {1: 5}, sync=sync)
    assert res.irs.shape == (1, 3, res.n_pre + int(sw["silence_s"] * fs) + 1)
    # ~0.06-sample sub-sample bias over this 4 s span; the S1 take spans 45 s
    assert res.ppm == pytest.approx(40.0, abs=1.5) and res.resid_samples < 0.3 and res.channel_ids == [5]
    m, arr = PS.ir_metrics(res.irs[0], fs, res.n_pre, (sw["f1"], sw["f2"]))
    hm = arr["ir"]
    pk = int(np.argmax(np.abs(hm)))
    if sync == 0:  # loopback: t = 0 is the played sweep start, the direct path sits at its delay
        assert abs(pk - (res.n_pre + int(0.010 * fs))) <= 1
        assert m["t_direct_s"] == pytest.approx(0.010, abs=2 / fs)
    else:  # acoustic sync: the direct path is (close to) zero lag
        assert abs(pk - res.n_pre) <= 3
    assert m["first_reflection_delay_s"] == pytest.approx(0.003, abs=2 / fs)
    bb = m["broadband"]
    assert bb["T20_s"] == pytest.approx(0.25, rel=0.15) and bb["T30_s"] == pytest.approx(0.25, rel=0.15)
    hb = an.filterbank(h, fs, [1000.0])[0]
    assert bb["C50_dB"] == pytest.approx(an.c50_db(h, fs), abs=1.0)
    assert bb["ir_snr_dB"] > 40 and bb["n_slopes"] == 1
    # DRR gate rule: 0.7 ms holds < 2 periods at 150 Hz (no broadband DRR) and at the 4 kHz octave's
    # 2.83 kHz lower edge; a 1 ms gate (still before the 3 ms reflection) opens the 4 kHz octave
    assert bb["DRR_dB"] is None
    assert all(o["DRR_dB"] is None for o in m["octaves"])
    m1, _ = PS.ir_metrics(res.irs[0], fs, res.n_pre, (sw["f1"], sw["f2"]), drr_gate_s=1e-3)
    o4 = next(o for o in m1["octaves"] if o["fc"] == 4000.0)
    assert o4["DRR_dB"] is not None and np.isfinite(o4["DRR_dB"])
    assert next(o for o in m1["octaves"] if o["fc"] == 1000.0)["DRR_dB"] is None
    o1 = next(o for o in m["octaves"] if o["fc"] == 1000.0)
    assert o1["DRR_dB"] is None and o1["T20_s"] == pytest.approx(0.25, rel=0.2)
    assert o1["C50_dB"] == pytest.approx(an.c50_db(hb, fs), abs=1.5)
    # identical sweeps: coherence ~ 1 where the sweep has energy
    assert min(c for c, fc in zip(m["band_coherence"], m["band_centers"]) if fc >= 500) > 0.99
    assert json.dumps(m)  # JSON-ready


def test_sweep_outside_recording_raises():
    fs = 16000
    played, sw = _sweep_file(fs, n_sweeps=2)
    x = _record(played, _room(fs), fs, ppm=0.0, lead_s=0.2)[:, : -int(1.5 * fs) - int(0.8 * fs)]
    with pytest.raises(ValueError, match="outside the recording"):
        PS.sweep_irs(x, fs, sw, {1: 0}, sync=0)


@pytest.mark.parametrize("ppm", [-30.0, 25.0])
def test_locate_s1_spacing_with_drift(ppm):
    """S1 layout (4 x 10 s sweeps, 5 s tails) with tens of ppm drift on acoustic sync: the coarse lag must
    not jump a whole sweep (whole-file correlation did, review 2026-11-09)."""
    fs = 8000
    played, sw = _sweep_file(fs, n_sweeps=4, T=10.0, silence_s=5.0)
    x = _record(played, _room(fs), fs, ppm=ppm, lead_s=2.0)
    res = PS.sweep_irs(x, fs, sw, {1: 0}, sync=1)
    assert res.ppm == pytest.approx(ppm, abs=0.3) and res.resid_samples < 0.3
    assert res.offset_s == pytest.approx(2.0 + 0.010, abs=2e-3)  # lead + direct delay
    m, _ = PS.ir_metrics(res.irs[0], fs, res.n_pre, (sw["f1"], sw["f2"]))
    assert m["broadband"]["T20_s"] == pytest.approx(0.25, rel=0.15)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # log10(0) on the silent channel, reported as None
def test_silent_channel_does_not_crash():
    """A hydrophone input that records digital silence: metrics come back without slope fits."""
    fs = 16000
    played, sw = _sweep_file(fs)
    x = _record(played, _room(fs), fs, ppm=0.0, lead_s=0.3)
    x = np.vstack([x, np.zeros((1, x.shape[1]))])
    res = PS.sweep_irs(x, fs, sw, {1: 0, 2: 1}, sync=1)
    m, _ = PS.ir_metrics(res.irs[1], fs, res.n_pre, (sw["f1"], sw["f2"]))
    assert m["broadband"]["n_slopes"] is None and json.dumps(m)


def test_noise_spectrum_levels_and_gate():
    fs, sd = 48000, 0.01
    rng = np.random.default_rng(0)
    x = sd * rng.standard_normal(20 * fs)
    s = PS.noise_spectrum(x, fs)
    df = s["freqs"][1] - s["freqs"][0]
    assert np.sum(10 ** (s["psd_db"] / 10)) * df == pytest.approx(sd ** 2, rel=0.03)  # Parseval
    # the SNR method's Butterworth band (uwse.bench.snr) passes ~0.4 dB less than an ideal 500-8000 Hz band
    assert s["band_db"]["500-8000"] == pytest.approx(10 * np.log10(sd ** 2 * 7500 / 24000), abs=0.6)
    i = int(np.argmin(np.abs(s["third_octave_fc"] - 1000)))
    bw = 1000 * (10 ** 0.05 - 10 ** -0.05)
    assert s["third_octave_db"][i] == pytest.approx(10 * np.log10(sd ** 2 / 24000 * bw), abs=0.5)
    # stationary: the spread left is the estimator's own (few bins per band at low f)
    assert np.all(s["third_octave_std_db"][s["third_octave_fc"] >= 1250] < 1.5)
    assert s["third_octave_fc"][-1] * 10 ** 0.05 < fs / 2 and s["third_octave_fc"][0] > 25  # 25 Hz: < 2 bins
    # a quiet 5 s span (thruster not yet up) is gated out; without the gate it pulls the level down
    y = x.copy()
    y[5 * fs:10 * fs] *= 0.01
    g = PS.noise_spectrum(y, fs)
    assert g["frames_kept"] < 0.8 * g["frames"]
    assert g["band_db"]["500-8000"] == pytest.approx(s["band_db"]["500-8000"], abs=0.3)
    assert g["third_octave_db"][i] == pytest.approx(s["third_octave_db"][i], abs=0.5)
    n = PS.noise_spectrum(y, fs, gate_db=None)
    assert n["frames_kept"] == n["frames"] and n["third_octave_db"][i] < s["third_octave_db"][i] - 1.0
    c = PS.compare_spectra({"quiet": s, "loud": PS.noise_spectrum(2 * x, fs)}, "quiet")
    assert list(c) == ["loud"] and np.allclose(c["loud"]["third_octave_re_ref_db"], 20 * np.log10(2), atol=1e-6)
    with pytest.raises(ValueError, match="band grids"):
        PS.compare_spectra({"a": s, "b": PS.noise_spectrum(x[: 10 * 16000], 16000)}, "a")


def test_analyze_session_cli(tmp_path):
    """plan -> recorder files at the standard paths -> analyze_session all."""
    fs = 48000
    bench, sdir = tmp_path / "bench", tmp_path / "sess"
    bench.mkdir()
    _thruster_bench(bench)
    played, sw = _sweep_file(fs, n_sweeps=2)
    audio.save(str(bench / "session" / "sweeps.wav"), played, fs, subtype="FLOAT")
    (bench / "session" / "sweeps.json").write_text(json.dumps(sw))
    rs = _script("run_session")
    assert rs.main(["plan", "--bench", str(bench), "--session", "1", "--channel-map", str(CMAP),
                    "--session-dir", str(sdir), "--out", str(tmp_path / "real"),
                    "--extra-noise", "quiet_pool", "thruster", "--extra-noise-s", "6", "--noise-s", "6"]) == 0
    rng = np.random.default_rng(2)
    # daisyA (no loopback, 96 kHz): both hydrophones hear the room; daisyB recorded nothing (missing file)
    hyd = _record(played, _room(fs), fs, ppm=-25.0, lead_s=0.6)[1]
    two = np.stack([hyd, 0.5 * hyd + 1e-5 * rng.standard_normal(hyd.size)])
    raw = sdir / "raw" / "daisyA"
    audio.save(str(raw / "s1_p1_sweeps_daisyA.wav"), audio.resample(two, fs, 96000), 96000, subtype="FLOAT")
    for tk, sd in (("p1_noise", 1e-3), ("noise_quiet_pool", 1e-3), ("noise_thruster", 4e-3)):
        audio.save(str(raw / f"s1_{tk}_daisyA.wav"), sd * rng.standard_normal((2, 6 * 96000)), 96000,
                   subtype="FLOAT")
    an_ = _script("analyze_session")
    assert an_.main(["all", "--session-dir", str(sdir)]) == 0
    out = sdir / "analysis"
    allrows = [json.loads(line) for line in (out / "sweeps.jsonl").read_text().splitlines()]
    rows = [r for r in allrows if "error" not in r]
    assert [(r["take_id"], r["channel_id"]) for r in rows] == [("p1_sweeps", 0), ("p1_sweeps", 1)]
    # the missing daisyB file is reported, not silently dropped (review 2026-11-09)
    assert [(r["recorder"], r["error"][:18]) for r in allrows if "error" in r] == [("daisyB", "no recording found")]
    for r in rows:
        assert r["fs_raw"] == 96000 and not r["sync_is_loopback"] and r["drift_ppm"] == pytest.approx(-25, abs=2)
        assert r["broadband"]["T20_s"] == pytest.approx(0.25, rel=0.2) and r["n_sweeps"] == 2
    z = np.load(out / "ir" / "p1_sweeps_daisyA_ch1.npz")
    assert z["irs"].shape[0] == 2 and "edc_db_1000" in z and int(z["fs"]) == fs
    nall = [json.loads(line) for line in (out / "noise.jsonl").read_text().splitlines()]
    nrows = [r for r in nall if "error" not in r]
    # p1_noise_on was not recorded on daisyA in this test; daisyB recorded nothing
    assert {(r["take_id"], r["recorder"]) for r in nall if "error" in r} == {
        ("p1_noise_on", "daisyA"), *((t, "daisyB") for t in ("p1_noise", "p1_noise_on", "noise_quiet_pool",
                                                              "noise_thruster"))}
    assert {(r["take_id"], r["channel_id"]) for r in nrows} == {
        (t, c) for t in ("p1_noise", "noise_quiet_pool", "noise_thruster") for c in (0, 1)}
    assert all(r["fs"] == 96000 for r in nrows)
    cmp_ = json.loads((out / "noise_compare.json").read_text())
    th = cmp_["daisyA_ch0"]["noise_thruster"]
    assert th["ref"] == "noise_quiet_pool" and th["band_re_ref_db"]["500-8000"] == pytest.approx(12.0, abs=0.5)
    assert (out / "noise" / "noise_thruster_daisyA_ch1.npz").exists()
    # --take restricts the work; unknown takes are refused
    assert an_.main(["noise", "--session-dir", str(sdir), "--take", "p1_noise", "--out", str(tmp_path / "o2")]) == 0
    assert len((tmp_path / "o2" / "noise.jsonl").read_text().splitlines()) == 3  # 2 channels + daisyB missing
    with pytest.raises(SystemExit, match="unknown takes"):
        an_.main(["noise", "--session-dir", str(sdir), "--take", "nope"])
