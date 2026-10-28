"""Tests for uwsim chain, noise, config, simulate and dataset (RESEARCH_PLAN 4.1, 4.2, 5.2, 5.3, 8.2)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf

from uwse.uwsim import chain as ch
from uwse.uwsim.config import SimConfig, from_dict, load_config, save_config, to_dict
from uwse.uwsim.noise import NoiseBank, band_snr_db, bubble_noise, energy_vad, mix_at_snr
from uwse.uwsim.simulate import Scene, sample_scene, simulate

FS = 48000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tone(f, dur=0.5, fs=FS):
    return np.sin(2 * np.pi * f * np.arange(int(dur * fs)) / fs)


def speechlike(rng, dur=1.0, fs=FS):
    """Band-limited noise gated on/off like speech (for VAD/SNR tests)."""
    x = ch.bandlimit(rng.standard_normal(int(dur * fs)), fs, 200, 6000)
    env = np.repeat(rng.random(int(np.ceil(dur * 10))) > 0.3, fs // 10)[: x.size]
    return x * env


def synth_rir(n=4800, d0=200, rng=None):
    rng = rng or np.random.default_rng(0)
    h = np.zeros(n)
    h[d0] = 1.0
    h[d0 + 30] = -0.8
    h[d0 + 60:] += 0.05 * rng.standard_normal(n - d0 - 60) * np.exp(-np.arange(n - d0 - 60) / 800)
    return h


# --------------------------------------------------------------------------- chain


def test_drive_lpf_attenuation():
    lvl = lambda y: 20 * np.log10(np.std(y[FS // 10:]))  # skip transient
    assert abs(lvl(ch.drive_lpf(tone(1000), FS)) - lvl(tone(1000))) < 0.5
    assert lvl(ch.drive_lpf(tone(16000), FS)) - lvl(tone(16000)) < -40  # 1 octave, 8th order


def test_hammerstein_identity_default(rng):
    x = rng.standard_normal(2000)
    assert np.allclose(ch.Projector().apply(x), x)
    assert np.allclose(ch.hammerstein(x, [np.array([1.0])]), x)
    y = ch.hammerstein(x, [np.array([1.0]), np.array([0.1]), None])
    assert np.allclose(y, x + 0.1 * x ** 2)


def test_warp_fir_preserves_dc():
    h = np.hanning(64)
    for f in (0.8, 1.2):
        w = ch.warp_fir(h, f)
        assert abs(w.sum() - h.sum()) / h.sum() < 0.02


def test_adc_clip_and_quantization(rng):
    y, clipped = ch.adc(np.array([0.1, -0.2, 0.3]), noise_floor_dbfs=None)
    assert not clipped
    q = 2.0 ** 23
    assert np.allclose(y * q, np.round(y * q))
    y, clipped = ch.adc(np.array([0.5, 1.5, -2.0]), rng, noise_floor_dbfs=None)
    assert clipped and y.max() <= 1.0 and y.min() >= -1.0
    y, _ = ch.adc(np.zeros(48000), rng, noise_floor_dbfs=-100.0)
    assert abs(20 * np.log10(np.std(y)) + 100) < 0.5


def test_piston_directivity():
    d = ch.Directivity.piston()
    assert np.allclose(d.gain_db(0.0), 0.0)
    g90 = d.gain_db(90.0)
    assert g90[-1] < -10 and g90[0] > -1  # 8 kHz beamed, 125 Hz omni (ka << 1)
    assert np.allclose(d.gain_db(30.0), d.gain_db(330.0))


def test_h_chain_identity_is_lpf_hpf(rng):
    c = ch.TransducerChain()
    s = rng.standard_normal(4800)
    ref = ch.preamp_hpf(ch.drive_lpf(s, FS), FS)
    assert np.allclose(c.h_chain_apply(s), ref)


# --------------------------------------------------------------------------- noise


@pytest.mark.parametrize("snr", [-10.0, 0.0, 12.5, 25.0])
def test_mix_at_snr_hits_target(rng, snr):
    s = speechlike(rng)
    n = rng.standard_normal(30000)  # shorter than speech -> tiled
    mix, ns = mix_at_snr(s, n, snr, FS, band=(500, 8000))
    assert mix.shape == s.shape
    assert abs(band_snr_db(s, ns, FS, (500, 8000)) - snr) < 0.2
    mask = energy_vad(s, FS)
    _, ns2 = mix_at_snr(s, n, snr, FS, band=(500, 8000), active_mask=mask)
    assert abs(band_snr_db(s, ns2, FS, (500, 8000), mask) - snr) < 0.2


def test_bubble_noise(rng):
    assert not np.any(bubble_noise(FS, 1.0, 0.0, rng=rng))
    b = bubble_noise(FS, 2.0, 20.0, (0.5e-3, 3e-3), rng)
    assert np.isfinite(b).all() and abs(np.sqrt(np.mean(b ** 2)) - 1) < 1e-6
    f = np.fft.rfftfreq(b.size, 1 / FS)
    pk = f[np.argmax(np.abs(np.fft.rfft(b)))]
    assert 900 < pk < 7500  # Minnaert 1.1-6.6 kHz


@pytest.fixture
def noise_dir(tmp_path, rng):
    for sub, name in (("thruster", "unitA_t50.wav"), ("thruster", "unitB_t50.wav"), ("ambient", "pumps_on.wav")):
        os.makedirs(tmp_path / sub, exist_ok=True)
        sf.write(tmp_path / sub / name, 0.1 * rng.standard_normal(16000), 16000)
    return str(tmp_path)


def test_noise_bank_splits_and_determinism(noise_dir):
    bank = NoiseBank(noise_dir, fs=FS)
    assert len(bank) == 3
    assert [it.path for it in bank.select("thruster", "test")] == ["thruster/unitB_t50.wav"]
    assert [it.path for it in bank.select("thruster", "train")] == ["thruster/unitA_t50.wav"]
    a, ia = bank.draw(30000, np.random.default_rng(5), "thruster", "train")
    b, ib = bank.draw(30000, np.random.default_rng(5), "thruster", "train")
    assert a.shape == (30000,) and np.array_equal(a, b) and ia == ib
    big, _ = bank.draw(200000, np.random.default_rng(1), "ambient", None)  # longer than file: looped
    assert big.shape == (200000,)


# --------------------------------------------------------------------------- config / scene


def test_config_yaml_roundtrip(tmp_path):
    cfg = SimConfig()
    p = str(tmp_path / "c.yaml")
    save_config(cfg, p)
    assert load_config(p) == cfg
    assert from_dict(to_dict(cfg)) == cfg
    assert load_config(os.path.join(ROOT, "configs", "uwsim_v0.yaml")) == cfg
    assert cfg.calibrated is False


def test_sample_scene_deterministic_and_in_range():
    cfg = SimConfig()
    assert sample_scene(cfg, 7).to_dict() == sample_scene(cfg, 7).to_dict()
    assert sample_scene(cfg, 7).to_dict() != sample_scene(cfg, 8).to_dict()
    n_gen = 0
    for seed in range(60):
        sc = sample_scene(cfg, seed)
        d = sc.to_dict()
        json.dumps(d)
        assert Scene.from_dict(d).to_dict() == d
        n_gen += sc.is_generic
        assert -1.0 <= sc.surface_R <= -0.9
        assert 0.030 <= sc.t_e <= 0.050
        assert 0.3 <= sc.range_m <= 25.0
        assert -10 <= sc.drive_db <= 0
        assert -10 <= sc.snr_db <= 25
        assert 0 <= sc.bubble_rate <= 20
        assert abs(sc.eps) <= 1e-3
        assert 0.8 <= sc.resonance_factor <= 1.2
        assert abs(sc.adc_floor_dbfs + 110) <= 3
        assert sc.src[2] >= 1.0 - 1e-9  # LL916 depth >= 1 m
        p = sc.pool
        for q in (sc.src, sc.rx):
            assert 0 <= q[0] <= p["L"] and 0 <= q[1] <= p["W"]
            assert q[2] <= p["H0"] + (p["H1"] - p["H0"]) * q[0] / p["L"]
        if sc.is_generic:
            assert 15 <= p["L"] <= 50 and 8 <= p["W"] <= 25
            assert 1.2 <= p["H0"] <= 5 and 1.2 <= p["H1"] <= 5
        if sc.thruster_on:
            assert 0.3 - 1e-6 <= np.linalg.norm(np.subtract(sc.thruster_pos, sc.rx)) <= 1.5 + 1e-6
        assert sc.f_x >= 500.0
    assert 3 <= n_gen <= 25  # ~20 % generic


# --------------------------------------------------------------------------- simulate


def _small_scene(seed=3):
    cfg = SimConfig()
    sc = sample_scene(cfg, seed)
    sc.range_m = 200 / FS * sc.c  # matches synth_rir's direct arrival at sample 200
    return sc


def test_simulate_with_given_rir(rng, noise_dir):
    sc = _small_scene()
    sc.thruster_on = True
    clean = speechlike(rng, 0.75)
    bank = NoiseBank(noise_dir, fs=FS)
    for fs_model in (16000, 48000):
        out = simulate(clean, sc, bank, np.random.default_rng(1), fs_model=fs_model,
                       rir=synth_rir(), thruster_rir=synth_rir(2400, 10))
        n = int(round(clean.size * fs_model / FS))
        for k in ("noisy", "target_R1", "target_R0"):
            assert out[k].shape == (n,) and out[k].dtype == np.float32 and np.isfinite(out[k]).all()
        assert out["fs"] == fs_model
        m = out["meta"]
        assert m["direct_index_48k"] == 200
        assert any(i.get("kind") == "thruster" and i["split"] == "train" for i in m["noise"])
        json.dumps(m)
    # deterministic given the same rng seed
    a = simulate(clean, sc, bank, np.random.default_rng(1), rir=synth_rir(), thruster_rir=synth_rir(2400, 10))
    b = simulate(clean, sc, bank, np.random.default_rng(1), rir=synth_rir(), thruster_rir=synth_rir(2400, 10))
    assert np.array_equal(a["noisy"], b["noisy"])


def test_target_r1_is_chain_only(rng):
    """With a pure-delay channel and no noise, R1 (ls gain) reproduces the received speech."""
    sc = _small_scene()
    sc.eps, sc.snr_db, sc.thruster_on, sc.ambient_on, sc.bubble_rate = 0.0, 80.0, False, False, 0.0
    sc.theta_deg, sc.adc_floor_dbfs = 0.0, -150.0
    sc.proj_eq = [[1000.0], [0.0]]
    h = np.zeros(1000)
    h[200] = 0.3
    clean = speechlike(rng, 0.75)
    out = simulate(clean, sc, None, np.random.default_rng(0), rir=h, fs_model=FS, target_gain="ls")
    noisy = ch.bandlimit(out["noisy"].astype(float), FS, 500, 8000)
    r1 = out["target_R1"].astype(float)
    err = np.sum((noisy - r1) ** 2) / np.sum(noisy ** 2)
    assert err < 1e-3


def test_simulate_full_render(rng):
    pytest.importorskip("uwse.uwsim.ism")
    pytest.importorskip("uwse.uwsim.tail")
    pytest.importorskip("uwse.uwsim.lowband")
    cfg = SimConfig()
    cfg.pool.calibrated_dims = {"L": 6.0, "W": 4.0, "H0": 2.0, "H1": 2.2}
    cfg.pool.generic_fraction = 0.0
    cfg.geometry.range_m = (0.5, 3.0)
    cfg.acoustics.t_e = (0.03, 0.03)
    cfg.acoustics.rir_length_s = 0.3
    sc = sample_scene(cfg, 1)
    out = simulate(speechlike(rng, 0.5), sc, None, np.random.default_rng(0))
    assert out["noisy"].shape == (8000,) and np.isfinite(out["noisy"]).all()
    assert np.isfinite(out["target_R1"]).all() and np.any(out["target_R1"])


# --------------------------------------------------------------------------- dataset


def test_dataset_determinism(tmp_path, rng, noise_dir):
    torch = pytest.importorskip("torch")
    from uwse.uwsim.dataset import RIRBank, UWSimPairs
    files = []
    for i in range(2):
        p = tmp_path / f"c{i}.wav"
        sf.write(p, 0.3 * speechlike(rng, 1.2, 16000), 16000)
        files.append(str(p))
    cfg = SimConfig()
    items = []
    for i in range(3):
        sc = sample_scene(cfg, i)
        sc.range_m = 200 / FS * sc.c
        items.append((sc, synth_rir(rng=np.random.default_rng(i)), synth_rir(2400, 10) if sc.thruster_on else None))
    bank = RIRBank.write(str(tmp_path / "bank"), items, shard_size=2)
    bank2 = RIRBank(str(tmp_path / "bank"))
    assert len(bank2) == 3
    s1, r1, _ = bank2.get(2)
    assert np.allclose(r1, items[2][1], atol=1e-6) and s1.to_dict() == items[2][0].to_dict()

    nb = NoiseBank(noise_dir, fs=FS)
    ds = UWSimPairs(files, nb, cfg, fs_model=16000, segment_s=0.5, seed=11, rir_bank=bank)
    a, ta, ma = ds[1]
    b, tb, mb = ds[1]
    assert a.shape == (8000,) and ta.shape == (8000,) and a.dtype == torch.float32
    assert torch.equal(a, b) and torch.equal(ta, tb) and ma == mb
    c, _, _ = ds[0]
    assert not torch.equal(a, c)
    ds.set_epoch(1)
    d, _, _ = ds[1]
    assert not torch.equal(a, d)
    ds0 = UWSimPairs(files, nb, cfg, fs_model=16000, segment_s=0.5, seed=11, rir_bank=bank, target="R0")
    _, t0, _ = ds0[1]
    assert not torch.equal(t0, ta)
    loader = torch.utils.data.DataLoader(ds0, batch_size=2, num_workers=0)
    xb, yb, mb = next(iter(loader))
    assert xb.shape == (2, 8000) and len(mb) == 2
