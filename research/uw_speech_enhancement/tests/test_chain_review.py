"""Adversarial-review regression tests for uwse.uwsim chain/noise/simulate/dataset.

Each test pins a defect found in review (RESEARCH_PLAN 4.1, 4.2 rows 14-18, 5.2, 5.4, 8.2).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

from uwse.uwsim import chain as ch
from uwse.uwsim.config import SimConfig
from uwse.uwsim.noise import NoiseBank, build_manifest
from uwse.uwsim.simulate import direct_path_index, sample_scene, simulate

FS = 48000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bl_noise(rng, n):
    return ch.bandlimit(rng.standard_normal(n), FS, 200, 6000)


def _quiet_scene(d0=200):
    sc = sample_scene(SimConfig(), 3)
    sc.range_m = d0 / FS * sc.c
    sc.eps, sc.snr_db, sc.thruster_on, sc.ambient_on, sc.bubble_rate = 0.0, 80.0, False, False, 0.0
    sc.theta_deg, sc.adc_floor_dbfs, sc.level_dbfs = 0.0, -150.0, -30.0
    sc.proj_eq = [[1000.0], [0.0]]
    return sc


def _lag(ref, x):
    cc = np.correlate(ref, x, "full")
    return int(np.argmax(cc) - (x.size - 1))


def test_direct_index_is_geometric_not_strongest_arrival():
    """Lloyd's mirror: direct + surface image nearly cancel, a later floor arrival dominates.

    The targets must still be aligned to the direct arrival r/c (section 8.2), not to
    whichever peak is largest within the search window.
    """
    sc = _quiet_scene(300)
    h = np.zeros(2000)
    h[300], h[301] = 0.06, -0.055   # band-limited direct + surface image (R_s ~ -1) after cancellation
    h[350] = 0.08                   # floor image, now the largest |h| sample
    assert direct_path_index(h, sc) == 300


def test_r0_aligned_with_received_speech(rng):
    """R0 must carry the chain's bulk delay (drive LPF + preamp HPF), like the noisy input."""
    sc = _quiet_scene()
    h = np.zeros(1000)
    h[200] = 0.3
    out = simulate(_bl_noise(rng, FS), sc, None, np.random.default_rng(0), rir=h, fs_model=FS,
                   target_gain="ls")
    nb = ch.bandlimit(out["noisy"].astype(float), FS, 500, 8000)
    assert _lag(nb, out["target_R0"].astype(float)) == 0
    assert _lag(nb, out["target_R1"].astype(float)) == 0


def test_targets_follow_time_variation(rng):
    """Row 14 resampling by (1+eps) must not leave the target drifting (4 s * 1e-3 = 4 ms)."""
    sc = _quiet_scene()
    sc.eps = 1e-3
    h = np.zeros(1000)
    h[200] = 0.3
    out = simulate(_bl_noise(rng, 4 * FS), sc, None, np.random.default_rng(0), rir=h, fs_model=FS,
                   target_gain="ls")
    nb = ch.bandlimit(out["noisy"].astype(float), FS, 500, 8000)
    r1 = out["target_R1"].astype(float)
    tail = slice(3 * FS, 4 * FS - 2000)
    assert abs(_lag(nb[tail], r1[tail])) <= 1
    err = np.sum((nb[FS // 10: -2000] - r1[FS // 10: -2000]) ** 2) / np.sum(nb[FS // 10: -2000] ** 2)
    assert err < 1e-2


def test_ambient_never_drawn_from_test_split(tmp_path, rng):
    """Section 5.4: training noise must not come from test sessions/units."""
    for sub, name in (("thruster", "unitA_t50.wav"), ("ambient/test", "pumps_S2.wav")):
        os.makedirs(tmp_path / sub, exist_ok=True)
        sf.write(tmp_path / sub / name, 0.1 * rng.standard_normal(16000), 16000)
    bank = NoiseBank(str(tmp_path), fs=FS)
    assert bank.select("ambient", "test") and not bank.select("ambient", "train")
    sc = _quiet_scene()
    sc.ambient_on, sc.thruster_on, sc.snr_db = True, True, 10.0
    h = np.zeros(1000)
    h[200] = 0.3
    out = simulate(_bl_noise(rng, FS // 2), sc, bank, np.random.default_rng(0), rir=h,
                   thruster_rir=h[:400])
    splits = [i.get("split") for i in out["meta"]["noise"] if "split" in i]
    assert splits and all(s == "train" for s in splits)


@pytest.mark.parametrize("rel,split,flagged", [
    ("thruster/unit-B_t50.wav", "test", False),
    ("thruster/unit B t50.wav", "test", False),
    ("thruster/T200_unitB/t20.wav", "test", False),
    ("thruster/unitA_t50.wav", "train", False),
    ("self/sub_pattern1.wav", "test", False),      # vehicle self-noise: test only (5.2)
    ("emi/S1_dummy_load.wav", "diag", False),      # EMI check: diagnostic (5.2)
    ("flow/jet_on_array.wav", "train", True),      # flow noise: flagged (row 15)
])
def test_manifest_inference_tags(tmp_path, rel, split, flagged):
    p = tmp_path / rel
    os.makedirs(p.parent, exist_ok=True)
    sf.write(p, np.zeros(160), 16000)
    (it,) = build_manifest(str(tmp_path), out=None)
    assert (it.split, it.flagged) == (split, flagged)


def test_noise_bank_rate_mismatch_rejected(tmp_path, rng):
    os.makedirs(tmp_path / "thruster")
    sf.write(tmp_path / "thruster" / "unitA.wav", 0.1 * rng.standard_normal(16000), 16000)
    bank = NoiseBank(str(tmp_path), fs=16000)
    sc = _quiet_scene()
    sc.thruster_on = True
    h = np.zeros(1000)
    h[200] = 0.3
    with pytest.raises(ValueError):
        simulate(_bl_noise(rng, FS // 4), sc, bank, np.random.default_rng(0), rir=h, thruster_rir=h)


def _load_script():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import make_rir_bank
    finally:
        sys.path.pop(0)
    return make_rir_bank


def test_datasets_guard_works_on_windows_paths():
    """os.path.abspath('/datasets/x') is 'C:\\datasets\\x' on Windows; the guard must still fire."""
    mrb = _load_script()
    assert mrb.is_under_datasets("/datasets/rir")
    assert mrb.is_under_datasets("C:\\datasets\\rir")
    assert mrb.is_under_datasets("/datasets")
    assert not mrb.is_under_datasets("/data/datasets_copy")
    assert not mrb.is_under_datasets(os.path.join(ROOT, "data", "rir_bank"))


def test_rir_bank_resume_rewrites_bank_json(tmp_path):
    """Resuming with a missing early shard must leave bank.json complete (n = N)."""
    mrb = _load_script()
    out = str(tmp_path / "bank")
    args = ["--n", "2", "--out", out, "--shard-size", "1", "--rir-length", "0.1",
            "--config", os.path.join(ROOT, "configs", "uwsim_v0.yaml")]
    mrb.main(args)
    os.remove(os.path.join(out, "shard_00000.npz"))
    mrb.main(args + ["--resume"])
    with open(os.path.join(out, "bank.json"), encoding="utf-8") as f:
        info = json.load(f)
    assert info["n"] == 2 and info["complete"] and info["shards"] == ["shard_00000.npz", "shard_00001.npz"]


def test_placement_fallback_is_flagged_and_inside():
    """An unsatisfiable range constraint must be logged in the scene, never silently violated."""
    cfg = SimConfig()
    cfg.pool.calibrated_dims = {"L": 3.0, "W": 2.0, "H0": 2.0, "H1": 2.0}
    cfg.pool.generic_fraction = 0.0
    cfg.geometry.range_m = (20.0, 25.0)
    sc = sample_scene(cfg, 0)
    assert sc.extra.get("placement_fallback") is True
    assert sc.src[2] >= 1.0 - 1e-9
    ok = sample_scene(SimConfig(), 0)
    assert "placement_fallback" not in ok.extra
