import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_nonspeech_set as M  # noqa: E402


def _utts(n=600):
    rng = np.random.default_rng(0)
    return [SimpleNamespace(utt_id=f"u{i:04d}", path=f"/x/{i}.flac", duration_s=float(rng.uniform(2, 15)))
            for i in range(n)]


def _recipe():
    r = M.load_recipe(ROOT / "configs" / "nonspeech_v0.yaml")
    r["n_per_cell"] = 3
    return r


def test_plan_balanced_and_disjoint():
    r, utts = _recipe(), _utts()
    used: set = set()
    dev = M.plan_segments(r, "dev", utts, [5.0, 8.0, 12.0], used)
    test = M.plan_segments(r, "test", utts, [5.0, 8.0, 12.0], used)
    for plan in (dev, test):
        assert len(plan) == 3 * len(r["noise_types"]) * len(r["ranges_m"])
        assert set(Counter((s["noise_type"], s["range_m"]) for s in plan).values()) == {3}
    ids = [s["utt_id"] for s in dev + test]
    assert len(ids) == len(set(ids))
    assert M.plan_segments(r, "dev", utts, [5.0, 8.0, 12.0], set()) == dev  # deterministic


def test_slope_bands_and_near_silent():
    r, utts = _recipe(), _utts()
    plan = M.plan_segments(r, "dev", utts, [6.0], set())
    for s in plan:
        sl = s["over"].get("stationary_slope_db")
        if s["noise_type"] != "stat_held" and sl is not None:
            assert -6 <= sl <= 0 and not (-4 < sl < -2)
        if s["noise_type"] == "stat_held":
            assert -4 <= sl <= -2
        if s["noise_type"] == "near_silent":
            assert 30 <= s["att_db"] <= 40 and s["snr_db"] >= 30
        else:
            assert 0 <= s["snr_db"] <= 15


def test_seeds_distinct_from_dev_wer():
    r = _recipe()
    seeds = {r["splits"]["dev"]["seed"], r["splits"]["test"]["seed"]}
    assert len(seeds) == 2 and not seeds & {20261104, 20261105, 20261106}


def test_group_median_gain():
    rows = [{"group": "a|1", "speech_gain": g, "seg_id": f"s{i}"} for i, g in enumerate([1.0, 3.0, 2.0])]
    rows.append({"group": "b|1", "speech_gain": 5.0, "seg_id": "s9"})
    out = M._assign_group_gains(rows)
    assert [r["gain_median"] for r in out] == [2.0, 2.0, 2.0, 5.0]


def test_render_one_segment_smoke(tmp_path):
    """1-segment end-to-end render (guards the CapChain batched-receive capture)."""
    import pytest

    ls = ROOT / "data" / "librispeech"
    if not ls.exists():
        pytest.skip("no librispeech")
    recipe = M.load_recipe(ROOT / "configs" / "nonspeech_v0.yaml")
    seg0 = M.plan_all(recipe, str(ls))["dev"][0]["seg_id"]
    rows = M.generate(recipe, "dev", tmp_path, str(ls), only=[seg0])
    assert len(rows) == 1
    r = rows[0]
    assert r["n_samples"] > 16000 and not r["clipped"] and np.isfinite(r["nonspeech_rms_dbfs"])
    assert np.isfinite(r["speech_gain"])
