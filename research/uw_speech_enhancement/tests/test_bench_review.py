"""Adversarial review tests for uwse.bench (RESEARCH_PLAN.md 5.1, 5.4, 6.5, 6.6, 6.8)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from uwse.bench import librispeech as ls
from uwse.bench import playlist as pl
from uwse.bench import schema

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ schema
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_schema_rejects_non_finite(bad):
    """NaN/inf are not JSON; metadata.jsonl must stay strict JSON (Croissant export, 6.8)."""
    row = schema.example_row()
    row["acoustic"]["snr_speechband_dB"] = bad
    with pytest.raises(schema.SchemaError, match="snr_speechband_dB"):
        schema.validate(row)


def test_write_jsonl_never_emits_nan(tmp_path):
    row = schema.example_row()
    row["drift_ppm"] = float("nan")
    with pytest.raises((schema.SchemaError, ValueError)):
        schema.write_jsonl(tmp_path / "m.jsonl", [row], check=False)


def test_schema_session_consistency():
    row = schema.example_row()  # session_id 2026-12-02_s2, utt_id ..._s2_...
    schema.validate(row)
    row["utt_id"] = "v0_s3_p3_b07_ch0_1089-134686-0000"
    with pytest.raises(schema.SchemaError, match="session"):
        schema.validate(row)


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r.update(src_xyz_m=[0, 0]), "src_xyz_m"),
    (lambda r: r.update(rx_xyz_m={"ch0": [0, 0]}), "rx_xyz_m"),
    (lambda r: r["thruster"].update(xyz_m=[1.0]), "thruster.xyz_m"),
    (lambda r: r["thruster"].update(throttle_pct=150), "throttle_pct"),
])
def test_schema_geometry_and_ranges(mutate, needle):
    row = schema.example_row()
    mutate(row)
    with pytest.raises(schema.SchemaError) as e:
        schema.validate(row)
    assert any(needle in m for m in e.value.errors), e.value.errors


# ------------------------------------------------------------------ librispeech
def test_manifest_raises_on_missing_flac(tmp_path):
    """A transcript line without its FLAC means a broken extraction; must not pass silently."""
    d = tmp_path / "test-clean" / "1" / "11"
    d.mkdir(parents=True)
    sf.write(d / "1-11-0000.flac", np.zeros(16000, np.float32), 16000)
    (d / "1-11.trans.txt").write_text("1-11-0000 A\n1-11-0001 B\n")
    with pytest.raises(ValueError, match="1-11-0001"):
        ls.build_manifest(tmp_path / "test-clean", progress=False)


# ------------------------------------------------------------------ playlist
def test_pack_blocks_rejects_nan_durations():
    """Manifests built with with_duration=False carry NaN; packing must not make one huge block."""
    items = [ls.Utterance(str(k), "1", "1", "", "", float("nan"), "test-clean") for k in range(20)]
    with pytest.raises(ValueError):
        pl.pack_blocks(items)


def test_chirp_rejects_nyquist_top():
    with pytest.raises(ValueError):
        pl.lfm_chirp(fs=16000, f1=8000.0)


def test_chirp_instantaneous_frequency():
    """Zero crossings follow f0 + (f1-f0) t / T (first-principles check of the sweep law)."""
    fs = 48000
    x = pl.lfm_chirp(fs, peak_dbfs=0.0, tukey_alpha=0.0)
    t = np.arange(len(x)) / fs
    T = len(x) / fs
    phase = 2 * np.pi * (500 * t + 0.5 * 7500 / T * t * t)
    assert np.allclose(x, np.sin(phase))
    finst = np.diff(phase) / (2 * np.pi) * fs
    assert abs(finst[0] - 500) < 20 and abs(finst[-1] - 8000) < 20


def test_prepare_utterance_level_before_lpf():
    """6.5: RMS-normalized to -26 dBFS *before* the LPF; for in-band speech the LPF keeps the level."""
    rng = np.random.default_rng(0)
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(4, 3000, fs=16000, output="sos"), rng.standard_normal(16000 * 3)) * 0.3
    y, info = pl.prepare_utterance(x, 16000)
    assert abs(info["rms_dBFS"] + 26.0) < 0.2
    assert len(y) == 3 * 48000


# ------------------------------------------------------------------ scripts
def test_make_playlists_rejects_schema_invalid_positions():
    mp = _load_script("make_playlists")
    with pytest.raises(SystemExit):
        mp.parse_args(["--positions", "p0.5", "p1"])
    a = mp.parse_args(["--positions", "p05", "p1"])
    assert a.positions == ["p05", "p1"]


def test_fetch_download_rejects_truncated(tmp_path, monkeypatch):
    fl = _load_script("fetch_librispeech")

    class _Resp:
        headers = {"Content-Length": "100"}

        def __init__(self):
            self._b = [b"x" * 40]

        def read(self, n):
            return self._b.pop() if self._b else b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(fl.urllib.request, "urlopen", lambda url, **kw: _Resp())
    dst = tmp_path / "x.tar.gz"
    with pytest.raises(IOError):
        fl.download("http://example.invalid/x.tar.gz", dst)
    assert not dst.exists()


def test_make_playlists_end_to_end(tmp_path):
    """Tiny tree, --no-render: selection, disjointness, schedule, budget written."""
    rng = np.random.default_rng(1)
    for subset, base, n_spk, n_per in (("test-clean", 1000, 40, 5), ("dev-clean", 2000, 40, 1)):
        for i in range(n_spk):
            s = str(base + i)
            d = tmp_path / "LibriSpeech" / subset / s / "1"
            d.mkdir(parents=True)
            lines = []
            for k in range(n_per):
                uid = f"{s}-1-{k:04d}"
                sf.write(d / f"{uid}.flac", (0.05 * rng.standard_normal(32000)).astype(np.float32), 16000)
                lines.append(f"{uid} W")
            (d / f"{s}-1.trans.txt").write_text("\n".join(lines) + "\n")
    mp = _load_script("make_playlists")
    out = tmp_path / "out"
    assert mp.main(["--librispeech-root", str(tmp_path), "--out", str(out), "--no-render"]) == 0
    sched = pl.read_schedule(out / "schedule.json")
    rep = json.loads((out / "budget.json").read_text())
    n_blocks = len({e.content_block for e in sched})
    assert len(sched) == 3 * 2 * n_blocks + 1 and sched[-1].anchor_repeat
    assert rep["schedule_entries"] == len(sched)
    # every scheduled utt_id built from the schedule must be schema-valid
    for e in sched[:3]:
        schema.make_utt_id("v0", 2, e.position_id, e.block_id, 0, "1000-1-0000")
