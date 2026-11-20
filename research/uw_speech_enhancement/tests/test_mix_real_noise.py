"""scripts/mix_real_noise.py: real noise mixed into real cuts at a target active-speech SNR."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from uwse import audio
from uwse.evaluation import normalize as N
from uwse.uwsim.noise import band_snr_db

ROOT = Path(__file__).resolve().parents[1]
FS = 48000
FS_RAW = 96000


def _script(name):
    spec = importlib.util.spec_from_file_location(f"t_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MX = _script("mix_real_noise")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _setup(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for i, pos in enumerate(["p1", "p3", "p3"]):
        uid = f"v1_s1_{pos}_b00_ch0_10{i}-1-000{i}"
        x = sosfilt(butter(4, (300, 4000), "band", fs=FS, output="sos"), rng.standard_normal(FS))
        r0 = np.zeros(int(1.5 * FS))
        r0[FS // 4: FS // 4 + FS] = 0.1 * x * np.hanning(FS)
        y = 0.5 * r0 + 1e-4 * rng.standard_normal(r0.size)
        wp, rp = tmp_path / f"{uid}.wav", tmp_path / f"{uid}_r0.wav"
        audio.save(str(wp), y, FS, subtype="FLOAT")
        audio.save(str(rp), r0, FS, subtype="FLOAT")
        base = {"utt_id": uid, "wav": str(wp), "r0_wav": str(rp), "transcript": "a b", "speaker": f"10{i}",
                "position": pos, "thruster": "none", "kind": "speech", "snr_db_500_8000": 40.0}
        rows += [{**base, "system": "C0"}, {**base, "system": "C8"},
                 {**base, "system": "C2", "wav": str(tmp_path / "se.wav")}]
    man = tmp_path / "m.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    noise = {}
    for name in ("pool_p1", "pool_p3", "bubble"):
        n = rng.standard_normal((2, 10 * FS_RAW)) * np.array([[0.01], [0.5]])  # ch1 must not be used
        noise[name] = tmp_path / f"{name}.wav"
        audio.save(str(noise[name]), n, FS_RAW, subtype="FLOAT")
    return man, rows, noise


def _run(tmp_path, man, noise, out, seed=0):
    return MX.main(["--manifest", str(man), "--noise", f"pool@p1={noise['pool_p1']}",
                    "--noise", f"pool@p3={noise['pool_p3']}", "--noise", f"bubble={noise['bubble']}",
                    "--snr", "0,5", "--seed", str(seed), "--out", str(out)])


def test_mix_snr_manifest_and_determinism(tmp_path):
    man, rows, noise = _setup(tmp_path)
    src = {r["wav"]: _sha(r["wav"]) for r in rows if r["system"] == "C0"}
    out = tmp_path / "added"
    assert _run(tmp_path, man, noise, out) == 0
    conds = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert conds == ["bubble0dB", "bubble5dB", "pool0dB", "pool5dB"]
    mix = [json.loads(l) for l in open(out / "mix.jsonl", encoding="utf-8")]
    assert len(mix) == 3 * 4
    for m in mix:
        y, fs = audio.load(m["source_wav"])
        z, _ = audio.load(m["wav"])
        assert fs == FS and z.size == y.size
        r0, _ = audio.load(next(r["r0_wav"] for r in rows if r["utt_id"] == m["utt_id"]))
        mask = N.energy_vad(r0, FS, N.VADConfig())
        assert band_snr_db(y, z - y, FS, (500.0, 8000.0), mask) == pytest.approx(m["snr_db"], abs=0.05)
        assert m["added_snr_db_500_8000"] == pytest.approx(m["snr_db"], abs=1e-6)
    # position-specific take, and one offset per (kind, cut) shared by both SNRs
    for m in mix:
        pos = "p1" if "_p1_" in m["utt_id"] else "p3"
        assert m["file"] == str(noise["bubble"] if m["kind"] == "bubble" else noise[f"pool_{pos}"])
    by = {}
    for m in mix:
        by.setdefault((m["kind"], m["utt_id"]), set()).add(m["offset_samples"])
    assert all(len(v) == 1 for v in by.values())
    # manifest: C0 + C8 on the noisy wav, C2 dropped, quiet SNR keys removed, labelled
    mf = [json.loads(l) for l in open(out / "pool0dB" / "manifest.jsonl", encoding="utf-8")]
    assert sorted({r["system"] for r in mf}) == ["C0", "C8"] and len(mf) == 6
    assert all(r["condition"] == "pool0dB" and r["added_noise"]["kind"] == "pool" for r in mf)
    assert all("snr_db_500_8000" not in r and r["r0_wav"] for r in mf)
    assert all(Path(r["wav"]).parent == out / "pool0dB" / "wav" for r in mf)
    info = json.loads((out / "mix_info.json").read_text(encoding="utf-8"))
    assert info["dropped_systems"] == ["C2"] and len(info["noise_files"]) == 3
    assert {r: _sha(r) for r in src} == src  # the real speech cuts are untouched
    first = {m["wav"]: audio.load(m["wav"])[0] for m in mix}  # samples (float WAV headers hold a timestamp)
    assert _run(tmp_path, man, noise, out) == 0  # rerun: identical outputs, nothing appended
    assert all(np.array_equal(audio.load(w)[0], x) for w, x in first.items())
    assert len(open(out / "mix.jsonl", encoding="utf-8").readlines()) == 12
    assert _run(tmp_path, man, noise, tmp_path / "other", seed=1) == 0
    mix1 = [json.loads(l) for l in open(tmp_path / "other" / "mix.jsonl", encoding="utf-8")]
    assert [m["offset_samples"] for m in mix1] != [m["offset_samples"] for m in mix]


def test_offset_and_errors(tmp_path):
    assert MX.noise_offset(0, "pool", "u", 1000) == MX.noise_offset(0, "pool", "u", 1000)
    assert 0 <= MX.noise_offset(3, "pool", "u", 10) <= 10
    with pytest.raises(ValueError):
        MX.noise_offset(0, "pool", "u", -1)
    with pytest.raises(SystemExit):
        MX.parse_noise(["pool0=x.wav"])  # digits would break condition names
    man, _, noise = _setup(tmp_path)
    with pytest.raises(SystemExit):  # no pool take for p3
        MX.main(["--manifest", str(man), "--noise", f"pool@p1={noise['pool_p1']}", "--out", str(tmp_path / "o")])
