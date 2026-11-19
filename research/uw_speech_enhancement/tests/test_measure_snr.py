"""Per-utterance SNR of pool takes (uwse.bench.snr, scripts/measure_snr.py; method signed off 2026-11-08)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from uwse import audio
from uwse.bench import snr as SN
from uwse.bench.schema import make_utt_id

ROOT = Path(__file__).resolve().parents[1]
FS = 48000


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bursts(fs, seconds=2.0, f0=1000.0, amp=0.1):
    """Tone bursts: on for the middle half of each 0.5 s, so the VAD mask is unambiguous."""
    t = np.arange(int(seconds * fs)) / fs
    on = (t % 0.5) > 0.25
    return amp * np.sin(2 * np.pi * f0 * t) * on, on


def test_speechband_snr_matches_analytic():
    rng = np.random.default_rng(0)
    s, on = _bursts(FS)
    sigma = 0.01
    ps, mask = SN.speech_powers(s, FS)
    assert mask[on].mean() > 0.95 and mask[~on].mean() < 0.05
    pn = SN.noise_powers(sigma * rng.standard_normal(20 * FS), FS)
    f = SN.snr_fields(ps, pn)
    want = 10 * np.log10((0.1 ** 2 / 2) / (sigma ** 2 * 7500 / (FS / 2)))
    assert f["acoustic"]["snr_speechband_dB"] == pytest.approx(want, abs=0.5)
    assert f["snr_db_500_8000"] == f["acoustic"]["snr_speechband_dB"]
    # a 1 kHz tone sits in the 1 kHz octave only
    oct_ = f["acoustic"]["snr_octave_dB"]
    assert set(oct_) == {"500", "1000", "2000", "4000"} and oct_["1000"] > oct_["500"] + 20 and oct_["1000"] > oct_["2000"] + 20


def test_low_frequency_speech_counts_only_in_wide_band():
    rng = np.random.default_rng(1)
    s_hi, _ = _bursts(FS, f0=1000.0)
    s_lo, _ = _bursts(FS, f0=200.0)
    pn = SN.noise_powers(0.01 * rng.standard_normal(10 * FS), FS)
    f = SN.snr_fields(SN.speech_powers(s_hi + s_lo, FS)[0], pn)
    g = SN.snr_fields(SN.speech_powers(s_hi, FS)[0], pn)
    assert f["snr_db_500_8000"] == pytest.approx(g["snr_db_500_8000"], abs=0.3)
    # the 200 Hz energy raises speech power, but white noise also gains 50-500 Hz: net up here
    assert f["snr_db_50_8000"] > g["snr_db_50_8000"] + 1.0


def test_level_drift():
    rng = np.random.default_rng(2)
    xs = [_bursts(FS, amp=a)[0] + 1e-4 * rng.standard_normal(2 * FS) for a in (0.1, 0.05, 0.2)]
    assert SN.level_drift_db(xs, [x * 10 ** (2 / 20) for x in xs], FS) == pytest.approx(2.0, abs=0.01)
    assert SN.level_drift_db(xs, xs, FS) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(SN.level_drift_db([], [], FS))


def _session(tmp_path, anchor_gain=1.0):
    """p3, two utterances; block 0 off, block 1 on, block 2 = anchor repeat of block 0 (off)."""
    rng = np.random.default_rng(3)
    items = ["1089-134686-0000", "1089-134686-0001"]
    sig = {it: _bursts(FS, amp=a)[0] for it, a in zip(items, (0.1, 0.05))}
    quiet, loud = 1e-4, 0.02
    wav = tmp_path / "wav"
    wav.mkdir()
    rows = []
    for block, state, anchor, sigma, g in ((0, "off", False, quiet, 1.0), (1, "on", False, loud, 1.0),
                                          (2, "off", True, quiet, anchor_gain)):
        for it in items:
            uid = make_utt_id("v0", 2, "p3", block, 0, it)
            p = wav / f"{uid}.wav"
            audio.save(str(p), g * sig[it] + sigma * rng.standard_normal(2 * FS), FS, subtype="FLOAT")
            rows.append({"utt_id": uid, "librispeech_id": it, "position_id": "p3", "thruster_state": state,
                         "anchor_repeat": anchor, "channel": 0, "file_channel": 1, "fs": FS, "wav": str(p)})
    (tmp_path / "align.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    for name, sigma in (("off", quiet), ("on", loud)):
        # 2-channel take; channel 0 is junk so a wrong channel shows up
        x = np.stack([rng.standard_normal(12 * FS), sigma * rng.standard_normal(12 * FS)])
        audio.save(str(tmp_path / f"noise_{name}.wav"), x, FS, subtype="FLOAT")
    return sig, items, quiet, loud


def test_measure_snr_cli(tmp_path):
    ms = _script("measure_snr")
    sig, items, quiet, loud = _session(tmp_path)
    assert ms.main(["--align", str(tmp_path / "align.jsonl"), "--noise", f"p3:off={tmp_path / 'noise_off.wav'}",
                    "--noise", f"p3:on={tmp_path / 'noise_on.wav'}"]) == 0
    out = {r["utt_id"]: r for r in map(json.loads, (tmp_path / "snr.jsonl").read_text().splitlines())}
    assert len(out) == 6 and all("acoustic" in r for r in out.values())
    for it, amp in zip(items, (0.1, 0.05)):
        on = out[make_utt_id("v0", 2, "p3", 1, 0, it)]
        off = out[make_utt_id("v0", 2, "p3", 0, 0, it)]
        # speech always from the off cut; noise from the take of the row's state
        assert on["speech_cut"] == off["speech_cut"] == make_utt_id("v0", 2, "p3", 0, 0, it)
        assert on["noise_take"].endswith("noise_on.wav") and off["noise_take"].endswith("noise_off.wav")
        want = 10 * np.log10((amp ** 2 / 2) / (loud ** 2 * 7500 / (FS / 2)))
        assert on["snr_db_500_8000"] == pytest.approx(want, abs=0.5)
        assert off["snr_db_500_8000"] > on["snr_db_500_8000"] + 30
        assert on["acoustic"]["snr_50_8000_dB"] == on["snr_db_50_8000"] and on["flags"] == []
    d = json.loads((tmp_path / "snr_drift.json").read_text())
    assert d["max_drift_db"] == SN.MAX_DRIFT_DB and abs(d["sessions"]["2"]["drift_db"]) < 0.05
    assert not d["sessions"]["2"]["flagged"] and d["sessions"]["2"]["n_utts"] == 2


def test_measure_snr_flags_level_drift_and_skips(tmp_path):
    ms = _script("measure_snr")
    _session(tmp_path, anchor_gain=10 ** (1.5 / 20))  # anchor 1.5 dB louder than the original
    assert ms.main(["--align", str(tmp_path / "align.jsonl"), "--noise", f"p3:off={tmp_path / 'noise_off.wav'}"]) == 0
    out = list(map(json.loads, (tmp_path / "snr.jsonl").read_text().splitlines()))
    d = json.loads((tmp_path / "snr_drift.json").read_text())["sessions"]["2"]
    assert d["drift_db"] == pytest.approx(1.5, abs=0.05) and d["flagged"]
    done = [r for r in out if "acoustic" in r]
    assert len(done) == 4 and all(r["flags"] == ["snr_level_drift"] for r in done)
    skipped = [r for r in out if "skipped" in r]
    assert len(skipped) == 2 and all("no on noise take" in r["skipped"] for r in skipped)
    with pytest.raises(SystemExit):
        ms.main(["--align", str(tmp_path / "align.jsonl"), "--noise", "p3:loud=x.wav"])


def test_schema_accepts_snr_fields():
    from uwse.bench import schema

    assert "snr_level_drift" in schema.FLAGS
    assert "snr_50_8000_dB" in {f.name for f in schema.dataclasses.fields(schema.Acoustic)}


def test_score_joins_snr_jsonl(tmp_path):
    spec = importlib.util.spec_from_file_location("score_cli", ROOT / "scripts" / "score.py")
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    p = tmp_path / "snr.jsonl"
    p.write_text(json.dumps({"utt_id": "u1", "acoustic": {}, "snr_db_500_8000": 1.0, "snr_db_50_8000": 4.0,
                             "flags": ["snr_level_drift"]}) + "\n" + json.dumps({"utt_id": "u2", "skipped": "x"}) + "\n")
    rows = [{"utt_id": "u1", "system": "C0"}, {"utt_id": "u1", "system": "C8"}, {"utt_id": "u2", "system": "C0"}]
    assert score.join_snr(rows, str(p)) == 2
    assert rows[1]["snr_db_50_8000"] == 4.0 and rows[0]["snr_level_drift"] and "snr_db_500_8000" not in rows[2]


def test_noise_gate_drops_thruster_off_spans():
    rng = np.random.default_rng(4)
    on = 0.02 * rng.standard_normal(20 * FS)
    take = np.concatenate([1e-4 * rng.standard_normal(3 * FS), on, 1e-4 * rng.standard_normal(4 * FS)])
    full = SN.noise_powers(on, FS, trim_s=0.0)["500-8000"]
    plain = SN.noise_powers(take, FS, gate_db=None)["500-8000"]
    gated = SN.noise_powers(take, FS, gate_db=6.0)["500-8000"]
    assert 10 * np.log10(full / plain) > 0.5  # off spans inside the take pull the plain mean down
    assert 10 * np.log10(full / gated) == pytest.approx(0.0, abs=0.1)
    # stationary take: the gate changes nothing material
    assert SN.NOISE_GATE_DB == 6.0 and SN.noise_powers(take, FS)["500-8000"] == gated  # gated by default
    assert 10 * np.log10(SN.noise_powers(on, FS)["500-8000"] / SN.noise_powers(on, FS, gate_db=None)["500-8000"]) \
        == pytest.approx(0.0, abs=0.05)


def test_noise_take_from_each_cuts_recorder(tmp_path):
    ms = _script("measure_snr")
    a = ms.parse_args(["--align", "x", "--session-dir", str(tmp_path)])
    sess = tmp_path / "raw"
    for rec in ("daisyA", "daisyB"):
        (sess / rec).mkdir(parents=True)
        for take in ("p3_noise", "p3_noise_on"):
            (sess / rec / f"s2_{take}_{rec}.wav").write_bytes(b"")
    r = {"raw_file": str(sess / "daisyB" / "s2_p3_speech_daisyB.wav")}
    assert ms.recorder_of(a, r) == "daisyB"
    assert ms.noise_path(a, 2, "p3", "on", ms.recorder_of(a, r)) == sess / "daisyB" / "s2_p3_noise_on_daisyB.wav"
    # outside raw/<rec>/ only --recorder can say which
    assert ms.recorder_of(a, {"raw_file": "elsewhere/take.wav"}) is None
    a2 = ms.parse_args(["--align", "x", "--session-dir", str(tmp_path), "--recorder", "daisyA"])
    assert ms.noise_path(a2, 2, "p3", "off", ms.recorder_of(a2, {})) == sess / "daisyA" / "s2_p3_noise_daisyA.wav"
