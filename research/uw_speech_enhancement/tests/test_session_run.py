"""Session tooling: take plan and cues, levels, take -> check -> import on synthetic takes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_ingest import FS, _bench, _record
from uwse import audio
from uwse.bench import session as ss
from uwse.bench.schema import parse_utt_id

ROOT = Path(__file__).resolve().parents[1]
CMAP = ROOT / "configs" / "session_channels_v0.json"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _thruster_bench(tmp_path):
    """3 blocks at p1 with states off, on, on (+ anchor); session WAVs written."""
    play, utts = _bench(tmp_path)
    states = ["off", "on", "on"]
    sched = json.loads((tmp_path / "schedule.json").read_text())
    for e in sched:
        e["thruster_state"] = states[e["block_id"]] if not e["anchor_repeat"] else "off"
    (tmp_path / "schedule.json").write_text(json.dumps(sched))
    idx = json.loads((tmp_path / "session" / "p1_speech.json").read_text())
    for b, s in zip(idx["blocks"], states):
        b["thruster_state"] = s
    (tmp_path / "session" / "p1_speech.json").write_text(json.dumps(idx))
    audio.save(str(tmp_path / "session" / "p1_speech.wav"), play, FS, subtype="PCM_24")
    first = idx["blocks"][0]
    audio.save(str(tmp_path / "session" / "anchor_p1.wav"),
               play[first["start_sample"]:first["start_sample"] + first["n_samples"]], FS, subtype="PCM_24")
    return play, utts


def test_plan_order_chirps_and_thruster_cues(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    _thruster_bench(bench)
    plan = ss.build_plan(bench, extra_noise=("quiet_pool",))
    assert [t.take_id for t in plan] == ["p1_noise", "p1_noise_on", "p1_speech", "p1_anchor", "noise_quiet_pool"]
    assert plan[1].kind == "noise" and plan[1].file is None and "THRUSTER ON" in plan[1].cues[0].text
    plan = [t for t in plan if t.take_id != "p1_noise_on"]
    sp = plan[1]
    assert sp.importable and sp.n_blocks == 3 and sp.n_utts == 6 and len(sp.chirps_s) == 6
    th = [c for c in sp.cues if c.text.startswith("THRUSTER")]
    assert [c.text.split()[1] for c in th] == ["OFF", "ON", "OFF"]
    # ON is cued right after block 0's end chirp, before block 1 starts
    assert sp.chirps_s[1] < th[1].t_s < sp.chirps_s[2]
    assert plan[2].kind == "anchor" and plan[2].n_blocks == 1
    rows = ss.utterance_rows(bench, "p1")
    assert len(rows) == 6 and rows[0]["start_s"] < rows[1]["start_s"]


def test_levels_and_drive():
    fs = 48000
    x = np.zeros(fs * 4)
    x[fs:2 * fs] = 0.05 * np.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)
    lv = ss.file_levels(x, fs)
    assert abs(lv["active_rms_dBFS"] - 20 * np.log10(0.05)) < 0.1 and abs(lv["active_fraction"] - 0.25) < 0.01
    # vrms_at_0dbfs is the meter reading of a full-scale sine (sample RMS -3.01 dBFS): a signal at
    # sample RMS 0 dBFS re the amp input drives sqrt(2) x that (fix of the 3.01 dB low estimate)
    d = ss.drive_estimate(-26.0, vrms_at_0dbfs=1.0, amp_gain_db=26.0, peak_dbfs=-20.0)
    assert d["drive_vrms"] == pytest.approx(np.sqrt(2), rel=1e-9)
    assert d["SL_est_dB"] == pytest.approx(180 + 20 * np.log10(np.sqrt(2) / 20), abs=1e-9)
    assert d["drive_peak_v"] == pytest.approx(np.sqrt(2) * 10 ** (6 / 20), rel=1e-9)
    assert abs(ss.loopback_drive(-40.0, in_dbfs_at_1vrms=-6.0, atten_db=40.0) - 10 ** (6 / 20)) < 1e-9


def test_import_channel_ids(tmp_path):
    play, _ = _bench(tmp_path)
    x = _record(play, lead_s=1.0, drift_ppm=20.0)
    rec = tmp_path / "t.wav"
    audio.save(str(rec), x * 0.9, FS, subtype="PCM_24")
    out = tmp_path / "out"
    mod = _script("import_recording")
    assert mod.main(["--bench", str(tmp_path), "--position", "p1", "--rec", str(rec), "--session", "2",
                     "--channels", "0,1", "--channel-ids", "2,3", "--sync-channel", "0", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in (out / "align.jsonl").read_text().splitlines()]
    assert sorted({r["channel"] for r in rows}) == [2, 3]
    for r in rows:
        assert parse_utt_id(r["utt_id"])["channel"] == r["channel"] == r["file_channel"] + 2
    with pytest.raises(SystemExit):
        mod.main(["--bench", str(tmp_path), "--position", "p1", "--rec", str(rec), "--session", "2",
                  "--channels", "0,1", "--channel-ids", "2", "--out", str(out)])


def test_take_check_import_end_to_end(tmp_path, monkeypatch):
    bench, sdir, real = tmp_path / "bench", tmp_path / "sess", tmp_path / "real"
    bench.mkdir()
    play, _ = _thruster_bench(bench)
    rs = _script("run_session")
    assert rs.main(["plan", "--bench", str(bench), "--session", "2", "--channel-map", str(CMAP),
                    "--session-dir", str(sdir), "--out", str(real), "--extra-noise"]) == 0
    sheet = (sdir / "RUN_SHEET.md").read_text(encoding="utf-8")
    assert "s2_p1_speech_daisyB.wav" in sheet and "THRUSTER ON" in sheet and "hydrophone ch3" in sheet
    # the thruster-on noise take must tell the operator to run the thruster (S1 dry run)
    assert "| p1_noise_on | (nothing; record only; THRUSTER ON and steady before recording) |" in sheet
    assert "### p1_noise_on: thruster cues" in sheet and "### p1_noise: thruster cues" not in sheet
    assert (sdir / "utterances_p1_speech.tsv").read_text().count("\n") == 7

    # the interface: loopback on in1, monitor hydrophone on in2 (the fake ignores the played pre-roll)
    take = _record(play, lead_s=2.0, drift_ppm=0.0)

    def fake_playrec(x, fs, rec, a, cues=None):
        assert len(x) == len(play) and rec.name == "iface"
        return take

    monkeypatch.setattr(rs, "_playrec", fake_playrec)
    assert rs.main(["take", "--session-dir", str(sdir), "--take-id", "p1_speech", "--range-m", "1.0",
                    "--water-temp-c", "26.5", "--pumps", "off"]) == 0
    row = json.loads((sdir / "takes.jsonl").read_text().splitlines()[-1])
    assert row["qc"]["iface"]["ok"] and row["range_m"] == 1.0 and row["pumps"] == "off"

    # a hydrophone-only recorder on its own clock and at 96 kHz (file ch 0/1 = hydrophones 2/3)
    hb = _record(play, lead_s=0.7, drift_ppm=-60.0, seed=3)[1]
    hb2 = np.stack([hb, hb + np.random.default_rng(4).standard_normal(hb.size) * 1e-3])
    db = sdir / "raw" / "daisyB" / "s2_p1_speech_daisyB.wav"
    audio.save(str(db), audio.resample(hb2, FS, 96000) * 0.9, 96000, subtype="PCM_24")
    assert rs.main(["check", "--session-dir", str(sdir), "--take-id", "p1_speech",
                    "--rec", f"daisyB={db}", "--run-import"]) == 0
    rows = [json.loads(line) for line in (real / "align.jsonl").read_text().splitlines()]
    assert len(rows) == 12 and sorted({r["channel"] for r in rows}) == [2, 3]
    assert all(r["fs_raw"] == 96000 and r["flags"] == [] and abs(r["drift_ppm"] + 60) < 5 for r in rows)
    assert {parse_utt_id(r["utt_id"])["session"] for r in rows} == {2}
    assert "--channel-ids 2,3" in (sdir / "import_cmds.txt").read_text()


def test_record_daisy_writes_batches(tmp_path):
    rd = _script("record_daisy")
    rng = np.random.default_rng(0)
    chunks = [(rng.uniform(-0.5, 0.5, (640, 2)).astype(np.float32), 96000, 24) for _ in range(20)]
    stats = {"dropped": 1}
    info = rd.record(iter(chunks), tmp_path / "d" / "x.wav", 96000, None, stats)
    assert info["n_samples"] == 12800 and info["dropped"] == 1
    y, fs = audio.load(str(tmp_path / "d" / "x.wav"), always_2d=True)
    assert fs == 96000 and y.shape == (2, 12800)
    assert np.max(np.abs(y - np.concatenate([c[0] for c in chunks]).T)) < 1e-6


def _sweeps(bench):
    audio.save(str(bench / "session" / "sweeps.wav"), 0.5 * np.ones(4 * FS), FS, subtype="FLOAT")
    (bench / "session" / "sweeps.json").write_text(json.dumps({"fs": FS, "chirp_start_sample": FS // 2,
                                                               "sweep_start_samples": [FS, 2 * FS]}))


def test_optional_take_groups(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    _thruster_bench(bench)
    _sweeps(bench)
    base = [t.take_id for t in ss.build_plan(bench, extra_noise=())]
    assert not any(t.startswith("e") for t in base)  # off by default: the speech-pool plan only
    plan = ss.build_plan(bench, extra_noise=(), groups={"e2": {"depths_m": [1.0, 1.5]}, "e3": {"points": ["r1", "r3", "r6"]},
                                                        "e4": {}})
    assert [t.take_id for t in plan[:len(base)]] == base
    extra = plan[len(base):]
    e2 = [t for t in extra if t.take_id.startswith("e2_")]
    assert len(e2) == 2 * 3 * 3 and "e2_d1p5_a45_att6" in {t.take_id for t in e2}
    t = next(t for t in e2 if t.take_id == "e2_d1p5_a45_att6")
    assert t.kind == "sweeps" and t.file == "sweeps.wav" and t.gain_db == -6.0 and not t.importable
    assert "1.5 m" in t.setup and "45 deg" in t.setup and len(t.chirps_s) == 1 and len(t.cues) == 2
    e3 = [t.take_id for t in extra if t.take_id.startswith("e3_")]
    assert e3 == ["e3_r1", "e3_r3", "e3_r6", "e3_r1_rep", "e3_r3_rep", "e3_r1_pumps_on", "e3_r1_pumps_off",
                  "e3_r1_noframe"]
    e4 = [t for t in extra if t.take_id.startswith("e4_")]
    assert len(e4) == 10 * 3 * 2 + 3 + 3 + 3 + 2 and all(t.file is None and t.kind == "noise_bank" for t in e4)
    assert all("THRUSTER ON" in t.cues[0].text and t.cues[-1].text.startswith("THRUSTER OFF") for t in e4)
    assert "e4_A_rev_t30_r0p5" in {t.take_id for t in e4} and len({t.take_id for t in plan}) == len(plan)
    with pytest.raises(ValueError, match="grid point"):
        ss.build_plan(bench, groups={"e3": {"points": []}})
    with pytest.raises(ValueError, match="unknown"):
        ss.build_plan(bench, groups={"e5": {}})


def test_take_groups_cli_sheet_and_gain(tmp_path, monkeypatch):
    bench, sdir = tmp_path / "bench", tmp_path / "sess"
    bench.mkdir()
    _thruster_bench(bench)
    _sweeps(bench)
    rs = _script("run_session")
    assert rs.main(["plan", "--bench", str(bench), "--session", "1", "--channel-map", str(CMAP),
                    "--session-dir", str(sdir), "--out", str(tmp_path / "real"), "--extra-noise",
                    "--e2", "--e2-angles-deg", "0", "--e2-atten-db", "0", "12", "--e3-points", "r1", "--e4"]) == 0
    sheet = (sdir / "RUN_SHEET.md").read_text(encoding="utf-8")
    assert "| e2_d1_a0_att12 | sweeps.wav at -12 dB |" in sheet and "E3: grid point r1, PUMPS ON" in sheet
    assert "| e4_flow_axis | (nothing; record only; THRUSTER ON and steady before recording) |" in sheet
    assert json.loads((sdir / "session.json").read_text())["groups"]["e3"]["points"] == ["r1"]
    got = {}

    def fake_playrec(x, fs, rec, a, cues=None):
        got["peak"] = float(np.max(np.abs(x)))
        return np.zeros((2, 10))

    monkeypatch.setattr(rs, "_playrec", fake_playrec)
    assert rs.main(["take", "--session-dir", str(sdir), "--take-id", "e2_d1_a0_att12"]) == 0
    assert got["peak"] == pytest.approx(0.5 * 10 ** (-12 / 20), rel=1e-3)
    # a plan made without groups still loads (session.json from before this change has no "groups")
    meta = json.loads((sdir / "session.json").read_text())
    meta.pop("groups")
    (sdir / "session.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="unknown take"):
        rs.main(["take", "--session-dir", str(sdir), "--take-id", "e3_r1"])
