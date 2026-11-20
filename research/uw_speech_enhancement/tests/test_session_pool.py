"""Pool plan (2026-11-19, no thruster): bench build, session files, plan, Daisy commands, verify, check, import.

A tiny synthetic LibriSpeech tree goes through make_playlists.py --plan pool,
make_session_audio.py and run_session.py plan / daisy-cmds / verify / check, and
synthetic Daisy recordings of the dry run and the chain-reference take are QC'd and
imported. Also covers the review fixes: dead session paths, iface without sync
channel, 3.01 dB drive error, idempotent check/import, redo naming, file length.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from test_ingest import FS, _record
from test_session_run import _script
from uwse import audio
from uwse.bench import ingest
from uwse.bench import playlist as pl
from uwse.bench import session as ss
from uwse.bench.schema import parse_utt_id

ROOT = Path(__file__).resolve().parents[1]
CMAP1 = ROOT / "configs" / "session_channels_v1.json"
CMAP_S1 = ROOT / "configs" / "session_channels_s1.json"


def _libri(root: Path, seed: int = 1) -> None:
    """test-clean: 40 speakers x 5 short utterances; dev-clean: 40 speakers x 2."""
    rng = np.random.default_rng(seed)
    for subset, base, n_spk, n_per in (("test-clean", 1000, 40, 5), ("dev-clean", 2000, 40, 2)):
        for i in range(n_spk):
            s = str(base + i)
            d = root / "LibriSpeech" / subset / s / "1"
            d.mkdir(parents=True)
            lines = []
            for k in range(n_per):
                uid = f"{s}-1-{k:04d}"
                m = int(16000 * (0.25 + 0.1 * rng.random()))
                sf.write(d / f"{uid}.flac", (0.05 * rng.standard_normal(m) * np.hanning(m)).astype(np.float32), 16000)
                lines.append(f"{uid} WORD{k}")
            (d / f"{s}-1.trans.txt").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def pool_bench(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pool")
    _libri(tmp)
    bench = tmp / "bench_v1"
    mp = _script("make_playlists")
    assert mp.main(["--plan", "pool", "--librispeech-root", str(tmp), "--out", str(bench), "--min-dur", "0.1",
                    "--target-speech-s", "6", "--bubble-min", "0.5"]) == 0
    ma = _script("make_session_audio")
    assert ma.main(["--bench", str(bench), "--max-take-min", "2", "--n-sweeps", "1", "--sweep-T", "1",
                    "--sweep-silence-s", "1", "--tone-s", "2"]) == 0
    return bench


def test_pool_bench_selection_schedule_and_files(pool_bench):
    b = pool_bench
    sel = {k: [json.loads(x) for x in (b / "selection" / f"{k}.jsonl").read_text().splitlines()]
           for k in ("v0-test", "v0-dev", "v0-pilot")}
    assert len(sel["v0-test"]) == 200 and len(sel["v0-pilot"]) == 20
    pilot = {u["utt_id"] for u in sel["v0-pilot"]}
    assert not pilot & {u["utt_id"] for u in sel["v0-dev"]} and not pilot & {u["utt_id"] for u in sel["v0-test"]}
    assert all(u["subset"] == "dev-clean" for u in sel["v0-pilot"])
    assert len({u["speaker"] for u in sel["v0-pilot"]}) == 20  # one per speaker
    rule = json.loads((b / "selection" / "pilot_rule.json").read_text())
    assert sorted(rule["utt_ids"]) == sorted(pilot) and "default_rng([seed, 3])" in rule["rule"]

    sched = pl.read_schedule(b / "schedule.json")
    assert pl.is_pool_schedule(sched) and all(e.thruster_state == "off" and e.position_id == "" for e in sched)
    assert len({e.block_id for e in sched}) == len(sched)  # unique block ids across programs
    progs = [e.program for e in sched]
    assert progs == sorted(progs, key=pl.POOL_PROGRAMS.index)
    speech = [e.content_block for e in sched if e.program == "speech"]
    anchor = [e for e in sched if e.program == "anchor"]
    assert len(anchor) == 1 and anchor[0].anchor_repeat and anchor[0].content_block == speech[0]
    bub = [e.content_block for e in sched if e.program == "bubble"]
    assert bub and set(bub) <= set(speech)
    assert json.loads((b / "selection" / "bubble_rule.json").read_text())["blocks"] == bub

    progs_json = json.loads((b / "session" / "programs.json").read_text())
    files = {f["file"]: f for f in progs_json["files"]}
    speech_parts = [f for f in progs_json["files"] if f["program"] == "speech"]
    assert len(speech_parts) >= 2 and all(f["minutes"] <= 2.0 for f in speech_parts)  # --max-take-min 2
    assert sum(f["n_utts"] for f in speech_parts) == 200
    assert {"pilot.wav", "anchor.wav"} <= set(files)
    for f in progs_json["files"]:  # index and audio agree; every block is where the index says
        fs, blocks = ingest.load_played(b, Path(f["file"]).stem)
        n = sf.info(str(b / "session" / f["file"])).frames
        assert blocks[-1].start_sample + blocks[-1].timing["n_samples"] <= n
    tone = json.loads((b / "session" / "tone_1k.json").read_text())
    x, fs = audio.load(str(b / "session" / "tone_1k.wav"))
    assert ss.file_levels(x, fs)["active_rms_dBFS"] == pytest.approx(tone["rms_dBFS"], abs=0.2)  # 50 ms fades


def test_split_parts():
    ma = _script("make_session_audio")
    assert ma.split_parts([10] * 6, 100) == [[0, 1, 2, 3, 4, 5]]
    g = ma.split_parts([10] * 6, 25)
    assert sum(g, []) == list(range(6)) and all(sum(10 for _ in p) <= 25 for p in g) and len(g) == 3
    assert ma.split_parts([50, 5, 5], 20) == [[0], [1, 2]]
    assert ma.split_parts([50, 15, 15, 15], 20) == [[0], [1], [2], [3]]  # an over-long item stands alone
    assert ma.split_parts([5] * 7, 20) == [[0, 1, 2, 3], [4, 5, 6]]


def test_make_playlists_refuses_frozen_benches():
    mp = _script("make_playlists")
    for name in ("v0_s1", "v0_pilot"):
        with pytest.raises(SystemExit):
            mp.parse_args(["--plan", "pool", "--out", str(ROOT / "data" / "bench" / name)])
    assert mp.parse_args(["--plan", "pool"]).out == ROOT / "data" / "bench" / "v1"


def _plan(pool_bench, sdir, *extra):
    rs = _script("run_session")
    assert rs.main(["plan", "--bench", str(pool_bench), "--session", "2", "--channel-map", str(CMAP1),
                    "--session-dir", str(sdir), "--out", str(sdir.parent / "real"), "--distances", "1", "3", "6",
                    "--pool-length-m", "10", "--pool-width-m", "4.5", "--pool-depth-m", "1.8", "--date", "2026-12-02",
                    *extra]) == 0
    return rs


def test_pool_plan_takes_and_run_sheet(pool_bench, tmp_path, capsys):
    sdir = tmp_path / "sess"
    rs = _plan(pool_bench, sdir)
    meta = json.loads((sdir / "session.json").read_text())
    assert meta["plan"] == "pool" and meta["pool"]["distances_m"] == [1.0, 3.0, 6.0]
    plan = ss.plan_from_meta(meta)
    ids = [t.take_id for t in plan]
    n_parts = len([f for f in json.loads((pool_bench / "session" / "programs.json").read_text())["files"]
                   if f["program"] == "speech"])
    per_d = lambda p: [f"{p}_noise", f"{p}_sweeps", *[f"{p}_speech_{k}" for k in range(1, n_parts + 1)],  # noqa: E731
                       f"{p}_anchor"]
    assert ids == ["dryrun", "ref_sweeps", "ref_speech", "pilot_speech", *per_d("p1"), *per_d("p3"), *per_d("p6"),
                   "p3bub_noise", "p3bub_speech"]
    t = {t.take_id: t for t in plan}
    assert t["dryrun"].dryrun and not t["dryrun"].importable and t["dryrun"].played == "anchor"
    assert t["ref_speech"].played == "pilot" and t["ref_speech"].position_id == "p1ref"
    assert t["pilot_speech"].position_id == "p3pilot" and t["pilot_speech"].n_utts == 20
    assert t["p3bub_noise"].duration_s == 120.0 and t["p3bub_noise"].bubbles
    assert t["p3bub_speech"].played == "bubble" and t["p6_speech_1"].distance_m == 6.0
    assert not any("THRUSTER" in c.text for tk in plan for c in tk.cues)
    # distances are a plan option, chosen on the day
    sd2 = tmp_path / "sess2"
    _plan(pool_bench, sd2, "--replan", "--no-bubbles", "--no-dry-run")
    meta2 = json.loads((sd2 / "session.json").read_text())
    meta2["pool"]["distances_m"] = [0.5, 2]
    assert [x.take_id for x in ss.plan_from_meta(meta2)][:4] == ["ref_sweeps", "ref_speech", "pilot_speech",
                                                                 "p0p5_noise"]
    with pytest.raises(SystemExit, match="exists"):
        _plan(pool_bench, sdir)

    sheet = (sdir / "RUN_SHEET.md").read_text(encoding="utf-8")
    for needle in ("no thruster", "BRIDGED", "GFCI", "NOBODY IN THE WATER", "20 Vrms", "5 kHz",
                   "length 10 m", "width 4.5 m", "depth 1.8 m", "date 2026-12-02", "Decision rule",
                   "dryrun/raw/daisyA/s2_dryrun_daisyA.wav", "raw/daisyB/s2_p3_speech_1_daisyB.wav",
                   "| take | attempt | time | distance (m) | speaker depth (m) | hydrophone depth (m) | orientation | "
                   "water T (°C) | amp Vrms (measured) | brick V | recorder gains | pump on/off | QC | notes |",
                   "--board 376636603433", "--board 376C36573433"):
        assert needle in sheet, needle
    assert "THRUSTER" not in sheet.upper().replace("NO THRUSTER", "")
    daisy_txt = (sdir / "daisy_cmds.txt").read_text()
    assert daisy_txt.count("record_daisy.py") == 2 * 2 * len(plan)  # attempt 1 + the attempt-2 redo lines
    assert "s2_p3_anchor_r2_daisyA.wav" in daisy_txt
    assert "--take-id <take> --run-import" not in sheet
    assert "/chainref" in sheet and "/pilot" in sheet and "/bubble" in sheet
    assert (sdir / "dryrun" / "raw" / "daisyA").is_dir() and not (sdir / "raw" / "iface").exists()
    assert (sdir / "utterances_pilot_speech.tsv").read_text().count("\n") == 21

    capsys.readouterr()
    assert rs.main(["daisy-cmds", "--session-dir", str(sdir), "--take-id", "p3_anchor", "--attempt", "2",
                    "--linux-sdir", "/home/op/uwse/sessions/s2"]) == 0
    out = capsys.readouterr().out.splitlines()
    cmd = [line for line in out if "record_daisy.py" in line]
    secs = ss.record_seconds(t["p3_anchor"])
    assert cmd[0] == (f"python scripts/record_daisy.py --board 376636603433 --rate 96000 --seconds {secs} "
                      "--out /home/op/uwse/sessions/s2/raw/daisyA/s2_p3_anchor_r2_daisyA.wav &")
    assert cmd[1].endswith("raw/daisyB/s2_p3_anchor_r2_daisyB.wav &") and out[out.index(cmd[1]) + 1] == "wait"


def _take96(play: np.ndarray, drift_ppm: float = -40.0, seed: int = 3) -> np.ndarray:
    """A hydrophone take of ``play`` on a 96 kHz Daisy clock (computed once per played file)."""
    hyd = _record(play, lead_s=0.8, drift_ppm=drift_ppm, seed=seed)[1]
    return audio.resample(hyd, FS, 96000) * 0.9


def _daisy(path: Path, hyd96: np.ndarray, seed: int) -> None:
    """Write a two-hydrophone Daisy file (+ record_daisy.py sidecar) from a precomputed take."""
    rng = np.random.default_rng(seed)
    two = np.stack([hyd96 + rng.standard_normal(hyd96.size) * 1e-4, hyd96 + rng.standard_normal(hyd96.size) * 1e-3])
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.save(str(path), two, 96000, subtype="PCM_24")
    path.with_suffix(".wav.json").write_text(json.dumps({"dropped": 0}))


def test_pool_take_verify_check_import(pool_bench, tmp_path, monkeypatch, capsys):
    sdir = tmp_path / "sess"
    rs = _plan(pool_bench, sdir)
    real = tmp_path / "real"
    played = []
    monkeypatch.setattr(rs, "_play_stream", lambda path, dur, a, cues, gain_db=0.0: played.append(path))
    # the interface plays only (no loopback): take logs metadata, records nothing
    assert rs.main(["take", "--session-dir", str(sdir), "--take-id", "ref_speech", "--distance-m", "0.98",
                    "--speaker-depth-m", "1.2", "--hyd-depth-m", "1.2", "--orientation", "mark A",
                    "--water-temp-c", "27", "--amp-vrms", "1.8", "--brick-v", "48", "--rec-gain", "A=20,B=20"]) == 0
    assert played[-1].name == "pilot.wav"
    row = json.loads((sdir / "takes.jsonl").read_text().splitlines()[-1])
    assert row["attempt"] == 1 and row["distance_m"] == 0.98 and row["pump"] == "off" and row["amp_vrms"] == 1.8
    assert set(row["recordings"]) == {"daisyA", "daisyB"}
    with pytest.raises(SystemExit, match="attempt 2"):  # never log the same attempt twice
        rs.main(["take", "--session-dir", str(sdir), "--take-id", "ref_speech"])
    with pytest.raises(SystemExit, match="attempt 2"):  # not even as metadata only
        rs.main(["take", "--session-dir", str(sdir), "--take-id", "ref_speech", "--no-audio"])

    # verify before the files are copied: missing
    assert rs.main(["verify", "--session-dir", str(sdir), "--take-id", "ref_speech"]) == 1
    play, _ = audio.load(str(pool_bench / "session" / "pilot.wav"))
    play = _take96(play)
    a1 = ss.raw_path(sdir, 2, "ref_speech", "daisyA")
    b1 = ss.raw_path(sdir, 2, "ref_speech", "daisyB")
    _daisy(a1, play, 3)
    assert rs.main(["verify", "--session-dir", str(sdir), "--take-id", "ref_speech"]) == 1  # daisyB still missing
    _daisy(b1, play, 4)
    assert rs.main(["verify", "--session-dir", str(sdir), "--take-id", "ref_speech"]) == 0

    assert rs.main(["check", "--session-dir", str(sdir), "--take-id", "ref_speech", "--run-import"]) == 0
    assert not (real / "align.jsonl").exists()  # the chain reference never enters the quiet manifest
    real = real / "chainref"
    rows = [json.loads(x) for x in (real / "align.jsonl").read_text().splitlines()]
    assert len(rows) == 20 * 4 and sorted({r["channel"] for r in rows}) == [0, 1, 2, 3]
    assert all(r["flags"] == [] and r["played"] == "pilot" and r["bench_split"] == "v0-pilot" for r in rows)
    ids = {parse_utt_id(r["utt_id"])["position_id"] for r in rows}
    assert ids == {"p1ref"} and len({r["utt_id"] for r in rows}) == len(rows)
    cmds = (sdir / "import_cmds.txt").read_text().splitlines()
    assert len(cmds) == 2 and all("--played pilot" in c and "real/chainref" in c for c in cmds)

    # re-running check + import is idempotent (no duplicate import lines or align rows)
    assert rs.main(["check", "--session-dir", str(sdir), "--take-id", "ref_speech", "--run-import"]) == 0
    assert (sdir / "import_cmds.txt").read_text().splitlines() == cmds
    assert len((real / "align.jsonl").read_text().splitlines()) == 80

    # a redo is attempt 2 with its own files; check picks the latest attempt, the rows are replaced
    _daisy(ss.raw_path(sdir, 2, "ref_speech", "daisyA", attempt=2), play, 5)
    _daisy(ss.raw_path(sdir, 2, "ref_speech", "daisyB", attempt=2), play, 6)
    assert rs.main(["check", "--session-dir", str(sdir), "--take-id", "ref_speech", "--run-import"]) == 0
    rows = [json.loads(x) for x in (real / "align.jsonl").read_text().splitlines()]
    assert len(rows) == 80 and all("_r2_" in Path(r["raw_file"]).name for r in rows)
    assert all("_r2_" in c for c in (sdir / "import_cmds.txt").read_text().splitlines())

    # noise and sweeps takes: no chirp QC, but an OK line
    _daisy(ss.raw_path(sdir, 2, "p1_noise", "daisyA"), play[:96000], 11)
    _daisy(ss.raw_path(sdir, 2, "p1_noise", "daisyB"), play[:96000], 12)
    capsys.readouterr()
    assert rs.main(["check", "--session-dir", str(sdir), "--take-id", "p1_noise"]) == 0
    out = capsys.readouterr().out
    assert "OK   p1_noise daisyA" in out and "OK   p1_noise daisyB" in out

    # a failing redo never replaces the passing import line; it is kept as a # QC FAIL comment
    _daisy(ss.raw_path(sdir, 2, "ref_speech", "daisyA", attempt=3), np.zeros_like(play), 13)
    _daisy(ss.raw_path(sdir, 2, "ref_speech", "daisyB", attempt=3), np.zeros_like(play), 14)
    assert rs.main(["check", "--session-dir", str(sdir), "--take-id", "ref_speech", "--run-import"]) == 1
    lines = (sdir / "import_cmds.txt").read_text().splitlines()
    live = [x for x in lines if not x.startswith("#")]
    assert len(live) == 2 and all("_r2_" in x for x in live)
    assert sum(x.startswith("# QC FAIL") and "_r3_" in x for x in lines) == 2
    assert len((real / "align.jsonl").read_text().splitlines()) == 80  # nothing imported from the failure

    # the dry run: files under dryrun/, QC only, never imported
    pa, _ = audio.load(str(pool_bench / "session" / "anchor.wav"))
    pa = _take96(pa, drift_ppm=25.0)
    _daisy(ss.raw_path(sdir, 2, "dryrun", "daisyA", dryrun=True), pa, 7)
    _daisy(ss.raw_path(sdir, 2, "dryrun", "daisyB", dryrun=True), pa, 8)
    assert not (sdir / "raw" / "daisyA" / "s2_dryrun_daisyA.wav").exists()
    capsys.readouterr()
    assert rs.main(["check", "--session-dir", str(sdir), "--take-id", "dryrun"]) == 0
    assert "dry run: PASS" in capsys.readouterr().out
    assert len((sdir / "import_cmds.txt").read_text().splitlines()) == 4
    # a wrong file fails QC instead of crashing
    _daisy(ss.raw_path(sdir, 2, "dryrun", "daisyA", attempt=2, dryrun=True), np.zeros_like(pa), 9)
    _daisy(ss.raw_path(sdir, 2, "dryrun", "daisyB", attempt=2, dryrun=True), pa, 10)
    assert rs.main(["check", "--session-dir", str(sdir), "--take-id", "dryrun"]) == 1


def test_check_skips_interface_without_sync(tmp_path):
    """Bug (b): the S1 map's interface records a monitor but no loopback; check must not crash on it."""
    from test_session_run import _thruster_bench

    bench, sdir = tmp_path / "bench", tmp_path / "sess"
    bench.mkdir()
    play, _ = _thruster_bench(bench)
    rs = _script("run_session")
    assert rs.main(["plan", "--bench", str(bench), "--session", "1", "--channel-map", str(CMAP_S1),
                    "--session-dir", str(sdir), "--out", str(tmp_path / "real"), "--extra-noise"]) == 0
    iface = ss.raw_path(sdir, 1, "p1_speech", "iface")
    audio.save(str(iface), np.zeros((2, FS)), FS, subtype="PCM_24")
    da = ss.raw_path(sdir, 1, "p1_speech", "daisyA")
    hyd = _record(play, lead_s=0.5, drift_ppm=10.0)[1]
    audio.save(str(da), np.stack([hyd, hyd]) * 0.9, FS, subtype="PCM_24")
    assert rs.main(["check", "--session-dir", str(sdir), "--take-id", "p1_speech"]) == 0
    qc = json.loads((sdir / "takes.jsonl").read_text().splitlines()[-1])["qc"]
    assert qc["iface"]["note"].startswith("not QC'd") and qc["daisyA"]["ok"]


def test_dead_session_paths_resolve(tmp_path):
    """Bug (a): a session.json written under the old repos/underwater_asr root still finds the bench."""
    old = Path("C:/Users/nobody/Documents/repos/underwater_asr/data/bench/v0_s1")
    assert ss.resolve_path(old) == ROOT / "data" / "bench" / "v0_s1"
    assert ss.resolve_path("data/bench/v1") == ROOT / "data" / "bench" / "v1"
    assert ss.portable_path(ROOT / "data" / "bench" / "v1") == "data/bench/v1"
    assert Path(ss.portable_path(tmp_path)).is_absolute()


def test_tone_and_drive_agree():
    """Bug (c): the tone check and the SL estimate agree; peak drive is reported."""
    fs = 48000
    full = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)  # 0 dBFS peak sine, sample RMS -3.01 dBFS
    lv = ss.file_levels(full, fs)
    assert lv["active_rms_dBFS"] == pytest.approx(ss.SINE_RMS_DBFS, abs=0.01)
    d = ss.drive_estimate(lv["active_rms_dBFS"], vrms_at_0dbfs=2.5, amp_gain_db=0.0, peak_dbfs=lv["peak_dBFS"])
    assert d["drive_vrms"] == pytest.approx(2.5, rel=1e-3)  # the meter reading of the full-scale sine
    assert d["drive_peak_v"] == pytest.approx(2.5 * np.sqrt(2), rel=1e-3)
    # the same chain seen through a -26 dBFS RMS tone read at V_t: identical drive for any file
    tone_rms, v_t = -26.0, 2.5 * 10 ** ((-26.0 - ss.SINE_RMS_DBFS) / 20)
    for rms, peak in ((-23.0, -0.5), (-30.0, -8.0)):
        a = ss.drive_estimate(rms, 2.5, 0.0, peak_dbfs=peak)
        b = ss.tone_drive(rms, peak, v_t, tone_rms)
        assert a["drive_vrms"] == pytest.approx(b["drive_vrms"]) and a["SL_est_dB"] == pytest.approx(b["SL_est_dB"])
        assert a["drive_peak_v"] == pytest.approx(b["drive_peak_v"])
    # 160 dB needs 2.0 Vrms speech-active, 150 dB 0.63 Vrms (LL916 nominal)
    assert ss.tone_target_vrms(160, -26.0, -26.0) == pytest.approx(2.0, rel=1e-6)
    assert ss.tone_target_vrms(150, -26.0, -26.0) == pytest.approx(0.632, abs=1e-3)
    assert ss.tone_target_vrms(160, -23.0, -26.0) == pytest.approx(2.0 * 10 ** (-3 / 20), rel=1e-6)


def test_levels_cli_tone(pool_bench, tmp_path, capsys):
    sdir = tmp_path / "sess"
    rs = _plan(pool_bench, sdir)
    capsys.readouterr()
    assert rs.main(["levels", "--session-dir", str(sdir), "--tone-vrms", "2.0", "--brick-v", "5"]) == 0
    out = capsys.readouterr().out
    lv = json.loads((sdir / "levels.json").read_text())
    assert "set the 1 kHz tone to" in out and lv["tone_target_vrms"] > 0
    f = next(r for r in lv["files"] if r["file"] == "pilot.wav")
    assert f["drive_peak_v"] > f["drive_vrms"] > 0 and "crest_dB" in f
    assert any("exceeds" in w for w in lv["warnings"])  # 5 V brick: speech peaks would clip

    # desk run (no tone reading yet): the drive is predicted at the tone target, so the brick check runs
    capsys.readouterr()
    assert rs.main(["levels", "--session-dir", str(sdir), "--target-sl-db", "160", "--brick-v", "5"]) == 0
    out = capsys.readouterr().out
    lv = json.loads((sdir / "levels.json").read_text())
    assert "PREDICTED" in out and lv["drive_basis"].startswith("PREDICTED")
    assert any("exceeds" in w for w in lv["warnings"])
    assert all("drive_peak_v" in r for r in lv["files"])
    # at exactly the target tone Vrms, the speech files sit at 160 dB: no ceiling warning, even though
    # sweeps.wav (louder sine sweeps) is above it
    tv = lv["tone_target_vrms"]
    assert rs.main(["levels", "--session-dir", str(sdir), "--tone-vrms", str(tv), "--brick-v", "1000"]) == 0
    lv = json.loads((sdir / "levels.json").read_text())
    speech = [r["SL_est_dB"] for r in lv["files"] if r["file"] != "sweeps.wav"]
    assert max(speech) == pytest.approx(160, abs=0.05)
    assert not any("ceiling" in w for w in lv["warnings"]), lv["warnings"]
    assert rs.main(["levels", "--session-dir", str(sdir), "--tone-vrms", str(round(tv * 1.2, 3))]) == 0
    assert any("ceiling" in w for w in json.loads((sdir / "levels.json").read_text())["warnings"])


def test_import_subdir_routing(pool_bench, tmp_path):
    """Pilot, chain-reference and bubble cuts never enter the main (quiet) align.jsonl."""
    sdir = tmp_path / "sess"
    rs = _plan(pool_bench, sdir)
    meta = json.loads((sdir / "session.json").read_text())
    plan = {t.take_id: t for t in ss.plan_from_meta(meta)}
    out = meta["out"]
    got = {tid: rs.import_out(meta, t) for tid, t in plan.items() if t.importable}
    assert got["ref_speech"] == f"{out}/chainref" and got["pilot_speech"] == f"{out}/pilot"
    assert got["p3bub_speech"] == f"{out}/bubble"
    main = [tid for tid, o in got.items() if o == out]
    assert main and all(tid.startswith(("p1_", "p3_", "p6_")) for tid in main)
    assert all(not tid.startswith(("p1_", "p3_", "p6_")) for tid, o in got.items() if o != out)


def test_load_channels_streams_and_flags_clips(tmp_path):
    fs = 96000
    rng = np.random.default_rng(0)
    x = 0.1 * rng.standard_normal((2, 3 * fs))
    x[1, 50000] = 1.0  # full scale on channel 1 only
    p = tmp_path / "t.wav"
    audio.save(str(p), x, fs, subtype="PCM_24")
    y, fs_raw, bits, n_ch, n_raw, clips = ingest.load_channels(p, [1, 0], 48000, block=4096)
    assert y.dtype == np.float32 and y.shape == (2, 3 * 48000) and (fs_raw, bits, n_ch, n_raw) == (96000, 24, 2, 3 * fs)
    ref = audio.resample(audio.load(str(p), always_2d=True)[0], fs, 48000, axis=-1)
    assert np.allclose(y[0], ref[1], atol=1e-5) and np.allclose(y[1], ref[0], atol=1e-5)
    assert list(clips[1]) == [50000] and clips[0].size == 0
    with pytest.raises(ValueError):
        ingest.load_channels(p, [2], 48000)


def test_record_daisy_refuses_overwrite(tmp_path):
    rd = _script("record_daisy")
    p = tmp_path / "x.wav"
    rd.refuse_overwrite(p)
    p.write_bytes(b"")
    with pytest.raises(SystemExit, match="attempt 2"):
        rd.refuse_overwrite(p)
    q = tmp_path / "y.wav"
    q.with_suffix(".wav.json").write_text("{}")
    with pytest.raises(SystemExit):
        rd.refuse_overwrite(q)


def test_take_label_and_latest_attempt(tmp_path):
    assert ss.take_label("p3_speech") == "p3_speech" and ss.take_label("p3_speech", 3) == "p3_speech_r3"
    t = ss.Take("p3_speech", "speech", "p3", "speech.wav", 10.0)
    assert ss.latest_attempt(tmp_path, 2, t, ["daisyA"]) == 1
    for k in (1, 2, 4):
        p = ss.raw_path(tmp_path, 2, t, "daisyA", k)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    assert ss.latest_attempt(tmp_path, 2, t, ["daisyA", "daisyB"]) == 4
    with pytest.raises(ValueError):
        ss.take_label("x", 0)
    assert ss.pos_label(0.5) == "p0p5" and ss.pos_label(3) == "p3" and ss.pos_label(4.5) == "p4p5"


def test_bench_sha256sums(pool_bench, tmp_path):
    """The pool build writes a reproducible SHA256SUMS.txt (the bench-freeze step of the checklist)."""
    import hashlib

    p = pool_bench / "SHA256SUMS.txt"
    data = p.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf") and b"\r" not in data
    lines = data.decode("utf-8").splitlines()
    paths = [x.split("  ", 1)[1] for x in lines]
    assert paths == sorted(paths) and "SHA256SUMS.txt" not in paths
    assert "session/programs.json" in paths and "schedule.json" in paths
    want = hashlib.sha256((pool_bench / "session" / "programs.json").read_bytes()).hexdigest()
    assert f"{want}  session/programs.json" in lines
    _, digest = ss.write_sha256sums(pool_bench)  # rewriting gives the same bytes
    assert p.read_bytes() == data and digest == hashlib.sha256(data).hexdigest()
    ma = _script("make_session_audio")
    with pytest.raises(SystemExit, match="frozen"):
        ma.main(["--bench", str(ROOT / "data" / "bench" / "v0_s1"), "--sums-only"])
