"""Take import on synthetic recordings: offset, drift, missed chirps, cuts, resampled takes."""
from __future__ import annotations

import json

import numpy as np
import pytest

from uwse import audio
from uwse.bench import ingest
from uwse.bench import playlist as pl
from uwse.evaluation import align as _align

FS = 48000


def _bench(tmp_path, n_blocks=3, n_utts=2, seed=0):
    """A tiny bench dir: blocks, schedule.json and session/p1_speech.json (+ anchor)."""
    rng = np.random.default_rng(seed)
    (tmp_path / "blocks").mkdir()
    (tmp_path / "session").mkdir()
    sched, index, parts, n = [], [], [], 0
    gap = np.zeros(FS)
    utts = {}
    for k in range(n_blocks):
        items = []
        for j in range(n_utts):
            m = int(FS // 2 * (0.6 + 0.4 * rng.random()))  # 16 kHz source, 0.3-0.5 s at 48 kHz... rendered
            x = rng.standard_normal(m) * np.hanning(m)
            items.append((f"{1000 + k}-{j}-0000", x, 16000))
        sig, t = pl.render_block(items, FS)
        cb = f"v0-test_{k:03d}"
        t["content_block"], t["bench_split"] = cb, "v0-test"
        for u in t["utterances"]:
            u["transcript"] = "X"
            utts[(cb, u["item_id"])] = sig[u["start"]:u["end"]]
        (tmp_path / "blocks" / f"{cb}.json").write_text(json.dumps(t))
        sched.append({"session_order_idx": k, "block_id": k, "content_block": cb, "position_id": "p1",
                      "thruster_state": "off", "block_order": "ABBA", "anchor_repeat": False})
        index.append({"session_order_idx": k, "content_block": cb, "thruster_state": "off",
                      "start_sample": n, "n_samples": len(sig)})
        parts += [sig, gap]
        n += len(sig) + len(gap)
    sched.append({**sched[0], "session_order_idx": n_blocks, "block_id": n_blocks, "anchor_repeat": True})
    (tmp_path / "schedule.json").write_text(json.dumps(sched))
    (tmp_path / "session" / "p1_speech.json").write_text(
        json.dumps({"position_id": "p1", "fs": FS, "gap_s": 1.0, "blocks": index}))
    return np.concatenate(parts), utts


def _record(play, lead_s, drift_ppm, delay_s=0.004, gain=0.3, snr_db=15.0, tail_s=1.0, seed=1):
    """Hydrophone take: lead silence, fast/slow ADC clock, delay, decaying echo, noise.

    Row 0 is the loopback (no acoustic path), row 1 the hydrophone.
    """
    rng = np.random.default_rng(seed)
    r = 1.0 + drift_ppm * 1e-6
    n = int((lead_s + len(play) / FS + tail_s) * FS * r)
    t_play = (np.arange(n) / r) - lead_s * FS  # playback sample at each take sample
    loop = _align.fractional_resample(play, t_play)
    hyd = _align.fractional_resample(play, t_play - delay_s * FS)
    hyd = gain * (hyd + 0.4 * np.concatenate([np.zeros(int(0.003 * FS)), hyd])[:n])
    sp = np.sqrt(np.mean(hyd[hyd != 0] ** 2))
    hyd = hyd + rng.standard_normal(n) * sp * 10 ** (-snr_db / 20)
    loop = 0.5 * loop + rng.standard_normal(n) * 1e-4
    return np.stack([loop, hyd])


def _check_cuts(streams, results, utts, pad_s=0.25, row=0, delay_s=0.0, tol=2):
    for res in results:
        cuts, bounds = ingest.cut_block(streams[row:row + 1], FS, res, pad_s)
        for k, u in enumerate(res.block.timing["utterances"]):
            ref = utts[(res.block.content_block, u["item_id"])]
            y = cuts[0][k]
            pad = int(pad_s * FS)
            assert len(y) == len(ref) + 2 * pad
            lag = _align.fine_lag(y[pad - 2000: pad - 2000 + len(ref) + 4000], ref, FS, max_lag_s=0.05)
            assert abs(lag.lag - (2000 + delay_s * FS)) <= tol, (res.block.content_block, k, lag.lag)
            assert bounds[k, 1] - bounds[k, 0] > len(ref)


@pytest.mark.parametrize("drift_ppm", [0.0, 80.0, -150.0])
def test_offset_and_drift_recovered_on_loopback(tmp_path, drift_ppm):
    play, utts = _bench(tmp_path)
    fs, blocks = ingest.load_session(tmp_path, "p1")
    x = _record(play, lead_s=3.217, drift_ppm=drift_ppm)
    results, qc = ingest.align_take(x, fs, blocks, sync=0)
    assert qc["n_chirps_good"] == qc["n_chirps"] == 6
    assert abs(qc["clock_offset"] - 3.217 * FS * (1 + drift_ppm * 1e-6)) < 2.0
    assert abs(qc["take_drift_ppm"] - drift_ppm) < 1.0
    for r in results:
        assert r.drift_source == "block" and not r.flags
        assert abs(r.drift_ppm - drift_ppm) < 5.0
    _check_cuts(x, results, utts, row=0, tol=2)


def test_hydrophone_only_sync_with_noise_and_delay(tmp_path):
    play, utts = _bench(tmp_path)
    fs, blocks = ingest.load_session(tmp_path, "p1")
    x = _record(play, lead_s=1.5, drift_ppm=60.0, delay_s=0.004, snr_db=10.0)
    results, qc = ingest.align_take(x, fs, blocks, sync=1)
    assert qc["n_chirps_good"] == 6
    assert abs(qc["take_drift_ppm"] - 60.0) < 5.0
    # the hydrophone chirps carry the acoustic delay, so the cuts line up with the played audio
    _check_cuts(x, results, utts, row=1, delay_s=0.0, tol=3)


def test_loopback_sync_cuts_hydrophone_with_acoustic_delay(tmp_path):
    play, utts = _bench(tmp_path)
    fs, blocks = ingest.load_session(tmp_path, "p1")
    x = _record(play, lead_s=2.0, drift_ppm=-40.0, delay_s=0.004)
    results, _ = ingest.align_take(x, fs, blocks, sync=0)
    _check_cuts(x, results, utts, row=1, delay_s=0.004, tol=3)


def test_missing_chirp_is_flagged_and_cut_from_clock_fit(tmp_path):
    play, utts = _bench(tmp_path)
    fs, blocks = ingest.load_session(tmp_path, "p1")
    x = _record(play, lead_s=2.5, drift_ppm=100.0)
    # wipe block 1's end chirp on the loopback
    r = 1.0 + 100e-6
    s = int((2.5 * FS + blocks[1].chirp_starts[1]) * r)
    x[0, s - 100: s + int(0.3 * FS)] = 0.0
    results, qc = ingest.align_take(x, fs, blocks, sync=0)
    assert qc["n_chirps_good"] == 5
    assert results[1].flags == ["align_low_psr"] and results[1].drift_source == "take_fit"
    assert not results[0].flags and not results[2].flags
    _check_cuts(x, results, utts, row=0, tol=2)


def test_take_at_other_rate_is_resampled(tmp_path):
    play, utts = _bench(tmp_path, n_blocks=2)
    fs, blocks = ingest.load_session(tmp_path, "p1")
    x = _record(play, lead_s=1.0, drift_ppm=30.0)
    path = tmp_path / "take44.wav"
    audio.save(str(path), audio.resample(x, FS, 44100, axis=-1) * 0.9, 44100, subtype="FLOAT")
    y, fs_raw, bits = ingest.load_take(path, FS)
    assert fs_raw == 44100 and bits == 32 and y.shape[0] == 2
    results, qc = ingest.align_take(y, fs, blocks, sync=0)
    assert qc["n_chirps_good"] == 4 and abs(qc["take_drift_ppm"] - 30.0) < 3.0
    _check_cuts(y, results, utts, row=0, tol=2)


def test_anchor_session_has_one_block(tmp_path):
    _bench(tmp_path)
    fs, blocks = ingest.load_session(tmp_path, "p1", anchor=True)
    assert len(blocks) == 1 and blocks[0].anchor_repeat and blocks[0].start_sample == 0
    assert blocks[0].content_block == "v0-test_000"


def test_cli_writes_cuts_and_schema_ids(tmp_path):
    import importlib.util
    from pathlib import Path

    from uwse.bench.schema import parse_utt_id

    play, _ = _bench(tmp_path)
    x = _record(play, lead_s=1.0, drift_ppm=20.0)
    rec = tmp_path / "take01.wav"
    audio.save(str(rec), x * 0.9, FS, subtype="PCM_24")
    spec = importlib.util.spec_from_file_location(
        "import_recording", Path(__file__).resolve().parents[1] / "scripts" / "import_recording.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    assert mod.main(["--bench", str(tmp_path), "--position", "p1", "--rec", str(rec), "--session", "2",
                     "--channels", "1", "--sync-channel", "0", "--out", str(out)]) == 0
    rows = [json.loads(l) for l in (out / "align.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    for r in rows:
        ids = parse_utt_id(r["utt_id"])
        assert ids["session"] == 2 and ids["channel"] == 1 and ids["position_id"] == "p1"
        assert ids["block_id"] == r["block_id"] and ids["item_id"] == r["librispeech_id"]
        assert r["flags"] == [] and abs(r["drift_ppm"] - 20.0) < 5.0
        assert (out / "wav" / f"{r['utt_id']}.wav").is_file()
        assert 0 <= r["start_sample_raw"] < r["end_sample_raw"] <= x.shape[1]
    qc = json.loads((out / "blocks_take01.json").read_text())
    assert qc["n_chirps_good"] == 6
