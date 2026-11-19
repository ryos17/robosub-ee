"""scripts/make_score_manifest.py: aligned R0 cuts and the score.py manifest."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from uwse import audio
from uwse.bench.schema import make_utt_id

ROOT = Path(__file__).resolve().parents[1]
FS = 48000


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path, delay=30):
    rng = np.random.default_rng(0)
    items = ["1089-134686-0000", "121-121726-0001"]
    block = np.zeros(6 * FS)
    utts = []
    for k, it in enumerate(items):
        s = FS + k * 2 * FS
        block[s:s + FS] = 0.1 * rng.standard_normal(FS)
        utts.append({"item_id": it, "start": s, "end": s + FS, "transcript": "a b"})
    (tmp_path / "bench" / "blocks").mkdir(parents=True)
    audio.save(str(tmp_path / "bench" / "blocks" / "v0-test_000.wav"), block, FS, subtype="FLOAT")
    (tmp_path / "bench" / "blocks" / "v0-test_000.json").write_text(json.dumps({"fs": FS, "utterances": utts}))
    (tmp_path / "wav").mkdir()
    rows = []
    for anchor, bid in ((False, 0), (True, 1)):
        for ch in (0, 1):
            for u in utts:
                uid = make_utt_id("v0", 2, "p3", bid, ch, u["item_id"])
                a = u["start"] - int(0.25 * FS)
                cut = np.zeros(int(1.5 * FS))
                cut[delay:] = 0.5 * block[a:a + cut.size - delay]
                cut += 1e-3 * rng.standard_normal(cut.size)
                p = tmp_path / "wav" / f"{uid}.wav"
                audio.save(str(p), cut, FS, subtype="FLOAT")
                rows.append({"utt_id": uid, "librispeech_id": u["item_id"], "transcript": "a b",
                             "content_block": "v0-test_000", "position_id": "p3", "thruster_state": "on",
                             "anchor_repeat": anchor, "channel": ch, "fs": FS, "pad_s": 0.25, "wav": str(p),
                             "flags": []})
    (tmp_path / "align.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    c0 = [r for r in rows if r["channel"] == 0 and not r["anchor_repeat"]]
    (tmp_path / "snr.jsonl").write_text(json.dumps({"utt_id": c0[0]["utt_id"], "acoustic": {}, "snr_db_500_8000": 2.0,
                                                    "snr_db_50_8000": 5.0, "flags": ["snr_level_drift"]}) + "\n")
    (tmp_path / "SPEAKERS.TXT").write_text(";ID|SEX\n1089 | M | test\n121 | F | test\n")
    (tmp_path / "se").mkdir()
    for r in c0:
        audio.save(str(tmp_path / "se" / f"{r['utt_id']}.wav"), np.zeros(10), FS, subtype="FLOAT")
    return c0


def test_r0_lag_and_manifest(tmp_path):
    msm = _script("make_score_manifest")
    c0 = _setup(tmp_path, delay=30)
    out = tmp_path / "out"
    assert msm.main(["--align", str(tmp_path / "align.jsonl"), "--bench", str(tmp_path / "bench"), "--out", str(out),
                     "--snr", str(tmp_path / "snr.jsonl"), "--speaker-sex", str(tmp_path / "SPEAKERS.TXT"),
                     "--c8", "--se", f"C2={tmp_path / 'se'}"]) == 0
    lags = [json.loads(l) for l in (out / "r0_lags.jsonl").read_text().splitlines()]
    assert len(lags) == 2 and all(x["lag_samples"] == pytest.approx(30, abs=0.2) and not x["low_psr"] for x in lags)
    # the written R0 lines up with the cut
    for r in c0:
        y, _ = audio.load(r["wav"])
        r0, _ = audio.load(str(out / "r0" / f"{r['utt_id']}.wav"))
        assert r0.size == y.size and np.corrcoef(y, r0)[0, 1] > 0.99
    man = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(man) == 6 and {m["system"] for m in man} == {"C0", "C8", "C2"}  # channel 0, anchors dropped
    by = {(m["system"], m["utt_id"]): m for m in man}
    m0 = by[("C0", c0[0]["utt_id"])]
    assert m0["wav"] == by[("C8", c0[0]["utt_id"])]["wav"] == c0[0]["wav"]
    assert m0["sex"] == "M" and by[("C0", c0[1]["utt_id"])]["sex"] == "F" and m0["speaker"] == "1089"
    assert m0["snr_db_50_8000"] == 5.0 and m0["snr_level_drift"] and "snr_db_500_8000" not in by[("C0", c0[1]["utt_id"])]
    assert "ref_wav" not in m0 and m0["r0_wav"].endswith(f"{c0[0]['utt_id']}.wav")
    # score.py accepts it
    score = _script("score")
    assert len(score.read_manifest(str(out / "manifest.jsonl"))) == 6


def test_missing_se_cut_and_bad_names(tmp_path):
    msm = _script("make_score_manifest")
    c0 = _setup(tmp_path)
    (tmp_path / "se" / f"{c0[1]['utt_id']}.wav").unlink()
    base = ["--align", str(tmp_path / "align.jsonl"), "--bench", str(tmp_path / "bench"), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="C2: missing"):
        msm.main(base + ["--se", f"C2={tmp_path / 'se'}"])
    with pytest.raises(SystemExit, match="reserved"):
        msm.main(base + ["--se", f"C8={tmp_path / 'se'}"])
