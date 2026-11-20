"""scripts/score_session.py -> scripts/make_paper_numbers.py on a synthetic session (fake ASR)."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from uwse import audio

ROOT = Path(__file__).resolve().parents[1]
FS = 16000


def _script(name):
    spec = importlib.util.spec_from_file_location(f"t_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SS = _script("score_session")
MP = _script("make_paper_numbers")


class FakeASR:
    """C8 (LoRA) is always right; the rest miss the last word of 'a b c d'."""

    def __init__(self, model_name, device=None, decode=None, lora_ckpt=None):
        self.name, self.lora_ckpt = "fake", lora_ckpt

    def transcribe(self, x):
        return {"text": "a b c d" if self.lora_ckpt else "a b c", "compression_ratio": 1.0}

    def info(self):
        return {"backend": "fake"}


def _rows(tmp_path, tag, rng, positions=("p1", "p3")):
    rows = []
    for pos in positions:
        for i in range(6):
            spk = f"{200 + i % 3}"
            uid = f"v1_s1_{pos}_b00_ch0_{spk}-1-{i:04d}"
            x = sosfilt(butter(4, (300, 4000), "band", fs=FS, output="sos"), rng.standard_normal(FS))
            r0 = np.zeros(2 * FS)
            r0[FS // 2: FS // 2 + FS] = 0.1 * x * np.hanning(FS)
            wp, rp = tmp_path / f"{tag}_{uid}.wav", tmp_path / f"{tag}_{uid}_r0.wav"
            audio.save(str(wp), 0.3 * r0 + 1e-3 * rng.standard_normal(r0.size), FS, subtype="FLOAT")
            audio.save(str(rp), r0, FS, subtype="FLOAT")
            rows.append({"utt_id": uid, "system": "C0", "wav": str(wp), "r0_wav": str(rp),
                         "transcript": "a b c d", "speaker": spk, "position": pos, "thruster": "none",
                         "kind": "speech"})
    return rows


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def session(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    monkeypatch.setattr(SS.SCORE.A, "WhisperASR", FakeASR)
    quiet = _rows(tmp_path, "q", rng)
    qm = _write(tmp_path / "quiet.jsonl", quiet)
    _write(tmp_path / "added" / "pool0dB" / "manifest.jsonl", _rows(tmp_path, "n", rng))
    bub = _write(tmp_path / "bub.jsonl", _rows(tmp_path, "b", rng, positions=("p3bub",)))
    se = quiet + [{**r, "system": "C2"} for r in quiet]
    sem = _write(tmp_path / "se.jsonl", se)
    ckpt = tmp_path / "c8.pt"
    ckpt.write_bytes(b"frozen")
    sha = hashlib.sha256(b"frozen").hexdigest()
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"role": "pilot", "wer": 0.04, "wer_lo": 0.02, "wer_hi": 0.07, "n_utts": 20,
                                 "n_speakers": 10, "threshold": 0.10, "decision": "ADDED-NOISE primary",
                                 "positions": ["p3"]}), encoding="utf-8")
    chain = tmp_path / "chain.json"
    chain.write_text(json.dumps({"role": "chainref", "wer": 0.021, "wer_lo": 0.01, "wer_hi": 0.03,
                                 "n_utts": 20, "n_speakers": 10, "positions": ["p0p5"]}), encoding="utf-8")
    out = tmp_path / "results"
    argv = ["--out", str(out), "--condition", f"quiet={qm}", "--condition", f"bubblelive={bub}", "--added-noise-dir", str(tmp_path / "added"),
            "--se-manifest", f"quiet={sem}", "--c8-ckpt", str(ckpt), "--c8-sha256-prefix", sha[:8],
            "--n-boot-ci", "40", "--n-boot-paired", "30", "--pilot", str(pilot), "--chainref", str(chain),
            "--no-metrics", "--device", "cpu"]
    assert SS.main(argv) == 0
    return out, argv, ckpt


def test_score_session_outputs(session):
    out, argv, _ = session
    idx = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert idx["positions"] == ["p1", "p3"] and idx["distances_m"] == {"p1": 1.0, "p3": 3.0, "p3bub": 3.0}
    assert idx["conditions"]["bubblelive"]["positions"] == ["p3bub"] and idx["main_condition"] == "quiet"
    assert idx["conditions"]["quiet"]["systems"] == ["C0", "C2", "C8"]
    assert idx["conditions"]["pool0dB"]["added_noise"] and not idx["conditions"]["quiet"]["added_noise"]
    for cond in ("quiet", "pool0dB"):
        for sub in ("all", "paired_ALL", "paired_p1", "paired_p3"):
            assert (out / cond / sub / "summary.csv").exists()
        assert (out / cond / "paired_p3" / "paired.csv").exists()
        info = json.loads((out / cond / "paired_p1" / "run_info.json").read_text(encoding="utf-8"))
        assert info["args"]["n_boot"] == 30 and info["args"]["level_normalize"]
    assert (out / "pilot.json").exists() and (out / "chainref.json").exists()


def test_score_session_refuses_wrong_ckpt(session):
    out, argv, ckpt = session
    bad = list(argv)
    bad[bad.index("--c8-sha256-prefix") + 1] = "00000000"
    with pytest.raises(SystemExit):
        SS.main(bad)


def test_macros(session, tmp_path):
    out, _, _ = session
    tex, fig = tmp_path / "paper" / "results_real.tex", tmp_path / "paper" / "fig_real.pdf"
    assert MP.main(["--results", str(out), "--tex", str(tex), "--fig", str(fig), "--no-fig"]) == 0
    assert not fig.exists()
    text = tex.read_text(encoding="utf-8")
    defs = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", text))
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in defs)
    for k in defs:  # every macro is documented in the header
        assert f"%   \\{k} " in text
    assert defs["WerCZeroQuietThree"] == "25.0" and defs["WerCEightQuietThree"] == "0.0"
    assert defs["WerCTwoQuietOne"] == "25.0"
    assert defs["DWerCEightQuietThree"] == r"\ensuremath{-25.0}"
    assert defs["RelDWerCEightQuietAll"] == r"\ensuremath{-100}"
    assert defs["DWerCEightPoolZeroOneCI"].startswith("[")
    assert defs["PrimCondName"] == "pool0dB" and defs["SecCondName"] == "quiet"
    assert defs["PrimWerCZeroThree"] == defs["WerCZeroPoolZeroThree"]
    assert defs["PrimDWerAll"] == defs["DWerCEightPoolZeroAll"]
    assert defs["SecDWerOne"] == defs["DWerCEightQuietOne"]
    assert defs["PilotWer"] == "4.0" and defs["PilotDecision"] == "added noise primary"
    assert defs["PilotDist"] == "3" and defs["ChainRefDist"] == "0.5" and defs["ChainRefWer"] == "2.1"
    assert defs["NUttsQuietThree"] == "6" and defs["NSpkQuietAll"] == "3" and defs["NUttsPerDist"] == "6"
    assert defs["WerCEightBubbleliveThree"] == "0.0" and "WerCEightBubbleliveOne" not in defs
    assert defs["DistList"] == "1 and 3" and defs["NDist"] == "2"
    assert defs["NBootCI"] == "40" and defs["NBootPaired"] == "30" and len(defs["CEightSha"]) == 8


def test_naming_and_missing_pilot(session, tmp_path):
    assert MP.numword(3.0) == "Three" and MP.numword(0.5) == "ZeroPtFive" and MP.numword("25") == "TwoFive"
    assert MP.camel("pool0dB") == "PoolZero" and MP.camel("C10") == "CTen" and MP.camel("bubble-5dB") == "BubbleMinusFive"
    out, _, _ = session
    nums = MP.collect(out)
    nums["pilot"] = None
    defs = {n: v for n, v, _ in MP.build_macros(nums)}
    assert defs["PrimCond"] == MP.MISSING and defs["PrimWerCZeroThree"] == MP.MISSING
    assert defs["PilotWer"] == MP.MISSING


def test_figure(tmp_path):
    pytest.importorskip("matplotlib")
    pos = ["p1", "p3", "p6"]
    wer = {}
    for k, s in enumerate(("C0", "C2", "C8")):
        for j, p in enumerate(pos):
            w = 0.03 * (j + 1) * (1.0 - 0.2 * (s == "C8"))
            wer[(s, p)] = {"wer": w, "lo": 0.8 * w, "hi": 1.2 * w, "n": 200}
    nums = {"positions": pos, "distances": {"p1": 1.0, "p3": 3.0, "p6": 6.0}, "chainref": {"wer": 0.02},
            "conds": {"quiet": {"wer": wer}, "pool0dB": {"wer": {k: {**v, "wer": 3 * v["wer"], "hi": 4 * v["wer"]}
                                                               for k, v in wer.items()}}}}
    fig = MP.make_figure(nums, tmp_path / "fig_real.pdf")
    assert fig.exists() and fig.stat().st_size > 1000
