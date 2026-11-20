"""scripts/pilot_wer.py: C0 pilot WER and the pre-registered quiet / added-noise decision."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from uwse import audio

ROOT = Path(__file__).resolve().parents[1]
FS = 48000


def _script(name):
    spec = importlib.util.spec_from_file_location(f"t_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PW = _script("pilot_wer")


def speechlike(rng, n, fs=FS):
    x = sosfilt(butter(4, (300, 4000), "band", fs=fs, output="sos"), rng.standard_normal(n))
    env = np.clip(np.sin(2 * np.pi * 4 * np.arange(n) / fs), 0, None) ** 2
    return 0.1 * x * env / np.sqrt(np.mean((x * env) ** 2))


def _fake(hyp, made):
    class FakeASR:
        def __init__(self, model_name, device=None, decode=None, lora_ckpt=None):
            self.name = "fake"
            made.append({"decode": decode, "lora": lora_ckpt, "device": device})

        def transcribe(self, x):
            return {"text": hyp, "compression_ratio": 1.0}

        def info(self):
            return {"backend": "fake"}

    return FakeASR


def _setup(tmp_path, n=6):
    rng = np.random.default_rng(0)
    man, align = [], []
    for i in range(n):
        spk = f"{100 + i % 3}"
        uid = f"v1_s1_p3_b00_ch0_{spk}-1-{i:04d}"
        r0 = np.zeros(2 * FS)
        r0[FS // 4: FS // 4 + FS] = speechlike(rng, FS)
        y = 0.3 * r0 + 1e-3 * rng.standard_normal(r0.size)
        wp, rp = tmp_path / f"{uid}.wav", tmp_path / f"{uid}_r0.wav"
        audio.save(str(wp), y, FS, subtype="FLOAT")
        audio.save(str(rp), r0, FS, subtype="FLOAT")
        man.append({"utt_id": uid, "system": "C0", "wav": str(wp), "r0_wav": str(rp), "transcript": "a b c d",
                    "speaker": spk, "position": "p3", "thruster": "none", "kind": "speech"})
        man.append({**man[-1], "system": "C8"})
        align.append({"utt_id": uid, "librispeech_id": f"{spk}-1-{i:04d}", "transcript": "a b c d",
                      "channel": 0, "anchor_repeat": False, "position_id": "p3", "thruster_state": "none",
                      "wav": str(wp), "flags": []})
        align.append({**align[-1], "channel": 1, "utt_id": uid.replace("ch0", "ch1")})
    (tmp_path / "m.jsonl").write_text("\n".join(json.dumps(r) for r in man) + "\n", encoding="utf-8")
    (tmp_path / "align.jsonl").write_text("\n".join(json.dumps(r) for r in align) + "\n", encoding="utf-8")
    return tmp_path / "m.jsonl", tmp_path / "align.jsonl"


def test_decide_boundary():
    assert PW.decide(0.10) == "QUIET primary"
    assert PW.decide(0.0999) == "ADDED-NOISE primary"
    assert PW.decide(0.5) == "QUIET primary"


@pytest.mark.parametrize("hyp,wer,decision", [("a b c d", 0.0, "ADDED-NOISE primary"),
                                              ("a b c x", 0.25, "QUIET primary")])
def test_pilot_manifest(tmp_path, monkeypatch, capsys, hyp, wer, decision):
    man, _ = _setup(tmp_path)
    made = []
    monkeypatch.setattr(PW.SCORE.A, "WhisperASR", _fake(hyp, made))
    out = tmp_path / "pilot.json"
    assert PW.main(["--manifest", str(man), "--out", str(out), "--device", "cpu", "--n-boot", "50"]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["wer"] == pytest.approx(wer)
    assert rec["decision"] == decision and decision in capsys.readouterr().out
    assert rec["n_utts"] == 6 and rec["n_speakers"] == 3  # C0 rows only (the C8 rows are ignored)
    assert rec["level_normalize"] and rec["asr_key"] == "whisper-large-v3+lnorm+guard"
    assert rec["timestamp"] and rec["threshold"] == 0.10 and rec["threshold_preregistered"]
    assert len(rec["input_sha256"]) == 64 and len(rec["per_utt"]) == 6
    assert made == [{"decode": PW.SCORE.A.DecodeConfig.guarded(), "lora": None, "device": "cpu"}]
    assert (tmp_path / "pilot_transcripts.jsonl").exists()
    with pytest.raises(SystemExit):  # the decision is written once
        PW.main(["--manifest", str(man), "--out", str(out)])


def test_align_input_and_chainref(tmp_path, monkeypatch):
    _, align = _setup(tmp_path)
    monkeypatch.setattr(PW.SCORE.A, "WhisperASR", _fake("a b", []))
    out = tmp_path / "chain.json"
    assert PW.main(["--align", str(align), "--out", str(out), "--role", "chainref", "--n-boot", "20"]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["role"] == "chainref" and "decision" not in rec
    assert rec["n_utts"] == 6 and not rec["level_normalize"] and rec["asr_key"] == "whisper-large-v3+guard"
    assert rec["wer"] == pytest.approx(0.5) and rec["positions"] == ["p3"]
