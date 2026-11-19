"""Guarded Whisper decoding: fallback temperatures, length cap and seeding reach transcribe()."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from uwse.evaluation.asr import ASR_FS, DecodeConfig, WhisperASR


class _FakeWhisper:
    dims = SimpleNamespace(n_text_ctx=448)

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return {"text": " hello world", "segments": [
            {"avg_logprob": -0.2, "no_speech_prob": 0.01, "temperature": 0.4}]}


def _asr(decode):
    asr = WhisperASR("tiny", device="cpu", fp16=False, decode=decode)
    asr._model = _FakeWhisper()
    return asr


def test_fixed_policy_unchanged():
    asr = _asr(DecodeConfig())
    res = asr.transcribe(np.zeros(5 * ASR_FS, np.float32))
    kw = asr._model.calls[0]
    assert kw["temperature"] == 0.0 and kw["beam_size"] == 5
    assert "sample_len" not in kw and "best_of" not in kw
    assert "max_temperature" not in res  # fixed-policy cache records keep their schema


def test_guarded_policy_caps_length_and_falls_back():
    asr = _asr(DecodeConfig.guarded())
    res = asr.transcribe(np.zeros(int(9.5 * ASR_FS), np.float32))
    kw = asr._model.calls[0]
    assert kw["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert kw["best_of"] == 5 and kw["compression_ratio_threshold"] == 2.4
    assert kw["sample_len"] == 76 + 16  # ceil(8 * 9.5) + margin
    assert res["sample_len"] == 92 and res["max_temperature"] == 0.4


def test_guarded_cap_never_exceeds_model_limit():
    asr = _asr(DecodeConfig.guarded())
    asr.transcribe(np.zeros(60 * ASR_FS, np.float32))
    assert asr._model.calls[0]["sample_len"] == 224


def test_guarded_sampling_is_seeded():
    import torch

    asr = _asr(DecodeConfig.guarded())
    asr.transcribe(np.zeros(ASR_FS, np.float32))
    a = torch.rand(1)
    asr.transcribe(np.zeros(ASR_FS, np.float32))
    assert torch.equal(a, torch.rand(1))


def test_on_device_load_matches_load_model():
    """The low-RAM loader gives the same weights and buffers as whisper.load_model (tiny, CPU)."""
    import os

    import pytest
    import torch

    whisper = pytest.importorskip("whisper")
    from uwse.evaluation.asr import _load_whisper_on_device

    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "whisper", "tiny.pt")):
        pytest.skip("whisper tiny checkpoint not cached")
    a = whisper.load_model("tiny", device="cpu")
    b = _load_whisper_on_device("tiny", "cpu")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]) for k in sa)
    assert torch.equal(a.alignment_heads.to_dense(), b.alignment_heads.to_dense())
    assert torch.equal(a.decoder.mask, b.decoder.mask)
