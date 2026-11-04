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
