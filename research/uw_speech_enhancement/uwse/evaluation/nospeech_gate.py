"""No-speech gate for Whisper output on hydrophone audio (system C8g).

Whisper's guarded decode still invents words on noise-only input: on the simulated non-speech
set (``scripts/make_nonspeech_set.py``) C8 writes text for 41.9% of segments. This gate removes
that output using Whisper's own no-speech probability:

1. Split the 16 kHz input into non-overlapping 30 s windows (Whisper's context length).
2. For each window, run the encoder on that window alone and read the softmax probability of
   ``<|nospeech|>`` at the start-of-transcript position, before any suppression (``p_ns``).
3. Drop every decoded segment whose start time falls in a window with ``p_ns >= TAU``.
   A clip of 30 s or less is therefore emptied when its one window fires.

``TAU = 0.05`` was chosen on the simulated dev data only (non-speech dev set plus
data/dev_wer_v0, two-fold out of fold) and then scored once on held-out simulated test data:
0/360 non-speech segments with output (C8: 151/360), and 0 of 1800 held-out speech clips gated
(0 of 480 on dev), so WER is unchanged (FINDINGS.md, "No-speech gate"). It has not been tested
on underwater recordings; the S1 pool session is its first test (RESULTS_E7_SIM_DRYRUN.md).
"""

from __future__ import annotations

import numpy as np

FS = 16000
WIN_S = 30
WIN = WIN_S * FS
TAU = 0.05


def split_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Non-overlapping 30 s windows; the last one may be shorter (the mel front end pads it)."""
    n = max(1, int(np.ceil(len(audio) / WIN)))
    return [audio[k * WIN:(k + 1) * WIN] for k in range(n)]


def first_window_p_ns(asr, audio: np.ndarray) -> float:
    """Pre-suppression ``<|nospeech|>`` probability at the SOT position of the first 30 s window.

    ``asr`` is a :class:`uwse.evaluation.asr.WhisperASR`; ``audio`` is the same 16 kHz array its
    ``transcribe`` gets (level normalization already applied)."""
    import torch
    import whisper
    from whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram
    from whisper.tokenizer import get_tokenizer

    m = asr.model
    tok = get_tokenizer(m.is_multilingual, num_languages=m.num_languages, language="en", task="transcribe")
    mel = log_mel_spectrogram(np.ascontiguousarray(audio, np.float32), m.dims.n_mels, padding=N_SAMPLES)
    seg = whisper.audio.pad_or_trim(mel[:, :N_FRAMES], N_FRAMES).to(m.device)
    if asr.fp16:
        seg = seg.half()
    with torch.no_grad():
        feats = m.embed_audio(seg[None])
        logits = m.logits(torch.tensor([[tok.sot]], device=m.device), feats)
        p = torch.softmax(logits[0, 0].float(), dim=-1)[tok.no_speech]
    return float(p)


def window_p_ns(asr, audio: np.ndarray, fn=None) -> list[float]:
    """``p_ns`` of every window, each computed from that window's audio alone."""
    fn = fn or first_window_p_ns
    return [float(fn(asr, w)) for w in split_windows(audio)]


def seg_window(start_s: float) -> int:
    """Index of the 30 s window a segment starting at ``start_s`` belongs to."""
    return int(np.floor(start_s * FS / WIN + 1e-9))


def gate_text(text: str, segments: list[dict], p: list[float], tau: float = TAU) -> tuple[str, list[bool]]:
    """Gated transcript: drop segments that start in a window with ``p_ns >= tau``.

    Returns (text, per-segment dropped flags). The text is returned unchanged when nothing is
    dropped; a clip with no segments is emptied when its first window fires."""
    if not segments:
        return ("" if p and p[0] >= tau else text), []
    dropped = [bool(p[min(seg_window(s["start"]), len(p) - 1)] >= tau) for s in segments]
    if not any(dropped):
        return text, dropped
    return " ".join(s["text"].strip() for s, d in zip(segments, dropped) if not d).strip(), dropped


def decode_with_segments(asr, x: np.ndarray) -> dict:
    """``asr.transcribe(x)`` unchanged, plus the long-form segments captured from the model call."""
    cap = {}
    orig = asr.model.transcribe

    def wrapped(*a, **k):
        o = orig(*a, **k)
        cap["segs"] = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]}
                       for s in o.get("segments", [])]
        return o

    asr.model.transcribe = wrapped
    try:
        res = asr.transcribe(x)
    finally:
        asr.model.transcribe = orig
    return {**res, "segments": cap.get("segs", [])}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))
