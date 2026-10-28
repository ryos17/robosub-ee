"""Evaluation pipeline for the real benchmark (RESEARCH_PLAN.md §6.5, §8.1-8.3, §8.5).

Modules
-------
align      chirp matched filter, clock-drift correction, fine lag, utterance cuts (§6.5)
normalize  R0-VAD speech-active RMS normalization to -26 dBFS, noise-only gain rule (§8.3)
asr        Whisper protocol, text normalization, corpus WER (S/D/I), hallucination flags (§8.3)
metrics    ESTOI/STOI (band-limited), PESQ-WB, SI-SDR, CI-SDR at 16 kHz (§8.1, §8.2)
stats      cluster bootstrap, paired CIs, TOST, Holm, MAPSSWE, Kendall tau (§8.5)

Heavy dependencies (whisper/torch, pystoi, pesq, jiwer, NeMo) are imported
lazily inside the functions that need them.
"""

from . import align, asr, metrics, normalize, stats  # noqa: F401

__all__ = ["align", "asr", "metrics", "normalize", "stats"]
