"""ASR scoring: Whisper protocol, text normalization, corpus WER, hallucination flags.

Implements RESEARCH_PLAN.md §8.3 "Whisper WER protocol" (and the Parakeet
secondary ASR of §8.1):

- ``openai-whisper`` ``large-v3`` by default, fp16 on CUDA; the package
  version and the checkpoint sha256 are recorded (:meth:`WhisperASR.info`).
  openai-whisper re-verifies the sha256 embedded in its download URL every
  time it loads a checkpoint, so the recorded hash is the verified one.
- Decoding: ``language="en"``, ``task="transcribe"``, ``temperature=0``
  (passed as a scalar, so the temperature fallback is disabled),
  ``beam_size=5``, ``condition_on_previous_text=False``, no prompt. The default
  thresholds (compression ratio 2.4, logprob -1.0, no-speech 0.6) are passed
  explicitly so they appear in the recorded config.
- ``whisper.normalizers.EnglishTextNormalizer`` on reference and hypothesis;
  corpus WER via jiwer as summed errors / summed reference words, with S, D
  and I reported.
- Hallucination flags: hyp/ref word-count ratio > 2 or zlib compression ratio
  of the hypothesis > 2.4. WER is reported with and without flagged
  utterances (:func:`wer_report`).
- Transcripts are cached in JSONL keyed by (system, utt_id, model).
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np
from tqdm import tqdm

__all__ = [
    "DecodeConfig",
    "ASRBackend",
    "WhisperASR",
    "ParakeetASR",
    "TranscriptCache",
    "transcribe_items",
    "get_normalizer",
    "normalize_text",
    "compression_ratio",
    "WERCounts",
    "utterance_counts",
    "corpus_wer",
    "word_alignment",
    "hallucination_flags",
    "wer_report",
]

ASR_FS = 16000


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_normalizer() -> Callable[[str], str]:
    """Whisper's ``EnglishTextNormalizer`` (§8.3), loaded once."""
    from whisper.normalizers import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize_text(text: str, normalizer: Callable[[str], str] | None = None) -> str:
    """Normalize with ``normalizer`` (default: Whisper English) and collapse spaces."""
    f = normalizer or get_normalizer()
    return " ".join(f(text or "").split())


def compression_ratio(text: str) -> float:
    """zlib compression ratio, identical to Whisper's ``compression_ratio`` check."""
    b = (text or "").encode("utf-8")
    if not b:
        return 0.0
    return len(b) / len(zlib.compress(b))


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------
@dataclass
class WERCounts:
    """Corpus or per-utterance edit counts."""

    substitutions: int
    deletions: int
    insertions: int
    hits: int
    n_ref: int
    n_utts: int = 1

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.n_ref if self.n_ref > 0 else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(errors=self.errors, wer=self.wer)
        return d


def _counts_one(ref: str, hyp: str) -> WERCounts:
    import jiwer

    r, h = ref.split(), hyp.split()
    # jiwer splits on single spaces only; feed it the same tokens counted here
    ref, hyp = " ".join(r), " ".join(h)
    if not r:
        return WERCounts(0, 0, len(h), 0, 0)
    if not h:
        return WERCounts(0, len(r), 0, 0, len(r))
    out = jiwer.process_words(ref, hyp)
    return WERCounts(out.substitutions, out.deletions, out.insertions, out.hits, len(r))


def utterance_counts(
    refs: Sequence[str],
    hyps: Sequence[str],
    normalize: bool = True,
    normalizer: Callable[[str], str] | None = None,
) -> list[WERCounts]:
    """Per-utterance S/D/I/hits/N after (optional) normalization of both sides.

    Empty references are kept (N = 0, every hypothesis word an insertion) so
    corpus sums stay correct; jiwer itself rejects empty references.
    """
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps differ in length")
    if normalize:
        refs = [normalize_text(r, normalizer) for r in refs]
        hyps = [normalize_text(h, normalizer) for h in hyps]
    return [_counts_one(r, h) for r, h in zip(refs, hyps)]


def corpus_wer(
    refs: Sequence[str],
    hyps: Sequence[str],
    normalize: bool = True,
    normalizer: Callable[[str], str] | None = None,
    mask: Sequence[bool] | None = None,
) -> WERCounts:
    """Corpus WER = sum(S+D+I) / sum(N_ref) (not the mean of per-utterance WERs).

    ``mask`` (True = keep) selects a subset, e.g. ``~flags`` to drop
    hallucination-flagged utterances.
    """
    per = utterance_counts(refs, hyps, normalize, normalizer)
    if mask is not None:
        per = [c for c, m in zip(per, mask) if m]
    return WERCounts(
        sum(c.substitutions for c in per),
        sum(c.deletions for c in per),
        sum(c.insertions for c in per),
        sum(c.hits for c in per),
        sum(c.n_ref for c in per),
        len(per),
    )


def word_alignment(ref: str, hyp: str) -> tuple[np.ndarray, np.ndarray]:
    """Word-level alignment for MAPSSWE (already-normalized strings).

    Returns ``ref_err`` (bool per reference word: substituted or deleted) and
    ``ins`` (int per gap, ``len(ref_words) + 1`` gaps; gap g precedes word g).
    """
    import jiwer

    r, h = ref.split(), hyp.split()
    ref, hyp = " ".join(r), " ".join(h)  # keep jiwer word indices consistent with r/h
    ref_err = np.zeros(len(r), dtype=bool)
    ins = np.zeros(len(r) + 1, dtype=np.int64)
    if not r:
        ins[0] = len(h)
        return ref_err, ins
    if not h:
        ref_err[:] = True
        return ref_err, ins
    out = jiwer.process_words(ref, hyp)
    for ch in out.alignments[0]:
        if ch.type in ("substitute", "delete"):
            ref_err[ch.ref_start_idx : ch.ref_end_idx] = True
        elif ch.type == "insert":
            ins[ch.ref_start_idx] += ch.hyp_end_idx - ch.hyp_start_idx
    return ref_err, ins


def hallucination_flags(
    refs: Sequence[str],
    hyps: Sequence[str],
    length_ratio: float = 2.0,
    compression_threshold: float = 2.4,
    normalize: bool = True,
    normalizer: Callable[[str], str] | None = None,
    raw_hyps: Sequence[str] | None = None,
) -> np.ndarray:
    """§8.3 hallucination flags: len(hyp)/len(ref) > 2 or compression ratio > 2.4.

    Word counts are taken after normalization. The compression ratio is
    computed on ``raw_hyps`` (Whisper's raw text, as in its own check) when
    given, else on the hypotheses as passed. Empty references flag any
    non-empty hypothesis (infinite length ratio).
    """
    raw = list(raw_hyps) if raw_hyps is not None else list(hyps)
    if normalize:
        refs = [normalize_text(r, normalizer) for r in refs]
        hyps = [normalize_text(h, normalizer) for h in hyps]
    flags = np.zeros(len(refs), dtype=bool)
    for i, (r, h, hr) in enumerate(zip(refs, hyps, raw)):
        nr, nh = len(r.split()), len(h.split())
        ratio = nh / nr if nr else (np.inf if nh else 0.0)
        flags[i] = ratio > length_ratio or compression_ratio(hr) > compression_threshold
    return flags


def wer_report(
    refs: Sequence[str],
    hyps: Sequence[str],
    normalizer: Callable[[str], str] | None = None,
    raw_hyps: Sequence[str] | None = None,
    **flag_kwargs: float,
) -> dict:
    """Corpus WER with S/D/I, with and without hallucination-flagged utterances."""
    flags = hallucination_flags(refs, hyps, normalizer=normalizer, raw_hyps=raw_hyps, **flag_kwargs)
    all_ = corpus_wer(refs, hyps, normalizer=normalizer)
    kept = corpus_wer(refs, hyps, normalizer=normalizer, mask=~flags)
    return {
        "all": all_.as_dict(),
        "unflagged": kept.as_dict(),
        "n_flagged": int(flags.sum()),
        "flag_rate": float(flags.mean()) if len(flags) else float("nan"),
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# ASR backends
# ---------------------------------------------------------------------------
class ASRBackend(Protocol):
    """Anything that turns 16 kHz mono float audio into text."""

    name: str

    def transcribe(self, audio: np.ndarray) -> dict: ...

    def info(self) -> dict: ...


@dataclass
class DecodeConfig:
    """Frozen §8.3 decoding settings (ablation A14 is decided on v0-dev)."""

    language: str = "en"
    task: str = "transcribe"
    temperature: float = 0.0  # scalar -> no temperature fallback
    beam_size: int | None = 5
    best_of: int | None = None
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    without_timestamps: bool = False


class WhisperASR:
    """openai-whisper backend with the §8.3 protocol; the model loads lazily."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str | None = None,
        decode: DecodeConfig | None = None,
        download_root: str | None = None,
        fp16: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.name = f"whisper-{model_name}"
        self._device = device
        self.decode = decode or DecodeConfig()
        self.download_root = download_root
        self._fp16 = fp16
        self._model = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def fp16(self) -> bool:
        return self.device.startswith("cuda") if self._fp16 is None else self._fp16

    @property
    def model(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.model_name, device=self.device, download_root=self.download_root)
        return self._model

    def transcribe(self, audio: np.ndarray) -> dict:
        """Transcribe 16 kHz mono audio. Returns text plus decoder diagnostics."""
        d = self.decode
        kw = dict(
            language=d.language,
            task=d.task,
            temperature=d.temperature,
            beam_size=d.beam_size,
            condition_on_previous_text=d.condition_on_previous_text,
            initial_prompt=d.initial_prompt,
            compression_ratio_threshold=d.compression_ratio_threshold,
            logprob_threshold=d.logprob_threshold,
            no_speech_threshold=d.no_speech_threshold,
            without_timestamps=d.without_timestamps,
            fp16=self.fp16,
            verbose=None,
        )
        if d.best_of is not None:
            kw["best_of"] = d.best_of
        out = self.model.transcribe(np.ascontiguousarray(audio, dtype=np.float32), **kw)
        segs = out.get("segments", [])
        text = out.get("text", "").strip()
        return {
            "text": text,
            "compression_ratio": compression_ratio(text),
            "avg_logprob": float(np.mean([s["avg_logprob"] for s in segs])) if segs else float("nan"),
            "no_speech_prob": float(np.max([s["no_speech_prob"] for s in segs])) if segs else float("nan"),
            "n_segments": len(segs),
        }

    def info(self) -> dict:
        """Package version, checkpoint URL sha256 and decoding config for the record."""
        import whisper

        try:
            from importlib.metadata import version

            ver = version("openai-whisper")
        except Exception:  # pragma: no cover - editable installs
            ver = getattr(whisper, "__version__", "unknown")
        url = getattr(whisper, "_MODELS", {}).get(self.model_name, "")
        sha = url.split("/")[-2] if url else None
        return {
            "backend": "openai-whisper",
            "whisper_version": ver,
            "model": self.model_name,
            "model_sha256": sha,
            "model_url": url,
            "device": self.device,
            "fp16": self.fp16,
            "decode": asdict(self.decode),
        }


class ParakeetASR:
    """Secondary ASR (§8.1): NVIDIA Parakeet-TDT-0.6B-v2 through NeMo, lazily loaded.

    Raises ``ImportError`` with install instructions if NeMo is missing; NeMo
    is deliberately not a dependency of the ``uwse`` env.
    """

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2", device: str | None = None) -> None:
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError as e:
                raise ImportError(
                    "ParakeetASR needs NVIDIA NeMo: pip install 'nemo_toolkit[asr]' "
                    "(not installed in the uwse env by default)"
                ) from e
            m = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
            if self._device:
                m = m.to(self._device)
            m.eval()
            self._model = m
        return self._model

    def transcribe(self, audio: np.ndarray) -> dict:
        out = self.model.transcribe([np.ascontiguousarray(audio, dtype=np.float32)])
        hyp = out[0]
        text = getattr(hyp, "text", hyp)
        if isinstance(text, (list, tuple)):  # some NeMo versions nest results
            text = text[0]
        text = str(text).strip()
        return {"text": text, "compression_ratio": compression_ratio(text)}

    def info(self) -> dict:
        try:
            from importlib.metadata import version

            ver = version("nemo_toolkit")
        except Exception:
            ver = "unknown"
        return {"backend": "nemo", "nemo_version": ver, "model": self.model_name}


# ---------------------------------------------------------------------------
# Cache and batch transcription
# ---------------------------------------------------------------------------
class TranscriptCache:
    """Append-only JSONL transcript cache keyed by (system, utt_id, model)."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self._d: dict[tuple[str, str, str], dict] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        self._d[(r["system"], r["utt_id"], r["model"])] = r

    def get(self, system: str, utt_id: str, model: str) -> dict | None:
        return self._d.get((system, utt_id, model))

    def __contains__(self, key: tuple[str, str, str]) -> bool:
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)

    def put(self, record: dict) -> None:
        key = (record["system"], record["utt_id"], record["model"])
        self._d[key] = record
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_16k(item: dict) -> np.ndarray:
    if item.get("audio") is not None:
        x = np.asarray(item["audio"], np.float64)
        fs = int(item.get("fs", ASR_FS))
    else:
        from uwse.audio import load

        x, fs = load(item["wav"])
        if x.ndim == 2:
            x = x[0]
    if fs != ASR_FS:
        from uwse.audio import resample

        x = resample(x, fs, ASR_FS)
    gain = item.get("gain")
    if gain is not None and np.isfinite(gain):
        x = x * float(gain)
    return x.astype(np.float32)


def transcribe_items(
    asr: ASRBackend,
    items: Iterable[dict],
    cache: TranscriptCache | None = None,
    model_key: str | None = None,
    progress: bool = True,
) -> list[dict]:
    """Transcribe items, reusing cached transcripts.

    Each item has ``system``, ``utt_id`` and either ``wav`` (path) or
    ``audio`` (+ ``fs``); an optional linear ``gain`` (from level
    normalization) is applied before decoding. Returns one record per item
    with ``system, utt_id, model, text, compression_ratio, ...``.
    """
    items = list(items)
    model_key = model_key or getattr(asr, "name", "asr")
    out = []
    for it in tqdm(items, desc=f"ASR {model_key}", disable=not progress):
        rec = cache.get(it["system"], it["utt_id"], model_key) if cache else None
        if rec is None:
            res = asr.transcribe(_load_16k(it))
            rec = {"system": it["system"], "utt_id": it["utt_id"], "model": model_key, **res}
            if cache is not None:
                cache.put(rec)
        out.append(rec)
    return out
