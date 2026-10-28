"""C8: Whisper large-v3 + LoRA adapted on uwsim audio, no SE (RESEARCH_PLAN.md §7.2-7.4, RQ2d).

Recipe (§7.2 row "Whisper large-v3 (C8)", Rong's Oceans 2025 setting):
LoRA r = 8, alpha = 32, dropout 0.05 on ``q_proj``/``v_proj`` (encoder and decoder),
peak LR 1e-3, weight decay 0, cross-entropy on the LibriSpeech transcripts of the
simulated audio. The base model is HF ``transformers`` ``WhisperForConditionalGeneration``
(fp16 weights on CUDA, LoRA weights in fp32), trained with `uwse.train.loop.fit`
(same schedule, clipping, accumulation, early stopping, checkpoints, JSONL log).

Data: any pair dataset whose items carry the clean utterance id in their meta
(``utt_id`` or ``clean_file``) and are whole utterances at 16 kHz (``segment_s=None``)
— transcripts cover the whole file, so random crops are not allowed. `TranscribedAudio`
turns such a dataset into ``(audio16k[T], transcript)`` items.

Evaluation WER (beam 5, normalizer, §8.3) lives in `uwse.evaluation`; here only the
validation CE is used for early stopping.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from uwse.train.loop import TrainConfig, TrainResult, fit, seed_everything

__all__ = [
    "LoRASpec",
    "load_librispeech_transcripts",
    "default_text_transform",
    "TranscribedAudio",
    "WhisperCollate",
    "build_whisper_lora",
    "WhisperLoRATrainable",
    "train_whisper_lora",
]

WHISPER_FS = 16000
WHISPER_MAX_S = 30.0


@dataclass
class LoRASpec:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


def load_librispeech_transcripts(root: str) -> dict[str, str]:
    """{utt_id: TRANSCRIPT} from every ``*.trans.txt`` under ``root`` (recursive)."""
    out: dict[str, str] = {}
    for p in glob.glob(os.path.join(root, "**", "*.trans.txt"), recursive=True):
        with open(p, encoding="utf-8") as f:
            for line in f:
                uid, _, text = line.strip().partition(" ")
                if uid:
                    out[uid] = text
    return out


def default_text_transform(text: str) -> str:
    """LibriSpeech is upper case without punctuation; Whisper emits sentence case.

    Lower-casing and capitalising the first letter keeps the targets close to
    Whisper's output distribution (WER is computed after the Whisper normalizer, §8.3,
    so casing does not change the metric, only how far the targets pull the LM).
    """
    t = text.strip().lower()
    return t[:1].upper() + t[1:]


def _utt_id(meta: Mapping[str, Any]) -> str | None:
    if meta.get("utt_id"):
        return str(meta["utt_id"])
    for k in ("clean_file", "clean_path", "speech_file", "speech_path", "path", "src"):
        if meta.get(k):
            return os.path.splitext(os.path.basename(str(meta[k])))[0]
    return None


class TranscribedAudio(Dataset):
    """``(noisy audio at 16 kHz [T], transcript)`` from a pair dataset + transcript map.

    The SE *target* is ignored: C8 adapts the recognizer to the channel, so the
    input is the simulated noisy hydrophone signal and the label is the text.
    Items longer than 30 s are cut to 30 s (Whisper's window); LibriSpeech
    utterances used here should be filtered to <= 30 s beforehand so this never
    truncates a transcript (a counter records how often it happened).
    """

    def __init__(self, pairs: Dataset, transcripts: Mapping[str, str], fs: int = WHISPER_FS,
                 text_transform: Callable[[str], str] | None = default_text_transform) -> None:
        self.pairs, self.transcripts, self.fs = pairs, transcripts, fs
        self.text_transform = text_transform
        self.n_truncated = 0

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, key: Any) -> tuple[Tensor, str]:
        item = self.pairs[key]
        wav, meta = item[0], (item[2] if len(item) > 2 else {})
        uid = _utt_id(meta)
        if uid is None or uid not in self.transcripts:
            raise KeyError(f"no transcript for item {key!r} (utt_id={uid!r})")
        if int(meta.get("fs", self.fs)) != self.fs:
            raise ValueError(f"expected {self.fs} Hz audio, got {meta.get('fs')}")
        wav = torch.as_tensor(wav, dtype=torch.float32).reshape(-1)
        n_max = int(WHISPER_MAX_S * self.fs)
        if wav.numel() > n_max:
            wav = wav[:n_max]
            self.n_truncated += 1
        text = self.transcripts[uid]
        if self.text_transform is not None:
            text = self.text_transform(text)
        return wav, text


class WhisperCollate:
    """Batch ``(audio, text)`` into log-mel ``input_features`` and label ids.

    Labels are the tokenizer's ``<|startoftranscript|><|en|><|transcribe|>
    <|notimestamps|> text <|endoftext|>`` sequence with the leading
    decoder-start token removed (the model re-inserts it when shifting labels
    right) and padding replaced by -100 (ignored by the CE loss).
    """

    def __init__(self, processor: Any, decoder_start_token_id: int, language: str = "english",
                 task: str = "transcribe") -> None:
        self.processor = processor
        self.start_id = int(decoder_start_token_id)
        tok = processor.tokenizer
        if hasattr(tok, "set_prefix_tokens"):
            tok.set_prefix_tokens(language=language, task=task, predict_timestamps=False)

    def __call__(self, items: Sequence[tuple[Tensor, str]]) -> dict[str, Any]:
        audios = [np.asarray(a.numpy() if isinstance(a, Tensor) else a, dtype=np.float32)
                  for a, _ in items]
        feats = self.processor.feature_extractor(audios, sampling_rate=WHISPER_FS,
                                                 return_tensors="pt").input_features
        tok = self.processor.tokenizer([t for _, t in items], padding=True, return_tensors="pt")
        labels = tok.input_ids.masked_fill(tok.attention_mask.eq(0), -100)
        if bool((labels[:, 0] == self.start_id).all()):
            labels = labels[:, 1:]
        audio_s = float(sum(a.size for a in audios)) / WHISPER_FS
        return {"input_features": feats, "labels": labels, "audio_s": audio_s}


def build_whisper_lora(model_name: str = "openai/whisper-large-v3", lora: LoRASpec | None = None,
                       device: str | torch.device = "cpu", fp16_base: bool | None = None,
                       gradient_checkpointing: bool = False) -> tuple[nn.Module, Any]:
    """Load Whisper + processor and wrap it with peft LoRA. Returns (peft_model, processor).

    ``fp16_base`` (default: True on CUDA) stores the frozen base in fp16 to fit
    large-v3 (1.55 B parameters) in 8 GB; peft keeps the LoRA weights in fp32.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    lora = lora or LoRASpec()
    device = torch.device(device)
    if fp16_base is None:
        fp16_base = device.type == "cuda"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16 if fp16_base else torch.float32)
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                     target_modules=list(lora.target_modules), bias="none")
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():  # LoRA in fp32 so GradScaler can unscale it
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    model.to(device)
    return model, processor


class WhisperLoRATrainable:
    """`uwse.train.loop.Trainable` for a peft-wrapped Whisper."""

    def __init__(self, model: nn.Module, model_name: str = "", lora: LoRASpec | None = None) -> None:
        self.model, self.model_name, self.lora = model, model_name, lora or LoRASpec()
        self._params = [p for p in model.parameters() if p.requires_grad]
        if not self._params:
            raise ValueError("no trainable LoRA parameters")
        # dtype of the frozen base weights: input features must match it
        self._in_dtype = next((p.dtype for p in model.parameters() if not p.requires_grad),
                              torch.float32)

    def parameters(self) -> list[nn.Parameter]:
        return self._params

    def batch_loss(self, batch: dict[str, Any], device: torch.device) -> tuple[Tensor, float]:
        feats = batch["input_features"].to(device, dtype=self._in_dtype, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        out = self.model(input_features=feats, labels=labels)
        return out.loss.float(), float(batch["audio_s"])

    def set_train(self, mode: bool) -> None:
        self.model.train(mode)

    def save(self, path: str, **extra: Any) -> None:
        from peft import get_peft_model_state_dict

        sd = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.model).items()}
        torch.save({"model": sd, "adapter": "whisper_lora", "base": self.model_name,
                    "lora": vars(self.lora), **extra}, path)

    def load(self, path: str) -> None:
        from peft import set_peft_model_state_dict

        st = torch.load(path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(self.model, st["model"])


def train_whisper_lora(train_ds: Dataset, val_ds: Dataset | None, cfg: TrainConfig | dict[str, Any],
                       model_name: str = "openai/whisper-large-v3", lora: LoRASpec | None = None,
                       gradient_checkpointing: bool = True,
                       model_and_processor: tuple[nn.Module, Any] | None = None) -> TrainResult:
    """Train C8 on ``(audio16k, transcript)`` datasets (e.g. `TranscribedAudio`).

    ``cfg.accum_steps`` should be set explicitly (audio length per item varies);
    otherwise it is derived from ``audio_s_per_step / (batch_size * segment_s)``
    with ``segment_s`` read as the mean utterance duration. ``cfg.param_mode`` is
    forced to "lora" (weight decay 0 unless set).
    """
    cfg = cfg if isinstance(cfg, TrainConfig) else TrainConfig.from_dict(cfg)
    cfg.param_mode = "lora"
    device = cfg.resolved_device()
    lora = lora or LoRASpec()
    if model_and_processor is None:
        seed_everything(cfg.seed, cfg.deterministic)  # LoRA A-matrix init follows the run seed
        model, processor = build_whisper_lora(model_name, lora, device,
                                              gradient_checkpointing=gradient_checkpointing)
    else:
        model, processor = model_and_processor
        model.to(device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    collate = WhisperCollate(processor, base.config.decoder_start_token_id)
    trainable = WhisperLoRATrainable(model, model_name, lora)
    res = fit(trainable, train_ds, val_ds, cfg, collate_fn=collate, device=device)
    if hasattr(model, "save_pretrained"):  # also export a standard peft adapter folder
        best = os.path.join(cfg.run_dir, "best.pt")
        if os.path.isfile(best):
            trainable.load(best)
        model.save_pretrained(os.path.join(cfg.run_dir, "lora_adapter_best"))
    return res
