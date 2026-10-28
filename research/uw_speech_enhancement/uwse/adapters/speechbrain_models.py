"""SpeechBrain adapters: MetricGAN+ and SepFormer enhancement models.

Implements three rows of RESEARCH_PLAN.md §7.1 and their §7.2 recipes:

- ``metricganplus``: ``speechbrain/metricgan-plus-voicebank`` (16 kHz). C1 zero-
  shot and C2 fine-tuning. Native loss = the MetricGAN+ loop (``MetricGANTrainer``):
  a discriminator learns normalized wide-band PESQ of (output, clean) and the
  generator minimizes (D(output, clean) - 1)^2. A plain spectral MSE loss is
  available as a fallback (``loss_name="spectral_mse"``). Trainable params:
  ``"full"`` (generator) or ``"lora"`` (generator Linear layers via
  ``uwse.adapters.lora.apply_lora``, BLSTM frozen, §7.2).
- ``sepformer_wham16k``: ``speechbrain/sepformer-wham16k-enhancement`` (16 kHz),
  loss = negative SI-SNR (native).
- ``sepformer_whamr``: ``speechbrain/sepformer-whamr-enhancement``. **8 kHz model**:
  ``sample_rate = 8000``, so the caller resamples to 8 kHz and the output carries
  no energy above 4 kHz. Per §7.1 it is scored with that band caveat and also
  compared within 0-4 kHz.

Checkpoints are fetched to ``pretrained_models/<name>`` under the project root.
On Windows, symlinks from the HF cache need developer mode, so the fetch uses
``LocalStrategy.COPY`` when the installed SpeechBrain supports it.
"""

from __future__ import annotations

import inspect
import logging
import random
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import SEAdapter, register
from .identity import as_batch, chunked_apply, eval_mode, fit_length, pretrained_dir, si_snr

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _sb_from_hparams(kind: str, source: str, savedir: str, device: torch.device):
    """Load a SpeechBrain inference interface with Windows-safe options.

    kind: "enhancement" (SpectralMaskEnhancement) or "separation"
    (SepformerSeparation). Uses `speechbrain.inference` (SB >= 1.0) with a
    fallback to the deprecated `speechbrain.pretrained`.
    """
    try:
        if kind == "enhancement":
            from speechbrain.inference.enhancement import SpectralMaskEnhancement as cls
        else:
            from speechbrain.inference.separation import SepformerSeparation as cls
    except ImportError:  # SpeechBrain < 1.0
        if kind == "enhancement":
            from speechbrain.pretrained import SpectralMaskEnhancement as cls
        else:
            from speechbrain.pretrained import SepformerSeparation as cls

    kwargs = dict(source=source, savedir=savedir, run_opts={"device": str(device)})
    params = inspect.signature(cls.from_hparams).parameters
    accepts_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    try:
        from speechbrain.utils.fetching import LocalStrategy

        if "local_strategy" in params or accepts_kw:
            kwargs["local_strategy"] = LocalStrategy.COPY
    except ImportError:
        pass
    # keep parameters trainable; Pretrained() freezes them by default
    kwargs["freeze_params"] = False
    try:
        return cls.from_hparams(**kwargs)
    except TypeError:
        kwargs.pop("freeze_params", None)
        return cls.from_hparams(**kwargs)


class _SpeechBrainAdapter(SEAdapter):
    source: str = ""
    kind: str = "enhancement"

    def __init__(self, device: str | torch.device = "cpu", savedir: str | None = None) -> None:
        super().__init__(device)
        self.savedir = savedir
        self.interface = None  # the SpeechBrain Pretrained object

    def load_pretrained(self) -> None:
        savedir = self.savedir or str(pretrained_dir(self.name))
        self.interface = _sb_from_hparams(self.kind, self.source, savedir, self.device)
        self.model = self._module_from_interface()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.to(self.device).eval()

    def _module_from_interface(self) -> nn.Module:
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load_pretrained()

    def to(self, device: str | torch.device) -> "SEAdapter":
        super().to(device)
        if self.interface is not None:
            self.interface.device = str(self.device)
            self.interface.mods.to(self.device)
        return self


# --------------------------------------------------------------------------
# MetricGAN+
# --------------------------------------------------------------------------
def normalized_pesq(ref: np.ndarray, deg: np.ndarray, fs: int = 16000) -> float:
    """Wide-band PESQ mapped to [0, 1] as in the SpeechBrain MetricGAN+ recipe.

    Recipe: (PESQ_wb + 0.5) / 5, i.e. the nominal range [-0.5, 4.5] -> [0, 1].
    Errors (e.g. no utterance detected in noise-only input) give 0, the bottom
    of the range; the recipe returns PESQ's -1 error value there, which we clip.
    """
    from pesq import pesq

    try:
        score = pesq(fs, np.asarray(ref, dtype=np.float64), np.asarray(deg, dtype=np.float64), "wb")
    except Exception:  # NoUtterancesError, BufferTooShortError
        return 0.0
    return float(np.clip((score + 0.5) / 5.0, 0.0, 1.0))


class MetricGANTrainer:
    """Discriminator + adversarial generator loss of MetricGAN+ (Fu et al., 2021).

    Mirrors the SpeechBrain ``recipes/Voicebank/enhance/MetricGAN`` recipe in:
      - the discriminator architecture (``speechbrain.lobes.models.MetricGAN.
        MetricDiscriminator``, input = [enhanced, clean] magnitude features
        stacked as 2 channels) and its MSE regression of normalized PESQ;
      - the three discriminator targets of MetricGAN+: Q(enhanced, clean),
        Q(noisy, clean) and 1 for (clean, clean);
      - the generator objective (D(enhanced, clean) - target)^2 with target 1;
      - a historical replay buffer of past (enhanced, clean, score) triples
        that the discriminator is also trained on.

    Differs from the recipe in scheduling: the recipe alternates whole
    sub-epochs (D on the training set and history, then G); here one D update
    on the current batch (+ a replay sample) precedes each G step, so a
    model-agnostic training loop can drive it through ``adapter.loss``. The
    pretrained HF checkpoint has no discriminator, so D starts from random init;
    call ``warmup`` on a few batches before fine-tuning the generator.

    PESQ runs on the CPU per utterance (pesq package), which dominates the step
    time; the W4 smoke test measures it (§7.2).
    """

    def __init__(
        self,
        adapter: "MetricGANPlusAdapter",
        lr: float = 5e-4,
        history_size: int = 512,
        history_portion: float = 0.2,
        target_score: float = 1.0,
    ) -> None:
        try:
            from speechbrain.lobes.models.MetricGAN import MetricDiscriminator
        except ImportError as e:  # pragma: no cover - depends on SB version
            raise ImportError("MetricGAN+ discriminator needs speechbrain.lobes.models.MetricGAN") from e
        self.adapter = adapter
        self.discriminator = MetricDiscriminator().to(adapter.device)
        self.opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.history: list[tuple[Tensor, Tensor, float]] = []
        self.history_size = history_size
        self.history_portion = history_portion
        self.target_score = target_score

    def to(self, device: torch.device) -> "MetricGANTrainer":
        self.discriminator.to(device)
        return self

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.discriminator.parameters()

    def _d(self, deg_spec: Tensor, ref_spec: Tensor) -> Tensor:
        # (B, T, F) x2 -> (B, 2, T, F) -> (B, 1)
        return self.discriminator(torch.stack([deg_spec, ref_spec], dim=1))

    def _scores(self, deg: Tensor, ref: Tensor, lengths: Sequence[int] | None = None) -> Tensor:
        fs = self.adapter.sample_rate
        d, r = deg.detach().cpu().numpy(), ref.detach().cpu().numpy()
        out = []
        for i in range(d.shape[0]):
            n = d.shape[1] if lengths is None else int(lengths[i])
            out.append(normalized_pesq(r[i, :n], d[i, :n], fs))
        return torch.tensor(out, dtype=torch.float32).unsqueeze(1)

    def discriminator_step(self, noisy: Tensor, clean: Tensor) -> float:
        """One D update on the current batch (+ replayed history). Returns D loss."""
        ad = self.adapter
        with torch.no_grad():
            noisy_spec = ad.features(noisy)
            clean_spec = ad.features(clean)
            enh_wav, enh_spec = ad._forward(noisy)
        q_enh = self._scores(enh_wav, clean)
        q_noisy = self._scores(noisy, clean)
        dev = ad.device
        self.discriminator.train()
        loss = (
            F.mse_loss(self._d(enh_spec, clean_spec), q_enh.to(dev))
            + F.mse_loss(self._d(noisy_spec, clean_spec), q_noisy.to(dev))
            + F.mse_loss(self._d(clean_spec, clean_spec), torch.ones_like(q_enh, device=dev))
        )
        # historical replay (MetricGAN+): fit D on past generator outputs too
        n_hist = int(round(self.history_portion * len(self.history)))
        if n_hist > 0:
            batch = random.sample(self.history, min(n_hist, noisy.shape[0]))
            for e_spec, c_spec, q in batch:
                loss = loss + F.mse_loss(
                    self._d(e_spec.to(dev).unsqueeze(0), c_spec.to(dev).unsqueeze(0)),
                    torch.tensor([[q]], device=dev),
                ) / len(batch)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        for i in range(enh_spec.shape[0]):
            self.history.append((enh_spec[i].cpu(), clean_spec[i].cpu(), float(q_enh[i])))
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size :]
        return float(loss.detach())

    def generator_loss(self, enh_spec: Tensor, clean: Tensor) -> Tensor:
        """(D(enh, clean) - target)^2 with D's weights frozen for this step.

        D stays in train mode, as in the recipe (Brain.fit keeps every module in
        train mode during the generator sub-stage): its input BatchNorm then
        normalizes with batch statistics, the same way D was fitted. Eval mode
        would use running statistics that, with momentum 0.01 and a randomly
        initialized D, lag far behind and give the generator a different D.
        """
        clean_spec = self.adapter.features(clean).detach()
        self.discriminator.train()
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        try:
            est = self._d(enh_spec, clean_spec)
        finally:
            for p in self.discriminator.parameters():
                p.requires_grad_(True)
        return F.mse_loss(est, torch.full_like(est, self.target_score))

    def warmup(self, batches: Iterable[tuple[Tensor, Tensor]]) -> list[float]:
        """Train D only on (noisy, clean) batches; returns the D losses."""
        return [self.discriminator_step(n.to(self.adapter.device), c.to(self.adapter.device)) for n, c in batches]


@register("metricganplus")
class MetricGANPlusAdapter(_SpeechBrainAdapter):
    """MetricGAN+ (BLSTM spectral-mask generator), 16 kHz.

    Forward (as SpeechBrain's ``SpectralMaskEnhancement`` and the recipe):
    features = log1p(|STFT|) (512-pt, 32 ms / 16 ms), mask = G(features) in
    (0, 1.2) (learnable sigmoid), enhanced = expm1(mask * features), resynthesized
    with the noisy phase by iSTFT. All steps are torch ops, so the waveform is
    differentiable w.r.t. the generator.
    """

    source = "speechbrain/metricgan-plus-voicebank"
    kind = "enhancement"
    sample_rate = 16000
    #: regexes for uwse.adapters.lora.apply_lora: the two nn.Linear layers after the BLSTM
    LORA_TARGETS = (r"^linear\d$",)

    def __init__(
        self,
        device: str | torch.device = "cpu",
        savedir: str | None = None,
        loss_name: str = "metricgan",
        min_mask: float = 0.05,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        d_lr: float = 5e-4,
    ) -> None:
        super().__init__(device, savedir)
        if loss_name not in ("metricgan", "spectral_mse"):
            raise ValueError("loss_name must be 'metricgan' or 'spectral_mse'")
        self.loss_name = loss_name
        self.min_mask = min_mask  # recipe hparam min_mask: 0.05 (training only)
        self.lora_cfg = dict(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.d_lr = d_lr
        self._trainer: MetricGANTrainer | None = None

    def _module_from_interface(self) -> nn.Module:
        return self.interface.mods.enhance_model

    @property
    def trainer(self) -> MetricGANTrainer:
        self._ensure_loaded()
        if self._trainer is None:
            self._trainer = MetricGANTrainer(self, lr=self.d_lr)
        return self._trainer

    def to(self, device):
        super().to(device)
        if self._trainer is not None:
            self._trainer.to(self.device)
        return self

    # -- maths ---------------------------------------------------------------
    def features(self, wav: Tensor) -> Tensor:
        """log1p spectral magnitude (B, frames, F), as SpectralMaskEnhancement."""
        self._ensure_loaded()
        return self.interface.compute_features(wav.to(self.device))

    def _forward(self, noisy: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (enhanced waveform (B, T), enhanced log1p-magnitude features)."""
        noisy = noisy.to(self.device)
        feats = self.features(noisy)
        lengths = torch.ones(noisy.shape[0], device=self.device)
        mask = self.model(feats, lengths=lengths)
        if self.model.training:
            mask = mask.clamp(min=self.min_mask)  # recipe applies min_mask in training
        enh_spec = mask * feats
        wav = self.interface.hparams.resynth(torch.expm1(enh_spec), noisy)
        return fit_length(wav, noisy.shape[-1]), enh_spec

    def enhance(self, wav: Tensor) -> Tensor:
        self._ensure_loaded()
        wav, was_1d = as_batch(wav)
        with eval_mode(self.model):
            out = self._forward(wav.float())[0]
        return out[0] if was_1d else out

    def forward_train(self, noisy: Tensor) -> Tensor:
        self._ensure_loaded()
        return self._forward(noisy.float())[0]

    def loss(self, noisy: Tensor, clean: Tensor) -> Tensor:
        """Generator loss. ``metricgan``: does one internal D update first (when
        the generator is in train mode), then returns (D(G(noisy), clean) - 1)^2.
        ``spectral_mse``: MSE between log1p magnitudes of output and clean."""
        self._ensure_loaded()
        noisy, clean = noisy.float().to(self.device), clean.float().to(self.device)
        if self.loss_name == "metricgan" and self.model.training:
            self.trainer.discriminator_step(noisy, clean)
        _, enh_spec = self._forward(noisy)
        if self.loss_name == "spectral_mse":
            return F.mse_loss(enh_spec, self.features(clean).detach())
        return self.trainer.generator_loss(enh_spec, clean)

    def trainable_params(self, mode: str = "full"):
        self._ensure_loaded()
        if mode == "full":
            return [p for p in self.model.parameters() if p.requires_grad]
        if mode == "lora":
            from .lora import apply_lora  # imported lazily (needs peft)

            # apply_lora freezes everything else: BLSTM and base weights (§7.2)
            return list(apply_lora(self.model, list(self.LORA_TARGETS), **self.lora_cfg))
        raise NotImplementedError(f"{self.name}: mode {mode!r} not supported")

    def save_checkpoint(self, path: str, **extra) -> None:
        if self._trainer is not None:
            extra.setdefault("discriminator", self._trainer.discriminator.state_dict())
        super().save_checkpoint(path, **extra)

    def load_checkpoint(self, path: str) -> None:
        super().load_checkpoint(path)
        state = torch.load(path, map_location=self.device)
        if "discriminator" in state:
            self.trainer.discriminator.load_state_dict(state["discriminator"])


# --------------------------------------------------------------------------
# SepFormer
# --------------------------------------------------------------------------
class _SepFormerAdapter(_SpeechBrainAdapter):
    """SepFormer enhancement (1 source): encoder -> masking transformer -> decoder.

    Loss = negative SI-SNR (native; the WHAM!/WHAMR! enhancement recipes use
    SI-SNR with a single source, so no PIT is needed).
    """

    kind = "separation"
    #: LoRA goes on the transformer feed-forward Linears. nn.MultiheadAttention
    #: uses its fused in_proj weight and out_proj.weight directly in
    #: F.multi_head_attention_forward, so peft adapters on q/k/v/o would be
    #: silently bypassed; §7.2's q,k,v,o targets need a custom MHA wrapper.
    LORA_TARGETS = (r"pos_ffn\.ffn\.\d+$",)

    def __init__(
        self,
        device: str | torch.device = "cpu",
        savedir: str | None = None,
        chunk_seconds: float = 20.0,
        overlap_seconds: float = 1.0,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__(device, savedir)
        self.chunk = int(chunk_seconds * self.sample_rate)
        self.overlap = int(overlap_seconds * self.sample_rate)
        self.lora_cfg = dict(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    def _module_from_interface(self) -> nn.Module:
        return self.interface.mods  # ModuleDict(encoder, masknet, decoder)

    def _forward(self, mix: Tensor) -> Tensor:
        """Same computation as SepformerSeparation.separate_batch, source 0."""
        mods = self.model
        mix = mix.to(self.device)
        mix_w = mods["encoder"](mix)
        est_mask = mods["masknet"](mix_w)  # (n_src, B, N, L)
        est = mods["decoder"](mix_w * est_mask[0])
        return fit_length(est, mix.shape[-1])

    def enhance(self, wav: Tensor) -> Tensor:
        self._ensure_loaded()
        wav, was_1d = as_batch(wav)
        with eval_mode(self.model):
            out = chunked_apply(
                self._forward, wav.float().to(self.device), self.chunk, self.overlap, match_gain=True
            )
        return out[0] if was_1d else out

    def forward_train(self, noisy: Tensor) -> Tensor:
        self._ensure_loaded()
        return self._forward(noisy.float())

    def loss(self, noisy: Tensor, clean: Tensor) -> Tensor:
        est = self.forward_train(noisy)
        return -si_snr(est, clean.float().to(self.device)).mean()

    def trainable_params(self, mode: str = "full"):
        self._ensure_loaded()
        if mode == "full":
            return [p for p in self.model.parameters() if p.requires_grad]
        if mode == "lora":
            from .lora import apply_lora

            return list(apply_lora(self.model, list(self.LORA_TARGETS), **self.lora_cfg))
        raise NotImplementedError(f"{self.name}: mode {mode!r} not supported")


@register("sepformer_wham16k")
class SepFormerWham16kAdapter(_SepFormerAdapter):
    source = "speechbrain/sepformer-wham16k-enhancement"
    sample_rate = 16000


@register("sepformer_whamr")
class SepFormerWhamrAdapter(_SepFormerAdapter):
    """WHAMR! denoising + dereverberation. 8 kHz model: output band <= 4 kHz.

    The caller must resample to 8 kHz in and back to the scoring rate out; any
    metric on the full 16 kHz band penalizes the missing 4-8 kHz band, so §7.1
    requires reporting this row with a band caveat and a 0-4 kHz comparison.
    """

    source = "speechbrain/sepformer-whamr-enhancement"
    sample_rate = 8000
