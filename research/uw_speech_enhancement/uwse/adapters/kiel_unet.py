"""Condition C10: the Kiel DAGA 2026 U-Net recipe, trained from scratch on uwsim.

Implements RESEARCH_PLAN.md §7.3 (row C10) and §7.1/§7.2 (adapter contract),
following the recipe summarised in RESEARCH_UPDATE_3.md ("U-Net (2026)",
Röhrdanz et al., DAGA 2026 pp. 814-817, doi 10.71568/daga2026.144).

Published (RESEARCH_UPDATE_3.md)               | Here
-----------------------------------------------|---------------------------------
6 kHz signal band                               | input resampled 16 -> 12 kHz, so
                                                | the STFT covers 0-6 kHz exactly
FFT 512, hop 32                                 | n_fft=512, hop=32 at 12 kHz
input block 128 x 256 log-magnitude             | 256 bins (DC .. fs/2 - 1 bin),
                                                | 128-frame random crops in training
4-level U-Net                                   | 4 encoder / 4 decoder levels + skips
IRM target, noisy phase reused                  | sigmoid mask, iSTFT with noisy phase
loss: MSE(mask) + lambda * L1(magnitude)        | same; lambda is a parameter
Adam 1e-4, 100 epochs, batch 32                 | training loop lives in uwse.train

Documented choices (not published by Kiel)
------------------------------------------
- **Sample rate.** The summary says "fs 6 kHz"; we read that as a 6 kHz *band*
  (fs = 12 kHz), as RESEARCH_UPDATE_3 §"New baseline" and the C10 row word it.
  ``internal_sr=6000`` reproduces the literal reading (band 0-3 kHz).
- **Output band.** The model's output is band-limited to 6 kHz: nothing above
  the internal Nyquist frequency survives the 16 -> 12 -> 16 kHz round trip.
- **Window** periodic Hann, ``center=True``; Nyquist bin (257th) not fed to the
  network and given mask 0, which yields the 256-bin block Kiel describes.
- **Input normalisation**: log(|Y| + 1e-6), then per-example zero mean / unit
  variance over the (F, T) block (makes the net level-invariant).
- **Architecture widths**: 16-32-64-128 channels, 256 at the bottleneck; two
  3x3 conv + BatchNorm + ReLU per level; 2x2 max-pool down, 2x2 transposed
  conv up (Ronneberger-style). About 1.9 M parameters at ``base_channels=16``.
- **IRM** = sqrt(|S|^2 / (|S|^2 + |N|^2)) with N = STFT(noisy - target) (the
  classic beta = 0.5 IRM of Wang et al.); target = whatever the dataset pairs
  with the noisy input (R1_sim in §7.2).
- **lambda_mag** default 0.1; the L1 magnitude term is normalised by the mean
  target magnitude so it is level-invariant (Kiel's lambda is unpublished).
- **Mixed precision.** Under ``torch.autocast`` (``TrainConfig.amp``) only the
  U-Net runs in reduced precision; resampling, STFT/iSTFT, features, IRM and
  loss stay float32 (a bf16/fp16 resampler corrupts the IRM target).
- **Normalisation statistics** are taken over whatever block the net sees:
  the 128-frame crop in training, the whole utterance in ``enhance``.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import firwin
from torch import Tensor, nn

from .base import SEAdapter, register

__all__ = [
    "KielUNet",
    "KielUNetAdapter",
    "ideal_ratio_mask",
    "resample_torch",
]

_LOG_EPS = 1e-6


def _fp32(device: torch.device):
    """Context that disables autocast so the DSP path stays float32."""
    return torch.autocast(device_type=device.type, enabled=False)
_IRM_EPS = 1e-10


# ---------------------------------------------------------------------------
# resampling (torch, polyphase; matches scipy.signal.resample_poly)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _poly_taps(up: int, down: int) -> np.ndarray:
    # Same design as scipy.signal.resample_poly's default: Kaiser(5.0) low-pass
    # with cutoff at the lower of the two Nyquist rates, gain `up`.
    max_rate = max(up, down)
    half_len = 10 * max_rate
    h = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0))
    return (h * up).astype(np.float64)


def resample_torch(x: Tensor, fs_in: int, fs_out: int) -> Tensor:
    """Polyphase resampling of (..., T) along the last axis, differentiable.

    Output length is ceil(T * up / down), as in ``scipy.signal.resample_poly``.
    Half-precision input is promoted to float32 and the filter always runs with
    autocast disabled.
    """
    if fs_in == fs_out:
        return x
    r = Fraction(fs_out, fs_in).limit_denominator(10000)
    up, down = r.numerator, r.denominator
    if x.is_floating_point() and x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    h = torch.as_tensor(_poly_taps(up, down), dtype=x.dtype, device=x.device)
    half = (h.numel() - 1) // 2
    lead = x.shape[:-1]
    n = x.shape[-1]
    x2 = x.reshape(-1, 1, n)
    if up > 1:
        u = x2.new_zeros(x2.shape[0], 1, n * up)
        u[..., ::up] = x2
    else:
        u = x2
    # symmetric FIR -> correlation == convolution; padding `half` keeps it zero-phase
    with _fp32(x.device):
        y = F.conv1d(u, h.view(1, 1, -1), padding=half)[..., ::down]
    n_out = -(-n * up // down)
    return y[..., :n_out].reshape(*lead, n_out)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def ideal_ratio_mask(target_mag: Tensor, noise_mag: Tensor, beta: float = 0.5) -> Tensor:
    """IRM = (|S|^2 / (|S|^2 + |N|^2))^beta, clipped to [0, 1]."""
    s2 = target_mag.square()
    n2 = noise_mag.square()
    return (s2 / (s2 + n2 + _IRM_EPS)).pow(beta).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
class _DoubleConv(nn.Sequential):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class KielUNet(nn.Module):
    """4-level 2-D U-Net mapping a normalised log-magnitude block to a mask.

    Input (B, 1, F, T) -> output (B, 1, F, T) in [0, 1]. F and T may be any
    size; they are zero-padded internally to multiples of 2**levels and the
    output is cropped back.
    """

    def __init__(self, base_channels: int = 16, levels: int = 4) -> None:
        super().__init__()
        self.levels = levels
        chans = [base_channels * 2**i for i in range(levels)]
        self.enc = nn.ModuleList()
        c_prev = 1
        for c in chans:
            self.enc.append(_DoubleConv(c_prev, c))
            c_prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(chans[-1], chans[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c_prev = chans[-1] * 2
        for c in reversed(chans):
            self.up.append(nn.ConvTranspose2d(c_prev, c, 2, stride=2))
            self.dec.append(_DoubleConv(2 * c, c))
            c_prev = c
        self.head = nn.Conv2d(chans[0], 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        f, t = x.shape[-2:]
        m = 2**self.levels
        pf, pt = (-f) % m, (-t) % m
        if pf or pt:
            x = F.pad(x, (0, pt, 0, pf))
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return torch.sigmoid(self.head(x))[..., :f, :t]


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------
@register("kiel_unet")
class KielUNetAdapter(SEAdapter):
    """SEAdapter for C10. Works on 16 kHz (B, T); processes 0-6 kHz internally."""

    sample_rate = 16000
    #: apply_lora patterns for trainable_params("lora"): every 3x3 conv of the
    #: encoder, bottleneck and decoder double-conv blocks (Conv2d, via peft).
    LORA_TARGETS = (r"^(enc|dec)\.\d+\.[03]$", r"^bottleneck\.[03]$")

    def __init__(
        self,
        device: str | torch.device = "cpu",
        checkpoint: str | None = None,
        internal_sr: int = 12000,
        n_fft: int = 512,
        hop: int = 32,
        base_channels: int = 16,
        lambda_mag: float = 0.1,
        crop_frames: int | None = 128,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__(device)
        self.lora_cfg = dict(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        self.checkpoint = checkpoint
        self.internal_sr = internal_sr
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = n_fft // 2  # network sees bins 0 .. n_fft/2 - 1
        self.base_channels = base_channels
        self.lambda_mag = lambda_mag
        self.crop_frames = crop_frames

    # -- lifecycle ---------------------------------------------------------
    def load_pretrained(self) -> None:
        """Fresh (random) weights, or `self.checkpoint` if one was given."""
        self.model = KielUNet(self.base_channels).to(self.device)
        if self.checkpoint is not None:
            self.load_checkpoint(self.checkpoint)

    def load_checkpoint(self, path: str) -> None:
        if self.model is None:
            self.model = KielUNet(self.base_channels).to(self.device)
        state = torch.load(path, map_location=self.device, weights_only=True)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        self.model.load_state_dict(sd)

    def _require_model(self) -> KielUNet:
        if self.model is None:
            self.load_pretrained()
        return self.model  # type: ignore[return-value]

    # -- signal path ---------------------------------------------------------
    def _stft(self, x: Tensor) -> Tensor:
        win = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        return torch.stft(x, self.n_fft, self.hop, window=win, center=True, return_complex=True)

    def _istft(self, spec: Tensor, length: int) -> Tensor:
        win = torch.hann_window(self.n_fft, device=spec.device, dtype=spec.real.dtype)
        return torch.istft(spec, self.n_fft, self.hop, window=win, center=True, length=length)

    @staticmethod
    def _features(mag: Tensor) -> Tensor:
        """(B, F, T) magnitude -> (B, 1, F, T) normalised log-magnitude."""
        lm = torch.log(mag + _LOG_EPS)
        mu = lm.mean(dim=(-2, -1), keepdim=True)
        sd = lm.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return ((lm - mu) / sd).unsqueeze(1)

    def _to_internal(self, wav: Tensor) -> Tensor:
        x = resample_torch(wav, self.sample_rate, self.internal_sr)
        # reflect padding in the centred STFT needs more than n_fft/2 samples
        if x.shape[-1] <= self.n_fft:
            x = F.pad(x, (0, self.n_fft + 1 - x.shape[-1]))
        return x

    def estimate_mask(self, noisy_spec: Tensor) -> Tensor:
        """(B, n_fft/2+1, T) complex STFT -> (B, n_fft/2+1, T) mask; Nyquist bin = 0."""
        model = self._require_model()
        with _fp32(noisy_spec.device):
            feats = self._features(noisy_spec[:, : self.n_bins].abs())
        m = model(feats).squeeze(1).float()
        return F.pad(m, (0, 0, 0, noisy_spec.shape[1] - self.n_bins))

    def forward_train(self, noisy: Tensor) -> Tensor:
        """Differentiable (B, T) -> (B, T) at 16 kHz."""
        squeeze = noisy.dim() == 1
        x = noisy.unsqueeze(0) if squeeze else noisy
        x = x.to(self.device, torch.float32)
        n = x.shape[-1]
        with _fp32(x.device):
            x12 = self._to_internal(x)
            spec = self._stft(x12)
        mask = self.estimate_mask(spec)
        with _fp32(x.device):
            est = self._istft(spec * mask, x12.shape[-1])
            y = resample_torch(est, self.internal_sr, self.sample_rate)
        y = F.pad(y, (0, max(0, n - y.shape[-1])))[..., :n]
        return y[0] if squeeze else y

    def enhance(self, wav: Tensor) -> Tensor:
        """Inference: no grad, BatchNorm in eval mode (the caller's mode is restored)."""
        model = self._require_model()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return self.forward_train(wav)
        finally:
            model.train(was_training)

    # -- training ------------------------------------------------------------
    def irm_target(self, noisy: Tensor, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (noisy STFT, target magnitude, IRM) at the internal rate, all bins."""
        with torch.no_grad(), _fp32(self.device):
            y = self._to_internal(noisy.to(self.device, torch.float32))
            s = self._to_internal(clean.to(self.device, torch.float32))
            spec_y = self._stft(y)
            spec_s = self._stft(s)
            spec_n = self._stft(y - s)
            irm = ideal_ratio_mask(spec_s.abs(), spec_n.abs())
        return spec_y, spec_s.abs(), irm

    def loss(self, noisy: Tensor, clean: Tensor) -> Tensor:
        """MSE(mask, IRM) + lambda_mag * normalised L1 on masked magnitude."""
        if noisy.dim() == 1:
            noisy, clean = noisy.unsqueeze(0), clean.unsqueeze(0)
        spec_y, mag_s, irm = self.irm_target(noisy, clean)
        model = self._require_model()
        n_frames = spec_y.shape[-1]
        if model.training and self.crop_frames and n_frames > self.crop_frames:
            i0 = int(torch.randint(0, n_frames - self.crop_frames + 1, (1,)))
            sl = slice(i0, i0 + self.crop_frames)
            spec_y, mag_s, irm = spec_y[..., sl], mag_s[..., sl], irm[..., sl]
        k = self.n_bins
        mag_y = spec_y[:, :k].abs()
        with _fp32(mag_y.device):
            feats = self._features(mag_y)
        mask = model(feats).squeeze(1).float()
        with _fp32(mag_y.device):
            loss = F.mse_loss(mask, irm[:, :k])
            if self.lambda_mag > 0:
                l1 = (mask * mag_y - mag_s[:, :k]).abs().mean()
                loss = loss + self.lambda_mag * l1 / (mag_s[:, :k].mean() + 1e-8)
        return loss

    def trainable_params(self, mode: str = "full") -> Iterable[nn.Parameter]:
        """"full": every parameter. "lora": LoRA factors (inserted on first call).

        In "lora" mode, if the model carries no LoRA factors yet,
        :func:`uwse.adapters.lora.apply_lora` is applied with ``LORA_TARGETS``
        and ``self.lora_cfg`` (base weights and BN statistics frozen).
        """
        model = self._require_model()
        if mode == "full":
            return [p for p in model.parameters() if p.requires_grad]
        if mode == "lora":
            ps = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
            if not ps:
                from .lora import apply_lora

                ps = list(apply_lora(model, list(self.LORA_TARGETS), **self.lora_cfg))
            return ps
        raise NotImplementedError(f"{self.name}: mode {mode!r} not supported")

    def n_params(self) -> int:
        return sum(p.numel() for p in self._require_model().parameters())

