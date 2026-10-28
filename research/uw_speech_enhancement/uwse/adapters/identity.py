"""C0 passthrough adapter and the small helpers every adapter shares.

Implements condition C0 ("Unprocessed (scored channel)", RESEARCH_PLAN.md
§7.3). For multichannel input (B, C, T) the adapter returns the scored channel;
which channel is scored is decided on v0-dev (§3, §5.4, ablation A12), so it is
a constructor argument here (default 0).

The helpers below (`fit_length`, `select_channel`, `as_batch`, `eval_mode`,
`chunked_apply`, `si_snr`, `multires_stft_loss`, `project_root`) live in this
module because it has no optional dependencies; the other adapter modules import
them from here.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import SEAdapter, register


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def project_root() -> Path:
    """`research/uw_speech_enhancement/` (two levels above this package)."""
    return Path(__file__).resolve().parents[2]


def pretrained_dir(name: str) -> Path:
    """Local download directory `pretrained_models/<name>` under the project root."""
    d = project_root() / "pretrained_models" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def as_batch(wav: Tensor) -> tuple[Tensor, bool]:
    """Promote (T,) to (1, T). Returns (tensor, was_1d)."""
    if wav.dim() == 1:
        return wav.unsqueeze(0), True
    return wav, False


def fit_length(x: Tensor, T: int) -> Tensor:
    """Zero-pad or trim the last axis to exactly T samples."""
    n = x.shape[-1]
    if n == T:
        return x
    if n > T:
        return x[..., :T]
    return F.pad(x, (0, T - n))


def select_channel(wav: Tensor, channel: int) -> Tensor:
    """(B, C, T) -> (B, T) at `channel`; (B, T) and (T,) pass through."""
    if wav.dim() == 3:
        if not -wav.shape[1] <= channel < wav.shape[1]:
            raise IndexError(f"scored channel {channel} out of range for {wav.shape[1]} channels")
        return wav[:, channel]
    return wav


@contextlib.contextmanager
def eval_mode(module: nn.Module | None) -> Iterator[None]:
    """`torch.no_grad()` + `module.eval()`, restoring the previous train flag."""
    was_training = bool(module is not None and module.training)
    if module is not None:
        module.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if module is not None and was_training:
            module.train(True)


def chunked_apply(
    fn: Callable[[Tensor], Tensor],
    wav: Tensor,
    chunk: int,
    overlap: int,
    match_gain: bool = False,
) -> Tensor:
    """Apply a (B, T) -> (B, T) function on overlapping chunks with linear crossfades.

    Used to bound memory for models whose cost grows with T (SepFormer
    attention). Inputs shorter than `chunk` are passed through `fn` unchanged.

    match_gain: models trained with a scale-invariant loss (SI-SNR) have an
    arbitrary output gain *and sign* per call, so independent chunks would not
    crossfade coherently. With match_gain each chunk k > 0 is multiplied by the
    least-squares gain g = <y_{k-1}, y_k> / <y_k, y_k> computed on the overlap,
    which carries the first chunk's gain through the file.
    """
    B, T = wav.shape
    if T <= chunk:
        return fit_length(fn(wav), T)
    if not 0 <= overlap < chunk:
        raise ValueError("need 0 <= overlap < chunk")
    hop = chunk - overlap
    out = torch.zeros(B, T, dtype=wav.dtype, device=wav.device)
    wsum = torch.zeros(T, dtype=wav.dtype, device=wav.device)
    ramp = torch.linspace(0.0, 1.0, overlap + 2, dtype=wav.dtype, device=wav.device)[1:-1]
    start = 0
    prev_tail: Tensor | None = None  # previous chunk's output over [start, start + overlap)
    while True:
        stop = min(start + chunk, T)
        n = stop - start
        seg = fit_length(fn(wav[:, start:stop]).to(out.device, out.dtype), n)
        if match_gain and prev_tail is not None and overlap > 0:
            k = min(overlap, n)
            cur = seg[:, :k]
            e_cur = cur.pow(2).sum(-1, keepdim=True)
            e_prev = prev_tail[:, :k].pow(2).sum(-1, keepdim=True)
            g = (prev_tail[:, :k] * cur).sum(-1, keepdim=True) / e_cur.clamp_min(1e-12)
            # a (near-)silent overlap carries no gain information; g would be ~0
            # and would silence this and every later chunk, so keep g = 1 there
            silent = (e_cur <= 1e-10 * k) | (e_prev <= 1e-10 * k)
            seg = seg * torch.where(silent, torch.ones_like(g), g)
        if overlap > 0 and stop < T:
            prev_tail = seg[:, hop : hop + overlap]  # n == chunk here
        # weights: linear fade-in over the overlap unless first chunk, fade-out
        # unless last chunk; the ramps of neighbours sum to 1 in the overlap
        w = torch.ones(n, dtype=wav.dtype, device=wav.device)
        k = min(overlap, n)
        if start > 0 and k > 0:
            w[:k] = ramp[:k]
        if stop < T and overlap > 0:
            w[-overlap:] = ramp.flip(0)
        out[:, start:stop] += seg * w
        wsum[start:stop] += w
        if stop == T:
            break
        start += hop
    return out / wsum.clamp_min(1e-8)


def si_snr(est: Tensor, ref: Tensor, eps: float = 1e-8) -> Tensor:
    """Scale-invariant SNR in dB per item, (B, T) x (B, T) -> (B,).

    SI-SNR = 10 log10(|s_t|^2 / |e|^2), s_t = <s', s> s / |s|^2, e = s' - s_t,
    with both signals zero-meaned first (Le Roux et al., 2019).
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) * ref / (ref.pow(2).sum(-1, keepdim=True) + eps)
    noise = est - proj
    return 10.0 * torch.log10(proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)


def multires_stft_loss(
    est: Tensor,
    ref: Tensor,
    fft_sizes: Sequence[int] = (256, 512, 1024),
    gamma: float = 0.3,
    factor_mag: float = 1.0,
    factor_complex: float = 1.0,
) -> Tensor:
    """Multi-resolution compressed-spectrum loss on waveforms (B, T).

    Same form as DeepFilterNet's `MultiResSpecLoss`: for each FFT size,
    MSE(|X|^g, |Y|^g) + MSE of the compressed complex spectra
    |X|^g e^{j angle X}. Hop = n_fft / 4, Hann window.
    """
    total = est.new_zeros(())
    for n in fft_sizes:
        win = torch.hann_window(n, device=est.device, dtype=est.dtype)
        X = torch.stft(est, n, n // 4, window=win, return_complex=True)
        Y = torch.stft(ref, n, n // 4, window=win, return_complex=True)
        Xa = X.abs().clamp_min(1e-12).pow(gamma)
        Ya = Y.abs().clamp_min(1e-12).pow(gamma)
        total = total + factor_mag * F.mse_loss(Xa, Ya)
        if factor_complex > 0:
            Xc = torch.view_as_real(Xa * torch.exp(1j * torch.angle(X)))
            Yc = torch.view_as_real(Ya * torch.exp(1j * torch.angle(Y)))
            total = total + factor_complex * F.mse_loss(Xc, Yc)
    return total


# --------------------------------------------------------------------------
# C0
# --------------------------------------------------------------------------
@register("identity")
class IdentityAdapter(SEAdapter):
    """C0: returns the input (scored channel for multichannel input)."""

    sample_rate = 16000
    multichannel = False

    def __init__(
        self,
        device: str | torch.device = "cpu",
        scored_channel: int = 0,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(device)
        self.scored_channel = scored_channel
        # passthrough has no native rate; callers may set the rate they score at
        self.sample_rate = sample_rate

    def load_pretrained(self) -> None:
        return None

    def enhance(self, wav: Tensor) -> Tensor:
        wav, was_1d = as_batch(wav)
        out = select_channel(wav, self.scored_channel).clone()
        return out[0] if was_1d else out

    def trainable_params(self, mode: str = "full"):
        return []
