"""Weighted prediction error (WPE) dereverberation, 1-channel and 4-channel.

Implements the classical C1 baseline of RESEARCH_PLAN.md §7.1 / §7.3 with the
`nara_wpe` NumPy implementation (fgnt, MIT). Starting point from §7.1:
STFT 512/128 at 16 kHz, prediction delay 3 frames, 20-40 taps (30 default;
40 taps x 8 ms hop = 320 ms prediction length, scaled toward a pool T60 of
0.25-1 s), 3 iterations. Taps are tuned on v0-dev (§5.4), so they are
constructor arguments.

WPE estimates, per frequency bin, a multichannel linear predictor of the late
reverberation from frames delayed by `delay` and subtracts it; the source
power is re-estimated each iteration (Nakatani et al., 2010; Drude et al.,
2018). `wpe4` uses all channels to predict the late reverb of every channel
and returns the scored one.

Implementation note: `wpe_freq` is nara_wpe's `wpe_v6` algorithm assembled from
nara_wpe's own helpers, with the batched linear solve done by torch because
NumPy's solve is pathologically slow on these systems in the uwse env (4-ch,
30 taps, 2 s: 16.5 s -> 1.0 s; outputs agree to 4e-8 relative).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .base import SEAdapter, register
from .identity import as_batch, fit_length, select_channel


def _solve(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Batched R^-1 P. torch.linalg.solve is ~100x faster than np.linalg.solve
    on these (F, D*taps, D*taps) systems with the OpenBLAS numpy build in the
    uwse env (3-7 s vs 30 ms for 257 x 120 x 120, identical results to 1e-13);
    falls back to nara_wpe's `_stable_solve` (solve, then lstsq) if singular."""
    try:
        return torch.linalg.solve(torch.from_numpy(R), torch.from_numpy(P)).numpy()
    except RuntimeError:
        from nara_wpe.wpe import _stable_solve

        return _stable_solve(R, P)


def wpe_freq(Y: np.ndarray, taps: int = 30, delay: int = 3, iterations: int = 3) -> np.ndarray:
    """WPE on an STFT of shape (F, D, frames); same algorithm as `nara_wpe.wpe.wpe_v6`
    with statistics_mode="full" and psd_context=0, built from nara_wpe's own
    helpers; only the linear solve is swapped (see `_solve`)."""
    from nara_wpe.wpe import build_y_tilde, get_power_inverse, hermite

    X = np.copy(Y)
    Y_tilde = build_y_tilde(Y, taps, delay)  # (F, D*taps, frames), delayed stack
    for _ in range(iterations):
        # ML weighting: 1 / (source power averaged over channels) per (f, t)
        inverse_power = get_power_inverse(X)
        Y_tilde_ip = Y_tilde * inverse_power[..., None, :]
        R = np.ascontiguousarray(np.matmul(Y_tilde_ip, hermite(Y_tilde)))
        P = np.ascontiguousarray(np.matmul(Y_tilde_ip, hermite(Y)))
        G = _solve(R, P)  # (F, D*taps, D) prediction filters
        X = Y - np.matmul(hermite(G), Y_tilde)  # subtract predicted late reverb
    return X


def wpe_numpy(
    x: np.ndarray,
    taps: int = 30,
    delay: int = 3,
    iterations: int = 3,
    n_fft: int = 512,
    hop: int = 128,
) -> np.ndarray:
    """Dereverberate (D, T) float audio with nara_wpe; returns (D, T)."""
    from nara_wpe.utils import istft, stft

    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    T = x.shape[-1]
    Y = stft(x, size=n_fft, shift=hop)  # (D, frames, F)
    Y = Y.transpose(2, 0, 1)  # (F, D, frames) as nara_wpe expects
    Z = wpe_freq(Y, taps=taps, delay=delay, iterations=iterations)
    z = istft(Z.transpose(1, 2, 0), size=n_fft, shift=hop)  # (D, T')
    if z.shape[-1] >= T:
        return z[..., :T]
    return np.pad(z, ((0, 0), (0, T - z.shape[-1])))


class _WPEBase(SEAdapter):
    sample_rate = 16000

    def __init__(
        self,
        device: str | torch.device = "cpu",
        taps: int = 30,
        delay: int = 3,
        iterations: int = 3,
        n_fft: int = 512,
        hop: int = 128,
        scored_channel: int = 0,
        progress: bool = False,
    ) -> None:
        super().__init__(device)
        self.taps, self.delay, self.iterations = taps, delay, iterations
        self.n_fft, self.hop = n_fft, hop
        self.scored_channel = scored_channel
        self.progress = progress

    def load_pretrained(self) -> None:
        return None

    def trainable_params(self, mode: str = "full"):
        return []

    def _run(self, x: Tensor) -> Tensor:
        """x: (B, D, T) -> dereverberated scored channel (B, T)."""
        B, D, T = x.shape
        xn = x.detach().to("cpu", torch.float64).numpy()
        outs = []
        it = tqdm(range(B), desc=self.name, leave=False) if self.progress else range(B)
        for b in it:
            z = wpe_numpy(xn[b], self.taps, self.delay, self.iterations, self.n_fft, self.hop)
            outs.append(z[self.scored_channel if D > 1 else 0])
        y = torch.from_numpy(np.stack(outs)).to(dtype=x.dtype if x.is_floating_point() else torch.float32)
        return fit_length(y, T).to(x.device)


@register("wpe1")
class WPE1Adapter(_WPEBase):
    """Single-channel WPE. (B, T) -> (B, T); (B, C, T) uses the scored channel only."""

    multichannel = False

    def enhance(self, wav: Tensor) -> Tensor:
        wav, was_1d = as_batch(wav)
        x = select_channel(wav, self.scored_channel).unsqueeze(1)  # (B, 1, T)
        y = self._run(x)
        return y[0] if was_1d else y


@register("wpe4")
class WPE4Adapter(_WPEBase):
    """Multichannel WPE over all C channels (4 hydrophones); returns the scored channel."""

    multichannel = True

    def __init__(self, device: str | torch.device = "cpu", n_channels: int | None = 4, **kw) -> None:
        super().__init__(device, **kw)
        self.n_channels = n_channels

    def enhance(self, wav: Tensor) -> Tensor:
        if wav.dim() == 2:  # (C, T) single item
            wav, was_1d = wav.unsqueeze(0), True
        elif wav.dim() == 3:
            was_1d = False
        else:
            raise ValueError(f"wpe4 expects (B, C, T) or (C, T), got {tuple(wav.shape)}")
        if self.n_channels is not None and wav.shape[1] != self.n_channels:
            raise ValueError(f"wpe4 expects {self.n_channels} channels, got {wav.shape[1]}")
        y = self._run(wav)
        return y[0] if was_1d else y
