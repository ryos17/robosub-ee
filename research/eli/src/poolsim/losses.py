"""Losses for fitting the renderer to recordings.

Multi-resolution STFT loss (DAFx/DDSP staple) on the early window, plus an
optional complex L1 on the very first milliseconds once envelopes agree.
"""
import torch


def stft_mag(x, n_fft, hop):
    w = torch.hann_window(n_fft, device=x.device)
    S = torch.stft(x, n_fft, hop_length=hop, window=w, return_complex=True)
    return S.abs()


def multires_stft_loss(pred, target, sizes=(2048, 512, 128), eps=1e-6):
    """pred, target: (C, T). Spectral convergence + log-magnitude L1 across resolutions."""
    loss = 0.0
    for n in sizes:
        hop = n // 4
        P = stft_mag(pred, n, hop)
        T = stft_mag(target, n, hop)
        sc = torch.linalg.norm(T - P) / (torch.linalg.norm(T) + eps)
        lm = (torch.log(P + eps) - torch.log(T + eps)).abs().mean()
        loss = loss + sc + lm
    return loss / len(sizes)


def early_window(x, fs, t0, t1):
    """Slice samples in [t0, t1) seconds along the last axis."""
    return x[..., int(t0 * fs):int(t1 * fs)]


def complex_l1(pred, target):
    return (pred - target).abs().mean()


def bandlimit(x, fs, f_lo, f_hi):
    """Zero spectrum outside [f_lo, f_hi] along last axis (for coarse-to-fine fitting)."""
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(x.shape[-1], 1.0 / fs, device=x.device)
    m = ((f >= f_lo) & (f <= f_hi)).to(X.dtype)
    return torch.fft.irfft(X * m, n=x.shape[-1])
