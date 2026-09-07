"""Differentiable early-reflection image-source renderer for a shoebox pool.

Frequency-domain rendering: each image source contributes
    g / d * R(f) * exp(-j 2 pi f d / c)
where R(f) is the product of per-wall reflection responses along the path.
Fractional delays are exact in the frequency domain, so everything is
differentiable in hydrophone positions, source position, wall positions,
wall filters, sound speed, per-channel gain/delay and the source response.

Coordinates: x along the pool length, y across, z up. Water surface at z=0,
floor at z=-depth. Walls at x=0, x=Lx, y=0, y=Ly. Surface reflection is
pressure-release (-1); floor and side walls have learnable filters.
"""
from __future__ import annotations
import itertools
import math
import torch
import torch.nn as nn

WALLS = ("x0", "x1", "y0", "y1", "floor", "surface")


def image_sources(order: int):
    """Enumerate image indices (nx, ny, nz) with |nx|+|ny|+|nz| <= order.
    Returns list of (nx, ny, nz). Reflection counts per wall follow from signs.
    """
    out = []
    r = range(-order, order + 1)
    for nx, ny, nz in itertools.product(r, r, r):
        if abs(nx) + abs(ny) + abs(nz) <= order:
            out.append((nx, ny, nz))
    return out


def image_position(src, n, L, depth):
    """Position of image (nx,ny,nz) of source `src` in a box [0,Lx]x[0,Ly]x[-depth,0].
    src: (...,3) tensor; L: (2,) tensor (Lx, Ly); depth: scalar tensor.
    """
    nx, ny, nz = n
    x, y, z = src[..., 0], src[..., 1], src[..., 2]
    # standard mirror formula: coordinate for index n along axis with box [a,b]:
    #   even n: x + n*(b-a) ; odd n: 2a - x + (n+1)*(b-a)  (mirror about a, then shift)
    def mirror(v, a, b, k):
        Lk = b - a
        if k % 2 == 0:
            return v + k * Lk
        return 2 * a - v + (k + 1) * Lk
    return torch.stack([
        mirror(x, torch.zeros_like(x), L[0], nx),
        mirror(y, torch.zeros_like(y), L[1], ny),
        mirror(z, -depth, torch.zeros_like(z), nz),
    ], dim=-1)


def wall_counts(n):
    """How many times an image path hits each wall, for index (nx,ny,nz)."""
    nx, ny, nz = n
    # along an axis with index k: |k| reflections, alternating between the two walls,
    # first reflection on the wall in the direction of k's sign.
    def split(k, lo, hi):
        c = {lo: 0, hi: 0}
        first, second = (hi, lo) if k > 0 else (lo, hi)
        a = abs(k)
        c[first] += (a + 1) // 2
        c[second] += a // 2
        return c
    cx = split(nx, "x0", "x1")
    cy = split(ny, "y0", "y1")
    cz = split(nz, "floor", "surface")
    return {**cx, **cy, **cz}


class WallFilter(nn.Module):
    """Frequency-dependent reflection coefficient: magnitude = sigmoid(a) * lowpass(f; fc)
    with learnable log fc, plus a learnable minimum-phase-ish phase term (small FIR).
    Kept simple and stable; swap for SVF/biquads later.
    """
    def __init__(self, r0=0.9, fc=8000.0):
        super().__init__()
        self.a = nn.Parameter(torch.logit(torch.tensor(float(r0))))
        self.log_fc = nn.Parameter(torch.log(torch.tensor(float(fc))))

    def forward(self, f):
        fc = torch.exp(self.log_fc)
        mag = torch.sigmoid(self.a) / torch.sqrt(1 + (f / fc) ** 2)
        phase = -torch.atan(f / fc)  # first-order lowpass phase
        return mag * torch.exp(1j * phase)


class PoolRenderer(nn.Module):
    def __init__(self, Lx=25.0, Ly=12.5, depth=2.0, c=1480.0, n_hyd=4, order=1,
                 fs=96000, n_fft=8192, hyd_init=None, wall_r0=0.9, floor_r0=0.7):
        super().__init__()
        self.fs, self.n_fft, self.order = fs, n_fft, order
        self.L = nn.Parameter(torch.tensor([Lx, Ly]))
        self.depth = nn.Parameter(torch.tensor(float(depth)))
        self.log_c = nn.Parameter(torch.log(torch.tensor(float(c))))
        if hyd_init is None:
            hyd_init = torch.tensor([[0, 0, 0], [0.06, 0, 0], [0, 0.06, 0], [0.06, 0.06, 0]]) + torch.tensor([5.0, 6.0, -1.5])
        self.hyd = nn.Parameter(torch.as_tensor(hyd_init, dtype=torch.float32).clone())
        self.ch_gain = nn.Parameter(torch.zeros(n_hyd))      # log gain
        self.ch_delay_us = nn.Parameter(torch.zeros(n_hyd))  # microseconds (keeps lr sane)
        self.walls = nn.ModuleDict({
            "x0": WallFilter(wall_r0), "x1": WallFilter(wall_r0),
            "y0": WallFilter(wall_r0), "y1": WallFilter(wall_r0),
            "floor": WallFilter(floor_r0),
        })
        # source response: log-magnitude on a coarse frequency grid, interpolated
        self.src_logmag_knots = nn.Parameter(torch.zeros(16))
        self.images = image_sources(order)
        self.surface_coef = -1.0  # pressure-release surface; set to +r for validation vs pyroomacoustics
        self.register_buffer("f", torch.fft.rfftfreq(n_fft, 1.0 / fs))

    @property
    def c(self):
        return torch.exp(self.log_c)

    def source_response(self):
        knots = self.src_logmag_knots
        x = torch.linspace(0, 1, knots.numel(), device=knots.device)
        fx = self.f / self.f[-1]
        idx = torch.clamp((fx * (knots.numel() - 1)).long(), 0, knots.numel() - 2)
        w = fx * (knots.numel() - 1) - idx
        lm = knots[idx] * (1 - w) + knots[idx + 1] * w
        return torch.exp(lm)

    def transfer(self, src, max_delay=None):
        """Return H: (n_hyd, n_freq) complex transfer functions for source at `src` (3,).
        max_delay (s): drop images arriving later than this (early-reflection window).
        """
        f = self.f
        H = torch.zeros(self.hyd.shape[0], f.numel(), dtype=torch.complex64, device=f.device)
        srcr = self.source_response()
        wall_R = {k: m(f) for k, m in self.walls.items()}
        for n in self.images:
            pos = image_position(src, n, self.L, self.depth)          # (3,)
            d = torch.linalg.norm(self.hyd - pos, dim=-1)              # (n_hyd,)
            tau = d / self.c + self.ch_delay_us * 1e-6                  # (n_hyd,)
            if max_delay is not None and bool((tau.min() > max_delay).item()):
                continue
            counts = wall_counts(n)
            R = torch.ones_like(f, dtype=torch.complex64)
            for k, cnt in counts.items():
                if cnt == 0:
                    continue
                if k == "surface":
                    R = R * (self.surface_coef ** cnt)
                else:
                    R = R * wall_R[k] ** cnt
            amp = torch.exp(self.ch_gain)[:, None] / d[:, None]
            H = H + amp * R[None, :] * srcr[None, :] * torch.exp(-2j * math.pi * f[None, :] * tau[:, None])
        return H

    def transfer_batch(self, srcs, max_delay=None, chunk=256):
        """srcs: (N,3) -> H: (N, n_hyd, n_freq). Vectorized over candidates; loops over images."""
        outs = []
        f = self.f
        srcr = self.source_response()
        wall_R = {k: m(f) for k, m in self.walls.items()}
        Rs = []
        for n in self.images:
            R = torch.ones_like(f, dtype=torch.complex64)
            for k, cnt in wall_counts(n).items():
                if cnt == 0:
                    continue
                R = R * ((self.surface_coef ** cnt) if k == "surface" else wall_R[k] ** cnt)
            Rs.append(R * srcr)
        for i in range(0, srcs.shape[0], chunk):
            S = srcs[i:i + chunk]                                        # (n,3)
            H = torch.zeros(S.shape[0], self.hyd.shape[0], f.numel(), dtype=torch.complex64, device=f.device)
            for n, R in zip(self.images, Rs):
                pos = image_position(S, n, self.L, self.depth)          # (n,3)
                d = torch.linalg.norm(self.hyd[None] - pos[:, None], dim=-1)   # (n, n_hyd)
                tau = d / self.c + self.ch_delay_us[None] * 1e-6
                if max_delay is not None:
                    keep = (tau.min(dim=1).values <= max_delay).to(d.dtype)[:, None]  # per-candidate gate
                else:
                    keep = 1.0
                amp = torch.exp(self.ch_gain)[None] / d * keep
                H = H + amp[..., None] * R[None, None] * torch.exp(-2j * math.pi * f[None, None] * tau[..., None])
            outs.append(H)
        return torch.cat(outs)

    def render(self, src, probe, max_delay=None):
        """Convolve probe (n_fft-length or shorter, 1-D) with the rendered CIRs.
        Returns (n_hyd, n_fft) time signals (circular; keep probe + tail < n_fft).
        """
        P = torch.fft.rfft(probe, n=self.n_fft)
        H = self.transfer(src, max_delay)
        return torch.fft.irfft(H * P[None, :], n=self.n_fft)

    def cir(self, src, max_delay=None):
        return torch.fft.irfft(self.transfer(src, max_delay), n=self.n_fft)
