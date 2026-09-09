"""Matched-field style localizer using a (fitted) PoolRenderer as the replica.

Coarse-to-fine grid search over source position maximizing the normalized
correlation between predicted and observed multichannel spectra. Plain torch,
batched over candidate positions so it runs on GPU (Orin) or CPU.
"""
import torch


def _replica_batch(model, cands, max_delay):
    return model.transfer_batch(cands, max_delay)


def _envelope(X, n_fft):
    """X: (..., F) one-sided spectrum -> analytic-signal magnitude (..., n_fft)."""
    Z = torch.zeros(*X.shape[:-1], n_fft, dtype=X.dtype, device=X.device)
    Z[..., :X.shape[-1]] = X
    Z[..., 1:X.shape[-1] - 1] *= 2
    return torch.fft.ifft(Z, n=n_fft).abs()


def match_score(model, obs_spec, cands, max_delay=None, band=(1500.0, 18000.0), mode="coherent"):
    """obs_spec: (C, F) complex observed spectrum (rfft of windowed recording deconvolved
    by the probe). mode: "coherent" = normalized complex inner product across channels and
    band (sharp, needs fine grid); "envelope" = normalized correlation of band-limited CIR
    envelopes (smooth, phase-free, good for the coarse level). Returns (N,) scores."""
    f = model.f
    m = ((f >= band[0]) & (f <= band[1])).to(obs_spec.dtype)
    O = obs_spec * m
    H = _replica_batch(model, cands, max_delay) * m
    if mode == "envelope":
        Oe = _envelope(O, model.n_fft)
        He = _envelope(H, model.n_fft)
        num = (He * Oe[None]).sum(dim=(1, 2))
        den = torch.linalg.norm(He.flatten(1), dim=1) * torch.linalg.norm(Oe) + 1e-9
        return num / den
    num = (H.conj() * O[None]).sum(dim=(1, 2)).abs()
    den = torch.linalg.norm(H.flatten(1), dim=1) * torch.linalg.norm(O) + 1e-9
    return num / den


DEFAULT_LEVELS = (
    (0.25, (2500.0, 17500.0), "envelope"),   # coarse: full-band envelope, phase-free
    (0.05, (2500.0, 12000.0), "coherent"),
    (0.01, (2500.0, 17500.0), "coherent"),
)


@torch.no_grad()
def localize(model, obs_spec, bounds, levels=DEFAULT_LEVELS, max_delay=None, topk=5):
    """Coarse-to-fine in space AND frequency: each level = (grid step m, (f_lo, f_hi), mode).
    Grid step must stay below ~lambda_min/2 of that level's band to keep the coherent
    score smooth enough to not skip the peak. Keeps top-k coarse candidates alive.
    bounds: ((xmin,xmax),(ymin,ymax),(zmin,zmax)). Returns best (3,) position and score."""
    dev = model.f.device
    (x0, x1), (y0, y1), (z0, z1) = bounds
    step, band, mode = levels[0]
    gx = torch.arange(x0, x1 + 1e-6, step, device=dev)
    gy = torch.arange(y0, y1 + 1e-6, step, device=dev)
    gz = torch.arange(z0, z1 + 1e-6, step, device=dev)
    cands = torch.cartesian_prod(gx, gy, gz)
    s = match_score(model, obs_spec, cands, max_delay, band, mode)
    seeds = cands[s.topk(min(topk, s.numel())).indices]
    prev = step
    for step, band, mode in levels[1:]:
        span = prev
        loc = torch.cartesian_prod(*[torch.arange(-span, span + 1e-6, step, device=dev)] * 3)
        cands = (seeds[:, None, :] + loc[None]).reshape(-1, 3)
        s = match_score(model, obs_spec, cands, max_delay, band, mode)
        seeds = cands[s.topk(min(topk, s.numel())).indices]
        prev = step
    return seeds[0], s.max()
