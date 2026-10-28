"""Late reverberation tail, octave filterbank and time-variation helpers.

Implements RESEARCH_PLAN.md section 4.2:

* row 7  (late tail): per-octave Gaussian noise with a single-slope envelope
  exp(-6.91 t / T60_b) or a two-slope envelope (sum of two decays with their own
  levels and T60s), energy-matched to the early ISM response at t_e and joined to it
  with a 5 ms crossfade (section 4.1, "tail_late(t) for t > t_e");
* row 14 (time variation): resampling by (1 + eps), |eps| <= 1e-3, and per-band IR
  gain perturbation with a per-band SD (to be set from the measured sweep-to-sweep
  statistics of E3; the defaults here are uncalibrated).

Octave filterbank
-----------------
:func:`octave_filterbank` is a zero-phase FFT-domain bank whose band magnitudes are
differences of smooth log-frequency steps,

    W_0 = 1 - s_0,   W_b = s_{b-1} - s_b,   W_{B-1} = s_{B-2},

where s_k(f) rises from 0 to 1 with a sin^2 ramp over +-``transition`` octaves around
the k-th nominal band edge (centre_k * sqrt 2).  The weights telescope, so
sum_b W_b(f) = 1 at every frequency: the bank reconstructs its input exactly (to FFT
round-off).  The outer bands absorb DC .. first edge and last edge .. Nyquist.  The
bank is non-causal (zero phase); inputs are zero-padded on both sides before the FFT
so the two-sided band impulse responses do not wrap.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from .physics import octave_bands

__all__ = [
    "octave_weights",
    "octave_filterbank",
    "decay_envelope",
    "late_tail",
    "splice_tail",
    "apply_time_scale",
    "perturb_ir_bands",
]

_LN1000 = 3.0 * np.log(10.0)  # 6.91: amplitude decays by 60 dB in T60


def _centres(centres: np.ndarray | None) -> np.ndarray:
    return octave_bands()[0] if centres is None else np.asarray(centres, dtype=np.float64)


def octave_weights(freqs: np.ndarray, centres: np.ndarray | None = None, transition: float = 0.25) -> np.ndarray:
    """Zero-phase band magnitudes W_b(f), shape (B, len(freqs)); sum over b is 1."""
    c = _centres(centres)
    if not 0 < transition <= 0.5:
        raise ValueError("transition (half-width, octaves) must be in (0, 0.5]")
    B = len(c)
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-12)
    lf = np.log2(f)
    steps = []
    for k in range(B - 1):
        edge = np.log2(c[k] * np.sqrt(2.0))
        u = np.clip((lf - (edge - transition)) / (2.0 * transition), 0.0, 1.0)
        steps.append(np.sin(0.5 * np.pi * u) ** 2)
    s = [np.ones_like(f)] + steps + [np.zeros_like(f)]
    return np.stack([s[b] - s[b + 1] for b in range(B)])


def octave_filterbank(
    x: np.ndarray,
    fs: int,
    centres: np.ndarray | None = None,
    transition: float = 0.25,
    pad: float = 0.05,
) -> np.ndarray:
    """Split ``x`` (..., N) into octave bands, returning (B, ..., N).

    Zero-phase, perfect reconstruction: ``octave_filterbank(x, fs).sum(0) == x``.
    ``pad`` seconds of zeros are added on each side before filtering.
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[-1]
    npad = int(round(pad * fs))
    nfft = next_fast_len(N + 2 * npad, real=True)
    X = rfft(x, n=nfft, axis=-1)
    # Pre-rotating by npad samples is equivalent to placing x at offset npad.
    k = np.arange(X.shape[-1])
    X = X * np.exp(-2j * np.pi * k * npad / nfft)
    W = octave_weights(np.fft.rfftfreq(nfft, 1.0 / fs), centres, transition)
    W = W.reshape((W.shape[0],) + (1,) * (x.ndim - 1) + (W.shape[1],))
    y = irfft(X[None] * W, n=nfft, axis=-1)
    return y[..., npad : npad + N]


def _match_band_amplitudes(
    p_target: np.ndarray, fs: int, centres: np.ndarray, transition: float, n_grid: int = 1 << 13, n_iter: int = 60
) -> np.ndarray:
    """Per-band synthesis RMS amplitudes A_b whose summed tail, analysed with the same
    bank, has expected band powers ``p_target``.

    :func:`late_tail` filters ONE white noise into bands, normalises band b to unit RMS
    and scales it by A_b, so the tail spectrum is |N(f)|^2 G(f)^2 with
    G = sum_b A_b W_b / sigma_b, sigma_b^2 = mean_f W_b^2.  Neighbouring bands overlap
    by +-transition octaves and add *coherently*, so the analysed power
    q_b = mean_f W_b^2 G^2 is not A_b^2 (valley bands read up to a few dB high).
    Solved by the multiplicative fixed point A_b <- A_b sqrt(p_b / q_b(A)), which
    converges because the overlap is small (the system is diagonally dominant).
    """
    f = np.linspace(0.0, fs / 2.0, n_grid)
    W = octave_weights(f, centres, transition)
    sig = np.sqrt(np.maximum((W**2).mean(axis=1), 1e-300))
    Wn = W / sig[:, None]
    W2 = W**2
    p = np.maximum(np.asarray(p_target, dtype=np.float64), 0.0)
    A = np.sqrt(p)
    for _ in range(n_iter):
        q = (W2 * (A @ Wn) ** 2).mean(axis=1)
        A = A * np.sqrt(np.where(q > 0, p / np.maximum(q, 1e-300), 0.0))
    return A


def _two_slope_params(t60: np.ndarray, B: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise ``t60`` to (T1, T2, rel_level_db) arrays of length B."""
    t60 = np.asarray(t60, dtype=np.float64)
    if t60.ndim == 0:
        t60 = np.full(B, float(t60))
    if t60.ndim == 1:
        if len(t60) != B:
            raise ValueError(f"t60 has {len(t60)} bands, expected {B}")
        return t60, t60, np.full(B, -np.inf)
    if t60.shape != (B, 3):
        raise ValueError("two-slope t60 must have shape (B, 3): [T60_1, T60_2, level2_dB]")
    return t60[:, 0], t60[:, 1], t60[:, 2]


def decay_envelope(t: np.ndarray, t60_1: float, t60_2: float | None = None, level2_db: float = -np.inf) -> np.ndarray:
    """Power envelope exp(-13.8 t/T1) + 10^(L2/10) exp(-13.8 t/T2), t in seconds.

    Both terms are referenced to t = 0 (the splice time t_e).  ``level2_db`` is the
    level of the second (usually slower) slope relative to the first at t = 0; -inf
    gives the single-slope model.
    """
    p = np.exp(-2.0 * _LN1000 * t / t60_1)
    if t60_2 is not None and np.isfinite(level2_db):
        p = p + 10.0 ** (level2_db / 10.0) * np.exp(-2.0 * _LN1000 * t / t60_2)
    return p


def late_tail(
    fs: int,
    length: int,
    t60: np.ndarray | float,
    rng: np.random.Generator,
    centres: np.ndarray | None = None,
    levels_db: np.ndarray | float = 0.0,
    t0: float = 0.0,
    transition: float = 0.25,
    return_bands: bool = False,
) -> np.ndarray:
    """Synthetic late tail of ``length`` samples.

    Band b is unit-RMS octave-filtered Gaussian noise times sqrt(decay_envelope(t - t0))
    times 10^(levels_db[b]/20), so the band's expected power at t = t0 is
    10^(levels_db[b]/10).  ``t60`` is (B,) for single slope or (B, 3) =
    [T60_1, T60_2, level2_dB] for two slopes.  Returns (length,) or (B, length).
    """
    c = _centres(centres)
    B = len(c)
    T1, T2, L2 = _two_slope_params(t60, B)
    lv = np.broadcast_to(np.asarray(levels_db, dtype=np.float64), (B,))
    noise = rng.standard_normal(length)
    bands = octave_filterbank(noise, fs, c, transition)
    bands /= np.sqrt(np.mean(bands**2, axis=-1, keepdims=True)) + 1e-30
    t = np.arange(length) / fs - t0
    for b in range(B):
        env = decay_envelope(t, T1[b], T2[b], L2[b])
        bands[b] *= np.sqrt(env) * 10.0 ** (lv[b] / 20.0)
    return bands if return_bands else bands.sum(0)


def splice_tail(
    early: np.ndarray,
    fs: int,
    t_e: float,
    t60: np.ndarray | float,
    rng: np.random.Generator,
    length: int,
    centres: np.ndarray | None = None,
    crossfade: float = 0.005,
    match_window: float = 0.010,
    transition: float = 0.25,
) -> np.ndarray:
    """Join an early ISM response to a synthetic late tail, returning ``length`` samples.

    Parameters
    ----------
    early : (N,) early response (ISM, images with delay <= t_e).
    t_e : splice time, s (RESEARCH_PLAN 4.2 row 6: 40 ms, randomized 30-50 ms).
    t60 : (B,) single-slope T60 per band, or (B, 3) two-slope [T60_1, T60_2, level2_dB].
    crossfade : length of the power-complementary (sin / cos) crossfade, s.  It occupies
        [t_e - crossfade, t_e], i.e. it ends at t_e where the ISM stops, so the early
        response is never faded in where it has already been truncated.
    match_window : the tail levels are set so that the expected power of the tail,
        analysed with the same octave bank, over [t_e - match_window, t_e] equals the
        early response's mean power there in every band (the "energy matched to the ISM
        EDC at t_e" of row 7).  Adjacent-band leakage of the overlapping bank is
        compensated by :func:`_match_band_amplitudes`.
        For two slopes the relative level of the second slope is kept and the overall
        level matched.

    Samples before t_e - crossfade are the early response unchanged.
    """
    c = _centres(centres)
    B = len(c)
    T1, T2, L2 = _two_slope_params(t60, B)
    n_e = int(round(t_e * fs))
    n_x = max(int(round(crossfade * fs)), 1)
    n_m = max(int(round(match_window * fs)), 1)
    if n_e - max(n_x, n_m) < 0:
        raise ValueError("t_e must exceed crossfade and match_window")
    if n_e >= length:
        raise ValueError("length must exceed t_e * fs")

    e = np.zeros(max(length, len(early)))
    e[: len(early)] = early
    e = e[:length]

    # Early band powers in the matching window (filter a slice with margin, not all of e).
    lo = max(n_e - n_m - int(0.02 * fs), 0)
    seg = octave_filterbank(e[lo : n_e + int(0.02 * fs)], fs, c, transition)
    p_early = np.mean(seg[:, n_e - n_m - lo : n_e - lo] ** 2, axis=-1)

    # Expected tail power over the same window per unit level, from the envelope.
    tw = (np.arange(n_e - n_m, n_e) - n_e) / fs
    p_unit = np.array([np.mean(decay_envelope(tw, T1[b], T2[b], L2[b])) for b in range(B)])
    # Overlapping bands leak into each other on re-analysis: solve for the synthesis
    # amplitudes over the window, then refer them to t_e via the envelope.
    A = _match_band_amplitudes(p_early, fs, c, transition)
    x = A**2
    levels_db = 10.0 * np.log10(np.maximum(x, 1e-300) / p_unit)
    levels_db[x <= 0] = -600.0

    n0 = n_e - n_x
    tail = late_tail(fs, length - n0, t60, rng, c, levels_db, t0=n_x / fs, transition=transition)

    u = (np.arange(n_x) + 0.5) / n_x
    out = e.copy()
    out[n0:n_e] = e[n0:n_e] * np.cos(0.5 * np.pi * u) + tail[:n_x] * np.sin(0.5 * np.pi * u)
    out[n_e:] = tail[n_x:]
    return out


# --------------------------------------------------------------------------- row 14


def apply_time_scale(x: np.ndarray, eps: float, half_taps: int = 32, max_eps: float = 1e-3) -> np.ndarray:
    """Resample ``x`` (..., N) by (1 + eps): y[n] = x(n (1 + eps)).

    eps > 0 compresses the signal (shorter output, all frequencies scaled up by 1 + eps,
    as for a closing range); eps < 0 stretches it.  Band-limited interpolation with a
    Kaiser-windowed sinc of 2 * ``half_taps`` taps (beta = 8, cutoff 0.95 Nyquist when
    compressing, so the top ~5 % of the band is attenuated).
    Output length floor((N - 1) / (1 + eps)) + 1.
    """
    if abs(eps) > max_eps:
        raise ValueError(f"|eps| = {abs(eps):g} exceeds max_eps = {max_eps:g}")
    x = np.asarray(x, dtype=np.float64)
    if eps == 0.0:
        return x.copy()
    N = x.shape[-1]
    M = int(np.floor((N - 1) / (1.0 + eps))) + 1
    fc = 0.95 / (1.0 + max(eps, 0.0))  # cutoff, fraction of Nyquist
    K = half_taps
    xp = np.concatenate([np.zeros(x.shape[:-1] + (K,)), x, np.zeros(x.shape[:-1] + (K + 1,))], axis=-1)
    win_beta = 8.0
    y = np.empty(x.shape[:-1] + (M,))
    chunk = 16384
    offs = np.arange(-K + 1, K + 1)
    for s in range(0, M, chunk):
        n = np.arange(s, min(s + chunk, M))
        pos = n * (1.0 + eps)
        base = np.floor(pos).astype(np.int64)
        frac = pos - base
        d = offs[None, :] - frac[:, None]  # tap offsets relative to the exact position
        # Kaiser window evaluated at the fractional tap offsets.
        w = np.i0(win_beta * np.sqrt(np.clip(1.0 - (d / K) ** 2, 0.0, 1.0))) / np.i0(win_beta)
        h = fc * np.sinc(fc * d) * w
        idx = base[:, None] + offs[None, :] + K
        y[..., s : s + len(n)] = np.einsum("...nk,nk->...n", xp[..., idx], h)
    return y


def perturb_ir_bands(
    rir: np.ndarray,
    fs: int,
    sd_db: np.ndarray | float,
    rng: np.random.Generator,
    centres: np.ndarray | None = None,
    transition: float = 0.25,
) -> np.ndarray:
    """Apply an independent N(0, sd_db[b]^2) dB gain to each octave band of ``rir``.

    Models sweep-to-sweep IR variability (row 14).  ``sd_db`` must come from the
    measured E3 statistics; there is no calibrated default.  sd_db = 0 returns ``rir``
    (perfect-reconstruction bank).
    """
    c = _centres(centres)
    sd = np.broadcast_to(np.asarray(sd_db, dtype=np.float64), (len(c),))
    g = 10.0 ** (rng.standard_normal(len(c)) * sd / 20.0)
    bands = octave_filterbank(rir, fs, c, transition)
    return np.tensordot(g, bands, axes=(0, 0))
