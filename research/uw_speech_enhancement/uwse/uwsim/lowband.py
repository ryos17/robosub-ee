"""Low-frequency band (f < f_x): Linkwitz-Riley crossover, measured-IR splice, modal sum.

Implements RESEARCH_PLAN.md section 4.2 rows 8 and 9 (section 4.1, "low band (f < f_x):
measured-IR splice (calibrated pool) | modal sum (generic pools)"):

* f_x = max(measured cutoff f_s, 500 Hz)  -> :func:`crossover_frequency`;
* LR4 crossover at f_x                    -> :func:`lr4_split`;
* splice of a low-passed IR into a high-passed simulated IR -> :func:`splice_lowband`;
* modal sum for a shoebox with a pressure-release top and rigid walls and floor
  -> :func:`shoebox_modes`, :func:`modal_rir`.

There is no separate high-pass for the waveguide cutoff (row 9): it emerges from the
modal sum, whose lowest eigenfrequency is c / (4 H) when L, W >> H.

Coordinates for the modal model: x in [0, L], y in [0, W], z = depth below the free
surface in [0, H] (z = 0 surface, z = H floor).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len, rfft
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

__all__ = [
    "crossover_frequency",
    "lr4_split",
    "splice_lowband",
    "shoebox_modes",
    "modal_rir",
]


def crossover_frequency(measured_cutoff: float | None = None, floor: float = 500.0) -> float:
    """f_x = max(measured cutoff, 500 Hz) (RESEARCH_PLAN 4.2 row 8)."""
    return float(floor if measured_cutoff is None else max(measured_cutoff, floor))


def lr4_split(x: np.ndarray, fs: int, f_x: float, zero_phase: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Fourth-order Linkwitz-Riley split of ``x`` (..., N) at ``f_x``; returns (low, high).

    zero_phase=True (default): forward-backward filtering (filtfilt) of a 2nd-order
    Butterworth low/high-pass.  The magnitudes are |H_BW2|^2, i.e. exactly the LR4
    magnitudes, with zero phase; because a Butterworth pair is power complementary,
    |H_lp|^2 + |H_hp|^2 = 1, so low + high == x (flat magnitude, zero phase, up to
    edge effects, which are removed by zero-padding 10 / f_x seconds on each side).
    The price is non-causality: a little pre-ringing of the low band before each
    arrival (about 1 / f_x).

    zero_phase=False: causal LR4 (each BW2 section applied twice); low + high is then
    an allpass of x (flat magnitude, 2nd-order phase rotation around f_x).
    """
    x = np.asarray(x, dtype=np.float64)
    if not 0 < f_x < fs / 2:
        raise ValueError("need 0 < f_x < fs/2")
    lp = butter(2, f_x, btype="low", fs=fs, output="sos")
    hp = butter(2, f_x, btype="high", fs=fs, output="sos")
    if zero_phase:
        npad = int(np.ceil(10.0 * fs / f_x))
        widths = [(0, 0)] * (x.ndim - 1) + [(npad, npad)]
        xp = np.pad(x, widths)
        lo = sosfiltfilt(lp, xp, axis=-1, padtype=None)[..., npad:-npad]
        hi = sosfiltfilt(hp, xp, axis=-1, padtype=None)[..., npad:-npad]
    else:
        lo = sosfilt(lp, sosfilt(lp, x, axis=-1), axis=-1)
        hi = sosfilt(hp, sosfilt(hp, x, axis=-1), axis=-1)
    return lo, hi


def _band_energy(h: np.ndarray, fs: int, f1: float, f2: float) -> float:
    n = next_fast_len(len(h), real=True)
    H = rfft(h, n=n)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    sel = (f >= f1) & (f <= f2)
    return float(np.sum(np.abs(H[sel]) ** 2))


def splice_lowband(
    rir_high_source: np.ndarray,
    rir_low_source: np.ndarray,
    fs: int,
    f_x: float,
    match_level: bool = False,
    zero_phase: bool = True,
) -> np.ndarray:
    """HP_LR4(rir_high_source) + LP_LR4(rir_low_source), crossover at ``f_x``.

    ``rir_high_source`` is the simulated ISM + tail IR; ``rir_low_source`` is the
    measured IR of the nearest calibration position (calibrated pool) or
    :func:`modal_rir` (generic pools).  Both must be on the same time axis (direct
    arrival at the same sample) and amplitude scale; the output length is the longer
    of the two.  ``match_level=True`` instead rescales the low source so its energy
    in [f_x / 2, 2 f_x] equals the high source's, for low sources of unknown scale.
    """
    n = max(len(rir_high_source), len(rir_low_source))
    a = np.zeros(n)
    b = np.zeros(n)
    a[: len(rir_high_source)] = rir_high_source
    b[: len(rir_low_source)] = rir_low_source
    if match_level:
        eb = _band_energy(b, fs, f_x / 2, 2 * f_x)
        if eb > 0:
            b *= np.sqrt(_band_energy(a, fs, f_x / 2, 2 * f_x) / eb)
    _, hi = lr4_split(a, fs, f_x, zero_phase)
    lo, _ = lr4_split(b, fs, f_x, zero_phase)
    return lo + hi


def shoebox_modes(L: float, W: float, H: float, c: float, f_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Eigenfrequencies <= f_max of the pressure-release-top, rigid-sided shoebox.

    f_lmn = (c / 2) sqrt((l/L)^2 + (m/W)^2 + ((2n+1)/(2H))^2),  l, m, n >= 0,
    eigenfunctions cos(l pi x / L) cos(m pi y / W) sin((2n+1) pi z / (2H)).
    Returns ``(freqs, idx)`` sorted by frequency, idx of shape (K, 3) = (l, m, n).
    """
    kmax = 2.0 * f_max / c  # bound on sqrt(...) term
    l = np.arange(int(np.floor(kmax * L)) + 1)
    m = np.arange(int(np.floor(kmax * W)) + 1)
    nmax = int(np.floor(kmax * H - 0.5))
    if nmax < 0:
        return np.zeros(0), np.zeros((0, 3), dtype=np.int64)
    n = np.arange(nmax + 1)
    lg, mg, ng = np.meshgrid(l, m, n, indexing="ij")
    f = 0.5 * c * np.sqrt((lg / L) ** 2 + (mg / W) ** 2 + ((2 * ng + 1) / (2 * H)) ** 2)
    sel = f <= f_max
    f, idx = f[sel], np.stack([lg[sel], mg[sel], ng[sel]], axis=1)
    order = np.argsort(f, kind="stable")
    return f[order], idx[order]


def modal_rir(
    src: tuple[float, float, float],
    rx: tuple[float, float, float],
    L: float,
    W: float,
    H: float,
    c: float,
    fs: int,
    length: int,
    f_max: float = 1500.0,
    damping: tuple[float, ...] | np.ndarray = (1.0, 0.8),
    damping_freqs: tuple[float, ...] | np.ndarray = (250.0, 500.0),
    chunk: int = 128,
    spreading: str = "1/r",
    oversample: float | None = 4.0,
) -> np.ndarray:
    """Modal-sum impulse response of a shoebox pool, ``length`` samples at ``fs``.

    h[k] = (g/fs) (c^2 / V) sum_N psi_N(src) psi_N(rx) / Lambda_N
                              * sin(w_N t) / w_N * exp(-delta_N t),   t = k / fs,

    the modal expansion of the Green's function of
    laplacian(G) - G_tt / c^2 = -delta(r - r_s) delta(t), whose free-field form is
    delta(t - r/c) / (4 pi r).  With the 1/fs factor a free-field arrival becomes one
    sample of height g / (4 pi r) (band-limited), i.e. the normalisation of an
    image-source model.  ``spreading`` sets g:

    * ``"1/r"`` (default): g = 4 pi, one sample of height 1 / r per image, the
      convention of :mod:`uwse.uwsim.ism` (A_i = R / r_i, pressure re the 1 m level),
      so the result can be spliced into an ISM + tail IR without a level fit;
    * ``"4pi"``: g = 1, the Green's-function normalisation 1 / (4 pi r).

    Lambda_N = (1/V) int psi_N^2 dV = e_l e_m / 2 with e_0 = 1, e_{>0} = 1/2.

    Damping: delta_N = 3 ln(10) / T60(f_N), T60 interpolated linearly in log f between
    ``damping`` values given at ``damping_freqs`` (default: T60 at 250 and 500 Hz;
    held constant outside).  The defaults (1.0 s, 0.8 s) are uncalibrated
    placeholders; pass the fitted 250/500 Hz decays (row 8).

    Evaluation: every mode is band-limited to <= f_max, so with ``oversample`` set the
    sum is evaluated at the internal rate fs / D, D = max(1, floor(fs / (oversample
    f_max))), and brought back to ``fs`` by polyphase interpolation (scipy
    ``resample_poly``; its anti-imaging filter passes <= ~0.8 x the internal Nyquist,
    which is >= 1.6 f_max for oversample = 4).  This cuts the cost by ~D (D = 12 for a
    48 kHz IR with f_max = 1 kHz).  ``oversample=None`` evaluates directly at ``fs``.

    The sum is truncated at ``f_max`` (Gibbs ringing there); use it only below a
    crossover f_x well under f_max (e.g. f_max >= 2 f_x) via :func:`splice_lowband`.
    Positions are (x, y, z) with z the depth below the surface.
    """
    for p in (src, rx):
        if not (0 <= p[0] <= L and 0 <= p[1] <= W and 0 <= p[2] <= H):
            raise ValueError(f"position {p} outside the {L} x {W} x {H} m box")
    if spreading == "1/r":
        g = 4.0 * np.pi
    elif spreading == "4pi":
        g = 1.0
    else:
        raise ValueError(f"spreading must be '1/r' or '4pi', got {spreading!r}")
    df = np.asarray(damping_freqs, dtype=np.float64)
    if df.size > 1 and np.any(np.diff(df) <= 0):
        raise ValueError("damping_freqs must be strictly increasing")
    freqs, idx = shoebox_modes(L, W, H, c, f_max)
    h = np.zeros(length)
    if len(freqs) == 0 or length == 0:
        return h
    D = 1 if oversample is None else max(1, int(np.floor(fs / (oversample * f_max))))
    # Internal grid: enough samples that the interpolated output covers `length`,
    # plus a margin so the interpolation filter's end transient falls outside it.
    margin = 0 if D == 1 else 16
    n_int = -(-length // D) + margin
    t = np.arange(n_int) * (D / fs)
    V = L * W * H
    l, m, n = idx[:, 0], idx[:, 1], idx[:, 2]

    def psi(p):
        return (
            np.cos(l * np.pi * p[0] / L)
            * np.cos(m * np.pi * p[1] / W)
            * np.sin((2 * n + 1) * np.pi * p[2] / (2 * H))
        )

    lam = np.where(l == 0, 1.0, 0.5) * np.where(m == 0, 1.0, 0.5) * 0.5
    w = 2.0 * np.pi * freqs
    amp = g * psi(src) * psi(rx) / lam * (c**2 / V) / fs / w
    t60 = np.interp(np.log(freqs), np.log(df), np.asarray(damping, dtype=np.float64))
    delta = 3.0 * np.log(10.0) / t60
    # Keep each (modes x time) block around <= 4M complex values (RAM).
    step = max(1, min(chunk, int(4_000_000 // max(n_int, 1))))
    s_cplx = -delta + 1j * w
    h_int = np.zeros(n_int)
    for s in range(0, len(freqs), step):
        sl = slice(s, s + step)
        # Im(exp((-delta + j w) t)) = exp(-delta t) sin(w t): one complex exp per entry.
        h_int += (amp[sl] @ np.exp(np.outer(s_cplx[sl], t))).imag
    if D == 1:
        return h_int[:length]
    # Polyphase interpolation keeps the internal samples h(k D / fs) (up to the
    # filter's passband ripple) and fills in the band-limited values between them.
    # The modal IR starts from zero with zero value at t = 0 (sin), so treating
    # t < 0 as zero is consistent.
    return resample_poly(h_int, D, 1)[:length]
