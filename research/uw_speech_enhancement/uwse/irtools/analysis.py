"""Impulse-response analysis.

Implements the IR metrics of RESEARCH_PLAN §4.3 step 2 (octave filtering,
Schroeder integration with Lundeby truncation, single- vs two-slope decay fits
for H1a) and §4.4 (arrival times / amplitudes of the first resolved
reflections, DRR and C50, EDC curvature class and knee time, 1/3-octave
coloration distance, T20 / EDT as descriptive quantities), plus the per-band IR
SNR required by §6.4 step 5 / G2.

Conventions: IRs are 1-D float arrays; levels are dB; times are seconds.
Octave bands are base-2 (fc = 1000 * 2^k), edges fc * 2^(+-1/(2 b)).

References
----------
M. R. Schroeder, "New method of measuring reverberation time", JASA 37, 1965.
A. Lundeby, T. E. Vigran, H. Bietz, M. Vorlaender, "Uncertainties of
measurements in room acoustics", Acustica 81, 344-355, 1995.
N. Xiang, P. M. Goggans, "Evaluation of decay times in coupled spaces:
Bayesian parameter estimation", JASA 110(3), 2001 (multi-slope EDC model).
ISO 3382-1:2009 (EDT, T20, T30, C50 definitions; onset at -20 dB re peak).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.fft import next_fast_len
from scipy.optimize import least_squares
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

__all__ = [
    "OCTAVE_CENTERS",
    "band_edges",
    "band_filter",
    "filterbank",
    "onset_index",
    "LundebyResult",
    "lundeby",
    "EDC",
    "schroeder_edc",
    "decay_time",
    "decay_times",
    "band_decay_times",
    "SlopeFit",
    "fit_decay_slopes",
    "drr_db",
    "clarity_db",
    "c50_db",
    "ir_snr_db",
    "Arrivals",
    "detect_arrivals",
    "third_octave_centers",
    "third_octave_spectrum",
    "spectral_distance_db",
]

OCTAVE_CENTERS = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
_K60 = 6.0 * np.log(10.0)  # energy decays by 60 dB at t = T60: exp(-_K60 / T60 * t)


# --------------------------------------------------------------------------- #
# Filterbank
# --------------------------------------------------------------------------- #
def band_edges(fc: float, fraction: int = 1) -> tuple[float, float]:
    """Base-2 band edges fc 2^(-+1/(2 fraction))."""
    g = 2.0 ** (1.0 / (2 * fraction))
    return fc / g, fc * g


def band_filter(x: np.ndarray, fs: int, fc: float, fraction: int = 1, order: int = 4) -> np.ndarray:
    """Zero-phase (sosfiltfilt) Butterworth band-pass; effective order 2 x order.

    The upper edge is clipped to 0.95 x Nyquist (e.g. the 8 kHz octave at
    fs = 16 kHz becomes a high-pass from 5.66 kHz).
    """
    lo, hi = band_edges(fc, fraction)
    nyq = fs / 2
    if lo >= 0.95 * nyq:
        raise ValueError(f"band {fc} Hz is above Nyquist at fs={fs}")
    if hi >= 0.95 * nyq:
        sos = butter(order, lo, btype="highpass", fs=fs, output="sos")
    else:
        sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def filterbank(x: np.ndarray, fs: int, centers=OCTAVE_CENTERS, fraction: int = 1, order: int = 4) -> np.ndarray:
    """(B, N) zero-phase band outputs for the given centres (250 Hz - 8 kHz octaves by default)."""
    return np.stack([band_filter(x, fs, fc, fraction, order) for fc in centers])


# --------------------------------------------------------------------------- #
# Schroeder / Lundeby
# --------------------------------------------------------------------------- #
def onset_index(h: np.ndarray, threshold_db: float = -20.0) -> int:
    """First sample whose energy is within ``threshold_db`` of the peak (ISO 3382-1 A.3.4)."""
    e = np.asarray(h, dtype=np.float64) ** 2
    return int(np.argmax(e >= e.max() * 10 ** (threshold_db / 10)))


def _block_levels(e: np.ndarray, n_blk: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean energy (dB) of consecutive blocks and block-centre sample indices."""
    n_blk = max(int(n_blk), 1)
    nb = len(e) // n_blk
    if nb < 1:
        return np.array([len(e) / 2]), np.array([10 * np.log10(max(e.mean(), 1e-300))])
    m = e[: nb * n_blk].reshape(nb, n_blk).mean(1)
    return (np.arange(nb) + 0.5) * n_blk, 10 * np.log10(np.maximum(m, 1e-300))


@dataclass
class LundebyResult:
    """Lundeby (1995) truncation point and late-decay regression.

    n_cross : crosspoint (samples, re the start of the analysed signal).
    noise_power : mean background-noise energy per sample.
    slope_db : late decay slope (dB / sample); intercept_db at sample 0.
    converged : whether the crosspoint iteration converged.
    """

    n_cross: int
    noise_power: float
    slope_db: float
    intercept_db: float
    iterations: int
    converged: bool

    def compensation(self) -> float:
        """Energy of the decay beyond the crosspoint, assuming it continues
        exponentially: integral e_c exp(-k n) dn = e_c / k."""
        if self.slope_db >= 0:
            return 0.0
        k = -self.slope_db * np.log(10) / 10  # 1/sample
        e_c = 10 ** ((self.intercept_db + self.slope_db * self.n_cross) / 10)
        return float(e_c / k)


def _linfit(x, y):
    if len(x) < 2:
        return None
    b, a = np.polyfit(x, y, 1)
    return a, b


def lundeby(
    h: np.ndarray,
    fs: int,
    init_block_s: float = 0.01,
    blocks_per_10db: int = 5,
    noise_tail_frac: float = 0.1,
    max_iter: int = 10,
) -> LundebyResult:
    """Iterative noise-floor / crosspoint estimation (Lundeby et al. 1995).

    ``h`` must start at the onset. Steps (numbering of the paper):
    1 block-average h^2 (10 ms); 2 noise from the last 10 %; 3 regression from
    0 dB to 10 dB above noise; 4 crosspoint; 5-6 re-average with
    ``blocks_per_10db`` blocks per 10 dB of decay; 7 noise from the point where
    the decay line is 10 dB below the noise (but at least the last 10 %);
    8 late-decay regression from 25 to 5 dB above noise; 9 new crosspoint;
    iterate 7-9 until it moves by less than one block.
    """
    e = np.asarray(h, dtype=np.float64) ** 2
    N = len(e)
    n_tail = max(int(noise_tail_frac * N), 1)
    noise = max(e[-n_tail:].mean(), 1e-300)
    tb, lv = _block_levels(e, init_block_s * fs)
    noise_db = 10 * np.log10(noise)
    i_pk = int(np.argmax(lv))
    stop = np.nonzero(lv[i_pk:] < noise_db + 10)[0]
    i_end = i_pk + (stop[0] if len(stop) else len(lv) - i_pk)
    fit = _linfit(tb[i_pk:i_end], lv[i_pk:i_end])
    if fit is None or fit[1] >= 0:
        return LundebyResult(N, noise, 0.0, float(lv.max()), 0, False)
    a, b = fit
    n_c = int(np.clip((noise_db - a) / b, 1, N))
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        n_blk = max(int(round(-10.0 / b / blocks_per_10db)), 1)  # b in dB/sample (< 0)
        # noise from the point 10 dB below the crosspoint on the decay line
        n_ns = int(min(n_c + (-10.0 / b), N - n_tail))
        n_ns = max(n_ns, 0)
        noise = max(e[n_ns:].mean(), 1e-300)
        noise_db = 10 * np.log10(noise)
        tb, lv = _block_levels(e[:N], n_blk)
        top = min(lv.max(), noise_db + 25.0)
        sel = (lv <= top) & (lv >= noise_db + 5.0) & (tb < n_c)
        idx = np.nonzero(sel)[0]
        if len(idx) < 2:
            break
        fit = _linfit(tb[idx], lv[idx])
        if fit is None or fit[1] >= 0:
            break
        a, b = fit
        n_new = int(np.clip((noise_db - a) / b, 1, N))
        if abs(n_new - n_c) < n_blk:
            n_c = n_new
            converged = True
            break
        n_c = n_new
    return LundebyResult(int(n_c), float(noise), float(b), float(a), it, converged)


@dataclass
class EDC:
    """Energy decay curve; t[0] = 0 at the onset, edc_db[0] = 0."""

    t: np.ndarray
    edc_db: np.ndarray
    onset: int
    lundeby: LundebyResult | None


def schroeder_edc(
    h: np.ndarray,
    fs: int,
    use_lundeby: bool = True,
    compensate: bool = True,
    onset_db: float = -20.0,
    subtract_noise: bool = False,
) -> EDC:
    """Schroeder backward integration EDC(t) = int_t^{t_c} h^2 + C.

    With ``use_lundeby`` the integral is truncated at the Lundeby crosspoint
    t_c and (``compensate``) the energy C of the decay assumed to continue
    exponentially past t_c is added (Lundeby 1995). ``subtract_noise``
    additionally subtracts the estimated noise power from h^2 before
    integrating (Chu's method; off by default).
    """
    h = np.asarray(h, dtype=np.float64)
    n0 = onset_index(h, onset_db)
    x = h[n0:]
    e = x**2
    lr = None
    n_c = len(e)
    C = 0.0
    if use_lundeby:
        lr = lundeby(x, fs)
        n_c = max(lr.n_cross, 2)
        if compensate:
            C = lr.compensation()
        if subtract_noise:
            e = e - lr.noise_power
    edc = np.cumsum(e[:n_c][::-1])[::-1] + C
    edc = np.maximum(edc, 1e-300)
    return EDC(np.arange(n_c) / fs, 10 * np.log10(edc / edc[0]), n0, lr)


def decay_time(t: np.ndarray, edc_db: np.ndarray, start_db: float, end_db: float) -> float:
    """Reverberation time -60/slope of a line fitted to the EDC between
    ``start_db`` and ``end_db`` (e.g. -5/-25 for T20, 0/-10 for EDT).
    NaN if the EDC does not reach ``end_db``."""
    if edc_db.min() > end_db:
        return float("nan")
    i0 = int(np.argmax(edc_db <= start_db))
    i1 = int(np.argmax(edc_db <= end_db))
    if i1 - i0 < 2:
        return float("nan")
    b, _ = np.polyfit(t[i0 : i1 + 1], edc_db[i0 : i1 + 1], 1)
    return float(-60.0 / b) if b < 0 else float("nan")


def decay_times(h: np.ndarray, fs: int, **edc_kw) -> dict[str, float]:
    """EDT (0/-10 dB), T20 (-5/-25 dB), T30 (-5/-35 dB) of one (band-limited) IR."""
    d = schroeder_edc(h, fs, **edc_kw)
    return {
        "EDT": decay_time(d.t, d.edc_db, 0.0, -10.0),
        "T20": decay_time(d.t, d.edc_db, -5.0, -25.0),
        "T30": decay_time(d.t, d.edc_db, -5.0, -35.0),
    }


def band_decay_times(h: np.ndarray, fs: int, centers=OCTAVE_CENTERS, **edc_kw) -> dict[str, np.ndarray]:
    """Per-octave EDT/T20/T30 (descriptive only per §4.4) and IR SNR."""
    centers = [c for c in centers if band_edges(c)[0] < 0.95 * fs / 2]
    bands = filterbank(h, fs, centers)
    out = {k: np.empty(len(centers)) for k in ("EDT", "T20", "T30")}
    for i, hb in enumerate(bands):
        d = decay_times(hb, fs, **edc_kw)
        for k in out:
            out[k][i] = d[k]
    out["centers"] = np.asarray(centers)
    out["snr_db"] = np.array([_snr_db(hb) for hb in bands])
    return out


# --------------------------------------------------------------------------- #
# Single- vs two-slope decay (H1a)
# --------------------------------------------------------------------------- #
@dataclass
class SlopeFit:
    """Result of ``fit_decay_slopes``.

    n_slopes : 1 or 2 (selected model).
    T60 : (T1,) or (T_early, T_late) of the selected model (s).
    levels_db : component EDC levels at t = 0 (dB re EDC(0)).
    knee_s : time where the two components' EDCs are equal (None if single).
    single, double : raw parameters of both fits [L, T] / [L1, T1, L2, T2].
    bic1, bic2 : BIC of both fits (lower is better).
    """

    n_slopes: int
    T60: tuple
    levels_db: tuple
    knee_s: float | None
    single: np.ndarray
    double: np.ndarray | None
    bic1: float
    bic2: float
    rms1_db: float
    rms2_db: float


def _two_slope_db(p, t):
    L1, lT1, L2, lT2 = p
    e = 10 ** (L1 / 10) * np.exp(-_K60 * t / np.exp(lT1)) + 10 ** (L2 / 10) * np.exp(-_K60 * t / np.exp(lT2))
    return 10 * np.log10(np.maximum(e, 1e-300))


def fit_decay_slopes(
    t: np.ndarray,
    edc_db: np.ndarray,
    floor_db: float = -60.0,
    tail_margin_db: float = 10.0,
    n_grid: int = 200,
    delta_bic: float = 10.0,
    min_ratio: float = 1.5,
    min_segment_db: float = 10.0,
) -> SlopeFit:
    """Fit single- and two-slope exponential models to an EDC and choose by BIC.

    Models (energy, Xiang & Goggans 2001 without the noise term, because the
    EDC is already Lundeby-truncated and compensated):
      1: E(t) = A exp(-K t / T)
      2: E(t) = A1 exp(-K t / T1) + A2 exp(-K t / T2),   K = 6 ln 10
    fitted by nonlinear least squares on the EDC in dB, on ``n_grid`` points
    uniform in t over the range EDC >= max(floor_db, EDC_end + tail_margin_db)
    (the last ``tail_margin_db`` before the crosspoint still contains noise
    energy and the compensation term, so it is excluded).

    BIC = n ln(RSS/n) + p ln n with p = 2 / 4. The grid size n is a modelling
    choice: EDC samples are strongly correlated, so BIC on raw samples would
    always pick the richer model; ``n_grid`` caps the effective count. The
    two-slope model is selected only if BIC2 < BIC1 - ``delta_bic`` AND
    T_late / T_early >= ``min_ratio`` AND the knee lies inside the fitted
    range AND each slope spans >= ``min_segment_db`` of decay inside the fitted
    range (60 knee / T_early and 60 (t_end - knee) / T_late). The last guard
    rejects a tiny late component that only absorbs the random EDC
    fluctuations near the end of the range; without it 4/10 single-decay
    250 Hz octave IRs were classified two-slope (review fix).
    """
    t = np.asarray(t, dtype=np.float64)
    edc_db = np.asarray(edc_db, dtype=np.float64)
    lim = max(floor_db, edc_db[-1] + tail_margin_db)
    use = np.nonzero(edc_db >= lim)[0]
    if len(use) < 8:
        raise ValueError("EDC too short for slope fitting")
    t_end = t[use[-1]]
    tg = np.linspace(0.0, t_end, n_grid)
    yg = np.interp(tg, t, edc_db)
    n = len(tg)

    b, a = np.polyfit(tg, yg, 1)
    T1s = -60.0 / b if b < 0 else np.inf
    rss1 = float(np.sum((a + b * tg - yg) ** 2))
    bic1 = n * np.log(max(rss1, 1e-12) / n) + 2 * np.log(n)

    # initial guesses: early slope from the first 30 %, late from the last 40 %
    q1, q2 = int(0.3 * n), int(0.6 * n)
    be, ae = np.polyfit(tg[:q1], yg[:q1], 1)
    bl, al = np.polyfit(tg[q2:], yg[q2:], 1)
    Te0 = -60 / be if be < 0 else max(T1s, 0.01)
    Tl0 = -60 / bl if bl < 0 else 2 * Te0
    if Tl0 <= Te0:
        Tl0 = 1.5 * Te0
    L20 = min(al, -3.0)
    L10 = 10 * np.log10(max(1 - 10 ** (L20 / 10), 1e-6))
    p0 = np.array([L10, np.log(Te0), L20, np.log(Tl0)])
    span = np.log(max(t_end, 1e-3))
    lo = [-40.0, np.log(1e-3), -120.0, np.log(1e-3)]
    hi = [10.0, span + 6, 10.0, span + 6]
    p0 = np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    res = least_squares(lambda p: _two_slope_db(p, tg) - yg, p0, bounds=(lo, hi), method="trf")
    p2 = res.x
    # order the components as (early = shorter T, late = longer T)
    if p2[1] > p2[3]:
        p2 = p2[[2, 3, 0, 1]]
    rss2 = float(np.sum(res.fun**2))
    bic2 = n * np.log(max(rss2, 1e-12) / n) + 4 * np.log(n)
    Te, Tl = np.exp(p2[1]), np.exp(p2[3])
    k1, k2 = _K60 / Te, _K60 / Tl
    knee = None
    if k1 > k2:
        knee = float((p2[0] - p2[2]) * np.log(10) / 10 / (k1 - k2))
    two = (
        bic2 < bic1 - delta_bic
        and Tl / Te >= min_ratio
        and knee is not None
        and 0.0 < knee < t_end
        and 60.0 * knee / Te >= min_segment_db
        and 60.0 * (t_end - knee) / Tl >= min_segment_db
    )
    rms1, rms2 = np.sqrt(rss1 / n), np.sqrt(rss2 / n)
    if two:
        return SlopeFit(2, (Te, Tl), (p2[0], p2[2]), knee, np.array([a, T1s]), p2, bic1, bic2, rms1, rms2)
    return SlopeFit(1, (T1s,), (a,), None, np.array([a, T1s]), p2, bic1, bic2, rms1, rms2)


# --------------------------------------------------------------------------- #
# DRR, clarity, SNR
# --------------------------------------------------------------------------- #
def drr_db(
    h: np.ndarray,
    fs: int,
    gate_s: float,
    t_first_reflection_s: float | None = None,
    band_lo_hz: float | None = None,
    min_cycles: float = 2.0,
    pre_s: float = 0.25e-3,
) -> float | None:
    """Direct-to-reverberant ratio with a direct-path time gate.

    Direct energy = h^2 summed over [t_d - pre_s, t_d + gate_s), t_d = peak of
    |h|; reverberant = everything after the gate. Returns None when the gate
    rule fails (§6.4 step 3, Appendix A "valid above 2-3/dt"):
      * the gate reaches the first reflection
        (``gate_s >= t_first_reflection_s``, delay re the direct path), or
      * the gate holds fewer than ``min_cycles`` periods of the lowest
        frequency analysed (``band_lo_hz * gate_s < min_cycles``).
    """
    if t_first_reflection_s is not None and gate_s >= t_first_reflection_s:
        return None
    if band_lo_hz is not None and band_lo_hz * gate_s < min_cycles:
        return None
    h = np.asarray(h, dtype=np.float64)
    nd = int(np.argmax(np.abs(h)))
    a = max(nd - int(round(pre_s * fs)), 0)
    b = nd + int(round(gate_s * fs))
    e_d = np.sum(h[a:b] ** 2)
    e_r = np.sum(h[b:] ** 2)
    if e_r <= 0:
        return float("inf")
    return float(10 * np.log10(e_d / e_r))


def clarity_db(h: np.ndarray, fs: int, t_split_s: float = 0.05, onset_db: float = -20.0) -> float:
    """C_t = 10 log10(E[0, t) / E[t, inf)), t = 0 at the ISO 3382 onset."""
    h = np.asarray(h, dtype=np.float64)
    n0 = onset_index(h, onset_db)
    ns = n0 + int(round(t_split_s * fs))
    e1, e2 = np.sum(h[n0:ns] ** 2), np.sum(h[ns:] ** 2)
    return float(10 * np.log10(e1 / e2)) if e2 > 0 else float("inf")


def c50_db(h: np.ndarray, fs: int, onset_db: float = -20.0) -> float:
    """C50 (dB)."""
    return clarity_db(h, fs, 0.05, onset_db)


def _snr_db(h: np.ndarray, noise_frac: float = 0.1) -> float:
    e = np.asarray(h, dtype=np.float64) ** 2
    n = max(int(noise_frac * len(e)), 1)
    return float(10 * np.log10(e.max() / max(e[-n:].mean(), 1e-300)))


def ir_snr_db(h: np.ndarray, fs: int, centers=OCTAVE_CENTERS, noise_frac: float = 0.1) -> np.ndarray:
    """Peak-to-late-noise ratio per band: 10 log10(max h_b^2 / mean of h_b^2 over
    the last ``noise_frac`` of the IR). ``centers=None``: broadband (scalar array).
    The IR must be long enough that its last part is noise only."""
    if centers is None:
        return np.array(_snr_db(h, noise_frac))
    centers = [c for c in centers if band_edges(c)[0] < 0.95 * fs / 2]
    return np.array([_snr_db(hb, noise_frac) for hb in filterbank(h, fs, centers)])


# --------------------------------------------------------------------------- #
# Resolved arrivals
# --------------------------------------------------------------------------- #
@dataclass
class Arrivals:
    """First resolved arrivals; index 0 is the direct path.

    times_s : absolute times in the IR; delays_s : re the direct path;
    levels_db : envelope peak level re the direct path.
    """

    times_s: np.ndarray
    delays_s: np.ndarray
    levels_db: np.ndarray


def detect_arrivals(
    h: np.ndarray,
    fs: int,
    n: int = 5,
    min_sep_s: float = 0.1e-3,
    threshold_db: float = -30.0,
    direct_db: float = -6.0,
    max_delay_s: float | None = None,
    smooth_s: float | None = None,
) -> Arrivals:
    """Direct path plus up to ``n`` resolved reflections by envelope peak picking.

    Envelope = |analytic signal| smoothed by a Hann window of ``smooth_s``
    (default ``min_sep_s / 2``; removes the 2/(pi n) Hilbert side-lobe ripple
    that would otherwise create spurious peaks). Peaks must be >= ``min_sep_s`` apart and above
    ``threshold_db`` re the global maximum. The direct path is the first peak
    within ``direct_db`` of the global maximum. Peak times / levels are refined
    by parabolic interpolation of the envelope (sub-sample). Returns n + 1
    arrivals at most. Resolution requires bandwidth >= ~2-3 / delay (§4.3).
    """
    h = np.asarray(h, dtype=np.float64)
    # odd-length (zero-delay) Hann smoother, at least 3 taps
    ns = max(2 * int(round((min_sep_s / 2 if smooth_s is None else smooth_s) * fs / 2)) + 1, 3)
    win = np.hanning(ns + 2)[1:-1]
    # Pad zeros in front (and use a linear, not circular, analytic signal) so an
    # arrival at the first sample keeps its two-sided envelope and is an
    # interior peak: find_peaks never reports edge samples, and a circular
    # Hilbert tail wraps onto the IR end (review fix: the direct path at
    # sample 0 was dropped and the first reflection became 'direct').
    npad = ns + 2
    hp = np.concatenate([np.zeros(npad), h])
    env = np.abs(hilbert(hp, N=next_fast_len(2 * len(hp)))[: len(hp)])
    env = np.convolve(env, win / win.sum(), mode="same")
    gmax = env.max()
    pk, _ = find_peaks(
        np.concatenate([env, [0.0]]),
        height=gmax * 10 ** (threshold_db / 20),
        distance=max(int(round(min_sep_s * fs)), 1),
    )
    if len(pk) == 0:
        pk = np.array([int(np.argmax(env))])
    i_d = int(np.argmax(env[pk] >= gmax * 10 ** (direct_db / 20)))
    pk = pk[i_d:]
    t_ref = pk[0] / fs
    if max_delay_s is not None:
        pk = pk[(pk / fs - t_ref) <= max_delay_s]
    pk = pk[: n + 1]
    tt, aa = [], []
    for p in pk:
        if 0 < p < len(env) - 1:
            y0, y1, y2 = env[p - 1], env[p], env[p + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            d = float(np.clip(d, -0.5, 0.5))
            amp = y1 - 0.25 * (y0 - y2) * d
        else:
            d, amp = 0.0, env[p]
        tt.append((p - npad + d) / fs)
        aa.append(amp)
    tt, aa = np.asarray(tt), np.asarray(aa)
    return Arrivals(tt, tt - tt[0], 20 * np.log10(aa / aa[0]))


# --------------------------------------------------------------------------- #
# Coloration
# --------------------------------------------------------------------------- #
def third_octave_centers(f_lo: float = 1000.0, f_hi: float = 8000.0) -> np.ndarray:
    """Base-2 1/3-octave centres 1000 2^(k/3) within [f_lo, f_hi]."""
    k = np.arange(np.ceil(3 * np.log2(f_lo / 1000) - 1e-9), np.floor(3 * np.log2(f_hi / 1000) + 1e-9) + 1)
    return 1000.0 * 2.0 ** (k / 3)


def third_octave_spectrum(h: np.ndarray, fs: int, f_lo: float = 1000.0, f_hi: float = 8000.0) -> tuple[np.ndarray, np.ndarray]:
    """1/3-octave band energy of an IR (dB re unit energy): by Parseval,
    E_band = (2 / nfft) sum_{f in band} |H_k|^2 (one-sided rfft bins), so the
    level does not depend on the zero-padded FFT length or the IR length
    (review fix: the raw bin sum grew by 10 log10 of the padding factor).

    The FFT is zero-padded to >= fs / 5 bins so the narrowest band holds
    enough bins. Bands whose upper edge exceeds Nyquist are dropped.
    """
    h = np.asarray(h, dtype=np.float64)
    nfft = next_fast_len(max(len(h), int(fs / 5)))
    P = 2.0 * np.abs(rfft(h, nfft)) ** 2 / nfft
    f = rfftfreq(nfft, 1 / fs)
    cs = [c for c in third_octave_centers(f_lo, f_hi) if band_edges(c, 3)[1] <= fs / 2]
    lv = []
    for c in cs:
        lo, hi = band_edges(c, 3)
        lv.append(10 * np.log10(max(P[(f >= lo) & (f < hi)].sum(), 1e-300)))
    return np.asarray(cs), np.asarray(lv)


def spectral_distance_db(
    h_a: np.ndarray, h_b: np.ndarray, fs: int, f_lo: float = 1000.0, f_hi: float = 8000.0, remove_gain: bool = True
) -> float:
    """RMS dB distance between the 1/3-octave spectra of two IRs (§4.4 coloration).

    ``remove_gain``: subtract the mean level difference first, so only the
    spectral shape (coloration) is compared, not the overall level.
    """
    _, la = third_octave_spectrum(h_a, fs, f_lo, f_hi)
    _, lb = third_octave_spectrum(h_b, fs, f_lo, f_hi)
    d = la - lb
    if remove_gain:
        d = d - d.mean()
    return float(np.sqrt(np.mean(d**2)))
