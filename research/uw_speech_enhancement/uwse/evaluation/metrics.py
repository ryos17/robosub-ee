"""Intrusive per-utterance metrics at 16 kHz: ESTOI, STOI, PESQ-WB, SI-SDR, CI-SDR.

Implements RESEARCH_PLAN.md §8.1 (metric list) and §8.2 (references R1/R0,
scoring rate 16 kHz, band [f_lo, 8 kHz]).

Band limiting of (E)STOI
------------------------
§8.1 scores ESTOI vs R1 "band-limited to [f_lo, 8 kHz]": the projector emits
nothing below f_lo, so content there is noise by construction. Both signals
are first band-pass filtered (zero-phase Butterworth). Filtering alone does
*not* remove the sub-f_lo bands from (E)STOI: the measure normalizes every
1/3-octave band envelope, so a band that is attenuated equally in both
signals scores exactly as before. The 1/3-octave bands whose centre lies
below ``f_lo`` (or whose upper edge exceeds ``f_hi``) are therefore also
excluded from the average (``drop_bands=True``, default). With
``f_lo <= 150 Hz`` and no filter this reduces exactly to ``pystoi``.

CI-SDR
------
Convolutive-transfer-function-invariant SDR (Boeddeker et al., arXiv
2011.15003): the reference is passed through the length-L FIR filter (L =
512 taps = 32 ms at 16 kHz) that best fits the estimate in the least-squares
sense, and the SDR is computed against that filtered reference. It absorbs
coloration but not the 0.25-1 s reverberant tail (§8.1).
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, fftconvolve, sosfiltfilt

__all__ = [
    "SCORE_FS",
    "bandlimit",
    "third_octave_mask",
    "stoi_banded",
    "estoi",
    "stoi",
    "pesq_wb",
    "si_sdr",
    "ci_sdr",
    "compute_metrics",
]

SCORE_FS = 16000
DEFAULT_F_LO = 500.0  # uncalibrated; set from the projector response (§3 "Band")
DEFAULT_F_HI = 8000.0
EPS = np.finfo(np.float64).eps


def _match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return np.asarray(a[:n], np.float64), np.asarray(b[:n], np.float64)


def bandlimit(x: np.ndarray, fs: int, f_lo: float = DEFAULT_F_LO, f_hi: float = DEFAULT_F_HI, order: int = 8) -> np.ndarray:
    """Zero-phase Butterworth band-pass to [f_lo, f_hi] (high-pass if f_hi >= 0.98*Nyquist).

    Zero phase (``sosfiltfilt``) so filtering introduces no lag between the
    estimate and the reference; the effective order is doubled.
    """
    x = np.asarray(x, np.float64)
    nyq = fs / 2.0
    hp = f_lo is not None and f_lo > 0
    lp = f_hi is not None and f_hi < 0.98 * nyq
    if hp and lp:
        sos = butter(order, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")
    elif hp:
        sos = butter(order, f_lo, btype="highpass", fs=fs, output="sos")
    elif lp:
        sos = butter(order, f_hi, btype="lowpass", fs=fs, output="sos")
    else:
        return x.copy()
    padlen = min(len(x) - 1, 3 * (2 * len(sos) + 1))
    return sosfiltfilt(sos, x, padlen=max(padlen, 0))


# ---------------------------------------------------------------------------
# (E)STOI
# ---------------------------------------------------------------------------
def third_octave_mask(f_lo: float | None, f_hi: float | None) -> np.ndarray:
    """Boolean mask over pystoi's 15 1/3-octave bands (centres 150 Hz * 2^(k/3)).

    Keeps bands with centre >= f_lo and upper edge <= f_hi (pystoi works at
    10 kHz, so f_hi above ~4.3 kHz keeps the top band).
    """
    from pystoi.stoi import CF

    cf = np.asarray(CF, np.float64)
    upper = cf * 2.0 ** (1.0 / 6.0)
    keep = np.ones(len(cf), dtype=bool)
    if f_lo is not None:
        keep &= cf >= f_lo
    if f_hi is not None:
        keep &= upper <= f_hi
    if not keep.any():
        raise ValueError(f"no 1/3-octave band inside [{f_lo}, {f_hi}]")
    return keep


def stoi_banded(
    ref: np.ndarray,
    est: np.ndarray,
    fs: int,
    extended: bool,
    band_mask: np.ndarray | None = None,
) -> float:
    """pystoi's (E)STOI restricted to the 1/3-octave bands in ``band_mask``.

    Mirrors ``pystoi.stoi`` step by step (resample to 10 kHz, silent-frame
    removal, 256-sample frames, 30-frame segments, beta = -15 dB clipping for
    STOI, row/column normalization for ESTOI); only the band set differs.
    """
    import importlib

    from pystoi import utils

    S = importlib.import_module("pystoi.stoi")  # the module (pystoi re-exports the function as `stoi`)

    x, y = _match(ref, est)
    if fs != S.FS:
        x = utils.resample_oct(x, S.FS, fs)
        y = utils.resample_oct(y, S.FS, fs)
    x, y = utils.remove_silent_frames(x, y, S.DYN_RANGE, S.N_FRAME, int(S.N_FRAME / 2))
    x_spec = utils.stft(x, S.N_FRAME, S.NFFT, overlap=2).transpose()
    y_spec = utils.stft(y, S.N_FRAME, S.NFFT, overlap=2).transpose()
    if x_spec.shape[-1] < S.N:
        warnings.warn("signal too short for (E)STOI after silent-frame removal; returning NaN")
        return float("nan")
    obm = S.OBM if band_mask is None else S.OBM[np.asarray(band_mask, bool)]
    x_tob = np.sqrt(np.matmul(obm, np.square(np.abs(x_spec))))
    y_tob = np.sqrt(np.matmul(obm, np.square(np.abs(y_spec))))
    xs = np.array([x_tob[:, m - S.N : m] for m in range(S.N, x_tob.shape[1] + 1)])
    ys = np.array([y_tob[:, m - S.N : m] for m in range(S.N, x_tob.shape[1] + 1)])
    if extended:
        xn = utils.row_col_normalize(xs)
        yn = utils.row_col_normalize(ys)
        return float(np.sum(xn * yn / S.N) / xn.shape[0])
    norm_const = np.linalg.norm(xs, axis=2, keepdims=True) / (np.linalg.norm(ys, axis=2, keepdims=True) + utils.EPS)
    yp = ys * norm_const
    clip_value = 10 ** (-S.BETA / 20)
    yp = np.minimum(yp, xs * (1 + clip_value))
    yp = yp - np.mean(yp, axis=2, keepdims=True)
    xs = xs - np.mean(xs, axis=2, keepdims=True)
    yp /= np.linalg.norm(yp, axis=2, keepdims=True) + utils.EPS
    xs /= np.linalg.norm(xs, axis=2, keepdims=True) + utils.EPS
    return float(np.sum(yp * xs) / (xs.shape[0] * xs.shape[1]))


def _stoi_common(est, ref, fs, f_lo, f_hi, extended, drop_bands, filter_order):
    x, y = _match(ref, est)
    if f_lo is not None or f_hi is not None:
        x = bandlimit(x, fs, f_lo or 0.0, f_hi if f_hi is not None else fs / 2, filter_order)
        y = bandlimit(y, fs, f_lo or 0.0, f_hi if f_hi is not None else fs / 2, filter_order)
    mask = third_octave_mask(f_lo, f_hi) if drop_bands else None
    return stoi_banded(x, y, fs, extended, mask)


def estoi(
    est: np.ndarray,
    ref: np.ndarray,
    fs: int = SCORE_FS,
    f_lo: float | None = DEFAULT_F_LO,
    f_hi: float | None = DEFAULT_F_HI,
    drop_bands: bool = True,
    filter_order: int = 8,
) -> float:
    """ESTOI (Jensen & Taal 2016) of ``est`` vs ``ref``, band-limited to [f_lo, f_hi]."""
    return _stoi_common(est, ref, fs, f_lo, f_hi, True, drop_bands, filter_order)


def stoi(
    est: np.ndarray,
    ref: np.ndarray,
    fs: int = SCORE_FS,
    f_lo: float | None = DEFAULT_F_LO,
    f_hi: float | None = DEFAULT_F_HI,
    drop_bands: bool = True,
    filter_order: int = 8,
) -> float:
    """STOI (Taal et al. 2011) of ``est`` vs ``ref``, band-limited to [f_lo, f_hi]."""
    return _stoi_common(est, ref, fs, f_lo, f_hi, False, drop_bands, filter_order)


# ---------------------------------------------------------------------------
# PESQ
# ---------------------------------------------------------------------------
def pesq_wb(est: np.ndarray, ref: np.ndarray, fs: int = SCORE_FS) -> float:
    """PESQ wide-band (ITU-T P.862.2) MOS-LQO; NaN on no-utterance / PESQ errors.

    Flagged as unreliable on band-limited, coloured and generative output (§8.1).
    """
    import pesq as _pesq

    x, y = _match(ref, est)
    if fs != SCORE_FS:
        from uwse.audio import resample

        x, y = resample(x, fs, SCORE_FS), resample(y, fs, SCORE_FS)
    if not (np.any(x) and np.any(y)):
        return float("nan")
    try:
        return float(_pesq.pesq(SCORE_FS, x, y, "wb"))
    except _pesq.NoUtterancesError:
        return float("nan")
    except _pesq.PesqError as e:
        # anything other than "no utterance" is unexpected: keep NaN but make it visible
        warnings.warn(f"PESQ failed ({type(e).__name__}: {e}); returning NaN")
        return float("nan")


# ---------------------------------------------------------------------------
# SDR family
# ---------------------------------------------------------------------------
def si_sdr(est: np.ndarray, ref: np.ndarray, zero_mean: bool = True) -> float:
    """Scale-invariant SDR (Le Roux et al. 2019), dB."""
    s, x = _match(est, ref)
    if zero_mean:
        s, x = s - s.mean(), x - x.mean()
    a = np.dot(s, x) / (np.dot(x, x) + EPS)
    t = a * x
    e = s - t
    return float(10.0 * np.log10((np.dot(t, t) + EPS) / (np.dot(e, e) + EPS)))


def ci_sdr(
    est: np.ndarray,
    ref: np.ndarray,
    filter_length: int = 512,
    diag_load: float = 1e-10,
    return_filter: bool = False,
) -> float | tuple[float, np.ndarray]:
    """CI-SDR (Boeddeker et al., arXiv 2011.15003), dB.

    Fits a causal FIR ``h`` of ``filter_length`` taps minimizing
    ``||est_pad - ref * h||^2`` where ``*`` is full convolution and ``est_pad``
    is ``est`` zero-padded by L-1 (the BSS-Eval convention). The normal
    equations are Toeplitz in the reference autocorrelation, solved by
    Levinson recursion. ``diag_load`` (relative to the zero-lag
    autocorrelation) regularizes band-limited references.
    """
    s, x = _match(est, ref)
    L = int(filter_length)
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n + L)))
    X = np.fft.rfft(x, nfft)
    S = np.fft.rfft(s, nfft)
    acf = np.fft.irfft(np.abs(X) ** 2, nfft)[:L]
    # r[k] = sum_n s[n] x[n-k], k = 0..L-1
    xcf = np.fft.irfft(S * np.conj(X), nfft)[:L]
    acf = acf.copy()
    acf[0] += diag_load * acf[0] + EPS
    h = solve_toeplitz(acf, xcf)
    target = fftconvolve(x, h)[: n + L - 1]
    s_pad = np.concatenate([s, np.zeros(L - 1)])
    e = s_pad - target
    val = float(10.0 * np.log10((np.dot(target, target) + EPS) / (np.dot(e, e) + EPS)))
    return (val, h) if return_filter else val


# ---------------------------------------------------------------------------
# Per-utterance bundle
# ---------------------------------------------------------------------------
ALL_METRICS = ("estoi", "stoi", "pesq_wb", "si_sdr", "ci_sdr")


def compute_metrics(
    est: np.ndarray,
    ref_r1: np.ndarray,
    fs: int = SCORE_FS,
    ref_r0: np.ndarray | None = None,
    f_lo: float | None = DEFAULT_F_LO,
    f_hi: float | None = DEFAULT_F_HI,
    metrics: Iterable[str] = ALL_METRICS,
) -> dict[str, float]:
    """All intrusive metrics of one utterance, at 16 kHz (§8.2), as a flat dict.

    Keys: ``estoi``, ``stoi`` (vs R1, band-limited), ``pesq_wb_r1``,
    ``si_sdr``, ``ci_sdr`` (vs R1) and, if ``ref_r0`` is given,
    ``pesq_wb_r0`` and ``estoi_r0`` (R0 band-limited as well, §8.2).
    """
    from uwse.audio import resample

    est = np.asarray(est, np.float64)
    ref_r1 = np.asarray(ref_r1, np.float64)
    if fs != SCORE_FS:
        est, ref_r1 = resample(est, fs, SCORE_FS), resample(ref_r1, fs, SCORE_FS)
        if ref_r0 is not None:
            ref_r0 = resample(np.asarray(ref_r0, np.float64), fs, SCORE_FS)
    metrics = set(metrics)
    out: dict[str, float] = {}
    if "estoi" in metrics:
        out["estoi"] = estoi(est, ref_r1, SCORE_FS, f_lo, f_hi)
        if ref_r0 is not None:
            out["estoi_r0"] = estoi(est, ref_r0, SCORE_FS, f_lo, f_hi)
    if "stoi" in metrics:
        out["stoi"] = stoi(est, ref_r1, SCORE_FS, f_lo, f_hi)
    if "pesq_wb" in metrics:
        out["pesq_wb_r1"] = pesq_wb(est, ref_r1, SCORE_FS)
        if ref_r0 is not None:
            out["pesq_wb_r0"] = pesq_wb(est, ref_r0, SCORE_FS)
    if "si_sdr" in metrics:
        out["si_sdr"] = si_sdr(est, ref_r1)
    if "ci_sdr" in metrics:
        out["ci_sdr"] = ci_sdr(est, ref_r1)
    return out
