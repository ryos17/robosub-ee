"""Exponential-sine-sweep (ESS) measurement tools.

Implements RESEARCH_PLAN §6.4 steps 4-5 (E2 projector chain, E3 pool IR grid)
and the measurement side of §4.2 rows 10, 11 and 14 and §4.3 step 2:

* sweep generation (Farina 2000; synchronized variant of Novak et al. 2015 so
  that harmonic IRs are phase-exact), optional low-frequency level pre-shaping
  so the projector is not over-driven, and a silence tail;
* regularized FFT deconvolution (Kirkeby-style in-band / out-of-band
  regularization) -> impulse response;
* harmonic IR separation at dt_k = T ln(k) / ln(f2/f1) and conversion of
  harmonic IRs to Hammerstein kernels h_1..h_K (Novak et al. 2010);
* loopback-referenced clock-drift estimation and correction, applied BEFORE
  deconvolution / averaging (§6.4 step 5: 50 ppm over 10 s = 0.5 ms skew);
* averaging of repeated sweeps with sweep-to-sweep coherence and per-band IR
  variance (§6.4 step 5; feeds §4.2 #14 and the σ_rep thresholds of §4.4).

References
----------
A. Farina, "Simultaneous measurement of impulse response and distortion with a
swept-sine technique", AES 108th Conv., 2000.
A. Novak, L. Simon, F. Kadlec, P. Lotton, "Nonlinear system identification using
exponential swept-sine signal", IEEE Trans. Instrum. Meas. 59(8), 2220-2229, 2010.
A. Novak, P. Lotton, L. Simon, "Synchronized swept-sine: theory, application and
implementation", J. Audio Eng. Soc. 63(10), 786-798, 2015.
O. Kirkeby, P. A. Nelson, "Digital filter design for inversion problems in sound
reproduction", J. Audio Eng. Soc. 47(7/8), 1999.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
from scipy.fft import next_fast_len
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve
from scipy.special import i0 as _i0
from tqdm import tqdm

from ..audio import fade
from .analysis import OCTAVE_CENTERS, filterbank

__all__ = [
    "ESS",
    "generate_ess",
    "harmonic_offsets",
    "deconvolve",
    "measure_ir",
    "HarmonicIRs",
    "harmonic_irs",
    "hammerstein_matrix",
    "hammerstein_kernels",
    "hammerstein_apply",
    "estimate_drift_ppm",
    "correct_drift",
    "SweepAverage",
    "average_sweeps",
    "processing_gain_db",
]


# --------------------------------------------------------------------------- #
# Sweep generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ESS:
    """A generated exponential sine sweep.

    Attributes
    ----------
    excitation : the shaped, faded sweep only (use this as the deconvolution
        reference).
    signal : ``excitation`` followed by the silence tail (what is played).
    fs, f1, f2 : sample rate and sweep band (Hz).
    T : actual sweep duration (s); differs slightly from the request when
        ``synchronized`` rounds the rate.
    L : sweep rate constant, f_inst(t) = f1 exp(t / L).
    amplitude : peak amplitude of the (unshaped) sweep.
    synchronized : whether f1 L is an integer (Novak 2015).
    """

    excitation: np.ndarray
    signal: np.ndarray
    fs: int
    f1: float
    f2: float
    T: float
    L: float
    amplitude: float
    synchronized: bool

    def harmonic_offset(self, k: int) -> float:
        """Time advance (s) of the k-th harmonic IR: L ln k."""
        return self.L * np.log(k)


def generate_ess(
    f1: float,
    f2: float,
    T: float,
    fs: int,
    amplitude: float = 1.0,
    fade_in_s: float = 0.05,
    fade_out_s: float = 0.01,
    silence_s: float = 4.0,
    synchronized: bool = True,
    lf_shelf_hz: float | None = None,
    lf_shelf_db: float = 0.0,
) -> ESS:
    """Exponential sine sweep x(t) = A g(f_inst(t)) sin(phi(t)).

    Parameters
    ----------
    f1, f2 : start / stop frequency (Hz), 0 < f1 < f2 <= fs/2.
    T : requested sweep duration (s). Plan default for E3: 10 s.
    fs : sample rate.
    amplitude : peak amplitude A.
    fade_in_s, fade_out_s : raised-cosine fades (avoid clicks at the ends).
    silence_s : silence appended after the sweep. §6.4 step 5 requires >= 4 s
        (3 x T60 for T60 up to 1.2 s).
    synchronized : round L so that f1 L is an integer (Novak 2015). Then
        k phi(t) = phi(t + L ln k) exactly, so harmonic IRs carry no spurious
        phase and the Hammerstein conversion is exact. Keep True for E2.
    lf_shelf_hz, lf_shelf_db : low-frequency level pre-shaping. Because an ESS
        maps time one-to-one onto frequency, a shelf is applied as an amplitude
        envelope g(f_inst(t)) with g_dB(f) = -lf_shelf_db / (1 + (f/fc)^4):
        about ``-lf_shelf_db`` well below fc, 0 dB well above. Deconvolution
        against ``ESS.excitation`` removes the shaping from the linear IR.
        Harmonic IRs are only exact for an unshaped sweep (the k-th harmonic
        sees g^k at frequency k f), so use shaping for E3, not for E2 THD.

    Returns
    -------
    ESS
    """
    if not 0 < f1 < f2 <= fs / 2:
        raise ValueError("need 0 < f1 < f2 <= fs/2")
    R = np.log(f2 / f1)
    L = T / R
    if synchronized:
        L = max(np.round(f1 * L), 1.0) / f1
        T = L * R
    n = int(round(T * fs))
    t = np.arange(n) / fs
    if synchronized:
        # phi(t) = 2 pi f1 L exp(t/L); phi(0) = 2 pi * integer, so x(0) = 0
        phase = 2 * np.pi * f1 * L * np.exp(t / L)
    else:
        phase = 2 * np.pi * f1 * L * (np.exp(t / L) - 1.0)
    x = amplitude * np.sin(phase)
    if lf_shelf_hz is not None and lf_shelf_db != 0.0:
        f_inst = f1 * np.exp(t / L)
        g_db = -lf_shelf_db / (1.0 + (f_inst / lf_shelf_hz) ** 4)
        x *= 10.0 ** (g_db / 20.0)
    x = fade(x, fs, fade_in_s, fade_out_s)
    sig = np.concatenate([x, np.zeros(int(round(silence_s * fs)))])
    return ESS(x, sig, fs, float(f1), float(f2), float(T), float(L), float(amplitude), synchronized)


def harmonic_offsets(f1: float, f2: float, T: float, K: int) -> np.ndarray:
    """dt_k = T ln(k) / ln(f2/f1) for k = 1..K (s). dt_1 = 0."""
    return T * np.log(np.arange(1, K + 1)) / np.log(f2 / f1)


def processing_gain_db(T: float, bandwidth_hz: float) -> float:
    """10 log10(T B): processing gain only, NOT the achieved IR SNR (§6.4)."""
    return float(10 * np.log10(T * bandwidth_hz))


# --------------------------------------------------------------------------- #
# Deconvolution
# --------------------------------------------------------------------------- #
def _regularization(freqs: np.ndarray, p_max: float, f_band, reg_in_db, reg_out_db, trans_oct=1 / 6):
    """Frequency-dependent Kirkeby regularization eps(f).

    eps = p_max 10^(reg_in/10) inside f_band, p_max 10^(reg_out/10) outside,
    raised-cosine transition (in log f) over ``trans_oct`` octaves just outside
    the band edges.
    """
    e_in = p_max * 10 ** (reg_in_db / 10)
    if f_band is None:
        return np.full_like(freqs, e_in)
    e_out = p_max * 10 ** (reg_out_db / 10)
    lo, hi = f_band
    with np.errstate(divide="ignore"):
        lf = np.log2(np.maximum(freqs, 1e-9))
    # distance outside the band in octaves (0 inside)
    d = np.maximum(np.log2(lo) - lf, 0.0) + np.maximum(lf - np.log2(hi), 0.0)
    w = np.clip(d / trans_oct, 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * w)  # 0 in band -> 1 outside
    # interpolate in log domain so the transition is smooth in dB
    return np.exp((1 - w) * np.log(e_in) + w * np.log(e_out))


def _deconv_spectrum(recorded, excitation, fs, f_band, reg_in_db, reg_out_db):
    recorded = np.asarray(recorded, dtype=np.float64)
    excitation = np.asarray(excitation, dtype=np.float64)
    # length >= len(y) + len(x): no circular overlap between the causal IR and
    # the anti-causal harmonic IRs (which sit at negative times down to -T)
    nfft = next_fast_len(len(recorded) + len(excitation))
    X = rfft(excitation, nfft)
    Y = rfft(recorded, nfft)
    P = np.abs(X) ** 2
    eps = _regularization(rfftfreq(nfft, 1 / fs), P.max(), f_band, reg_in_db, reg_out_db)
    return Y * np.conj(X) / (P + eps), nfft


def deconvolve(
    recorded: np.ndarray,
    excitation: np.ndarray,
    fs: int,
    f_band: tuple[float, float] | None = None,
    n_out: int | None = None,
    reg_in_db: float = -60.0,
    reg_out_db: float = 20.0,
    n_pre: int = 0,
) -> np.ndarray:
    """Impulse response by regularized spectral division h = IFFT(Y X* / (|X|^2 + eps)).

    Parameters
    ----------
    recorded : recorded response (1-D). Drift-correct it first (``correct_drift``).
    excitation : the signal that was actually played (e.g. ``ESS.excitation``,
        including any pre-shaping and fades), or the loopback recording of it.
    fs : sample rate.
    f_band : excited band (f1, f2). Outside it the regularization rises to
        ``reg_out_db`` (re max |X|^2) so out-of-band noise is not amplified.
        None: uniform ``reg_in_db``.
    n_out : number of causal IR samples returned. Default
        ``len(recorded) - len(excitation) + 1`` (the tail that was recorded).
    reg_in_db, reg_out_db : regularization levels re max |X|^2.
    n_pre : also return this many negative-time samples in front (then
        h[n_pre] is zero lag). The band limitation W = |X|^2 / (|X|^2 + eps) is
        zero-phase, so a band-limited IR rings symmetrically about each
        arrival; keep a few ms of pre-roll when the band edges are steep.

    Returns
    -------
    h : (n_pre + n_out,) impulse response, h[n_pre] = zero lag.
    """
    H, nfft = _deconv_spectrum(recorded, excitation, fs, f_band, reg_in_db, reg_out_db)
    if n_out is None:
        n_out = max(len(recorded) - len(excitation) + 1, 1)
    n_out = min(n_out, nfft - len(excitation))
    h = irfft(H, nfft)
    if n_pre:
        return np.concatenate([h[nfft - n_pre :], h[:n_out]])
    return h[:n_out]


def measure_ir(recorded: np.ndarray, ess: ESS, n_out: int | None = None, n_pre: int = 0, **kw) -> np.ndarray:
    """``deconvolve`` against ``ess.excitation`` with f_band = (f1, f2)."""
    return deconvolve(recorded, ess.excitation, ess.fs, f_band=(ess.f1, ess.f2), n_out=n_out, n_pre=n_pre, **kw)


# --------------------------------------------------------------------------- #
# Harmonic IRs and Hammerstein kernels
# --------------------------------------------------------------------------- #
@dataclass
class HarmonicIRs:
    """Separated harmonic IRs.

    irs[k-1] is the k-th harmonic IR; sample index ``pre`` is t = 0 (the
    harmonic's own arrival). All rows have the same time axis.
    """

    irs: np.ndarray
    fs: int
    pre: int
    offsets_s: np.ndarray
    amplitude: float

    @property
    def K(self) -> int:
        return self.irs.shape[0]


def harmonic_irs(
    recorded: np.ndarray,
    ess: ESS,
    K: int = 3,
    n_ir: int | None = None,
    pre_s: float = 1e-3,
    fade_frac: float = 0.1,
    reg_in_db: float = -60.0,
    reg_out_db: float = 20.0,
) -> HarmonicIRs:
    """Separate harmonic IRs k = 1..K from one ESS recording (Farina 2000).

    After deconvolution with the linear inverse, the k-th harmonic's IR sits at
    t = -dt_k = -L ln k. Each is brought to t = 0 by an exact fractional
    (frequency-domain) shift and windowed: half-Hann fade-in over ``pre_s``
    before the arrival, flat, half-Hann fade-out over the last ``fade_frac``.

    The default window length is the largest that cannot overlap the
    neighbouring harmonic: the gap between harmonics K+1 and K,
    L ln((K+1)/K), which is the smallest relevant gap.

    Scaling: deconvolution is against ``ess.excitation`` (amplitude A
    included), so H_k = sum_n A^(n-1) c_{n,k} G_n (see ``hammerstein_matrix``).
    """
    if K < 1:
        raise ValueError("K >= 1")
    fs = ess.fs
    H, nfft = _deconv_spectrum(recorded, ess.excitation, fs, (ess.f1, ess.f2), reg_in_db, reg_out_db)
    pre = int(round(pre_s * fs))
    gap = int(np.floor(ess.L * np.log((K + 1) / K) * fs))
    if n_ir is None:
        n_ir = gap
    if n_ir > gap:
        raise ValueError(f"n_ir={n_ir} exceeds the harmonic gap ({gap} samples)")
    if n_ir <= pre:
        raise ValueError("window shorter than pre_s; increase T or decrease K / pre_s")
    n_post = n_ir - pre
    win = np.ones(n_ir)
    if pre:
        win[:pre] = 0.5 - 0.5 * np.cos(np.pi * np.arange(pre) / pre)
    n_fo = max(int(round(fade_frac * n_post)), 1)
    win[-n_fo:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, n_fo + 1) / n_fo)

    omega = 2 * np.pi * rfftfreq(nfft, 1 / fs)
    offsets = np.array([ess.harmonic_offset(k) for k in range(1, K + 1)])
    out = np.empty((K, n_ir))
    for i, dt in enumerate(offsets):
        # delay by dt + pre/fs: the harmonic at -dt moves to sample `pre`
        hk = irfft(H * np.exp(-1j * omega * (dt + pre / fs)), nfft)[:n_ir]
        out[i] = hk * win
    return HarmonicIRs(out, fs, pre, offsets, ess.amplitude)


def hammerstein_matrix(K: int, amplitude: float = 1.0) -> np.ndarray:
    """Complex matrix M (K x K) with H_k(f) = sum_n M[k-1, n-1] G_n(f).

    Novak et al. (2010), Sec. III: for an input A sin(phi), the power-reduction
    expansion sin^n(phi) = sum_k c_{n,k} sin(k phi + psi_{n,k}) maps each
    Hammerstein branch g_n onto harmonics k <= n (k = n, n-2, ...). With the
    synchronized sweep, sin(k phi(t)) is the sweep advanced by L ln k, so after
    deconvolution with the sweep of amplitude A:

        M[k, n] = A^(n-1) * c_{n,k} * exp(j psi_{n,k}).

    For K = 3 this is H_1 = G_1 + (3/4) A^2 G_3, H_2 = -(j/2) A G_2,
    H_3 = -(1/4) A^2 G_3 (j: +90 deg at positive frequencies, numpy rfft
    convention). c and psi are obtained here numerically from the DFT of
    sin^n over one period, which is exact for integer n.
    """
    M = 256
    phi = 2 * np.pi * np.arange(M) / M
    mat = np.zeros((K, K), dtype=complex)
    for n in range(1, K + 1):
        F = np.fft.fft(np.sin(phi) ** n) / M
        for k in range(1, n + 1):
            # sin(k phi) has DFT coefficient 1/(2j) at bin k; ratio = complex gain
            coef = F[k] * 2j
            if abs(coef) > 1e-12:
                mat[k - 1, n - 1] = amplitude ** (n - 1) * coef
    return mat


def hammerstein_kernels(hir: HarmonicIRs, K: int | None = None) -> np.ndarray:
    """Hammerstein kernels g_1..g_K (time domain, same axis as ``hir``: index
    ``hir.pre`` is t = 0) from harmonic IRs by inverting ``hammerstein_matrix``
    per frequency (Novak et al. 2010). The model is y = sum_k g_k * x^k with x
    in the same units as the ESS amplitude.
    """
    K = hir.K if K is None else K
    if K > hir.K:
        raise ValueError("K exceeds the number of harmonic IRs")
    n = hir.irs.shape[1]
    Hf = rfft(hir.irs[:K], axis=1)  # (K, F)
    Minv = np.linalg.inv(hammerstein_matrix(K, hir.amplitude))
    Gf = Minv @ Hf
    return irfft(Gf, n, axis=1)


def hammerstein_apply(x: np.ndarray, kernels: np.ndarray, pre: int = 0) -> np.ndarray:
    """y = sum_k g_k * x^k, aligned to x (drops ``pre`` leading kernel samples)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros(len(x))
    xk = np.ones_like(x)
    for g in kernels:
        xk = xk * x
        y += fftconvolve(xk, g)[pre : pre + len(x)]
    return y


# --------------------------------------------------------------------------- #
# Clock drift (loopback referenced)
# --------------------------------------------------------------------------- #
def _xcorr_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full cross-correlation r[l] = sum_n a[n + l] b[n], lags -(len(b)-1)..len(a)-1."""
    n = next_fast_len(len(a) + len(b) - 1)
    r = irfft(rfft(a, n) * np.conj(rfft(b, n)), n)
    return np.concatenate([r[n - (len(b) - 1) :], r[: len(a)]])


def _xcorr_env(win: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Envelope |analytic r[k]| of r[k] = sum_n win[n + k] seg[n] for the valid
    lags k = 0..len(win) - len(seg). Picking the integer delay on the envelope
    (group delay) rather than on |r| avoids carrier-cycle slips for the
    narrowband low-frequency segments of a sweep under noise (review fix)."""
    n = next_fast_len(len(win) + len(seg) - 1)
    nr = n // 2 + 1
    Z = np.zeros(n, dtype=complex)
    Z[:nr] = rfft(win, n) * np.conj(rfft(seg, n))
    Z[1 : (n + 1) // 2] *= 2.0
    return np.abs(np.fft.ifft(Z)[: len(win) - len(seg) + 1])


def _frac_delay(seg_rec: np.ndarray, seg_ref: np.ndarray, band_db: float = -20.0) -> tuple[float, float]:
    """Sub-sample group delay of seg_rec re seg_ref from a weighted LS fit of the
    cross-spectrum phase, angle(Y X*) = phi0 - w d. The intercept phi0 is
    needed because a time-scaled (drifted) chirp segment is narrowband with an
    arbitrary carrier-phase offset; forcing the line through the origin would
    bias d.

    Only bins where the *reference* holds energy (within ``band_db`` of its
    peak) enter the fit, weighted by |X|^2: the phase variance of a bin of
    Y X* under additive noise is ~ sigma^2 / |X|^2, so |X|^2 is the WLS weight.
    Selecting bins by |Y X*| instead lets out-of-band noise bins (with random
    phase) into the unwrap, which broke the estimate at low SNR (review fix).

    Returns (d, info) where info = sum |X|^2 (w - w_mean)^2 is proportional to
    the inverse variance of d under white noise (Fisher information of the
    slope with a free intercept); use it as the segment weight.
    """
    m = len(seg_ref)
    w = np.hanning(m)
    nfft = next_fast_len(2 * m)
    X = rfft(seg_ref * w, nfft)
    P = np.abs(X) ** 2
    keep = np.nonzero(P >= P.max() * 10 ** (band_db / 10))[0]
    if len(keep) < 2:
        return 0.0, 0.0
    C = rfft(seg_rec * w, nfft)[keep] * np.conj(X[keep])
    om = 2 * np.pi * keep / nfft
    wt = P[keep]
    om_m = np.sum(wt * om) / np.sum(wt)
    den = np.sum(wt * (om - om_m) ** 2)
    if den <= 0:
        return 0.0, 0.0
    # Coarse slope without unwrapping (Kay's weighted phase-difference
    # estimator over adjacent kept bins): np.unwrap of noisy low-|X| band-edge
    # bins inserts 2 pi jumps that shift every higher bin (review fix: 1.7
    # sample errors at 0 dB input SNR, where the CRLB is ~0.05 sample).
    adj = np.nonzero(np.diff(keep) == 1)[0]
    d0 = -np.angle(np.sum(C[adj + 1] * np.conj(C[adj]))) / (2 * np.pi / nfft) if len(adj) else 0.0
    # Fine WLS (with intercept) on the residual phase, which is now ~ constant
    # across the band, so the wrapped angle needs no unwrapping.
    Cr = C * np.exp(1j * om * d0)
    ph = np.angle(Cr * np.exp(-1j * np.angle(np.sum(wt * Cr / np.maximum(np.abs(Cr), 1e-300)))))
    ph_m = np.sum(wt * ph) / np.sum(wt)
    return float(d0 - np.sum(wt * (om - om_m) * (ph - ph_m)) / den), float(den)


def _coarse_lag(rec: np.ndarray, ref: np.ndarray, starts: np.ndarray, m: int, half_span: int) -> int:
    """Global lag of ``rec`` re ``ref`` by non-coherent integration of the
    per-segment matched-filter envelopes, smoothed over the possible drift span.

    The coherent full-reference cross-correlation is ambiguous for references
    made of repeated identical markers: drift misaligns the end marker at the
    true lag by eps * len(ref) samples, so the true lag and the
    marker-1-vs-marker-2 lag score about equally (review fix). Summing |r_i|^2
    of each segment at lag l = k - s_i and box-smoothing over +-``half_span``
    samples gathers all segments at the true lag regardless of drift.
    """
    nfft = next_fast_len(len(rec) + len(ref))
    nr = nfft // 2 + 1
    Rf = rfft(rec, nfft)
    # analytic-signal weights (one-sided spectrum) -> envelope without carrier ripple
    one = np.full(nr, 2.0)
    one[0] = 1.0
    if nfft % 2 == 0:
        one[-1] = 1.0
    acc = np.zeros(nfft)
    Z = np.zeros(nfft, dtype=complex)
    for s in starts:
        Z[:nr] = Rf * np.conj(rfft(ref[s : s + m], nfft)) * one
        acc += np.roll(np.abs(np.fft.ifft(Z)) ** 2, -int(s))
    acc = uniform_filter1d(acc, 2 * half_span + 1, mode="wrap")
    j = int(np.argmax(acc))
    return j - nfft if j >= len(rec) else j


def estimate_drift_ppm(
    recording: np.ndarray,
    reference: np.ndarray,
    fs: int,
    segment_s: float = 0.5,
    min_rel_energy_db: float = -30.0,
    max_ppm: float = 1000.0,
    return_details: bool = False,
):
    """Clock drift of ``recording`` relative to ``reference`` (ppm).

    Convention: ppm > 0 means the recording clock runs fast, i.e. an event at
    reference sample s appears at recording sample ~ s (1 + ppm 1e-6) + d0.
    Pass ``correct_drift(recording, ppm)`` to map it back onto the reference
    clock. Use the loopback channel (§6.4 step 5, §6.5) recorded on the same
    ADC as the hydrophones: the drift is a property of the board clock.

    Method: the reference is cut into ``segment_s`` segments (silent ones
    skipped, e.g. between start/end chirp markers); a coarse global lag is
    found by non-coherent integration of the segment matched-filter envelopes
    (robust to repeated identical markers); each segment is then located in
    the recording by matched filtering within +-2 x the maximum drift span,
    refined to sub-sample precision by the cross-spectrum phase slope, and a
    weighted line d_i = d0 + eps s_i is fitted (weights: in-band
    sum |X|^2 (w - w_mean)^2 of the Hann-windowed segment, i.e. energy x centred
    rms bandwidth^2, the inverse variance of the delay with a free carrier
    phase under white noise; the earlier rectangular-window bandwidth was
    dominated by sinc leakage and over-weighted low-frequency segments). For an ESS any time-scaling bias (Doppler-delay coupling,
    L ln(1+eps)) is the same for every segment and only enters d0, so the slope
    is unbiased; the same holds for identical start/end chirp markers.
    Segments should be shorter than the markers' spacing; with markers, pass a
    ``segment_s`` no longer than the marker.
    """
    rec = np.asarray(recording, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    m = int(round(segment_s * fs))
    starts = np.arange(0, len(ref) - m + 1, m)
    if len(starts) < 2:
        raise ValueError("reference shorter than two segments")
    energies = np.array([np.sum(ref[s : s + m] ** 2) for s in starts])
    ok = energies >= energies.max() * 10 ** (min_rel_energy_db / 10)
    # maximum drift span over the record (samples)
    span = int(np.ceil(max_ppm * 1e-6 * (len(ref) + len(rec)))) + 8
    lag0 = _coarse_lag(rec, ref, starts[ok], m, span)
    W = 2 * span  # the smoothed peak can sit anywhere inside the drift cluster

    pos, dly, wts = [], [], []
    for s in starts[ok]:
        seg = ref[s : s + m]
        a, b = s + lag0 - W, s + lag0 + m + W
        if a < 0 or b > len(rec):
            continue
        win = rec[a:b]
        li = int(np.argmax(_xcorr_env(win, seg)))  # lags 0..2W in win
        if li == 0 or li == 2 * W:
            continue
        d_int = a + li - s
        frac, info = _frac_delay(rec[s + d_int : s + d_int + m], seg)
        if info <= 0:
            continue
        pos.append(s + m / 2)
        dly.append(d_int + frac)
        wts.append(info)
    if len(pos) < 2:
        raise ValueError("need >= 2 usable segments to estimate drift")
    pos, dly, wts = map(np.asarray, (pos, dly, wts))
    wts = wts / wts.max()
    A = np.stack([np.ones_like(pos), pos], 1) * np.sqrt(wts)[:, None]
    coef, *_ = np.linalg.lstsq(A, dly * np.sqrt(wts), rcond=None)
    ppm = float(coef[1] * 1e6)
    if return_details:
        return ppm, {"lag0": lag0, "positions": pos, "delays": dly, "weights": wts, "offset": coef[0]}
    return ppm


def correct_drift(
    x: np.ndarray,
    ppm: float,
    half_taps: int = 32,
    beta: float = 8.0,
    chunk: int = 16384,
    n_phase: int = 1024,
) -> np.ndarray:
    """Resample ``x`` onto the reference clock: y[m] = x(m (1 + ppm 1e-6)).

    Kaiser-windowed sinc interpolation (2 x ``half_taps`` taps), accurate to
    about 0.9 x Nyquist. Apply BEFORE deconvolution and before averaging
    sweeps (§4.3 step 2, §6.4 step 5). Output length is
    floor((len(x) - 1) / (1 + eps)) + 1.

    The kernel is tabulated at ``n_phase`` fractional offsets and linearly
    interpolated between them (about -130 dB from evaluating the Kaiser sinc
    directly at every output sample, and ~10x faster: review fix, 10 s at
    48 kHz took ~3-9 s).
    """
    x = np.asarray(x, dtype=np.float64)
    r = 1.0 + ppm * 1e-6
    n_out = int(np.floor((len(x) - 1) / r)) + 1
    xp = np.concatenate([np.zeros(half_taps), x, np.zeros(half_taps + 1)])
    taps = np.arange(-half_taps + 1, half_taps + 1)
    fr = np.arange(n_phase + 1) / n_phase
    u = taps[None, :] - fr[:, None]  # distance from each tap to the output instant
    tab = np.sinc(u) * _i0(beta * np.sqrt(np.clip(1 - (u / half_taps) ** 2, 0, None))) / _i0(beta)
    dtab = np.diff(tab, axis=0)
    frames = np.lib.stride_tricks.sliding_window_view(xp, 2 * half_taps)  # frames[j] = xp[j : j + 2 half_taps]
    y = np.empty(n_out)
    for c0 in range(0, n_out, chunk):
        mm = np.arange(c0, min(c0 + chunk, n_out))
        p = mm * r
        n0 = np.floor(p).astype(np.int64)
        q = (p - n0) * n_phase
        i = np.minimum(q.astype(np.int64), n_phase - 1)
        k = tab[i] + (q - i)[:, None] * dtab[i]
        # taps n0 - half_taps + 1 .. n0 + half_taps of x = xp[n0 + 1 : n0 + 1 + 2 half_taps]
        y[mm] = np.einsum("ij,ij->i", frames[n0 + 1], k)
    return y


# --------------------------------------------------------------------------- #
# Averaging repeated sweeps
# --------------------------------------------------------------------------- #
@dataclass
class SweepAverage:
    """Mean IR and sweep-to-sweep statistics.

    coherence : gamma^2(f) = |sum_i H_i|^2 / (S sum_i |H_i|^2), in [0, 1]
        (1: identical sweeps; expected ~1/S for incoherent sweeps).
    band_coherence : the same ratio with energies summed over each band.
    band_var_db : normalized IR variance per band,
        10 log10( sum_i ||h_ib - mean_b||^2 / (S - 1) / ||mean_b||^2 ).
    """

    ir_mean: np.ndarray
    freqs: np.ndarray
    coherence: np.ndarray
    band_centers: np.ndarray
    band_coherence: np.ndarray
    band_var_db: np.ndarray
    n_sweeps: int = field(default=0)


def average_sweeps(irs: np.ndarray, fs: int, centers=OCTAVE_CENTERS) -> SweepAverage:
    """Average S aligned, drift-corrected IRs (S, n) and report coherence and
    per-band variance (§6.4 step 5). If coherence is low at high frequency the
    plan uses single-sweep IRs instead of the average.
    """
    irs = np.asarray(irs, dtype=np.float64)
    if irs.ndim != 2 or irs.shape[0] < 2:
        raise ValueError("irs must be (S >= 2, n)")
    S, n = irs.shape
    mean = irs.mean(0)
    Hf = rfft(irs, axis=1)
    num = np.abs(Hf.sum(0)) ** 2
    den = S * np.sum(np.abs(Hf) ** 2, axis=0)
    coh = num / np.maximum(den, 1e-300)
    freqs = rfftfreq(n, 1 / fs)
    centers = np.asarray([c for c in centers if c / np.sqrt(2) < fs / 2], dtype=float)
    bcoh = np.empty(len(centers))
    bvar = np.empty(len(centers))
    banded = np.stack([filterbank(h, fs, centers) for h in tqdm(irs, desc="sweep bands", leave=False, disable=S < 8)])
    for b, fc in enumerate(centers):
        sel = (freqs >= fc / np.sqrt(2)) & (freqs < fc * np.sqrt(2))
        bcoh[b] = num[sel].sum() / max(den[sel].sum(), 1e-300)
        hb = banded[:, b]
        mb = hb.mean(0)
        v = np.sum((hb - mb) ** 2) / (S - 1)
        bvar[b] = 10 * np.log10(max(v, 1e-300) / max(np.sum(mb**2), 1e-300))
    return SweepAverage(mean, freqs, coh, centers, bcoh, bvar, S)
