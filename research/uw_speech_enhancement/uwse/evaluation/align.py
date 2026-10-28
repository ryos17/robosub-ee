"""Real-benchmark alignment: chirps, clock drift, fine lag, utterance cuts.

Implements RESEARCH_PLAN.md §6.5 ("Playback material and synchronization",
alignment steps 1-4) and the alignment fields of the §6.8 metadata schema
(``drift_ppm``, ``fine_lag_samples_16k``, ``chirp_psr_dB``, flag
``align_low_psr``).

Playback block layout (§6.5)::

    [0.5 s silence][LFM chirp 0.25 s, f_lo-8 kHz][0.5 s][utt1][1.0 s]...[uttN]
    [0.5 s][LFM chirp][2.0 s silence]

Pipeline
--------
1. :func:`detect_chirps` matched-filters the chirp on a stream and returns the
   (sub-sample) chirp onsets with their peak-to-sidelobe ratio (PSR).
2. :func:`estimate_drift_ppm` compares the start->end chirp interval with the
   nominal interval (from the playlist or from the loopback stream), and
   :func:`correct_drift` resamples the stream onto the reference clock.
3. :func:`fine_lag` searches the per-utterance lag against R1 at 16 kHz within
   +-20 ms and flags PSR < 6 dB.
4. :func:`cut_utterances` cuts the utterances (with padding) using the
   playlist timing anchored on the first chirp.

:func:`align_block` chains 1, 2 and 4 for one block. The QC thresholds are
parameters because §5.4 freezes them on v0-dev.

Conventions: sample indices are float where sub-sample precision matters;
``lag`` L means ``y[n] ~ g * ref[n - L]`` (positive L: ``y`` is late).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.signal import correlate, hilbert
from scipy.special import i0

__all__ = [
    "BlockLayout",
    "ChirpDetection",
    "LagResult",
    "lfm_chirp",
    "lfm_chirp_at",
    "assemble_block",
    "layout_from_timing",
    "matched_filter",
    "peak_to_sidelobe_db",
    "detect_chirps",
    "estimate_drift_ppm",
    "fractional_resample",
    "correct_drift",
    "fine_lag",
    "shift_signal",
    "fit_gain",
    "cut_utterances",
    "align_block",
]

#: §6.5 default chirp parameters. f_lo is set per projector (§3 "Band"):
#: f_lo = max(projector -10 dB point, 1.5 x measured cutoff), expected
#: about 500-650 Hz. 500 Hz is an uncalibrated default.
DEFAULT_F_LO = 500.0
DEFAULT_F_HI = 8000.0
DEFAULT_CHIRP_S = 0.25
#: Canonical synchronization chirp shared with ``uwse.bench.playlist`` (which
#: imports this generator, so playback and matched-filter template are the
#: same function). The window is a continuous Tukey(0.1): raised-cosine
#: tapers of 0.1 * 0.25 s / 2 = 12.5 ms at each end.
DEFAULT_TUKEY_ALPHA = 0.1
DEFAULT_FADE_S = DEFAULT_TUKEY_ALPHA * DEFAULT_CHIRP_S / 2.0
#: Played chirp peak level (uncalibrated default; set against projector drive in S1).
DEFAULT_CHIRP_PEAK_DBFS = -20.0
DEFAULT_CHIRP_AMPLITUDE = 10.0 ** (DEFAULT_CHIRP_PEAK_DBFS / 20.0)


# ---------------------------------------------------------------------------
# Chirp and block generation
# ---------------------------------------------------------------------------
def lfm_chirp_at(
    t: np.ndarray,
    f_lo: float = DEFAULT_F_LO,
    f_hi: float = DEFAULT_F_HI,
    duration: float = DEFAULT_CHIRP_S,
    fade_s: float = DEFAULT_FADE_S,
) -> np.ndarray:
    """Evaluate the LFM chirp at arbitrary times ``t`` (s, chirp onset at 0).

    Zero outside ``[0, duration)``. Raised-cosine fades of ``fade_s`` (a
    continuous Tukey window; default Tukey(0.1)) limit spectral splatter. Used to synthesise streams on a drifted clock exactly.
    """
    t = np.asarray(t, dtype=np.float64)
    k = (f_hi - f_lo) / duration  # sweep rate, Hz/s
    phase = 2.0 * np.pi * (f_lo * t + 0.5 * k * t * t)
    y = np.sin(phase)
    inside = (t >= 0.0) & (t < duration)
    w = np.ones_like(t)
    if fade_s > 0:
        rise = t < fade_s
        w[rise] = 0.5 - 0.5 * np.cos(np.pi * t[rise] / fade_s)
        fall = t > duration - fade_s
        w[fall] = 0.5 - 0.5 * np.cos(np.pi * (duration - t[fall]) / fade_s)
    return np.where(inside, y * w, 0.0)


def lfm_chirp(
    fs: int,
    f_lo: float = DEFAULT_F_LO,
    f_hi: float = DEFAULT_F_HI,
    duration: float = DEFAULT_CHIRP_S,
    fade_s: float = DEFAULT_FADE_S,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Sampled LFM chirp (§6.5: 0.25 s, f_lo to 8 kHz), ``round(duration*fs)`` samples.

    ``f_hi`` must be below Nyquist; at ``fs = 16 kHz`` pass ``f_hi < 8000``.
    """
    if f_hi >= fs / 2:
        raise ValueError(f"f_hi={f_hi} must be < fs/2={fs / 2}")
    n = int(round(duration * fs))
    return amplitude * lfm_chirp_at(np.arange(n) / fs, f_lo, f_hi, duration, fade_s)


@dataclass
class BlockLayout:
    """Nominal timing of one playback block (§6.5), in seconds."""

    utt_durations_s: Sequence[float]
    lead_silence_s: float = 0.5
    chirp_s: float = DEFAULT_CHIRP_S
    post_chirp_gap_s: float = 0.5
    inter_utt_gap_s: float = 1.0
    pre_chirp_gap_s: float = 0.5
    tail_silence_s: float = 2.0

    @property
    def chirp1_start_s(self) -> float:
        return self.lead_silence_s

    @property
    def utt_starts_s(self) -> np.ndarray:
        d = np.asarray(self.utt_durations_s, dtype=np.float64)
        first = self.lead_silence_s + self.chirp_s + self.post_chirp_gap_s
        offsets = np.concatenate([[0.0], np.cumsum(d[:-1] + self.inter_utt_gap_s)]) if len(d) else d
        return first + offsets

    @property
    def utt_ends_s(self) -> np.ndarray:
        return self.utt_starts_s + np.asarray(self.utt_durations_s, dtype=np.float64)

    @property
    def chirp2_start_s(self) -> float:
        ends = self.utt_ends_s
        last = ends[-1] if len(ends) else self.lead_silence_s + self.chirp_s + self.post_chirp_gap_s
        return float(last + self.pre_chirp_gap_s)

    @property
    def chirp_interval_s(self) -> float:
        """Nominal onset-to-onset interval between the two chirps."""
        return self.chirp2_start_s - self.chirp1_start_s

    @property
    def total_s(self) -> float:
        return self.chirp2_start_s + self.chirp_s + self.tail_silence_s


def assemble_block(
    utterances: Sequence[np.ndarray],
    fs: int,
    f_lo: float = DEFAULT_F_LO,
    f_hi: float = DEFAULT_F_HI,
    chirp_amplitude: float = DEFAULT_CHIRP_AMPLITUDE,
    **layout_kwargs: float,
) -> tuple[np.ndarray, BlockLayout]:
    """Build a playback block in the §6.5 format.

    Utterance durations are taken as ``len(u) / fs``; durations are exact in
    samples so the layout and the signal agree to the sample.
    """
    layout = BlockLayout([len(u) / fs for u in utterances], **layout_kwargs)
    x = np.zeros(int(round(layout.total_s * fs)))
    c = lfm_chirp(fs, f_lo, f_hi, layout.chirp_s, amplitude=chirp_amplitude)
    for t0 in (layout.chirp1_start_s, layout.chirp2_start_s):
        i = int(round(t0 * fs))
        x[i : i + len(c)] += c
    for u, t0 in zip(utterances, layout.utt_starts_s):
        i = int(round(t0 * fs))
        x[i : i + len(u)] += u
    return x, layout


def layout_from_timing(timing: dict) -> BlockLayout:
    """Build a :class:`BlockLayout` from a ``uwse.bench.playlist`` timing JSON.

    The bench writes sample offsets at ``timing["fs"]`` plus its own
    ``layout`` dict (``post_chirp_s``/``inter_utt_s``/``pre_chirp_s`` naming);
    this converts both to the alignment layout so the cut times are the ones
    that were actually played. The chirp ``f0_hz``/``f1_hz`` to pass to
    :func:`detect_chirps` are in ``timing["chirp"]``.
    """
    fs = float(timing["fs"])
    durs = [(u["end"] - u["start"]) / fs for u in timing["utterances"]]
    lay = dict(timing.get("layout", {}))
    names = {
        "lead_silence_s": "lead_silence_s",
        "chirp_s": "chirp_s",
        "post_chirp_s": "post_chirp_gap_s",
        "inter_utt_s": "inter_utt_gap_s",
        "pre_chirp_s": "pre_chirp_gap_s",
        "tail_silence_s": "tail_silence_s",
    }
    kw = {names[k]: float(v) for k, v in lay.items() if k in names}
    return BlockLayout(durs, **kw)


# ---------------------------------------------------------------------------
# Matched filter and PSR
# ---------------------------------------------------------------------------
@dataclass
class ChirpDetection:
    """One detected chirp onset."""

    index: int  # integer onset sample in the stream
    frac_index: float  # sub-sample onset (parabolic on the envelope)
    peak: float  # matched-filter envelope at the peak
    psr_db: float  # peak-to-sidelobe ratio


def matched_filter(x: np.ndarray, template: np.ndarray, envelope: bool = True) -> np.ndarray:
    """Matched-filter output ``c[k] = sum_n x[k+n] template[n]`` for every onset k.

    Length ``len(x) - len(template) + 1``. With ``envelope=True`` the magnitude
    of the analytic signal is returned: an LFM correlation oscillates at the
    band centre, and peak picking / interpolation on the envelope is unbiased.
    """
    c = correlate(np.asarray(x, np.float64), np.asarray(template, np.float64), mode="valid", method="fft")
    if envelope:
        c = np.abs(hilbert(c))
    return c


def peak_to_sidelobe_db(
    c: np.ndarray,
    peak_idx: int,
    mainlobe: int,
    window: int | None = None,
) -> float:
    """PSR = 20 log10(|c[peak]| / max |c| outside the main lobe), in dB.

    The sidelobe region is ``|k - peak| <= window`` (whole array if None) minus
    ``|k - peak| <= mainlobe``. Returns +inf if the sidelobe region is empty or
    exactly zero.
    """
    a = np.abs(c)
    lo = 0 if window is None else max(0, peak_idx - window)
    hi = len(a) if window is None else min(len(a), peak_idx + window + 1)
    seg = a[lo:hi].copy()
    m0 = max(0, peak_idx - mainlobe - lo)
    m1 = min(len(seg), peak_idx + mainlobe + 1 - lo)
    seg[m0:m1] = 0.0
    side = seg.max() if seg.size else 0.0
    if side <= 0:
        return float("inf")
    return float(20.0 * np.log10(a[peak_idx] / side))


def _parabolic(y: np.ndarray, i: int) -> float:
    """Vertex offset (in samples, |.| <= 0.5) of a parabola through y[i-1:i+2]."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2.0 * b + c
    if den == 0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def detect_chirps(
    x: np.ndarray,
    fs: int,
    n_chirps: int = 2,
    f_lo: float = DEFAULT_F_LO,
    f_hi: float = DEFAULT_F_HI,
    duration: float = DEFAULT_CHIRP_S,
    min_separation_s: float = 1.0,
    mainlobe_s: float | None = None,
    sidelobe_window_s: float = 0.1,
    template: np.ndarray | None = None,
) -> list[ChirpDetection]:
    """Detect the ``n_chirps`` strongest chirp onsets in a stream (§6.5 step 1).

    Peaks are picked greedily on the matched-filter envelope with a
    ``min_separation_s`` exclusion zone and returned in time order. The main
    lobe of an LFM envelope is about 1/B wide; by default the main lobe
    excluded from the PSR is ``3/B`` (B = f_hi - f_lo), and sidelobes are
    searched within ``sidelobe_window_s`` of the peak (the remainder of the
    block contains the other chirp and speech).
    """
    if template is None:
        template = lfm_chirp(fs, f_lo, f_hi, duration)
    env = matched_filter(x, template, envelope=True)
    bw = f_hi - f_lo
    mainlobe = int(np.ceil((mainlobe_s if mainlobe_s is not None else 3.0 / bw) * fs))
    mainlobe = max(mainlobe, 2)
    sep = int(round(min_separation_s * fs))
    win = int(round(sidelobe_window_s * fs))
    work = env.copy()
    found: list[ChirpDetection] = []
    for _ in range(n_chirps):
        i = int(np.argmax(work))
        if work[i] <= 0:
            break
        psr = peak_to_sidelobe_db(env, i, mainlobe, win)
        found.append(ChirpDetection(i, i + _parabolic(env, i), float(env[i]), psr))
        work[max(0, i - sep) : i + sep + 1] = 0.0
    return sorted(found, key=lambda d: d.index)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def estimate_drift_ppm(start_idx: float, end_idx: float, nominal_samples: float) -> float:
    """Clock drift (ppm) of a stream from its chirp onsets (§6.5 step 2).

    ``nominal_samples`` is the onset-to-onset interval on the reference clock,
    either ``layout.chirp_interval_s * fs`` (playlist) or the interval measured
    on the loopback stream. Positive drift = the stream's ADC clock runs fast
    (more samples than nominal).
    """
    return float(((end_idx - start_idx) / nominal_samples - 1.0) * 1e6)


def _kaiser_sinc(t: np.ndarray, half_width: int, beta: float, cutoff: float) -> np.ndarray:
    arg = np.clip(1.0 - (t / half_width) ** 2, 0.0, None)
    win = np.where(np.abs(t) < half_width, i0(beta * np.sqrt(arg)) / i0(beta), 0.0)
    return cutoff * np.sinc(cutoff * t) * win


def fractional_resample(
    x: np.ndarray,
    positions: np.ndarray,
    half_width: int = 16,
    beta: float = 8.6,
    cutoff: float = 0.97,
    chunk: int = 1 << 15,
) -> np.ndarray:
    """Band-limited interpolation of ``x`` at fractional sample ``positions``.

    Kaiser-windowed sinc with ``2*half_width`` taps; ``cutoff`` is relative to
    the input Nyquist (0.97 leaves a transition band). Samples outside ``x``
    are treated as zero. Processed in chunks to bound memory.
    """
    x = np.asarray(x, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    xp = np.concatenate([np.zeros(half_width), x, np.zeros(half_width + 1)])
    taps = np.arange(-half_width + 1, half_width + 1)
    out = np.empty(len(positions))
    for s in range(0, len(positions), chunk):
        p = positions[s : s + chunk]
        base = np.floor(p).astype(np.int64)
        frac = p - base
        idx = base[:, None] + taps[None, :] + half_width
        valid = (idx >= 0) & (idx < len(xp))
        vals = xp[np.clip(idx, 0, len(xp) - 1)] * valid
        h = _kaiser_sinc(frac[:, None] - taps[None, :], half_width, beta, cutoff)
        out[s : s + chunk] = np.sum(vals * h, axis=1)
    return out


def correct_drift(
    x: np.ndarray,
    drift_ppm: float,
    anchor: float = 0.0,
    n_out: int | None = None,
    **interp_kwargs: float,
) -> np.ndarray:
    """Resample a drifting stream onto the reference clock (§6.5 step 2).

    Output sample n (reference clock) is read at stream position
    ``anchor + (n - anchor) * (1 + drift_ppm*1e-6)``; ``anchor`` (e.g. the first
    chirp onset) keeps that instant fixed so playlist offsets measured from it
    stay valid.
    """
    r = 1.0 + drift_ppm * 1e-6
    if n_out is None:
        n_out = int(np.floor((len(x) - 1 - anchor) / r + anchor)) + 1
    pos = anchor + (np.arange(n_out) - anchor) * r
    return fractional_resample(x, pos, **interp_kwargs)


# ---------------------------------------------------------------------------
# Fine lag
# ---------------------------------------------------------------------------
@dataclass
class LagResult:
    """Per-utterance fine-alignment result (§6.5 steps 3-4)."""

    lag: float  # samples at fs; y[n] ~ g * ref[n - lag]
    lag_int: int
    psr_db: float
    flagged: bool  # PSR below threshold -> metadata flag "align_low_psr"
    peak: float  # normalized cross-correlation at the peak (signed)
    extra: dict = field(default_factory=dict)


def fine_lag(
    y: np.ndarray,
    ref: np.ndarray,
    fs: int = 16000,
    max_lag_s: float = 0.020,
    subsample: bool = False,
    mainlobe_s: float = 0.001,
    psr_threshold_db: float = 6.0,
) -> LagResult:
    """Fine lag of ``y`` against ``ref`` (R1) within +-``max_lag_s`` (§6.5 step 3).

    Cross-correlation over integer lags; the peak is the maximum of |c| (the
    hydrophone polarity is not assumed). With ``subsample`` the lag is refined
    by parabolic interpolation of |c|. PSR = peak over the largest |c| within
    the search window outside +-``mainlobe_s``; utterances with PSR below
    ``psr_threshold_db`` (6 dB, frozen on v0-dev) are flagged (step 4).
    """
    y = np.asarray(y, np.float64)
    ref = np.asarray(ref, np.float64)
    m = int(round(max_lag_s * fs))
    full = correlate(y, ref, mode="full", method="fft")
    zero = len(ref) - 1  # index of lag 0 in `full`
    lo, hi = max(0, zero - m), min(len(full), zero + m + 1)
    c = full[lo:hi]
    a = np.abs(c)
    k = int(np.argmax(a))
    lag_int = k + lo - zero
    lag = float(lag_int) + (_parabolic(a, k) if subsample else 0.0)
    mainlobe = max(1, int(round(mainlobe_s * fs)))
    psr = peak_to_sidelobe_db(c, k, mainlobe, None)
    denom = np.sqrt(np.sum(y * y) * np.sum(ref * ref))
    peak = float(c[k] / denom) if denom > 0 else 0.0
    return LagResult(lag, lag_int, psr, bool(psr < psr_threshold_db), peak)


def shift_signal(y: np.ndarray, lag: float, n_out: int | None = None) -> np.ndarray:
    """Undo a lag: ``out[n] = y[n + lag]`` (integer or fractional), zero outside."""
    n_out = len(y) if n_out is None else n_out
    if float(lag).is_integer():
        lag = int(lag)
        out = np.zeros(n_out)
        src0, src1 = max(0, lag), min(len(y), lag + n_out)
        if src1 > src0:
            out[src0 - lag : src1 - lag] = y[src0:src1]
        return out
    return fractional_resample(y, np.arange(n_out) + lag)


def fit_gain(y: np.ndarray, ref: np.ndarray) -> float:
    """Least-squares broadband gain g minimizing ||y - g*ref|| (§8.2 R1 fit)."""
    n = min(len(y), len(ref))
    d = float(np.dot(ref[:n], ref[:n]))
    return float(np.dot(y[:n], ref[:n]) / d) if d > 0 else 0.0


# ---------------------------------------------------------------------------
# Cutting
# ---------------------------------------------------------------------------
def cut_utterances(
    x: np.ndarray,
    fs: int,
    layout: BlockLayout,
    chirp1_index: float,
    drift_ppm: float = 0.0,
    pad_s: float = 0.25,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Cut each utterance of a block, with ``pad_s`` padding either side (§8.3).

    Utterance k starts at ``chirp1_index + (t_k - t_chirp1) * fs * (1 + drift)``
    stream samples, so the stream need not be drift-corrected first (pass
    ``drift_ppm=0`` if it was). Out-of-range samples are zero-filled. Returns
    the cuts and an ``(N, 2)`` array of integer (start, end) stream indices of
    each cut including padding.
    """
    r = 1.0 + drift_ppm * 1e-6
    pad = int(round(pad_s * fs))
    cuts, bounds = [], []
    for t0, t1 in zip(layout.utt_starts_s, layout.utt_ends_s):
        s = int(round(chirp1_index + (t0 - layout.chirp1_start_s) * fs * r)) - pad
        e = int(round(chirp1_index + (t1 - layout.chirp1_start_s) * fs * r)) + pad
        seg = np.zeros(e - s)
        a, b = max(0, s), min(len(x), e)
        if b > a:
            seg[a - s : b - s] = x[a:b]
        cuts.append(seg)
        bounds.append((s, e))
    return cuts, np.asarray(bounds, dtype=np.int64).reshape(-1, 2)


def align_block(
    stream: np.ndarray,
    fs: int,
    layout: BlockLayout,
    f_lo: float = DEFAULT_F_LO,
    f_hi: float = DEFAULT_F_HI,
    nominal_interval_samples: float | None = None,
    pad_s: float = 0.25,
    psr_threshold_db: float = 6.0,
    resample: bool = True,
) -> dict:
    """Steps 1, 2 and 4 of §6.5 for one block of one stream.

    Detects both chirps, estimates drift against ``nominal_interval_samples``
    (loopback interval; defaults to the playlist interval), optionally resamples
    to the reference clock anchored at the first chirp, then cuts the
    utterances. Fine lag (step 3) is done per utterance at 16 kHz with
    :func:`fine_lag` because it needs R1.

    Returns a dict with ``chirps``, ``drift_ppm``, ``chirp_psr_db`` (min of the
    two), ``low_psr`` flag, ``cuts`` and ``bounds``.
    """
    dets = detect_chirps(stream, fs, 2, f_lo, f_hi, layout.chirp_s)
    if len(dets) < 2:
        raise RuntimeError("fewer than two chirps detected")
    nominal = layout.chirp_interval_s * fs if nominal_interval_samples is None else nominal_interval_samples
    drift = estimate_drift_ppm(dets[0].frac_index, dets[1].frac_index, nominal)
    psr = min(d.psr_db for d in dets)
    if resample:
        x = correct_drift(stream, drift, anchor=dets[0].frac_index)
        cuts, bounds = cut_utterances(x, fs, layout, dets[0].frac_index, 0.0, pad_s)
    else:
        cuts, bounds = cut_utterances(stream, fs, layout, dets[0].frac_index, drift, pad_s)
    return {
        "chirps": dets,
        "drift_ppm": drift,
        "chirp_psr_db": psr,
        "low_psr": bool(psr < psr_threshold_db),
        "cuts": cuts,
        "bounds": bounds,
    }
