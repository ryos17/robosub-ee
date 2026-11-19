"""Same-day analysis of a pool session's non-speech takes (RESEARCH_PLAN.md 4.3, 4.4, 5.2, 6.4).

Hardware-free, so it is tested on synthetic takes; ``scripts/analyze_session.py``
finds the takes of a session directory and writes the results.

- **Sweep takes** (``sweeps.wav``: sync chirp, then S exponential sweeps with
  silence tails). `sweep_irs`: clock drift and offset of the recording against the
  played file on the recorder's sync channel (loopback if recorded, else its first
  hydrophone), drift correction, one IR per sweep and hydrophone by regularized
  deconvolution against the regenerated excitation. `ir_metrics`: mean IR, sweep
  coherence and between-sweep variance, broadband and per-octave EDC (Schroeder with
  Lundeby truncation), EDT/T20/T30, single- vs two-slope fit, C50, DRR (direct-path
  gate; None where the gate rule fails, 8.2) and per-band IR SNR. Uncalibrated:
  levels are relative, times are absolute.
- **Noise takes** (``<pos>_noise``, ``<pos>_noise_on``, the ``noise_<name>`` bank and
  any E4 take). `noise_spectrum`: mean PSD over the kept frames (same trim and 6 dB
  median gate as the signed-off SNR method), 1/3-octave band levels with their
  frame-to-frame spread (stationarity), and the SNR-method band powers, in dBFS.
  `compare_spectra`: band levels of each condition relative to a reference take.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from uwse.bench import snr as SN
from uwse.irtools import analysis as an
from uwse.irtools import ess as es

__all__ = [
    "IR_PRE_S",
    "DRR_GATE_S",
    "ARRIVAL_DB",
    "excitation_from_meta",
    "locate",
    "SweepIRs",
    "sweep_irs",
    "ir_metrics",
    "third_octave_edges",
    "noise_spectrum",
    "compare_spectra",
]

IR_PRE_S = 0.010  # negative-time samples kept in front of each IR (band-limit ringing, sync on a hydrophone)
DRR_GATE_S = 0.7e-3  # direct-path gate (RESEARCH_PLAN.md 6.4: >= 0.7 ms, valid above about 3 kHz)
#: arrivals weaker than this (dB re the strongest) are not counted as reflections: the deconvolved
#: IR's band-edge ringing sits near -28 dB and would otherwise be taken for the first reflection
ARRIVAL_DB = -20.0


# ------------------------------------------------------------------ sweeps
def excitation_from_meta(sw: dict) -> es.ESS:
    """The sweep that ``make_session_audio.py`` wrote, regenerated from ``session/sweeps.json``."""
    e = es.generate_ess(sw["f1"], sw["f2"], sw["T"], sw["fs"], amplitude=10 ** (sw["peak_dBFS"] / 20),
                        silence_s=sw["silence_s"], synchronized=sw.get("synchronized", True),
                        lf_shelf_hz=sw.get("lf_shelf_hz"), lf_shelf_db=sw.get("lf_shelf_db", 0.0))
    if len(e.excitation) != sw["sweep_n_samples"]:
        raise ValueError(f"regenerated sweep has {len(e.excitation)} samples, sweeps.json says "
                         f"{sw['sweep_n_samples']}: generator or metadata changed")
    return e


def _parabolic(y: np.ndarray, i: int) -> float:
    if 0 < i < len(y) - 1:
        den = y[i - 1] - 2 * y[i] + y[i + 1]
        if den != 0:
            return i + float(np.clip(0.5 * (y[i - 1] - y[i + 1]) / den, -0.5, 0.5))
    return float(i)


def locate(sync: np.ndarray, e: es.ESS, starts: Sequence[int], fs: int, max_ppm: float = 1000.0) -> dict[str, float]:
    """Clock drift (ppm) and offset of a recording against the played file, from the sweeps themselves.

    ``starts`` are the sweep starts in the played file (``sweeps.json``). A played sample s
    is at recording sample ``s (1 + ppm 1e-6) + offset``; after ``correct_drift(x, ppm)``
    it is at ``s + offset / (1 + ppm 1e-6)``.

    Coarse lag: the earliest strong match of one sweep (>= half the best match), taken
    as sweep 0. Correlating the whole played file instead is ambiguous: its S sweeps are
    identical, and with tens of ppm drift the S matched pairs no longer add up, so the
    peak lands a whole number of sweeps off (review, 2026-11-09). Then each sweep is
    deconvolved where it should be, sweep k's IR is cross-correlated with sweep 0's
    (sub-sample, parabolic), and a line ``arrival_k = r s_k + offset`` is fitted, as
    `uwse.bench.ingest` fits the chirp onsets. Comparing whole IRs, not single peaks,
    keeps the fit on one arrival when the sync channel is a reverberant hydrophone;
    there ``offset`` puts the IR's strongest arrival at zero lag. ``resid_samples``
    (rms fit residual) is the QC: more than a sample or two means a sweep locked onto
    a different arrival or a sweep is missing.
    """
    from scipy.signal import correlate, hilbert

    sync = np.asarray(sync, float)
    if len(sync) < len(e.excitation):
        raise ValueError(f"recording ({len(sync) / fs:.2f} s) is shorter than one sweep")
    env = np.abs(hilbert(correlate(sync, e.excitation, mode="valid", method="fft")))
    strong = np.flatnonzero(env >= 0.5 * env.max())
    first = strong[0]
    # refine to the envelope peak of that first cluster (a strong reflection may follow within ms)
    j = first + int(np.argmax(env[first:first + int(0.05 * fs)]))
    coarse = int(j) - int(starts[0])
    # drift moves sweep k by up to max_ppm x its distance from sweep 0 (the coarse anchor)
    pad = int(np.ceil(max_ppm * 1e-6 * (starts[-1] - starts[0] + len(e.excitation)))) + int(0.01 * fs)
    n_h = 2 * pad + int(0.05 * fs)
    seg_starts, irs = [], []
    for s in starts:
        a = s + coarse - pad
        if a < 0 or a + len(e.excitation) + n_h > len(sync):
            raise ValueError(f"sweep at {s / fs:.2f} s of the played file is outside the recording "
                             f"({len(sync) / fs:.2f} s, coarse offset {coarse / fs:+.2f} s)")
        irs.append(es.deconvolve(sync[a:a + len(e.excitation) + n_h], e.excitation, fs, f_band=(e.f1, e.f2),
                                 n_out=n_h))
        seg_starts.append(a)
    env = np.abs(hilbert(irs[0]))
    p0 = _parabolic(env, int(np.argmax(env)))
    arr = []
    for a, h in zip(seg_starts, irs):
        xc = correlate(h, irs[0], mode="full", method="fft")
        arr.append(a + p0 + _parabolic(xc, int(np.argmax(xc))) - (len(h) - 1))
    s = np.asarray(starts, float)
    arr = np.asarray(arr)
    if len(s) >= 2:
        r, d0 = np.polyfit(s, arr, 1)
        resid = float(np.sqrt(np.mean((arr - (r * s + d0)) ** 2)))
    else:
        r, d0, resid = 1.0, float(arr[0] - s[0]), float("nan")
    return {"ppm": float((r - 1.0) * 1e6), "offset": float(d0), "resid_samples": resid,
            "coarse_offset": coarse}


@dataclass
class SweepIRs:
    irs: np.ndarray  # (C, S, n_pre + n_out), zero lag at index n_pre
    fs: int
    n_pre: int
    channel_ids: list[int]
    ppm: float
    offset_s: float
    sync: int
    resid_samples: float


def sweep_irs(x: np.ndarray, fs: int, sw: dict, hydrophones: dict[int, int], sync: int,
              max_ppm: float = 1000.0, pre_s: float = IR_PRE_S) -> SweepIRs:
    """One IR per sweep and hydrophone of a recorded sweep take.

    ``x`` is the recording (C, T) at ``fs`` (= ``sw['fs']``), ``sw`` the bench's
    ``session/sweeps.json``, ``hydrophones`` file channel -> benchmark channel id.
    """
    if fs != sw["fs"]:
        raise ValueError(f"recording at {fs} Hz, sweeps at {sw['fs']} Hz: resample first")
    e = excitation_from_meta(sw)
    loc = locate(x[sync], e, sw["sweep_start_samples"], fs, max_ppm)
    r = 1.0 + loc["ppm"] * 1e-6
    shift = loc["offset"] / r
    n_pre = int(round(pre_s * fs))
    n_sig = len(e.signal)
    n_out = n_sig - len(e.excitation) + 1
    chans = sorted(hydrophones)
    out = np.zeros((len(chans), len(sw["sweep_start_samples"]), n_pre + n_out))
    for i, c in enumerate(chans):
        y = es.correct_drift(x[c], loc["ppm"])
        for k, s in enumerate(sw["sweep_start_samples"]):
            a = int(round(s + shift))
            if a < 0 or a + n_sig > len(y):
                raise ValueError(f"sweep {k + 1} at {a / fs:.2f} s is outside the recording ({len(y) / fs:.2f} s)")
            out[i, k] = es.measure_ir(y[a:a + n_sig], e, n_out=n_out, n_pre=n_pre)
    return SweepIRs(out, fs, n_pre, [hydrophones[c] for c in chans], loc["ppm"], loc["offset"] / fs, sync,
                    loc["resid_samples"])


def _f(v: Any) -> float | None:
    """JSON-safe float (None for NaN / inf / None)."""
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def _slopes(t: np.ndarray, edc_db: np.ndarray) -> dict:
    """Single- vs two-slope fit fields; None where the EDC is too short to fit (e.g. a silent channel)."""
    try:
        sl = an.fit_decay_slopes(t, edc_db)
    except ValueError:
        return {"n_slopes": None, "T60_slopes_s": None, "knee_s": None}
    return {"n_slopes": sl.n_slopes, "T60_slopes_s": [_f(v) for v in sl.T60], "knee_s": _f(sl.knee_s)}


def ir_metrics(irs: np.ndarray, fs: int, n_pre: int, f_band: tuple[float, float],
               centers: Sequence[float] = an.OCTAVE_CENTERS, drr_gate_s: float = DRR_GATE_S) -> tuple[dict, dict]:
    """Metrics of one hydrophone's sweep IRs (S, n): (JSON-ready row, arrays for the npz)."""
    irs = np.asarray(irs, float)
    S = irs.shape[0]
    if S >= 2:
        avg = es.average_sweeps(irs, fs, centers)
        h = avg.ir_mean
        coh = {"band_centers": avg.band_centers.tolist(), "band_coherence": [_f(v) for v in avg.band_coherence],
               "band_var_db": [_f(v) for v in avg.band_var_db]}
    else:
        h, coh = irs[0], {}
    # Metrics run on the whole IR, pre-roll included: with acoustic sync the direct path is at zero lag
    # and cutting there drops the zero-phase octave filters' pre-ringing (octave C50 off by up to 18 dB).
    # EDC, C50 and EDT/T20/T30 start at the ISO 3382 onset; arrival times are re zero lag.
    # octaves inside the excited band only (and below 0.95 Nyquist)
    centers = [c for c in centers if an.band_edges(c)[0] >= f_band[0] and an.band_edges(c)[1] <= min(f_band[1], 0.95 * fs / 2)]
    arr = an.detect_arrivals(h, fs, n=3, threshold_db=ARRIVAL_DB)
    t_first = float(arr.delays_s[1]) if len(arr.delays_s) > 1 else None
    edc = an.schroeder_edc(h, fs)
    dt = an.decay_times(h, fs)
    row: dict[str, Any] = {
        "n_sweeps": S, "t_direct_s": _f(arr.times_s[0] - n_pre / fs), "first_reflection_delay_s": _f(t_first),
        "arrival_delays_s": [_f(v) for v in arr.delays_s], "arrival_levels_db": [_f(v) for v in arr.levels_db],
        "broadband": {"EDT_s": _f(dt["EDT"]), "T20_s": _f(dt["T20"]), "T30_s": _f(dt["T30"]),
                      "C50_dB": _f(an.c50_db(h, fs)),
                      "DRR_dB": _f(an.drr_db(h, fs, drr_gate_s, t_first, band_lo_hz=f_band[0])),
                      **_slopes(edc.t, edc.edc_db), "ir_snr_dB": _f(an.ir_snr_db(h, fs, centers=None))},
        "drr_gate_s": drr_gate_s, **coh,
    }
    bands = an.filterbank(h, fs, centers)
    per = []
    arrays = {"ir": h, "edc_t": edc.t, "edc_db": edc.edc_db, "band_centers": np.asarray(centers, float)}
    for fc, hb in zip(centers, bands):
        d = an.schroeder_edc(hb, fs)
        bt = an.decay_times(hb, fs)
        per.append({"fc": fc, "EDT_s": _f(bt["EDT"]), "T20_s": _f(bt["T20"]), "T30_s": _f(bt["T30"]),
                    "C50_dB": _f(an.c50_db(hb, fs)),
                    "DRR_dB": _f(an.drr_db(hb, fs, drr_gate_s, t_first, band_lo_hz=an.band_edges(fc)[0])),
                    **_slopes(d.t, d.edc_db),
                    "ir_snr_dB": _f(an.ir_snr_db(hb, fs, centers=None))})
        arrays[f"edc_db_{int(fc)}"] = d.edc_db
    row["octaves"] = per
    return row, arrays


# ------------------------------------------------------------------ noise
def third_octave_edges(fs: int, f_lo: float = 25.0, f_hi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Base-10 1/3-octave centres (1000 * 10^(k/10)) from ``f_lo`` up to the last band below Nyquist."""
    f_hi = f_hi or fs / 2
    k = np.arange(np.ceil(10 * np.log10(f_lo / 1000)), np.floor(10 * np.log10(f_hi / 1000)) + 1)
    fc = 1000.0 * 10 ** (k / 10)
    fc = fc[fc * 10 ** 0.05 < fs / 2]
    return fc, np.stack([fc * 10 ** -0.05, fc * 10 ** 0.05], 1)


def noise_spectrum(x: np.ndarray, fs: int, trim_s: float = 1.0, gate_db: float | None = SN.NOISE_GATE_DB,
                   nperseg: int | None = None) -> dict:
    """PSD and band levels of a noise-only take (one channel), in dBFS.

    Trim and frame gate as `uwse.bench.snr.noise_powers` (frames within ``gate_db``
    below the median 50-8000 Hz frame level are kept). ``nperseg`` defaults to
    about 85 ms (Hann, 50 % overlap). The PSD is the mean one-sided density over the
    kept frames in dB re 1 FS^2/Hz; a band level is 10 log10 of the band's mean square
    (a full-scale sine is -3 dB).
    """
    x = np.asarray(x, float)
    k = int(round(trim_s * fs))
    xt = x[k:x.size - k] if x.size > 4 * k else x
    nper = nperseg or 1 << int(round(np.log2(0.085 * fs)))
    if xt.size < nper:
        raise ValueError(f"noise take of {xt.size / fs:.2f} s after trimming is shorter than one frame")
    w = np.hanning(nper + 1)[:-1]  # periodic Hann
    fr = np.lib.stride_tricks.sliding_window_view(xt, nper)[::nper // 2]
    P = (np.abs(np.fft.rfft(fr * w, axis=1)) ** 2).T / (fs * np.sum(w ** 2))  # (F, frames)
    P[1:(nper + 1) // 2] *= 2.0  # one-sided: double all but DC (and Nyquist for even nper)
    f = np.fft.rfftfreq(nper, 1 / fs)
    df = f[1] - f[0]
    wb = (f >= 50) & (f < 8000)
    lev = 10 * np.log10(P[wb].sum(0) * df + 1e-30)
    keep = lev >= np.median(lev) - gate_db if gate_db is not None else np.ones(lev.size, bool)
    psd = P[:, keep].mean(1)
    fc, edges = third_octave_edges(fs)
    band = np.stack([(f >= lo) & (f < hi) for lo, hi in edges])
    ok = band.sum(1) >= 2  # low bands narrower than two FFT bins are dropped
    fc, band = fc[ok], band[ok]
    fr_lv = 10 * np.log10(band.astype(float) @ P[:, keep] * df + 1e-30)  # (bands, kept frames)
    pw = SN.noise_powers(x, fs, trim_s=trim_s, gate_db=gate_db)
    return {
        "fs": fs, "duration_s": x.size / fs, "frames": int(lev.size), "frames_kept": int(keep.sum()),
        "freqs": f, "psd_db": 10 * np.log10(psd + 1e-30),
        "third_octave_fc": fc, "third_octave_db": 10 * np.log10(band.astype(float) @ psd * df + 1e-30),
        "third_octave_std_db": fr_lv.std(1), "third_octave_p10_db": np.percentile(fr_lv, 10, axis=1),
        "third_octave_p90_db": np.percentile(fr_lv, 90, axis=1),
        "band_db": {b: 10 * np.log10(p + 1e-30) for b, p in pw.items()},
        "broadband_db": float(10 * np.log10(np.mean(xt ** 2) + 1e-30)),
    }


def compare_spectra(spectra: dict[str, dict], ref: str) -> dict[str, dict]:
    """Per condition: 1/3-octave and SNR-band levels minus those of ``ref`` (same fs and channel)."""
    r = spectra[ref]
    out = {}
    for name, s in spectra.items():
        if name == ref:
            continue
        if not np.array_equal(s["third_octave_fc"], r["third_octave_fc"]):
            raise ValueError(f"{name} and {ref} have different band grids (sample rates {s['fs']} vs {r['fs']})")
        out[name] = {"ref": ref, "third_octave_fc": r["third_octave_fc"],
                     "third_octave_re_ref_db": s["third_octave_db"] - r["third_octave_db"],
                     "band_re_ref_db": {b: s["band_db"][b] - r["band_db"][b] for b in r["band_db"]}}
    return out
