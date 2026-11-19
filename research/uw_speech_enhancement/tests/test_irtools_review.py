"""Adversarial review tests for uwse.irtools (RESEARCH_PLAN §4.3, §4.4, §6.4).

Each test pins a defect found in review (failing before the fix) or a property
the original suite did not exercise.
"""

import numpy as np
import pytest

from uwse.irtools import analysis as an
from uwse.irtools import ess as es

K60 = 6 * np.log(10)


def _ess_at(s, t):
    """Analytic synchronized ESS (no fades) evaluated at times t (s); 0 outside."""
    y = np.zeros_like(t)
    on = (t >= 0) & (t < len(s.excitation) / s.fs)
    y[on] = s.amplitude * np.sin(2 * np.pi * s.f1 * s.L * np.exp(t[on] / s.L))
    return y


def _drifted(sources, fs, ppm, n_total):
    """Sum of analytic sweeps ``(ess, start_s)`` sampled by a clock fast by ``ppm``."""
    t = np.arange(n_total) / (fs * (1 + ppm * 1e-6))
    return sum(_ess_at(s, t - t0) for s, t0 in sources)


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ppm", [50.0, -40.0])
def test_drift_from_identical_start_end_markers(ppm):
    """Brief: drift 'from matched-filter peak positions of start/end markers'.
    With two identical markers the coherent full-reference cross-correlation is
    ambiguous (drift decorrelates the end marker at the true lag), which used to
    lock the global lag onto marker-1-vs-marker-2 and fit the slope inside one
    0.2 s marker (48.4 ppm for 50)."""
    fs, dur = 16000, 10.0
    mk = es.generate_ess(500, 6000, 0.2, fs, fade_in_s=0, fade_out_s=0, silence_s=0)
    tm = len(mk.excitation) / fs
    ref = np.zeros(int(dur * fs))
    ref[: len(mk.excitation)] = mk.excitation
    n_end = len(ref) - len(mk.excitation)
    ref[n_end:] = mk.excitation
    d0 = 0.05
    rec = _drifted([(mk, d0), (mk, d0 + n_end / fs)], fs, ppm, len(ref) + int(0.2 * fs))
    assert tm < 0.25
    est = es.estimate_drift_ppm(rec, ref, fs, segment_s=0.1)
    assert abs(est - ppm) < 1.0, est


@pytest.mark.parametrize("ppm", [50.0, -50.0, 20.0])
def test_drift_from_faded_markers(ppm):
    """Same as above with the default raised-cosine fades on the markers. The
    drifted recording is made by band-limited resampling (``correct_drift`` with
    the inverse ratio; validated against an analytic sweep in test_irtools)."""
    fs = 16000
    mk = es.generate_ess(500, 6000, 0.2, fs, silence_s=0).excitation
    ref = np.zeros(10 * fs)
    ref[: len(mk)] = mk
    ref[-len(mk) :] = mk
    x = np.concatenate([np.zeros(800), ref, np.zeros(800)])
    eps = ppm * 1e-6
    rec = es.correct_drift(x, -eps / (1 + eps) * 1e6)  # rec[m] = x(m / (1 + eps))
    est = es.estimate_drift_ppm(rec, ref, fs, segment_s=0.1)
    assert abs(est - ppm) < 1.0, est


@pytest.mark.parametrize("snr_db,tol", [(20.0, 1.0), (0.0, 10.0)])
def test_drift_robust_to_noise(snr_db, tol):
    """White noise on the loopback. The estimator must degrade gracefully, not
    break down. Before the fix noise bins entered the phase unwrap and the
    integer lag slipped carrier cycles: 0 dB gave 1 ppm (and 77 ppm after a
    partial fix) for a true 50 ppm. The residual spread is noise-limited (it
    scales as 10^(-SNR/20): SD ~0.33 / 1.1 / 3.8 ppm at 20 / 10 / 0 dB over 8
    seeds) because an ESS puts its precise high-frequency segments at the end,
    so the drift lever arm is short; a realistic electrical loopback (> 40 dB)
    is well inside 1 ppm."""
    fs, ppm = 16000, 50.0
    s = es.generate_ess(100, 6000, 10.0, fs, fade_in_s=0, fade_out_s=0, silence_s=1.0)
    rng = np.random.default_rng(7)
    rec = _drifted([(s, 0.1)], fs, ppm, len(s.signal) + int(0.3 * fs))
    rec = rec + np.std(rec) * 10 ** (-snr_db / 20) * rng.standard_normal(len(rec))
    est = es.estimate_drift_ppm(rec, s.signal, fs)
    assert abs(est - ppm) < tol, est


@pytest.mark.parametrize("ppm", [50.0, -73.0, 400.0])
def test_correct_drift_matches_direct_kaiser_sinc(rng, ppm):
    """The tabulated kernel must equal the directly evaluated Kaiser sinc
    y[m] = sum_n x[n] sinc(n - m r) w(n - m r) (here evaluated independently)."""
    from scipy.special import i0

    x = rng.standard_normal(3000)
    r, ht, beta = 1 + ppm * 1e-6, 32, 8.0
    y = es.correct_drift(x, ppm)
    m = np.arange(len(y))[:, None]
    u = np.arange(len(x))[None, :] - m * r
    w = np.where(np.abs(u) < ht, i0(beta * np.sqrt(np.clip(1 - (u / ht) ** 2, 0, None))) / i0(beta), 0.0)
    # the implementation uses taps n0-31..n0+32 (u in (-32, 32]); match that support
    w = np.where((u > -ht) & (u <= ht), w, 0.0)
    y_ref = (np.sinc(u) * w) @ x
    err = 10 * np.log10(np.sum((y - y_ref) ** 2) / np.sum(y_ref**2))
    assert err < -100, err


def test_average_sweeps_coherence_lower_bound_is_zero(rng):
    """gamma^2 = |sum H|^2 / (S sum |H|^2) can reach 0 (two sweeps of opposite
    sign); the docstring's [1/S, 1] claim was wrong. Expected value for
    incoherent sweeps is ~1/S."""
    h = rng.standard_normal(2000)
    r = es.average_sweeps(np.stack([h, -h]), 16000)
    assert np.all(r.coherence < 1e-20)
    assert "[0, 1]" in es.SweepAverage.__doc__


# --------------------------------------------------------------------------- #
# Decay-slope model selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fc", [250, 500, 1000, 4000])
def test_single_slope_band_limited_not_misclassified(fc):
    """§4.4 'same model class': a single exponential decay, octave-filtered, must
    not be called two-slope. At 250 Hz the narrow band's EDC fluctuations used to
    be fitted by a tiny late component that spans < 10 dB (4/10 seeds)."""
    fs, T60 = 16000, 0.8
    n_two = 0
    for seed in range(10):
        r = np.random.default_rng(seed)
        t = np.arange(int(3 * fs)) / fs
        h = r.standard_normal(len(t)) * np.exp(-0.5 * K60 * t / T60) + 1e-3 * r.standard_normal(len(t))
        e = an.schroeder_edc(an.band_filter(h, fs, fc), fs)
        n_two += an.fit_decay_slopes(e.t, e.edc_db).n_slopes == 2
    assert n_two == 0, n_two


@pytest.mark.parametrize("Te,Tl,a2_db", [(0.25, 1.2, -25.0), (0.3, 0.9, -15.0), (0.2, 0.6, -20.0)])
def test_two_slope_still_detected(Te, Tl, a2_db):
    """The extra guard must not remove genuine double decays (ratio >= 2.5)."""
    fs = 16000
    t = np.arange(int(3.0 * fs)) / fs
    r = np.random.default_rng(11)
    a2 = 10 ** (a2_db / 20)
    h = r.standard_normal(len(t)) * np.exp(-0.5 * K60 * t / Te) + a2 * r.standard_normal(len(t)) * np.exp(
        -0.5 * K60 * t / Tl
    )
    h = h + 1e-5 * r.standard_normal(len(t))
    e = an.schroeder_edc(h, fs)
    fit = an.fit_decay_slopes(e.t, e.edc_db)
    assert fit.n_slopes == 2
    knee_true = np.log((Te / Tl) / a2**2) / (K60 / Te - K60 / Tl)
    assert abs(fit.knee_s / knee_true - 1) < 0.2, (fit.knee_s, knee_true)


# --------------------------------------------------------------------------- #
# Arrivals
# --------------------------------------------------------------------------- #
def test_arrivals_direct_at_first_sample():
    """An IR whose direct path is at sample 0 (simulator output, or a measured IR
    trimmed at zero lag) lost its direct path: find_peaks never reports edge
    samples, so the first reflection became 'direct' (delays shifted, levels +6 dB)."""
    fs = 48000
    h = np.zeros(4800)
    h[0], h[100], h[200] = 1.0, 0.5, 0.25
    arr = an.detect_arrivals(h, fs)
    np.testing.assert_allclose(arr.delays_s * fs, [0, 100, 200], atol=0.5)
    np.testing.assert_allclose(arr.levels_db, 20 * np.log10([1, 0.5, 0.25]), atol=0.5)


# --------------------------------------------------------------------------- #
# Coloration
# --------------------------------------------------------------------------- #
def test_third_octave_level_independent_of_ir_length(rng):
    """Zero-padding an IR does not change its spectrum; the band energy must not
    scale with the FFT length (it was a sum of |H|^2 bins: +7 dB for 8x padding),
    otherwise ``spectral_distance_db(remove_gain=False)`` compares IRs of
    different lengths with a spurious offset."""
    fs = 16000
    h = rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300)
    hp = np.concatenate([h, np.zeros(14000)])
    assert an.spectral_distance_db(h, hp, fs, 1000, 6400, remove_gain=False) < 0.05
    # Parseval: the band energies of a white IR sum to ~ its energy in the analysed range
    w = rng.standard_normal(4096)
    _, lv = an.third_octave_spectrum(w, fs, 1000, 6400)
    lo, hi = an.band_edges(1000, 3)[0], an.band_edges(6400, 3)[1]
    expect = np.sum(w**2) * (hi - lo) / (fs / 2)
    assert abs(10 * np.log10(np.sum(10 ** (lv / 10)) / expect)) < 0.5
