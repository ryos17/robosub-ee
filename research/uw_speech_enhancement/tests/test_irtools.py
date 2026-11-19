"""Synthetic / analytic tests for uwse.irtools (RESEARCH_PLAN §4.3, §4.4, §6.4)."""

import numpy as np
import pytest
from numpy.fft import rfft, rfftfreq
from scipy.signal import fftconvolve

from uwse.irtools import analysis as an
from uwse.irtools import ess as es
from uwse.irtools import soundspeed as ss

K60 = 6 * np.log(10)


def _inband_err_db(h_est, h_true, fs, lo, hi, nfft):
    f = rfftfreq(nfft, 1 / fs)
    sel = (f >= lo) & (f <= hi)
    He, Ht = rfft(h_est, nfft)[sel], rfft(h_true, nfft)[sel]
    return 10 * np.log10(np.sum(np.abs(He - Ht) ** 2) / np.sum(np.abs(Ht) ** 2))


# --------------------------------------------------------------------------- #
# ESS
# --------------------------------------------------------------------------- #
def test_ess_basic_properties():
    s = es.generate_ess(50, 7000, 2.0, 16000, silence_s=0.5)
    assert abs(s.f1 * s.L - round(s.f1 * s.L)) < 1e-9  # synchronized
    assert abs(s.T - 2.0) < 0.05
    assert len(s.signal) == len(s.excitation) + 8000
    np.testing.assert_allclose(es.harmonic_offsets(50, 7000, s.T, 3), [s.harmonic_offset(k) for k in (1, 2, 3)])
    assert abs(es.processing_gain_db(10, 20000) - 53.0) < 0.1
    # LF pre-shaping lowers the start of the sweep, leaves the top alone
    sh = es.generate_ess(50, 7000, 2.0, 16000, lf_shelf_hz=400, lf_shelf_db=12, fade_in_s=0, fade_out_s=0)
    n = len(sh.excitation)
    assert np.abs(sh.excitation[: n // 10]).max() < 10 ** (-10 / 20)
    assert np.abs(sh.excitation[-n // 10 :]).max() > 0.99


@pytest.mark.parametrize("shaped", [False, True])
def test_ess_deconvolution_recovers_fir(rng, shaped):
    fs = 16000
    kw = dict(lf_shelf_hz=300, lf_shelf_db=15) if shaped else {}
    s = es.generate_ess(50, 7000, 2.0, fs, silence_s=0.5, **kw)
    fir = rng.standard_normal(256) * np.exp(-np.arange(256) / 40)
    rec = fftconvolve(s.signal, fir)
    # band-limited IR rings symmetrically (zero-phase band limitation): keep pre-roll
    pre = 2048
    h = es.measure_ir(rec, s, n_out=8000, n_pre=pre)
    err = _inband_err_db(h, np.concatenate([np.zeros(pre), fir]), fs, 100, 6000, 16384)
    assert err < -40, err


def test_harmonic_levels_memoryless_polynomial():
    fs, A, a2, a3 = 16000, 0.5, 0.1, 0.2
    s = es.generate_ess(50, 7000, 2.0, fs, amplitude=A, silence_s=0.3)
    x = s.signal
    y = x + a2 * x**2 + a3 * x**3
    hir = es.harmonic_irs(y, s, K=3)
    f = rfftfreq(hir.irs.shape[1], 1 / fs)
    band = (f > 400) & (f < 5000)
    lev = [np.median(np.abs(rfft(h))[band]) for h in hir.irs]
    # H1 = 1 + 3/4 a3 A^2, |H2| = a2 A / 2, |H3| = a3 A^2 / 4 (Novak 2010)
    np.testing.assert_allclose(lev, [1 + 0.75 * a3 * A**2, a2 * A / 2, a3 * A**2 / 4], rtol=0.02)


def test_hammerstein_kernels_recovered(rng):
    fs, A = 16000, 0.7
    s = es.generate_ess(50, 7000, 2.0, fs, amplitude=A, silence_s=0.3)
    g = [rng.standard_normal(48) * np.exp(-np.arange(48) / 8) for _ in range(3)]
    g[1] *= 0.1
    g[2] *= 0.05
    y = es.hammerstein_apply(s.signal, g)
    hir = es.harmonic_irs(y, s, K=3)
    ker = es.hammerstein_kernels(hir)
    n = ker.shape[1]
    for k in range(3):
        true = np.zeros(n)
        true[hir.pre : hir.pre + 48] = g[k]
        err = _inband_err_db(ker[k], true, fs, 400, 5000, n)
        assert err < -25, (k + 1, err)
    # the matrix for K = 3 has the closed form of Novak et al. (2010)
    M = es.hammerstein_matrix(3, A)
    np.testing.assert_allclose(M[0], [1, 0, 0.75 * A**2], atol=1e-12)
    np.testing.assert_allclose(M[1], [0, -0.5j * A, 0], atol=1e-12)
    np.testing.assert_allclose(M[2], [0, 0, -0.25 * A**2], atol=1e-12)


def _drifted_sweep(s, ppm, d0_s, n_total):
    """Analytic synchronized ESS sampled by a clock running fast by ``ppm``."""
    n = np.arange(n_total)
    t = n / (s.fs * (1 + ppm * 1e-6)) - d0_s
    on = (t >= 0) & (t < len(s.excitation) / s.fs)
    y = np.zeros(n_total)
    y[on] = s.amplitude * np.sin(2 * np.pi * s.f1 * s.L * np.exp(t[on] / s.L))
    return y


def test_drift_estimate_and_correction():
    fs, ppm, d0 = 16000, 50.0, 0.1
    s = es.generate_ess(100, 6000, 10.0, fs, fade_in_s=0, fade_out_s=0, silence_s=1.0)
    rec = _drifted_sweep(s, ppm, d0, len(s.signal) + int(0.3 * fs))
    est = es.estimate_drift_ppm(rec, s.signal, fs)
    assert abs(est - ppm) < 1.0, est
    # correction with the true value reproduces the reference timeline
    cor = es.correct_drift(rec, ppm)
    n0 = int(d0 * fs)
    seg = slice(fs // 2, len(s.excitation) - fs // 2)
    ref = s.excitation[seg]
    got = cor[n0:][seg]
    err = 10 * np.log10(np.sum((got - ref) ** 2) / np.sum(ref**2))
    assert err < -40, err
    # after correcting with the estimate, the residual drift is < 1 ppm
    cor_est = es.correct_drift(rec, est)
    assert abs(es.estimate_drift_ppm(cor_est, s.signal, fs)) < 1.0


def test_average_sweeps_coherence(rng):
    fs = 16000
    h = rng.standard_normal(4000) * np.exp(-np.arange(4000) / 800)
    same = es.average_sweeps(np.stack([h] * 4), fs)
    assert np.all(same.band_coherence > 0.999)
    noisy = es.average_sweeps(np.stack([h + 0.3 * rng.standard_normal(4000) for _ in range(4)]), fs)
    assert np.all(noisy.band_coherence < same.band_coherence)
    assert np.all(np.isfinite(noisy.band_var_db))
    np.testing.assert_allclose(noisy.ir_mean.shape, h.shape)


# --------------------------------------------------------------------------- #
# Decay analysis
# --------------------------------------------------------------------------- #
def _decay_ir(rng, fs, T60, dur, noise_db):
    t = np.arange(int(dur * fs)) / fs
    h = rng.standard_normal(len(t)) * np.exp(-0.5 * K60 * t / T60)
    return np.concatenate([np.zeros(100), h]) + 10 ** (noise_db / 20) * rng.standard_normal(len(t) + 100)


def test_lundeby_t60_within_5pct(rng):
    fs, T60 = 16000, 0.8
    h = _decay_ir(rng, fs, T60, 3.0, -60)
    d = an.decay_times(h, fs)
    for k in ("T20", "T30"):
        assert abs(d[k] / T60 - 1) < 0.05, (k, d[k])
    edc = an.schroeder_edc(h, fs)
    # crosspoint near where the decay meets the -60 dB floor (0.8 s)
    assert 0.6 < edc.lundeby.n_cross / fs < 1.1
    # without truncation the noise floor bends T30 upward
    raw = an.schroeder_edc(h, fs, use_lundeby=False)
    assert an.decay_time(raw.t, raw.edc_db, -5, -35) > d["T30"]


def test_band_decay_times(rng):
    fs, T60 = 16000, 0.6
    h = _decay_ir(rng, fs, T60, 2.5, -70)
    bt = an.band_decay_times(h, fs, centers=(500, 1000, 2000, 4000))
    assert np.all(np.abs(bt["T20"] / T60 - 1) < 0.1)
    assert np.all(bt["snr_db"] > 40)


def test_two_slope_detected_single_not(rng):
    fs, Te, Tl, a2_db = 16000, 0.25, 1.2, -25.0
    t = np.arange(int(3.0 * fs)) / fs
    a2 = 10 ** (a2_db / 20)
    h = rng.standard_normal(len(t)) * np.exp(-0.5 * K60 * t / Te) + a2 * rng.standard_normal(len(t)) * np.exp(
        -0.5 * K60 * t / Tl
    )
    h = h + 1e-5 * rng.standard_normal(len(t))
    edc = an.schroeder_edc(h, fs)
    fit = an.fit_decay_slopes(edc.t, edc.edc_db)
    assert fit.n_slopes == 2
    knee_true = np.log((Te / Tl) / a2**2) / (K60 / Te - K60 / Tl)
    assert abs(fit.knee_s / knee_true - 1) < 0.2, (fit.knee_s, knee_true)
    assert abs(fit.T60[0] / Te - 1) < 0.15 and abs(fit.T60[1] / Tl - 1) < 0.15
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        hs = _decay_ir(r, fs, 0.8, 3.0, -60)
        e = an.schroeder_edc(hs, fs)
        assert an.fit_decay_slopes(e.t, e.edc_db).n_slopes == 1, seed


def test_drr_c50_analytic():
    fs, nd, nr, tau, g = 48000, 480, 96, 0.05, 0.05
    N = fs
    q = np.exp(-1 / (tau * fs))
    h = np.zeros(N)
    h[nd] = 1.0
    M = N - nd - nr
    h[nd + nr :] = g * q ** np.arange(M)

    def geo(j0, j1):  # sum_{j=j0}^{j1-1} q^{2j}
        return (q ** (2 * j0) - q ** (2 * j1)) / (1 - q**2)

    drr = an.drr_db(h, fs, gate_s=1e-3, t_first_reflection_s=nr / fs)
    assert abs(drr - (-10 * np.log10(g**2 * geo(0, M)))) < 1e-9
    n50 = int(0.05 * fs) - nr
    c50 = an.c50_db(h, fs)
    expect = 10 * np.log10((1 + g**2 * geo(0, n50)) / (g**2 * geo(n50, M)))
    assert abs(c50 - expect) < 1e-9
    # gate rule: gate reaching the first reflection, or too few cycles in band
    assert an.drr_db(h, fs, gate_s=3e-3, t_first_reflection_s=nr / fs) is None
    assert an.drr_db(h, fs, gate_s=1e-3, band_lo_hz=250) is None


def test_ir_snr_and_arrivals(rng):
    fs = 48000
    h = np.zeros(fs // 2)
    for t_ms, a in ((1.0, 1.0), (1.69, -0.5), (2.5, 0.25), (4.0, 0.125)):
        h[int(round(t_ms * 1e-3 * fs))] = a
    arr = an.detect_arrivals(h, fs, n=5)
    np.testing.assert_allclose(arr.delays_s * 1e3, [0, 0.6875, 1.5, 3.0], atol=1 / 48)
    np.testing.assert_allclose(arr.levels_db, 20 * np.log10([1, 0.5, 0.25, 0.125]), atol=0.5)
    hn = h + 1e-3 * rng.standard_normal(len(h))
    snr = an.ir_snr_db(hn, fs, centers=None)
    assert 50 < float(snr) < 70


def test_third_octave_distance(rng):
    fs = 16000
    h = rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300)
    c, lv = an.third_octave_spectrum(h, fs, 1000, 8000)
    assert np.allclose(c[0], 1000) and len(c) == 9  # 8 kHz band upper edge > Nyquist dropped
    assert an.spectral_distance_db(h, 3 * h, fs, 1000, 6400) < 1e-9
    hb = h + an.band_filter(h, fs, 2000, fraction=3)  # +6 dB bump at 2 kHz
    assert an.spectral_distance_db(h, hb, fs, 1000, 6400) > 1.0


# --------------------------------------------------------------------------- #
# Sound speed
# --------------------------------------------------------------------------- #
def test_marczak_appendix_values():
    np.testing.assert_allclose(ss.marczak_c([20, 25, 27, 28]), [1482.4, 1496.7, 1501.9, 1504.4], atol=0.06)
    assert abs(ss.marczak_dcdT(26) - 2.6) < 0.1


def test_plan_relative_se_reproduces():
    """§4.3 / App. A: ranges {1,3,6,10} m, +-3 cm -> rel SE(c) ~ 0.44 % (0.74 % at +-5 cm).
    Reproduces when "+-3 cm" is read as a standard deviation: sigma_r / sqrt(Sxx)."""
    r = [1, 3, 6, 10]
    assert abs(ss.predicted_rel_se_c(r, 0.03) - 0.0044) < 0.00005
    assert abs(ss.predicted_rel_se_c(r, 0.05) - 0.0074) < 0.00005
    mc = ss.monte_carlo_tof_fit(r, range_sd_m=0.03, n_trials=40000, rng=np.random.default_rng(1))
    assert abs(mc["rel_sd_c"] / 0.00442 - 1) < 0.03
    # as a uniform half-width the figure would be 0.255 %, not 0.44 %
    mcu = ss.monte_carlo_tof_fit(r, range_sd_m=0.03, n_trials=40000, rng=np.random.default_rng(2), uniform=True)
    assert abs(mcu["rel_sd_c"] / ss.predicted_rel_se_c(r, ss.uniform_halfwidth_to_sd(0.03)) - 1) < 0.03


def test_sound_speed_fit_unbiased_and_marczak_check():
    r = np.array([1.0, 3.0, 6.0, 10.0])
    T = 26.0
    c = ss.marczak_c(T)
    exact = ss.fit_tof(r, r / c + 2e-3)
    assert abs(exact.c - c) < 1e-6 and abs(exact.tau0 - 2e-3) < 1e-12
    mc = ss.monte_carlo_tof_fit(r, c=c, tau0=2e-3, range_sd_m=0.03, n_trials=40000, rng=np.random.default_rng(3))
    # errors-in-variables attenuation bias is ~(n-1) sigma^2 / Sxx = 0.006 %: negligible
    assert abs(mc["mean_c"] / c - 1) < 2e-4
    assert abs(mc["mean_tau0"] - 2e-3) < 2e-6
    rng = np.random.default_rng(4)
    agree = 0
    for _ in range(200):
        f = ss.fit_tof(r + rng.normal(0, 0.03, 4), r / c + 2e-3, range_sd_m=0.03)
        agree += ss.compare_marczak(f, T, use_known_se=True).agree
    assert agree / 200 > 0.9  # ~95 % coverage at 2 SE
    off = ss.fit_tof(r, r / (1.03 * c) + 2e-3, range_sd_m=0.03)
    assert not ss.compare_marczak(off, T, use_known_se=True).agree
