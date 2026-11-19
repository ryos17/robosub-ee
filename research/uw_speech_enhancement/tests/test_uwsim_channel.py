"""Tests for uwse.uwsim.physics / tail / lowband (RESEARCH_PLAN 4.2 rows 1, 2, 7-9, 14, 16)."""

import numpy as np
import pytest

from uwse.uwsim.lowband import crossover_frequency, lr4_split, modal_rir, shoebox_modes, splice_lowband
from uwse.uwsim.physics import (
    absorption_db,
    francois_garrison_alpha,
    minnaert_frequency,
    octave_bands,
    salinity_correction,
    sound_speed_marczak,
    waveguide_cutoff,
)
from uwse.uwsim.tail import (
    apply_time_scale,
    late_tail,
    octave_filterbank,
    perturb_ir_bands,
    splice_tail,
)

FS = 16000


# ----------------------------------------------------------------- local helpers


def schroeder_db(h=None, energy=None):
    """Schroeder EDC in dB from an IR, or from an (ensemble-averaged) energy h^2."""
    e = h**2 if energy is None else energy
    e = np.cumsum(e[::-1])[::-1]
    return 10 * np.log10(e / e[0] + 1e-300)


def fit_t60(h, fs, hi_db=-5.0, lo_db=-25.0, energy=None):
    """T60 from a line fit to the Schroeder EDC between hi_db and lo_db."""
    edc = schroeder_db(h, energy)
    sel = (edc <= hi_db) & (edc >= lo_db)
    t = np.arange(len(edc))[sel] / fs
    slope = np.polyfit(t, edc[sel], 1)[0]
    return -60.0 / slope


# ----------------------------------------------------------------- physics


def test_marczak_values():
    assert sound_speed_marczak(20.0) == pytest.approx(1482.4, abs=0.2)
    assert sound_speed_marczak(25.0) == pytest.approx(1496.7, abs=0.2)
    c = sound_speed_marczak(np.array([25.0, 27.0]))
    assert np.all((c > 1496) & (c < 1503))


def test_salinity_correction_range():
    per_ppt = salinity_correction(1.0, 25.0)
    assert 1.0 < per_ppt < 1.35
    assert salinity_correction(0.0, 25.0) == 0.0


def test_francois_garrison():
    # fresh water, 8 kHz, 50 m: negligible (row 2: absorption off)
    assert absorption_db(8000.0, 50.0, T_C=25.0, S_ppt=0.0) < 1e-3
    # seawater sanity: about 1 dB/km at 10 kHz, cold water
    a = francois_garrison_alpha(10e3, T_C=4.0, S_ppt=35.0, depth_m=0.0, pH=8.0)
    assert 0.7 < a < 1.5
    # monotone in f
    f = np.array([1e3, 4e3, 8e3])
    assert np.all(np.diff(francois_garrison_alpha(f, 25.0, 0.0)) > 0)


def test_waveguide_cutoff():
    c = sound_speed_marczak(26.0)
    fc = waveguide_cutoff(c, np.array([3.35, 4.27]))
    assert 108 < fc[0] < 113 and 86 < fc[1] < 90


def test_minnaert():
    assert minnaert_frequency(1e-3) == pytest.approx(3290.0, rel=0.01)
    f = minnaert_frequency(np.array([0.5e-3, 3e-3]))
    assert f[0] == pytest.approx(6.6e3, rel=0.02) and f[1] == pytest.approx(1.1e3, rel=0.02)
    assert minnaert_frequency(1e-3, depth_m=10.0) > minnaert_frequency(1e-3)


def test_octave_bands():
    c, lo, hi = octave_bands()
    np.testing.assert_allclose(c, [250, 500, 1000, 2000, 4000, 8000])
    np.testing.assert_allclose(hi[:-1], lo[1:])


# ----------------------------------------------------------------- filterbank / tail


def test_filterbank_perfect_reconstruction(rng):
    x = rng.standard_normal(4000)
    bands = octave_filterbank(x, FS)
    assert bands.shape == (6, 4000)
    np.testing.assert_allclose(bands.sum(0), x, atol=1e-10)


def test_single_slope_t60_recovered(rng):
    # A single realization of the 250 Hz band scatters by ~5 % (few degrees of
    # freedom), so the EDC is taken from the ensemble-mean energy of 8 realizations.
    t60 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    n_real = 8
    e_bands = 0.0
    e_reband = 0.0
    for _ in range(n_real):
        bands = late_tail(FS, int(1.5 * FS), t60, rng, return_bands=True)
        e_bands = e_bands + bands**2
        # and after summing and re-analysing with the same bank
        e_reband = e_reband + octave_filterbank(bands.sum(0), FS) ** 2
    for b in range(6):
        assert fit_t60(None, FS, energy=e_bands[b]) == pytest.approx(t60[b], rel=0.05)
        assert fit_t60(None, FS, energy=e_reband[b]) == pytest.approx(t60[b], rel=0.05)


def test_two_slope_shape(rng):
    B, n_real, n = 6, 6, int(2.0 * FS)
    params = np.tile([0.2, 1.0, -15.0], (B, 1))
    e2 = sum(octave_filterbank(late_tail(FS, n, params, rng), FS)[3] ** 2 for _ in range(n_real))
    e1 = sum(octave_filterbank(late_tail(FS, n, np.full(B, 0.2), rng), FS)[3] ** 2 for _ in range(n_real))
    t_early = fit_t60(None, FS, -1.0, -8.0, energy=e2)
    t_late = fit_t60(None, FS, -25.0, -40.0, energy=e2)
    assert t_early < 0.45  # dominated by the fast slope
    assert t_late == pytest.approx(1.0, rel=0.10)  # knee, then the slow slope
    # single slope of the same T1 has no knee
    assert fit_t60(None, FS, -25.0, -40.0, energy=e1) == pytest.approx(0.2, rel=0.10)


def test_splice_continuity(rng):
    t_e, t60 = 0.04, np.full(6, 0.5)
    n_e = int(t_e * FS)
    length = int(1.0 * FS)
    # "early" = a dense exponential decay (stand-in for the ISM), truncated at t_e
    early = late_tail(FS, n_e, t60, rng)
    out = splice_tail(early, FS, t_e, t60, rng, length, match_window=0.03)
    assert out.shape == (length,)
    n_x = int(0.005 * FS)
    np.testing.assert_array_equal(out[: n_e - n_x], early[: n_e - n_x])
    # broadband power continuous across t_e (expected ratio from the decay itself)
    w = int(0.015 * FS)
    p_before = np.mean(out[n_e - n_x - w : n_e - n_x] ** 2)
    p_after = np.mean(out[n_e : n_e + w] ** 2)
    expected_db = -60.0 * (n_x + w) / FS / 0.5
    assert 10 * np.log10(p_after / p_before) == pytest.approx(expected_db, abs=2.0)
    # the spliced IR keeps the decay
    assert fit_t60(out, FS) == pytest.approx(0.5, rel=0.05)


def test_splice_two_slope_runs(rng):
    early = late_tail(FS, int(0.04 * FS), np.full(6, 0.3), rng)
    p = np.tile([0.3, 0.9, -10.0], (6, 1))
    out = splice_tail(early, FS, 0.04, p, rng, int(1.0 * FS))
    assert np.all(np.isfinite(out)) and np.abs(out[int(0.05 * FS) :]).max() > 0


# ----------------------------------------------------------------- time variation


def test_apply_time_scale(rng):
    eps = 8e-4
    n = np.arange(FS)
    f0 = 1000.0
    x = np.sin(2 * np.pi * f0 * n / FS)
    y = apply_time_scale(x, eps)
    assert len(y) == int(np.floor((FS - 1) / (1 + eps))) + 1
    m = np.arange(len(y))
    ref = np.sin(2 * np.pi * f0 * (1 + eps) * m / FS)
    mid = slice(100, len(y) - 100)
    assert np.max(np.abs(y[mid] - ref[mid])) < 2e-3
    np.testing.assert_array_equal(apply_time_scale(x, 0.0), x)
    with pytest.raises(ValueError):
        apply_time_scale(x, 2e-3)


def test_perturb_ir_bands(rng):
    h = late_tail(FS, FS // 4, np.full(6, 0.3), rng)
    np.testing.assert_allclose(perturb_ir_bands(h, FS, 0.0, rng), h, atol=1e-10)
    hp = perturb_ir_bands(h, FS, 2.0, rng)
    assert not np.allclose(hp, h)
    e0 = np.sum(octave_filterbank(h, FS) ** 2, axis=-1)
    e1 = np.sum(octave_filterbank(hp, FS) ** 2, axis=-1)
    assert np.all(np.abs(10 * np.log10(e1 / e0)) < 12)


# ----------------------------------------------------------------- low band


@pytest.mark.parametrize("zero_phase", [True, False])
def test_lr4_sum_flat(zero_phase):
    x = np.zeros(8192)
    x[4096] = 1.0
    lo, hi = lr4_split(x, FS, 500.0, zero_phase=zero_phase)
    mag = 20 * np.log10(np.abs(np.fft.rfft(lo + hi)))
    assert np.max(np.abs(mag)) < 0.1
    f = np.fft.rfftfreq(len(x), 1 / FS)
    k = np.argmin(np.abs(f - 500.0))
    assert 20 * np.log10(np.abs(np.fft.rfft(lo))[k]) == pytest.approx(-6.02, abs=0.3)


def test_splice_lowband_identity(rng):
    h = late_tail(FS, FS // 2, np.full(6, 0.3), rng)
    out = splice_lowband(h, h, FS, 500.0)
    edge = int(0.02 * FS)
    np.testing.assert_allclose(out[edge:-edge], h[edge:-edge], atol=1e-6 * np.abs(h).max())
    assert crossover_frequency(None) == 500.0
    assert crossover_frequency(620.0) == 620.0 and crossover_frequency(110.0) == 500.0


def test_modal_lowest_mode_is_cutoff():
    c, H = 1500.0, 2.0
    f, idx = shoebox_modes(500.0, 300.0, H, c, f_max=200.0)
    assert f[0] == pytest.approx(waveguide_cutoff(c, H))
    assert tuple(idx[0]) == (0, 0, 0)
    assert f[1] == pytest.approx(c / (4 * H), rel=1e-3)  # large L, W: dense just above
    assert np.all(f >= c / (4 * H) - 1e-9)
    # next depth mode is 3 c / (4 H)
    f2, idx2 = shoebox_modes(5.0, 4.0, H, c, f_max=600.0)
    assert np.any((idx2[:, 0] == 0) & (idx2[:, 1] == 0) & (idx2[:, 2] == 1))
    assert f2[(idx2[:, 0] == 0) & (idx2[:, 1] == 0) & (idx2[:, 2] == 1)][0] == pytest.approx(3 * c / (4 * H))


def test_modal_rir_spectrum_peaks_at_lowest_mode():
    fs, c = 4000, 1500.0
    L, W, H = 3.0, 2.5, 2.0  # lowest mode c/(4H) = 187.5 Hz, next lateral ones far above
    h = modal_rir((1.0, 1.0, 1.5), (2.0, 1.5, 1.8), L, W, H, c, fs, fs, f_max=400.0, damping=(2.0, 2.0))
    spec = np.abs(np.fft.rfft(h))
    f = np.fft.rfftfreq(len(h), 1 / fs)
    sel = f < 400
    fpk = f[sel][np.argmax(spec[sel])]
    assert fpk == pytest.approx(c / (4 * H), abs=2.0) or fpk > c / (4 * H)
    assert spec[f < 0.5 * c / (4 * H)].max() < 0.1 * spec.max()
    with pytest.raises(ValueError):
        modal_rir((4.0, 1.0, 1.0), (1.0, 1.0, 1.0), L, W, H, c, fs, 100)


@pytest.mark.slow
def test_modal_matches_image_sum():
    """Modal sum (Green's-function 1/(4 pi r) normalisation) == exact image sum, low-passed."""
    fs, c = 8000, 1500.0
    L, W, H = 6.0, 5.0, 2.0
    src, rx = (1.3, 2.1, 0.7), (4.2, 3.3, 1.2)
    T = 0.06
    N = int(T * fs)
    h_m = modal_rir(src, rx, L, W, H, c, fs, N, f_max=1500.0, damping=(1e6, 1e6), spreading="4pi")
    p = np.arange(-15, 16)
    xs = np.concatenate([src[0] + 2 * p * L, -src[0] + 2 * p * L])
    ys = np.concatenate([src[1] + 2 * p * W, -src[1] + 2 * p * W])
    q = np.arange(-20, 21)[:, None] * 4 * H
    zz = np.concatenate([src[2] + q, -src[2] + q, 2 * H - src[2] + q, 2 * H + src[2] + q]).ravel()
    sz = np.concatenate([np.ones_like(q), -np.ones_like(q), np.ones_like(q), -np.ones_like(q)]).ravel()
    X, Y, Z = np.meshgrid(xs, ys, zz, indexing="ij")
    S = np.broadcast_to(sz, X.shape)
    d = np.sqrt((X - rx[0]) ** 2 + (Y - rx[1]) ** 2 + (Z - rx[2]) ** 2)
    keep = d / c < T + 0.01
    d, S = d[keep], S[keep]
    t = np.arange(N) / fs
    h_i = np.zeros(N)
    for i in range(0, len(d), 2048):
        h_i += (S[i : i + 2048] / (4 * np.pi * d[i : i + 2048])) @ np.sinc(fs * (t[None] - d[i : i + 2048, None] / c))
    lm, _ = lr4_split(h_m, fs, 300.0)
    li, _ = lr4_split(h_i, fs, 300.0)
    sl = slice(int(0.01 * fs), int(0.045 * fs))
    assert np.linalg.norm(lm[sl] - li[sl]) / np.linalg.norm(li[sl]) < 0.01
