"""Equivalence of the optimized uwsim hot paths with straightforward scipy references.

The simulator's FIR convolution (overlap-save), IIR filtering (cached designs, subnormal
dither), paired time scaling and shared transmit/h_chain work must reproduce the plain
implementations to float round-off; see uwse/uwsim/chain.py and simulate.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, fftconvolve, sosfilt, sosfiltfilt
from scipy.signal import resample as fft_resample

from uwse.uwsim import chain as ch
from uwse.uwsim.noise import band_snr_db, bubble_noise
from uwse.uwsim.simulate import _conv32, time_scale, time_scale_pair

FS = 48000


def _close(y, ref, tol=1e-12):
    ref = np.asarray(ref)
    assert y.shape == ref.shape
    peak = max(float(np.max(np.abs(ref))), 1e-300)
    assert float(np.max(np.abs(y - ref))) <= tol * peak


def _sparse(rng, n=4 * FS):
    """Bubble-like noise: exact-zero gaps of up to ~1 s (the subnormal-stall case)."""
    return bubble_noise(FS, n / FS, 2.0, (0.5e-3, 3e-3), rng)


# --------------------------------------------------------------------------- FIR


@pytest.mark.parametrize("n,m", [(192000, 513), (192000, 257), (60000, 400), (8192, 513), (100, 5)])
@pytest.mark.parametrize("rows", [None, 3])
def test_fir_full_matches_fftconvolve(rng, n, m, rows):
    x = rng.standard_normal((rows, n) if rows else n)
    h = rng.standard_normal(m)
    ref = fftconvolve(x, h[None] if rows else h, mode="full", axes=-1)
    _close(ch.fir_full(x, h), ref)


@pytest.mark.parametrize("centred", [False, True])
def test_apply_fir_matches_direct_convolution(rng, centred):
    x = rng.standard_normal(50000)
    h = ch.eq_fir([125, 1000, 8000], [2.0, -3.0, 1.0], FS, 257)
    full = np.convolve(x, h)
    off = (h.size - 1) // 2 if centred else 0
    _close(ch.apply_fir(x, h, centred=centred), full[off: off + x.size])
    # rows are filtered independently
    x2 = np.stack([x, 2 * x[::-1]])
    y2 = ch.apply_fir(x2, h, centred=centred)
    _close(y2[0], full[off: off + x.size])
    _close(y2[1], np.convolve(2 * x[::-1], h)[off: off + x.size])


# --------------------------------------------------------------------------- IIR


def test_butter_cache_matches_design():
    for args in ((8, 8000.0, "low"), (2, 100.0, "high"), (4, (500.0, 8000.0), "band")):
        order, wn, bt = args
        ref = butter(order, list(wn) if isinstance(wn, tuple) else wn, btype=bt, fs=FS, output="sos")
        np.testing.assert_array_equal(ch.butter_sos(order, wn, bt, FS), ref)
        key = (order, tuple(map(float, wn)) if isinstance(wn, tuple) else float(wn), bt, float(FS))
        assert ch._butter(*key)[0] is ch._butter(*key)[0]  # cached design, returned as a copy


def test_sosfilt_ftz_bit_identical_without_zeros(rng):
    sos = butter(8, 8000, fs=FS, output="sos")
    x = rng.standard_normal((2, 30000))
    np.testing.assert_array_equal(ch.sosfilt_ftz(sos, x), sosfilt(sos, x, axis=-1))
    zi = rng.standard_normal((sos.shape[0], 2, 2))
    y, zf = ch.sosfilt_ftz(sos, x, zi)
    y_ref, zf_ref = sosfilt(sos, x, axis=-1, zi=zi)
    np.testing.assert_array_equal(y, y_ref)
    np.testing.assert_array_equal(zf, zf_ref)


@pytest.mark.parametrize("design", [(4, [500, 8000], "band"), (2, 100, "high"), (8, 8000, "low")])
def test_sosfilt_ftz_sparse_input_no_subnormals(rng, design):
    order, wn, bt = design
    sos = butter(order, wn, btype=bt, fs=FS, output="sos")
    x = _sparse(rng)
    assert np.mean(x == 0) > 0.5
    y = ch.sosfilt_ftz(sos, x)
    _close(y, sosfilt(sos, x), tol=1e-100)
    tiny = np.finfo(np.float64).tiny
    assert not np.any((y != 0) & (np.abs(y) < tiny))  # nothing decays into the subnormal range
    yy = ch.sosfiltfilt_ftz(sos, x)
    _close(yy, sosfiltfilt(sos, x), tol=1e-100)
    assert not np.any((yy != 0) & (np.abs(yy) < tiny))


def test_sosfiltfilt_ftz_matches_scipy(rng):
    sos = butter(4, [500, 8000], btype="band", fs=FS, output="sos")
    x = rng.standard_normal((3, 20000))
    np.testing.assert_array_equal(ch.sosfiltfilt_ftz(sos, x), sosfiltfilt(sos, x, axis=-1))
    np.testing.assert_array_equal(ch.sosfiltfilt_ftz(sos, x[1]), sosfiltfilt(sos, x[1]))
    with pytest.raises(ValueError):
        ch.sosfiltfilt_ftz(sos, x[0, :20])


def test_bandlimit_and_band_snr_match_reference(rng):
    x = rng.standard_normal((2, 40000))
    sos = butter(4, [500, 8000], btype="band", fs=FS, output="sos")
    np.testing.assert_array_equal(ch.bandlimit(x, FS, 500, 8000), sosfiltfilt(sos, x, axis=-1))
    np.testing.assert_array_equal(ch.bandlimit(x[0], FS, 500, None),
                                  sosfiltfilt(butter(4, 500, "high", fs=FS, output="sos"), x[0]))
    s, nz = x[0], _sparse(rng, 40000)
    m = np.abs(s) > 0.3
    ref = 10 * np.log10((np.mean(sosfiltfilt(sos, s)[m] ** 2) + 1e-30)
                        / (np.mean(sosfiltfilt(sos, nz)[m] ** 2) + 1e-30))
    assert abs(band_snr_db(s, nz, FS, (500.0, 8000.0), m) - ref) < 1e-10


def test_adc_matches_reference_formula():
    x = np.random.default_rng(3).standard_normal(5000) * 0.3
    y, clipped = ch.adc(x, np.random.default_rng(7), -90.0, 16)
    r = x + 10 ** (-90 / 20) * np.random.default_rng(7).standard_normal(x.shape)
    q = 2.0 ** 15
    np.testing.assert_array_equal(y, np.clip(np.round(r * q), -q, q - 1) / q)
    assert clipped == bool(np.any(np.abs(r) >= 1.0))


# --------------------------------------------------------------------------- chain / simulate helpers


@pytest.mark.parametrize("eps", [7.3e-4, -4.1e-4, 0.0])
@pytest.mark.parametrize("n", [192000, 33331])
def test_time_scale_pair_matches_two_real_calls(rng, eps, n):
    a, b = rng.standard_normal(n), 0.1 * rng.standard_normal(n)
    ya, yb = time_scale_pair(a, b, eps)
    _close(ya, time_scale(a, eps))
    _close(yb, time_scale(b, eps))
    if eps:  # the real path is scipy's FFT resample cropped / zero-padded to n
        num = int(round(n / (1 + eps)))
        ref = fft_resample(a, num)
        ref = ref[:n] if ref.size >= n else np.pad(ref, (0, n - ref.size))
        _close(ya, ref)


@pytest.mark.parametrize("eps", [7.3e-4, -9.9e-4])
def test_time_scale_pair_single_precision(rng, eps):
    """complex64 path: each output within 1e-6 of its own peak, even at 1e-4 relative scale."""
    n = 192000
    a, b = rng.standard_normal(n), 1e-4 * rng.standard_normal(n)
    ya, yb = time_scale_pair(a, b, eps, single=True)
    assert ya.dtype == yb.dtype == np.float64
    _close(ya, time_scale(a, eps), tol=1e-6)
    _close(yb, time_scale(b, eps), tol=1e-6)
    z0, z1 = time_scale_pair(a, np.zeros(n), eps, single=True)
    _close(z0, ya, tol=1e-6)
    assert np.max(np.abs(z1)) <= 1e-6 * np.max(np.abs(a))  # cross-talk at float32 round-off


def test_conv32_close_to_float64(rng):
    x, h = rng.standard_normal(48000), rng.standard_normal(4800) * np.exp(-np.arange(4800) / 800)
    y = _conv32(x, h)
    assert y.dtype == np.float64
    _close(y, fftconvolve(x, h), tol=1e-6)


def test_fir_edge_cases(rng):
    x = rng.standard_normal(1000)
    assert ch.apply_fir(x, np.array([])).shape == fftconvolve(x, np.array([])).shape
    sos = ch.butter_sos(4, 1000.0, "low", FS)
    sos[:] = 0.0  # the returned array is a copy: the cached design must stay intact
    np.testing.assert_array_equal(ch.butter_sos(4, 1000.0, "low", FS), butter(4, 1000.0, fs=FS, output="sos"))


def _reference_transmit_and_h_chain(chn, s, drive_db, theta, proj_eq, rf):
    """The pre-optimization formulas: transmit() and h_chain_apply() before receive()."""
    x = sosfilt(butter(chn.drive_lpf_order, chn.drive_lpf_hz, fs=chn.fs, output="sos"), s)
    pk = np.max(np.abs(x))
    xs = x / pk * 10.0 ** (drive_db / 20.0)
    eq = ch.eq_fir(proj_eq[0], proj_eq[1], chn.fs)
    y = np.zeros_like(xs)
    for k, h in enumerate(chn.projector.kernels(rf), start=1):
        if h is not None and np.any(h):
            y += fftconvolve(xs ** k, h)[: xs.size] if h.size > 1 else xs ** k * h[0]
    y = fftconvolve(y, eq)[256: 256 + s.size]
    d = chn.directivity.fir(theta, chn.fs)
    tx = fftconvolve(y, d)[(d.size - 1) // 2: (d.size - 1) // 2 + s.size]
    h1 = ch.warp_fir(chn.projector.h1, rf)
    lin = fftconvolve(x, h1)[: x.size] if h1.size > 1 else x * h1[0]
    return tx, fftconvolve(lin, eq)[256: 256 + s.size]


@pytest.mark.parametrize("nonlinear", [False, True])
def test_transmit_and_h_chain_shared_work(rng, nonlinear):
    s = ch.bandlimit(rng.standard_normal(3 * FS), FS, 100, 7000)
    proj = ch.Projector()
    if nonlinear:
        nn = np.arange(300)
        proj = ch.Projector(h1=np.exp(-nn / 50.0) * np.cos(nn / 3.0), h2=0.05 * np.exp(-nn[:100] / 20.0),
                            h3=0.02 * rng.standard_normal(64))
    chn = ch.TransducerChain(projector=proj)
    proj_eq = ch.random_third_octave_gains(rng, 3.0)
    tx, y = chn._transmit_and_h_chain_tx(s, -4.0, 37.0, proj_eq, 1.13)
    tx_ref, y_ref = _reference_transmit_and_h_chain(chn, s, -4.0, 37.0, proj_eq, 1.13)
    _close(tx, tx_ref)
    _close(y, y_ref)
    # and against the public methods
    _close(tx, chn.transmit(s, -4.0, 37.0, proj_eq, 1.13))
    hyd_eq = ch.tilt_gains(1.5)
    _close(chn.receive(y, hyd_eq), chn.h_chain_apply(s, proj_eq, hyd_eq, 1.13))


def test_receive_rows_match_single_calls(rng):
    chn = ch.TransducerChain(hydrophone=ch.Hydrophone(fir=np.array([1.0, -0.3, 0.1])))
    hyd_eq = ch.tilt_gains(-1.2)
    x = np.stack([rng.standard_normal(30000), _sparse(rng, 30000), rng.standard_normal(30000)])
    y = chn.receive(x, hyd_eq)
    for r in range(3):
        _close(y[r], chn.receive(x[r], hyd_eq))
