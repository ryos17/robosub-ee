"""Analytic tests for uwse.uwsim.{reflection, geometry, ism} (RESEARCH_PLAN 4.1--4.2)."""

import itertools

import numpy as np
import pytest

from uwse.uwsim.geometry import Pool, avery_competition_pool, cutoff_frequency, maas_dive_well, mirror_point
from uwse.uwsim.ism import (
    exact_floor_images,
    image_sources,
    ism_rir,
    mean_depth_error,
    render_freq,
    render_sinc,
    shoebox_images,
)
from uwse.uwsim.reflection import (
    Boundary,
    CONCRETE_UNCALIBRATED,
    anechoic,
    fluid_fluid_R,
    halfspace_R,
    pressure_release_R,
    surface_R,
)

C, FS, RHO_W = 1500.0, 48000, 997.0


def _dtft(x, f, fs=FS):
    n = np.arange(len(x))
    return np.exp(-2j * np.pi * np.outer(f, n) / fs) @ x


# ---------------------------------------------------------------- reflection
def test_halfspace_normal_incidence():
    b = Boundary(cp=4200.0, cs=2400.0, rho=2350.0, eta=0.0)
    z1, z2 = RHO_W * C, b.rho * b.cp
    R0 = halfspace_R(0.0, C, RHO_W, b.cp, b.cs, b.rho)
    assert np.isclose(R0, (z2 - z1) / (z2 + z1), atol=1e-12)
    assert np.isclose(b.normal_incidence_R(C, RHO_W), R0.real)


def test_halfspace_beyond_critical_lossless_and_lossy():
    b = Boundary(cp=4000.0, cs=2300.0, rho=2300.0, eta=0.0)
    tp, ts = b.critical_angles(C)
    th = np.linspace(max(tp, ts) + 0.02, np.pi / 2 - 0.01, 50)
    R = b.R(th, C, RHO_W)
    assert np.allclose(np.abs(R), 1.0, atol=1e-12)
    assert np.all(np.abs(np.angle(R)) > 1e-3)  # genuine phase shift, not just a sign
    lossy = Boundary(cp=4000.0, cs=2300.0, rho=2300.0, eta=0.05).R(th, C, RHO_W)
    assert np.all(np.abs(lossy) < 1.0)
    # grazing incidence -> -1
    assert np.isclose(b.R(np.pi / 2, C, RHO_W), -1.0, atol=1e-9)


def test_fluid_fluid_special_case():
    th = np.linspace(0, 1.5, 30)
    a = halfspace_R(th, C, RHO_W, 1700.0, 0.0, 1900.0)
    b = fluid_fluid_R(th, C, RHO_W, 1700.0, 1900.0)
    assert np.allclose(a, b)
    tc = np.arcsin(C / 1700.0)
    beyond = th > tc + 1e-3
    assert np.allclose(np.abs(b[beyond]), 1.0)
    assert np.allclose(anechoic(C, RHO_W).R(th, C, RHO_W), 0.0)


def test_surface_constants():
    assert surface_R().item() == -1.0
    assert np.all(surface_R(np.zeros(4), -0.95) == -0.95)
    assert -1.0 < pressure_release_R() < -0.999


# ---------------------------------------------------------------- geometry
def test_pool_geometry():
    p = avery_competition_pool()
    assert np.isclose(p.depth_at(0), 3.35) and np.isclose(p.depth_at(37), 4.27)
    assert np.isclose(p.mean_depth_along([10, 5, 1], [20, 5, 2]), p.depth_at(15))
    assert p.contains(np.array([1.0, 1.0, 3.5])) == False  # below the floor at x = 1  # noqa: E712
    with pytest.raises(ValueError):
        p.check_position([1.0, 1.0, -0.1])
    m = maas_dive_well()
    assert m.is_flat and np.isclose(m.H0, 5.4864)
    assert np.isclose(cutoff_frequency(1500, 2.0), 187.5)
    lo, hi = p.cutoff_range(1500)
    assert lo < hi
    # mirror across the tilted floor: equidistant, on the normal
    n, d = p.floor_plane()
    q = np.array([7.0, 3.0, 2.0])
    qm = mirror_point(q, n, d)
    assert np.isclose(q @ n - d, -(qm @ n - d))


# ---------------------------------------------------------------- ISM basics
def test_direct_path_exact():
    pool = Pool(20.0, 10.0, 4.0)
    src, rx = np.array([5.0, 5.0, 1.0]), np.array([8.0, 6.0, 1.5])
    res = ism_rir(src, rx, pool, C, FS, t_max=0.01, walls=0.0, floor=0.0, surface_R=0.0, rolloff=0.0)
    im = res.images
    assert len(im) == 1
    r = np.linalg.norm(src - rx)
    assert np.isclose(im.delay[0], r / C) and np.isclose(im.amp[0], 1.0 / r)
    # integer-sample delay renders as a single sample of value 1/r
    src2 = np.array([5.0, 5.0, 1.0])
    rx2 = src2 + np.array([1.5, 0.0, 0.0])  # 1 ms = 48 samples
    res2 = ism_rir(src2, rx2, pool, C, FS, t_max=0.004, walls=0.0, floor=0.0, surface_R=0.0, rolloff=0.0)
    k = int(np.argmax(np.abs(res2.rir)))
    assert k == 48 and np.isclose(res2.rir[k], 1 / 1.5, rtol=1e-6)
    assert np.max(np.abs(np.delete(res2.rir, k))) < 1e-6


def test_single_surface_image_sign_and_delay():
    pool = Pool(30.0, 20.0, 5.0)
    src, rx = np.array([10.0, 10.0, 1.0]), np.array([11.0, 10.0, 1.0])
    im = image_sources(src, rx, pool, C, 0.01, surface_R=-1.0, walls=0.0, floor=0.0)[0]
    assert len(im) == 2
    surf = im.subset(im.counts[:, 4] == 1)
    r2 = np.sqrt(1.0 + 2.0**2)
    assert np.isclose(surf.delay[0], r2 / C) and np.isclose(surf.amp[0], -1.0 / r2)
    assert np.allclose(surf.pos[0], [10.0, 10.0, -1.0])
    assert np.isclose(surf.theta[0, 4], np.arctan2(1.0, 2.0))


def test_lloyds_mirror_appendix_a():
    """Two-point Lloyd's mirror (surface only, R = -1, both ends 1 m deep, r = 1 m):
    plan Appendix A row: -1.3, +2.9, -1.0, +2.4, +1.6 dB at 0.2/0.5/1/2/4 kHz;
    nulls of |1 - (r1/r2) exp(-j 2 pi f dtau)| at f = k / dtau."""
    pool = Pool(40.0, 30.0, 10.0)
    src, rx = np.array([20.0, 15.0, 1.0]), np.array([21.0, 15.0, 1.0])
    res = ism_rir(src, rx, pool, C, FS, t_max=0.004, walls=0.0, floor=0.0, rolloff=0.0, n_fft=1 << 16)
    f = np.array([200.0, 500.0, 1000.0, 2000.0, 4000.0])
    level = 20 * np.log10(np.abs(_dtft(res.rir, f)) * 1.0)  # direct path is 1/r = 1
    assert np.allclose(level, [-1.3, 2.9, -1.0, 2.4, 1.6], atol=0.15)
    r1, r2 = 1.0, np.sqrt(5.0)
    dtau = (r2 - r1) / C
    nulls = np.arange(1, 6) / dtau
    peaks = (np.arange(1, 6) - 0.5) / dtau
    Hn, Hp = np.abs(_dtft(res.rir, nulls)), np.abs(_dtft(res.rir, peaks))
    assert np.allclose(Hn, 1 / r1 - 1 / r2, rtol=0.02)
    assert np.allclose(Hp, 1 / r1 + 1 / r2, rtol=0.02)


# ---------------------------------------------------------------- tilted floor
def test_exact_floor_flat_matches_shoebox():
    pool = Pool(12.0, 8.0, 3.0, 3.0)
    src, rx = np.array([3.0, 2.0, 0.7]), np.array([7.5, 5.0, 2.1])
    pe, ce, _ = exact_floor_images(src, rx, pool, 1)
    assert len(pe) == 1 and np.allclose(pe[0], [3.0, 2.0, 2 * 3.0 - 0.7])
    pe2, ce2, _ = exact_floor_images(src, rx, pool, 2)
    ps, cs = shoebox_images(src, rx, (12.0, 8.0, 3.0), 100.0, max_order=2)
    sel = cs[:, 5] > 0
    key = lambda p, c: sorted(tuple(np.round(np.r_[pp, cc], 9)) for pp, cc in zip(p, c))  # noqa: E731
    assert key(pe2, ce2) == key(ps[sel], cs[sel])
    a = ism_rir(src, rx, pool, C, FS, 0.03, walls=CONCRETE_UNCALIBRATED, exact_floor_order=2)
    b = ism_rir(src, rx, pool, C, FS, 0.03, walls=CONCRETE_UNCALIBRATED, exact_floor_order=0)
    assert np.allclose(a.rir, b.rir, atol=1e-12)


def test_tilted_floor_images_and_mean_depth_error():
    pool = avery_competition_pool()
    src, rx = np.array([5.0, 8.0, 1.0]), np.array([9.0, 9.0, 1.5])
    pe, ce, te = exact_floor_images(src, rx, pool, 1)
    n, d = pool.floor_plane()
    assert len(pe) == 1
    assert np.isclose(src @ n - d, -(pe[0] @ n - d))  # true mirror across the sloped plane
    # specular point lies on the floor
    err = mean_depth_error(src, rx, pool)
    assert np.all(np.isfinite(err.delay_error))
    assert 0 < err.max_abs_delay_error < 2e-4  # nonzero for a slope, small (< 0.2 ms) here
    flat = mean_depth_error(src, rx, Pool(37.0, 20.0, 3.8, 3.8))
    assert flat.max_abs_delay_error < 1e-12


# ---------------------------------------------------------------- rendering
def _brute_force(src, rx, dims, R_wall, R_floor, R_s, t_max, n_out):
    """Independent loop implementation: explicit image lattice + ideal sinc."""
    L, W, H = dims
    D = C * t_max
    rir = np.zeros(n_out)
    n = np.arange(n_out)
    M = int(D // (2 * min(dims))) + 2
    count = 0
    for px, py, pz in itertools.product((0, 1), repeat=3):
        for mx, my, mz in itertools.product(range(-M, M + 1), repeat=3):
            img = np.array(
                [(1 - 2 * px) * src[0] + 2 * mx * L, (1 - 2 * py) * src[1] + 2 * my * W, (1 - 2 * pz) * src[2] + 2 * mz * H]
            )
            r = np.linalg.norm(img - rx)
            if r > D:
                continue
            nw = abs(mx - px) + abs(mx) + abs(my - py) + abs(my)
            amp = R_wall**nw * R_s ** abs(mz - pz) * R_floor ** abs(mz) / r
            rir += amp * np.sinc(n - r / C * FS)
            count += 1
    return rir, count


@pytest.mark.parametrize("R_wall,R_floor", [(0.8, 0.6), (-0.7, 0.9)])
def test_freq_synthesis_matches_brute_force(R_wall, R_floor):
    dims = (6.0, 4.0, 2.0)
    pool = Pool(*dims)
    src, rx = np.array([1.3, 1.1, 0.6]), np.array([4.2, 2.9, 1.4])
    t_max = 0.012
    res = ism_rir(src, rx, pool, C, FS, t_max, -1.0, R_wall, R_floor, rolloff=0.0, n_fft=1 << 15)
    ref, count = _brute_force(src, rx, dims, R_wall, R_floor, -1.0, t_max, len(res.rir))
    assert len(res.images) == count
    assert np.max(np.abs(res.rir - ref)) < 2e-3 * np.max(np.abs(ref))


def test_sinc_method_matches_freq_in_band():
    pool = Pool(6.0, 4.0, 2.0)
    src, rx = np.array([1.3, 1.1, 0.6]), np.array([4.2, 2.9, 1.4])
    kw = dict(walls=CONCRETE_UNCALIBRATED, floor=Boundary(cp=5000, cs=2800, rho=2400, eta=0.0), rolloff=0.0)
    a = ism_rir(src, rx, pool, C, FS, 0.012, method="freq", **kw)
    b = ism_rir(src, rx, pool, C, FS, 0.012, method="sinc", **kw)
    assert np.iscomplexobj(a.images.R) and np.max(np.abs(a.images.R.imag)) > 0.1
    f = np.linspace(2000, 16000, 50)
    Ha, Hb = _dtft(a.rir, f), _dtft(b.rir, f)
    assert np.max(np.abs(Ha - Hb)) / np.max(np.abs(Ha)) < 0.1


def test_complex_amplitude_rendering():
    """A single image with A = exp(j phi) must have the constant phase phi (after
    removing the delay) at every in-band frequency: checks the Hilbert part."""
    phi, tau, n_out = 0.9, 100.37 / FS, 400
    h = render_freq(np.array([tau]), np.array([np.exp(1j * phi)]), FS, n_out, n_fft=1 << 16)
    f = np.linspace(500, 20000, 40)
    ph = np.angle(_dtft(h, f) * np.exp(2j * np.pi * f * tau))
    assert np.allclose(ph, phi, atol=0.05)
    hs = render_sinc(np.array([tau]), np.array([np.exp(1j * phi)]), FS, n_out)
    f2 = np.linspace(3000, 18000, 20)
    ph2 = np.angle(_dtft(hs, f2) * np.exp(2j * np.pi * f2 * tau))
    assert np.allclose(ph2, phi, atol=0.1)


# ---------------------------------------------------------------- sanity
def test_image_count_and_energy_sanity():
    dims = (10.0, 8.0, 3.0)
    pool = Pool(*dims)
    src, rx = np.array([3.0, 3.0, 1.0]), np.array([6.0, 5.0, 2.0])
    t_max = 0.04
    D = C * t_max
    im = image_sources(src, rx, pool, C, t_max, -1.0, 1.0, 1.0)[0]
    expected = 4.0 / 3.0 * np.pi * D**3 / np.prod(dims)
    assert abs(len(im) / expected - 1) < 0.05
    # energy: sum 1/r^2 ~ 4 pi D / V for |R| = 1
    e = np.sum(np.abs(im.amp[1:]) ** 2)
    assert abs(e / (4 * np.pi * D / np.prod(dims)) - 1) < 0.15
    assert np.all(im.delay <= t_max + 1e-12) and np.all(np.diff(im.delay) >= 0)
    # max_order cap
    im2 = image_sources(src, rx, pool, C, t_max, -1.0, 1.0, 1.0, max_order=2)[0]
    assert len(im2) == 1 + 6 + 18 and im2.order.max() == 2


def test_pyroomacoustics_image_positions():
    pra = pytest.importorskip("pyroomacoustics")
    dims = [7.0, 5.0, 2.5]
    src, rx = [2.0, 1.5, 0.8], [5.1, 3.3, 1.7]
    room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(0.1), max_order=3)
    room.add_source(src)
    room.add_microphone(rx)
    room.image_source_model()
    ref = np.sort(np.linalg.norm(room.sources[0].images.T - np.array(rx), axis=1))
    im = image_sources(np.array(src), np.array(rx), Pool(*dims), C, 1.0, 1.0, 1.0, 1.0, max_order=3)[0]
    assert len(im) == len(ref)
    assert np.allclose(np.sort(im.dist), ref)


def test_invalid_positions():
    pool = Pool(10.0, 5.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        ism_rir([1.0, 1.0, 2.5], [5.0, 2.0, 1.0], pool)  # below the floor at x = 1 (H = 2.1)
    with pytest.raises(ValueError):
        ism_rir([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], pool)
