"""Adversarial review tests for uwse.uwsim.{reflection, geometry, ism} (RESEARCH_PLAN 4.1--4.2).

These check the module against independent derivations rather than its own formulas:
the elastic boundary-condition system for R, the law of reflection for the exact
tilted-floor images, source--receiver reciprocity, and the per-image angle.
"""

import numpy as np
import pytest

from uwse.uwsim.geometry import Pool, avery_competition_pool
from uwse.uwsim.ism import exact_floor_images, image_sources, ism_rir, render_freq
from uwse.uwsim.reflection import Boundary, as_reflection, halfspace_R

C, FS, RHO_W = 1500.0, 48000, 997.0


def _q(c, p):
    """Vertical slowness with Im(q) < 0 (decaying into the lower medium, exp(+j w t))."""
    q = np.sqrt(1.0 / complex(c) ** 2 - p**2 + 0j)
    if q.imag > 0 or (q.imag == 0 and q.real < 0):
        q = -q
    return q


def _R_bc(theta, cp, cs, rho, eta):
    """R from the 3x3 fluid--solid boundary-condition system (independent of the closed form).

    exp(+j w t), w = 1, z positive down into the solid.  Water: phi = e^{-j qw z} + R e^{+j qw z};
    solid: phi = T e^{-j qp z}, psi = S e^{-j qs z}; all times e^{-j p x}.
    u = grad(phi) + curl(psi y): ux = phi_x - psi_z, uz = phi_z + psi_x.
    BCs at z = 0: uz continuous, sigma_zz = -p_fluid = -rho_w phi_w, sigma_xz = 0.
    """
    p = np.sin(theta) / C
    qw = np.cos(theta) / C
    cpc, csc = cp * (1 + 0.5j * eta), cs * (1 + 0.5j * eta)
    qp, qs = _q(cpc, p), _q(csc, p)
    mu = rho * csc**2
    lam = rho * cpc**2 - 2 * mu
    dx = -1j * p
    # columns: R, T, S ; derivative factors d/dz: R-wave +j qw, T -j qp, S -j qs
    dzR, dzT, dzS = 1j * qw, -1j * qp, -1j * qs
    # uz: water (incident dz -j qw) + R dzR = T dzT + S dx
    row1 = [dzR, -dzT, -dx]
    rhs1 = 1j * qw  # -(incident uz) = -(-j qw)
    # sigma_zz solid: lam (dx ux + dz uz) + 2 mu dz uz
    #   phi part: lam (dx^2 + dz^2) + 2 mu dz^2 ; psi part: lam (dx(-dz) + dz dx) + 2 mu dz dx = 2 mu dz dx
    szT = lam * (dx**2 + dzT**2) + 2 * mu * dzT**2
    szS = 2 * mu * dzS * dx
    # fluid: sigma_zz = -rho_w (1 + R) (w = 1, phi -> p = rho w^2 phi)
    row2 = [-RHO_W, -szT, -szS]
    rhs2 = RHO_W
    # sigma_xz = mu (dz ux + dx uz): phi: 2 dx dz ; psi: dx^2 - dz^2
    row3 = [0.0, mu * 2 * dx * dzT, mu * (dx**2 - dzS**2)]
    A = np.array([row1, row2, row3], dtype=complex)
    b = np.array([rhs1, rhs2, 0.0], dtype=complex)
    return np.linalg.solve(A, b)[0]


@pytest.mark.parametrize("eta", [0.0, 0.03])
def test_halfspace_matches_boundary_condition_solve(eta):
    cp, cs, rho = 4000.0, 2300.0, 2300.0
    th = np.linspace(0.0, np.pi / 2 - 1e-3, 61)
    th = th[np.abs(np.sin(th) - C / cs) > 1e-3]  # avoid the exact critical angle (singular system)
    th = th[np.abs(np.sin(th) - C / cp) > 1e-3]
    ref = np.array([_R_bc(t, cp, cs, rho, eta) for t in th])
    got = halfspace_R(th, C, RHO_W, cp, cs, rho, eta)
    assert np.allclose(got, ref, atol=1e-9)


def test_halfspace_lossless_limit_is_continuous():
    """The lossless branch choice must be the eta -> 0+ limit of the lossy one."""
    th = np.linspace(0, np.pi / 2 - 1e-3, 200)
    a = halfspace_R(th, C, RHO_W, 4000.0, 2300.0, 2300.0, 0.0)
    b = halfspace_R(th, C, RHO_W, 4000.0, 2300.0, 2300.0, 1e-9)
    assert np.allclose(a, b, atol=1e-5)


def test_wall_image_uses_its_own_incidence_angle():
    pool = Pool(30.0, 20.0, 6.0)
    src, rx = np.array([1.0, 10.0, 3.0]), np.array([1.5, 18.0, 3.5])
    b = Boundary(cp=4000.0, cs=2300.0, rho=2300.0, eta=0.0)
    im = image_sources(src, rx, pool, C, 0.02, 0.0, b, 0.0, max_order=1)[0]
    w = im.subset(im.counts[:, 0] == 1)
    assert len(w) == 1
    img = np.array([-1.0, 10.0, 3.0])
    r = np.linalg.norm(rx - img)
    theta = np.arccos(abs(rx[0] - img[0]) / r)
    assert np.isclose(w.theta[0, 0], theta)
    assert np.isclose(w.amp[0], b.R(theta, C, RHO_W) / r)
    # near-grazing geometry is beyond critical -> complex R actually exercised
    assert theta > np.arcsin(C / b.cs) and abs(w.R[0].imag) > 0.1


def _specular_path(src, rx, seq, pool):
    """Explicit reflection points for plane sequence ``seq`` (src -> rx), or None if the
    path violates the law of reflection / leaves the faces.  Independent re-derivation:
    checks equal angles and path length directly instead of trusting the image construction."""
    planes = pool.planes()
    imgs, q = [], src
    for nm in seq:
        n, d = planes[nm]
        q = q - 2 * (q @ n - d) * n
        imgs.append(q)
    pts, cur = [], rx
    for nm, tgt in zip(reversed(seq), reversed(imgs)):
        n, d = planes[nm]
        v = tgt - cur
        if abs(n @ v) < 1e-12:
            return None
        cur = cur + (d - n @ cur) / (n @ v) * v
        pts.append(cur)
    path = [rx] + pts + [src]
    angles = {}
    for k, nm in enumerate(reversed(seq)):
        n, _ = planes[nm]
        a = path[k] - path[k + 1]
        b = path[k + 2] - path[k + 1]
        ca, cb = a @ n / np.linalg.norm(a), b @ n / np.linalg.norm(b)
        if not (np.isclose(ca, cb, atol=1e-9) and ca < 0 and pool.contains(path[k + 1], tol=1e-7)):
            return None
        angles[nm] = np.arccos(-ca)
    length = sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))
    return imgs[-1], length, angles


def test_exact_tilted_images_obey_law_of_reflection():
    pool = Pool(20.0, 10.0, 2.0, 5.0)  # strong slope
    src, rx = np.array([4.0, 3.0, 1.2]), np.array([12.0, 6.0, 2.5])
    pe, ce, te = exact_floor_images(src, rx, pool, 2)
    names = list(pool.planes())
    seqs = [("floor",)] + [s for o in names if o != "floor" for s in (("floor", o), (o, "floor"))]
    valid = [v for v in (_specular_path(src, rx, s, pool) for s in seqs) if v is not None]
    assert len(pe) == len(valid) >= 4
    for img, length, angles in valid:
        i = int(np.argmin(np.linalg.norm(pe - img, axis=1)))
        assert np.allclose(pe[i], img)
        assert np.isclose(np.linalg.norm(rx - pe[i]), length)  # unfolded length == true path length
        for nm, th in angles.items():
            assert np.isclose(te[i, names.index(nm)], th)


@pytest.mark.parametrize("pool", [Pool(12.0, 8.0, 3.0), avery_competition_pool()])
def test_reciprocity(pool):
    """Swapping source and receiver must give the same response (all reflection
    models are reciprocal; mean depth uses the symmetric midpoint)."""
    a, b = np.array([3.0, 2.0, 0.7]), np.array([9.5, 6.0, 2.1])
    kw = dict(c=C, fs=FS, t_max=0.03, walls=Boundary(eta=0.01), floor=Boundary(5000, 2800, 2400, 0.0))
    r1 = ism_rir(a, b, pool, **kw)
    r2 = ism_rir(b, a, pool, **kw)
    assert len(r1.images) == len(r2.images)
    assert np.allclose(r1.rir, r2.rir, atol=1e-9 * np.max(np.abs(r1.rir)))


def test_freq_render_no_wraparound():
    """An image at the end of the window must not leak before it (time aliasing)."""
    n_out = 2000
    h = render_freq(np.array([1990.3 / FS]), np.array([1.0]), FS, n_out)
    assert np.max(np.abs(h[:1500])) < 2e-3


def test_as_reflection_accepts_numpy_scalars():
    f = as_reflection(np.float64(-0.5))
    assert np.allclose(f(np.zeros(3)), -0.5)
    g = as_reflection(np.array(0.3))
    assert np.allclose(g(np.zeros(2)), 0.3)


def test_tilted_mean_depth_below_source_raises_or_is_sane():
    """Deep-end source below the mean depth of the path: every reflected path must be
    longer than the direct path (a reflection can never arrive first)."""
    pool = avery_competition_pool()
    src, rx = np.array([36.0, 10.0, 4.2]), np.array([2.0, 10.0, 1.0])
    im = image_sources(src, rx, pool, C, 0.05, -1.0, 0.9, 0.9)[0]
    direct = np.linalg.norm(src - rx)
    assert np.isclose(im.dist[0], direct) and im.order[0] == 0
    assert np.all(im.dist[1:] > direct)


def test_appendix_a_asymmetric_depth_delays():
    """Plan Appendix A / 4.3 step 1: source 0.7 m, receiver 1.2 m, horizontal range 1 m, H = 2 m:
    surface image 0.69 ms after the direct path, floor image 0.12 ms after the surface image."""
    pool = Pool(30.0, 20.0, 2.0)
    src, rx = np.array([10.0, 10.0, 0.7]), np.array([11.0, 10.0, 1.2])
    im = image_sources(src, rx, pool, C, 0.004, -1.0, 0.0, 0.8, max_order=1)[0]
    d0 = im.delay[im.order == 0][0]
    ds = im.delay[im.counts[:, 4] == 1][0]
    dfl = im.delay[im.counts[:, 5] == 1][0]
    assert round((ds - d0) * 1e3, 2) == 0.69
    assert round((dfl - ds) * 1e3, 2) == 0.12
    assert im.amp[im.counts[:, 4] == 1][0] < 0 < im.amp[im.counts[:, 5] == 1][0]


@pytest.mark.parametrize("k", [-1, 3])
def test_exact_floor_order_out_of_range_raises(k):
    with pytest.raises(ValueError):
        image_sources(np.array([1.0, 1.0, 1.0]), np.array([3.0, 2.0, 1.5]), Pool(10.0, 5.0, 2.0, 3.0),
                      exact_floor_order=k)


def test_render_freq_rejects_aliasing_fft_length():
    """n_fft == n_out lets the sinc / Hilbert tails of late images wrap onto t = 0."""
    with pytest.raises(ValueError):
        render_freq(np.array([390.5 / FS]), np.array([1.0]), FS, 400, n_fft=512)
