"""Signed, angle-dependent image-source model (ISM) of a tilted-floor pool.

Implements RESEARCH_PLAN.md section 4.1 (ISM_early), section 4.2 rows 3--6 and
12--13, the section 4.2 "Implementation" bullets, and the geometry needed by
the section 4.3 calibration.

Model
-----
* Point source, point receiver, spherical spreading: an image at distance r_i
  contributes A_i = (prod of reflection coefficients) / r_i at delay
  tau_i = r_i / c (pressure re the 1 m level of the direct path).
* Free surface: real, signed R_s raised to the number of surface bounces.
* Walls and floor: R(theta) of :mod:`uwse.uwsim.reflection`, evaluated at each
  reflection's own incidence angle.  For shoebox images every bounce on a
  given pair of parallel planes has the same incidence angle, read off the
  unfolded image--receiver vector.
* Geometry (row 3): images that involve the floor and have total order <= 2
  (configurable, ``exact_floor_order``) are computed **exactly** by mirroring
  the source across the tilted floor plane (and the other plane, in both
  orders) and keeping only the specularly valid ones (back-traced reflection
  points must lie on the faces).  All other images come from the
  Allen--Berkley shoebox lattice with depth H = the mean floor depth along the
  src--rx path (:meth:`Pool.mean_depth_along`).  This is an approximation for
  orders >= 3 with a sloped floor; :func:`mean_depth_error` reports its error
  against the exact 2nd-order images.  For a flat floor both are identical.

Rendering
---------
Default ``method="freq"``: frequency-domain synthesis
H(f_k) = sum_i A_i exp(-j 2 pi f_k tau_i) on an rfft grid, then irfft.  This is
chosen because beyond the critical angle R is complex with a constant phase
(R at f > 0, conj(R) at f < 0), i.e. a band-limited Hilbert-transform pulse
whose 1/t tail a short time-domain kernel cannot represent; the frequency
domain renders it, and fractional delays, exactly (ideal band limit).  The FFT
length is padded to >= 4x the output length so the periodic-sinc wrap-around is
negligible, and only the first ``n_out`` samples are returned.  An optional
raised-cosine roll-off over the top ``rolloff`` fraction of the band shortens
the sinc ringing.  Cost is O(N_images x n_bins) multiply-adds done as a matrix
product (about 2 s for 7e4 images over 0.1 s at 48 kHz on a laptop CPU).
``method="sinc"`` is the plan's 64-tap Hann-windowed fractional-delay sinc
(plus a windowed discrete Hilbert kernel for Im(A)); it is O(N_images x 64)
and meant for very long responses with many images, at the cost of
low-frequency error on complex-R images (below roughly fs / 64).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .geometry import SURFACE_NAMES, Pool, mirror_point
from .reflection import (
    C_WATER,
    CONCRETE_UNCALIBRATED,
    RHO_WATER,
    SURFACE_R_DEFAULT,
    ReflectionLike,
    as_reflection,
)

_IX0, _IXL, _IY0, _IYW, _IS, _IF = range(6)
_SINC_TAPS = 64


@dataclass
class ImageSet:
    """Image sources, one row per image (direct path included, order 0).

    Attributes
    ----------
    pos : (N, 3) image positions [m] (for exact tilted-floor images, the true mirror point).
    dist : (N,) unfolded path length = |rx - pos| [m].
    delay : (N,) dist / c [s].
    R : (N,) complex product of all reflection coefficients along the path.
    amp : (N,) complex amplitude R / dist.
    counts : (N, 6) int reflections per surface, columns :data:`SURFACE_NAMES`
        (x0, xL, y0, yW, surface, floor).
    theta : (N, 6) incidence angle from the normal [rad] of the reflections on
        each surface (nan where that surface is not hit).
    exact : (N,) True for images computed against the exact tilted floor plane.
    """

    pos: np.ndarray
    dist: np.ndarray
    delay: np.ndarray
    R: np.ndarray
    amp: np.ndarray
    counts: np.ndarray
    theta: np.ndarray
    exact: np.ndarray

    columns = SURFACE_NAMES

    def __len__(self) -> int:
        return len(self.dist)

    @property
    def order(self) -> np.ndarray:
        return self.counts.sum(axis=1)

    def subset(self, mask: np.ndarray) -> "ImageSet":
        return ImageSet(*(getattr(self, k)[mask] for k in self.__dataclass_fields__))

    def sorted(self) -> "ImageSet":
        return self.subset(np.argsort(self.delay, kind="stable"))

    @staticmethod
    def concat(sets: list["ImageSet"]) -> "ImageSet":
        return ImageSet(*(np.concatenate([getattr(s, k) for s in sets]) for k in ImageSet.__dataclass_fields__))


@dataclass
class ISMResult:
    """Output of :func:`ism_rir`.  ``rir[n]`` is the response at t = n / fs (t = 0 at emission)."""

    rir: np.ndarray
    fs: int
    images: ImageSet
    c: float
    t_max: float
    h_mean: float

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.rir)) / self.fs


# --------------------------------------------------------------------------
# image enumeration
# --------------------------------------------------------------------------
def _axis_images(xs: float, xr: float, L: float, D: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-D Allen--Berkley images x = (1 - 2p) xs + 2 m L within D of xr.

    Returns (coordinate, bounces on the plane at 0, bounces on the plane at L);
    the counts are |m - p| and |m|.
    """
    m_max = int(np.ceil((D + abs(xr) + abs(xs)) / (2.0 * L))) + 1
    m = np.arange(-m_max, m_max + 1)
    coord = np.concatenate([xs + 2 * m * L, -xs + 2 * m * L])
    n_lo = np.concatenate([np.abs(m), np.abs(m - 1)])
    n_hi = np.concatenate([np.abs(m), np.abs(m)])
    keep = np.abs(coord - xr) <= D
    return coord[keep], n_lo[keep], n_hi[keep]


def shoebox_images(
    src: np.ndarray,
    rx: np.ndarray,
    dims: tuple[float, float, float],
    D: float,
    max_order: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """All shoebox images within distance D of rx (vectorized).

    ``dims`` = (L, W, H) with the surface at z = 0 and the floor at z = H.
    Returns (pos (N, 3), counts (N, 6)).
    """
    src = np.asarray(src, float)
    rx = np.asarray(rx, float)
    ax = [_axis_images(src[i], rx[i], dims[i], D) for i in range(3)]
    (cx, x0, xL), (cy, y0, yW), (cz, zs, zf) = ax
    D2 = D * D
    dx2, dy2, dz2 = (cx - rx[0]) ** 2, (cy - rx[1]) ** 2, (cz - rx[2]) ** 2
    mxy = dx2[:, None] + dy2[None, :] <= D2
    if max_order is not None:
        mxy &= (x0 + xL)[:, None] + (y0 + yW)[None, :] <= max_order
    ix, iy = np.nonzero(mxy)
    rxy = dx2[ix] + dy2[iy]
    oxy = (x0 + xL)[ix] + (y0 + yW)[iy]
    # Chunk over xy pairs so the (pairs x z) mask stays small.
    chunk = max(1, int(2**24 // max(len(cz), 1)))
    out_xy, out_z = [], []
    for s in range(0, len(ix), chunk):
        m = rxy[s : s + chunk, None] + dz2[None, :] <= D2
        if max_order is not None:
            m &= oxy[s : s + chunk, None] + (zs + zf)[None, :] <= max_order
        j, k = np.nonzero(m)
        out_xy.append(j + s)
        out_z.append(k)
    j = np.concatenate(out_xy) if out_xy else np.zeros(0, int)
    k = np.concatenate(out_z) if out_z else np.zeros(0, int)
    a, b = ix[j], iy[j]
    pos = np.stack([cx[a], cy[b], cz[k]], axis=1)
    counts = np.stack([x0[a], xL[a], y0[b], yW[b], zs[k], zf[k]], axis=1).astype(int)
    return pos, counts


def _shoebox_angles(pos: np.ndarray, rx: np.ndarray, counts: np.ndarray) -> np.ndarray:
    v = np.abs(rx[None, :] - pos)
    dist = np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ang = np.arccos(np.clip(v / dist[:, None], 0.0, 1.0))  # angle to the x, y, z normals
    theta = ang[:, [0, 0, 1, 1, 2, 2]]
    theta[counts == 0] = np.nan
    return theta


def exact_floor_images(
    src: np.ndarray, rx: np.ndarray, pool: Pool, max_order: int = 2, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Floor-involving images of order <= max_order (1 or 2) against the exact tilted plane.

    Each candidate sequence (first-hit plane, second-hit plane) is built by
    successive mirroring and kept only if the back-traced reflection points
    lie on the corresponding faces of the (convex) pool, i.e. the specular path
    exists.  For a flat floor this reproduces the shoebox images exactly.

    Returns (pos (N, 3), counts (N, 6), theta (N, 6)), with theta the incidence
    angle of each reflection, computed from the actual ray segments.
    """
    if max_order not in (1, 2):
        raise ValueError("exact floor images are implemented for orders 1 and 2")
    src = np.asarray(src, float)
    rx = np.asarray(rx, float)
    planes = pool.planes()
    names = list(SURFACE_NAMES)
    seqs: list[tuple[str, ...]] = [("floor",)]
    if max_order >= 2:
        for other in names:
            if other != "floor":
                seqs += [("floor", other), (other, "floor")]
    pos_l, cnt_l, th_l = [], [], []
    for seq in seqs:
        imgs = []
        p = src
        for name in seq:
            p = mirror_point(p, *planes[name])
            imgs.append(p)
        counts = np.zeros(6, int)
        theta = np.full(6, np.nan)
        cur, ok = rx, True
        for name, target in zip(reversed(seq), reversed(imgs)):
            n, d = planes[name]
            v = target - cur
            denom = n @ v
            if denom <= tol:
                ok = False
                break
            t = (d - n @ cur) / denom
            if not (tol < t < 1.0 - tol):
                ok = False
                break
            hit = cur + t * v
            if not pool.contains(hit, tol=1e-7):
                ok = False
                break
            col = names.index(name)
            counts[col] += 1
            theta[col] = np.arccos(min(1.0, abs(denom) / np.linalg.norm(v)))
            cur = hit
        if ok:
            pos_l.append(imgs[-1])
            cnt_l.append(counts)
            th_l.append(theta)
    if not pos_l:
        return np.zeros((0, 3)), np.zeros((0, 6), int), np.zeros((0, 6))
    return np.array(pos_l), np.array(cnt_l), np.array(th_l)


def _reflection_product(
    counts: np.ndarray, theta: np.ndarray, R_s: float, R_wall, R_floor
) -> np.ndarray:
    R = np.ones(len(counts), dtype=complex)
    for col, func in ((_IX0, R_wall), (_IXL, R_wall), (_IY0, R_wall), (_IYW, R_wall), (_IF, R_floor)):
        n = counts[:, col]
        m = n > 0
        if m.any():
            R[m] *= func(theta[m, col]) ** n[m]
    R *= float(R_s) ** counts[:, _IS]
    return R


def image_sources(
    src: np.ndarray,
    rx: np.ndarray,
    pool: Pool,
    c: float = C_WATER,
    t_max: float = 0.04,
    surface_R: float = SURFACE_R_DEFAULT,
    walls: ReflectionLike = CONCRETE_UNCALIBRATED,
    floor: ReflectionLike = CONCRETE_UNCALIBRATED,
    max_order: int | None = None,
    exact_floor_order: int = 2,
    rho_w: float = RHO_WATER,
    drop_zero: bool = True,
) -> tuple[ImageSet, float]:
    """All images with delay <= t_max (and order <= max_order).  Returns (images, H_mean)."""
    if exact_floor_order not in (0, 1, 2):
        raise ValueError("exact_floor_order must be 0, 1 or 2 (exact tilted-floor images beyond order 2 are not implemented)")
    src = pool.check_position(src, "src")
    rx = pool.check_position(rx, "rx")
    if np.allclose(src, rx):
        raise ValueError("src and rx coincide")
    D = c * t_max
    h_mean = pool.mean_depth_along(src, rx)
    pos, counts = shoebox_images(src, rx, (pool.L, pool.W, h_mean), D, max_order)
    theta = _shoebox_angles(pos, rx, counts)
    exact = np.zeros(len(pos), bool)
    if exact_floor_order > 0:
        k = exact_floor_order
        keep = ~((counts[:, _IF] > 0) & (counts.sum(1) <= k))
        pe, ce, te = exact_floor_images(src, rx, pool, k)
        if max_order is not None:
            sel = ce.sum(1) <= max_order
            pe, ce, te = pe[sel], ce[sel], te[sel]
        sel = np.linalg.norm(pe - rx, axis=1) <= D
        pos = np.concatenate([pos[keep], pe[sel]])
        counts = np.concatenate([counts[keep], ce[sel]])
        theta = np.concatenate([theta[keep], te[sel]])
        exact = np.concatenate([exact[keep], np.ones(int(sel.sum()), bool)])
    dist = np.linalg.norm(pos - rx, axis=1)
    R = _reflection_product(
        counts, theta, surface_R, as_reflection(walls, c, rho_w), as_reflection(floor, c, rho_w)
    )
    imgs = ImageSet(pos, dist, dist / c, R, R / dist, counts, theta, exact)
    if drop_zero:
        imgs = imgs.subset(imgs.R != 0)
    return imgs.sorted(), h_mean


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 2))))


def render_freq(
    delay: np.ndarray,
    amp: np.ndarray,
    fs: float,
    n_out: int,
    n_fft: int | None = None,
    rolloff: float = 0.0,
    max_elems: int = 2**22,
) -> np.ndarray:
    """Band-limited sum of complex-weighted delayed impulses via the frequency domain.

    rir[n] = irfft(sum_i A_i exp(-j 2 pi f_k tau_i))[n], n < n_out.  A complex A_i
    acts as A_i at f > 0 and conj(A_i) at f < 0 (DC uses Re(A_i)).  ``n_fft`` must
    be >= 2 n_out (default: next power of two >= 4 n_out) so the periodic
    synthesis does not wrap late-image tails onto the start.
    """
    n_fft = _next_pow2(4 * n_out) if n_fft is None else int(n_fft)
    if n_fft < 2 * n_out:
        # The synthesis is periodic in n_fft: with less than n_out of padding the
        # band-limited tails of the latest images wrap onto the start of the response.
        raise ValueError(f"n_fft={n_fft} must be >= 2 * n_out = {2 * n_out} to avoid time aliasing")
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    K = len(f)
    amp = np.asarray(amp, dtype=complex)
    delay = np.asarray(delay, dtype=float)
    # Factor the bin index k = j*B + b so exp(-j w_k tau) = exp(-j w_{jB} tau) exp(-j w_b tau):
    # only (J + B) complex exponentials per image instead of K, and the sum over
    # images becomes one (J x n) @ (n x B) matrix product.
    B = int(np.ceil(np.sqrt(K)))
    J = int(np.ceil(K / B))
    df = fs / n_fft
    fj = np.arange(J) * B * df
    fb = np.arange(B) * df
    Hm = np.zeros((J, B), dtype=complex)
    chunk = max(1, max_elems // (J + B))
    starts = range(0, len(delay), chunk)
    for s in tqdm(starts, desc="ism freq synth", disable=len(starts) < 8, leave=False):
        tau = delay[s : s + chunk]
        base = np.exp(-2j * np.pi * np.outer(fj, tau)) * amp[s : s + chunk][None, :]
        step = np.exp(-2j * np.pi * np.outer(tau, fb))
        Hm += base @ step
    H = Hm.ravel()[:K]
    H[0] = amp.real.sum()
    if rolloff > 0:
        f_nyq = fs / 2.0
        f1 = (1.0 - rolloff) * f_nyq
        w = np.ones_like(f)
        band = f > f1
        w[band] = 0.5 * (1.0 + np.cos(np.pi * (f[band] - f1) / (f_nyq - f1)))
        H *= w
    return np.fft.irfft(H, n_fft)[:n_out]


def render_sinc(
    delay: np.ndarray, amp: np.ndarray, fs: float, n_out: int, n_taps: int = _SINC_TAPS, max_images: int = 2**15
) -> np.ndarray:
    """64-tap Hann-windowed fractional-delay sinc (plan row 6).

    Re(A) is rendered with sinc(d); Im(A) with the band-limited kernel of
    j sgn(f), g(d) = -(1 - cos(pi d)) / (pi d).  The truncated Hilbert kernel is
    inaccurate below roughly fs / n_taps.
    """
    half = n_taps // 2
    k = np.arange(-half + 1, half + 1)
    rir = np.zeros(n_out)
    amp = np.asarray(amp, dtype=complex)
    starts = range(0, len(delay), max_images)
    for s in tqdm(starts, desc="ism sinc synth", disable=len(starts) < 8, leave=False):
        pos = np.asarray(delay[s : s + max_images]) * fs
        a = amp[s : s + max_images]
        idx = np.floor(pos).astype(int)[:, None] + k[None, :]
        d = idx - pos[:, None]
        w = 0.5 * (1.0 + np.cos(np.pi * d / half))
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(np.abs(d) < 1e-12, 0.0, -(1.0 - np.cos(np.pi * d)) / (np.pi * d))
        h = (a.real[:, None] * np.sinc(d) + a.imag[:, None] * g) * w
        valid = (idx >= 0) & (idx < n_out)
        rir += np.bincount(idx[valid], weights=h[valid], minlength=n_out)
    return rir


def ism_rir(
    src: np.ndarray,
    rx: np.ndarray,
    pool: Pool,
    c: float = C_WATER,
    fs: int = 48000,
    t_max: float = 0.04,
    surface_R: float = SURFACE_R_DEFAULT,
    walls: ReflectionLike = CONCRETE_UNCALIBRATED,
    floor: ReflectionLike = CONCRETE_UNCALIBRATED,
    max_order: int | None = None,
    n_fft: int | None = None,
    exact_floor_order: int = 2,
    method: str = "freq",
    rolloff: float = 0.05,
    rho_w: float = RHO_WATER,
) -> ISMResult:
    """Early room impulse response of a tilted-floor pool by the signed ISM.

    Parameters
    ----------
    src, rx : (x, y, z) [m], z positive down; must be inside the water.
    pool : :class:`~uwse.uwsim.geometry.Pool`.
    c : sound speed [m/s] (use the measured TOF slope, section 4.3).
    fs : sample rate [Hz] (48 kHz internal rate, section 4.1).
    t_max : include every image with delay <= t_max [s] (t_e = 40 ms in v0).
    surface_R : signed real surface coefficient, raised to the bounce count.
    walls, floor : :class:`Boundary` (angle-dependent half-space R), a constant
        (real or complex, angle-independent) or a callable R(theta).
    max_order : optional cap on the total reflection order.
    n_fft : FFT length for ``method="freq"`` (default next pow2 >= 4 n_out).
    exact_floor_order : floor images up to this total order (0, 1 or 2) are
        exact against the tilted plane; higher ones use the mean depth.
    method : "freq" (exact band-limited synthesis, default) or "sinc".
    rolloff : raised-cosine taper over the top fraction of the band ("freq" only).
    rho_w : water density [kg/m^3].

    Returns
    -------
    :class:`ISMResult`; ``rir`` has length ceil(t_max fs) + 32 so the last
    image's main lobe is not cut, sample n at t = n / fs.
    """
    imgs, h_mean = image_sources(
        src, rx, pool, c, t_max, surface_R, walls, floor, max_order, exact_floor_order, rho_w
    )
    n_out = int(np.ceil(t_max * fs)) + _SINC_TAPS // 2
    if method == "freq":
        rir = render_freq(imgs.delay, imgs.amp, fs, n_out, n_fft, rolloff)
    elif method == "sinc":
        rir = render_sinc(imgs.delay, imgs.amp, fs, n_out)
    else:
        raise ValueError(f"unknown method {method!r}")
    return ISMResult(rir=rir, fs=fs, images=imgs, c=c, t_max=t_max, h_mean=h_mean)


# --------------------------------------------------------------------------
# approximation error of the mean-depth floor
# --------------------------------------------------------------------------
@dataclass
class MeanDepthError:
    """Exact vs mean-depth floor images of order 1--2, matched by reflection counts."""

    counts: np.ndarray  # (N, 6)
    delay_exact: np.ndarray  # [s]
    delay_approx: np.ndarray  # [s]
    level_error_db: np.ndarray  # 20 log10(r_exact / r_approx): spreading-level error of the approx [dB]
    h_mean: float

    @property
    def delay_error(self) -> np.ndarray:
        """delay_approx - delay_exact [s]."""
        return self.delay_approx - self.delay_exact

    @property
    def max_abs_delay_error(self) -> float:
        return float(np.max(np.abs(self.delay_error))) if len(self.delay_error) else 0.0


def mean_depth_error(src: np.ndarray, rx: np.ndarray, pool: Pool, c: float = C_WATER) -> MeanDepthError:
    """Error of the mean-depth (flat-floor) approximation against the exact
    first- and second-order tilted-floor images (plan row 3 requires it be reported).

    Each valid exact image is paired with the mean-depth shoebox image having
    the same per-surface reflection counts (nearest in delay if several).
    """
    src = pool.check_position(src, "src")
    rx = pool.check_position(rx, "rx")
    h_mean = pool.mean_depth_along(src, rx)
    pe, ce, _ = exact_floor_images(src, rx, pool, 2)
    de = np.linalg.norm(pe - rx, axis=1)
    D = (de.max() if len(de) else 0.0) + 2.0 * (pool.L + pool.W + max(pool.H0, pool.H1))
    pa, ca = shoebox_images(src, rx, (pool.L, pool.W, h_mean), D, max_order=2)
    da = np.linalg.norm(pa - rx, axis=1)
    approx = np.empty(len(de))
    for i in range(len(de)):  # at most 11 exact images
        cand = np.all(ca == ce[i], axis=1)
        if not cand.any():
            approx[i] = np.nan
            continue
        dd = da[cand]
        approx[i] = dd[np.argmin(np.abs(dd - de[i]))]
    return MeanDepthError(
        counts=ce,
        delay_exact=de / c,
        delay_approx=approx / c,
        level_error_db=20 * np.log10(de / approx),
        h_mean=h_mean,
    )
