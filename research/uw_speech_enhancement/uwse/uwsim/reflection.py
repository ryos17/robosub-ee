"""Plane-wave reflection coefficients for the pool boundaries.

Implements RESEARCH_PLAN.md section 4.2 rows 4 (surface) and 5 (walls and floor):

* the free surface is a real, *signed* constant R_s (default -1.0);
* walls and floor are fluid--elastic half-spaces (water over concrete or tile)
  with an angle-dependent reflection coefficient R(theta), textbook form of
  Brekhovskikh (Waves in Layered Media) / Jensen et al. (Computational Ocean
  Acoustics, ch. 1):

      R = (Z_p cos^2(2 theta_s) + Z_s sin^2(2 theta_s) - Z_w)
          / (Z_p cos^2(2 theta_s) + Z_s sin^2(2 theta_s) + Z_w),
      Z_w = rho_w c_w / cos(theta),  Z_p = rho c_p / cos(theta_p),
      Z_s = rho c_s / cos(theta_s),

  with Snell's law sin(theta_p)/c_p = sin(theta_s)/c_s = sin(theta)/c_w.

Conventions
-----------
* ``theta`` is the incidence angle measured from the boundary normal
  (0 = normal incidence, pi/2 = grazing).
* Time dependence is **exp(+j 2 pi f t)** (engineering convention), the same
  one used by the ISM synthesis (H(f) = sum_i A_i exp(-j 2 pi f tau_i)).  The
  returned complex R applies to positive frequencies; negative frequencies use
  conj(R).  Jensen et al. use exp(-i omega t), so their R is the complex
  conjugate of ours.
* Losses enter through complex wave speeds c* = c (1 + j eta / 2) (eta is the
  loss factor; for eta << 1 this is an attenuation of pi*eta nepers, i.e.
  27.3*eta dB, per wavelength).  With exp(+j omega t) this gives Im(k) < 0, a
  decaying wave.  The vertical slowness in the lower medium is taken on the
  branch Im(q) <= 0 so that beyond a critical angle the transmitted field is
  evanescent and, for a lossless solid, |R| = 1 exactly with a phase shift.
* The plane-wave R of a homogeneous half-space has no length scale, so it is
  frequency-independent; R(theta, f) of the plan reduces to R(theta) here.

The material values below are **uncalibrated textbook priors** (plan table,
row 5, marked "unverified"); the calibrated values come from the pool fit
(section 4.3, step 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

RHO_WATER = 997.0
"""Fresh-water density at about 25 degC [kg/m^3]."""

C_WATER = 1500.0
"""Nominal sound speed [m/s]; use the measured TOF-slope value in practice (section 4.3)."""

SURFACE_R_DEFAULT = -1.0
"""Signed pressure-release surface (plan row 4)."""

ReflectionLike = Union["Boundary", float, complex, Callable[[np.ndarray], np.ndarray]]


def _vertical_slowness(c: complex, sin_theta: np.ndarray, c_w: float) -> np.ndarray:
    """q = sqrt(1/c^2 - sin^2(theta)/c_w^2) on the branch Im(q) <= 0, Re(q) >= 0.

    q = cos(theta_m)/c_m is the vertical slowness in medium m.  For a lossless
    medium beyond its critical angle the radicand is negative real and numpy's
    principal root returns +j|q|; that branch would grow into the solid under
    exp(+j omega t), so it is conjugated to -j|q|.  For lossy media the radicand
    has Im < 0 and the principal root already lies on the decaying branch.
    """
    q = np.sqrt(1.0 / complex(c) ** 2 - (sin_theta / c_w) ** 2 + 0j)
    return np.where(q.imag > 0, np.conj(q), q)


def fluid_fluid_R(
    theta_inc: np.ndarray | float,
    c_w: float,
    rho_w: float,
    c2: float,
    rho2: float,
    eta: float = 0.0,
) -> np.ndarray:
    """Rayleigh reflection coefficient of a fluid half-space (no shear).

    R = (rho2 cos(theta)/c_w - rho_w q2) / (rho2 cos(theta)/c_w + rho_w q2), q2 the
    vertical slowness in the lower fluid (see module docstring for conventions).
    """
    if eta < 0:
        raise ValueError("eta must be >= 0")
    theta = np.asarray(theta_inc, dtype=float)
    s = np.sin(theta)
    qw = np.cos(theta) / c_w
    q2 = _vertical_slowness(c2 * (1.0 + 0.5j * eta), s, c_w)
    return (rho2 * qw - rho_w * q2) / (rho2 * qw + rho_w * q2)


def halfspace_R(
    theta_inc: np.ndarray | float,
    c_w: float,
    rho_w: float,
    cp: float,
    cs: float,
    rho: float,
    eta: float = 0.0,
    eta_s: float | None = None,
) -> np.ndarray:
    """Plane-wave reflection coefficient of water over an elastic half-space.

    Parameters
    ----------
    theta_inc : incidence angle(s) from the normal [rad], in [0, pi/2].
    c_w, rho_w : water sound speed [m/s] and density [kg/m^3].
    cp, cs, rho : compressional and shear speeds [m/s] and density of the solid.
        ``cs == 0`` reduces to the fluid--fluid case (:func:`fluid_fluid_R`).
    eta : loss factor of the compressional wave (c_p* = c_p (1 + j eta/2)).
    eta_s : loss factor of the shear wave; defaults to ``eta``.

    Returns
    -------
    complex ndarray with the shape of ``theta_inc`` (exp(+j omega t) convention).
    Normal incidence gives (rho cp - rho_w c_w)/(rho cp + rho_w c_w); grazing
    incidence gives -1.
    """
    if cs == 0:
        return fluid_fluid_R(theta_inc, c_w, rho_w, cp, rho, eta)
    eta_s = eta if eta_s is None else eta_s
    if eta < 0 or eta_s < 0:
        raise ValueError("loss factors must be >= 0")
    theta = np.asarray(theta_inc, dtype=float)
    s = np.sin(theta)
    qw = np.cos(theta) / c_w
    cp_c = cp * (1.0 + 0.5j * eta)
    cs_c = cs * (1.0 + 0.5j * eta_s)
    qp = _vertical_slowness(cp_c, s, c_w)
    qs = _vertical_slowness(cs_c, s, c_w)
    # Only squares of sin/cos(theta_s) appear, so no extra branch choice is needed.
    S = (cs_c * s / c_w) ** 2  # sin^2(theta_s)
    cos2_sq = (1.0 - 2.0 * S) ** 2  # cos^2(2 theta_s)
    sin2_sq = 4.0 * S * (1.0 - S)  # sin^2(2 theta_s)
    # Z_tot = Z_p cos^2 2ts + Z_s sin^2 2ts = N / D with Z_p = rho/qp, Z_s = rho/qs;
    # Z_w = rho_w/qw.  Clearing denominators avoids 1/0 at grazing and critical angles.
    N = rho * (cos2_sq * qs + sin2_sq * qp)
    D = qp * qs
    return (N * qw - rho_w * D) / (N * qw + rho_w * D)


def surface_R(theta_inc: np.ndarray | float | None = None, R_s: float = SURFACE_R_DEFAULT) -> np.ndarray:
    """Signed, angle-independent free-surface coefficient (plan row 4).

    Returns ``R_s`` broadcast to the shape of ``theta_inc`` (a scalar array if None).
    """
    shape = () if theta_inc is None else np.shape(theta_inc)
    return np.full(shape, float(R_s))


def pressure_release_R(
    c_w: float = C_WATER, rho_w: float = RHO_WATER, c_air: float = 343.0, rho_air: float = 1.2
) -> float:
    """Normal-incidence water-to-air coefficient (about -0.9994; plan row 4 note)."""
    z_w, z_a = rho_w * c_w, rho_air * c_air
    return (z_a - z_w) / (z_a + z_w)


@dataclass(frozen=True)
class Boundary:
    """Elastic half-space parameters of one boundary (plan row 5).

    Defaults are **uncalibrated** mid-range textbook priors for pool concrete
    (plan: c_p 3500--5500 m/s, c_s 2000--3000 m/s, rho 2200--2500 kg/m^3; eta is
    a guess).  Replace with the section 4.3 fit for the real pool.
    """

    cp: float = 4000.0
    cs: float = 2300.0
    rho: float = 2300.0
    eta: float = 0.02
    eta_s: float | None = None

    def R(self, theta_inc: np.ndarray | float, c_w: float = C_WATER, rho_w: float = RHO_WATER) -> np.ndarray:
        return halfspace_R(theta_inc, c_w, rho_w, self.cp, self.cs, self.rho, self.eta, self.eta_s)

    def critical_angles(self, c_w: float = C_WATER) -> tuple[float, float]:
        """(compressional, shear) critical angles [rad]; nan where c <= c_w."""
        tp = np.arcsin(c_w / self.cp) if self.cp > c_w else np.nan
        ts = np.arcsin(c_w / self.cs) if self.cs > c_w else np.nan
        return float(tp), float(ts)

    def normal_incidence_R(self, c_w: float = C_WATER, rho_w: float = RHO_WATER) -> float:
        z2, z1 = self.rho * self.cp, rho_w * c_w
        return (z2 - z1) / (z2 + z1)


CONCRETE_UNCALIBRATED = Boundary(cp=4000.0, cs=2300.0, rho=2300.0, eta=0.02)
"""Textbook-prior concrete (uncalibrated)."""

TILE_UNCALIBRATED = Boundary(cp=5000.0, cs=2800.0, rho=2400.0, eta=0.01)
"""Textbook-prior ceramic tile on concrete, treated as one half-space (uncalibrated)."""


def anechoic(c_w: float = C_WATER, rho_w: float = RHO_WATER) -> Boundary:
    """Impedance-matched fluid boundary: R = 0 at every angle (for tests and ablations)."""
    return Boundary(cp=c_w, cs=0.0, rho=rho_w, eta=0.0)


def as_reflection(
    spec: ReflectionLike, c_w: float = C_WATER, rho_w: float = RHO_WATER
) -> Callable[[np.ndarray], np.ndarray]:
    """Turn a Boundary, a constant (real or complex) or a callable into R(theta)."""
    if isinstance(spec, Boundary):
        return lambda th: spec.R(th, c_w, rho_w)
    if callable(spec):
        return lambda th: np.asarray(spec(th), dtype=complex)
    if np.isscalar(spec) or (isinstance(spec, np.ndarray) and spec.ndim == 0):
        val = complex(spec)
        return lambda th: np.full(np.shape(th), val, dtype=complex)
    raise TypeError(f"cannot interpret {spec!r} as a reflection coefficient")
