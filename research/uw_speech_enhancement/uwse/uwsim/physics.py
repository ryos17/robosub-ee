"""Closed-form water-acoustics helpers for the pool channel simulator.

Implements RESEARCH_PLAN.md section 4.2:

* row 1  (sound speed): Marczak (1997) pure-water c(T), plus a small-salinity
  correction for salt-chlorinated pools;
* row 2  (absorption): Francois & Garrison (1982) alpha(f, T, S, D, pH);
* row 9  (cutoff): quarter-wave cutoff of a pressure-release / rigid-bottom
  layer, f_c = c / (4 H);
* row 16 (bubbles): Minnaert resonance f0(a);

and the octave-band grid (250 Hz - 8 kHz) used throughout section 4 (tail
fitting, per-band perturbation, validation metrics in section 4.4).

Units: SI unless the name says otherwise (alpha is returned in dB/km because
that is how F&G tabulate it).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sound_speed_marczak",
    "salinity_correction",
    "sound_speed",
    "francois_garrison_alpha",
    "absorption_db",
    "waveguide_cutoff",
    "minnaert_frequency",
    "octave_bands",
]

# Marczak (1997), "Water as a standard in the measurements of speed of sound in
# liquids", JASA 102(5), 2776-2779; 5th-order fit to Del Grosso & Mader, Kroebel &
# Mahrt and Fujii & Masui, valid 0-95 degC at 1 atm.  Coefficients as tabulated by
# NPL ("Speed of sound in pure water"); they give 1482.36 m/s at 20 degC.
_MARCZAK = (
    1.402385e3,
    5.038813e0,
    -5.799136e-2,
    3.287156e-4,
    -1.398845e-6,
    2.787860e-9,
)


def sound_speed_marczak(T_C: float | np.ndarray) -> float | np.ndarray:
    """Speed of sound in pure (fresh) water at atmospheric pressure, m/s.

    Marczak (1997) polynomial, valid 0-95 degC.  Reference values: 1482.4 m/s at
    20 degC, 1496.7 at 25, 1501.9 at 27, 1504.4 at 28 (RESEARCH_PLAN Appendix A).
    """
    T = np.asarray(T_C, dtype=np.float64)
    c = np.polynomial.polynomial.polyval(T, _MARCZAK)
    return float(c) if c.ndim == 0 else c


def salinity_correction(S_ppt: float | np.ndarray, T_C: float | np.ndarray = 25.0) -> float | np.ndarray:
    """Additive sound-speed correction for a small salinity S (g/kg), m/s.

    Linearization of the salinity terms of Mackenzie (1981, JASA 70, 807),
    dc/dS = 1.340 - 1.025e-2 * T  [m/s per g/kg], which is about +1.1 m/s per g/kg at
    25 degC and +1.3 at 5 degC.  Intended for salt-chlorinated pools (S ~ 2-6 g/kg)
    on top of the pure-water Marczak value; the Mackenzie fit itself is nominally for
    S = 25-40, so treat the result as a +-0.1 m/s-per-g/kg estimate (uncalibrated).
    """
    S = np.asarray(S_ppt, dtype=np.float64)
    T = np.asarray(T_C, dtype=np.float64)
    dc = (1.340 - 1.025e-2 * T) * S
    return float(dc) if dc.ndim == 0 else dc


def sound_speed(T_C: float | np.ndarray, S_ppt: float | np.ndarray = 0.0) -> float | np.ndarray:
    """Marczak c(T) plus :func:`salinity_correction` (m/s)."""
    return sound_speed_marczak(T_C) + salinity_correction(S_ppt, T_C)


def francois_garrison_alpha(
    f_Hz: float | np.ndarray,
    T_C: float = 25.0,
    S_ppt: float = 0.0,
    depth_m: float = 1.0,
    pH: float = 7.5,
) -> float | np.ndarray:
    """Francois & Garrison (1982) absorption coefficient, dB/km.

    alpha = A1 P1 f1 f^2/(f1^2 + f^2)   (boric acid)
          + A2 P2 f2 f^2/(f2^2 + f^2)   (magnesium sulphate)
          + A3 P3 f^2                    (pure water, viscous)
    with f in kHz.  JASA 72(3) 896-907 and 72(6) 1879-1890.  For fresh water (S = 0)
    the relaxation frequencies collapse (f1 = 0, A2 = 0) and only the pure-water term
    remains: about 0.012 dB/km at 8 kHz and 25 degC, i.e. < 0.001 dB over 50 m, which
    is why absorption is off in uwsim v0 (row 2).  pH only enters the boric term.
    """
    f = np.asarray(f_Hz, dtype=np.float64) / 1e3  # kHz
    T, S, D = float(T_C), float(S_ppt), float(depth_m)
    theta = T + 273.0
    c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * D  # F&G's own c, not Marczak

    # Boric acid
    A1 = 8.86 / c * 10.0 ** (0.78 * pH - 5.0)
    f1 = 2.8 * np.sqrt(S / 35.0) * 10.0 ** (4.0 - 1245.0 / theta)
    # Magnesium sulphate
    A2 = 21.44 * S / c * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * D + 6.2e-9 * D**2
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / theta) / (1.0 + 0.0018 * (S - 35.0))
    # Pure water
    if T <= 20.0:
        A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.50e-8 * T**3
    else:
        A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.5e-10 * T**3
    P3 = 1.0 - 3.83e-5 * D + 4.9e-10 * D**2

    f2sq = f**2
    with np.errstate(invalid="ignore", divide="ignore"):
        boric = np.where(f1 > 0, A1 * f1 * f2sq / (f1**2 + f2sq), 0.0)
    mgso4 = A2 * P2 * f2 * f2sq / (f2**2 + f2sq)
    water = A3 * P3 * f2sq
    a = boric + mgso4 + water
    return float(a) if a.ndim == 0 else a


def absorption_db(f_Hz: float | np.ndarray, range_m: float, **kw) -> float | np.ndarray:
    """Total absorption loss over ``range_m`` metres, dB (F&G, see above)."""
    return francois_garrison_alpha(f_Hz, **kw) * range_m / 1e3


def waveguide_cutoff(c: float | np.ndarray, H: float | np.ndarray) -> float | np.ndarray:
    """Lowest-mode cutoff c / (4 H) of a water layer of depth H, Hz.

    Pressure-release surface and rigid floor make the depth eigenfunction
    sin((2n+1) pi z / (2H)); the n = 0 mode is a quarter wave.  Below this frequency no
    propagating mode exists in a laterally unbounded layer (RESEARCH_PLAN 4.2 row 9).
    """
    fc = np.asarray(c, dtype=np.float64) / (4.0 * np.asarray(H, dtype=np.float64))
    return float(fc) if fc.ndim == 0 else fc


def minnaert_frequency(
    a_m: float | np.ndarray,
    depth_m: float = 0.0,
    gamma: float = 1.4,
    p_atm: float = 101325.0,
    rho: float = 998.0,
    g: float = 9.81,
) -> float | np.ndarray:
    """Minnaert resonance of a gas bubble of radius ``a_m`` (metres), Hz.

    f0 = (1 / (2 pi a)) sqrt(3 gamma p0 / rho),  p0 = p_atm + rho g depth.
    Adiabatic air (gamma = 1.4), no surface tension or thermal damping.  At the surface
    with rho = 998 kg/m^3 this is f0 = 3.29 / a  (a in m), e.g. 6.6 kHz at a = 0.5 mm and
    1.1 kHz at a = 3 mm (RESEARCH_PLAN 4.2 row 16).
    """
    a = np.asarray(a_m, dtype=np.float64)
    p0 = p_atm + rho * g * depth_m
    f0 = np.sqrt(3.0 * gamma * p0 / rho) / (2.0 * np.pi * a)
    return float(f0) if f0.ndim == 0 else f0


def octave_bands(f_lo: float = 250.0, f_hi: float = 8000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Octave-band centres f_lo * 2^k up to f_hi and their nominal edges.

    Returns ``(centres, lower, upper)`` with lower = centre / sqrt(2) and
    upper = centre * sqrt(2).  The default gives 250, 500, 1k, 2k, 4k, 8k Hz.
    """
    if f_lo <= 0 or f_hi < f_lo:
        raise ValueError("need 0 < f_lo <= f_hi")
    n = int(np.floor(np.log2(f_hi / f_lo) + 1e-9)) + 1
    centres = f_lo * 2.0 ** np.arange(n)
    return centres, centres / np.sqrt(2.0), centres * np.sqrt(2.0)
