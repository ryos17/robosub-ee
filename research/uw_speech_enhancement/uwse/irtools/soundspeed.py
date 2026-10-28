"""Sound speed from the time-of-flight vs range slope, cross-checked with Marczak c(T).

Implements RESEARCH_PLAN §4.3 step 3 (and §4.2 row 1, §6.3, §6.9, Appendix A):

* fit TOF = r / c + tau0 by ordinary least squares over >= 4 tape ranges
  (1, 3, 6, 10 m); the slope gives c, the intercept tau0 absorbs electronic,
  transducer and group delay (TOF is referenced to the loopback);
* report c +- SE(c) next to Marczak c(T) and test agreement within 2 SE.

Uncertainty model for the plan's "+-3 cm -> rel. SE(c) ~ 0.44 %": the range
error is treated as a standard deviation sigma_r (not a half-width). An error
dr in a tape range is equivalent to a TOF error dr / c, so with known
sigma_r and OLS on the slope a = 1/c,

    SE(a) = (sigma_r / c) / sqrt(Sxx),  Sxx = sum (r_i - mean r)^2,
    SE(c) / c = SE(a) / a = sigma_r / sqrt(Sxx).

For {1, 3, 6, 10} m, Sxx = 46 m^2: 0.03 / sqrt(46) = 0.442 % (reproduces the
plan) and 0.05 / sqrt(46) = 0.737 % ("0.74 %"). If "+-3 cm" were instead the
half-width of a uniform error, sigma_r = 0.03 / sqrt(3) and SE(c)/c = 0.255 %.
The fit itself estimates SE from the residuals (n - 2 = 2 dof for 4 ranges),
which is unbiased in variance but itself very uncertain; pass ``range_sd_m``
to also get the known-variance SE.

Marczak (1997) pure-water equation, valid 0-95 degC, as quoted by NPL
("Speed of sound in pure water"): checked against Appendix A
(1482.4, 1496.7, 1501.9, 1504.4 m/s at 20, 25, 27, 28 degC). A local copy is
kept here on purpose (``uwse.uwsim.physics`` is maintained separately).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "marczak_c",
    "marczak_dcdT",
    "SoundSpeedFit",
    "fit_tof",
    "predicted_rel_se_c",
    "uniform_halfwidth_to_sd",
    "MarczakComparison",
    "compare_marczak",
    "monte_carlo_tof_fit",
]

# Marczak (1997) coefficients, c in m/s, T in degC (ITS-90)
_MARCZAK = (1.402385e3, 5.038813, -5.799136e-2, 3.287156e-4, -1.398845e-6, 2.787860e-9)


def marczak_c(T_c):
    """Speed of sound in pure water (m/s), Marczak (1997), 0-95 degC."""
    T = np.asarray(T_c, dtype=np.float64)
    c = np.zeros_like(T)
    for i, a in enumerate(_MARCZAK):
        c = c + a * T**i
    return c if c.ndim else float(c)


def marczak_dcdT(T_c):
    """dc/dT of the Marczak equation (m/s per degC); about 2.6 at 26 degC."""
    T = np.asarray(T_c, dtype=np.float64)
    d = np.zeros_like(T)
    for i, a in enumerate(_MARCZAK[1:], start=1):
        d = d + i * a * T ** (i - 1)
    return d if d.ndim else float(d)


@dataclass
class SoundSpeedFit:
    """OLS fit of TOF = r / c + tau0.

    se_c, se_tau0 : from the residual variance (dof = n - 2).
    se_c_known, se_tau0_known : from the stated range / TOF SDs (if given).
    residuals_s : TOF residuals (s).
    """

    c: float
    se_c: float
    tau0: float
    se_tau0: float
    slope: float
    se_slope: float
    residuals_s: np.ndarray
    dof: int
    sigma_t: float
    se_c_known: float | None = None
    se_tau0_known: float | None = None


def fit_tof(
    ranges_m,
    tof_s,
    range_sd_m: float | None = None,
    tof_sd_s: float = 0.0,
) -> SoundSpeedFit:
    """Least-squares fit of TOF = r / c + tau0 (slope a = 1/c, intercept tau0).

    SE(c) = SE(a) / a^2 (delta method). The plan requires >= 4 ranges; at least
    3 are needed for a residual-based SE.
    """
    r = np.asarray(ranges_m, dtype=np.float64)
    t = np.asarray(tof_s, dtype=np.float64)
    n = len(r)
    if n < 3 or len(t) != n:
        raise ValueError("need >= 3 (plan: >= 4) matching ranges and TOFs")
    rm = r.mean()
    Sxx = np.sum((r - rm) ** 2)
    a = np.sum((r - rm) * (t - t.mean())) / Sxx
    tau0 = t.mean() - a * rm
    res = t - (a * r + tau0)
    dof = n - 2
    s = np.sqrt(np.sum(res**2) / dof)
    se_a = s / np.sqrt(Sxx)
    se_tau = s * np.sqrt(1 / n + rm**2 / Sxx)
    c = 1.0 / a
    out = SoundSpeedFit(float(c), float(se_a / a**2), float(tau0), float(se_tau), float(a), float(se_a), res, dof, float(s))
    if range_sd_m is not None:
        sk = np.sqrt((range_sd_m * a) ** 2 + tof_sd_s**2)
        out.se_c_known = float(sk / np.sqrt(Sxx) / a**2)
        out.se_tau0_known = float(sk * np.sqrt(1 / n + rm**2 / Sxx))
    return out


def predicted_rel_se_c(ranges_m, range_sd_m: float, tof_sd_s: float = 0.0, c: float = 1500.0) -> float:
    """Known-variance relative SE of c: sqrt(sigma_r^2 + (c sigma_t)^2) / sqrt(Sxx)."""
    r = np.asarray(ranges_m, dtype=np.float64)
    Sxx = np.sum((r - r.mean()) ** 2)
    return float(np.sqrt(range_sd_m**2 + (c * tof_sd_s) ** 2) / np.sqrt(Sxx))


def uniform_halfwidth_to_sd(half_width: float) -> float:
    """SD of a uniform error on [-w, w]: w / sqrt(3)."""
    return half_width / np.sqrt(3.0)


@dataclass
class MarczakComparison:
    c_fit: float
    se_c: float
    c_marczak: float
    diff: float
    se_combined: float
    z: float
    agree: bool


def compare_marczak(
    fit: SoundSpeedFit,
    temperature_c: float,
    temperature_sd_c: float = 0.0,
    k: float = 2.0,
    use_known_se: bool = False,
) -> MarczakComparison:
    """Agreement test |c_fit - c_Marczak(T)| <= k SE (§4.3 step 3, k = 2).

    SE combines the fit SE with the temperature uncertainty propagated through
    dc/dT (+-0.2 degC -> +-0.5 m/s at 26 degC, §6.9). Pure water is assumed;
    salinity from chlorination is ignored.
    """
    se_fit = fit.se_c_known if (use_known_se and fit.se_c_known is not None) else fit.se_c
    cm = marczak_c(temperature_c)
    se = float(np.hypot(se_fit, marczak_dcdT(temperature_c) * temperature_sd_c))
    d = fit.c - cm
    z = d / se if se > 0 else np.inf
    return MarczakComparison(fit.c, se_fit, cm, d, se, float(z), bool(abs(d) <= k * se))


def monte_carlo_tof_fit(
    ranges_m,
    c: float = 1500.0,
    tau0: float = 1e-3,
    range_sd_m: float = 0.03,
    tof_sd_s: float = 0.0,
    n_trials: int = 20000,
    rng: np.random.Generator | None = None,
    uniform: bool = False,
) -> dict[str, float]:
    """Monte Carlo of ``fit_tof`` with Gaussian (or uniform +-range_sd_m if
    ``uniform``) tape-range errors. Vectorized; returns mean / SD of c_hat,
    relative SD, and the mean residual-based relative SE."""
    rng = np.random.default_rng() if rng is None else rng
    r = np.asarray(ranges_m, dtype=np.float64)
    if uniform:
        dr = rng.uniform(-range_sd_m, range_sd_m, (n_trials, len(r)))
    else:
        dr = rng.normal(0.0, range_sd_m, (n_trials, len(r)))
    # true TOF at the true range; the analyst only knows the taped (erroneous) range
    t = r / c + tau0 + rng.normal(0.0, tof_sd_s, (n_trials, len(r))) if tof_sd_s > 0 else np.broadcast_to(r / c + tau0, dr.shape)
    x = r + dr
    xm = x.mean(1, keepdims=True)
    Sxx = np.sum((x - xm) ** 2, 1)
    a = np.sum((x - xm) * (t - t.mean(1, keepdims=True)), 1) / Sxx
    b = t.mean(1) - a * xm[:, 0]
    res = t - (a[:, None] * x + b[:, None])
    s = np.sqrt(np.sum(res**2, 1) / (len(r) - 2))
    ch = 1 / a
    return {
        "mean_c": float(ch.mean()),
        "sd_c": float(ch.std(ddof=1)),
        "rel_sd_c": float(ch.std(ddof=1) / c),
        "mean_rel_se_resid": float(np.mean(s / np.sqrt(Sxx) / a)),
        "mean_tau0": float(b.mean()),
    }
