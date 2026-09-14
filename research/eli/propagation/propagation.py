"""Frequency and phase response of water as a propagation medium.

Two figures, each a Bode-style pair (magnitude + phase) with fresh (pool)
water and seawater overlaid -- fresh water bold, seawater dotted:

    acoustic.*   sound in water
    em.*         electromagnetic waves in water

Everything plotted is the response of the *medium* only: a plane wave
travelling distance r has transfer function

    H(f) = exp(-alpha(f) * r) * exp(-j * beta(f) * r)

Geometric spreading is frequency-flat and therefore deliberately left out;
it would shift the magnitude curves down without changing their shape.

Models
------
Acoustic absorption: Francois & Garrison (1982), JASA 72(6) 1879 and
72(3) 896 -- boric acid relaxation, MgSO4 relaxation, and the viscous
absorption of pure water.  Setting S = 0 collapses both salt relaxations
to zero, which is exactly the fresh-water case.

Acoustic dispersion: each Francois-Garrison relaxation term is the
imaginary part of a Debye-relaxing sound speed,

    1/c(w)^2 = 1/c_inf^2 * (1 + sum_i eps_i / (1 + j*w/w_i))

so the same fitted (A_i, f_i) that give the absorption also fix the phase
velocity, with no extra free parameters.  The relaxation strengths eps_i
are O(1e-5), so this is evaluated in the linearised limit.

Electromagnetic: single-Debye water permittivity plus ionic conduction,

    eps_r(w) = eps_inf + (eps_s - eps_inf)/(1 + j*w*tau) - j*sigma/(w*eps_0)

which covers both the conduction-dominated low-frequency regime and the
~17 GHz dipole relaxation.

Run:  python propagation.py [--outdir figs]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

NP_PER_DB = 1.0 / 8.685889638065035  # nepers per decibel
EPS0 = 8.8541878128e-12              # F/m
MU0 = 4.0e-7 * np.pi                 # H/m
C0 = 299_792_458.0                   # m/s


# --------------------------------------------------------------------------
# acoustics
# --------------------------------------------------------------------------

@dataclass
class Water:
    """Water column conditions for the acoustic model."""
    name: str
    T: float        # temperature, deg C
    S: float        # salinity, practical salinity units (ppt)
    D: float        # depth, m
    pH: float
    range_m: float  # reference propagation range for the right-hand axis

    @property
    def c(self) -> float:
        """Sound speed used inside Francois-Garrison, m/s."""
        return 1412.0 + 3.21 * self.T + 1.19 * self.S + 0.0167 * self.D


def francois_garrison(f_khz: np.ndarray, w: Water):
    """Absorption in dB/km, split into its three physical terms.

    Returns (total, boric, mgso4, pure_water) plus the relaxation
    parameters needed for the dispersion calculation.
    """
    T, S, D, pH, c = w.T, w.S, w.D, w.pH, w.c

    # boric acid, B(OH)3 -- only relaxation below ~10 kHz
    A1 = 8.86 / c * 10.0 ** (0.78 * pH - 5.0)
    P1 = 1.0
    f1 = 2.8 * np.sqrt(S / 35.0) * 10.0 ** (4.0 - 1245.0 / (273.0 + T))

    # magnesium sulphate, MgSO4 -- relaxation near 100 kHz
    A2 = 21.44 * S / c * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * D + 6.2e-9 * D * D
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / (273.0 + T)) / (1.0 + 0.0018 * (S - 35.0))

    # viscous absorption of pure water
    if T <= 20.0:
        A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.50e-8 * T**3
    else:
        A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.50e-10 * T**3
    P3 = 1.0 - 3.83e-5 * D + 4.9e-10 * D * D

    # a zero relaxation frequency (S = 0 kills f1) must not divide by zero
    boric = np.zeros_like(f_khz) if f1 == 0.0 else \
        A1 * P1 * f1 * f_khz**2 / (f1**2 + f_khz**2)
    mgso4 = np.zeros_like(f_khz) if (A2 == 0.0 or f2 == 0.0) else \
        A2 * P2 * f2 * f_khz**2 / (f2**2 + f_khz**2)
    pure = A3 * P3 * f_khz**2

    relax = [("B(OH)$_3$", A1 * P1, f1), ("MgSO$_4$", A2 * P2, f2)]
    return boric + mgso4 + pure, boric, mgso4, pure, relax, (A3 * P3)


def acoustic_response(f_hz: np.ndarray, w: Water):
    """Attenuation (dB/m) and phase constant (rad/m) for sound in water."""
    f_khz = f_hz / 1e3
    total_db_km, boric, mgso4, pure, relax, A3P3 = francois_garrison(f_khz, w)
    alpha_db_m = total_db_km / 1e3

    # Relaxation strengths from the same fitted coefficients.  The Debye
    # model gives alpha_i -> eps_i * pi * f_i / c_inf as f >> f_i, while
    # Francois-Garrison gives A_i * f_i dB/km; equating the two asymptotes
    # fixes eps_i with no new parameters.
    eps = []
    for _label, A, f_r_khz in relax:
        if A == 0.0 or f_r_khz == 0.0:
            eps.append((0.0, f_r_khz * 1e3))
            continue
        alpha_inf_np_m = (A * f_r_khz / 1e3) * NP_PER_DB   # Np/m
        eps.append((alpha_inf_np_m * w.c / (np.pi * f_r_khz * 1e3), f_r_khz * 1e3))

    # The pure-water f^2 term is the low-frequency tail of water's structural
    # relaxation near 1 GHz.  Folding it in with that assumed f_r shows it is
    # irrelevant in band (its dispersion scales as (f/f_r)^2).
    F_STRUCT = 1e9
    alpha_struct_inf = (A3P3 * (F_STRUCT / 1e3) / 1e3) * NP_PER_DB
    eps.append((alpha_struct_inf * w.c / (np.pi * F_STRUCT), F_STRUCT))

    # Re k = (w/c_inf) * [1 + sum eps_i/2 / (1 + x_i^2)],  x_i = f/f_i
    omega = 2.0 * np.pi * f_hz
    corr = np.zeros_like(f_hz)
    corr_dc = 0.0
    for e_i, f_r in eps:
        if e_i == 0.0:
            continue
        x = f_hz / f_r
        corr += 0.5 * e_i / (1.0 + x**2)
        corr_dc += 0.5 * e_i
    c_inf = w.c
    beta = omega / c_inf * (1.0 + corr)
    v_ph = omega / beta
    c_zero = c_inf / (1.0 + corr_dc)          # f -> 0 limit
    ppm = (v_ph / c_zero - 1.0) * 1e6
    return alpha_db_m, beta, v_ph, ppm, dict(
        boric=boric / 1e3, mgso4=mgso4 / 1e3, pure=pure / 1e3,
        relax=relax, c_zero=c_zero)


# --------------------------------------------------------------------------
# electromagnetics
# --------------------------------------------------------------------------

@dataclass
class EMWater:
    """Single-Debye water plus ionic conduction."""
    name: str
    eps_s: float     # static relative permittivity
    eps_inf: float   # high-frequency relative permittivity
    tau: float       # Debye relaxation time, s
    sigma: float     # ionic conductivity, S/m
    T: float
    S: float
    range_m: float
    sigma_note: str = ""
    marks: list = field(default_factory=list)
    bands: list = field(default_factory=list)

    @property
    def f_debye(self) -> float:
        return 1.0 / (2.0 * np.pi * self.tau)


def em_response(f_hz: np.ndarray, w: EMWater):
    """Attenuation (dB/m) and phase constant (rad/m) for EM waves in water."""
    omega = 2.0 * np.pi * f_hz
    # exp(+j w t) convention -> lossy permittivity has a negative imaginary part
    eps_r = w.eps_inf + (w.eps_s - w.eps_inf) / (1.0 + 1j * omega * w.tau) \
        - 1j * w.sigma / (omega * EPS0)
    k = omega / C0 * np.sqrt(eps_r)          # principal branch: Re k > 0, Im k < 0
    beta = np.real(k)
    alpha_np_m = -np.imag(k)
    alpha_db_m = alpha_np_m / NP_PER_DB
    v_ph = omega / beta
    skin_depth = 1.0 / alpha_np_m
    return alpha_db_m, beta, v_ph, skin_depth, eps_r


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.9,
})

SIG = "#1f77b4"
ACC = "#d62728"
AUX = "#7f7f7f"
SPEECH = "#2ca02c"
SONAR = "#ff7f0e"
SEA_C = "#101010"

# fresh water is the case we actually operate in, so it gets the bold line
FRESH_KW = dict(color=SIG, ls="-", lw=2.4)
SEA_KW = dict(color=SEA_C, ls=":", lw=1.9)
FRESH_R = dict(color=ACC, ls="-", lw=2.0)
SEA_R = dict(color=ACC, ls=":", lw=1.6)


def _eng(f: float) -> str:
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz")):
        if f >= scale:
            v = f / scale
            return f"{v:.0f} {unit}" if v >= 10 else f"{v:g} {unit}"
    return f"{f:g} Hz"


def _band(ax, lo, hi, label, color=ACC, row=0):
    ax.axvspan(lo, hi, color=color, alpha=0.13, lw=0, zorder=0)
    if not label:
        return
    x0, x1 = ax.get_xlim()
    frac = np.log10(np.sqrt(lo * hi) / x0) / np.log10(x1 / x0)
    x, ha = (hi, "right") if frac > 0.85 else (np.sqrt(lo * hi), "center")
    ax.text(x, 0.975 - 0.055 * row, label, transform=ax.get_xaxis_transform(),
            ha=ha, va="top", fontsize=7, color=color, weight="bold")


def _line(ax, f, label, color=AUX):
    ax.axvline(f, color=color, ls=":", lw=0.9, zorder=0)
    ax.text(f, 0.02, " " + label, transform=ax.get_xaxis_transform(),
            rotation=90, ha="left", va="bottom", fontsize=6.5, color=color)


SUBS = tuple(np.arange(2, 10) * 0.1)


def _decade_ticks(ax):
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=SUBS, numticks=100))


def _graticule(ax, log_y=True):
    """Major and minor decade rules on every log axis."""
    if log_y:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=SUBS, numticks=100))
    ax.grid(True, which="major", alpha=0.38, lw=0.7)
    ax.grid(True, which="minor", alpha=0.15, lw=0.5)
    ax.tick_params(which="minor", length=2.5)


# --------------------------------------------------------------------------
# figures: both media overlaid on one pair of axes
# --------------------------------------------------------------------------

def figure_acoustic(fresh: Water, sea: Water, marks, range_m=25.0):
    """Fresh water and seawater on one figure.

    A single reference range is used for the right-hand axis, since it is just
    a rescaling of the left axis and so applies to both curves at once.
    """
    f = np.logspace(np.log10(20.0), 6, 2400)
    a_f, b_f, _, ppm_f, ex_f = acoustic_response(f, fresh)
    a_s, b_s, _, ppm_s, ex_s = acoustic_response(f, sea)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=False,
                                   gridspec_kw=dict(hspace=0.34))

    # ---- magnitude ----
    ax1.loglog(f, a_f, label="fresh water (pool), S = 0", **FRESH_KW)
    ax1.loglog(f, a_s, label="seawater, S = 35 psu", **SEA_KW)
    # the gap between the two curves is exactly seawater's salt relaxations
    ax1.loglog(f, ex_s["boric"], ls="--", lw=1.0, color=AUX,
               label="sea: boric acid relaxation")
    ax1.loglog(f, ex_s["mgso4"], ls="-.", lw=1.0, color=AUX,
               label="sea: MgSO$_4$ relaxation")
    ax1.set_ylabel("absorption  [dB / m]\n(more loss downward $\\downarrow$)")
    ax1.set_xlabel("frequency  [Hz]")
    ax1.set_ylim(1e-11, 1e1)
    ax1.invert_yaxis()

    axr = ax1.twinx()
    axr.set_yscale("log")
    axr.set_ylim(*[v * range_m for v in ax1.get_ylim()])
    axr.set_ylabel(f"loss over {range_m:g} m  [dB]", color=ACC)
    axr.tick_params(axis="y", colors=ACC, labelsize=8)
    axr.grid(False)
    axr.axhline(3.0, color=ACC, ls="--", lw=0.9)
    axr.text(10.0 ** (np.log10(f[0]) + 0.40 * np.log10(f[-1] / f[0])), 3.0,
             "3 dB", color=ACC, fontsize=7, ha="center", va="top")

    ax1.set_title("Acoustic propagation in water — fresh vs sea\n"
                  f"T = {fresh.T:g} °C;  fresh pH {fresh.pH:g}, {fresh.D:g} m, "
                  f"$c_0$ = {ex_f['c_zero']:.0f} m/s;  sea pH {sea.pH:g}, "
                  f"{sea.D:g} m, $c_0$ = {ex_s['c_zero']:.0f} m/s")
    ax1.legend(loc="upper right", bbox_to_anchor=(0.995, 0.85))

    # ---- phase ----
    ax2.loglog(f, np.degrees(b_f), label=r"$\beta$, fresh", **FRESH_KW)
    ax2.loglog(f, np.degrees(b_s), label=r"$\beta$, sea", **SEA_KW)
    ax2.set_ylabel("phase shift  [deg / m]")
    ax2.set_xlabel("frequency  [Hz]")

    ax2b = ax2.twinx()
    ax2b.set_xscale("log")
    ax2b.plot(f, ppm_f, label=r"$\Delta c/c_0$, fresh", **FRESH_R)
    ax2b.plot(f, ppm_s, label=r"$\Delta c/c_0$, sea", **SEA_R)
    ax2b.set_ylabel(r"phase-velocity dispersion  $\Delta c/c_0$  [ppm]", color=ACC)
    ax2b.tick_params(axis="y", colors=ACC, labelsize=8)
    ax2b.grid(False)
    ax2b.set_ylim(-1.5, 28.0)

    for ax in (ax1, ax2):
        _decade_ticks(ax)
        _graticule(ax)
        ax.set_xlim(f[0], f[-1])
    for row, (lo, hi, lab, col) in enumerate(marks):
        _band(ax1, lo, hi, lab, col, row)
        _band(ax2, lo, hi, "", col)

    i30 = int(np.argmin(np.abs(f - 30e3)))
    note = (f"at 30 kHz, over {range_m:g} m:\n"
            f"  fresh  {a_f[i30]*range_m:8.2e} dB,  "
            f"$\\Delta c/c_0$ = {ppm_f[i30]:.2f} ppm\n"
            f"  sea    {a_s[i30]*range_m:8.2e} dB,  "
            f"$\\Delta c/c_0$ = {ppm_s[i30]:.2f} ppm")
    ax2.text(0.015, 0.97, note, transform=ax2.transAxes, fontsize=7.2,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=AUX, lw=0.6, alpha=0.92))

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right",
               bbox_to_anchor=(0.995, 0.09), ncol=2)
    return fig


def figure_em(fresh: EMWater, sea: EMWater):
    """Fresh water and seawater on one figure.

    Skin depth on the right axis is 1/alpha, a pure function of the left axis,
    so the one scale serves both curves.
    """
    f = np.logspace(1, 11, 2600)
    a_f, b_f, v_f, d_f, _ = em_response(f, fresh)
    a_s, b_s, v_s, d_s, _ = em_response(f, sea)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=False,
                                   gridspec_kw=dict(hspace=0.34))

    # ---- magnitude ----
    ax1.loglog(f, a_f, label=f"fresh water (pool), $\\sigma$ = {fresh.sigma:g} S/m",
               **FRESH_KW)
    ax1.loglog(f, a_s, label=f"seawater, $\\sigma$ = {sea.sigma:g} S/m", **SEA_KW)
    ax1.set_ylabel("attenuation  [dB / m]\n(more loss downward $\\downarrow$)")
    ax1.set_xlabel("frequency  [Hz]")
    ax1.set_ylim(1e-3, 1e6)
    ax1.invert_yaxis()

    axr = ax1.twinx()
    axr.set_yscale("log")
    lo, hi = ax1.get_ylim()
    axr.set_ylim(1.0 / (lo * NP_PER_DB), 1.0 / (hi * NP_PER_DB))
    axr.set_ylabel("skin depth  $\\delta = 1/\\alpha$  [m]", color=ACC)
    axr.tick_params(axis="y", colors=ACC, labelsize=8)
    axr.grid(False)

    ax1.set_title("Electromagnetic propagation in water — fresh vs sea\n"
                  f"T = {fresh.T:g} °C;  fresh $\\varepsilon_s$ = {fresh.eps_s:g}, "
                  f"$f_D$ = {fresh.f_debye/1e9:.1f} GHz;  "
                  f"sea $\\varepsilon_s$ = {sea.eps_s:g}, "
                  f"$f_D$ = {sea.f_debye/1e9:.1f} GHz")
    ax1.legend(loc="lower left")

    # ---- phase ----
    ax2.loglog(f, np.degrees(b_f), label=r"$\beta$, fresh", **FRESH_KW)
    ax2.loglog(f, np.degrees(b_s), label=r"$\beta$, sea", **SEA_KW)
    ax2.set_ylabel("phase shift  [deg / m]")
    ax2.set_xlabel("frequency  [Hz]")
    ax2.set_ylim(1e-2, 1e6)

    ax2b = ax2.twinx()
    ax2b.loglog(f, v_f, label=r"$v_p$, fresh", **FRESH_R)
    ax2b.loglog(f, v_s, label=r"$v_p$, sea", **SEA_R)
    ax2b.set_ylabel("phase velocity  [m/s]", color=ACC)
    ax2b.tick_params(axis="y", colors=ACC, labelsize=8)
    ax2b.grid(False)
    ax2b.set_ylim(1e4, 1e9)
    ax2b.set_xlim(f[0], f[-1])

    x_mid = 10.0 ** (np.log10(f[0]) + 0.52 * np.log10(f[-1] / f[0]))
    ax2b.axhline(C0, color=ACC, ls=":", lw=0.9)
    ax2b.text(x_mid, C0, "c (vacuum)", color=ACC, fontsize=7,
              va="bottom", ha="center")
    v_lo = C0 / np.sqrt(max(fresh.eps_s, sea.eps_s))
    v_hi = C0 / np.sqrt(min(fresh.eps_s, sea.eps_s))
    ax2b.axhline(v_hi, color=ACC, ls=":", lw=0.9)
    ax2b.text(x_mid, v_hi,
              f"c/$\\sqrt{{\\varepsilon_s}}$ = {v_lo/1e6:.0f}–{v_hi/1e6:.0f} Mm/s",
              color=ACC, fontsize=7, va="bottom", ha="center")

    for ax in (ax1, ax2):
        _decade_ticks(ax)
        _graticule(ax)
        ax.set_xlim(f[0], f[-1])
    for row, (lo_b, hi_b, lab, col) in enumerate(fresh.bands):
        _band(ax1, lo_b, hi_b, lab, col, row)
        _band(ax2, lo_b, hi_b, "", col)
    for fx, lab in fresh.marks:
        _line(ax1, fx, lab)
        _line(ax2, fx, lab)
    f_d = 0.5 * (fresh.f_debye + sea.f_debye)
    for ax in (ax1, ax2):
        _line(ax, f_d, f"Debye $f_D$ $\\approx$ {f_d/1e9:.0f} GHz", color=ACC)

    rows = ["            fresh          sea"]
    for fx in (10e3, 1e6, 2.4e9):
        i = int(np.argmin(np.abs(f - fx)))
        rows.append(f"{_eng(fx):>8}  {d_f[i]:8.3g} m  {d_s[i]:8.3g} m")
    ax2.text(0.015, 0.97, "skin depth\n" + "\n".join(rows),
             transform=ax2.transAxes, fontsize=7.2, va="top", ha="left",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=AUX, lw=0.6, alpha=0.92))

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right",
               bbox_to_anchor=(0.995, 0.03), ncol=2)
    return fig


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(Path(__file__).parent / "figs"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 25 m competition pool; ocean reference range 1 km
    fresh = Water("fresh water (pool)", T=20.0, S=0.0, D=3.0, pH=7.4, range_m=25.0)
    sea = Water("seawater", T=20.0, S=35.0, D=10.0, pH=8.0, range_m=1000.0)

    # 85 Hz .. 8 kHz spans voiced fundamentals through fricative energy
    SPEECH_BAND = (85.0, 8e3, "human speech", SPEECH)
    acoustic_marks = [SPEECH_BAND,
                      (25e3, 40e3, "RoboSub pinger", ACC),
                      (675e3, 825e3, "Ping360", SONAR)]

    em_fresh = EMWater("fresh water (pool)", eps_s=80.2, eps_inf=5.5, tau=9.4e-12,
                       sigma=0.05, T=20.0, S=0.0, range_m=1.0,
                       sigma_note=" (pool, 0.02–0.1 typical)",
                       marks=[(25e3, "VLF"), (2.4e9, "Wi-Fi 2.4 G")],
                       bands=[(85.0, 8e3, "speech band", SPEECH)])
    em_sea = EMWater("seawater", eps_s=73.0, eps_inf=5.5, tau=9.0e-12,
                     sigma=4.8, T=20.0, S=35.0, range_m=1.0,
                     sigma_note="", marks=[(25e3, "VLF"), (2.4e9, "Wi-Fi 2.4 G")],
                     bands=[(85.0, 8e3, "speech band", SPEECH)])

    jobs = [
        ("acoustic", lambda: figure_acoustic(fresh, sea, acoustic_marks)),
        ("em", lambda: figure_em(em_fresh, em_sea)),
    ]
    for name, build in jobs:
        fig = build()
        fig.subplots_adjust(left=0.11, right=0.88, top=0.90, bottom=0.08)
        for ext in ("png", "pdf"):
            p = outdir / f"{name}.{ext}"
            fig.savefig(p, bbox_inches="tight")
            print("wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
