"""Transducer and receive chain of the pool simulator.

Implements RESEARCH_PLAN.md section 4.1 (the non-channel stages of the signal
chain) and section 4.2 rows 10-12, 19 and 20:

    drive LPF (8 kHz) -> projector (Hammerstein, k <= 3) -> directivity G_b(theta)
    ... channel ... -> hydrophone FIR -> preamp HPF -> gain -> ADC

Every quantity that must later come from pool measurements (chain FIR, harmonic
IRs, directivity table, hydrophone FIR, preamp corner, codec floor) is a
parameter with an *uncalibrated* default; objects that carry such data expose a
``calibrated`` flag so downstream code and logs can tell the two apart.

Filter convention. Causal filters model real analog/digital stages (drive LPF,
measured FIRs, preamp HPF). The randomization equalizers (1/3-octave projector
ripple, hydrophone tilt) and the directivity filter are linear-phase FIRs applied
*centred* (zero added delay), so they shift no arrival times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.signal import butter, fftconvolve, firwin2, resample, sosfilt, sosfiltfilt
from scipy.special import j1

# Octave centres used for the directivity table and the band-wise equalizers.
OCTAVE_CENTRES = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])

# LL916H radius: 9.25" (234.95 mm) diameter per Lubell -> a = 0.1175 m (section 4.2 row 12).
LL916_RADIUS_M = 0.117


# --------------------------------------------------------------------------- utilities


def apply_fir(x: np.ndarray, h: np.ndarray, centred: bool = False) -> np.ndarray:
    """Convolve ``x`` with FIR ``h`` along the last axis, output length = len(x).

    ``centred=True`` removes the (N-1)/2 group delay of a linear-phase FIR.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size == 1:
        return x * h[0]
    y = fftconvolve(x, h, mode="full", axes=-1)
    off = (h.size - 1) // 2 if centred else 0
    return y[..., off: off + x.shape[-1]]


def eq_fir(freqs_hz: Sequence[float], gains_db: Sequence[float], fs: int,
           numtaps: int = 513) -> np.ndarray:
    """Linear-phase FIR whose magnitude interpolates ``gains_db`` at ``freqs_hz``.

    Interpolation is linear in dB over log-frequency; gains are held flat below the
    first and above the last point.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    g = np.asarray(gains_db, dtype=np.float64)
    order = np.argsort(f)
    f, g = f[order], g[order]
    grid = np.linspace(0.0, fs / 2.0, 1025)
    lg = np.interp(np.log2(np.maximum(grid, 1.0)), np.log2(f), g)
    if numtaps % 2 == 0:
        numtaps += 1
    return firwin2(numtaps, grid, 10.0 ** (lg / 20.0), fs=fs)


def third_octave_centres(f_lo: float = 100.0, f_hi: float = 10000.0) -> np.ndarray:
    """Base-2 1/3-octave centres re 1 kHz within [f_lo, f_hi]."""
    n = np.arange(np.floor(3 * np.log2(f_lo / 1000.0)), np.ceil(3 * np.log2(f_hi / 1000.0)) + 1)
    return 1000.0 * 2.0 ** (n / 3.0)


def random_third_octave_gains(rng: np.random.Generator, max_db: float,
                              f_lo: float = 100.0, f_hi: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """Random 1/3-octave gains U(-max_db, max_db) (section 4.2 row 10 randomization)."""
    fc = third_octave_centres(f_lo, f_hi)
    return fc, rng.uniform(-max_db, max_db, size=fc.size)


def tilt_gains(tilt_db: float, f_ref: float = 1000.0,
               f_lo: float = 125.0, f_hi: float = 8000.0) -> tuple[np.ndarray, np.ndarray]:
    """Spectral tilt that is 0 dB at ``f_ref`` and reaches +/-``tilt_db`` at the band edges.

    Linear in log-frequency; used for the hydrophone +/-2 dB randomization (row 19).
    """
    fc = OCTAVE_CENTRES[(OCTAVE_CENTRES >= f_lo) & (OCTAVE_CENTRES <= f_hi)]
    half_span = np.log2(f_hi / f_lo) / 2.0
    return fc, tilt_db * np.log2(fc / f_ref) / half_span


def warp_fir(h: np.ndarray, factor: float) -> np.ndarray:
    """Shift every resonance of FIR ``h`` by ``factor`` (f -> factor*f).

    Done by time-scaling the impulse response by 1/factor (band-limited
    interpolation); used for the +/-20 % resonance-shift randomization (row 10).
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size <= 1 or factor == 1.0:
        return h.copy()
    # Zero-pad before the FFT resample so the (decaying) IR does not wrap around.
    hp = np.concatenate([h, np.zeros(h.size)])
    n_out = max(1, int(round(h.size / factor)))
    y = resample(hp, max(2, int(round(hp.size / factor))))
    return y[:n_out] * factor


# --------------------------------------------------------------------------- drive LPF


def drive_lpf(x: np.ndarray, fs: int, fc: float = 8000.0, order: int = 8,
              zero_phase: bool = False) -> np.ndarray:
    """Butterworth anti-imaging low-pass on the drive signal (section 4.1, first stage).

    Mirrors the 8 kHz LPF of the real playback chain. Causal by default, as in hardware.
    """
    if fc >= fs / 2:
        return np.array(x, dtype=np.float64, copy=True)
    sos = butter(order, fc, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1) if zero_phase else sosfilt(sos, x, axis=-1)


# --------------------------------------------------------------------------- projector


def hammerstein(x: np.ndarray, kernels: Sequence[np.ndarray | None]) -> np.ndarray:
    """Hammerstein model y = sum_k h_k * (x^k), k = 1..len(kernels) (section 4.2 row 11).

    ``kernels[k-1]`` is the FIR for the k-th power branch; ``None`` disables a branch.
    Output length equals len(x).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    for k, h in enumerate(kernels, start=1):
        if h is None:
            continue
        h = np.asarray(h, dtype=np.float64)
        if not np.any(h):
            continue
        y += apply_fir(x ** k, h)
    return y


@dataclass
class Projector:
    """Projector linear FIR plus 2nd/3rd harmonic Hammerstein kernels.

    Defaults are *uncalibrated*: h1 = unit impulse (flat chain), h2 = h3 = 0 (linear).
    The measured h1 is the chain FIR h_chain from E2; h2, h3 come from ESS harmonic IRs.
    """

    h1: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    h2: np.ndarray | None = None
    h3: np.ndarray | None = None
    calibrated: bool = False

    def kernels(self, resonance_factor: float = 1.0) -> list[np.ndarray | None]:
        ks = [self.h1, self.h2, self.h3]
        return [None if h is None else warp_fir(h, resonance_factor) for h in ks]

    def apply(self, x: np.ndarray, resonance_factor: float = 1.0) -> np.ndarray:
        """Full nonlinear projector output."""
        return hammerstein(x, self.kernels(resonance_factor))

    def apply_linear(self, x: np.ndarray, resonance_factor: float = 1.0) -> np.ndarray:
        """Linear part only (h1); this is what enters h_chain and the R1 target (section 8.2)."""
        return apply_fir(np.asarray(x, dtype=np.float64), warp_fir(self.h1, resonance_factor))


# --------------------------------------------------------------------------- directivity


def piston_directivity(theta_rad: np.ndarray | float, f_hz: np.ndarray | float,
                       a: float = LL916_RADIUS_M, c: float = 1500.0) -> np.ndarray:
    """Baffled circular piston far-field amplitude D = 2 J1(ka sin t) / (ka sin t).

    Textbook model used as the *uncalibrated* stand-in for the measured LL916 table.
    For |theta| > 90 deg the 90 deg value is used (no rear-radiation model).
    """
    th = np.minimum(np.abs(np.asarray(theta_rad, dtype=np.float64)), np.pi / 2)
    ka_s = 2 * np.pi * np.asarray(f_hz, dtype=np.float64) / c * a * np.sin(th)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(np.abs(ka_s) < 1e-8, 1.0, 2.0 * j1(ka_s) / np.where(ka_s == 0, 1.0, ka_s))
    return np.abs(d)


@dataclass
class Directivity:
    """Per-octave directivity gains at measured angles, interpolated in angle (row 12).

    ``gains_db[i, b]`` is the gain at ``angles_deg[i]`` in octave ``bands_hz[b]``,
    re on-axis. Angles beyond the last measured angle are clamped to it.
    """

    angles_deg: np.ndarray
    bands_hz: np.ndarray
    gains_db: np.ndarray
    calibrated: bool = False

    @classmethod
    def piston(cls, a: float = LL916_RADIUS_M, c: float = 1500.0,
               angles_deg: Sequence[float] = (0.0, 45.0, 90.0),
               bands_hz: Sequence[float] = tuple(OCTAVE_CENTRES),
               floor_db: float = -30.0) -> "Directivity":
        """Uncalibrated default: piston model sampled at the measurement angles."""
        ang = np.asarray(angles_deg, dtype=np.float64)
        bands = np.asarray(bands_hz, dtype=np.float64)
        d = piston_directivity(np.deg2rad(ang)[:, None], bands[None, :], a=a, c=c)
        g = np.maximum(20 * np.log10(np.maximum(d, 1e-12)), floor_db)
        return cls(ang, bands, g, calibrated=False)

    def gain_db(self, theta_deg: float) -> np.ndarray:
        """Interpolated per-band gain (dB) at off-axis angle ``theta_deg``."""
        th = float(np.clip(abs(theta_deg) % 360.0, 0.0, 360.0))
        if th > 180.0:
            th = 360.0 - th
        th = min(th, float(self.angles_deg[-1]))
        return np.array([np.interp(th, self.angles_deg, self.gains_db[:, b])
                         for b in range(self.bands_hz.size)])

    def fir(self, theta_deg: float, fs: int, numtaps: int = 257) -> np.ndarray:
        """Linear-phase FIR realizing the per-octave gains at ``theta_deg`` (apply centred)."""
        return eq_fir(self.bands_hz, self.gain_db(theta_deg), fs, numtaps)

    def apply(self, x: np.ndarray, theta_deg: float, fs: int) -> np.ndarray:
        return apply_fir(x, self.fir(theta_deg, fs), centred=True)


# --------------------------------------------------------------------------- receive side


@dataclass
class Hydrophone:
    """Per-channel hydrophone FIR (row 19). Default: unit impulse, uncalibrated."""

    fir: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    calibrated: bool = False

    def apply(self, x: np.ndarray) -> np.ndarray:
        return apply_fir(np.asarray(x, dtype=np.float64), self.fir)


def preamp_hpf(x: np.ndarray, fs: int, fc: float = 100.0, order: int = 2) -> np.ndarray:
    """Causal Butterworth high-pass of the preamp. The corner is uncalibrated (100 Hz)."""
    if fc <= 0:
        return np.array(x, dtype=np.float64, copy=True)
    sos = butter(order, fc, btype="high", fs=fs, output="sos")
    return sosfilt(sos, x, axis=-1)


def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) * 10.0 ** (gain_db / 20.0)


def adc(x: np.ndarray, rng: np.random.Generator | None = None,
        noise_floor_dbfs: float | None = -110.0, bits: int = 24) -> tuple[np.ndarray, bool]:
    """ADC model: additive white codec floor, hard clip at 0 dBFS, ``bits``-bit quantization.

    ``noise_floor_dbfs`` is the RMS of the white floor re full scale (1.0); ``None``
    disables it. The -110 dBFS default is an uncalibrated placeholder for the
    measured Daisy codec floor (row 20). Returns (y, clipped) where ``clipped`` is
    True if any sample reached or exceeded full scale.
    """
    y = np.array(x, dtype=np.float64, copy=True)
    if noise_floor_dbfs is not None:
        rng = rng if rng is not None else np.random.default_rng()
        y += 10.0 ** (noise_floor_dbfs / 20.0) * rng.standard_normal(y.shape)
    q = 2.0 ** (bits - 1)
    clipped = bool(np.any(np.abs(y) >= 1.0))
    y = np.clip(np.round(y * q), -q, q - 1) / q
    return y, clipped


# --------------------------------------------------------------------------- full chain


@dataclass
class TransducerChain:
    """Container for every measured transducer item, with per-scene randomization.

    Linear chain used for the target:  h_chain = drive LPF -> h1 (warped) -> projector EQ
    -> hydrophone FIR -> hydrophone tilt -> preamp HPF.  Directivity and the
    nonlinear branches are *not* in h_chain: h_chain is measured on-axis at close
    range with the gated/substitution protocol (section 8.2).
    """

    fs: int = 48000
    drive_lpf_hz: float = 8000.0
    drive_lpf_order: int = 8
    projector: Projector = field(default_factory=Projector)
    directivity: Directivity = field(default_factory=Directivity.piston)
    hydrophone: Hydrophone = field(default_factory=Hydrophone)
    preamp_hpf_hz: float = 100.0
    preamp_hpf_order: int = 2

    @property
    def calibrated(self) -> bool:
        return bool(self.projector.calibrated and self.directivity.calibrated
                    and self.hydrophone.calibrated)

    # transmit side -------------------------------------------------------------
    def transmit(self, s: np.ndarray, drive_db: float, theta_deg: float,
                 proj_eq: tuple[np.ndarray, np.ndarray] | None = None,
                 resonance_factor: float = 1.0) -> np.ndarray:
        """Clean speech -> acoustic source signal radiated toward the receiver.

        Drive level: the LPF output is peak-normalized so 0 dB drive = full-scale
        amplifier input, then scaled by ``drive_db`` (only matters for h2, h3).
        """
        x = drive_lpf(s, self.fs, self.drive_lpf_hz, self.drive_lpf_order)
        pk = np.max(np.abs(x)) if x.size else 0.0
        if pk > 0:
            x = x / pk * 10.0 ** (drive_db / 20.0)
        y = self.projector.apply(x, resonance_factor)
        if proj_eq is not None:
            y = apply_fir(y, eq_fir(proj_eq[0], proj_eq[1], self.fs), centred=True)
        return self.directivity.apply(y, theta_deg, self.fs)

    # receive side --------------------------------------------------------------
    def receive(self, p: np.ndarray, hyd_eq: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """Acoustic pressure at the hydrophone -> preamp output (before gain/ADC)."""
        y = self.hydrophone.apply(p)
        if hyd_eq is not None:
            y = apply_fir(y, eq_fir(hyd_eq[0], hyd_eq[1], self.fs), centred=True)
        return preamp_hpf(y, self.fs, self.preamp_hpf_hz, self.preamp_hpf_order)

    # linear chain for R1 -------------------------------------------------------
    def h_chain_apply(self, s: np.ndarray,
                      proj_eq: tuple[np.ndarray, np.ndarray] | None = None,
                      hyd_eq: tuple[np.ndarray, np.ndarray] | None = None,
                      resonance_factor: float = 1.0) -> np.ndarray:
        """s * h_chain (no pool path, no directivity, no nonlinearity), section 8.2."""
        x = drive_lpf(s, self.fs, self.drive_lpf_hz, self.drive_lpf_order)
        y = self.projector.apply_linear(x, resonance_factor)
        if proj_eq is not None:
            y = apply_fir(y, eq_fir(proj_eq[0], proj_eq[1], self.fs), centred=True)
        return self.receive(y, hyd_eq)


def bandlimit(x: np.ndarray, fs: int, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass [f_lo, f_hi]; edges at/above Nyquist are dropped."""
    hi_ok = f_hi is not None and f_hi < 0.49 * fs
    lo_ok = f_lo is not None and f_lo > 0
    if lo_ok and hi_ok:
        sos = butter(order, [f_lo, f_hi], btype="band", fs=fs, output="sos")
    elif lo_ok:
        sos = butter(order, f_lo, btype="high", fs=fs, output="sos")
    elif hi_ok:
        sos = butter(order, f_hi, btype="low", fs=fs, output="sos")
    else:
        return np.array(x, dtype=np.float64, copy=True)
    return sosfiltfilt(sos, x, axis=-1)
