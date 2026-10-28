"""Simulator configuration: every randomization range of RESEARCH_PLAN.md section 4.2.

A :class:`SimConfig` is a tree of dataclasses that round-trips through YAML
(``configs/uwsim_v0.yaml``). Ranges are ``[lo, hi]`` pairs sampled uniformly.
``calibrated: false`` marks configs whose values are textbook/uncalibrated
placeholders; the frozen v0 (gate G3) replaces them with fitted values from
section 4.3 and sets the flag.

Row numbers below refer to the section 4.2 table.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any

import yaml

Range = tuple[float, float]


@dataclass
class BoundaryRanges:
    """Fluid-elastic half-space priors for walls/floor (row 5). Unverified textbook values."""

    cp: Range = (3500.0, 5500.0)
    cs: Range = (2000.0, 3000.0)
    rho: Range = (2200.0, 2500.0)
    eta: Range = (0.01, 0.05)  # loss factor; uncalibrated guess


@dataclass
class PoolConfig:
    """Rows 3-5: geometry, surface, boundaries."""

    calibrated_name: str = "avery"          # geometry preset of the calibrated pool
    calibrated_dims: dict | None = None     # override {L, W, H0, H1}; None -> geometry preset
    generic_fraction: float = 0.2           # row 3: 20 % generic pools
    dim_jitter: float = 0.10                # L, W, H +/-10 %
    slope_jitter: float = 0.50              # floor slope +/-50 %
    generic_L: Range = (15.0, 50.0)
    generic_W: Range = (8.0, 25.0)
    generic_H: Range = (1.2, 5.0)
    generic_rel_slope: Range = (0.0, 0.4)   # (H1-H0)/H_mean for generic pools; uncalibrated
    surface_R: Range = (-1.0, -0.9)         # row 4
    walls: BoundaryRanges = field(default_factory=BoundaryRanges)
    floor: BoundaryRanges = field(default_factory=BoundaryRanges)


@dataclass
class AcousticsConfig:
    """Rows 1, 6-8, 14."""

    temperature_c: float = 26.0             # c from Marczak at this T unless c_nominal is set
    c_nominal: float | None = None          # measured c (TOF slope) replaces this when calibrated
    c_jitter: float = 0.005                 # +/-0.5 %
    t_e: Range = (0.030, 0.050)             # row 6, s
    # Row 7 single-slope T60 per octave (s). Uncalibrated guess inside the 0.25-1 s pool range.
    t60_bands: dict = field(default_factory=lambda: {125: 1.0, 250: 0.9, 500: 0.8, 1000: 0.7,
                                                     2000: 0.6, 4000: 0.5, 8000: 0.4})
    t60_two_slope: dict | None = None       # set when H1a selects the two-slope model
    t60_scale: Range = (0.7, 1.3)
    f_x_min: float = 500.0                  # row 8: f_x = max(f_s, 500 Hz)
    modal_f_max_factor: float = 2.0         # modal sum computed up to factor * f_x
    rir_length_s: float = 1.0
    eps_max: float = 1e-3                   # row 14: |eps| <= 1e-3
    ir_perturb_db: float = 0.0              # row 14: per-octave sweep-to-sweep SD (E3); 0 = off


@dataclass
class GeometryConfig:
    """Row 13 positions and row 12 orientation."""

    source_depth_min: float = 1.0           # LL916 rated 1-18 m
    receiver_depth_min: float = 0.1
    wall_margin: float = 0.1
    range_m: Range = (0.3, 25.0)
    orientation_deg: Range = (0.0, 360.0)
    range_sampling: str = "volume"          # "volume" (uniform in volume, rejection) | "uniform_range"
    thruster_dist: Range = (0.3, 1.5)       # row 15: thruster 0.3-1.5 m from the array


@dataclass
class ChainConfig:
    """Rows 10-12, 19, 20. File fields point to measured .npy/.npz; None = uncalibrated default."""

    drive_lpf_hz: float = 8000.0
    drive_lpf_order: int = 8
    drive_level_db: Range = (-10.0, 0.0)    # row 11
    h1_path: str | None = None              # measured chain FIR (E2)
    h2_path: str | None = None              # ESS harmonic IRs
    h3_path: str | None = None
    proj_eq_db: float = 3.0                 # row 10: +/-3 dB random 1/3-octave
    resonance_shift: float = 0.20           # row 10: +/-20 %
    directivity_path: str | None = None     # npz with angles_deg, bands_hz, gains_db
    piston_radius_m: float = 0.117          # LL916 a (row 12)
    hydrophone_fir_path: str | None = None
    hyd_tilt_db: float = 2.0                # row 19
    preamp_hpf_hz: float = 100.0            # uncalibrated corner
    preamp_hpf_order: int = 2
    adc_floor_dbfs: float = -110.0          # uncalibrated codec floor
    adc_jitter_db: float = 3.0              # row 20
    adc_bits: int = 24
    level_dbfs: Range = (-35.0, -15.0)      # speech-active RMS at the ADC; uncalibrated


@dataclass
class NoiseConfig:
    """Rows 15-18."""

    snr_measured: Range = (0.0, 15.0)       # placeholder for the E6 per-take distribution
    snr_widen_db: float = 10.0
    snr_clip: Range = (-10.0, 25.0)
    band: Range = (500.0, 8000.0)
    thruster_prob: float = 0.7
    thruster_split: str = "train"           # unit A only for training (section 5.2)
    ambient_prob: float = 1.0
    ambient_rel_db: Range = (-20.0, -5.0)   # re thruster
    bubble_rate: Range = (0.0, 20.0)        # per s
    bubble_radius_m: Range = (0.5e-3, 3e-3)
    bubble_rel_db: Range = (-20.0, 0.0)
    recorded_bubbles_prob: float = 0.0


@dataclass
class TargetConfig:
    kind: str = "R1"                         # "R1" (chain reference) | "R0" (band-limited clean)
    f_lo: float = 500.0
    f_hi: float = 8000.0
    gain: str = "rms"                        # "rms" (level-matched) | "ls" (least-squares fit, section 8.2)


@dataclass
class SimConfig:
    version: str = "uwsim_v0"
    calibrated: bool = False
    fs: int = 48000
    pool: PoolConfig = field(default_factory=PoolConfig)
    acoustics: AcousticsConfig = field(default_factory=AcousticsConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    target: TargetConfig = field(default_factory=TargetConfig)


# --------------------------------------------------------------------------- (de)serialization


def _to_plain(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {f.name: _to_plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, tuple):
        return [_to_plain(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    return x


def _from_plain(cls: type, d: dict) -> Any:
    if d is None:
        return cls()
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kw = {}
    for k, v in d.items():
        t = hints[k]
        if dataclasses.is_dataclass(t):
            kw[k] = _from_plain(t, v)
        elif typing.get_origin(t) is tuple and isinstance(v, (list, tuple)):
            kw[k] = tuple(float(a) for a in v)
        else:
            kw[k] = v
    return cls(**kw)


def to_dict(cfg: SimConfig) -> dict:
    return _to_plain(cfg)


def from_dict(d: dict) -> SimConfig:
    return _from_plain(SimConfig, d)


def load_config(path: str) -> SimConfig:
    with open(path, encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f) or {})


def save_config(cfg: SimConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
