"""Scene sampling, RIR rendering and the end-to-end pool simulation.

Implements the full chain of RESEARCH_PLAN.md section 4.1 with the section 4.2
randomization, the noise model of rows 15-18, and the training targets of section
8.2:

* ``R1_sim = s * h_chain``: chain coloration only, **no pool path** (identical to the
  real R1), band-limited to [f_lo, 8 kHz]; aligned to the direct-path arrival and
  given a least-squares broadband gain against the received speech, which is the
  same delay+gain fit applied to the real R1 (RealMAN-style).
* ``R0 = s`` band-limited, aligned and scaled the same way.

All processing runs at 48 kHz; outputs are resampled to the model rate (16 kHz by
default, 48 kHz for DFN3).

Sibling modules (geometry, reflection, ism, tail, lowband, physics) are imported
lazily, so scene sampling and ``simulate`` with a precomputed RIR work without them.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.signal import fftconvolve
from scipy.signal import resample as fft_resample

from uwse.audio import resample
from uwse.uwsim.chain import (
    OCTAVE_CENTRES,
    Directivity,
    Hydrophone,
    Projector,
    TransducerChain,
    adc,
    apply_fir,
    bandlimit,
    eq_fir,
    random_third_octave_gains,
    tilt_gains,
)
from uwse.uwsim.config import SimConfig
from uwse.uwsim.noise import (
    NoiseBank,
    bubble_noise,
    combine_noises,
    energy_vad,
    mix_at_snr,
    synthetic_noise,
)

FS_SIM = 48000

# Used only if uwse.uwsim.geometry is unavailable. Approximate, NOT survey data:
# copies of the geometry presets (Avery competition 37 x 20 m, 3.35-4.27 m; Maas well 25 x 17 m, 18 ft).
_FALLBACK_POOLS = {
    "avery": {"L": 37.0, "W": 20.0, "H0": 3.35, "H1": 4.27},
    "maas": {"L": 25.0, "W": 17.0, "H0": 5.4864, "H1": 5.4864},
}


# --------------------------------------------------------------------------- helpers


def sound_speed(temperature_c: float) -> float:
    """Marczak pure-water c(T) (physics module if present, same polynomial otherwise)."""
    try:
        from uwse.uwsim.physics import sound_speed_marczak
        return float(sound_speed_marczak(temperature_c))
    except Exception:
        T = temperature_c
        return float(1.402385e3 + 5.038813 * T - 5.799136e-2 * T ** 2 + 3.287156e-4 * T ** 3
                     - 1.398845e-6 * T ** 4 + 2.787860e-9 * T ** 5)


def _calibrated_dims(cfg: SimConfig) -> dict:
    if cfg.pool.calibrated_dims:
        return {k: float(cfg.pool.calibrated_dims[k]) for k in ("L", "W", "H0", "H1")}
    name = cfg.pool.calibrated_name.lower()
    try:
        from uwse.uwsim import geometry
        fn = {"avery": "avery_competition_pool", "maas": "maas_dive_well"}.get(name, name)
        p = getattr(geometry, fn)()
        return {"L": float(p.L), "W": float(p.W), "H0": float(p.H0), "H1": float(p.H1)}
    except Exception:
        return dict(_FALLBACK_POOLS[name])


def floor_depth(pool: dict, x: float) -> float:
    """Depth of the floor tilted along x (z positive down from the surface)."""
    return pool["H0"] + (pool["H1"] - pool["H0"]) * x / pool["L"]


def _u(rng: np.random.Generator, r: Sequence[float]) -> float:
    return float(rng.uniform(r[0], r[1]))


def _sample_point(rng: np.random.Generator, pool: dict, z_min: float, margin: float) -> np.ndarray:
    """Uniform point in the water volume (rejection against the tilted floor)."""
    h_max = max(pool["H0"], pool["H1"])
    for _ in range(1000):
        x = rng.uniform(margin, pool["L"] - margin)
        y = rng.uniform(margin, pool["W"] - margin)
        hz = floor_depth(pool, x) - margin
        z_lo = min(z_min, hz)
        z = rng.uniform(z_lo, max(z_lo, h_max - margin))
        if z <= hz:
            return np.array([x, y, z])
    return np.array([pool["L"] / 2, pool["W"] / 2, floor_depth(pool, pool["L"] / 2) / 2])


def _inside(pool: dict, p: np.ndarray, z_min: float, margin: float) -> bool:
    x, y, z = p
    return (margin <= x <= pool["L"] - margin and margin <= y <= pool["W"] - margin
            and z_min <= z <= floor_depth(pool, x) - margin)


def _random_dir(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- scene


@dataclass
class Scene:
    """Every sampled parameter of one simulated take (JSON-serializable via to_dict)."""

    seed: int
    rir_seed: int
    calibrated: bool
    is_generic: bool
    pool: dict
    c: float
    surface_R: float
    walls: dict
    floor: dict
    t_e: float
    t60_bands: dict
    t60_two_slope: dict | None
    f_x: float
    modal_f_max: float
    rir_length_s: float
    src: list
    rx: list
    range_m: float
    orientation_deg: float
    theta_deg: float
    drive_db: float
    resonance_factor: float
    proj_eq: list
    hyd_eq: list
    eps: float
    ir_perturb: list | None
    thruster_on: bool
    thruster_pos: list | None
    thruster_split: str
    ambient_on: bool
    ambient_rel_db: float
    bubble_rate: float
    bubble_radius_m: list
    bubble_rel_db: float
    snr_db: float
    noise_band: list
    adc_floor_dbfs: float
    adc_bits: int
    level_dbfs: float
    target_band: list
    lowband_src_idx: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(dataclasses.asdict(self), default=_json_default))

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def _sample_boundary(rng: np.random.Generator, br) -> dict:
    return {"cp": _u(rng, br.cp), "cs": _u(rng, br.cs), "rho": _u(rng, br.rho), "eta": _u(rng, br.eta)}


def sample_scene(cfg: SimConfig, rng: np.random.Generator | int, generic: bool | None = None) -> Scene:
    """Draw one scene from the section 4.2 randomization ranges in ``cfg``.

    ``rng`` may be a Generator or an int seed. ``generic`` forces a generic (True)
    or calibrated-pool (False) scene; None draws it with ``cfg.pool.generic_fraction``.
    """
    seed = int(rng) if isinstance(rng, (int, np.integer)) else int(rng.integers(2 ** 31))
    rng = np.random.default_rng(seed)
    pc, ac, gc, cc, nc = cfg.pool, cfg.acoustics, cfg.geometry, cfg.chain, cfg.noise

    is_generic = bool(rng.random() < pc.generic_fraction) if generic is None else bool(generic)
    if is_generic:
        L, W, Hm = _u(rng, pc.generic_L), _u(rng, pc.generic_W), _u(rng, pc.generic_H)
        s = _u(rng, pc.generic_rel_slope) * (1 if rng.random() < 0.5 else -1)
        lo, hi = pc.generic_H
        pool = {"L": L, "W": W, "H0": float(np.clip(Hm * (1 - s / 2), lo, hi)),
                "H1": float(np.clip(Hm * (1 + s / 2), lo, hi))}
    else:
        b = _calibrated_dims(cfg)
        j = pc.dim_jitter
        sL, sW, sH = (float(rng.uniform(1 - j, 1 + j)) for _ in range(3))
        slope = (b["H1"] - b["H0"]) * sH * float(rng.uniform(1 - pc.slope_jitter, 1 + pc.slope_jitter))
        H0 = b["H0"] * sH
        pool = {"L": b["L"] * sL, "W": b["W"] * sW, "H0": H0, "H1": H0 + slope}

    c0 = ac.c_nominal if ac.c_nominal is not None else sound_speed(ac.temperature_c)
    c = c0 * float(rng.uniform(1 - ac.c_jitter, 1 + ac.c_jitter))
    t60_scale = _u(rng, ac.t60_scale)
    t60 = {str(float(k)): float(v) * t60_scale for k, v in ac.t60_bands.items()}
    two = None
    if ac.t60_two_slope:
        two = json.loads(json.dumps(ac.t60_two_slope))
        two["t60_scale"] = t60_scale
    # Row 8: waveguide cutoff c/(4H) (pressure-release top, rigid bottom) at the deepest point.
    f_s = c / (4.0 * max(pool["H0"], pool["H1"]))
    f_x = max(f_s, ac.f_x_min)

    margin = gc.wall_margin
    r_lo, r_hi = gc.range_m
    src = rx = None
    extra: dict = {}
    for _ in range(2000):
        rx_c = _sample_point(rng, pool, gc.receiver_depth_min, margin)
        if gc.range_sampling == "uniform_range":
            src_c = rx_c + _u(rng, gc.range_m) * _random_dir(rng)
            if not _inside(pool, src_c, gc.source_depth_min, margin):
                continue
        else:
            src_c = _sample_point(rng, pool, gc.source_depth_min, margin)
        if r_lo <= np.linalg.norm(src_c - rx_c) <= r_hi:
            src, rx = src_c, rx_c
            break
    if src is None:  # pool too small/large for the range constraint: place at r_lo along +/-x
        rx = _sample_point(rng, pool, gc.receiver_depth_min, margin)
        sgn = 1.0 if rx[0] + r_lo <= pool["L"] - margin else -1.0
        src = rx + np.array([sgn * r_lo, 0.0, 0.0])
        src[2] = max(src[2], min(gc.source_depth_min, floor_depth(pool, src[0]) - margin))
        extra["placement_fallback"] = True  # constraint not met by sampling; logged, not hidden
    rng_m = float(np.linalg.norm(src - rx))

    # Projector axis horizontal at azimuth phi; theta = angle off-axis toward the receiver.
    phi = _u(rng, gc.orientation_deg)
    axis = np.array([np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi)), 0.0])
    u = (rx - src) / max(rng_m, 1e-9)
    theta = float(np.rad2deg(np.arccos(np.clip(axis @ u, -1.0, 1.0))))

    pf, pg = random_third_octave_gains(rng, cc.proj_eq_db)
    hf, hg = tilt_gains(float(rng.uniform(-cc.hyd_tilt_db, cc.hyd_tilt_db)))
    ir_pert = None
    if ac.ir_perturb_db > 0:
        ir_pert = [OCTAVE_CENTRES.tolist(), rng.normal(0, ac.ir_perturb_db, OCTAVE_CENTRES.size).tolist()]

    thr_on = bool(rng.random() < nc.thruster_prob)
    thr_pos = None
    if thr_on:
        for _ in range(500):
            p = rx + _u(rng, gc.thruster_dist) * _random_dir(rng)
            if _inside(pool, p, 0.05, 0.05):
                thr_pos = p
                break
        if thr_pos is None:
            thr_pos = rx + np.array([0.0, 0.0, -min(gc.thruster_dist[0], rx[2] * 0.5)])

    snr_lo = max(nc.snr_measured[0] - nc.snr_widen_db, nc.snr_clip[0])
    snr_hi = min(nc.snr_measured[1] + nc.snr_widen_db, nc.snr_clip[1])

    return Scene(
        seed=seed, rir_seed=int(rng.integers(2 ** 31)), calibrated=bool(cfg.calibrated),
        is_generic=is_generic, pool=pool, c=c, surface_R=_u(rng, pc.surface_R),
        walls=_sample_boundary(rng, pc.walls), floor=_sample_boundary(rng, pc.floor),
        t_e=_u(rng, ac.t_e), t60_bands=t60, t60_two_slope=two, f_x=float(f_x),
        modal_f_max=float(f_x * ac.modal_f_max_factor), rir_length_s=float(ac.rir_length_s),
        src=src.tolist(), rx=rx.tolist(), range_m=rng_m, orientation_deg=phi, theta_deg=theta,
        drive_db=_u(rng, cc.drive_level_db),
        resonance_factor=float(rng.uniform(1 - cc.resonance_shift, 1 + cc.resonance_shift)),
        proj_eq=[pf.tolist(), pg.tolist()], hyd_eq=[hf.tolist(), hg.tolist()],
        eps=float(rng.uniform(-ac.eps_max, ac.eps_max)), ir_perturb=ir_pert,
        thruster_on=thr_on, thruster_pos=None if thr_pos is None else np.asarray(thr_pos).tolist(),
        thruster_split=nc.thruster_split,
        ambient_on=bool(rng.random() < nc.ambient_prob), ambient_rel_db=_u(rng, nc.ambient_rel_db),
        bubble_rate=_u(rng, nc.bubble_rate), bubble_radius_m=list(nc.bubble_radius_m),
        bubble_rel_db=_u(rng, nc.bubble_rel_db),
        snr_db=float(rng.uniform(snr_lo, snr_hi)), noise_band=list(nc.band),
        adc_floor_dbfs=cc.adc_floor_dbfs + float(rng.uniform(-cc.adc_jitter_db, cc.adc_jitter_db)),
        adc_bits=int(cc.adc_bits), level_dbfs=_u(rng, cc.level_dbfs),
        target_band=[cfg.target.f_lo, cfg.target.f_hi],
        lowband_src_idx=int(rng.integers(2 ** 31)), extra=extra,
    )


# --------------------------------------------------------------------------- RIR rendering


def _pool_obj(scene: Scene):
    from uwse.uwsim.geometry import Pool
    p = scene.pool
    return Pool(p["L"], p["W"], p["H0"], p["H1"])


def _boundary(d: dict):
    from uwse.uwsim.reflection import Boundary
    return Boundary(**d)


def _t60_arg(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """(centres, t60) for tail.splice_tail: t60 is (B,) single-slope or (B, 3) two-slope.

    Two-slope config format: {"centres": [...], "t60_1": [...], "t60_2": [...],
    "level2_db": [...]}; both T60s are multiplied by the scene's t60_scale.
    """
    ts = scene.t60_two_slope
    if ts:
        k = float(ts.get("t60_scale", 1.0))
        t60 = np.stack([np.asarray(ts["t60_1"], float) * k, np.asarray(ts["t60_2"], float) * k,
                        np.asarray(ts["level2_db"], float)], axis=1)
        return np.asarray(ts["centres"], float), t60
    items = sorted((float(f), float(v)) for f, v in scene.t60_bands.items())
    return np.array([f for f, _ in items]), np.array([v for _, v in items])


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:n] if x.size >= n else np.pad(x, (0, n - x.size))


def _early_plus_tail(src, rx, scene: Scene, rng: np.random.Generator, n: int) -> np.ndarray:
    """ISM up to t_e after the direct arrival, then the late tail.

    t_e is measured from the direct arrival (r/c), not from emission, so that the
    early part spans the same reflection window at 0.3 m and at 25 m.
    """
    from uwse.uwsim.ism import ism_rir
    from uwse.uwsim.tail import splice_tail
    t_abs = float(np.linalg.norm(np.asarray(src) - np.asarray(rx))) / scene.c + scene.t_e
    res = ism_rir(np.asarray(src, float), np.asarray(rx, float), _pool_obj(scene), scene.c, FS_SIM, t_abs,
                  surface_R=scene.surface_R, walls=_boundary(scene.walls), floor=_boundary(scene.floor))
    early = np.asarray(res.rir, dtype=np.float64)
    centres, t60 = _t60_arg(scene)
    return _fit(splice_tail(early, FS_SIM, t_abs, t60, rng, n, centres=centres), n)


def render_rir(scene: Scene, measured_low_irs: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """Source -> receiver pool RIR at 48 kHz (section 4.1 channel h_ch).

    ISM early part (t <= t_e, signed angle-dependent R) + late tail, then the
    band below f_x is replaced (LR4 crossover) by a low-band IR: the measured IR
    from ``measured_low_irs`` for calibrated-pool scenes when given (index jittered
    by ``scene.lowband_src_idx``), otherwise the shoebox modal sum at the mean
    depth with damping 6.91/T60 of the lowest tail band.
    """
    from uwse.uwsim.lowband import modal_rir, splice_lowband
    rng = np.random.default_rng(scene.rir_seed)
    n = int(round(scene.rir_length_s * FS_SIM))
    high = _early_plus_tail(scene.src, scene.rx, scene, rng, n)
    if measured_low_irs and not scene.is_generic:
        low = _fit(measured_low_irs[scene.lowband_src_idx % len(measured_low_irs)], n)
    else:
        # Modal model is a flat box at the mean depth: clamp positions into it.
        p = scene.pool
        H = 0.5 * (p["H0"] + p["H1"])
        centres, t60 = _t60_arg(scene)
        t60 = t60 if t60.ndim == 1 else t60[:, 0]
        damp = (float(np.interp(np.log(250.0), np.log(centres), t60)),
                float(np.interp(np.log(500.0), np.log(centres), t60)))

        def clamp(q):
            return (float(np.clip(q[0], 0, p["L"])), float(np.clip(q[1], 0, p["W"])),
                    float(np.clip(q[2], 0.01 * H, 0.99 * H)))

        low = _fit(modal_rir(clamp(scene.src), clamp(scene.rx), p["L"], p["W"], H, scene.c, FS_SIM, n,
                             f_max=scene.modal_f_max, damping=damp, damping_freqs=(250.0, 500.0)), n)
    return _fit(splice_lowband(high, low, FS_SIM, scene.f_x), n)


def render_thruster_rir(scene: Scene) -> np.ndarray | None:
    """Thruster -> receiver IR (row 15): ISM early + tail, no low-band splice."""
    if not scene.thruster_on or scene.thruster_pos is None:
        return None
    rng = np.random.default_rng(scene.rir_seed + 1)
    n = int(round(scene.rir_length_s * FS_SIM))
    return _early_plus_tail(scene.thruster_pos, scene.rx, scene, rng, n)


def direct_path_index(rir: np.ndarray, scene: Scene, fs: int = FS_SIM, search: int = 96,
                      mode: str = "geometric") -> int:
    """Sample index of the direct arrival.

    ``mode="geometric"`` (default): round(range / c * fs), the exact direct-path delay of
    the ISM (which renders image i at r_i / c with no added latency). Peak picking is
    unreliable: at range the direct path and the R_s ~ -1 surface image arrive within a
    sample or two and cancel (Lloyd's mirror), so the largest |rir| sample is often a
    later floor/wall arrival 0.5-1.5 ms after the direct one.
    ``mode="peak"``: |rir| peak within +/-``search`` of range/c (for measured IRs whose
    time origin is not the emission time).
    """
    n0 = int(round(scene.range_m / scene.c * fs))
    if mode == "geometric":
        return n0
    if mode != "peak":
        raise ValueError("mode must be 'geometric' or 'peak'")
    lo, hi = max(0, n0 - search), min(rir.size, n0 + search + 1)
    if hi <= lo:
        return int(np.argmax(np.abs(rir)))
    return lo + int(np.argmax(np.abs(rir[lo:hi])))


# --------------------------------------------------------------------------- chain from config


def _load_arr(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    a = np.load(path)
    if isinstance(a, np.lib.npyio.NpzFile):
        a = a[a.files[0]]
    return np.asarray(a, dtype=np.float64).ravel()


def build_chain(cfg: SimConfig, c: float = 1500.0) -> TransducerChain:
    """TransducerChain from the measured files in ``cfg.chain`` (uncalibrated defaults if absent)."""
    cc = cfg.chain
    h1 = _load_arr(cc.h1_path)
    proj = Projector(h1=h1 if h1 is not None else np.array([1.0]), h2=_load_arr(cc.h2_path),
                     h3=_load_arr(cc.h3_path), calibrated=h1 is not None)
    if cc.directivity_path:
        z = np.load(cc.directivity_path)
        direc = Directivity(np.asarray(z["angles_deg"], float), np.asarray(z["bands_hz"], float),
                            np.asarray(z["gains_db"], float), calibrated=True)
    else:
        direc = Directivity.piston(a=cc.piston_radius_m, c=c)
    hf = _load_arr(cc.hydrophone_fir_path)
    hyd = Hydrophone(fir=hf if hf is not None else np.array([1.0]), calibrated=hf is not None)
    return TransducerChain(fs=cfg.fs, drive_lpf_hz=cc.drive_lpf_hz, drive_lpf_order=cc.drive_lpf_order,
                           projector=proj, directivity=direc, hydrophone=hyd,
                           preamp_hpf_hz=cc.preamp_hpf_hz, preamp_hpf_order=cc.preamp_hpf_order)


# --------------------------------------------------------------------------- simulate


def time_scale(x: np.ndarray, eps: float) -> np.ndarray:
    """y[n] = x(n (1 + eps)) (row 14; same convention as tail.apply_time_scale), length kept.

    FFT resampling (one FFT of the signal) instead of a per-sample sinc, for memory.
    """
    if eps == 0.0 or x.size < 2:
        return x
    y = fft_resample(x, max(2, int(round(x.size / (1.0 + eps)))))
    return _fit(y, x.size)


def _shift(x: np.ndarray, d: int) -> np.ndarray:
    return np.concatenate([np.zeros(d), x[: x.size - d]]) if d > 0 else x


def chain_delay(chain: TransducerChain, proj_eq=None, hyd_eq=None, resonance_factor: float = 1.0,
                band: tuple[float, float] = (500.0, 8000.0), fs: int = FS_SIM, n: int = 8192,
                pre: int = 1024) -> int:
    """Bulk delay (samples) of the linear chain h_chain in ``band``: peak of its band-limited IR.

    This is the broadband delay a per-utterance delay fit (section 8.2) would assign to
    R0 = s against the received speech; R1 already contains it because it is s * h_chain.
    """
    d = np.zeros(n)
    d[pre] = 1.0
    h = bandlimit(chain.h_chain_apply(d, proj_eq, hyd_eq, resonance_factor), fs, *band)
    return int(np.argmax(np.abs(h))) - pre


def _ls_gain(t: np.ndarray, y: np.ndarray) -> float:
    den = float(t @ t)
    return float(t @ y) / den if den > 0 else 0.0


def simulate(clean_48k: np.ndarray, scene: Scene, noise_bank: NoiseBank | None,
             rng: np.random.Generator, *, fs_model: int = 16000, chain: TransducerChain | None = None,
             rir: np.ndarray | None = None, thruster_rir: np.ndarray | None = None,
             measured_low_irs: Sequence[np.ndarray] | None = None, target_gain: str = "rms") -> dict:
    """Run clean speech (48 kHz, 1-D) through the full section 4.1 chain.

    ``rir``/``thruster_rir`` may be supplied (e.g. from an RIRBank); otherwise they are
    rendered from ``scene``. ``noise_bank=None`` (or a bank lacking a kind) falls back to
    synthetic noise / bubbles so smoke tests run without recordings.

    ``target_gain``: "ls" applies the broadband least-squares gain of the target against
    the received speech (the literal section 8.2 fit; near the Lloyd's-mirror nulls it
    can be tiny or negative); "rms" (default) matches the target's speech-active RMS to
    the received speech's, which keeps training targets at a stable level. Both gains
    are recorded in ``meta``.

    Returns dict(noisy, target_R1, target_R0, fs, meta); signals are float32 at
    ``fs_model`` with len = round(len(clean) * fs_model / 48000).
    """
    fs = FS_SIM
    s = np.asarray(clean_48k, dtype=np.float64).ravel()
    n = s.size
    chain = chain if chain is not None else TransducerChain(fs=fs, directivity=Directivity.piston(c=scene.c))
    proj_eq = (np.asarray(scene.proj_eq[0]), np.asarray(scene.proj_eq[1]))
    hyd_eq = (np.asarray(scene.hyd_eq[0]), np.asarray(scene.hyd_eq[1]))
    nb, tb = tuple(scene.noise_band), tuple(scene.target_band)
    meta: dict = {"scene": scene.to_dict(), "fs_sim": fs, "fs_model": fs_model,
                  "chain_calibrated": chain.calibrated, "noise": []}

    # ---- transmit, channel, time variation
    x = chain.transmit(s, scene.drive_db, scene.theta_deg, proj_eq, scene.resonance_factor)
    if rir is None:
        rir = render_rir(scene, measured_low_irs)
    rir = np.asarray(rir, dtype=np.float64)
    if scene.ir_perturb:
        rir = apply_fir(rir, eq_fir(scene.ir_perturb[0], scene.ir_perturb[1], fs), centred=True)
    p = time_scale(fftconvolve(x, rir)[:n], scene.eps)
    d0 = direct_path_index(rir, scene)
    meta["direct_index_48k"] = d0

    # ---- noise (rows 15-18)
    s_del = _shift(s, d0)
    mask = energy_vad(s_del, fs)
    comps: list[tuple[np.ndarray, float]] = []
    have_bank = noise_bank is not None and len(noise_bank) > 0
    if have_bank and int(noise_bank.fs) != fs:
        raise ValueError(f"noise_bank.fs = {noise_bank.fs} but simulate runs at {fs} Hz; "
                         f"build the NoiseBank with fs={fs}")
    # One split for every recorded noise kind (section 5.4: training noise = unit A /
    # training sessions only). ``thruster_split`` is the scene's noise split.
    split = scene.thruster_split
    if scene.thruster_on:
        if have_bank and noise_bank.has("thruster", scene.thruster_split):
            if thruster_rir is None:
                thruster_rir = render_thruster_rir(scene)
            if thruster_rir is not None:
                Lt = int(np.asarray(thruster_rir).size)
                seg, info = noise_bank.draw(n + Lt, rng, "thruster", scene.thruster_split)
                comps.append((fftconvolve(seg, thruster_rir)[Lt: Lt + n], 0.0))
            else:
                seg, info = noise_bank.draw(n, rng, "thruster", scene.thruster_split)
                comps.append((seg, 0.0))
            meta["noise"].append(info)
    if scene.ambient_on and have_bank and noise_bank.has("ambient", split):
        seg, info = noise_bank.draw(n, rng, "ambient", split)
        comps.append((seg, scene.ambient_rel_db))
        meta["noise"].append(info)
    if scene.bubble_rate > 0:
        b = bubble_noise(fs, n / fs, scene.bubble_rate, tuple(scene.bubble_radius_m), rng)
        if np.any(b):
            comps.append((b, scene.bubble_rel_db))
            meta["noise"].append({"kind": "bubbles_synthetic", "rate": scene.bubble_rate})
    noise = combine_noises(comps) if comps else None
    if noise is None or not np.any(noise):
        noise = synthetic_noise(n, fs, rng)
        meta["noise"].append({"kind": "synthetic_fallback"})
    _, noise = mix_at_snr(p, noise, scene.snr_db, fs, nb, mask)

    # ---- receive chain, gain, ADC (rows 19, 20)
    sp_rx = chain.receive(p, hyd_eq)
    mix_rx = sp_rx + chain.receive(noise, hyd_eq)
    act = mask if mask.any() else np.ones(n, bool)
    lvl = np.sqrt(np.mean(mix_rx[act] ** 2)) + 1e-20
    gain = 10.0 ** (scene.level_dbfs / 20.0) / lvl
    sp_out = sp_rx * gain
    noisy, clipped = adc(mix_rx * gain, rng, scene.adc_floor_dbfs, scene.adc_bits)
    meta.update(gain_db=float(20 * np.log10(gain)), clipped=clipped)

    # ---- targets (section 8.2): delay = direct arrival, broadband gain vs received speech.
    # R0 = s also gets the chain's bulk delay (R1 = s * h_chain contains it already), and
    # both get the row-14 time scaling, so that the only target/input misalignment left is
    # the pool path itself (otherwise eps = 1e-3 drifts the target 4 ms over 4 s).
    sp_b = bandlimit(sp_out, fs, *tb)
    d_chain = chain_delay(chain, proj_eq, hyd_eq, scene.resonance_factor, tb, fs)
    meta["chain_delay_48k"] = d_chain
    r1 = _shift(bandlimit(chain.h_chain_apply(s, proj_eq, hyd_eq, scene.resonance_factor), fs, *tb), d0)
    r0 = _shift(bandlimit(s, fs, *tb), max(d0 + d_chain, 0))
    r1, r0 = time_scale(r1, scene.eps), time_scale(r0, scene.eps)
    g1, g0 = _ls_gain(r1, sp_b), _ls_gain(r0, sp_b)
    ref = np.sqrt(np.mean(sp_b[act] ** 2))
    q1 = ref / (np.sqrt(np.mean(r1[act] ** 2)) + 1e-20)
    q0 = ref / (np.sqrt(np.mean(r0[act] ** 2)) + 1e-20)
    meta.update(ls_gain_R1=g1, ls_gain_R0=g0, rms_gain_R1=float(q1), rms_gain_R0=float(q0),
                target_gain=target_gain)
    if target_gain == "ls":
        r1, r0 = r1 * g1, r0 * g0
    elif target_gain == "rms":
        r1, r0 = r1 * q1, r0 * q0
    else:
        raise ValueError("target_gain must be 'ls' or 'rms'")

    out = {}
    n_out = int(round(n * fs_model / fs))
    for k, v in (("noisy", noisy), ("target_R1", r1), ("target_R0", r0)):
        out[k] = _fit(resample(v, fs, fs_model), n_out).astype(np.float32)
    out["fs"] = fs_model
    out["meta"] = meta
    return out
