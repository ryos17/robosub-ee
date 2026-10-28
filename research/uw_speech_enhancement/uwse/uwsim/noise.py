"""Noise bank, synthetic bubble noise and speech-band SNR mixing.

Implements RESEARCH_PLAN.md section 5.2 (noise bank with split tags: T200 unit A
= train, unit B = test only; ambient; bubbles; flagged flow noise) and section 4.2
rows 15-18 (thruster, Minnaert bubbles, ambient, speech-band SNR).

Manifest format (``manifest.jsonl`` in the bank root, one JSON object per file)::

    {"path": "thruster/unitA_t50_1m.wav", "kind": "thruster", "split": "train",
     "unit": "A", "throttle": 50, "session": "S1", "flagged": false}

``kind`` is one of thruster | ambient | bubbles | flow | self | emi (free text is
allowed); ``split`` is train | test | diag. Flagged files (e.g. flow noise with the
jet on the array) are excluded unless ``include_flagged=True``. Without a manifest,
:func:`build_manifest` infers kind from the first sub-folder and puts paths
containing ``unitB``/``unit_b``/``test`` in the test split.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from uwse.audio import resample

try:  # sibling module (section 4.2 row 16); local fallback keeps this module usable alone
    from uwse.uwsim.physics import minnaert_frequency as _minnaert_physics
except Exception:  # pragma: no cover - depends on parallel work
    _minnaert_physics = None


# --------------------------------------------------------------------------- noise bank


@dataclass
class NoiseItem:
    path: str
    kind: str
    split: str = "train"
    unit: str | None = None
    throttle: float | None = None
    session: str | None = None
    flagged: bool = False
    fs: int = 0
    frames: int = 0
    extra: dict = field(default_factory=dict)


# Folder kinds whose split/flag is fixed by section 5.2 regardless of file names.
_KIND_SPLIT = {"self": "test", "emi": "diag"}
_FLAGGED_KINDS = {"flow"}
_UNIT_B = re.compile(r"unit[\s_\-]*b(?![a-z])")
_UNIT_A = re.compile(r"unit[\s_\-]*a(?![a-z])")
_TEST = re.compile(r"(^|/)test(/|_|-|$)")


def _infer_item(root: str, rel: str) -> NoiseItem:
    """Tags from the path (fallback when no manifest exists; prefer an explicit manifest).

    Errs toward *test*: anything naming unit B or a ``test`` folder is held out, so a
    naming slip cannot leak held-out noise into training (section 5.4).
    """
    rel = rel.replace("\\", "/")
    parts = rel.split("/")
    kind = parts[0] if len(parts) > 1 else "ambient"
    low = rel.lower()
    unit = "B" if _UNIT_B.search(low) else ("A" if _UNIT_A.search(low) else None)
    split = "test" if (unit == "B" or _TEST.search(low)) else _KIND_SPLIT.get(kind.lower(), "train")
    flagged = "flag" in low or kind.lower() in _FLAGGED_KINDS
    return NoiseItem(path=rel, kind=kind, split=split, unit=unit, flagged=flagged)


def build_manifest(root: str, out: str | None = "manifest.jsonl") -> list[NoiseItem]:
    """Scan ``root`` for WAVs and infer tags from folder names; optionally write the manifest."""
    items = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.lower().endswith((".wav", ".flac")):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                items.append(_infer_item(root, rel))
    items.sort(key=lambda it: it.path)
    if out:
        with open(os.path.join(root, out), "w", encoding="utf-8") as f:
            for it in items:
                d = asdict(it)
                d.pop("fs"), d.pop("frames")
                f.write(json.dumps(d) + "\n")
    return items


class NoiseBank:
    """Index over a folder of noise WAVs with split/kind tags and seeded segment draws.

    Files are never fully loaded: ``draw`` reads only the requested segment.
    """

    def __init__(self, root: str, fs: int = 48000, manifest: str | None = "manifest.jsonl",
                 include_flagged: bool = False) -> None:
        self.root = root
        self.fs = fs
        mpath = os.path.join(root, manifest) if manifest else None
        if mpath and os.path.isfile(mpath):
            items = []
            with open(mpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    known = {k: d.pop(k) for k in list(d) if k in NoiseItem.__dataclass_fields__}
                    items.append(NoiseItem(**known, extra=d))
        else:
            items = build_manifest(root, out=None)
        for it in items:
            info = sf.info(os.path.join(root, it.path))
            it.fs, it.frames = int(info.samplerate), int(info.frames)
        self.items = [it for it in items if (include_flagged or not it.flagged) and it.frames > 0]

    def __len__(self) -> int:
        return len(self.items)

    def select(self, kind: str | Iterable[str] | None = None,
               split: str | Iterable[str] | None = "train") -> list[NoiseItem]:
        kinds = {kind} if isinstance(kind, str) else (set(kind) if kind is not None else None)
        splits = {split} if isinstance(split, str) else (set(split) if split is not None else None)
        return [it for it in self.items
                if (kinds is None or it.kind in kinds) and (splits is None or it.split in splits)]

    def has(self, kind: str, split: str | None = "train") -> bool:
        return len(self.select(kind, split)) > 0

    def draw(self, n_samples: int, rng: np.random.Generator, kind: str | Iterable[str] | None = None,
             split: str | Iterable[str] | None = "train") -> tuple[np.ndarray, dict]:
        """Random segment of ``n_samples`` at ``self.fs`` (looped if the file is short).

        Returns (segment, info) where info names the file and offset for the metadata.
        """
        pool = self.select(kind, split)
        if not pool:
            raise ValueError(f"no noise files with kind={kind!r} split={split!r} in {self.root}")
        it = pool[int(rng.integers(len(pool)))]
        need = int(np.ceil(n_samples * it.fs / self.fs)) + 16
        if it.frames >= need:
            start = int(rng.integers(0, it.frames - need + 1))
            x, _ = sf.read(os.path.join(self.root, it.path), start=start, stop=start + need,
                           dtype="float64", always_2d=True)
        else:
            start = 0
            x, _ = sf.read(os.path.join(self.root, it.path), dtype="float64", always_2d=True)
            x = np.tile(x, (int(np.ceil(need / x.shape[0])), 1))[:need]
        x = x[:, 0]  # first channel; multichannel noise is out of scope for v0
        if it.fs != self.fs:
            x = resample(x, it.fs, self.fs)
        x = x[:n_samples]
        if x.size < n_samples:
            x = np.pad(x, (0, n_samples - x.size))
        return x, {"path": it.path, "kind": it.kind, "split": it.split, "unit": it.unit,
                   "start": start, "fs_file": it.fs}


def synthetic_noise(n_samples: int, fs: int, rng: np.random.Generator,
                    slope_db_per_oct: float = -3.0) -> np.ndarray:
    """Gaussian noise with a power slope in dB/octave (fallback when no bank is given).

    For smoke tests only: it is not a model of any recorded underwater noise.
    """
    w = rng.standard_normal(n_samples)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n_samples, 1.0 / fs)
    f[0] = f[1] if f.size > 1 else 1.0
    W *= (f / 1000.0) ** (slope_db_per_oct / (20 * np.log10(2)))
    y = np.fft.irfft(W, n_samples)
    return y / (np.std(y) + 1e-20)


# --------------------------------------------------------------------------- bubbles


def minnaert_frequency(a: np.ndarray | float, depth_m: float = 0.0) -> np.ndarray:
    """Minnaert resonance f0 = (1/2 pi a) sqrt(3 gamma p / rho) (about 3.29/a Hz near the surface)."""
    if _minnaert_physics is not None:
        return np.asarray(_minnaert_physics(a, depth_m), dtype=np.float64)
    gamma, rho, p = 1.4, 998.0, 101325.0 + 998.0 * 9.81 * depth_m
    return np.sqrt(3 * gamma * p / rho) / (2 * np.pi * np.asarray(a, dtype=np.float64))


def bubble_noise(fs: int, dur: float, rate: float, radius_range: tuple[float, float] = (0.5e-3, 3e-3),
                 rng: np.random.Generator | None = None, damping: float = 0.03) -> np.ndarray:
    """Poisson train of Minnaert bursts (section 4.2 row 16).

    Each bubble is a damped sinusoid sin(2 pi f0 t) exp(-pi delta f0 t) with
    radius drawn log-uniformly from ``radius_range`` (m), amplitude log-normal
    (sigma 6 dB) and total dimensionless damping ``delta`` (0.03: an
    uncalibrated mid-range value for mm bubbles). Returns unit-RMS noise
    (zeros if rate == 0).
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = int(round(fs * dur))
    y = np.zeros(n)
    if rate <= 0 or n == 0:
        return y
    n_b = rng.poisson(rate * dur)
    if n_b == 0:
        return y
    starts = rng.integers(0, n, size=n_b)
    radii = np.exp(rng.uniform(np.log(radius_range[0]), np.log(radius_range[1]), size=n_b))
    f0 = minnaert_frequency(radii)
    amps = 10.0 ** (rng.normal(0.0, 6.0, size=n_b) / 20.0)
    for s0, f, a in zip(starts, f0, amps):
        alpha = np.pi * damping * f
        L = min(n - s0, int(np.ceil(7.0 / alpha * fs)))  # ~60 dB decay
        t = np.arange(L) / fs
        y[s0: s0 + L] += a * np.sin(2 * np.pi * f * t) * np.exp(-alpha * t)
    r = np.sqrt(np.mean(y ** 2))
    return y / r if r > 0 else y


# --------------------------------------------------------------------------- SNR mixing


def energy_vad(x: np.ndarray, fs: int, frame_s: float = 0.02, threshold_db: float = -40.0) -> np.ndarray:
    """Sample mask of frames whose energy is within ``threshold_db`` of the loudest frame."""
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(round(frame_s * fs)))
    n_fr = int(np.ceil(x.size / hop))
    xp = np.pad(x, (0, n_fr * hop - x.size))
    e = np.mean(xp.reshape(n_fr, hop) ** 2, axis=1)
    if not np.any(e > 0):
        return np.zeros(x.size, dtype=bool)
    act = 10 * np.log10(e + 1e-30) >= 10 * np.log10(e.max()) + threshold_db
    return np.repeat(act, hop)[: x.size]


def _band(x: np.ndarray, fs: int, band: tuple[float, float] | None) -> np.ndarray:
    if band is None:
        return x
    lo, hi = band
    hi_ok = hi is not None and hi < 0.49 * fs
    if lo and hi_ok:
        sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    elif lo:
        sos = butter(4, lo, btype="high", fs=fs, output="sos")
    elif hi_ok:
        sos = butter(4, hi, btype="low", fs=fs, output="sos")
    else:
        return x
    return sosfiltfilt(sos, x)


def band_snr_db(speech: np.ndarray, noise: np.ndarray, fs: int,
                band: tuple[float, float] | None = (500.0, 8000.0),
                active_mask: np.ndarray | None = None) -> float:
    """Speech-band SNR over speech-active samples (section 4.2 row 18)."""
    if active_mask is None:
        active_mask = energy_vad(speech, fs)
    m = np.asarray(active_mask, dtype=bool)
    if not m.any():
        m = np.ones_like(m)
    ps = np.mean(_band(speech, fs, band)[m] ** 2)
    pn = np.mean(_band(noise, fs, band)[m] ** 2)
    return float(10 * np.log10((ps + 1e-30) / (pn + 1e-30)))


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Tile or crop ``x`` to length ``n``."""
    if x.size >= n:
        return x[:n]
    return np.tile(x, int(np.ceil(n / max(x.size, 1))))[:n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, fs: int,
               band: tuple[float, float] | None = (500.0, 8000.0),
               active_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scale ``noise`` so the speech-band, speech-active SNR equals ``snr_db``.

    SNR = 10 log10( mean(s_b^2 | active) / mean(n_b^2 | active) ), where _b is the
    zero-phase band-pass ``band`` and ``active`` comes from ``active_mask`` or an
    energy VAD on ``speech``. Returns (speech + scaled_noise, scaled_noise).
    """
    speech = np.asarray(speech, dtype=np.float64)
    noise = fit_length(np.asarray(noise, dtype=np.float64), speech.size)
    cur = band_snr_db(speech, noise, fs, band, active_mask)
    if not np.isfinite(cur) or np.all(noise == 0):
        return speech.copy(), noise
    g = 10.0 ** ((cur - snr_db) / 20.0)
    n = noise * g
    return speech + n, n


def combine_noises(components: Sequence[tuple[np.ndarray, float]]) -> np.ndarray:
    """Sum unit-RMS-normalized components with relative levels in dB."""
    out = None
    for x, rel_db in components:
        r = np.sqrt(np.mean(x ** 2))
        if r <= 0:
            continue
        y = x / r * 10.0 ** (rel_db / 20.0)
        out = y if out is None else out + y
    return out
