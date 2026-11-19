"""Small audio helpers shared by every subpackage (NumPy, float64 internally)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load(path: str, always_2d: bool = False) -> tuple[np.ndarray, int]:
    """Read a WAV as float64. Mono -> (T,), multichannel -> (C, T)."""
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    x = x.T
    if x.shape[0] == 1 and not always_2d:
        x = x[0]
    return x, fs


def save(path: str, x: np.ndarray, fs: int, subtype: str = "PCM_24") -> None:
    """Write (T,) or (C, T) float audio."""
    x = np.asarray(x)
    sf.write(path, x.T if x.ndim == 2 else x, fs, subtype=subtype)


def resample(x: np.ndarray, fs_in: int, fs_out: int, axis: int = -1) -> np.ndarray:
    """Polyphase resampling with an exact rational ratio."""
    if fs_in == fs_out:
        return x
    r = Fraction(fs_out, fs_in).limit_denominator(10000)
    return resample_poly(x, r.numerator, r.denominator, axis=axis)


def rms(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=axis))


def db(x: np.ndarray | float, floor: float = 1e-20) -> np.ndarray:
    """Power-safe 20*log10 of an amplitude."""
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def db_pow(p: np.ndarray | float, floor: float = 1e-40) -> np.ndarray:
    """10*log10 of a power."""
    return 10.0 * np.log10(np.maximum(p, floor))


def fade(x: np.ndarray, fs: int, t_in: float = 0.005, t_out: float = 0.005) -> np.ndarray:
    """Raised-cosine fade in/out along the last axis."""
    y = np.array(x, dtype=np.float64, copy=True)
    n_in, n_out = int(round(t_in * fs)), int(round(t_out * fs))
    if n_in:
        y[..., :n_in] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(n_in) / n_in)
    if n_out:
        y[..., -n_out:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, n_out + 1) / n_out)
    return y
