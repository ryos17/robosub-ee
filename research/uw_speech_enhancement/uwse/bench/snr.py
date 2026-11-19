"""Per-utterance SNR of pool recordings (measurement method signed off 2026-11-08).

Method (RESULTS_E7_SIM_DRYRUN.md, frozen protocol):

- **Speech power** of an utterance comes from its own thruster-off cut at the same
  session, position and channel (every content block is played in both states,
  adjacent in time, 6.5). It is the band power over speech-active samples, from an
  energy VAD on that cut: the same definition as the simulator's ``mix_at_snr``.
- **Noise power** comes from that position's noise-only take in the row's thruster
  state (``<pos>_noise`` for off, ``<pos>_noise_on`` for on), band power over the
  take minus ``trim_s`` at each end, over the 0.1 s frames whose 50-8000 Hz level is
  within ``NOISE_GATE_DB`` below the take's median frame level (amendment 2026-11-08:
  drops pre-roll and spin-up inside a thruster-on take).
- **Bands.** 500-8000 Hz (``snr_speechband_dB``), 50-8000 Hz (``snr_50_8000_dB``,
  reporting only) and the octaves 500-4000 Hz (``snr_octave_dB``).
- **Speech-level stability.** The drift anchor repeats the session's first block at
  the end. The median over its utterances of the 500-8000 Hz speech-active level of
  the anchor cut minus the original cut (same state, same VAD mask) is the session's
  level drift. When ``|drift| > max_drift_db`` every row of that session is flagged
  ``snr_level_drift``.

Speech and noise go through the same chain, so no absolute calibration is needed
(RESEARCH_PLAN.md 6.9).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from uwse.bench.schema import OCTAVE_BANDS
from uwse.uwsim.noise import _band, energy_vad

__all__ = [
    "SNR_BANDS",
    "MAX_DRIFT_DB",
    "NOISE_GATE_DB",
    "band_powers",
    "speech_powers",
    "noise_powers",
    "snr_fields",
    "level_drift_db",
]

MAX_DRIFT_DB = 1.0  # fixed before any real data (2026-11-08)
NOISE_GATE_DB = 6.0  # method amendment 2026-11-08 (S1 dry run), before any real data


def _bands() -> dict[str, tuple[float, float]]:
    out = {"500-8000": (500.0, 8000.0), "50-8000": (50.0, 8000.0)}
    for fc in OCTAVE_BANDS:
        f = float(fc)
        out[fc] = (f / math.sqrt(2.0), f * math.sqrt(2.0))
    return out


SNR_BANDS = _bands()


def band_powers(x: np.ndarray, fs: int, mask: np.ndarray | None = None,
                bands: dict[str, tuple[float, float]] = SNR_BANDS) -> dict[str, float]:
    """Mean square of the zero-phase band-passed ``x`` over ``mask`` (all samples if None)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.ones(x.size, bool) if mask is None else np.asarray(mask, bool)
    if not m.any():
        m = np.ones(x.size, bool)
    return {k: float(np.mean(_band(x, fs, b)[m] ** 2)) for k, b in bands.items()}


def speech_powers(speech: np.ndarray, fs: int, bands=SNR_BANDS) -> tuple[dict[str, float], np.ndarray]:
    """Band powers over the speech-active samples of a cut; returns (powers, VAD mask)."""
    mask = energy_vad(speech, fs)
    return band_powers(speech, fs, mask, bands), mask


def noise_powers(noise: np.ndarray, fs: int, trim_s: float = 1.0, bands=SNR_BANDS,
                 gate_db: float | None = NOISE_GATE_DB, frame_s: float = 0.1) -> dict[str, float]:
    """Band powers of a noise-only take, dropping ``trim_s`` at each end (operator handling).

    With ``gate_db`` (None = no gate), only ``frame_s`` frames whose 50-8000 Hz level is within ``gate_db``
    below the median frame level are kept, so pre-roll, spin-up or a late thruster start
    inside a thruster-on take do not pull the noise power down (S1 dry run, 2026-11-08).
    """
    noise = np.asarray(noise, dtype=np.float64)
    k = int(round(trim_s * fs))
    if noise.size > 4 * k:
        noise = noise[k:noise.size - k]
    mask = None
    if gate_db is not None:
        hop = max(1, int(round(frame_s * fs)))
        n_fr = noise.size // hop
        wb = _band(noise, fs, SNR_BANDS["50-8000"])[: n_fr * hop].reshape(n_fr, hop)
        lev = 10 * np.log10(np.mean(wb ** 2, axis=1) + 1e-30)
        keep = lev >= np.median(lev) - gate_db
        mask = np.zeros(noise.size, bool)
        mask[: n_fr * hop] = np.repeat(keep, hop)
    return band_powers(noise, fs, mask, bands)


def _db(ps: float, pn: float) -> float:
    return float(10.0 * np.log10((ps + 1e-30) / (pn + 1e-30)))


def snr_fields(ps: dict[str, float], pn: dict[str, float]) -> dict:
    """The 6.8 ``acoustic`` SNR fields plus score.py's two-band keys."""
    a = {"snr_speechband_dB": _db(ps["500-8000"], pn["500-8000"]),
         "snr_50_8000_dB": _db(ps["50-8000"], pn["50-8000"]),
         "snr_octave_dB": {fc: _db(ps[fc], pn[fc]) for fc in OCTAVE_BANDS}}
    return {"acoustic": a, "snr_db_500_8000": a["snr_speechband_dB"], "snr_db_50_8000": a["snr_50_8000_dB"]}


def level_drift_db(original: Sequence[np.ndarray], anchor: Sequence[np.ndarray], fs: int) -> float:
    """Median over utterances of the anchor cut's 500-8000 Hz speech-active level minus the
    original cut's, with the VAD mask of the original applied to both."""
    band = {"500-8000": SNR_BANDS["500-8000"]}
    d = []
    for x, y in zip(original, anchor):
        n = min(x.size, y.size)
        x, y = np.asarray(x[:n], float), np.asarray(y[:n], float)
        m = energy_vad(x, fs)
        d.append(_db(band_powers(y, fs, m, band)["500-8000"], band_powers(x, fs, m, band)["500-8000"]))
    return float(np.median(d)) if d else float("nan")
