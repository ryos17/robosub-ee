"""Level normalization of evaluation cuts before ASR.

Implements RESEARCH_PLAN.md §8.3 "Level normalization (fixes the
peak-normalization bug)":

- every cut is RMS-normalized to -26 dBFS over the speech-active region;
- speech activity comes from a VAD on the *aligned R0*, never on the system
  output, so every system is measured over the same samples;
- noise-only segments get the median gain of that system's speech utterances
  at the same position and condition (:func:`assign_gains`);
- silence / residual noise is never amplified to full scale (no peak
  normalization; an empty VAD mask yields no gain rather than a huge one);
- padding is 0.25 s either side, and the same cut feeds every system (the
  padding itself is applied when cutting, see ``align.cut_utterances``).

The VAD defaults are uncalibrated starting points; §5.4 freezes the VAD
settings on v0-dev.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "TARGET_DBFS",
    "PAD_S",
    "VADConfig",
    "energy_vad",
    "active_rms_dbfs",
    "normalization_gain",
    "normalize_cut",
    "assign_gains",
]

TARGET_DBFS = -26.0
PAD_S = 0.25


@dataclass
class VADConfig:
    """Energy-VAD parameters (frozen on v0-dev, §5.4).

    A frame is active if its energy exceeds
    ``max(noise_floor + margin_db, peak - dynamic_range_db, abs_floor_dbfs)``
    where noise_floor / peak are the ``floor_percentile`` / ``peak_percentile``
    of the frame energies (percentiles make it robust to clicks and to long
    silences). Activity is extended by ``hangover_s`` after and ``preroll_s``
    before each active run; runs shorter than ``min_active_s`` are dropped
    and gaps shorter than ``min_gap_s`` are bridged.
    """

    frame_s: float = 0.020
    hop_s: float = 0.010
    floor_percentile: float = 10.0
    peak_percentile: float = 95.0
    margin_db: float = 12.0
    dynamic_range_db: float = 40.0
    abs_floor_dbfs: float = -70.0
    hangover_s: float = 0.20
    preroll_s: float = 0.05
    min_active_s: float = 0.05
    min_gap_s: float = 0.10


def _runs(mask: np.ndarray) -> np.ndarray:
    """(start, end) index pairs of True runs, end exclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1)


def energy_vad(x: np.ndarray, fs: int, config: VADConfig | None = None) -> np.ndarray:
    """Per-sample boolean speech-activity mask of ``x`` (run it on aligned R0)."""
    cfg = config or VADConfig()
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    flen = max(1, int(round(cfg.frame_s * fs)))
    hop = max(1, int(round(cfg.hop_s * fs)))
    if n == 0:
        return np.zeros(0, dtype=bool)
    n_frames = max(1, 1 + (n - flen) // hop) if n >= flen else 1
    starts = np.arange(n_frames) * hop
    # frame energies via cumulative sum (no framing copy)
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    ends = np.minimum(starts + flen, n)
    e = (cs[ends] - cs[starts]) / np.maximum(ends - starts, 1)
    e_db = 10.0 * np.log10(np.maximum(e, 1e-20))
    floor = np.percentile(e_db, cfg.floor_percentile)
    peak = np.percentile(e_db, cfg.peak_percentile)
    thr = max(floor + cfg.margin_db, peak - cfg.dynamic_range_db, cfg.abs_floor_dbfs)
    act = e_db > thr
    if not act.any():
        return np.zeros(n, dtype=bool)

    def frames(s: float) -> int:
        return int(round(s / cfg.hop_s))

    # bridge short gaps, then drop short bursts
    gap = frames(cfg.min_gap_s)
    for s, t in _runs(~act):
        if s > 0 and t < len(act) and t - s < gap:
            act[s:t] = True
    min_run = max(1, frames(cfg.min_active_s))
    for s, t in _runs(act):
        if t - s < min_run:
            act[s:t] = False
    # hangover / pre-roll on the sample grid
    mask = np.zeros(n, dtype=bool)
    hang, pre = int(round(cfg.hangover_s * fs)), int(round(cfg.preroll_s * fs))
    for s, t in _runs(act):
        a = max(0, starts[s] - pre)
        b = min(n, ends[t - 1] + hang)
        mask[a:b] = True
    return mask


def active_rms_dbfs(x: np.ndarray, mask: np.ndarray) -> float:
    """RMS level (dBFS, full scale = 1.0 amplitude) of ``x`` over ``mask``; -inf if empty."""
    n = min(len(x), len(mask))
    sel = np.asarray(x[:n], np.float64)[np.asarray(mask[:n], bool)]
    if sel.size == 0:
        return float("-inf")
    p = np.mean(sel * sel)
    return float(10.0 * np.log10(p)) if p > 0 else float("-inf")


def normalization_gain(
    x: np.ndarray,
    mask: np.ndarray,
    target_dbfs: float = TARGET_DBFS,
    max_gain_db: float = 60.0,
) -> float:
    """Linear gain putting the active-region RMS of ``x`` at ``target_dbfs``.

    Returns NaN when the mask is empty or the active region is digital
    silence: such cuts must get a gain from :func:`assign_gains` instead of
    being blown up to full scale. Gains above ``max_gain_db`` are clipped
    (guards against near-silent system outputs).
    """
    lvl = active_rms_dbfs(x, mask)
    if not np.isfinite(lvl):
        return float("nan")
    g_db = min(target_dbfs - lvl, max_gain_db)
    return float(10.0 ** (g_db / 20.0))


def normalize_cut(
    x: np.ndarray,
    r0_aligned: np.ndarray,
    fs: int,
    target_dbfs: float = TARGET_DBFS,
    vad: VADConfig | None = None,
    gain: float | None = None,
    max_gain_db: float = 60.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """RMS-normalize one system's cut using the VAD of the aligned R0 cut.

    ``x`` and ``r0_aligned`` are the same (padded) cut of the system output and
    of R0. If ``gain`` is given (noise-only segment, from :func:`assign_gains`)
    it is applied unchanged. Returns ``(y, gain, mask)``; ``gain`` is NaN and
    ``y`` is ``x`` unchanged if no speech was found and no gain was supplied.
    """
    mask = energy_vad(r0_aligned, fs, vad)
    if len(mask) < len(x):
        mask = np.concatenate([mask, np.zeros(len(x) - len(mask), bool)])
    mask = mask[: len(x)]
    g = normalization_gain(x, mask, target_dbfs, max_gain_db) if gain is None else float(gain)
    y = np.asarray(x, np.float64) * g if np.isfinite(g) else np.asarray(x, np.float64)
    return y, g, mask


def assign_gains(
    rows: Iterable[dict],
    group_keys: Sequence[str] = ("system", "position", "condition"),
    kind_key: str = "kind",
    gain_key: str = "gain",
) -> list[dict]:
    """Fill the gain of noise-only rows with the median speech gain of their group.

    Each row is a dict with ``group_keys`` (system, position, condition),
    ``kind`` ("speech" or "noise") and, for speech rows, ``gain`` (linear).
    Noise rows get ``median(gain of speech rows in the same group)`` (§8.3).
    NaN speech gains are ignored. Raises ``ValueError`` if a noise row's group
    has no valid speech gain. Returns new dicts; inputs are not modified.
    """
    rows = [dict(r) for r in rows]
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get(kind_key, "speech") == "speech":
            g = r.get(gain_key)
            if g is not None and np.isfinite(g):
                groups[tuple(r[k] for k in group_keys)].append(float(g))
    for r in rows:
        if r.get(kind_key, "speech") != "speech":
            key = tuple(r[k] for k in group_keys)
            if not groups.get(key):
                raise ValueError(f"no speech gains for group {key} to assign noise-only gain")
            r[gain_key] = float(np.median(groups[key]))
            r["gain_source"] = "median_speech_gain"
    return rows
