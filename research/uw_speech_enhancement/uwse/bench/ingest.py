"""Import a pool recording of a session file: chirp sync, clock drift, utterance cuts.

A take is one recorder file that captured one played session file
(``<pos>_speech.wav`` or ``anchor_<pos>.wav`` from ``scripts/make_session_audio.py``),
started at an unknown time before playback and stopped after it. Every block in
the session file carries its own start and end chirp (RESEARCH_PLAN.md 6.5),
so each block is aligned on its own:

1. **Coarse offset.** The matched-filter envelope of the sync channel is
   computed in chunks and max-pooled to ``hop_s``. It is correlated with an
   impulse train at the nominal chirp onsets of the whole session file, after
   widening each impulse by the largest drift expected over the take, so a
   single missed chirp does not move the offset.
2. **Per-block chirps.** Around each block's predicted position the two
   chirps are detected (`uwse.evaluation.align.detect_chirps`). A chirp that is
   missing, far from its prediction or below the PSR threshold is replaced by
   the prediction from the take-level clock fit, and the block is flagged
   ``align_low_psr``.
3. **Clock drift.** Each block's drift comes from its own chirp interval when
   both chirps are good, otherwise from the take-level fit of chirp onset vs
   nominal time (least squares over all good chirps).
4. **Cuts.** Each recorded channel is resampled onto the playback clock
   anchored at the block's first chirp (`correct_drift`) and every utterance
   is cut with ``pad_s`` padding either side (`cut_utterances`).

The sync channel is the loopback input when one was recorded (same ADC clock as
the hydrophones, high PSR) and otherwise the hydrophone itself. With a
loopback, cuts include the acoustic delay (a few ms), which the 0.25 s padding
covers; the fine lag against R1 (6.5 step 3) is a scoring-time step.

Recordings at a rate other than the playback rate are first resampled to it
with an exact rational ratio; raw sample indices in the output refer to the
original file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import correlate, hilbert

from uwse import audio
from uwse.evaluation import align as _align

__all__ = [
    "SessionBlock",
    "load_session",
    "envelope_pooled",
    "coarse_offset",
    "align_take",
]


@dataclass
class SessionBlock:
    """One block of a played session file, with its timing JSON."""

    session_order_idx: int
    block_id: int
    content_block: str
    thruster_state: str
    anchor_repeat: bool
    start_sample: int  # in the session file, at the playback rate
    timing: dict

    @property
    def chirp_starts(self) -> list[int]:
        return [self.start_sample + int(c["start"]) for c in self.timing["chirps"]]

    @property
    def layout(self) -> _align.BlockLayout:
        return _align.layout_from_timing(self.timing)


def load_session(bench: str | Path, position: str, anchor: bool = False) -> tuple[int, list[SessionBlock]]:
    """Blocks of ``session/<pos>_speech.json`` (or the anchor file) with their timing JSONs.

    Returns ``(fs, blocks)``. ``block_id`` and ``anchor_repeat`` come from
    ``schedule.json``.
    """
    bench = Path(bench)
    sched = json.loads((bench / "schedule.json").read_text(encoding="utf-8"))
    by_order = {e["session_order_idx"]: e for e in sched}

    def timing(cb: str) -> dict:
        return json.loads((bench / "blocks" / f"{cb}.json").read_text(encoding="utf-8"))

    if anchor:
        e = next((e for e in sched if e["anchor_repeat"] and e["position_id"] == position), None)
        if e is None:
            raise ValueError(f"no anchor repeat scheduled at {position}")
        t = timing(e["content_block"])
        return int(t["fs"]), [SessionBlock(e["session_order_idx"], e["block_id"], e["content_block"],
                                           e["thruster_state"], True, 0, t)]
    idx = json.loads((bench / "session" / f"{position}_speech.json").read_text(encoding="utf-8"))
    blocks = []
    for b in idx["blocks"]:
        e = by_order[b["session_order_idx"]]
        if e["content_block"] != b["content_block"]:
            raise ValueError(f"session index and schedule disagree at order {b['session_order_idx']}")
        blocks.append(SessionBlock(b["session_order_idx"], e["block_id"], b["content_block"],
                                   b["thruster_state"], bool(e["anchor_repeat"]), int(b["start_sample"]),
                                   timing(b["content_block"])))
    return int(idx["fs"]), blocks


def envelope_pooled(x: np.ndarray, template: np.ndarray, hop: int, chunk: int = 1 << 21) -> np.ndarray:
    """Matched-filter envelope of ``x`` (onset-indexed), max-pooled over ``hop`` samples.

    Computed in overlapping chunks so a long take never needs one FFT of its
    full length. Element ``j`` is the max envelope over onsets ``[j*hop, (j+1)*hop)``.
    """
    n_on = len(x) - len(template) + 1
    if n_on <= 0:
        return np.zeros(0)
    out = np.zeros(int(math.ceil(n_on / hop)))
    chunk = max(chunk - chunk % hop, hop)
    edge = len(template)  # analytic-signal edge effects
    for s in range(0, n_on, chunk):
        a = max(0, s - edge)
        b = min(len(x), s + chunk + len(template) - 1 + edge)
        c = correlate(x[a:b], template, mode="valid", method="fft")
        env = np.abs(hilbert(c))[s - a: s - a + min(chunk, n_on - s)]
        m = len(env) // hop * hop
        j = s // hop
        if m:
            out[j: j + m // hop] = env[:m].reshape(-1, hop).max(axis=1)
        if len(env) > m:
            out[j + m // hop] = env[m:].max()
    return out


def coarse_offset(env: np.ndarray, hop: int, chirp_starts: Sequence[int], max_drift_ppm: float) -> int:
    """Take sample index of session sample 0, to within about ``hop`` plus drift.

    The pooled envelope is widened by the drift possible over the session
    length, then correlated with impulses at the nominal chirp onsets.
    """
    starts = np.asarray(chirp_starts, dtype=np.int64)
    span = int(starts.max() - starts.min()) if len(starts) else 0
    w = 2 * int(math.ceil(max_drift_ppm * 1e-6 * span / hop)) + 1
    e = maximum_filter1d(env, size=w, mode="constant") if w > 1 else env
    e = e / (np.median(e) + 1e-12)
    imp = np.zeros(int(starts.max() // hop) + 1)
    imp[(starts // hop).astype(np.int64)] = 1.0
    if len(e) < len(imp):
        raise ValueError("recording is shorter than the session file")
    score = correlate(e, imp, mode="valid", method="fft")
    return int(np.argmax(score)) * hop


@dataclass
class _Chirp:
    nominal: int  # session sample
    found: float | None  # take sample (playback rate), None if rejected
    psr_db: float
    predicted: float


@dataclass
class BlockResult:
    block: SessionBlock
    chirps: list[_Chirp]
    chirp1: float  # take sample of the first chirp onset used for the cuts
    drift_ppm: float
    drift_source: str  # "block" or "take_fit"
    flags: list[str] = field(default_factory=list)


def _fit_clock(pairs: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares take = a + r * session; returns (a, r)."""
    s = np.array([p[0] for p in pairs], dtype=np.float64)
    t = np.array([p[1] for p in pairs], dtype=np.float64)
    if len(pairs) == 1 or np.ptp(s) == 0:
        return float(t[0] - s[0]), 1.0
    r, a = np.polyfit(s, t, 1)
    return float(a), float(r)


def align_take(
    streams: np.ndarray,
    fs: int,
    blocks: Sequence[SessionBlock],
    sync: int = 0,
    max_drift_ppm: float = 200.0,
    search_s: float = 0.5,
    psr_threshold_db: float = 6.0,
    pad_s: float = 0.25,
    hop_s: float = 0.01,
) -> tuple[list[BlockResult], dict[str, Any]]:
    """Align every block of a take. ``streams`` is ``(C, T)`` at the playback rate ``fs``.

    ``sync`` is the row used for chirp detection. Returns per-block results
    (chirps, drift, flags) and take-level QC (offset, clock fit). Cutting is
    done by :func:`cut_block`.
    """
    streams = np.atleast_2d(np.asarray(streams, dtype=np.float64))
    x = streams[sync]
    ch0 = blocks[0].timing["chirp"]
    f_lo, f_hi, dur = float(ch0["f0_hz"]), float(ch0["f1_hz"]), float(ch0["duration_s"])
    template = _align.lfm_chirp(fs, f_lo, f_hi, dur)
    hop = max(1, int(round(hop_s * fs)))
    all_starts = [s for b in blocks for s in b.chirp_starts]
    offset = coarse_offset(envelope_pooled(x, template, hop), hop, all_starts, max_drift_ppm)

    # chirp detection around each prediction; the clock fit is refined as good chirps accrue
    pairs: list[tuple[int, float]] = []
    results: list[BlockResult] = []
    win = int(round(search_s * fs))
    for b in blocks:
        chirps = []
        for s_nom in b.chirp_starts:
            a, r = _fit_clock(pairs) if pairs else (float(offset), 1.0)
            pred = a + r * s_nom
            lo = max(0, int(pred) - win)
            hi = min(len(x), int(pred) + win + len(template))
            found, psr = None, float("-inf")
            if hi - lo >= len(template):
                d = _align.detect_chirps(x[lo:hi], fs, 1, f_lo, f_hi, dur, template=template)
                if d:
                    psr = d[0].psr_db
                    t = lo + d[0].frac_index
                    if psr >= psr_threshold_db and abs(t - pred) <= win:
                        found = t
            chirps.append(_Chirp(s_nom, found, psr, pred))
            if found is not None:
                pairs.append((s_nom, found))
        results.append(BlockResult(b, chirps, 0.0, 0.0, ""))

    if not pairs:
        raise RuntimeError("no chirp passed the PSR check; wrong file, channel or chirp band?")
    a, r = _fit_clock(pairs)
    take_drift = (r - 1.0) * 1e6 if len(pairs) > 1 else 0.0
    for res in results:
        c1, c2 = res.chirps
        if c1.found is not None and c2.found is not None:
            res.drift_ppm = _align.estimate_drift_ppm(c1.found, c2.found, c2.nominal - c1.nominal)
            res.drift_source = "block"
        else:
            res.drift_ppm = take_drift
            res.drift_source = "take_fit"
            res.flags.append("align_low_psr")
        if c1.found is not None:
            res.chirp1 = c1.found
        elif c2.found is not None:  # back off from the end chirp on the block's clock
            res.chirp1 = c2.found - (c2.nominal - c1.nominal) * (1.0 + res.drift_ppm * 1e-6)
        else:
            res.chirp1 = a + r * c1.nominal
    resid = [f - (a + r * s) for s, f in pairs]
    qc = {
        "fs": fs,
        "coarse_offset": offset,
        "clock_offset": a,
        "take_drift_ppm": take_drift,
        "n_chirps": len(all_starts),
        "n_chirps_good": len(pairs),
        "fit_resid_max_samples": float(np.max(np.abs(resid))) if resid else 0.0,
    }
    return results, qc


def cut_block(streams: np.ndarray, fs: int, res: BlockResult, pad_s: float = 0.25) -> tuple[list[np.ndarray], np.ndarray]:
    """Drift-correct each row of ``streams`` around the block and cut its utterances.

    Returns ``cuts[c][k]`` (channel c, utterance k) and the ``(K, 2)`` take
    sample bounds of each cut (padding included, playback rate).
    """
    streams = np.atleast_2d(np.asarray(streams, dtype=np.float64))
    lay = res.block.layout
    r = 1.0 + res.drift_ppm * 1e-6
    first = res.chirp1 - lay.chirp1_start_s * fs * r  # take sample of block start
    a = max(0, int(math.floor(first)) - int(pad_s * fs) - 64)
    b = min(streams.shape[1], int(math.ceil(first + lay.total_s * fs * r)) + int(pad_s * fs) + 64)
    anchor = res.chirp1 - a
    cuts = []
    bounds = None
    for row in streams:
        seg = _align.correct_drift(row[a:b], res.drift_ppm, anchor=anchor)
        c, bd = _align.cut_utterances(seg, fs, lay, anchor, 0.0, pad_s)
        cuts.append(c)
        if bounds is None:  # back to take samples on the recording clock
            bounds = np.rint(a + anchor + (bd - anchor) * r).astype(np.int64)
    return cuts, bounds


def load_take(path: str | Path, fs_target: int) -> tuple[np.ndarray, int, int]:
    """Read a take as ``(C, T)`` at ``fs_target``; returns (streams, fs_raw, bits)."""
    import soundfile as sf

    info = sf.info(str(path))
    x, fs_raw = audio.load(str(path), always_2d=True)
    if fs_raw != fs_target:
        x = audio.resample(x, fs_raw, fs_target, axis=-1)
    bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}.get(info.subtype, 0)
    return x, fs_raw, bits
