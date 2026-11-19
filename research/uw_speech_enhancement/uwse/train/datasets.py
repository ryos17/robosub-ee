"""Training-pair datasets for fine-tuning (RESEARCH_PLAN.md §5.3, §7.3, §8.2).

Every dataset here yields ``(noisy[T], target[T], meta)`` with float32 tensors at
the model rate, so `uwse.train.loop` never knows where pairs came from:

- `SimPairsDataset`   C2/C3/C10: thin wrapper around `uwse.uwsim.dataset.UWSimPairs`.
- `InAirPairs`        C7 in-air control: pyroomacoustics ShoeBox RIRs with c = 343 m/s
                      and T60 drawn from the *pool* distribution; same noise, same
                      SNR rule and the same target definition as uwsim.
- `ReplayPairs`       C9: measured IRs from calibration/validation positions (never
                      benchmark positions, §5.3), perturbed by time-stretch ±10 %,
                      per-octave gain ±2 dB and an optional tail resynthesis hook.
- `SyntheticPairs`    harmonic "speech" + reverb + noise for tests and the W4 smoke run.
- `SegmentDataset`    random (training) or deterministic (validation) crops to a fixed
                      segment length, keyed by ``(index, crop_seed)`` so the stream is
                      resumable (`uwse.train.loop.StepBatchSampler`).
- `Resampled`         rate conversion to ``adapter.sample_rate``.
- `build_val_set`     fixed-seed validation set (sim-val), optionally frozen to WAVs.

Target definition (§8.2): R1 = s * h_chain, aligned to the direct-path arrival of the
noisy signal. When no chain FIR is supplied the target is s itself (identity chain,
i.e. R0); pass the measured h_chain once E2 exists.

Determinism: item ``i`` of a pair dataset depends only on ``(seed, i)``, so the
validation set and a resumed run are reproducible.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from scipy.signal import fftconvolve, resample_poly
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from uwse import audio

__all__ = [
    "list_audio_files",
    "SegmentDataset",
    "Resampled",
    "SimPairsDataset",
    "InAirPairs",
    "ReplayPairs",
    "SyntheticPairs",
    "WavPairs",
    "build_val_set",
    "collate_pairs",
    "direct_path_index",
    "time_stretch_ir",
    "random_band_gain",
    "resynth_tail",
    "mix_pair",
]

Key = int | tuple[int, int]
NoiseFn = Callable[[int, int, np.random.Generator], np.ndarray]


def _split_key(key: Key) -> tuple[int, int | None]:
    if isinstance(key, (tuple, list)):
        return int(key[0]), int(key[1])
    return int(key), None


def _to_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x.detach().float().cpu()
    return torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist() if v.size <= 16 else f"<array {v.shape}>"
        elif isinstance(v, Tensor):
            v = v.tolist() if v.numel() <= 16 else f"<tensor {tuple(v.shape)}>"
        elif isinstance(v, dict):
            v = _jsonable(v)
        elif not isinstance(v, (str, int, float, bool, list, tuple, type(None))):
            v = str(v)
        out[k] = v
    return out


def list_audio_files(spec: str | Sequence[str], exts: tuple[str, ...] = (".wav", ".flac")) -> list[str]:
    """Expand glob pattern(s), directories (recursive) or a .txt list into sorted files."""
    specs = [spec] if isinstance(spec, str) else list(spec)
    files: list[str] = []
    for s in specs:
        if s.endswith(".txt") and os.path.isfile(s):
            with open(s, encoding="utf-8") as f:
                files += [ln.strip() for ln in f if ln.strip()]
        elif os.path.isdir(s):
            for e in exts:
                files += glob.glob(os.path.join(s, "**", "*" + e), recursive=True)
        else:
            files += [p for p in glob.glob(s, recursive=True) if p.lower().endswith(exts)]
    return sorted(set(files))


# ---------------------------------------------------------------------------
# cropping, resampling, collate
# ---------------------------------------------------------------------------
class SegmentDataset(Dataset):
    """Crop (noisy, target) pairs of `base` to exactly `n_samples` (zero-padding short ones).

    ``ds[(i, crop_seed)]`` takes a random offset from ``default_rng(crop_seed)``
    (training); ``ds[i]`` takes a deterministic offset from ``default_rng(i)``
    (validation), so validation crops never change between evaluations.
    """

    def __init__(self, base: Dataset, n_samples: int | None) -> None:
        self.base, self.n = base, n_samples

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, key: Key) -> tuple[Tensor, Tensor, dict]:
        i, crop_seed = _split_key(key)
        item = self.base[i]
        noisy, target = _to_tensor(item[0]), _to_tensor(item[1])
        meta = dict(item[2]) if len(item) > 2 else {}
        if self.n is None:
            return noisy, target, meta
        T = min(noisy.shape[-1], target.shape[-1])
        noisy, target = noisy[..., :T], target[..., :T]
        if T > self.n:
            rng = np.random.default_rng(crop_seed if crop_seed is not None else [i, 7])
            off = int(rng.integers(0, T - self.n + 1))
            noisy, target = noisy[..., off:off + self.n], target[..., off:off + self.n]
            meta["crop_offset"] = off
        elif T < self.n:
            pad = self.n - T
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            target = torch.nn.functional.pad(target, (0, pad))
        return noisy, target, meta


def collate_pairs(items: list[tuple[Tensor, Tensor, dict]]) -> tuple[Tensor, Tensor, list[dict]]:
    """Stack equal-length items into (noisy[B,T], target[B,T], metas)."""
    noisy = torch.stack([it[0] for it in items])
    target = torch.stack([it[1] for it in items])
    return noisy, target, [it[2] for it in items]


class Resampled(Dataset):
    """Resample the (noisy, target) of `base` from `fs_in` to `fs_out` (polyphase)."""

    def __init__(self, base: Dataset, fs_in: int, fs_out: int) -> None:
        self.base, self.fs_in, self.fs_out = base, int(fs_in), int(fs_out)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, key: Key):
        item = self.base[key]
        if self.fs_in == self.fs_out:
            return item
        conv = []
        for x in item[:2]:
            x = x.numpy() if isinstance(x, Tensor) else np.asarray(x)
            conv.append(_to_tensor(audio.resample(x.astype(np.float64), self.fs_in, self.fs_out)))
        meta = dict(item[2]) if len(item) > 2 else {}
        meta["fs"] = self.fs_out
        return conv[0], conv[1], meta


# ---------------------------------------------------------------------------
# IR helpers (C7, C9)
# ---------------------------------------------------------------------------
def direct_path_index(h: np.ndarray, fs: int, rel: float = 0.5, search_s: float = 0.001) -> int:
    """Index of the direct-path peak: first sample above ``rel`` x max, then the local max
    within ``search_s`` after it (robust to a later reflection being the global max)."""
    a = np.abs(h)
    m = a.max()
    if m <= 0:
        return 0
    i0 = int(np.argmax(a >= rel * m))
    w = max(1, int(search_s * fs))
    return i0 + int(np.argmax(a[i0:i0 + w]))


def time_stretch_ir(h: np.ndarray, factor: float, max_den: int = 200) -> np.ndarray:
    """h'(t) = h(t / factor): all delays and decay times scale by ``factor``.

    Implemented as polyphase resampling by the rational approximation of ``factor``
    (so the spectrum also scales by 1/factor, as for a uniformly scaled room).
    """
    if abs(factor - 1.0) < 1e-9:
        return np.asarray(h, dtype=np.float64).copy()
    r = Fraction(float(factor)).limit_denominator(max_den)
    return resample_poly(np.asarray(h, dtype=np.float64), r.numerator, r.denominator)


def random_band_gain(h: np.ndarray, fs: int, rng: np.random.Generator, max_db: float = 2.0,
                     centres: Sequence[float] = (125, 250, 500, 1000, 2000, 4000, 8000)) -> np.ndarray:
    """Independent U(-max_db, max_db) gain at each octave centre, interpolated linearly in
    dB over log-frequency (constant beyond the end centres), applied as a zero-phase
    spectral weight. Smooth across bands, so no filterbank ripple is introduced."""
    h = np.asarray(h, dtype=np.float64)
    c = np.asarray([f for f in centres if f < fs / 2], dtype=np.float64)
    g_db = rng.uniform(-max_db, max_db, c.size)
    nfft = int(2 ** np.ceil(np.log2(2 * h.size)))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    lf = np.log2(np.maximum(f, 1.0))
    gain = 10.0 ** (np.interp(lf, np.log2(c), g_db) / 20.0)
    return np.fft.irfft(np.fft.rfft(h, nfft) * gain, nfft)[: h.size]


def _band_t60(band: np.ndarray, fs: int, lo_db: float = -5.0, hi_db: float = -25.0) -> float | None:
    """T60 from a linear fit to the Schroeder EDC between lo_db and hi_db (T20 style)."""
    e = np.cumsum(band[::-1] ** 2)[::-1]
    if e[0] <= 0:
        return None
    edc = 10.0 * np.log10(np.maximum(e / e[0], 1e-30))
    m = (edc <= lo_db) & (edc >= hi_db)
    if m.sum() < 8:
        return None
    t = np.nonzero(m)[0] / fs
    slope = np.polyfit(t, edc[m], 1)[0]  # dB/s, negative
    return float(-60.0 / slope) if slope < 0 else None


def resynth_tail(h: np.ndarray, fs: int, rng: np.random.Generator, t_e: float = 0.04,
                 fallback_t60: float = 0.5) -> np.ndarray:
    """Replace the measured tail after t_e (re the direct path) by a statistical tail.

    Per octave (250 Hz-8 kHz), T60 is fitted on the measured late part (Schroeder,
    -5..-25 dB) and `uwse.uwsim.tail.splice_tail` energy-matches Gaussian noise with
    that decay to the response at t_e (§4.2 row 7). Keeps the measured early part,
    randomizes the late fine structure: a new tail realization per call.
    """
    from uwse.uwsim.tail import octave_filterbank, splice_tail
    from uwse.uwsim.physics import octave_bands

    h = np.asarray(h, dtype=np.float64)
    p = direct_path_index(h, fs)
    n_e = p + int(round(t_e * fs))
    if h.size <= n_e + int(0.03 * fs):
        return h.copy()
    centres = octave_bands(250.0, min(8000.0, fs / 2.0 / np.sqrt(2.0)))[0]
    late = h[n_e:]
    bands = octave_filterbank(late, fs, centres)
    t60 = np.array([_band_t60(b, fs) or np.nan for b in bands])
    if np.all(np.isnan(t60)):
        t60[:] = fallback_t60
    else:
        t60[np.isnan(t60)] = np.nanmedian(t60)
    t60 = np.clip(t60, 0.05, 5.0)
    out = splice_tail(h[:n_e], fs, n_e / fs, t60, rng, h.size, centres)
    return out


# ---------------------------------------------------------------------------
# mixing (shared by C7 / C9 so both follow the uwsim recipe)
# ---------------------------------------------------------------------------
def _as_noise_fn(noise: Any, fs: int, split: str = "train", kind: Any = None) -> NoiseFn:
    """Accept a `uwse.uwsim.noise.NoiseBank`, a callable (n, fs, rng) -> x, a list of
    arrays at ``fs``, or None (pink-ish synthetic noise, smoke tests only)."""
    if noise is None:
        from uwse.uwsim.noise import synthetic_noise
        return lambda n, fs_, rng: synthetic_noise(n, fs_, rng)
    if hasattr(noise, "draw"):
        if getattr(noise, "fs", fs) != fs:
            noise.fs = fs  # NoiseBank resamples each draw to its .fs
        return lambda n, fs_, rng: noise.draw(n, rng, kind=kind, split=split)[0]
    if callable(noise):
        return noise
    arrays = [np.asarray(a, dtype=np.float64) for a in noise]

    def draw(n: int, fs_: int, rng: np.random.Generator) -> np.ndarray:
        a = arrays[int(rng.integers(len(arrays)))]
        if a.size < n:
            a = np.tile(a, int(np.ceil(n / a.size)) + 1)
        off = int(rng.integers(0, a.size - n + 1))
        return a[off:off + n]

    return draw


def mix_pair(clean: np.ndarray, h_room: np.ndarray, noise: np.ndarray, snr_db: float, fs: int,
             level_dbfs: float, chain: Callable[[np.ndarray], np.ndarray] | None = None,
             pre_roll: int = 0, direct_index: int | None = None,
             ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """noisy = (s*h_chain)*h_room + g*noise, target = s*h_chain aligned to the direct path.

    ``clean`` may carry ``pre_roll`` extra samples in front so the reverberant segment
    contains the decay of the preceding speech; they are dropped from both outputs.
    ``direct_index``: the direct-path sample of ``h_room`` when it is known from
    geometry (simulated IRs; the same rule as ``uwse.uwsim.simulate.direct_path_index``
    mode "geometric"). ``None`` falls back to the peak picker, for measured IRs.
    SNR is the speech-band, speech-active SNR of `uwse.uwsim.noise.mix_at_snr` (the
    uwsim rule, §4.2 row 18). Both outputs get one common gain so that the noisy RMS
    is ``level_dbfs``; both are scaled down together if the noisy peak exceeds 0.99.
    """
    from uwse.uwsim.noise import mix_at_snr

    s_tx = chain(clean) if chain is not None else np.asarray(clean, dtype=np.float64)
    p = int(direct_index) if direct_index is not None else direct_path_index(h_room, fs)
    guard = min(p, 64)  # keep the acausal half of the fractional-delay sinc
    h = h_room[p - guard:]
    rev = fftconvolve(s_tx, h)[: s_tx.size]
    target = np.concatenate([np.zeros(guard), s_tx])[: s_tx.size]  # delayed by guard: aligned
    rev, target = rev[pre_roll:], target[pre_roll:]
    noisy, _ = mix_at_snr(rev, noise[: rev.size], snr_db, fs)
    r = np.sqrt(np.mean(noisy ** 2))
    g = 10.0 ** (level_dbfs / 20.0) / r if r > 0 else 1.0
    peak = np.max(np.abs(noisy)) * g
    if peak > 0.99:
        g *= 0.99 / peak
    return noisy * g, target * g, {"gain_db": float(20 * np.log10(g)), "align_guard": guard}


class _PairBase(Dataset):
    """Common sampling of clean speech, noise, SNR and level for C7/C9 (and the tests).

    ``clean``: list of audio files, or list of arrays at ``fs_clean``. Item i uses
    ``default_rng([seed, i])`` for every random choice. ``segment_s=None`` returns
    whole utterances (needed for Whisper-LoRA, whose transcripts cover the file).
    """

    def __init__(self, clean: Sequence[str | np.ndarray], noise: Any, fs_model: int, *,
                 fs_sim: int | None = None, n_items: int = 100_000, seed: int = 0,
                 segment_s: float | None = 4.0, snr_range: tuple[float, float] = (-5.0, 20.0),
                 level_dbfs_range: tuple[float, float] = (-35.0, -15.0),
                 chain: Callable[[np.ndarray], np.ndarray] | np.ndarray | None = None,
                 fs_clean: int = 16000, noise_split: str = "train", noise_kind: Any = None) -> None:
        if len(clean) == 0:
            raise ValueError("no clean speech given")
        self.clean = list(clean)
        self.fs_model = int(fs_model)
        self.fs = int(fs_sim or fs_model)
        self.n_items, self.seed, self.segment_s = int(n_items), int(seed), segment_s
        self.snr_range, self.level_range = snr_range, level_dbfs_range
        self.fs_clean = fs_clean
        # an FIR is stored as an array and applied by a bound method (not a lambda), so the
        # dataset pickles into spawn-started DataLoader workers (Windows)
        self._h_chain: np.ndarray | None = None
        if chain is not None and not callable(chain):
            self._h_chain = np.asarray(chain, dtype=np.float64)
            self.chain = self._apply_chain_fir
        else:
            self.chain = chain
        self._noise = noise
        self._noise_split, self._noise_kind = noise_split, noise_kind
        self._noise_fn: NoiseFn | None = None

    def __len__(self) -> int:
        return self.n_items

    def __getstate__(self) -> dict[str, Any]:
        st = self.__dict__.copy()
        st["_noise_fn"] = None  # closure; rebuilt lazily in the worker
        return st

    def _apply_chain_fir(self, s: np.ndarray) -> np.ndarray:
        return fftconvolve(np.asarray(s, dtype=np.float64), self._h_chain)[: np.size(s)]

    def _noise_draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self._noise_fn is None:  # lazily, so the dataset pickles into workers cheaply
            self._noise_fn = _as_noise_fn(self._noise, self.fs, self._noise_split, self._noise_kind)
        return np.asarray(self._noise_fn(n, self.fs, rng), dtype=np.float64)

    def _load_clean(self, j: int) -> tuple[np.ndarray, str]:
        c = self.clean[j]
        if isinstance(c, (str, os.PathLike)):
            x, fs = audio.load(str(c))
            if x.ndim > 1:
                x = x[0]
            name = str(c)
        else:
            x, fs, name = np.asarray(c, dtype=np.float64), self.fs_clean, f"array{j}"
        return audio.resample(x, fs, self.fs), name

    def _room_ir(self, rng: np.random.Generator, i: int) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def __getitem__(self, key: Key) -> tuple[Tensor, Tensor, dict]:
        i, _ = _split_key(key)
        rng = np.random.default_rng([self.seed, i])
        j = int(rng.integers(len(self.clean)))
        s, name = self._load_clean(j)
        h, meta = self._room_ir(rng, i)
        pre = 0
        if self.segment_s is not None:
            n = int(round(self.segment_s * self.fs))
            pre = min(h.size, s.size)  # reverberation of earlier speech enters the segment
            if s.size > n:
                off = int(rng.integers(0, s.size - n + 1))
                lo = max(0, off - pre)
                pre = off - lo
                s = s[lo:off + n]
            else:
                # as uwsim: zero-pad the clean speech to the segment, then simulate, so the
                # noise (and the reverberant decay) run over the whole segment
                pre = 0
                s = np.pad(s, (0, n - s.size))
        noise = self._noise_draw(s.size - pre, rng)
        snr = float(rng.uniform(*self.snr_range))
        lvl = float(rng.uniform(*self.level_range))
        noisy, target, mix_meta = mix_pair(s, h, noise, snr, self.fs, lvl, self.chain, pre,
                                           direct_index=meta.get("direct_index"))
        if self.fs != self.fs_model:
            noisy = audio.resample(noisy, self.fs, self.fs_model)
            target = audio.resample(target, self.fs, self.fs_model)
        utt = os.path.splitext(os.path.basename(name))[0]
        meta.update(mix_meta, snr_db=snr, level_dbfs=lvl, clean_file=name, utt_id=utt,
                    fs=self.fs_model, index=i)
        return _to_tensor(noisy), _to_tensor(target), meta


# ---------------------------------------------------------------------------
# C7: in-air control
# ---------------------------------------------------------------------------
class InAirPairs(_PairBase):
    """C7: same recipe as uwsim but with in-air shoebox RIRs (pyroomacoustics, c = 343 m/s).

    T60 is drawn from the *pool* distribution: either ``t60_samples`` (measured pool
    T60s, preferred once E3 exists) or U(``t60_range``). The default 0.25-1.0 s is the
    uncalibrated pool range quoted in §7.1 (WPE row), not a measurement.
    Wall absorption follows from Sabine's inverse for the drawn room and T60
    (`pyroomacoustics.inverse_sabine`, frequency-flat, all walls equal, no sign flip:
    exactly what an in-air simulator gives). ISM order is capped at ``max_order`` for
    speed and, if ``synth_tail``, the response after ``t_e`` is replaced by the same
    energy-matched exponential tail uwsim uses (§4.2 row 7), so the decay reaches the
    drawn T60 regardless of the cap. Room sizes default to the generic-pool ranges of
    §4.2 row 3; distances to §4.2 row 13.

    RIRs are cached: item i uses RIR number ``rng.integers(n_rirs)``, generated on first
    use with ``default_rng([seed, 1, k])``.
    """

    def __init__(self, clean: Sequence[str | np.ndarray], noise: Any, fs_model: int, *,
                 t60_range: tuple[float, float] = (0.25, 1.0),
                 t60_samples: Sequence[float] | None = None,
                 room_dim_range: Sequence[tuple[float, float]] = ((15.0, 50.0), (8.0, 25.0), (1.2, 5.0)),
                 distance_range: tuple[float, float] = (0.3, 25.0), c: float = 343.0,
                 max_order: int = 12, t_e: float = 0.04, synth_tail: bool = True,
                 n_rirs: int = 512, margin: float = 0.3, **kw: Any) -> None:
        super().__init__(clean, noise, fs_model, **kw)
        self.t60_range, self.t60_samples = t60_range, t60_samples
        self.room_dim_range = [tuple(r) for r in room_dim_range]
        self.distance_range, self.c = distance_range, float(c)
        self.max_order, self.t_e, self.synth_tail = int(max_order), float(t_e), synth_tail
        self.n_rirs, self.margin = int(n_rirs), float(margin)
        self._cache: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}

    def _draw_t60(self, rng: np.random.Generator) -> float:
        if self.t60_samples is not None and len(self.t60_samples):
            return float(self.t60_samples[int(rng.integers(len(self.t60_samples)))])
        return float(rng.uniform(*self.t60_range))

    def make_rir(self, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        import pyroomacoustics as pra

        for _ in range(50):
            dims = np.array([rng.uniform(*r) for r in self.room_dim_range])
            t60 = self._draw_t60(rng)
            try:
                e_abs, order = pra.inverse_sabine(t60, dims, c=self.c)
            except ValueError:  # T60 unreachable in this room (absorption > 1): redraw
                continue
            lo, hi = self.margin, dims - self.margin
            if np.any(hi <= lo):
                continue
            for _ in range(50):
                src, mic = rng.uniform(lo, hi), rng.uniform(lo, hi)
                d = float(np.linalg.norm(src - mic))
                if self.distance_range[0] <= d <= self.distance_range[1]:
                    break
            else:
                continue
            pra.constants.set("c", self.c)
            room = pra.ShoeBox(dims, fs=self.fs, materials=pra.Material(float(e_abs)),
                               max_order=int(min(order, self.max_order)), air_absorption=False)
            room.add_source(src)
            room.add_microphone(mic)
            room.compute_rir()
            h = np.asarray(room.rir[0][0], dtype=np.float64)
            # geometric direct-path index (as uwsim): pyroomacoustics delays every
            # image by frac_delay_length // 2 samples (its fractional-delay centre)
            p = int(round(d / self.c * self.fs)) + pra.constants.get("frac_delay_length") // 2
            if self.synth_tail:
                from uwse.uwsim.tail import splice_tail

                n_len = p + int(round((t60 + self.t_e) * self.fs))
                n_e = p + int(round(self.t_e * self.fs))
                h = splice_tail(h[:n_e], self.fs, n_e / self.fs, t60, rng, n_len)
            meta = {"t60": t60, "room_dim": dims.round(3).tolist(), "distance": d,
                    "max_order": int(min(order, self.max_order)), "abs": float(e_abs),
                    "c": self.c, "direct_index": p}
            return h, meta
        raise RuntimeError("could not draw a valid in-air room; check t60/room ranges")

    def _room_ir(self, rng: np.random.Generator, i: int) -> tuple[np.ndarray, dict[str, Any]]:
        k = int(rng.integers(self.n_rirs))
        if k not in self._cache:
            self._cache[k] = self.make_rir(np.random.default_rng([self.seed, 1, k]))
        h, meta = self._cache[k]
        return h, {**meta, "rir_id": k}


# ---------------------------------------------------------------------------
# C9: measured-IR replay
# ---------------------------------------------------------------------------
class ReplayPairs(_PairBase):
    """C9: measured IRs (calibration + validation positions only, §5.3) with perturbations.

    ``irs``: a folder / glob / list of WAV IR files (multichannel files contribute
    each channel), or a list of arrays at ``fs_ir``. The IRs should be chain-
    compensated room responses; the chain is re-applied through ``chain`` so the
    target is R1 exactly as in uwsim. Files whose path contains any of
    ``exclude`` (e.g. "bench", "P1") are dropped: benchmark-position IRs must never
    enter training.

    Perturbations per item: time-stretch U(1 - stretch, 1 + stretch) (default ±10 %),
    independent per-octave gain U(±band_gain_db) (default ±2 dB), and with probability
    ``p_tail`` a tail resynthesis ``tail_fn(h, fs, rng)`` (default `resynth_tail`;
    pass uwsim's own tail model when it is frozen).
    """

    def __init__(self, clean: Sequence[str | np.ndarray], noise: Any, fs_model: int, *,
                 irs: str | Sequence[str] | Sequence[np.ndarray], fs_ir: int | None = None,
                 stretch: float = 0.10, band_gain_db: float = 2.0, p_tail: float = 0.5,
                 tail_fn: Callable[[np.ndarray, int, np.random.Generator], np.ndarray] | None = resynth_tail,
                 exclude: Sequence[str] = ("bench",), max_ir_s: float = 2.0, **kw: Any) -> None:
        super().__init__(clean, noise, fs_model, **kw)
        self.stretch, self.band_gain_db, self.p_tail = stretch, band_gain_db, p_tail
        self.tail_fn = tail_fn
        self.irs: list[np.ndarray] = []
        self.ir_names: list[str] = []
        if isinstance(irs, str) or (len(irs) and isinstance(irs[0], (str, os.PathLike))):
            files = list_audio_files(irs if isinstance(irs, str) else [str(x) for x in irs])
            root = irs if isinstance(irs, str) and os.path.isdir(irs) else None

            def _name(f: str) -> str:  # match exclusions below the IR root only, so an
                # ancestor directory named e.g. "benchmarks" does not drop every file
                return (os.path.relpath(f, root) if root else f).lower()

            files = [f for f in files if not any(e.lower() in _name(f) for e in exclude)]
            for f in tqdm(files, desc="loading IRs", disable=len(files) < 50):
                x, fs = audio.load(f, always_2d=True)
                x = audio.resample(x, fs, self.fs)
                for ch in range(x.shape[0]):
                    self.irs.append(x[ch, : int(max_ir_s * self.fs)])
                    self.ir_names.append(f"{f}#ch{ch}")
        else:
            fs_in = int(fs_ir or self.fs)
            for k, x in enumerate(irs):
                self.irs.append(audio.resample(np.asarray(x, dtype=np.float64), fs_in, self.fs))
                self.ir_names.append(f"array{k}")
        if not self.irs:
            raise ValueError("no measured IRs found (after exclusions)")

    def _room_ir(self, rng: np.random.Generator, i: int) -> tuple[np.ndarray, dict[str, Any]]:
        k = int(rng.integers(len(self.irs)))
        h = self.irs[k]
        a = float(rng.uniform(1.0 - self.stretch, 1.0 + self.stretch))
        h = time_stretch_ir(h, a)
        h = random_band_gain(h, self.fs, rng, self.band_gain_db)
        tail = bool(self.tail_fn is not None and rng.random() < self.p_tail)
        if tail:
            h = self.tail_fn(h, self.fs, rng)
        return h, {"ir": self.ir_names[k], "stretch": a, "tail_resynth": tail}


# ---------------------------------------------------------------------------
# C2: uwsim wrapper
# ---------------------------------------------------------------------------
class SimPairsDataset(Dataset):
    """C2/C3/C10 training pairs from `uwse.uwsim.dataset.UWSimPairs`.

    Builds the simulator config (``SimConfig`` from YAML), the noise bank
    (`uwse.uwsim.noise.NoiseBank` at the simulator rate; the train split is drawn
    inside uwsim) and the optional `RIRBank`, and converts every item to float32
    tensors at ``fs_model`` with a dict meta (uwsim returns JSON text).

    ``segment_s=None`` simulates each utterance whole (needed for C8, whose
    transcripts cover the file): the inner segment length is set to the file's
    duration before every item.
    """

    def __init__(self, clean_files: Sequence[str], noise_root: str | None, sim_config: str | Any,
                 fs_model: int, segment_s: float | None, seed: int = 0, rir_bank: Any = None,
                 target: str = "R1", n_items: int | None = None, **kw: Any) -> None:
        from uwse.uwsim import config as simcfg
        from uwse.uwsim.dataset import RIRBank, UWSimPairs

        if isinstance(sim_config, (str, os.PathLike)):
            loader = getattr(simcfg.SimConfig, "load_yaml", None) or simcfg.load_config
            cfg = loader(str(sim_config))
        else:
            cfg = sim_config
        bank = None
        if noise_root is not None:
            from uwse.uwsim.noise import NoiseBank
            bank = NoiseBank(noise_root, fs=int(getattr(cfg, "fs", 48000)))
        if isinstance(rir_bank, (str, os.PathLike)):
            rir_bank = RIRBank(str(rir_bank))
        self.whole = segment_s is None
        self.inner = UWSimPairs(list(clean_files), bank, cfg, fs_model, segment_s or 4.0, seed,
                                rir_bank=rir_bank, target=target, length=n_items, **kw)
        self.fs_model = fs_model

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, key: Key):
        i, _ = _split_key(key)
        if self.whole:  # per-item length = the utterance (each worker owns its copy)
            import soundfile as sf
            path = self.inner.clean_files[i % len(self.inner.clean_files)]
            self.inner.segment_s = float(sf.info(path).duration)
        item = self.inner[i]
        meta = item[2] if len(item) > 2 else {}
        meta = json.loads(meta) if isinstance(meta, str) else dict(meta)
        meta.setdefault("fs", self.fs_model)
        meta.setdefault("index", i)
        return _to_tensor(item[0]), _to_tensor(item[1]), meta


# ---------------------------------------------------------------------------
# synthetic (tests, W4 smoke run)
# ---------------------------------------------------------------------------
class SyntheticPairs(Dataset):
    """Voiced-speech-like harmonic signal + exponential reverb + noise. No real audio needed.

    clean: harmonics of f0 ~ U(90, 250) Hz with 1/k amplitudes and a 3-6 Hz syllabic
    envelope; noisy: clean convolved with a decaying-noise IR (T60 ~ U(0.2, 0.6) s)
    plus white noise at SNR ~ U(snr_range). Deterministic per (seed, index).
    """

    def __init__(self, n_items: int, fs: int, dur_s: float = 2.0, seed: int = 0,
                 snr_range: tuple[float, float] = (0.0, 15.0), reverb: bool = True) -> None:
        self.n_items, self.fs, self.dur_s, self.seed = n_items, fs, dur_s, seed
        self.snr_range, self.reverb = snr_range, reverb

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, key: Key):
        i, _ = _split_key(key)
        rng = np.random.default_rng([self.seed, i])
        n = int(self.dur_s * self.fs)
        t = np.arange(n) / self.fs
        f0 = rng.uniform(90, 250) * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t))
        ph = 2 * np.pi * np.cumsum(f0) / self.fs
        kmax = int(min(30, (0.45 * self.fs) // 250))
        s = sum(np.sin(k * ph) / k for k in range(1, kmax + 1))
        env = np.clip(np.sin(2 * np.pi * rng.uniform(3, 6) * t + rng.uniform(0, 6.28)), 0, None) ** 2
        s = s * env
        s /= np.max(np.abs(s)) + 1e-9
        s *= 0.3
        y = s
        if self.reverb:
            t60 = rng.uniform(0.2, 0.6)
            L = int(t60 * self.fs)
            h = rng.standard_normal(L) * np.exp(-6.91 * np.arange(L) / self.fs / t60) * 0.3
            h[0] = 1.0
            y = fftconvolve(s, h)[:n]
        snr = rng.uniform(*self.snr_range)
        w = rng.standard_normal(n)
        w *= np.sqrt(np.mean(y ** 2) / np.mean(w ** 2)) * 10 ** (-snr / 20)
        return _to_tensor(y + w), _to_tensor(s), {"snr_db": float(snr), "index": i, "fs": self.fs}


# ---------------------------------------------------------------------------
# frozen validation set
# ---------------------------------------------------------------------------
class WavPairs(Dataset):
    """Pairs frozen by `build_val_set`: <dir>/manifest.jsonl + noisy/*.wav + target/*.wav."""

    def __init__(self, root: str) -> None:
        self.root = root
        with open(os.path.join(root, "manifest.jsonl"), encoding="utf-8") as f:
            self.rows = [json.loads(ln) for ln in f if ln.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: Key):
        i, _ = _split_key(key)
        r = self.rows[i]
        x, _ = audio.load(os.path.join(self.root, r["noisy"]))
        y, _ = audio.load(os.path.join(self.root, r["target"]))
        return _to_tensor(x), _to_tensor(y), r.get("meta", {})


def build_val_set(ds: Dataset, n_items: int, out_dir: str | None = None, fs: int | None = None,
                  overwrite: bool = False) -> Dataset:
    """Fixed-seed validation set (sim-val, §5.3): the first ``n_items`` of a dataset that
    was built with its own seed and dev-clean speakers.

    With ``out_dir`` the pairs are written once as float WAVs plus a manifest and a
    `WavPairs` over them is returned (a frozen copy, identical across runs and models
    even if the simulator code later changes). An existing complete set is reused.
    Without ``out_dir`` a lazy index subset is returned (valid because our datasets are
    deterministic per index).
    """
    n_items = min(n_items, len(ds))
    if out_dir is None:
        return torch.utils.data.Subset(ds, list(range(n_items)))
    man = os.path.join(out_dir, "manifest.jsonl")
    if os.path.isfile(man) and not overwrite:
        existing = WavPairs(out_dir)
        if len(existing) == n_items:
            return existing
    if fs is None:
        raise ValueError("fs is required to write a frozen validation set")
    os.makedirs(os.path.join(out_dir, "noisy"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "target"), exist_ok=True)
    rows = []
    for i in tqdm(range(n_items), desc=f"freezing val set -> {out_dir}"):
        item = ds[i]
        noisy, target = item[0], item[1]
        meta = item[2] if len(item) > 2 else {}
        name = f"{i:06d}.wav"
        audio.save(os.path.join(out_dir, "noisy", name), np.asarray(noisy, np.float64), fs, "FLOAT")
        audio.save(os.path.join(out_dir, "target", name), np.asarray(target, np.float64), fs, "FLOAT")
        rows.append({"noisy": f"noisy/{name}", "target": f"target/{name}", "meta": _jsonable(meta)})
    with open(man + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(man + ".tmp", man)
    return WavPairs(out_dir)
