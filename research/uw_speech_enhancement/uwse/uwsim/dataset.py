"""On-the-fly PyTorch training-pair mixer and the precomputed RIR bank.

Implements RESEARCH_PLAN.md section 5.3 (RIR bank: N RIRs, calibrated-pool vs
generic fraction, noise and SNR drawn on the fly) and the "PyTorch on-the-fly
mixer" of section 4.2 (Implementation).

Determinism: item ``i`` of epoch ``e`` uses ``np.random.default_rng((seed, e, i))``
for every random choice (crop, scene or bank entry, noise segments, ADC noise), so
results are identical regardless of DataLoader worker count or order.

RIR bank layout (written by ``scripts/make_rir_bank.py`` / :meth:`RIRBank.write`)::

    <root>/bank.json             {"n": N, "fs": 48000, "shards": [...], "shard_size": S}
    <root>/scenes.jsonl          one Scene dict per RIR (global index order)
    <root>/shard_00000.npz       rir (S, Lmax) float32, rir_len (S,), thr (S, Lt) float32,
                                 thr_len (S,), index (S,)
"""

from __future__ import annotations

import json
import os
import warnings
from collections import OrderedDict
from typing import Sequence

import numpy as np

from uwse.audio import load, resample
from uwse.uwsim.config import SimConfig
from uwse.uwsim.noise import NoiseBank
from uwse.uwsim.simulate import FS_SIM, Scene, build_chain, sample_scene, simulate

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch is in the env; keep RIRBank importable without it
    torch = None
    Dataset = object


class RIRBank:
    """Sharded npz store of precomputed (scene, rir, thruster_rir) triples.

    Shards are loaded on demand and kept in a small LRU cache (``cache_shards``);
    each DataLoader worker holds its own cache.
    """

    def __init__(self, root: str, cache_shards: int = 4) -> None:
        self.root = root
        with open(os.path.join(root, "bank.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        self.n = int(self.info["n"])
        self.shard_size = int(self.info["shard_size"])
        self.shards: list[str] = list(self.info["shards"])
        with open(os.path.join(root, "scenes.jsonl"), encoding="utf-8") as f:
            self.scenes = [json.loads(line) for line in f if line.strip()]
        self.cache_shards = cache_shards
        self._cache: OrderedDict[int, dict] = OrderedDict()

    def __len__(self) -> int:
        return self.n

    def _shard(self, k: int) -> dict:
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        with np.load(os.path.join(self.root, self.shards[k])) as z:
            d = {name: z[name] for name in z.files}
        self._cache[k] = d
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return d

    def get(self, i: int) -> tuple[Scene, np.ndarray, np.ndarray | None]:
        k, j = divmod(int(i), self.shard_size)
        d = self._shard(k)
        rir = d["rir"][j, : int(d["rir_len"][j])].astype(np.float64)
        tl = int(d["thr_len"][j])
        thr = d["thr"][j, :tl].astype(np.float64) if tl > 0 else None
        return Scene.from_dict(self.scenes[i]), rir, thr

    def draw(self, rng: np.random.Generator) -> tuple[Scene, np.ndarray, np.ndarray | None]:
        return self.get(int(rng.integers(self.n)))

    @staticmethod
    def write_shard(root: str, k: int, rirs: Sequence[np.ndarray], thrs: Sequence[np.ndarray | None],
                    index: Sequence[int]) -> str:
        """Write one shard; returns its file name."""
        L = max(r.size for r in rirs)
        Lt = max([t.size for t in thrs if t is not None] + [1])
        R = np.zeros((len(rirs), L), np.float32)
        T = np.zeros((len(rirs), Lt), np.float32)
        rl = np.zeros(len(rirs), np.int64)
        tl = np.zeros(len(rirs), np.int64)
        for j, (r, t) in enumerate(zip(rirs, thrs)):
            R[j, : r.size], rl[j] = r, r.size
            if t is not None:
                T[j, : t.size], tl[j] = t, t.size
        name = f"shard_{k:05d}.npz"
        np.savez(os.path.join(root, name), rir=R, rir_len=rl, thr=T, thr_len=tl,
                 index=np.asarray(index, np.int64))
        return name

    @staticmethod
    def write(root: str, items: Sequence[tuple[Scene, np.ndarray, np.ndarray | None]],
              shard_size: int = 200) -> "RIRBank":
        """Write a whole (small) bank from memory. Large banks use make_rir_bank.py."""
        os.makedirs(root, exist_ok=True)
        shards = []
        with open(os.path.join(root, "scenes.jsonl"), "w", encoding="utf-8") as f:
            for i, (sc, _, _) in enumerate(items):
                f.write(json.dumps(sc.to_dict()) + "\n")
        for k in range(0, len(items), shard_size):
            chunk = items[k: k + shard_size]
            shards.append(RIRBank.write_shard(root, k // shard_size, [c[1] for c in chunk],
                                              [c[2] for c in chunk], range(k, k + len(chunk))))
        with open(os.path.join(root, "bank.json"), "w", encoding="utf-8") as f:
            json.dump({"n": len(items), "fs": FS_SIM, "shard_size": shard_size, "shards": shards}, f)
        return RIRBank(root)


class UWSimPairs(Dataset):
    """(noisy, target, meta_json) training pairs simulated on the fly.

    Parameters
    ----------
    clean_files : clean speech paths (any rate; resampled to 48 kHz).
    noise_bank : NoiseBank or None (synthetic fallback noise).
    cfg : SimConfig.
    fs_model : output rate (16000; 48000 for DFN3).
    segment_s : crop length in seconds (zero-padded if the file is shorter).
    seed : base seed; see module docstring for the per-item scheme.
    rir_bank : optional RIRBank; if None each item renders its own RIR (slow, needs
        the ISM sibling modules).
    target : "R1" (chain reference, section 8.2) or "R0".
    length : virtual epoch length (default len(clean_files)).
    """

    def __init__(self, clean_files: Sequence[str], noise_bank: NoiseBank | None, cfg: SimConfig,
                 fs_model: int = 16000, segment_s: float = 4.0, seed: int = 0,
                 rir_bank: RIRBank | None = None, target: str = "R1", length: int | None = None) -> None:
        if target not in ("R1", "R0"):
            raise ValueError("target must be 'R1' or 'R0'")
        self.clean_files = list(clean_files)
        self.noise_bank = noise_bank
        self.cfg = cfg
        self.fs_model = fs_model
        self.segment_s = segment_s
        self.seed = seed
        self.rir_bank = rir_bank
        self.target = target
        self.length = length if length is not None else len(self.clean_files)
        self.epoch = 0
        self._chain = None
        if noise_bank is None or len(noise_bank) == 0:
            warnings.warn("UWSimPairs without a noise bank: every item uses synthetic fallback noise "
                          "(smoke tests only, not a section 5.2 noise model)", stacklevel=2)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def _clean_segment(self, path: str, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        x, fs = load(path)
        if x.ndim > 1:
            x = x[0]
        n_in = int(round(self.segment_s * fs))
        start = int(rng.integers(0, x.size - n_in + 1)) if x.size > n_in else 0
        x = x[start: start + n_in]
        if fs != FS_SIM:
            x = resample(x, fs, FS_SIM)
        n = int(round(self.segment_s * FS_SIM))
        x = x[:n] if x.size >= n else np.pad(x, (0, n - x.size))
        return x, start

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, self.epoch, int(i)))
        path = self.clean_files[int(i) % len(self.clean_files)]
        s, start = self._clean_segment(path, rng)
        if self.rir_bank is not None:
            scene, rir, thr = self.rir_bank.draw(rng)
            # Per-take parameters (noise, SNR, levels, drive, EQs) are re-drawn; geometry stays.
            fresh = sample_scene(self.cfg, rng, generic=scene.is_generic)
            for k in ("pool", "c", "surface_R", "walls", "floor", "t_e", "t60_bands", "t60_two_slope",
                      "f_x", "modal_f_max", "rir_length_s", "src", "rx", "range_m", "orientation_deg",
                      "theta_deg", "thruster_on", "thruster_pos", "rir_seed", "is_generic"):
                setattr(fresh, k, getattr(scene, k))
            scene = fresh
        else:
            scene, rir, thr = sample_scene(self.cfg, rng), None, None
        if self._chain is None:
            self._chain = build_chain(self.cfg)
        out = simulate(s, scene, self.noise_bank, rng, fs_model=self.fs_model, chain=self._chain,
                       rir=rir, thruster_rir=thr, target_gain=self.cfg.target.gain)
        meta = out["meta"]
        meta.update(clean_path=path, clean_start=start, index=int(i), epoch=self.epoch)
        noisy = out["noisy"]
        tgt = out["target_" + self.target]
        if torch is None:
            return noisy, tgt, json.dumps(meta)
        return torch.from_numpy(noisy), torch.from_numpy(tgt), json.dumps(meta)
