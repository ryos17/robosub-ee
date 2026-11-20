"""Added-noise conditions: real pool / bubble noise mixed into the real speech cuts.

Pool plan of 2026-11-19. The water path stays real: each noisy cut is the recorded pool cut
(unchanged; the original stays where it is) plus a segment of a real noise-only recording
(take ``p<d>_noise`` per distance, ``p3bub_noise`` from the bubble pass), scaled so that

    SNR = 10 log10( mean(s_b^2 | active) / mean(n_b^2 | active) )

equals the target (default 0 and 5 dB). ``_b`` is the zero-phase band-pass ``--band``
(default 500-8000 Hz, the simulator's and score.py's first band) and ``active`` is score.py's
level-normalization VAD (``normalize.energy_vad``) on the aligned R0 cut, i.e. where the
played utterance is; the mixing itself is ``uwse.uwsim.noise.mix_at_snr``. ``s`` is the
recorded cut, so it already holds the (low) ambient noise of the quiet take; the stated SNR
is that of the added noise only. Every output is labelled added noise.

Determinism: the noise offset of a cut is drawn from a generator seeded by sha256 of
(``--seed``, noise kind, utt_id), uniform over the usable part of the noise take
(``--trim-s`` dropped at each end). The same offset is used for every SNR of a (kind, cut),
so the 0 and 5 dB versions differ only in the noise gain.

Noise takes are given as ``KIND=PATH`` (used at every position) or ``KIND@POS=PATH`` (used
at that position only; a position-specific take wins), read on ``--noise-channel`` (file
channel, default 0 = Daisy A ch0, the scored hydrophone) and resampled to the cut rate::

    python scripts/mix_real_noise.py --manifest data/real/v1_s2/score/manifest.jsonl \
        --noise pool@p1=sessions/<date>_s2/raw/daisyA/s2_p1_noise_daisyA.wav \
        --noise pool@p3=sessions/<date>_s2/raw/daisyA/s2_p3_noise_daisyA.wav \
        --noise pool@p6=sessions/<date>_s2/raw/daisyA/s2_p6_noise_daisyA.wav \
        --noise bubble=sessions/<date>_s2/raw/daisyA/s2_p3bub_noise_daisyA.wav --snr 0,5 --out data/real/v1_s2/added

Input rows: the C0 rows of the score.py manifest (and C8 rows, which share the C0 wav, are
re-made on the noisy wav); enhanced rows are dropped (they must be re-enhanced on the noisy
cuts). Outputs under ``--out``:

- ``<kind><snr>dB/wav/<utt_id>.wav``  (FLOAT, cut rate, cut length)
- ``<kind><snr>dB/manifest.jsonl``    score.py manifest: the input rows with ``wav`` replaced,
  ``condition``, ``added_noise`` {kind, snr_db, file, offset_samples, gain}, and the measured
  ``added_snr_db_500_8000`` / ``added_snr_db_50_8000``; the quiet-take SNR keys are removed
- ``mix.jsonl`` (one row per noisy cut) and ``mix_info.json`` (args, noise-file sha256)

Outputs are rewritten from scratch on every run (deterministic), never appended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uwse import audio  # noqa: E402
from uwse.evaluation import normalize as N  # noqa: E402
from uwse.uwsim.noise import band_snr_db, energy_vad, mix_at_snr  # noqa: E402

SNR_KEYS = ("snr_db_500_8000", "snr_db_50_8000", "snr_level_drift")
SAME_WAV_SYSTEMS = ("C0", "C8")


def cond_name(kind: str, snr_db: float) -> str:
    return f"{kind}{snr_db:g}dB"


def parse_noise(specs: list[str]) -> dict[tuple[str, str | None], Path]:
    """``KIND=PATH`` / ``KIND@POS=PATH`` -> {(kind, pos or None): path}."""
    out: dict[tuple[str, str | None], Path] = {}
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"--noise expects KIND=PATH or KIND@POS=PATH, got {s!r}")
        k, p = s.split("=", 1)
        kind, _, pos = k.partition("@")
        if not kind.isalpha():
            raise SystemExit(f"noise kind must be letters only (used in condition names), got {kind!r}")
        key = (kind, pos or None)
        if key in out:
            raise SystemExit(f"--noise {k} given twice")
        out[key] = Path(p)
    return out


def noise_offset(seed: int, kind: str, utt_id: str, usable: int) -> int:
    """Deterministic offset in [0, usable], independent of row order and of the SNR."""
    if usable < 0:
        raise ValueError("noise take shorter than the cut")
    h = hashlib.sha256(f"{seed}|{kind}|{utt_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return int(rng.integers(0, usable + 1))


class NoiseTakes:
    """Noise-only recordings, one channel, trimmed, resampled once per rate."""

    def __init__(self, paths: dict[tuple[str, str | None], Path], channel: int, trim_s: float):
        self.paths, self.channel, self.trim_s = paths, channel, trim_s
        self._cache: dict[tuple[Path, int], np.ndarray] = {}

    def kinds(self) -> list[str]:
        return sorted({k for k, _ in self.paths})

    def path(self, kind: str, pos: str) -> Path | None:
        return self.paths.get((kind, pos)) or self.paths.get((kind, None))

    def get(self, path: Path, fs: int) -> np.ndarray:
        if (path, fs) not in self._cache:
            x, fs_raw = audio.load(str(path), always_2d=True)
            if self.channel >= x.shape[0]:
                raise SystemExit(f"{path} has {x.shape[0]} channels, --noise-channel {self.channel}")
            x = np.asarray(x[self.channel], np.float64)
            t = int(round(self.trim_s * fs_raw))
            if x.size <= 2 * t:
                raise SystemExit(f"{path}: {x.size / fs_raw:.1f} s is too short for --trim-s {self.trim_s}")
            x = x[t: x.size - t]
            self._cache[(path, fs)] = audio.resample(x, fs_raw, fs) if fs_raw != fs else x
        return self._cache[(path, fs)]


def active_mask(row: dict, y: np.ndarray, fs: int) -> np.ndarray:
    """score.py's R0 VAD when the row has an aligned R0 cut, else an energy VAD on the cut."""
    if row.get("r0_wav"):
        r0, fs0 = audio.load(row["r0_wav"])
        r0 = r0 if r0.ndim == 1 else r0[0]
        if fs0 != fs:
            r0 = audio.resample(r0, fs0, fs)
        m = N.energy_vad(r0, fs, N.VADConfig())
        m = np.concatenate([m, np.zeros(max(0, y.size - m.size), bool)])[: y.size]
        if m.any():
            return m
    return energy_vad(y, fs)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="score.py manifest of the quiet cuts")
    ap.add_argument("--noise", action="append", default=[], required=True, metavar="KIND[@POS]=PATH")
    ap.add_argument("--noise-channel", type=int, default=0, help="file channel of the noise takes")
    ap.add_argument("--snr", default="0,5", help="comma list of target SNRs (dB)")
    ap.add_argument("--band", default="500,8000", help="SNR band lo,hi in Hz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trim-s", type=float, default=1.0, help="seconds dropped at each end of a noise take")
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args(argv)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    a = parse_args(argv)
    snrs = [float(s) for s in a.snr.split(",") if s.strip()]
    band = tuple(float(v) for v in a.band.split(","))
    takes = NoiseTakes(parse_noise(a.noise), a.noise_channel, a.trim_s)
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    c0 = [r for r in rows if r["system"] == "C0" and r.get("kind", "speech") == "speech"]
    if not c0:
        raise SystemExit(f"no C0 speech rows in {a.manifest}")
    other = {r["system"] for r in rows} - {"C0"}
    mirrored = [s for s in SAME_WAV_SYSTEMS if s in other]
    dropped = sorted(other - set(mirrored))
    c0_wav = {r["utt_id"]: r["wav"] for r in c0}
    for r in rows:
        if r["system"] in mirrored and r["wav"] != c0_wav.get(r["utt_id"]):
            raise SystemExit(f"{r['system']} row {r['utt_id']} does not use the C0 wav")

    for kind in takes.kinds():
        for snr in snrs:
            d = a.out / cond_name(kind, snr)
            if d.exists():
                shutil.rmtree(d)
            (d / "wav").mkdir(parents=True)
    mix_rows = []
    manifests: dict[str, list[dict]] = {cond_name(k, s): [] for k in takes.kinds() for s in snrs}
    for r in tqdm(c0, desc="mix"):
        y, fs = audio.load(r["wav"])
        y = np.asarray(y if y.ndim == 1 else y[0], np.float64)
        mask = active_mask(r, y, fs)
        for kind in takes.kinds():
            p = takes.path(kind, str(r["position"]))
            if p is None:
                raise SystemExit(f"no {kind} noise take for position {r['position']}")
            nz = takes.get(p, fs)
            off = noise_offset(a.seed, kind, r["utt_id"], nz.size - y.size)
            seg = nz[off: off + y.size]
            for snr in snrs:
                cond = cond_name(kind, snr)
                mix, scaled = mix_at_snr(y, seg, snr, fs, band=band, active_mask=mask)
                gain = float(np.sqrt(np.mean(scaled ** 2) / max(np.mean(seg ** 2), 1e-30)))
                wp = a.out / cond / "wav" / f"{r['utt_id']}.wav"
                audio.save(str(wp), mix, fs, subtype="FLOAT")
                meas = {f"added_snr_db_{int(lo)}_{int(hi)}": band_snr_db(y, scaled, fs, (lo, hi), mask)
                        for lo, hi in (band, (50.0, 8000.0))}
                info = {"kind": kind, "snr_db": snr, "file": str(p), "offset_samples": off,
                        "offset_s": off / fs, "gain": gain, "band_hz": list(band), "seed": a.seed}
                mix_rows.append({"utt_id": r["utt_id"], "condition": cond, "wav": str(wp),
                                 "source_wav": r["wav"], **info, **meas})
                base = {k: v for k, v in r.items() if k not in SNR_KEYS}
                for sysname in ("C0", *mirrored):
                    manifests[cond].append({**base, "system": sysname, "wav": str(wp), "condition": cond,
                                            "added_noise": info, **meas})
    for cond, ms in manifests.items():
        with open(a.out / cond / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(m) + "\n" for m in ms)
    with open(a.out / "mix.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(m) + "\n" for m in mix_rows)
    info = {"args": {k: str(v) for k, v in vars(a).items()}, "snrs_db": snrs, "band_hz": list(band),
            "conditions": sorted(manifests), "systems": ["C0", *mirrored], "dropped_systems": dropped,
            "n_cuts": len(c0), "manifest_sha256": _sha(a.manifest),
            "noise_files": {f"{k}@{p or '*'}": {"path": str(v), "sha256": _sha(v)}
                            for (k, p), v in takes.paths.items()}}
    (a.out / "mix_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"{len(c0)} cuts x {len(manifests)} conditions ({', '.join(sorted(manifests))}) -> {a.out}"
          + (f"; dropped systems {dropped} (re-enhance the noisy cuts)" if dropped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
