"""Front-end inference on real cuts: frozen C1 / C2 / C10 outputs for scripts/score.py.

The frozen protocol (RESULTS_E7_SIM_DRYRUN.md, 2026-11-09 amendment) scores the SE front
ends minus C0 as secondary endpoints "once a front-end inference script for real cuts
exists"; this is that script. No training, and no change to weights.

Systems and weights (the labels are the manifest's ``system`` values):

    C1   wpe1        no weights, always on the CPU (as sim_baseline)
    C2   dfn3        runs/c2_dfn3_s0/best.pt       (dfn3_c2 in RESULTS_E7)
    C10  kiel_unet   runs/c10_kiel_unet_s0/best.pt

Each system is built by ``sim_baseline._load_system`` and each cut goes through
``sim_baseline._enhance_one``, the functions the simulation used: the whole cut at its own
rate is resampled (``audio.resample``) to the adapter's rate, enhanced in one call, resampled
to 16 kHz and fitted to the length score.py gives the C0 cut. A checkpoint is refused if the
path does not exist or if the loaded state differs from it (``dfn3`` loads with
strict=False over the zero-shot weights, and ``kiel_unet`` starts from random ones).

Input level is a protocol decision, so ``--input-level`` is required and recorded. The
simulator set every input to -26 dBFS over the active speech, and C2's output depends on
level (17 dB match at +/-10 dB, 9.5 dB at -30 dB on one cut):

    none   the cut at the recorder's level, unchanged
    sim    each cut is scaled by the gain score.py's level normalization gives its C0 cut
           (R0 VAD, -26 dBFS active; noise rows the median speech gain), and the output is
           scaled back, so outputs stay on the recorder's scale and comparable with C0

The per-cut gains are in cuts.json, and per-thruster-state active levels in run_info.json.

The frozen systems see their simulation rates. A cut below an adapter's rate (e.g. a 16 kHz
clip for 48 kHz C2) has no 8-24 kHz band, which is not the frozen system: it is refused
unless ``--allow-upsampled-input``, and then labelled e.g. ``C2_16kin`` in the manifest so it
cannot be pooled with C2.

Inputs (exactly one):

    --manifest   score.py JSONL; its C0 rows are the cuts, other systems' rows are dropped
    --wav-list   text file with one wav path per line (+ optional --root for relative ids)

Outputs in ``--out-dir``: <system>/<utt_id>.wav (FLOAT, 16 kHz, one per cut), cuts.json
(SHA-256, rate, length, levels, gain of every cut), run_info.json (checkpoint SHA-256, adapter,
device, RTF, failed cuts) and manifest.jsonl (the C0 rows, then each finished system's rows, which copy
the C0 row and change only ``system`` and ``wav``).

Resumable: finished outputs are skipped, files appear atomically, and a resume is refused if
a cut, a gain, a checkpoint or the level policy changed. A cut that fails is recorded in
run_info.json and the job goes on; its system stays out of the manifest and the exit status
is non-zero. One model is in memory at a time. Separate calls with ``--systems C2`` etc.
accumulate into the same manifest, but calls on one ``--out-dir`` must be sequential (a lock
file enforces it).

Example (long: launch detached per the repo convention)::

    python scripts/enhance_cuts.py --manifest data/real/v0_s1/score/manifest.jsonl \
        --out-dir runs/enhance/v0_s1 --input-level sim
    python scripts/score.py --manifest runs/enhance/v0_s1/manifest.jsonl \
        --out-dir runs/score/v0_s1_se --level-normalize --baseline C0 --n-boot 2000
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from uwse import audio  # noqa: E402
from uwse.evaluation import normalize as N  # noqa: E402

import sim_baseline as sb  # noqa: E402

FS_SCORE = sb.FS_SCORE
BASELINE = "C0"
LEVEL_WARN_DB = 3.0  # warn when a thruster state's median active level is this far from the simulator's -26 dBFS


@dataclasses.dataclass(frozen=True)
class SystemSpec:
    label: str  # manifest system name and output directory
    adapter: str  # uwse.adapters registry key
    ckpt: str | None = None  # fine-tuned weights; None for systems without weights
    adapter_fs: int = 16000  # the adapter's input rate; checked against the loaded model
    upsampled_input: bool = False  # cuts below adapter_fs, allowed explicitly (not the frozen system)


FROZEN = {
    "C1": SystemSpec("C1", "wpe1"),
    "C2": SystemSpec("C2", "dfn3", str(ROOT / "runs" / "c2_dfn3_s0" / "best.pt"), adapter_fs=48000),
    "C10": SystemSpec("C10", "kiel_unet", str(ROOT / "runs" / "c10_kiel_unet_s0" / "best.pt")),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def n_out_16k(frames: int, fs: int) -> int:
    """Samples of the cut at 16 kHz as score.py loads it: resample_poly's ceil(T * up / down)."""
    if fs == FS_SCORE:
        return frames
    r = Fraction(FS_SCORE, fs).limit_denominator(10000)
    return -(-frames * r.numerator // r.denominator)


def out_path(out_dir: Path, label: str, utt_id: str) -> Path:
    """<out>/<system>/<utt_id>.wav, keeping an utt_id's <session>/ subdirectory."""
    rel = Path(utt_id[:-4] if utt_id.endswith(".wav") else utt_id)
    if rel.is_absolute() or rel.drive or ".." in rel.parts:
        raise SystemExit(f"unsafe utt_id {utt_id!r}")
    return out_dir / label / rel.with_name(rel.name + ".wav")


def resolve_device(adapter: str, device: str) -> str:
    """sim_baseline's rule: identity and WPE always run on the CPU."""
    return "cpu" if adapter in ("identity", "wpe1") else device


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_wav(path: Path, y: np.ndarray) -> None:
    """Atomic FLOAT 16 kHz write, so an interrupted run never leaves a partial file to skip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    sf.write(str(tmp), y.astype(np.float32), FS_SCORE, subtype="FLOAT", format="WAV")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def read_c0_rows(path: str) -> tuple[list[dict], dict[str, int]]:
    """The verbatim C0 rows of a score.py manifest, and the count of dropped rows per system.

    Validation is score.py's own. Dropping the other systems also keeps the secondary run's
    Holm family (score.py corrects across every system in the manifest) to the front ends."""
    import score

    score.read_manifest(path)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    dropped: dict[str, int] = {}
    for r in rows:
        if r["system"] != BASELINE:
            dropped[r["system"]] = dropped.get(r["system"], 0) + 1
    c0 = [r for r in rows if r["system"] == BASELINE]
    if not c0:
        raise SystemExit(f"{path}: no {BASELINE} rows")
    return c0, dropped


def _unlabeled_row(utt_id: str, session: str, wav: str) -> dict:
    """A C0 row for a clip with no labels: the session is the only known cluster and the
    clip is its own VAD source (no R0), the same for every system."""
    return {"utt_id": utt_id, "system": BASELINE, "wav": wav, "r0_wav": wav, "transcript": "",
            "speaker": session, "position": session, "thruster": "unknown", "kind": "speech"}


def rows_from_wav_list(path: str, root: str | None) -> list[dict]:
    """One wav per line; utt_id is the path relative to ``root`` (else the file stem)."""
    rows, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            wav = os.path.join(root, p) if root and not os.path.isabs(p) else p
            uid = Path(os.path.relpath(wav, root)).as_posix() if root else Path(p).stem
            if uid in seen:
                raise SystemExit(f"{path}: duplicate utt_id {uid!r} (give --root to keep subdirectories)")
            seen.add(uid)
            rows.append(_unlabeled_row(uid, Path(wav).parent.name or "unknown", wav))
    if not rows:
        raise SystemExit(f"{path}: no wav paths")
    return rows


def inspect_cuts(rows: list[dict]) -> list[dict]:
    """Hash and measure every cut. A cut must be one mono file: nothing upstream writes
    anything else, and picking a channel here could silently score the wrong hydrophone."""
    cuts = []
    for r in tqdm(rows, desc="inspect cuts"):
        p = r["wav"]
        if not os.path.isfile(p):
            raise SystemExit(f"{r['utt_id']}: missing cut {p}")
        try:
            x, fs = audio.load(p)
        except RuntimeError as e:
            raise SystemExit(f"{r['utt_id']}: cannot read cut {p}: {e}")
        if x.ndim != 1:
            raise SystemExit(f"{p}: {x.shape[0]} channels; a cut must be mono")
        if x.size == 0:
            raise SystemExit(f"{r['utt_id']}: empty cut {p}")
        cuts.append({"utt_id": r["utt_id"], "wav": p, "fs": int(fs), "frames": int(x.size),
                     "n_out": n_out_16k(int(x.size), int(fs)), "sha256": sha256(p),
                     "rms_dbfs": float(audio.db(audio.rms(x))), "peak_dbfs": float(audio.db(np.max(np.abs(x))))})
    return cuts


def check_output_paths(rows: list[dict], out_dir: Path) -> None:
    """Two cuts must never share an output file, also on a case-insensitive filesystem."""
    seen: dict[str, str] = {}
    for r in rows:
        key = os.path.normcase(str(out_path(out_dir, BASELINE, r["utt_id"])))
        if key in seen:
            raise SystemExit(f"utt_ids {seen[key]!r} and {r['utt_id']!r} collide on one output file")
        seen[key] = r["utt_id"]


def level_gains(rows: list[dict]) -> list[float | None]:
    """score.py's level-normalization gain (dB) of each C0 cut; None where it finds no speech."""
    import score

    bad = next((r for r in rows if not os.path.isfile(r.get("r0_wav") or r.get("ref_wav") or "")), None)
    if bad is not None:
        raise SystemExit(f"{bad['utt_id']}: no r0_wav/ref_wav file to measure the active level on")
    scored = score.level_normalize([{**r, "kind": r.get("kind") or "speech", "position": str(r.get("position")),
                                     "thruster": str(r.get("thruster"))} for r in rows], N.VADConfig())
    return [float(20.0 * np.log10(r["gain"])) if np.isfinite(r["gain"]) else None for r in scored]


def apply_level_policy(rows: list[dict], cuts: list[dict], input_level: str) -> None:
    """Fill each cut's active level and the gain to apply in front of the systems."""
    try:
        gains: list[float | None] = level_gains(rows)
    except (SystemExit, ValueError) as e:
        if input_level == "sim":
            raise SystemExit(f"--input-level sim needs score.py's gain for every cut: {e}")
        print(f"[warn] active speech level unavailable, so it cannot be compared with the simulator's: {e}")
        gains = [None] * len(cuts)
    for c, r, g in zip(cuts, rows, gains):
        c["thruster"] = str(r.get("thruster"))
        c["score_gain_db"] = g
        speech = (r.get("kind") or "speech") == "speech"
        c["active_dbfs"] = N.TARGET_DBFS - g if g is not None and speech else None
        c["input_gain_db"] = g if input_level == "sim" and g is not None else 0.0


def check_cuts(out_dir: Path, cuts: list[dict]) -> None:
    """Refuse to resume over a cut whose audio or gain changed; a superset of an earlier run is fine."""
    path = out_dir / "cuts.json"
    seen = _read_json(path) or {}
    changed = [c["utt_id"] for c in cuts if c["utt_id"] in seen and
               (seen[c["utt_id"]]["sha256"] != c["sha256"]
                or seen[c["utt_id"]].get("input_gain_db", c["input_gain_db"]) != c["input_gain_db"])]
    if changed:
        raise SystemExit(f"{len(changed)} cuts differ from the ones enhanced in {out_dir} "
                         f"(e.g. {changed[0]}); use a new --out-dir")
    seen.update({c["utt_id"]: {k: c[k] for k in ("wav", "fs", "frames", "sha256", "rms_dbfs", "peak_dbfs",
                                                 "active_dbfs", "score_gain_db", "input_gain_db")}
                 for c in cuts})
    _write_json(path, seen)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def verify_checkpoint_loaded(se, path: str) -> None:
    """Every tensor of the loaded model must equal the checkpoint's, key for key."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    want = state["model"] if isinstance(state, dict) and "model" in state else state
    have = {k: v.detach().cpu() for k, v in se.model.state_dict().items()}
    bad = sorted(set(want) ^ set(have)) + [k for k in sorted(set(want) & set(have))
                                           if not torch.equal(have[k], want[k].cpu())]
    if bad:
        raise SystemExit(f"{se.name}: model state differs from {path} at {len(bad)} keys (e.g. {bad[0]!r})")


def load_system(spec: SystemSpec, device: str):
    """sim_baseline's loader, then the check that the frozen weights really are in the model."""
    from uwse.adapters import available, import_errors

    if spec.adapter not in available():
        raise SystemExit(f"{spec.label}: adapter {spec.adapter!r} unavailable: {import_errors or 'not registered'}")
    ckpts = {spec.label: spec.ckpt} if spec.ckpt else {}
    se = sb._load_system(spec.label, device, {spec.label: spec.adapter}, ckpts)
    if spec.ckpt:
        verify_checkpoint_loaded(se, spec.ckpt)
    return se


def _free(device: str) -> None:
    import torch

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def out_dir_lock(out_dir: Path):
    """One process per --out-dir: run_info, cuts and the manifest are read-modify-write. The
    OS drops the lock when the process dies, so a crashed run cannot leave a stale lock."""
    f = open(out_dir / ".lock", "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit(f"another enhance_cuts.py is running on {out_dir}; calls on one --out-dir must be sequential")
    try:
        yield
    finally:
        if os.name == "nt":
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()


def _info_base(out_dir: Path) -> dict:
    return _read_json(out_dir / "run_info.json") or {"systems": {}}


def stamp_system(out_dir: Path, spec: SystemSpec, ckpt_sha: str | None, device: str) -> None:
    """Record provenance before the first output exists; refuse a resume with other weights."""
    info = _info_base(out_dir)
    old = info["systems"].get(spec.label)
    if old is None:
        d = out_dir / spec.label
        if d.is_dir() and next(d.rglob("*.wav"), None) is not None:
            raise SystemExit(f"{d} holds wavs with no recorded checkpoint; use a new --out-dir")
    elif old["adapter"] != spec.adapter or old["ckpt_sha256"] != ckpt_sha:
        raise SystemExit(f"{spec.label}: {out_dir} was made with {old['adapter']} sha256 {old['ckpt_sha256']}, "
                         f"now {spec.adapter} sha256 {ckpt_sha}; use a new --out-dir")
    info["systems"].setdefault(spec.label, {}).update(adapter=spec.adapter, ckpt=spec.ckpt, ckpt_sha256=ckpt_sha,
                                                      device=device, status="running",
                                                      upsampled_input=spec.upsampled_input)
    _write_json(out_dir / "run_info.json", info)


def _output_problem(path: Path, n_out: int) -> str | None:
    """Why ``path`` is not a usable finished output of a cut, or None if it is."""
    if not path.exists():
        return "missing"
    try:
        i = sf.info(str(path))
    except RuntimeError as e:  # soundfile's LibsndfileError
        return f"unreadable ({e})"
    if i.samplerate != FS_SCORE or i.frames != n_out or i.channels != 1:
        return f"{i.channels} ch, {i.frames} frames at {i.samplerate} Hz; expected mono, {n_out} at {FS_SCORE}"
    return None


def _is_done(path: Path, n_out: int) -> bool:
    if not path.exists():
        return False
    problem = _output_problem(path, n_out)
    if problem:
        raise SystemExit(f"{path}: {problem}. Delete this file and rerun to re-enhance it")
    return True


def enhance_system(spec: SystemSpec, ckpt_sha: str | None, cuts: list[dict], out_dir: Path, device: str,
                   loader: Callable = load_system) -> dict:
    """Enhance the cuts without an output yet; the model is loaded only if there are any.
    A cut that fails is recorded in the returned stats and the remaining cuts still run."""
    dev = resolve_device(spec.adapter, device)
    stamp_system(out_dir, spec, ckpt_sha, dev)
    todo = [c for c in cuts if not _is_done(out_path(out_dir, spec.label, c["utt_id"]), c["n_out"])]
    failures: list[dict] = []
    stats: dict = {"n_cuts": len(cuts), "n_skipped": len(cuts) - len(todo), "n_enhanced": 0}
    if todo:
        se = loader(spec, dev)
        if int(se.sample_rate) != spec.adapter_fs:
            raise SystemExit(f"{spec.label}: adapter runs at {se.sample_rate} Hz, declared {spec.adapter_fs}")
        stats["adapter_sample_rate"] = int(se.sample_rate)
        t0, n_audio = time.perf_counter(), 0.0
        for c in tqdm(todo, desc=spec.label):
            try:
                x, fs = audio.load(c["wav"])
                g = 10.0 ** (c["input_gain_db"] / 20.0)  # exactly 1.0 for --input-level none
                y16 = sb._enhance_one(se, x * g, fs, c["n_out"], dev) / g
                if not np.all(np.isfinite(y16)):
                    raise ValueError("non-finite output")
                _write_wav(out_path(out_dir, spec.label, c["utt_id"]), y16)
            except Exception as e:  # noqa: BLE001 (recorded and reported by run())
                failures.append({"utt_id": c["utt_id"], "error": f"{type(e).__name__}: {e}"})
                tqdm.write(f"[error] {spec.label} {c['utt_id']}: {failures[-1]['error']}")
                _free(dev)
                continue
            stats["n_enhanced"] += 1
            n_audio += x.size / fs
        stats["rtf"] = (time.perf_counter() - t0) / max(n_audio, 1e-9)
        del se
        _free(dev)
    stats["failures"] = failures
    info = _info_base(out_dir)
    info["systems"][spec.label].update(status="failed_cuts" if failures else "done", **stats)
    _write_json(out_dir / "run_info.json", info)
    return stats


def write_manifest(rows: list[dict], cuts: list[dict], out_dir: Path) -> tuple[Path, list[str]]:
    """C0 rows, then each stamped system with a valid output for every row (in FROZEN order)."""
    stamped = _info_base(out_dir)["systems"]
    order = [s for s in FROZEN if s in stamped] + [s for s in stamped if s not in FROZEN]
    done = []
    for s in order:
        bad = {c["utt_id"]: p for c in cuts if (p := _output_problem(out_path(out_dir, s, c["utt_id"]), c["n_out"]))}
        if bad:
            uid, why = next(iter(bad.items()))
            print(f"[warn] {s} left out of the manifest: {len(bad)} of {len(cuts)} outputs not usable (e.g. {uid}: {why})")
        else:
            done.append(s)
    out = list(rows)
    for s in done:
        out.extend({**r, "system": s, "wav": str(out_path(out_dir, s, r["utt_id"]).resolve())} for r in rows)
    path = out_dir / "manifest.jsonl"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    os.replace(tmp, path)
    return path, done


def _level_summary(cuts: list[dict]) -> dict:
    v = np.array([c["rms_dbfs"] for c in cuts], float)
    by: dict[str, list[float]] = {}
    for c in cuts:
        if c["active_dbfs"] is not None:
            by.setdefault(c["thruster"], []).append(c["active_dbfs"])
    return {"n": int(v.size), "rms_dbfs_min": float(v.min()), "rms_dbfs_median": float(np.median(v)),
            "rms_dbfs_max": float(v.max()), "fs_in": sorted({c["fs"] for c in cuts}),
            "active_dbfs_by_thruster": {t: {"n": len(a), "median": float(np.median(a))} for t, a in sorted(by.items())},
            "n_without_active_level": int(sum(c["active_dbfs"] is None for c in cuts))}


def _report_levels(summary: dict, input_level: str) -> None:
    """Per-state active level against the simulator's; a mismatch matters only with no gain."""
    for t, s in summary["active_dbfs_by_thruster"].items():
        off = s["median"] - N.TARGET_DBFS
        print(f"thruster {t}: median active level {s['median']:.1f} dBFS ({off:+.1f} dB re simulator, n={s['n']})")
        if input_level == "none" and abs(off) > LEVEL_WARN_DB:
            print(f"[warn] thruster {t}: {off:+.1f} dB from the simulator's {N.TARGET_DBFS:.0f} dBFS input; "
                  "C2's output depends on level and --input-level none applies no gain")
    if summary["n_without_active_level"]:
        print(f"[warn] {summary['n_without_active_level']} cuts have no active speech level "
              "(noise-only cuts, or no speech found)" + ("; those without a gain get none" if input_level == "sim" else ""))


def apply_rate_policy(specs: list[SystemSpec], cuts: list[dict], allow: bool) -> list[SystemSpec]:
    """A cut below an adapter's rate lacks the band the frozen system was fed: refuse it, or
    with ``allow`` give that system its own label so it cannot be pooled with the frozen one."""
    fs_min = min(c["fs"] for c in cuts)
    out = []
    for s in specs:
        if fs_min >= s.adapter_fs:
            out.append(s)
            continue
        n = sum(c["fs"] < s.adapter_fs for c in cuts)
        if not allow:
            raise SystemExit(f"{s.label}: {n} cuts are below the adapter's {s.adapter_fs} Hz (lowest {fs_min} Hz), so "
                             "it would not be the frozen system; give --allow-upsampled-input to run it anyway "
                             "under its own label")
        label = f"{s.label}_{fs_min // 1000}kin"
        print(f"[warn] {s.label}: {n} cuts upsampled to {s.adapter_fs} Hz; not the frozen system, labelled {label}")
        out.append(dataclasses.replace(s, label=label, upsampled_input=True))
    return out


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(rows: list[dict], specs: list[SystemSpec], out_dir: Path, device: str, meta: dict | None = None,
        loader: Callable = load_system, input_level: str = "none",
        allow_upsampled: bool = False) -> tuple[Path, list[str]]:
    """Enhance ``rows`` (C0 rows) with ``specs``; returns the manifest path and finished systems.
    Every checkpoint is checked before any audio is read. Exits non-zero after writing the
    manifest if any cut failed."""
    check_output_paths(rows, out_dir)
    shas: dict[str, str | None] = {}
    for s in specs:
        if s.ckpt is not None and not Path(s.ckpt).is_file():
            raise SystemExit(f"{s.label}: checkpoint {s.ckpt} does not exist; refusing to run without its weights")
        shas[s.label] = sha256(s.ckpt) if s.ckpt else None
    sb._check_out(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_dir_lock(out_dir):
        return _run_locked(rows, specs, shas, out_dir, device, meta, loader, input_level, allow_upsampled)


def _run_locked(rows, specs, shas, out_dir, device, meta, loader, input_level, allow_upsampled):
    cuts = inspect_cuts(rows)
    specs_in = specs
    specs = apply_rate_policy(specs, cuts, allow_upsampled)
    shas = {new.label: shas[old.label] for old, new in zip(specs_in, specs)}
    apply_level_policy(rows, cuts, input_level)
    info = _info_base(out_dir)
    if info.get("systems") and info.get("input_level_policy") != input_level:
        raise SystemExit(f"{out_dir} was made with --input-level {info.get('input_level_policy')}, "
                         f"now {input_level}; use a new --out-dir")
    check_cuts(out_dir, cuts)
    info.update(meta or {}, git_head=_git_head(), n_cuts=len(cuts), input_level=_level_summary(cuts),
                fs_score=FS_SCORE, input_level_policy=input_level)
    _write_json(out_dir / "run_info.json", info)
    _report_levels(info["input_level"], input_level)
    failed: dict[str, list[dict]] = {}
    try:
        for s in specs:
            stats = enhance_system(s, shas[s.label], cuts, out_dir, device, loader)
            print(f"{s.label}: {stats}")
            if stats["failures"]:
                failed[s.label] = stats["failures"]
    finally:  # finished systems stay usable whatever stopped the job
        path, done = write_manifest(rows, cuts, out_dir)
        print(f"{len(rows)} cuts x {1 + len(done)} systems ({', '.join([BASELINE] + done)}) -> {path}")
    import score

    score.read_manifest(str(path))  # the output must be a manifest score.py accepts
    if failed:
        lines = [f"{s}: {f['utt_id']}: {f['error']}" for s, fs in failed.items() for f in fs]
        raise SystemExit(f"{len(lines)} cuts failed (details in run_info.json; rerun to retry them):\n" + "\n".join(lines))
    return path, done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="score.py manifest; its C0 rows are the cuts")
    src.add_argument("--wav-list", help="text file, one wav path per line")
    ap.add_argument("--root", default=None, help="clip root for --wav-list (relative utt_ids)")
    ap.add_argument("--out-dir", required=True, help="calls on one directory must be sequential (locked)")
    ap.add_argument("--input-level", required=True, choices=("none", "sim"),
                    help="protocol decision, recorded in run_info.json. none: the recorder's level; sim: each cut "
                         "at the simulator's -26 dBFS active level with score.py's C0 gain, undone on the output")
    ap.add_argument("--allow-upsampled-input", action="store_true",
                    help="run a system on cuts below its adapter rate, under a distinct label (e.g. C2_16kin)")
    ap.add_argument("--systems", default=",".join(FROZEN), help="comma list from " + ",".join(FROZEN))
    ap.add_argument("--ckpt", action="append", default=[], metavar="SYSTEM=PATH",
                    help="other weights for a system (default: the frozen checkpoints)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="first N cuts only (smoke test)")
    args = ap.parse_args(argv)

    labels = [s for s in args.systems.split(",") if s]
    unknown = sorted(set(labels) - set(FROZEN))
    if unknown:
        raise SystemExit(f"unknown systems {unknown}; known: {list(FROZEN)}")
    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    if set(ckpts) - {s for s in labels if FROZEN[s].ckpt}:
        raise SystemExit(f"--ckpt names a system without weights or not in --systems: {sorted(ckpts)}")
    specs = [dataclasses.replace(FROZEN[s], ckpt=ckpts.get(s, FROZEN[s].ckpt)) for s in labels]

    dropped: dict[str, int] = {}
    if args.manifest:
        rows, dropped = read_c0_rows(args.manifest)
        if dropped:
            print(f"[note] kept the {BASELINE} rows only; dropped {dropped}")
    else:
        rows = rows_from_wav_list(args.wav_list, args.root)
    rows = rows[: args.limit] if args.limit else rows

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    meta = {"args": vars(args), "dropped_rows": dropped, "torch": torch.__version__}
    if args.manifest:
        meta["input_manifest_sha256"] = sha256(args.manifest)
    print(f"{len(rows)} cuts, systems {labels}, device {device}")
    run(rows, specs, Path(args.out_dir), device, meta, input_level=args.input_level,
        allow_upsampled=args.allow_upsampled_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
