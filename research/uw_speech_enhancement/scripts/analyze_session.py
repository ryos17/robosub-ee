"""Same-day analysis of a session's sweep and noise takes: IRs, EDC, DRR, C50; noise spectra.

Run after the session (CPU only), on the session directory that ``run_session.py``
wrote. Recorder files come from ``takes.jsonl`` (as ``check`` reads them) or, if not
logged, from the standard name ``raw/<rec>/s<N>_<take>_<rec>.wav``. See
`uwse.bench.postsession` for the methods.

    python scripts/analyze_session.py sweeps --session-dir sessions/2026-11-09_s1
    python scripts/analyze_session.py noise  --session-dir sessions/2026-11-09_s1
    python scripts/analyze_session.py all    --session-dir sessions/2026-11-09_s1 --take p3_sweeps

Writes under ``<session-dir>/analysis/`` (or ``--out``):

- ``sweeps.jsonl``  one row per sweep take x recorder x hydrophone: drift, offset, sync
  channel, arrivals, broadband and per-octave EDT/T20/T30, C50, DRR (null where the
  gate rule fails), single/two-slope fit, IR SNR, sweep coherence and variance.
- ``ir/<take>_<rec>_ch<id>.npz``  per-sweep IRs, the mean IR, broadband and per-octave EDCs.
- ``noise.jsonl``  one row per noise take x recorder x hydrophone: kept frames,
  broadband and SNR-method band levels (dBFS), 1/3-octave levels and their spread.
- ``noise/<take>_<rec>_ch<id>.npz``  mean PSD and 1/3-octave arrays.
- ``noise_compare.json``  per recorder and channel, each condition's band levels
  relative to ``--noise-ref`` (default ``noise_quiet_pool``, else the first ``<pos>_noise``).

Levels are uncalibrated (dBFS through the recording chain); compare within a channel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwse import audio  # noqa: E402
from uwse.bench import ingest  # noqa: E402
from uwse.bench import postsession as PS  # noqa: E402
from uwse.bench import session as ss  # noqa: E402
from uwse.bench import snr as SN  # noqa: E402

NOISE_KINDS = ("noise", "noise_bank")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("sweeps", "noise", "all"))
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--take", action="append", default=None, help="only these take ids (repeatable)")
    ap.add_argument("--rec", action="append", default=[], metavar="TAKE:NAME=PATH",
                    help="a recorder file not logged in takes.jsonl and not at the standard path")
    ap.add_argument("--out", type=Path, default=None, help="default <session-dir>/analysis")
    ap.add_argument("--max-ppm", type=float, default=1000.0, help="largest clock drift searched (sweeps)")
    ap.add_argument("--drr-gate-ms", type=float, default=PS.DRR_GATE_S * 1e3)
    ap.add_argument("--trim-s", type=float, default=1.0, help="seconds dropped at each end of a noise take")
    ap.add_argument("--noise-gate-db", type=float, default=SN.NOISE_GATE_DB,
                    help="frame gate below the median level, as measure_snr.py; <= 0 disables")
    ap.add_argument("--noise-ref", default=None, help="reference take for noise_compare.json")
    return ap.parse_args(argv)


def recordings(sdir: Path, meta: dict, plan: list[ss.Take], cmap: ss.ChannelMap,
               extra: list[str]) -> dict[str, dict[str, Path]]:
    """take_id -> recorder -> file: takes.jsonl rows, then --rec, then the standard path if it exists."""
    out: dict[str, dict[str, Path]] = {}
    log = sdir / "takes.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("type") == "take":
                    out.setdefault(r["take_id"], {}).update({k: Path(v) for k, v in r["recordings"].items()})
    for spec in extra:
        tk, rest = spec.split(":", 1)
        name, p = rest.split("=", 1)
        out.setdefault(tk, {})[name] = Path(p)
    for t in plan:
        for name in cmap.recorders:
            p = sdir / "raw" / name / f"s{meta['session']}_{t.take_id}_{name}.wav"
            if name not in out.get(t.take_id, {}) and p.exists():
                out.setdefault(t.take_id, {})[name] = p
    for tk, recs in out.items():
        for name, p in recs.items():
            if not p.is_absolute() and not p.exists() and (sdir / p).exists():
                recs[name] = sdir / p
    return out


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, np.ndarray):
        return _jsonable(d.tolist())
    if isinstance(d, (np.floating, float)):
        return float(d) if np.isfinite(d) else None
    if isinstance(d, np.integer):
        return int(d)
    return d


def _files(t: ss.Take, recs: dict[str, dict[str, Path]], cmap: ss.ChannelMap, rows: list[dict]):
    """(name, recorder, path) of every hydrophone recorder of a take; an error row for each missing file."""
    for name, rec in cmap.recorders.items():
        if not rec.hydrophones:
            continue
        path = recs.get(t.take_id, {}).get(name)
        base = {"take_id": t.take_id, "recorder": name, "file": str(path) if path else None, "setup": t.setup}
        if path is None or not path.exists():
            err = ("no recording found (not in takes.jsonl or --rec, not at the standard raw path)"
                   if path is None else "file missing")
            rows.append({**base, "error": err})
            print(f"{t.take_id} {name}: {rows[-1]['error']}")
            continue
        yield name, rec, path, base


def run_sweeps(a, meta, takes, recs, cmap, out: Path) -> list[dict]:
    bench = Path(meta["bench"])
    sw = json.loads((bench / "session" / "sweeps.json").read_text(encoding="utf-8"))
    fs = sw["fs"]
    rows: list[dict] = []
    (out / "ir").mkdir(parents=True, exist_ok=True)
    for t in tqdm(takes, desc="sweep takes"):
        for name, rec, path, base in list(_files(t, recs, cmap, rows)):
            base["gain_db"] = t.gain_db
            x, fs_raw, _ = ingest.load_take(path, fs)
            sync = cmap.sync_channel(rec)
            try:
                res = PS.sweep_irs(x, fs, sw, rec.hydrophones, sync, max_ppm=a.max_ppm)
            except ValueError as e:
                rows.append({**base, "error": str(e)})
                print(f"{t.take_id} {name}: {e}")
                continue
            for i, cid in enumerate(res.channel_ids):
                row = {**base, "channel_id": cid, "fs_raw": fs_raw, "fs": fs, "sync_channel": sync,
                       "sync_is_loopback": rec.loopback == sync, "drift_ppm": res.ppm, "offset_s": res.offset_s,
                       "drift_fit_resid_samples": res.resid_samples, "ir_pre_s": res.n_pre / fs}
                try:
                    m, arrays = PS.ir_metrics(res.irs[i], fs, res.n_pre, (sw["f1"], sw["f2"]),
                                              drr_gate_s=a.drr_gate_ms * 1e-3)
                except Exception as e:  # noqa: BLE001 - one bad channel must not lose the session's other rows
                    rows.append({**row, "error": f"ir_metrics: {type(e).__name__}: {e}"})
                    print(f"{t.take_id} {name} ch{cid}: {rows[-1]['error']}")
                    np.savez_compressed(out / "ir" / f"{t.take_id}_{name}_ch{cid}.npz", irs=res.irs[i], fs=fs,
                                        n_pre=res.n_pre)
                    continue
                rows.append({**row, **m})
                np.savez_compressed(out / "ir" / f"{t.take_id}_{name}_ch{cid}.npz", irs=res.irs[i], fs=fs,
                                    n_pre=res.n_pre, **arrays)
                bb = m["broadband"]
                print(f"{t.take_id} {name} ch{cid}: drift {res.ppm:+.2f} ppm (resid {res.resid_samples:.2f} smp), "
                      f"T20 {bb['T20_s']}, C50 {bb['C50_dB']}, DRR {bb['DRR_dB']}, slopes {bb['n_slopes']}")
    return rows


def run_noise(a, meta, takes, recs, cmap, out: Path) -> tuple[list[dict], dict]:
    rows, spectra = [], {}
    (out / "noise").mkdir(parents=True, exist_ok=True)
    gate = a.noise_gate_db if a.noise_gate_db > 0 else None
    for t in tqdm(takes, desc="noise takes"):
        for name, rec, path, base in list(_files(t, recs, cmap, rows)):
            base["kind"] = t.kind
            x, fs = audio.load(str(path), always_2d=True)
            for c, cid in sorted(rec.hydrophones.items()):
                try:
                    s = PS.noise_spectrum(x[c], fs, trim_s=a.trim_s, gate_db=gate)
                except Exception as e:  # noqa: BLE001 - keep the other channels and takes
                    rows.append({**base, "channel_id": cid, "error": f"{type(e).__name__}: {e}"})
                    continue
                spectra.setdefault((name, cid), {})[t.take_id] = s
                rows.append({**base, "channel_id": cid, "fs": fs, "duration_s": s["duration_s"],
                             "frames": s["frames"], "frames_kept": s["frames_kept"],
                             "broadband_dBFS": s["broadband_db"], "band_dBFS": s["band_db"],
                             "third_octave_fc": s["third_octave_fc"], "third_octave_dBFS": s["third_octave_db"],
                             "third_octave_std_db": s["third_octave_std_db"]})
                np.savez_compressed(out / "noise" / f"{t.take_id}_{name}_ch{cid}.npz",
                                    **{k: v for k, v in s.items() if isinstance(v, np.ndarray)}, fs=fs)
                print(f"{t.take_id} {name} ch{cid}: {s['broadband_db']:.1f} dBFS, 500-8000 "
                      f"{s['band_db']['500-8000']:.1f} dBFS, kept {s['frames_kept']}/{s['frames']} frames")
    compare = {}
    for (name, cid), sp in spectra.items():
        ref = a.noise_ref or ("noise_quiet_pool" if "noise_quiet_pool" in sp else
                              next((k for k in sp if k.endswith("_noise")), None))
        if ref is None or ref not in sp or len(sp) < 2:
            continue
        try:
            compare[f"{name}_ch{cid}"] = PS.compare_spectra(sp, ref)
        except ValueError as e:
            compare[f"{name}_ch{cid}"] = {"error": str(e)}
    return rows, compare


def main(argv=None) -> int:
    a = parse_args(argv)
    sdir = a.session_dir
    meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    plan = ss.build_plan(meta["bench"], noise_s=meta["noise_s"], extra_noise=meta["extra_noise"],
                         extra_noise_s=meta["extra_noise_s"], groups=meta.get("groups"))
    cmap = ss.ChannelMap(meta["channel_map"])
    if a.take:
        unknown = set(a.take) - {t.take_id for t in plan}
        if unknown:
            raise SystemExit(f"unknown takes {sorted(unknown)}")
        plan = [t for t in plan if t.take_id in a.take]
    recs = recordings(sdir, meta, plan, cmap, a.rec)
    out = a.out or sdir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    if a.what in ("sweeps", "all"):
        takes = [t for t in plan if t.kind == "sweeps"]
        rows = run_sweeps(a, meta, takes, recs, cmap, out)
        (out / "sweeps.jsonl").write_text("".join(json.dumps(_jsonable(r)) + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} sweep rows -> {out / 'sweeps.jsonl'}")
    if a.what in ("noise", "all"):
        takes = [t for t in plan if t.kind in NOISE_KINDS]
        rows, compare = run_noise(a, meta, takes, recs, cmap, out)
        (out / "noise.jsonl").write_text("".join(json.dumps(_jsonable(r)) + "\n" for r in rows), encoding="utf-8")
        (out / "noise_compare.json").write_text(json.dumps(_jsonable(compare), indent=1), encoding="utf-8")
        print(f"{len(rows)} noise rows -> {out / 'noise.jsonl'}")
    print("rows with an \"error\" field name every take/recorder that was missing or failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
