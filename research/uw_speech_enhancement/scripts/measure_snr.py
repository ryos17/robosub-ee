"""Measure per-utterance SNR of imported pool takes (method signed off 2026-11-08).

Speech power from each utterance's thruster-off cut (same session, position and
channel), noise power from that position's noise-only take in the row's thruster
state, both bands (500-8000 and 50-8000 Hz) and the octaves; the drift anchor checks
speech-level stability. See `uwse.bench.snr` for the definitions.

    python scripts/measure_snr.py --align data/real/v0_s2/align.jsonl --session-dir sessions/2026-12-02_s2

With ``--session-dir`` the noise take comes from the cut's own recorder (the parent
directory of its ``raw_file``, ``raw/<rec>/``; ``--recorder`` only when that cannot be
inferred), read on the cut's ``file_channel``. For a single recorder the noise takes
can instead be given explicitly (repeatable)::

        --noise p3:off=raw/daisy/s2_p3_noise_daisy.wav --noise p3:on=raw/daisy/s2_p3_noise_on_daisy.wav

Writes next to ``--align`` (or ``--out``):

- ``snr.jsonl``  per cut: ``utt_id``, ``acoustic`` (``snr_speechband_dB``,
  ``snr_50_8000_dB``, ``snr_octave_dB``), ``snr_db_500_8000`` / ``snr_db_50_8000``
  (the keys ``scripts/score.py`` reads), the speech cut and noise take used, and
  ``flags`` (``snr_level_drift``). Cuts without a thruster-off pair or a noise take
  are listed with ``skipped``.
- ``snr_drift.json``  anchor level drift per session and the threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwse import audio  # noqa: E402
from uwse.bench import ingest  # noqa: E402
from uwse.bench import snr as SN  # noqa: E402
from uwse.bench.schema import parse_utt_id  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--align", type=Path, required=True, help="align.jsonl from import_recording.py")
    ap.add_argument("--noise", action="append", default=[], metavar="POS:STATE=PATH",
                    help="noise-only take for a position and thruster state (off|on)")
    ap.add_argument("--session-dir", type=Path, default=None,
                    help="run_session.py session dir; noise takes resolved as "
                         "raw/<rec>/s<N>_<pos>_noise[_on]_<rec>.wav")
    ap.add_argument("--recorder", default=None,
                    help="recorder for --session-dir when a cut's raw_file is not under raw/<rec>/")
    ap.add_argument("--max-drift-db", type=float, default=SN.MAX_DRIFT_DB)
    ap.add_argument("--trim-s", type=float, default=1.0, help="seconds dropped at each end of a noise take")
    ap.add_argument("--noise-gate-db", type=float, default=SN.NOISE_GATE_DB,
                    help="keep only noise-take frames within this many dB below the median level "
                         "(drops thruster-off/spin-up spans; method amendment 2026-11-08); <= 0 disables")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: the --align dir)")
    return ap.parse_args(argv)


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r.update({f"_{k}": v for k, v in parse_utt_id(r["utt_id"]).items()})
                rows.append(r)
    return rows


def recorder_of(a: argparse.Namespace, r: dict) -> str | None:
    p = Path(r.get("raw_file") or "")
    return p.parent.name if p.parent.parent.name == "raw" else a.recorder


def noise_path(a: argparse.Namespace, session: int, pos: str, state: str, rec: str | None) -> Path | None:
    given = {}
    for kv in a.noise:
        key, p = kv.split("=", 1)
        given[tuple(key.split(":", 1))] = Path(p)
    if (pos, state) in given:
        return given[(pos, state)]
    if a.session_dir and rec:
        take = f"{pos}_noise" + ("_on" if state == "on" else "")
        p = a.session_dir / "raw" / rec / f"s{session}_{take}_{rec}.wav"
        return p if p.exists() else None
    return None


def _key(r: dict) -> tuple:
    return (r["_session"], r["position_id"], r["channel"], r["librispeech_id"])


def main(argv=None) -> int:
    a = parse_args(argv)
    for kv in a.noise:
        if "=" not in kv or ":" not in kv.split("=", 1)[0] or kv.split("=", 1)[0].split(":", 1)[1] not in ("off", "on"):
            raise SystemExit(f"--noise expects POS:off=PATH or POS:on=PATH, got {kv!r}")
    rows = read_rows(a.align)
    out_dir = a.out or a.align.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    off = {}
    for r in rows:
        if r["thruster_state"] == "off" and not r.get("anchor_repeat"):
            off.setdefault(_key(r), r)

    # speech-level stability from the drift anchor, per session
    pairs = defaultdict(lambda: ([], []))
    by_state = {(_key(r), r["thruster_state"]): r for r in rows if not r.get("anchor_repeat")}
    for r in rows:
        o = by_state.get((_key(r), r["thruster_state"])) if r.get("anchor_repeat") else None
        if o is not None:
            x, _ = audio.load(o["wav"])
            y, _ = audio.load(r["wav"])
            pairs[r["_session"]][0].append(x)
            pairs[r["_session"]][1].append(y)
    drift = {}
    for s in sorted({r["_session"] for r in rows}):
        if s in pairs:
            fs = int(next(r["fs"] for r in rows if r["_session"] == s))
            d = SN.level_drift_db(*pairs[s], fs)
            drift[s] = {"drift_db": d, "n_utts": len(pairs[s][0]), "flagged": bool(abs(d) > a.max_drift_db)}
        else:
            drift[s] = {"drift_db": None, "n_utts": 0, "flagged": False, "note": "no anchor cuts: level not checked"}

    speech_cache: dict[str, dict] = {}
    noise_cache: dict[tuple, dict] = {}
    out = []
    for r in tqdm(rows, desc="snr"):
        rec = {"utt_id": r["utt_id"]}
        o = off.get(_key(r))
        npath = noise_path(a, r["_session"], r["position_id"], r["thruster_state"], recorder_of(a, r))
        if o is None or npath is None:
            rec["skipped"] = "no thruster-off cut of this utterance" if o is None else \
                f"no {r['thruster_state']} noise take for {r['position_id']}"
            out.append(rec)
            continue
        fs = int(o["fs"])
        if o["wav"] not in speech_cache:
            x, fx = audio.load(o["wav"])
            if fx != fs:
                raise SystemExit(f"{o['wav']}: {fx} Hz, align.jsonl says {fs}")
            speech_cache[o["wav"]] = SN.speech_powers(x, fs)[0]
        nk = (str(npath), int(r["file_channel"]), fs)
        if nk not in noise_cache:
            nx, _, _ = ingest.load_take(npath, fs)
            if nk[1] >= nx.shape[0]:
                raise SystemExit(f"{npath} has {nx.shape[0]} channels, cut uses file channel {nk[1]}")
            noise_cache[nk] = SN.noise_powers(nx[nk[1]], fs, trim_s=a.trim_s, gate_db=a.noise_gate_db if a.noise_gate_db > 0 else None)
        rec.update(SN.snr_fields(speech_cache[o["wav"]], noise_cache[nk]))
        rec.update(speech_cut=o["utt_id"], noise_take=str(npath), thruster_state=r["thruster_state"],
                   flags=["snr_level_drift"] if drift[r["_session"]]["flagged"] else [])
        out.append(rec)

    with open(out_dir / "snr.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    with open(out_dir / "snr_drift.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"max_drift_db": a.max_drift_db, "sessions": {str(k): v for k, v in drift.items()}}, f, indent=1)
    done = [o for o in out if "acoustic" in o]
    if done:
        v = np.array([o["snr_db_500_8000"] for o in done])
        w = np.array([o["snr_db_50_8000"] for o in done])
        print(f"{len(done)}/{len(out)} cuts measured; SNR 500-8000 Hz median {np.median(v):+.1f} dB "
              f"[{v.min():+.1f}, {v.max():+.1f}], 50-8000 Hz median {np.median(w):+.1f} dB")
    for s, d in drift.items():
        print(f"session {s}: anchor level drift "
              + (f"{d['drift_db']:+.2f} dB over {d['n_utts']} cuts" if d["drift_db"] is not None else "not checked")
              + (" -> FLAGGED snr_level_drift" if d["flagged"] else ""))
    print(f"-> {out_dir / 'snr.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
