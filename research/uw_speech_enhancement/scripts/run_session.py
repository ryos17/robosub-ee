"""Run a pool recording session: take plan, levels, playback + loopback capture, per-take QC.

Run after ``make_playlists.py`` and ``make_session_audio.py``. See
docs/RECORDING_SESSION_CHECKLIST.md for the order of work at the pool.

    # before the session (desk): plan, run sheet, utterance lists, expected levels
    python scripts/run_session.py plan   --bench data/bench/v0 --session 2 --channel-map configs/session_channels_v0.json --session-dir sessions/2026-12-02_s2
    python scripts/run_session.py levels --session-dir sessions/2026-12-02_s2 --vrms-at-0dbfs 1.23 --amp-gain-db 20

    # at the pool, once per take (plays the file, records the interface inputs, prints thruster cues, runs chirp QC)
    python scripts/run_session.py take   --session-dir sessions/2026-12-02_s2 --take-id p1_speech --range-m 1.00 ...

    # after copying the hydrophone recorders' files: QC every file of the take, then import it
    python scripts/run_session.py check  --session-dir sessions/2026-12-02_s2 --take-id p1_speech \
        --rec daisyA=raw/daisyA/s2_p1_speech_daisyA.wav --rec daisyB=raw/daisyB/s2_p1_speech_daisyB.wav --run-import

Session directory layout (``plan`` creates it):

- ``session.json``  bench path, session number, release, channel map (copied in).
- ``RUN_SHEET.md``  take order, file names, chirp counts, thruster cues, blank log fields.
- ``utterances_<pos>.tsv``  every utterance of each played file with its time in the file.
- ``raw/<recorder>/s<N>_<take_id>_<recorder>.wav``  one file per recorder per take.
- ``takes.jsonl``  append-only log: one ``take`` row per take, ``check`` rows per QC pass.
- ``import_cmds.txt``  the ``import_recording.py`` command for every checked file.

Live playback needs ``sounddevice`` (``pip install sounddevice``); ``--no-audio``
logs a take played and recorded by other software.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwse import audio  # noqa: E402
from uwse.bench import playlist as pl  # noqa: E402
from uwse.bench import session as ss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ session dir
def _load_session(sdir: Path) -> tuple[dict, list[ss.Take], ss.ChannelMap]:
    meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
    plan = ss.build_plan(meta["bench"], noise_s=meta["noise_s"], extra_noise=meta["extra_noise"],
                         extra_noise_s=meta["extra_noise_s"])
    return meta, plan, ss.ChannelMap(meta["channel_map"])


def _take(plan: list[ss.Take], take_id: str) -> ss.Take:
    for t in plan:
        if t.take_id == take_id:
            return t
    raise SystemExit(f"unknown take {take_id!r}; plan: {', '.join(t.take_id for t in plan)}")


def rec_path(sdir: Path, session: int, take_id: str, rec: str) -> Path:
    return sdir / "raw" / rec / f"s{session}_{take_id}_{rec}.wav"


def _append(sdir: Path, row: dict) -> None:
    with open(sdir / "takes.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------------------ plan
def cmd_plan(a: argparse.Namespace) -> int:
    sdir: Path = a.session_dir
    sdir.mkdir(parents=True, exist_ok=True)
    cmap_d = json.loads(a.channel_map.read_text(encoding="utf-8"))
    cmap = ss.ChannelMap(cmap_d)
    sched = pl.read_schedule(a.bench / "schedule.json")
    blocks = json.loads((a.bench / "blocks" / f"{sched[0].content_block}.json").read_text(encoding="utf-8"))
    split = blocks["bench_split"]
    plan = ss.build_plan(a.bench, noise_s=a.noise_s, extra_noise=a.extra_noise, extra_noise_s=a.extra_noise_s)
    meta = {"bench": str(a.bench.resolve()), "session": a.session, "bench_split": split,
            "out": str((a.out or ROOT / "data" / "real" / f"{split.split('-')[0]}_s{a.session}").resolve()),
            "noise_s": a.noise_s, "extra_noise": list(a.extra_noise), "extra_noise_s": a.extra_noise_s,
            "channel_map": cmap_d, "created": _now()}
    (sdir / "session.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for r in cmap.recorders:
        (sdir / "raw" / r).mkdir(parents=True, exist_ok=True)

    positions = [t.position_id for t in plan if t.kind in ("speech", "anchor")]
    for t in plan:
        if t.kind not in ("speech", "anchor"):
            continue
        rows = ss.utterance_rows(a.bench, t.position_id, anchor=t.kind == "anchor")
        cols = list(rows[0])
        lines = ["\t".join(cols)] + ["\t".join(str(r[c]) for c in cols) for r in rows]
        (sdir / f"utterances_{t.take_id}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (sdir / "RUN_SHEET.md").write_text(run_sheet(meta, plan, cmap), encoding="utf-8")
    (sdir / "session_plan.json").write_text(json.dumps(ss.plan_json(plan), indent=1), encoding="utf-8")
    total = sum(t.duration_s for t in plan) / 60
    print(f"{len(plan)} takes ({', '.join(dict.fromkeys(positions))}), {total:.0f} min of recording "
          f"-> {sdir}/RUN_SHEET.md")
    return 0


def run_sheet(meta: dict, plan: list[ss.Take], cmap: ss.ChannelMap) -> str:
    s = meta["session"]
    recs = list(cmap.recorders.values())
    chan_rows = []
    for r in recs:
        roles = {**{c: "loopback" for c in ([r.loopback] if r.loopback is not None else [])},
                 **{c: f"hydrophone ch{i}" for c, i in r.hydrophones.items()}, **r.other}
        for c in sorted(roles):
            chan_rows.append(f"| {r.name} | {r.fs} | in {c + 1} (file ch {c}) | {roles[c]} | {r.note} |")
    take_rows = []
    for t in plan:
        files = ", ".join(f"`s{s}_{t.take_id}_{r.name}.wav`" for r in recs)
        what = t.file or "(nothing; record only)"
        extra = f"{len(t.chirps_s)} chirps, {t.n_blocks} blocks, {t.n_utts} utts" if t.importable else \
            (f"{len(t.chirps_s)} chirp + sweeps" if t.kind == "sweeps" else "")
        take_rows.append(f"| {t.take_id} | {what} | {t.duration_s / 60:.1f} | {extra} | {files} | | |")
    cue_sections = []
    for t in plan:
        if t.kind in ("speech", "anchor"):
            changes = [c for c in t.cues if c.text.startswith("THRUSTER")]
            body = "\n".join(f"| {c.t_s:8.1f} | {int(c.t_s // 60)}:{c.t_s % 60:04.1f} | {c.text} |" for c in changes)
            cue_sections.append(f"### {t.take_id}: thruster cues (time from the file start)\n\n"
                                f"| t (s) | m:ss | action |\n|---|---|---|\n{body}\n")
    log_rows = "\n".join(f"| {t.take_id} | | | | | | | | |" for t in plan)
    return f"""# Run sheet: session {s} ({meta['bench_split']})

Generated by `scripts/run_session.py plan` on {meta['created']}. Bench: `{meta['bench']}`.
Imports go to `{meta['out']}`.

## Channel map

| recorder | fs (Hz) | input | role | note |
|---|---|---|---|---|
{chr(10).join(chan_rows)}

Hydrophone channel ids are the `ch<N>` in every utt_id. Scored channel: ch{cmap.scored_channel}.

## Takes, in order

Start EVERY recorder before the file starts and stop it after the file ends (both chirps of every block
must be on the recording). Name each recorder's file exactly as listed.

| take | play | min | content | recorder files | done | notes |
|---|---|---|---|---|---|---|
{chr(10).join(take_rows)}

{chr(10).join(cue_sections)}
## Per-take log

| take | time | range (m) | proj depth (m) | hyd depth (m) | water T (°C) | thruster unit / throttle % | QC | notes |
|---|---|---|---|---|---|---|---|---|
{log_rows}

Session: date ______ pool ______ amp ______ amp gain ______ dB  interface out ______ Vrms @ 0 dBFS
Recorder gains: ______  pumps on/off ______  pool activity ______  operators ______
"""


# ------------------------------------------------------------------ levels
def cmd_levels(a: argparse.Namespace) -> int:
    meta, plan, cmap = _load_session(a.session_dir)
    bench = Path(meta["bench"])
    rows, worst = [], -np.inf
    for f in dict.fromkeys(t.file for t in plan if t.file):
        x, fs = audio.load(str(bench / "session" / f))
        lv = ss.file_levels(x, fs)
        row = {"file": f, **{k: round(v, 2) for k, v in lv.items()}}
        if a.vrms_at_0dbfs and a.amp_gain_db is not None:
            row.update({k: round(v, 2) for k, v in
                        ss.drive_estimate(lv["active_rms_dBFS"], a.vrms_at_0dbfs, a.amp_gain_db).items()})
            worst = max(worst, row["SL_est_dB"])
        rows.append(row)
    for r in rows:
        print("  ".join(f"{k}={v}" for k, v in r.items()))
    if np.isfinite(worst):
        print(f"highest nominal SL {worst:.1f} dB re 1 uPa @ 1 m (target 150-160, plan 6.5; "
              f"no swimmers while transmitting; any person in the water: RL <= {ss.RL_LIMIT_DB:.0f} dB)")
        if worst > 160:
            print("WARNING: above the 160 dB plan ceiling; lower the amplifier gain")
    out = {"files": rows, "vrms_at_0dbfs": a.vrms_at_0dbfs, "amp_gain_db": a.amp_gain_db, "time": _now()}
    if a.live:
        out["live"] = live_level(a, meta, plan, cmap)
    (a.session_dir / "levels.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def live_level(a: argparse.Namespace, meta: dict, plan: list[ss.Take], cmap: ss.ChannelMap) -> dict:
    """Play the first speech block's opening seconds and measure the loopback."""
    rec = _iface(cmap, a.iface)
    t = next(t for t in plan if t.kind == "speech")
    x, fs = audio.load(str(Path(meta["bench"]) / "session" / t.file))
    x = x[: int(a.live_s * fs)]
    y = _playrec(x, fs, rec, a)
    lv = ss.file_levels(y[rec.loopback], rec.fs)
    res = {"file": t.file, "seconds": a.live_s, "loopback_active_rms_dBFS": round(lv["active_rms_dBFS"], 2),
           "loopback_peak_dBFS": round(lv["peak_dBFS"], 2)}
    if a.in_dbfs_at_1vrms is not None:
        v = ss.loopback_drive(lv["active_rms_dBFS"], a.in_dbfs_at_1vrms, a.loopback_atten_db)
        res.update({"drive_vrms_measured": round(v, 3),
                    "SL_est_dB": round(ss.LL916_SL_DB_AT_20VRMS + 20 * np.log10(v / 20), 1)})
    print("live:", res)
    return res


# ------------------------------------------------------------------ audio I/O
def _iface(cmap: ss.ChannelMap, name: str | None) -> ss.Recorder:
    if name:
        return cmap.recorders[name]
    lb = [r for r in cmap.recorders.values() if r.loopback is not None]
    if not lb:
        raise SystemExit("no recorder with a loopback in the channel map; pass --iface")
    return lb[0]


def _playrec(x: np.ndarray, fs: int, rec: ss.Recorder, a: argparse.Namespace,
             cues: list[ss.Cue] | None = None) -> np.ndarray:
    """Play ``x`` on output channel ``--out-channel`` with pre/post roll; return (C, T) inputs of ``rec``."""
    import sounddevice as sd

    if fs != rec.fs:
        x = audio.resample(x, fs, rec.fs)
    pre, post = int(a.pre_roll_s * rec.fs), int(a.post_roll_s * rec.fs)
    play = np.concatenate([np.zeros(pre), x, np.zeros(post)]).astype(np.float32)[:, None]
    n_in = 1 + max([rec.loopback or 0, *rec.hydrophones, *rec.other])
    y = sd.playrec(play, samplerate=rec.fs, input_mapping=list(range(1, n_in + 1)),
                   output_mapping=[a.out_channel], device=a.device, blocking=False)
    t0 = time.monotonic()
    for c in cues or []:
        wait = a.pre_roll_s + c.t_s - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)
        print(f"[{time.monotonic() - t0:7.1f} s] {c.text}", flush=True)
    sd.wait()
    return np.asarray(y, float).T


# ------------------------------------------------------------------ take
METADATA = ("range_m", "proj_depth_m", "hyd_depth_m", "water_temp_c", "amp_gain_db", "rec_gain_db", "pumps",
            "pool_activity", "thruster_unit", "throttle_pct", "operator", "notes")


def cmd_take(a: argparse.Namespace) -> int:
    sdir = a.session_dir
    meta, plan, cmap = _load_session(sdir)
    t = _take(plan, a.take_id)
    rec = _iface(cmap, a.iface)
    path = rec_path(sdir, meta["session"], t.take_id, rec.name)
    started = _now()
    if a.no_audio:
        print(f"{t.take_id}: --no-audio, logging only")
    else:
        if path.exists() and not a.overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite or use a new --take-id suffix")
        if t.file:
            x, fs = audio.load(str(Path(meta["bench"]) / "session" / t.file))
        else:
            fs, x = rec.fs, np.zeros(int(t.duration_s * rec.fs))
        print(f"{t.take_id}: playing {t.file or '(silence)'} ({t.duration_s / 60:.1f} min) on {rec.name}")
        y = _playrec(x, fs, rec, a, cues=t.cues)
        path.parent.mkdir(parents=True, exist_ok=True)
        audio.save(str(path), y, rec.fs, subtype="PCM_24")
    row = {"type": "take", "take_id": t.take_id, "kind": t.kind, "position_id": t.position_id,
           "played_file": t.file, "started": started, "session": meta["session"],
           "recordings": {rec.name: str(path)} if not a.no_audio else {},
           "pre_roll_s": a.pre_roll_s, "post_roll_s": a.post_roll_s,
           **{k: getattr(a, k) for k in METADATA}}
    for spec in a.rec or []:
        name, p = spec.split("=", 1)
        row["recordings"][name] = p
    if t.importable and not a.no_audio and rec.loopback is not None:
        q = ss.qc_take(path, meta["bench"], t.position_id, t.kind == "anchor", rec.loopback)
        row["qc"] = {rec.name: q}
        print(("PASS" if q["ok"] else "FAIL -> redo the take") + f" {rec.name} loopback: {q}")
    _append(sdir, row)
    return 0


# ------------------------------------------------------------------ check / import
def cmd_check(a: argparse.Namespace) -> int:
    sdir = a.session_dir
    meta, plan, cmap = _load_session(sdir)
    t = _take(plan, a.take_id)
    recs = {}
    for line in (sdir / "takes.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["take_id"] == t.take_id and r["type"] == "take":
            recs.update(r["recordings"])
    for spec in a.rec or []:
        name, p = spec.split("=", 1)
        recs[name] = p
    if not recs:
        raise SystemExit(f"no recordings logged for {t.take_id}; pass --rec NAME=PATH")
    results, cmds, all_ok = {}, [], True
    for name, p in recs.items():
        rec = cmap.recorders[name]
        p = Path(p)
        if not p.is_absolute() and not p.exists() and (sdir / p).exists():
            p = sdir / p
        if not t.importable:
            results[name] = {"ok": p.exists(), "file": str(p)}
            continue
        q = ss.qc_take(p, meta["bench"], t.position_id, t.kind == "anchor", cmap.sync_channel(rec))
        results[name] = q
        all_ok &= q["ok"]
        print(("PASS" if q["ok"] else "FAIL") + f" {t.take_id} {name}: {q}")
        if rec.hydrophones:
            cmds.append(ss.import_argv(t, rec, p, cmap, meta["bench"], meta["session"], meta["out"]))
    _append(sdir, {"type": "check", "take_id": t.take_id, "time": _now(), "qc": results})
    if cmds:
        with open(sdir / "import_cmds.txt", "a", encoding="utf-8", newline="\n") as f:
            for argv in cmds:
                f.write("python scripts/import_recording.py " + " ".join(shlex.quote(x) for x in argv) + "\n")
    if a.run_import:
        if not all_ok and not a.force:
            print("QC failed: not importing (pass --force to import anyway; failed blocks are flagged)")
            return 1
        import importlib.util

        spec = importlib.util.spec_from_file_location("import_recording", ROOT / "scripts" / "import_recording.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for argv in cmds:
            mod.main(argv)
    return 0 if all_ok else 1


# ------------------------------------------------------------------ CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="take plan, run sheet and utterance lists")
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--session", type=int, required=True, help="session number in utt_ids (s<N>)")
    p.add_argument("--channel-map", type=Path, required=True)
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="import output dir (default data/real/<release>_s<N>)")
    p.add_argument("--noise-s", type=float, default=60.0, help="noise-only take per position (s)")
    p.add_argument("--extra-noise", nargs="*", default=list(ss.EXTRA_NOISE))
    p.add_argument("--extra-noise-s", type=float, default=90.0)

    def audio_args(p):
        p.add_argument("--session-dir", type=Path, required=True)
        p.add_argument("--iface", default=None, help="recorder name of the playback interface (default: the one "
                                                     "with a loopback)")
        p.add_argument("--device", default=None, help="sounddevice device name or index")
        p.add_argument("--out-channel", type=int, default=1, help="interface output feeding the amplifier (1-based)")
        p.add_argument("--pre-roll-s", type=float, default=2.0)
        p.add_argument("--post-roll-s", type=float, default=3.0)

    p = sub.add_parser("levels", help="file levels, drive and nominal source level; --live measures the loopback")
    audio_args(p)
    p.add_argument("--vrms-at-0dbfs", type=float, default=None, help="interface output Vrms for a 0 dBFS sine")
    p.add_argument("--amp-gain-db", type=float, default=None, help="amplifier voltage gain (dB)")
    p.add_argument("--live", action="store_true")
    p.add_argument("--live-s", type=float, default=15.0)
    p.add_argument("--in-dbfs-at-1vrms", type=float, default=None, help="interface input level for 1 Vrms")
    p.add_argument("--loopback-atten-db", type=float, default=40.0)

    p = sub.add_parser("take", help="play one take, record the interface, QC the loopback, log metadata")
    audio_args(p)
    p.add_argument("--take-id", required=True)
    p.add_argument("--no-audio", action="store_true", help="played/recorded elsewhere; log metadata only")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--rec", action="append", help="other recorders' files, NAME=PATH (can also go to check)")
    p.add_argument("--range-m", type=float)
    p.add_argument("--proj-depth-m", type=float)
    p.add_argument("--hyd-depth-m", type=float)
    p.add_argument("--water-temp-c", type=float)
    p.add_argument("--amp-gain-db", type=float)
    p.add_argument("--rec-gain-db", type=float)
    p.add_argument("--pumps", choices=("on", "off"))
    p.add_argument("--pool-activity", choices=("empty", "swimmers"), default="empty")
    p.add_argument("--thruster-unit", choices=("A", "B"))
    p.add_argument("--throttle-pct", type=float)
    p.add_argument("--operator")
    p.add_argument("--notes", default="")

    p = sub.add_parser("check", help="QC every recorder file of a take; write (and run) the import commands")
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--take-id", required=True)
    p.add_argument("--rec", action="append", help="NAME=PATH, added to the files logged by take")
    p.add_argument("--run-import", action="store_true")
    p.add_argument("--force", action="store_true", help="import even if QC failed")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    return {"plan": cmd_plan, "levels": cmd_levels, "take": cmd_take, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
