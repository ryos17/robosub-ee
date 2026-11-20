"""Run a pool recording session: take plan, levels, playback, Daisy commands, file check, QC, import.

Run after ``make_playlists.py`` and ``make_session_audio.py``. Pool plan (2026-11-19, no
thruster, no loopback; bench from ``make_playlists.py --plan pool``)::

    # desk, before the day: plan, run sheet, utterance lists, Daisy command lines
    python scripts/run_session.py plan --bench data/bench/v1 --session 2 \
        --channel-map configs/session_channels_v1.json --session-dir sessions/2026-12-02_s2 \
        --distances 1 3 6 --pool-length-m 10 --pool-width-m 5 --pool-depth-m 2 --date 2026-12-02
    python scripts/run_session.py levels --session-dir sessions/2026-12-02_s2 --target-sl-db 160 --brick-v 48

    # at the pool: set the level with the 1 kHz tone (DMM across the amp terminals), then check it
    python scripts/run_session.py tone   --session-dir sessions/2026-12-02_s2
    python scripts/run_session.py levels --session-dir sessions/2026-12-02_s2 --tone-vrms 1.80 --brick-v 48

    # per take: start the two record_daisy.py lines on the Linux capture machine (copied from
    # RUN_SHEET.md / daisy_cmds.txt, or printed on the PC by `daisy-cmds`; the Linux machine has
    # no session dir or bench, so run_session.py does not run there), then on the Windows PC
    python scripts/run_session.py take --session-dir sessions/2026-12-02_s2 --take-id p3_speech \
        --distance-m 3.02 --speaker-depth-m 1.2 --hyd-depth-m 1.2 --water-temp-c 27 --amp-vrms 1.80 ...
    # copy the Daisy files to <session>/raw/daisyA|daisyB/ (dry run: <session>/dryrun/raw/...), then
    python scripts/run_session.py verify --session-dir sessions/2026-12-02_s2 --take-id p3_speech
    python scripts/run_session.py check  --session-dir sessions/2026-12-02_s2 --take-id p3_speech [--run-import]

``check`` routes every import by take (`uwse.bench.session.import_subdir`): the per-distance
``p<d>_speech`` / ``p<d>_anchor`` takes go to the import dir ``<out>`` (the quiet condition),
``ref_speech`` to ``<out>/chainref``, ``pilot_speech`` to ``<out>/pilot`` and the bubble pass to
``<out>/bubble``, so ``--run-import`` is safe on every take and no ``--out`` is edited by hand.

A redo of a take is attempt 2, 3, ...: ``--attempt 2`` on ``daisy-cmds`` / ``take`` names the files
``s<N>_<take>_r2_<rec>.wav``; nothing is ever overwritten. ``daisy_cmds.txt`` also lists the
attempt-2 lines. ``verify`` and ``check`` default to the latest attempt on disk. ``check`` keeps
one import line per take x recorder in ``import_cmds.txt``: the latest attempt that PASSED QC (a
failing attempt is written only as a ``# QC FAIL`` comment). Re-running ``check`` rewrites the file
(no duplicates) and re-importing replaces the take's rows in ``align.jsonl``.

Session directory layout (``plan`` creates it):

- ``session.json``  plan options, bench and import paths (relative to the project root), channel map.
- ``RUN_SHEET.md``  safety and level steps, take order, Daisy lines, per-take log columns.
- ``daisy_cmds.txt``  the ``record_daisy.py`` lines of every take (attempt 1, then the attempt-2 redo lines).
- ``utterances_<take>.tsv``  every utterance of each played file with its time in the file.
- ``raw/<recorder>/s<N>_<take>[_r<k>]_<recorder>.wav``  one file per Daisy board per take;
  the dry run goes to ``dryrun/raw/...`` instead.
- ``takes.jsonl``  append-only log: ``take`` rows (metadata), ``verify`` and ``check`` rows.
- ``import_cmds.json`` / ``import_cmds.txt``  the ``import_recording.py`` command per take x recorder
  (latest QC-passing attempt, ``--out`` already routed by take).

Legacy thruster benches (no ``session/programs.json``, e.g. the frozen data/bench/v0_s1) keep the
old per-position plan; their ``take`` records the interface inputs when the map has a loopback.
Live playback needs ``sounddevice``; ``--no-audio`` logs a take played by other software.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shlex
import sys
import time
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
    return meta, ss.plan_from_meta(meta), ss.ChannelMap(meta["channel_map"])


def _bench(meta: dict) -> Path:
    return ss.resolve_path(meta["bench"])


def _out(meta: dict) -> Path:
    return ss.resolve_path(meta["out"])


def _take(plan: list[ss.Take], take_id: str) -> ss.Take:
    for t in plan:
        if t.take_id == take_id:
            return t
    raise SystemExit(f"unknown take {take_id!r}; plan: {', '.join(t.take_id for t in plan)}")


def rec_path(sdir: Path, session: int, take_id: str, rec: str, attempt: int = 1, dryrun: bool = False) -> Path:
    return ss.raw_path(sdir, session, take_id, rec, attempt, dryrun=dryrun)


def _append(sdir: Path, row: dict) -> None:
    with open(sdir / "takes.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


def _log_rows(sdir: Path) -> list[dict]:
    p = sdir / "takes.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _linux_sdir(sdir: Path) -> str:
    """The session dir as the Linux machine sees it from the project root (relative when inside it)."""
    p = ss.portable_path(sdir, ROOT)
    return p if not Path(p).is_absolute() else Path(p).as_posix()


# ------------------------------------------------------------------ plan
def cmd_plan(a: argparse.Namespace) -> int:
    sdir: Path = a.session_dir
    cmap_d = json.loads(a.channel_map.read_text(encoding="utf-8"))
    cmap = ss.ChannelMap(cmap_d)
    pool = ss.is_pool_bench(a.bench)
    groups = take_groups(a)
    if pool and groups:
        raise SystemExit("--e2/--e3/--e4 are legacy thruster-plan groups; the pool plan has its own takes")
    if not pool and (a.distances is not None):
        raise SystemExit(f"{a.bench} is a legacy per-position bench (no session/programs.json); "
                         "--distances needs a pool-plan bench (make_playlists.py --plan pool)")
    if (sdir / "session.json").exists() and not a.replan:
        raise SystemExit(f"{sdir / 'session.json'} exists; pass --replan to rewrite the plan files "
                         "(recordings and takes.jsonl are kept)")
    sdir.mkdir(parents=True, exist_ok=True)
    sched = pl.read_schedule(a.bench / "schedule.json")
    split = json.loads((a.bench / "blocks" / f"{sched[0].content_block}.json").read_text(encoding="utf-8"))["bench_split"]
    meta = {"bench": ss.portable_path(a.bench, ROOT), "session": a.session, "bench_split": split,
            "channel_map": cmap_d, "created": _now()}
    if pool:
        opts = {"distances_m": a.distances or ss.POOL_DEFAULTS["distances_m"], "ref_m": a.ref_m,
                "pilot_m": a.pilot_m, "bubble_m": None if a.no_bubbles else a.bubble_m, "noise_s": a.noise_s,
                "bubble_noise_s": a.bubble_noise_s, "dry_run": not a.no_dry_run}
        plan = ss.build_pool_plan(a.bench, **opts)
        out = a.out or ROOT / "data" / "real" / f"{a.bench.resolve().name}_s{a.session}"
        meta.update({"plan": "pool", "pool": opts, "out": ss.portable_path(out, ROOT),
                     "header": {"date": a.date, "pool_length_m": a.pool_length_m, "pool_width_m": a.pool_width_m,
                                "pool_depth_m": a.pool_depth_m, "pool": a.pool_name}})
    else:
        plan = ss.build_plan(a.bench, noise_s=a.noise_s, extra_noise=a.extra_noise, extra_noise_s=a.extra_noise_s,
                             groups=groups)
        out = a.out or ROOT / "data" / "real" / f"{split.split('-')[0]}_s{a.session}"
        meta.update({"plan": "legacy", "out": ss.portable_path(out, ROOT), "noise_s": a.noise_s,
                     "extra_noise": list(a.extra_noise), "extra_noise_s": a.extra_noise_s, "groups": groups})
    (sdir / "session.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for r in cmap.recorders.values():
        if r.records:
            (sdir / "raw" / r.name).mkdir(parents=True, exist_ok=True)
            if pool and any(t.dryrun for t in plan):
                (sdir / "dryrun" / "raw" / r.name).mkdir(parents=True, exist_ok=True)

    for t in plan:
        if t.kind not in ("speech", "anchor") or not t.importable:
            continue
        rows = ss.utterance_rows(a.bench, t.position_id, anchor=t.kind == "anchor", played=t.played)
        cols = list(rows[0])
        lines = ["\t".join(cols)] + ["\t".join(str(r[c]) for c in cols) for r in rows]
        (sdir / f"utterances_{t.take_id}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if pool:
        lsd = _linux_sdir(sdir)
        blocks = [f"# {t.take_id} ({t.duration_s / 60:.1f} min)\n" + "\n".join(
            ss.daisy_commands(t, cmap, a.session, lsd)) for t in plan]
        redo = [f"# {ss.take_label(t.take_id, 2)} (redo, attempt 2: then `take --attempt 2`)\n" + "\n".join(
            ss.daisy_commands(t, cmap, a.session, lsd, attempt=2)) for t in plan]
        (sdir / "daisy_cmds.txt").write_text(
            "\n\n".join(blocks) + "\n\n# ---------------- redo lines (attempt 2; for attempt k write _r<k> in "
            "--out)\n\n" + "\n\n".join(redo) + "\n", encoding="utf-8")
        sheet = pool_run_sheet(meta, plan, cmap, lsd)
    else:
        sheet = run_sheet(meta, plan, cmap)
    (sdir / "RUN_SHEET.md").write_text(sheet, encoding="utf-8")
    (sdir / "session_plan.json").write_text(json.dumps(ss.plan_json(plan), indent=1), encoding="utf-8")
    total = sum(t.duration_s for t in plan) / 60
    longest = max(plan, key=lambda t: t.duration_s)
    print(f"{len(plan)} takes, {total:.0f} min of recording (longest {longest.take_id} "
          f"{longest.duration_s / 60:.1f} min) -> {sdir / 'RUN_SHEET.md'}")
    return 0


def take_groups(a: argparse.Namespace) -> dict:
    """The optional legacy E2/E3/E4 groups asked for on the command line (empty = none)."""
    g: dict = {}
    if a.e2:
        g["e2"] = {"depths_m": a.e2_depths_m, "angles_deg": a.e2_angles_deg, "atten_db": a.e2_atten_db}
    if a.e3_points:
        g["e3"] = {"points": a.e3_points, "repeat_points": a.e3_repeat_points}
    if a.e4:
        g["e4"] = {}
    return g


def _chan_rows(cmap: ss.ChannelMap) -> list[str]:
    rows = []
    for r in cmap.recorders.values():
        roles = {**{c: "loopback" for c in ([r.loopback] if r.loopback is not None else [])},
                 **{c: f"hydrophone ch{i}" for c, i in r.hydrophones.items()}, **r.other}
        if not roles:
            rows.append(f"| {r.name} | {r.fs} | (none) | plays only | {r.note} |")
        for c in sorted(roles):
            rows.append(f"| {r.name} | {r.fs} | in {c + 1} (file ch {c}) | {roles[c]} | {r.note} |")
    return rows


POOL_SAFETY = """\
- The amplifier is BRIDGED (BTL): never tie a speaker terminal, the AC205C primary or the cable
  shield to any ground (interface, laptop, RCA shell, meter). Use one channel's + and - to the
  AC205C primary. Measure across those two terminals with a floating, battery-powered DMM.
- Mains brick and amplifier on a TESTED GFCI, on dry ground well back from the water.
- Power up: amplifier knob at MINIMUM, interface first, then the amplifier (power down in reverse).
- Projector at least 1 m below the surface (Lubell: 1-18 m). Speaker and hydrophone at the same
  depth, hydrophone on the speaker's horizontal axis, projector rotation fixed and marked for the
  whole session (5 kHz notch and clover-leaf polar pattern near 5 kHz).
- NOBODY IN THE WATER while the projector plays (otherwise received level <= ~150 dB re 1 uPa).
- Do NOT pre-compensate the 5 kHz notch; scoring uses the measured chain reference (ref takes)."""


def pool_run_sheet(meta: dict, plan: list[ss.Take], cmap: ss.ChannelMap, linux_sdir: str) -> str:
    s, h = meta["session"], meta.get("header", {})

    def hv(k: str, unit: str = "") -> str:
        v = h.get(k)
        return f"{v:g}{unit}" if isinstance(v, (int, float)) else (str(v) if v else "______")

    recs = [r for r in cmap.recorders.values() if r.hydrophones]
    take_rows = []
    for t in plan:
        what = t.file or "(nothing played; record only)"
        files = ", ".join(f"`{ss.raw_path('', s, t, r.name).as_posix()}`" for r in recs)
        if t.importable or t.dryrun:
            content = f"{len(t.chirps_s)} chirps, {t.n_blocks} blocks, {t.n_utts} utts"
        elif t.kind == "sweeps":
            content = f"{len(t.chirps_s)} chirp + sweeps"
        else:
            content = "noise only"
        if t.setup:
            content += f"; {t.setup}"
        dist = "-" if t.distance_m is None else f"{t.distance_m:g}"
        take_rows.append(f"| {t.take_id} | {t.position_id} | {dist} | {what} | {t.duration_s / 60:.1f} | "
                         f"{ss.record_seconds(t)} | {content} | {files} |")
    cmds = []
    for t in plan:
        lines = "\n".join(ss.daisy_commands(t, cmap, s, linux_sdir))
        win = f"python scripts/run_session.py take --session-dir {linux_sdir} --take-id {t.take_id}"
        cmds.append(f"### {t.take_id}\n\nLinux (repository's project root; type or paste these), start "
                    f"first:\n\n```bash\n{lines}\n```\n\n"
                    f"Windows, within {20:d} s:\n\n```\n{win} --distance-m ... --speaker-depth-m ... "
                    f"--hyd-depth-m ... --water-temp-c ... --amp-vrms ... --brick-v ... --pump "
                    f"{'on' if t.bubbles else 'off'}\n```\n")
    cols = ("take", "attempt", "time", "distance (m)", "speaker depth (m)", "hydrophone depth (m)", "orientation",
            "water T (°C)", "amp Vrms (measured)", "brick V", "recorder gains", "pump on/off", "QC", "notes")
    log_rows = "\n".join(f"| {t.take_id} |" + " |" * (len(cols) - 1) for t in plan)
    pilot = next((t for t in plan if t.take_id == "pilot_speech"), None)
    ref = next((t for t in plan if t.take_id == "ref_speech"), None)
    dists = ", ".join(f"{d:g} m" for d in meta["pool"]["distances_m"])
    return f"""# Run sheet: session {s} (pool plan, no thruster)

Generated by `scripts/run_session.py plan` on {meta['created']}. Bench: `{meta['bench']}`.
Imports go to `{meta['out']}`. Distances: {dists}.

Session: date {hv('date')}  pool {hv('pool')}  length {hv('pool_length_m', ' m')}  width {hv('pool_width_m', ' m')}
depth {hv('pool_depth_m', ' m')}  operators ______  amp knob position (taped) ______

## Safety (read before powering anything)

{POOL_SAFETY}

## Level setting (once, before take 1; the knob is never touched again)

1. `python scripts/run_session.py levels --session-dir <this dir> --target-sl-db 160 --brick-v <brick V>`
   prints the tone Vrms that puts speech at the target (about 2.0 Vrms speech-active for 160 dB
   re 1 uPa @ 1 m, 0.63 Vrms for 150 dB) and the PREDICTED peak drive at that setting; on a
   peak-vs-brick warning lower `--target-sl-db` (desk, before the day).
2. `python scripts/run_session.py tone --session-dir <this dir>` plays the 1 kHz tone; raise the knob
   from minimum ONLY until the DMM reads that Vrms; tape the knob. Hard cap 20 Vrms on the LL916.
3. `python scripts/run_session.py levels --session-dir <this dir> --tone-vrms <reading> --brick-v <brick V>`
   checks every file: the speech peaks (crest about 23 dB) must stay under {ss.PEAK_SUPPLY_FRACTION:g} x the
   brick voltage, else lower the target and repeat. The 160 dB ceiling applies to the speech files;
   `sweeps.wav` is listed on its own line (about 3 dB above speech) and is not a reason to turn down.

## Channel map

| recorder | fs (Hz) | input | role | note |
|---|---|---|---|---|
{chr(10).join(_chan_rows(cmap))}

Hydrophone channel ids are the `ch<N>` in every utt_id. Scored channel: ch{cmap.scored_channel}. The PC plays
only; both Daisy boards record on the Linux machine and every file aligns on its own hydrophone chirps.

## Takes, in order

Start BOTH Daisy recorders first (they stop on their own after `--seconds`), then start the take on the
PC within 20 s. Name every file exactly as listed (relative to this session dir).
{'The dry run goes to `dryrun/raw/...` and is never imported; its QC must PASS before the pool takes.' if any(t.dryrun for t in plan) else ''}
{f'The chain reference (`ref_sweeps`, `ref_speech`, {ref.distance_m:g} m) gives the measured transducer-chain reference R1.' if ref else ''}
{f'Decision rule (pre-registered): after `pilot_speech` ({pilot.distance_m:g} m) is imported, score it with C0; pilot C0 WER >= 10 % -> quiet recordings primary (added noise secondary); < 10 % -> added noise at 0 dB SNR primary (quiet secondary).' if pilot else ''}

| take | position | d (m) | play | min | record s | content | Daisy files |
|---|---|---|---|---|---|---|---|
{chr(10).join(take_rows)}

After each take: copy the Daisy files to this session dir on the PC, then

    python scripts/run_session.py verify --session-dir <this dir> --take-id <take>
    python scripts/run_session.py check  --session-dir <this dir> --take-id <take>

`check` prints PASS/FAIL per file (speech, anchor, dry run) or OK (noise, sweeps: file present) and
writes the take's import lines to `import_cmds.txt` with `--out` already routed: `p<d>_speech` /
`p<d>_anchor` -> `{meta['out']}`, `ref_speech` -> `{meta['out']}/chainref`, `pilot_speech` ->
`{meta['out']}/pilot`, bubble takes -> `{meta['out']}/bubble`. Adding `--run-import` imports the take
there at once (slow: minutes per channel); do it at least for `pilot_speech` (the decision rule).

A FAIL means redo: `--attempt 2` on daisy-cmds and take (files `..._r2_<rec>.wav`; nothing is overwritten).
The attempt-2 Daisy lines are at the end of `daisy_cmds.txt`; for attempt k, write `_r<k>` in `--out`
(or run `daisy-cmds --take-id <take> --attempt <k>` on the PC and type the printed lines on Linux).

## Per-take log

| {' | '.join(cols)} |
|{'---|' * len(cols)}
{log_rows}

## Commands per take (attempt 1)

{chr(10).join(cmds)}"""


def run_sheet(meta: dict, plan: list[ss.Take], cmap: ss.ChannelMap) -> str:
    """Legacy (thruster schedule) run sheet."""
    s = meta["session"]
    recs = list(cmap.recorders.values())
    take_rows = []
    for t in plan:
        files = ", ".join(f"`s{s}_{t.take_id}_{r.name}.wav`" for r in recs if r.records)
        what = t.file or "(nothing; record only" + ("; THRUSTER ON and steady before recording)" if any(
            "THRUSTER ON" in c.text for c in t.cues) else ")")
        if t.file and t.gain_db:
            what += f" at {t.gain_db:+g} dB"
        extra = f"{len(t.chirps_s)} chirps, {t.n_blocks} blocks, {t.n_utts} utts" if t.importable else \
            (f"{len(t.chirps_s)} chirp + sweeps" if t.kind == "sweeps" else "")
        if t.setup:
            extra = f"{extra}; {t.setup}" if extra else t.setup
        take_rows.append(f"| {t.take_id} | {what} | {t.duration_s / 60:.1f} | {extra} | {files} | | |")
    cue_sections = []
    for t in plan:
        changes = [c for c in t.cues if "THRUSTER" in c.text]
        if t.kind in ("speech", "anchor") or (t.kind == "noise" and any("THRUSTER ON" in c.text for c in changes)):
            body = "\n".join(f"| {c.t_s:8.1f} | {int(c.t_s // 60)}:{c.t_s % 60:04.1f} | {c.text} |" for c in changes)
            cue_sections.append(f"### {t.take_id}: thruster cues (time from the file start)\n\n"
                                f"| t (s) | m:ss | action |\n|---|---|---|\n{body}\n")
    log_rows = "\n".join(f"| {t.take_id} | | | | | | | | |" for t in plan)
    return f"""# Run sheet: session {s} ({meta['bench_split']}, legacy thruster plan)

Generated by `scripts/run_session.py plan` on {meta['created']}. Bench: `{meta['bench']}`.
Imports go to `{meta['out']}`.

## Channel map

| recorder | fs (Hz) | input | role | note |
|---|---|---|---|---|
{chr(10).join(_chan_rows(cmap))}

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
SWEEPS_FILE = "sweeps.wav"


def cmd_levels(a: argparse.Namespace) -> int:
    """File levels and the drive they give.

    The drive comes from the DMM reading of the 1 kHz tone (``--tone-vrms``, at the pool) or, at
    the desk, from the tone Vrms that the target needs (``--target-sl-db``), so the peak-vs-brick
    and 20 Vrms checks run before the day too. The 160 dB ceiling applies to the speech-type
    files only; ``sweeps.wav`` (sine sweeps, about 3 dB above the speech active RMS) is reported
    on its own line and checked only against the 20 Vrms cap and the brick voltage.
    """
    meta, plan, cmap = _load_session(a.session_dir)
    bench = _bench(meta)
    tone_meta_p = bench / "session" / "tone_1k.json"
    tone = json.loads(tone_meta_p.read_text(encoding="utf-8")) if tone_meta_p.exists() else None
    if a.tone_vrms is not None and tone is None:
        raise SystemExit("--tone-vrms needs session/tone_1k.json (make_session_audio.py, pool plan)")
    levels = {}
    for f in dict.fromkeys(t.file for t in plan if t.file):
        x, fs = audio.load(str(bench / "session" / f))
        levels[f] = ss.file_levels(x, fs)
        del x
    speech_active = [lv["active_rms_dBFS"] for f, lv in levels.items() if f != SWEEPS_FILE]
    max_active = max(speech_active) if speech_active else -np.inf
    out = {"time": _now(), "tone_vrms": a.tone_vrms, "brick_v": a.brick_v, "target_sl_db": a.target_sl_db,
           "vrms_at_0dbfs": a.vrms_at_0dbfs, "amp_gain_db": a.amp_gain_db}
    warn = []
    tv = None
    if tone is not None and np.isfinite(max_active):
        tv = ss.tone_target_vrms(a.target_sl_db, max_active, tone["rms_dBFS"])
        out["tone_target_vrms"] = round(tv, 3)
        if tv > ss.LL916_MAX_VRMS:
            warn.append(f"the tone target {tv:.1f} Vrms is above the LL916 {ss.LL916_MAX_VRMS:g} Vrms cap")
    if a.tone_vrms is not None:
        basis, tone_v = f"measured tone {a.tone_vrms:g} Vrms", a.tone_vrms
    elif tv is not None:
        basis, tone_v = f"PREDICTED at the tone target {tv:.2f} Vrms ({a.target_sl_db:g} dB)", tv
    else:
        basis, tone_v = None, None
    out["drive_basis"] = basis or ("legacy --vrms-at-0dbfs/--amp-gain-db" if a.vrms_at_0dbfs else None)
    rows, worst = [], {"speech": (-np.inf, 0.0), "sweeps": (-np.inf, 0.0)}
    worst_pk, worst_pk_file = 0.0, None
    for f, lv in levels.items():
        row = {"file": f, **{k: round(v, 2) for k, v in lv.items()},
               "crest_dB": round(lv["peak_dBFS"] - lv["active_rms_dBFS"], 1)}
        d = None
        if tone_v is not None:
            d = ss.tone_drive(lv["active_rms_dBFS"], lv["peak_dBFS"], tone_v, tone["rms_dBFS"])
        elif a.vrms_at_0dbfs and a.amp_gain_db is not None:
            d = ss.drive_estimate(lv["active_rms_dBFS"], a.vrms_at_0dbfs, a.amp_gain_db, peak_dbfs=lv["peak_dBFS"])
        if d is not None:
            row.update({k: round(v, 2) for k, v in d.items()})
            kind = "sweeps" if f == SWEEPS_FILE else "speech"
            sl, v = worst[kind]
            worst[kind] = (max(sl, d["SL_est_dB"]), max(v, d["drive_vrms"]))
            if d["drive_peak_v"] > worst_pk:
                worst_pk, worst_pk_file = d["drive_peak_v"], f
        rows.append(row)
    for r in rows:
        print("  ".join(f"{k}={v}" for k, v in r.items()))
    out["files"] = rows
    if tv is not None:
        print(f"set the 1 kHz tone to {tv:.2f} Vrms for {a.target_sl_db:g} dB re 1 uPa @ 1 m nominal speech-active SL "
              f"(loudest speech file active RMS {max_active:.1f} dBFS, tone {tone['rms_dBFS']:.1f} dBFS)")
    if basis:
        print(f"drive below: {basis}")
    sl, v = worst["speech"]
    if np.isfinite(sl):
        print(f"speech files: highest nominal SL {sl:.1f} dB re 1 uPa @ 1 m (target 150-160; nobody in the water "
              f"while transmitting; any person in the water: RL <= {ss.RL_LIMIT_DB:.0f} dB); drive {v:.2f} Vrms active")
        if sl > 160 + 0.05:
            warn.append(f"speech SL {sl:.1f} dB is above the 160 dB plan ceiling; turn the knob down")
    sl_sw, v_sw = worst["sweeps"]
    if np.isfinite(sl_sw):
        print(f"sweeps.wav (sine sweeps; not held to the 160 dB speech ceiling): nominal SL {sl_sw:.1f} dB, "
              f"drive {v_sw:.2f} Vrms")
    if worst_pk_file is not None:
        print(f"peak drive {worst_pk:.1f} V ({worst_pk_file})" + (
            f"; limit {ss.PEAK_SUPPLY_FRACTION:g} x {a.brick_v:g} V brick = {ss.PEAK_SUPPLY_FRACTION * a.brick_v:.1f} V"
            if a.brick_v else "; pass --brick-v to check it against the amplifier supply"))
        vmax = max(v, v_sw)
        if vmax > ss.LL916_MAX_VRMS:
            warn.append(f"drive {vmax:.1f} Vrms is above the LL916 {ss.LL916_MAX_VRMS:g} Vrms cap")
        if a.brick_v:
            lim = ss.PEAK_SUPPLY_FRACTION * a.brick_v
            if worst_pk > lim:
                warn.append(f"peak drive {worst_pk:.1f} V ({worst_pk_file}) exceeds {ss.PEAK_SUPPLY_FRACTION:g} x "
                            f"{a.brick_v:g} V supply = {lim:.1f} V: the amplifier will clip; lower the target")
    for w in warn:
        print("WARNING: " + w)
    out["warnings"] = warn
    if a.live:
        out["live"] = live_level(a, meta, plan, cmap)
    (a.session_dir / "levels.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def live_level(a: argparse.Namespace, meta: dict, plan: list[ss.Take], cmap: ss.ChannelMap) -> dict:
    """Legacy: play the first speech block's opening seconds and measure the loopback."""
    rec = _iface(cmap, a.iface)
    if rec.loopback is None:
        raise SystemExit("--live needs a loopback input; without one set the level with the tone and a DMM")
    t = next(t for t in plan if t.kind == "speech")
    x, fs = audio.load(str(_bench(meta) / "session" / t.file))
    x = x[: int(a.live_s * fs)]
    y = _playrec(x, fs, rec, a)
    lv = ss.file_levels(y[rec.loopback], rec.fs)
    res = {"file": t.file, "seconds": a.live_s, "loopback_active_rms_dBFS": round(lv["active_rms_dBFS"], 2),
           "loopback_peak_dBFS": round(lv["peak_dBFS"], 2)}
    if a.in_dbfs_at_1vrms is not None:
        v = ss.loopback_drive(lv["active_rms_dBFS"], a.in_dbfs_at_1vrms, a.loopback_atten_db)
        res.update({"drive_vrms_measured": round(v, 3), "SL_est_dB": round(ss.sl_from_vrms(v), 1)})
    print("live:", res)
    return res


# ------------------------------------------------------------------ audio I/O
def _sounddevice():
    """``sounddevice`` (live playback), with the fix in the error when the env lacks it."""
    try:
        import sounddevice as sd
    except ImportError as e:
        raise SystemExit("live playback needs the 'sounddevice' package, which this Python env lacks "
                         f"({e}). Install it into the uwse env (conda env update -f environment.yml, or "
                         "pip install sounddevice), then run: python -c \"import sounddevice; "
                         "print(sounddevice.query_devices())\" and pass the interface as --device") from None
    return sd


def _iface(cmap: ss.ChannelMap, name: str | None) -> ss.Recorder:
    if name:
        return cmap.recorders[name]
    lb = [r for r in cmap.recorders.values() if r.loopback is not None]
    if lb:
        return lb[0]
    if "iface" in cmap.recorders:
        return cmap.recorders["iface"]
    raise SystemExit("no playback interface in the channel map; pass --iface")


def _playrec(x: np.ndarray, fs: int, rec: ss.Recorder, a: argparse.Namespace,
             cues: list[ss.Cue] | None = None) -> np.ndarray:
    """Legacy: play ``x`` on ``--out-channel`` with pre/post roll; return (C, T) inputs of ``rec``."""
    sd = _sounddevice()

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


def _play_stream(path: Path | None, duration_s: float, a: argparse.Namespace, cues: list[ss.Cue],
                 gain_db: float = 0.0, block: int = 4800) -> None:
    """Play a file block by block (never loaded whole) on ``--out-channel``, printing cues on time.

    ``path=None`` plays nothing and only runs the cue clock for ``duration_s`` (noise takes).
    """
    pending = sorted(cues, key=lambda c: c.t_s)

    def tick(t_file: float, t0: float) -> None:
        while pending and pending[0].t_s <= t_file:
            c = pending.pop(0)
            print(f"[{time.monotonic() - t0:7.1f} s] {c.text}", flush=True)

    t0 = time.monotonic()
    if path is None:
        while time.monotonic() - t0 < duration_s:
            tick(time.monotonic() - t0, t0)
            time.sleep(0.2)
        tick(np.inf, t0)
        return
    sd = _sounddevice()
    import soundfile as sf

    g = 10 ** (gain_db / 20)
    with sf.SoundFile(str(path)) as f:
        fs, nch = f.samplerate, int(a.out_channel)
        with sd.OutputStream(samplerate=fs, channels=nch, device=a.device, dtype="float32") as st:
            st.write(np.zeros((int(a.pre_roll_s * fs), nch), np.float32))
            n = 0
            while True:
                x = f.read(block, dtype="float32", always_2d=True)
                if not len(x):
                    break
                buf = np.zeros((len(x), nch), np.float32)
                buf[:, nch - 1] = x[:, 0] * g
                tick(n / fs, t0)
                st.write(buf)
                n += len(x)
            tick(np.inf, t0)
            st.write(np.zeros((int(a.post_roll_s * fs), nch), np.float32))


# ------------------------------------------------------------------ tone
def cmd_tone(a: argparse.Namespace) -> int:
    meta, plan, cmap = _load_session(a.session_dir)
    p = _bench(meta) / "session" / "tone_1k.wav"
    if not p.exists():
        raise SystemExit(f"{p} missing (make_session_audio.py on a pool-plan bench writes it)")
    print("1 kHz level tone: knob from MINIMUM, raise only until the DMM reads the target Vrms; tape the knob")
    for k in range(a.repeat):
        _play_stream(p, 0.0, a, [ss.Cue(0.0, f"tone {k + 1}/{a.repeat}")])
    return 0


# ------------------------------------------------------------------ take
POOL_METADATA = ("distance_m", "speaker_depth_m", "hyd_depth_m", "orientation", "water_temp_c", "amp_vrms",
                 "brick_v", "rec_gain", "pump", "operator", "qc_note", "notes")


def cmd_take(a: argparse.Namespace) -> int:
    sdir = a.session_dir
    meta, plan, cmap = _load_session(sdir)
    t = _take(plan, a.take_id)
    if meta.get("plan") == "pool":
        return _take_pool(a, sdir, meta, t, cmap)
    return _take_legacy(a, sdir, meta, t, cmap)


def _take_pool(a: argparse.Namespace, sdir: Path, meta: dict, t: ss.Take, cmap: ss.ChannelMap) -> int:
    done = [r for r in _log_rows(sdir) if r.get("type") == "take" and r["take_id"] == t.take_id
            and int(r.get("attempt", 1)) == a.attempt]
    if done:  # also with --no-audio: one metadata row per attempt
        raise SystemExit(f"{t.take_id} attempt {a.attempt} is already logged ({done[-1]['started']}); "
                         f"a redo is --attempt {a.attempt + 1} (record it with daisy-cmds --attempt {a.attempt + 1})")
    started = _now()
    if t.setup:
        print(f"{t.take_id}: CHECK FIRST: {t.setup}")
    if a.no_audio:
        print(f"{t.take_id}: --no-audio, logging only")
    else:
        path = _bench(meta) / "session" / t.file if t.file else None
        print(f"{t.take_id}: {'playing ' + t.file if t.file else 'nothing played (noise only)'} "
              f"({t.duration_s / 60:.1f} min); both Daisy boards must already be recording", flush=True)
        _play_stream(path, t.duration_s, a, t.cues, gain_db=t.gain_db)
    md = {k: getattr(a, k) for k in POOL_METADATA}
    if md["distance_m"] is None:
        md["distance_m"] = t.distance_m
    if md["pump"] is None:
        md["pump"] = "on" if t.bubbles else "off"
    rows = {r.name: str(ss.raw_path(sdir, meta["session"], t, r.name, a.attempt)) for r in cmap.recorders.values()
            if r.hydrophones}
    _append(sdir, {"type": "take", "take_id": t.take_id, "attempt": a.attempt, "kind": t.kind,
                   "position_id": t.position_id, "played_file": t.file, "started": started, "session": meta["session"],
                   "recordings": rows, "pre_roll_s": a.pre_roll_s, "post_roll_s": a.post_roll_s,
                   "nominal_distance_m": t.distance_m, **md})
    print(f"{t.take_id}: logged; copy the Daisy files, then run verify and check")
    return 0


LEGACY_METADATA = ("range_m", "proj_depth_m", "hyd_depth_m", "water_temp_c", "amp_gain_db", "rec_gain_db",
                   "pumps", "operator", "notes")


def _take_legacy(a: argparse.Namespace, sdir: Path, meta: dict, t: ss.Take, cmap: ss.ChannelMap) -> int:
    rec = _iface(cmap, a.iface)
    path = rec_path(sdir, meta["session"], t.take_id, rec.name, a.attempt)
    started = _now()
    recorded = False
    if a.no_audio:
        print(f"{t.take_id}: --no-audio, logging only")
    else:
        if t.setup:
            print(f"{t.take_id}: SET UP FIRST: {t.setup}")
        print(f"{t.take_id}: playing {t.file or '(silence)'} ({t.duration_s / 60:.1f} min) on {rec.name}")
        if rec.records:
            if path.exists():
                raise SystemExit(f"{path} exists; a redo is --attempt {a.attempt + 1}")
            if t.file:
                x, fs = audio.load(str(_bench(meta) / "session" / t.file))
                x = x * 10 ** (t.gain_db / 20)
            else:
                fs, x = rec.fs, np.zeros(int(t.duration_s * rec.fs))
            y = _playrec(x, fs, rec, a, cues=t.cues)
            path.parent.mkdir(parents=True, exist_ok=True)
            audio.save(str(path), y, rec.fs, subtype="PCM_24")
            recorded = True
        else:  # the interface records nothing (no loopback): play only, streamed
            _play_stream(_bench(meta) / "session" / t.file if t.file else None, t.duration_s, a, t.cues,
                         gain_db=t.gain_db)
    row = {"type": "take", "take_id": t.take_id, "attempt": a.attempt, "kind": t.kind, "position_id": t.position_id,
           "played_file": t.file, "started": started, "session": meta["session"],
           "recordings": {rec.name: str(path)} if recorded else {},
           "pre_roll_s": a.pre_roll_s, "post_roll_s": a.post_roll_s,
           **{k: getattr(a, k, None) for k in LEGACY_METADATA}}
    for spec in a.rec or []:
        name, p = spec.split("=", 1)
        row["recordings"][name] = p
    if t.importable and recorded and rec.loopback is not None:
        q = ss.qc_take(path, _bench(meta), t.position_id, t.kind == "anchor", rec.loopback)
        row["qc"] = {rec.name: q}
        print(("PASS" if q["ok"] else "FAIL -> redo the take") + f" {rec.name} loopback: {q}")
    _append(sdir, row)
    return 0


# ------------------------------------------------------------------ Daisy commands / verify
def _takes_arg(plan: list[ss.Take], ids: list[str] | None) -> list[ss.Take]:
    return [_take(plan, i) for i in ids] if ids else plan


def cmd_daisy(a: argparse.Namespace) -> int:
    meta, plan, cmap = _load_session(a.session_dir)
    lsd = a.linux_sdir or _linux_sdir(a.session_dir)
    for t in _takes_arg(plan, a.take_id):
        print(f"# {ss.take_label(t.take_id, a.attempt)}: {t.kind}, {t.file or 'nothing played'}, "
              f"{t.duration_s / 60:.1f} min; start these, then the take on the PC within {a.lead_s:g} s")
        print("\n".join(ss.daisy_commands(t, cmap, meta["session"], lsd, a.attempt, a.lead_s, a.tail_s,
                                          python=a.python)))
        print()
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    sdir = a.session_dir
    meta, plan, cmap = _load_session(sdir)
    names = [r.name for r in cmap.recorders.values() if r.hydrophones]
    all_ok = True
    for t in _takes_arg(plan, a.take_id):
        att = a.attempt or ss.latest_attempt(sdir, meta["session"], t, names)
        v = ss.verify_files(sdir, meta["session"], t, cmap, att, a.min_fraction)
        all_ok &= v["ok"]
        for name, r in v["files"].items():
            msg = r.get("error") or f"{r['seconds']} s, {r['channels']} ch, {r['fs']} Hz"
            if r.get("warning"):
                msg += f" (WARNING: {r['warning']})"
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {ss.take_label(t.take_id, att)} {name}: {msg}")
        if a.take_id:
            _append(sdir, {"type": "verify", "time": _now(), **v})
    if not all_ok:
        print("missing or short files: copy them from the Linux machine (or redo the take) before check")
    return 0 if all_ok else 1


# ------------------------------------------------------------------ check / import
def import_out(meta: dict, t: ss.Take) -> str:
    """``--out`` of a take's import: the session's import dir, or its chainref/pilot/bubble sub-dir."""
    if meta.get("plan") != "pool":
        return str(_out(meta))
    sub = ss.import_subdir(t)
    return meta["out"].rstrip("/") + (f"/{sub}" if sub else "")


QC_FAIL_KEY = "|qcfail"


def _write_import_cmds(sdir: Path, new: dict[str, tuple[list[str], bool]]) -> None:
    """Merge ``new`` (take|recorder -> (argv, qc_ok)) into import_cmds.json and rewrite import_cmds.txt.

    Only a QC-passing attempt becomes an import line; it replaces the take x recorder's earlier
    line and clears any failure note. A failing attempt never replaces a passing line: it is kept
    as a ``# QC FAIL`` comment (not run by a copy-paste of the file) until a later attempt passes.
    """
    pj = sdir / "import_cmds.json"
    cur = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {}
    for key, (argv, ok) in new.items():
        if ok:
            cur[key] = argv
            cur.pop(key + QC_FAIL_KEY, None)
        else:
            cur[key + QC_FAIL_KEY] = argv
    pj.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    with open(sdir / "import_cmds.txt", "w", encoding="utf-8", newline="\n") as f:
        for key, argv in cur.items():
            line = "python scripts/import_recording.py " + " ".join(shlex.quote(x) for x in argv)
            if key.endswith(QC_FAIL_KEY):
                kept = ("the passing attempt above is kept" if key[: -len(QC_FAIL_KEY)] in cur else
                        "no attempt has passed: redo the take (or import with check --force)")
                line = f"# QC FAIL, not imported ({kept}): {line}"
            f.write(line + "\n")


def cmd_check(a: argparse.Namespace) -> int:
    sdir = a.session_dir
    meta, plan, cmap = _load_session(sdir)
    t = _take(plan, a.take_id)
    bench, pool = _bench(meta), meta.get("plan") == "pool"
    names = [r.name for r in cmap.recorders.values() if r.records]
    att = a.attempt or ss.latest_attempt(sdir, meta["session"], t, names)
    recs: dict[str, str] = {}
    for r in _log_rows(sdir):
        if r["take_id"] == t.take_id and r["type"] == "take" and int(r.get("attempt", 1)) == att:
            recs.update(r["recordings"])
    for name in names:  # the standard file names, if not logged
        p = ss.raw_path(sdir, meta["session"], t, name, att)
        if name not in recs and (p.exists() or (pool and cmap.recorders[name].hydrophones)):
            recs[name] = str(p)
    for spec in a.rec or []:
        name, p = spec.split("=", 1)
        recs[name] = p
    if not recs:
        raise SystemExit(f"no recordings for {t.take_id}; pass --rec NAME=PATH")
    results, cmds, all_ok = {}, {}, True
    label = ss.take_label(t.take_id, att)
    for name, p in recs.items():
        rec = cmap.recorders[name]
        p = Path(p)
        if not p.is_absolute() and not p.exists() and (sdir / p).exists():
            p = sdir / p
        if not p.exists():
            results[name] = {"ok": False, "file": str(p), "error": "missing"}
            all_ok = False
            print(f"FAIL {label} {name}: {p} missing")
            continue
        if not rec.has_sync:  # e.g. an interface without loopback: nothing to align on
            results[name] = {"ok": True, "file": str(p), "note": "not QC'd: no loopback or hydrophone"}
            continue
        if not (t.importable or t.dryrun):
            results[name] = {"ok": True, "file": str(p)}
            print(f"OK   {label} {name}: {p.name} present (no chirp QC for {t.kind} takes; verify checked "
                  "length, rate and channels)")
            continue
        q = ss.qc_take(p, bench, t.position_id, t.kind == "anchor", cmap.sync_channel(rec), played=t.played)
        results[name] = q
        all_ok &= q["ok"]
        print(("PASS" if q["ok"] else "FAIL") + f" {label} {name}: {q}")
        if rec.hydrophones and t.importable:
            bench_arg = meta["bench"] if pool else str(bench)
            cmds[f"{t.take_id}|{name}"] = (ss.import_argv(t, rec, p, cmap, bench_arg, meta["session"],
                                                          import_out(meta, t)), q["ok"])
    _append(sdir, {"type": "check", "take_id": t.take_id, "attempt": att, "time": _now(), "qc": results})
    if t.dryrun:
        print("dry run: " + ("PASS, the chain works" if all_ok else "FAIL: fix the chain before the pool takes"))
    if cmds:
        _write_import_cmds(sdir, cmds)
        print(f"import lines -> {sdir / 'import_cmds.txt'} (--out {import_out(meta, t)})")
    if a.run_import and cmds:
        if not all_ok and not a.force:
            print("QC failed: not importing (pass --force to import anyway; failed blocks are flagged)")
            return 1
        import importlib.util

        spec = importlib.util.spec_from_file_location("import_recording", ROOT / "scripts" / "import_recording.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for argv, _ok in cmds.values():
            argv = list(argv)
            for k in ("--bench", "--out"):  # session.json stores them relative to the project root
                i = argv.index(k)
                argv[i + 1] = str(ss.resolve_path(argv[i + 1]))
            mod.main(argv)
    return 0 if all_ok else 1


# ------------------------------------------------------------------ CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="take plan, run sheet, utterance lists, Daisy command lines")
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--session", type=int, required=True, help="session number in utt_ids (s<N>)")
    p.add_argument("--channel-map", type=Path, required=True)
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="import output dir (default data/real/<bench name>_s<N>; legacy data/real/<release>_s<N>)")
    p.add_argument("--replan", action="store_true", help="rewrite the plan files of an existing session dir")
    p.add_argument("--noise-s", type=float, default=60.0, help="noise-only take per distance/position (s)")
    g = p.add_argument_group("pool plan (bench with session/programs.json)")
    g.add_argument("--distances", type=float, nargs="+", default=None, metavar="M",
                   help="projector-hydrophone distances of the main takes, in order (default 1 3 6)")
    g.add_argument("--ref-m", type=float, default=ss.POOL_DEFAULTS["ref_m"], help="chain-reference distance (0.5-1 m)")
    g.add_argument("--pilot-m", type=float, default=ss.POOL_DEFAULTS["pilot_m"])
    g.add_argument("--bubble-m", type=float, default=ss.POOL_DEFAULTS["bubble_m"])
    g.add_argument("--no-bubbles", action="store_true", help="drop the bubble pass")
    g.add_argument("--bubble-noise-s", type=float, default=ss.POOL_DEFAULTS["bubble_noise_s"])
    g.add_argument("--no-dry-run", action="store_true", help="drop the desk dry-run take")
    g.add_argument("--date", default=None)
    g.add_argument("--pool-name", default=None)
    g.add_argument("--pool-length-m", type=float, default=None)
    g.add_argument("--pool-width-m", type=float, default=None)
    g.add_argument("--pool-depth-m", type=float, default=None)
    g = p.add_argument_group("legacy thruster plan only")
    g.add_argument("--extra-noise", nargs="*", default=list(ss.EXTRA_NOISE))
    g.add_argument("--extra-noise-s", type=float, default=90.0)
    g.add_argument("--e2", action="store_true", help="E2 projector chain: sweeps x depths x angles x drive levels")
    g.add_argument("--e2-depths-m", type=float, nargs="+", default=ss.GROUP_DEFAULTS["e2"]["depths_m"])
    g.add_argument("--e2-angles-deg", type=float, nargs="+", default=ss.GROUP_DEFAULTS["e2"]["angles_deg"])
    g.add_argument("--e2-atten-db", type=float, nargs="+", default=ss.GROUP_DEFAULTS["e2"]["atten_db"])
    g.add_argument("--e3-points", nargs="+", default=None, metavar="LABEL")
    g.add_argument("--e3-repeat-points", nargs="*", default=None, metavar="LABEL")
    g.add_argument("--e4", action="store_true", help="E4 thruster noise bank")

    def audio_args(p):
        p.add_argument("--session-dir", type=Path, required=True)
        p.add_argument("--iface", default=None, help="recorder name of the playback interface")
        p.add_argument("--device", default=None, help="sounddevice device name or index")
        p.add_argument("--out-channel", type=int, default=1, help="interface output feeding the amplifier (1-based)")
        p.add_argument("--pre-roll-s", type=float, default=2.0)
        p.add_argument("--post-roll-s", type=float, default=3.0)

    p = sub.add_parser("levels", help="file levels, drive (active Vrms, peak V) and nominal source level")
    audio_args(p)
    p.add_argument("--tone-vrms", type=float, default=None, help="DMM reading of the 1 kHz tone (pool plan)")
    p.add_argument("--target-sl-db", type=float, default=160.0, help="speech-active nominal SL to dial in")
    p.add_argument("--brick-v", type=float, default=None, help="amplifier DC supply voltage (peak headroom check)")
    p.add_argument("--vrms-at-0dbfs", type=float, default=None,
                   help="legacy: interface output Vrms for a full-scale (0 dBFS peak) sine")
    p.add_argument("--amp-gain-db", type=float, default=None, help="legacy: amplifier voltage gain (dB)")
    p.add_argument("--live", action="store_true", help="legacy: measure the loopback")
    p.add_argument("--live-s", type=float, default=15.0)
    p.add_argument("--in-dbfs-at-1vrms", type=float, default=None)
    p.add_argument("--loopback-atten-db", type=float, default=40.0)

    p = sub.add_parser("tone", help="play the 1 kHz level tone (streamed) for the DMM")
    audio_args(p)
    p.add_argument("--repeat", type=int, default=3, help="times to play the tone file")

    p = sub.add_parser("take", help="play one take (streamed), print its cues, log its metadata")
    audio_args(p)
    p.add_argument("--take-id", required=True)
    p.add_argument("--attempt", type=int, default=1, help="redo number (files ..._r<k>_<rec>.wav for k >= 2)")
    p.add_argument("--no-audio", action="store_true", help="played elsewhere; log metadata only")
    p.add_argument("--rec", action="append", help="legacy: other recorders' files, NAME=PATH")
    p.add_argument("--distance-m", type=float, help="measured projector-hydrophone distance")
    p.add_argument("--speaker-depth-m", type=float)
    p.add_argument("--hyd-depth-m", type=float)
    p.add_argument("--orientation", default=None, help="projector rotation mark / hydrophone bearing")
    p.add_argument("--water-temp-c", type=float)
    p.add_argument("--amp-vrms", type=float, help="DMM reading at the tone check")
    p.add_argument("--brick-v", type=float)
    p.add_argument("--rec-gain", default=None, help="recorder gains (free text, e.g. A=20dB,B=20dB)")
    p.add_argument("--pump", choices=("on", "off"), default=None, help="air pump (default: on for bubble takes)")
    p.add_argument("--qc-note", default=None, help="operator's on-site QC remark")
    p.add_argument("--operator")
    p.add_argument("--notes", default="")
    # legacy metadata
    p.add_argument("--range-m", type=float)
    p.add_argument("--proj-depth-m", type=float)
    p.add_argument("--amp-gain-db", type=float)
    p.add_argument("--rec-gain-db", type=float)
    p.add_argument("--pumps", choices=("on", "off"))

    p = sub.add_parser("daisy-cmds", help="print the record_daisy.py lines of takes (run on the PC; type them on Linux)")
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--take-id", nargs="+", default=None, help="default: every take")
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--linux-sdir", default=None, help="session dir on the Linux machine (default: same relative path)")
    p.add_argument("--lead-s", type=float, default=20.0, help="seconds recorded before the PC starts the take")
    p.add_argument("--tail-s", type=float, default=10.0)
    p.add_argument("--python", default="python")

    p = sub.add_parser("verify", help="are the copied Daisy files of a take (or all takes) present and complete?")
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--take-id", nargs="+", default=None, help="default: every take")
    p.add_argument("--attempt", type=int, default=None, help="default: the latest attempt on disk")
    p.add_argument("--min-fraction", type=float, default=1.0, help="shortest file as a fraction of the take")

    p = sub.add_parser("check", help="QC every recorder file of a take; write (and run) the import commands")
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--take-id", required=True)
    p.add_argument("--attempt", type=int, default=None, help="default: the latest attempt on disk")
    p.add_argument("--rec", action="append", help="NAME=PATH, overrides the standard file name")
    p.add_argument("--run-import", action="store_true")
    p.add_argument("--force", action="store_true", help="import even if QC failed")
    a = ap.parse_args(argv)
    if getattr(a, "attempt", None) is not None and a.attempt < 1:
        ap.error("--attempt must be >= 1")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    return {"plan": cmd_plan, "levels": cmd_levels, "tone": cmd_tone, "take": cmd_take, "daisy-cmds": cmd_daisy,
            "verify": cmd_verify, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
