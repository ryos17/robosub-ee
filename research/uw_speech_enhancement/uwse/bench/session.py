"""Run-time support for a pool recording session (RESEARCH_PLAN.md 6.4-6.6, 6.8).

Everything here is hardware-free, so it can be tested on synthetic takes; the
audio I/O lives in ``scripts/run_session.py``.

Pool plan (2026-11-19; no thruster; bench from ``make_playlists.py --plan pool``):

- `build_pool_plan`: (0) ``dryrun`` desk/bucket dry run of ``anchor.wav`` (files go to
  ``<session>/dryrun/``, never the session raw dir); (1) chain reference at ``ref_m``:
  ``ref_sweeps`` + ``ref_speech`` (the pilot sentences); (2) ``pilot_speech`` at ``pilot_m``;
  (3) per distance d (chosen on the day): ``<p>_noise``, ``<p>_sweeps``, ``<p>_speech``
  (``<p>_speech_<k>`` if the bench split the file), ``<p>_anchor``; (4) bubble pass at
  ``bubble_m``: ``<p>bub_noise`` and ``<p>bub_speech``. Position labels are ``p`` + distance
  (`pos_label`: 1 -> p1, 0.5 -> p0p5); the chain reference, pilot and bubble takes get the
  suffixes ``ref``, ``pilot``, ``bub`` so their utt_ids are separable from the main takes.
- `plan_from_meta`: the plan of a session.json (pool or legacy).
- `raw_path` / `take_label`: recorder file names, with ``_r<k>`` for redo attempt k >= 2.
- `daisy_commands` / `verify_files`: the Linux ``record_daisy.py`` lines of a take and the
  check that the copied Daisy files are all there before ``run_session.py check``.
- `tone_drive` / `tone_target_vrms`: the drive (active Vrms, peak V) and nominal source level
  of every file from the measured Vrms of the 1 kHz level tone.

Legacy plan (thruster schedules, e.g. the frozen data/bench/v0_s1):

- `build_plan`: the ordered take list for one session (noise with the thruster off
  and, where the position has thruster-on blocks, on; sweeps; speech per position;
  then the drift anchor and the extra noise-bank takes), with each take's chirp onsets and the thruster cues an operator must follow.
  Optional E2/E3/E4 take groups (`extra_groups`, off by default) append the
  characterisation takes of RESEARCH_PLAN.md 6.4 and 5.2.
- `utterance_rows`: the utterance list of a position's speech file, in play order.
- `file_levels` / `drive_estimate`: digital levels of the session files and the
  projector drive they imply at a given amplifier setting.
- `ChannelMap`: which recorder file channel is which benchmark channel.
- `qc_take`: chirp alignment of a recorded take (the first stage of
  `scripts/import_recording.py`), to accept or redo a take at the pool.
- `import_argv`: the ``import_recording.py`` arguments for one recorder file of a take.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uwse import audio
from uwse.bench import ingest
from uwse.bench import playlist as pl

#: nominal LL916C source level, dB re 1 uPa @ 1 m at 1 kHz for 20 Vrms drive (Lubell; RESEARCH_PLAN.md 6.1)
LL916_SL_DB_AT_20VRMS = 180.0
#: Lubell maximum drive of the LL916: hard cap
LL916_MAX_VRMS = 20.0
#: received-level rule (6.2): <= 150 dB re 1 uPa where anyone is in the water
RL_LIMIT_DB = 150.0
#: sample RMS of a full-scale (peak 1.0) sine, dBFS: 20 log10(1/sqrt 2) = -3.01
SINE_RMS_DBFS = 20.0 * float(np.log10(np.sqrt(0.5)))
#: peak drive must stay under this fraction of the amplifier's DC supply (bridged TPA3255 headroom)
PEAK_SUPPLY_FRACTION = 0.9
EXTRA_NOISE = ("quiet_pool", "thruster", "bubbles_splashing", "pumps")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """A session.json path: relative to the project root, or absolute.

    An absolute path that no longer exists (the project moved, e.g. from
    ``repos/underwater_asr`` to the robosub-ee worktree) is re-rooted at the first
    ``data``/``sessions``/``runs`` component, if that exists under ``root``.
    """
    p = Path(p)
    if not p.is_absolute():
        return root / p
    if p.exists():
        return p
    parts = p.parts
    for i, part in enumerate(parts):
        if part in ("data", "sessions", "runs"):
            cand = root.joinpath(*parts[i:])
            if cand.exists() or (root / part).is_dir():
                return cand
    return p


def portable_path(p: str | Path, root: Path = PROJECT_ROOT) -> str:
    """Store a path relative to the project root when it is inside it (posix separators)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(p)


@dataclass
class Cue:
    t_s: float  # seconds from the start of the played file
    text: str


@dataclass
class Take:
    take_id: str
    kind: str  # noise | sweeps | speech | anchor | noise_bank
    position_id: str | None
    file: str | None  # played file, relative to <bench>/session; None = nothing played
    duration_s: float
    chirps_s: list[float] = field(default_factory=list)
    cues: list[Cue] = field(default_factory=list)
    n_blocks: int = 0
    n_utts: int = 0
    importable: bool = False  # speech / anchor takes go through import_recording.py
    gain_db: float = 0.0  # digital gain applied to the played file (E2 drive levels)
    setup: str = ""  # rig setup an operator must make before the take
    # pool plan
    played: str | None = None  # stem of the played pool-plan session file (load_played); None = legacy
    program: str | None = None  # pilot | speech | anchor | bubble | sweeps | noise
    distance_m: float | None = None  # nominal projector-hydrophone distance
    dryrun: bool = False  # files go to <session>/dryrun/, never imported
    bubbles: bool = False  # air stone running near the speaker


def _speech_take(bench: Path, pos: str, anchor: bool) -> Take:
    fs, blocks = ingest.load_session(bench, pos, anchor=anchor)
    chirp_n = int(round(blocks[0].timing["chirps"][0]["end"] - blocks[0].timing["chirps"][0]["start"]))
    chirps, cues, state = [], [], None
    for k, b in enumerate(blocks):
        starts = b.chirp_starts
        chirps += [s / fs for s in starts]
        if b.thruster_state != state:
            # switch right after the previous block's end chirp: the 2 s tail + gap is the spin-up time
            t = 0.0 if k == 0 else (blocks[k - 1].chirp_starts[-1] + chirp_n) / fs
            cues.append(Cue(round(t, 2), f"THRUSTER {b.thruster_state.upper()} (for block {b.block_id})"))
            state = b.thruster_state
        cues.append(Cue(round(b.start_sample / fs, 2), f"block {b.block_id} {b.content_block} "
                                                       f"[{b.thruster_state}]"))
    last = blocks[-1]
    end = last.start_sample + int(last.timing["n_samples"])
    if anchor:
        name = f"anchor_{pos}.wav"
    else:
        name = f"{pos}_speech.wav"
    dur = audio_duration(bench / "session" / name, fallback=end / fs)
    cues.append(Cue(round(end / fs, 2), "last end chirp done; keep recording to the end of the file"))
    if state == "on":
        cues.append(Cue(round(end / fs, 2), "THRUSTER OFF"))
    return Take(take_id=f"{pos}_anchor" if anchor else f"{pos}_speech", kind="anchor" if anchor else "speech",
                position_id=pos, file=name, duration_s=dur, chirps_s=[round(c, 4) for c in chirps], cues=cues,
                n_blocks=len(blocks), n_utts=sum(len(b.timing["utterances"]) for b in blocks), importable=True)


def audio_duration(path: Path, fallback: float = 0.0) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return fallback


# ------------------------------------------------------------------ pool plan (2026-11-19)
POOL_DEFAULTS = {"distances_m": [1.0, 3.0, 6.0], "ref_m": 1.0, "pilot_m": 3.0, "bubble_m": 3.0,
                 "noise_s": 60.0, "bubble_noise_s": 120.0, "dry_run": True}
SAFETY_CUE = "NOBODY IN THE WATER while the projector plays"
BUBBLE_SETUP = "AIR STONE RUNNING near the speaker (pump on, steady) BEFORE the recorders start"


def is_pool_bench(bench: str | Path) -> bool:
    return (Path(bench) / "session" / "programs.json").exists()


def pos_label(d: float) -> str:
    """Position id of a distance: 1 -> p1, 0.5 -> p0p5, 4.5 -> p4p5 (utt_id grammar p[0-9a-z]+)."""
    if not (np.isfinite(d) and d > 0):
        raise ValueError(f"distance must be > 0 m, got {d}")
    return "p" + _num(round(float(d), 3))


def _played_take(bench: Path, stem: str, take_id: str, kind: str, pos: str, program: str,
                 distance_m: float | None, setup: str = "", dryrun: bool = False, bubbles: bool = False) -> Take:
    fs, blocks = ingest.load_played(bench, stem)
    chirps, cues = [], [Cue(0.0, SAFETY_CUE)]
    if setup:
        cues.append(Cue(0.0, f"CHECK: {setup}"))
    for b in blocks:
        chirps += [s / fs for s in b.chirp_starts]
        cues.append(Cue(round(b.start_sample / fs, 2), f"block {b.block_id} {b.content_block}"))
    last = blocks[-1]
    end = last.start_sample + int(last.timing["n_samples"])
    cues.append(Cue(round(end / fs, 2), "last end chirp done; keep recording to the end of the file"))
    return Take(take_id=take_id, kind=kind, position_id=pos, file=f"{stem}.wav",
                duration_s=audio_duration(bench / "session" / f"{stem}.wav", fallback=end / fs),
                chirps_s=[round(c, 4) for c in chirps], cues=cues, n_blocks=len(blocks),
                n_utts=sum(len(b.timing["utterances"]) for b in blocks), importable=not dryrun, setup=setup,
                played=stem, program=program, distance_m=distance_m, dryrun=dryrun, bubbles=bubbles)


def _sweeps_take(bench: Path, take_id: str, pos: str, distance_m: float, sw: dict) -> Take:
    fs = sw["fs"]
    return Take(take_id, "sweeps", pos, "sweeps.wav", audio_duration(bench / "session" / "sweeps.wav"),
                chirps_s=[round(sw["chirp_start_sample"] / fs, 4)],
                cues=[Cue(0.0, SAFETY_CUE)] + [Cue(round(s / fs, 2), f"sweep {k + 1}")
                                               for k, s in enumerate(sw["sweep_start_samples"])],
                program="sweeps", distance_m=distance_m)


def _noise_take(take_id: str, pos: str, distance_m: float, dur: float, setup: str = "",
                bubbles: bool = False) -> Take:
    text = "NOTHING PLAYING: noise-only take" + (f"; {setup}" if setup else "")
    return Take(take_id, "noise", pos, None, dur, cues=[Cue(0.0, text)], setup=setup, program="noise",
                distance_m=distance_m, bubbles=bubbles)


def build_pool_plan(bench: str | Path, distances_m: Sequence[float] = POOL_DEFAULTS["distances_m"],
                    ref_m: float = POOL_DEFAULTS["ref_m"], pilot_m: float = POOL_DEFAULTS["pilot_m"],
                    bubble_m: float | None = POOL_DEFAULTS["bubble_m"], noise_s: float = POOL_DEFAULTS["noise_s"],
                    bubble_noise_s: float = POOL_DEFAULTS["bubble_noise_s"],
                    dry_run: bool = POOL_DEFAULTS["dry_run"]) -> list[Take]:
    """Take order of the no-thruster pool plan (module docstring). ``bubble_m=None`` drops the bubble pass."""
    bench = Path(bench)
    progs = json.loads((bench / "session" / "programs.json").read_text(encoding="utf-8"))
    by_prog: dict[str, list[str]] = {}
    for f in progs["files"]:
        by_prog.setdefault(f["program"], []).append(Path(f["file"]).stem)
    missing = [p for p in ("pilot", "speech", "anchor") if p not in by_prog]
    if missing:
        raise ValueError(f"{bench}: programs.json lacks {missing}")
    if len(by_prog["anchor"]) != 1 or len(by_prog["pilot"]) != 1:
        raise ValueError("the anchor and pilot programs must be single files")
    sw = json.loads((bench / "session" / "sweeps.json").read_text(encoding="utf-8"))
    dists = [float(d) for d in distances_m]
    labels = [pos_label(d) for d in dists]
    if not dists or len(set(labels)) != len(labels):
        raise ValueError(f"distances must be distinct and non-empty, got {dists}")
    plan: list[Take] = []
    anchor = by_prog["anchor"][0]
    if dry_run:
        plan.append(_played_take(bench, anchor, "dryrun", "dryrun", "pdry", "anchor", None, dryrun=True,
                                 setup="DESK DRY RUN, projector SUBMERGED in a bucket (never driven in air), knob "
                                       "just above minimum (tone about 0.1 Vrms): full chain (PC, interface, amp, "
                                       "AC205C, projector, hydrophones, both Daisy boards); QC must PASS"))
    rp = pos_label(ref_m) + "ref"
    plan.append(_sweeps_take(bench, "ref_sweeps", rp, ref_m, sw))
    plan.append(_played_take(bench, by_prog["pilot"][0], "ref_speech", "speech", rp, "pilot", ref_m,
                             setup=f"CHAIN REFERENCE: hydrophone {ref_m:g} m from the projector, on axis"))
    pp = pos_label(pilot_m) + "pilot"
    plan.append(_played_take(bench, by_prog["pilot"][0], "pilot_speech", "speech", pp, "pilot", pilot_m,
                             setup=f"PILOT at {pilot_m:g} m; score with C0 on site before the distance takes"))
    for d, p in zip(dists, labels):
        plan.append(_noise_take(f"{p}_noise", p, d, noise_s))
        plan.append(_sweeps_take(bench, f"{p}_sweeps", p, d, sw))
        parts = by_prog["speech"]
        for k, stem in enumerate(parts, start=1):
            tid = f"{p}_speech" if len(parts) == 1 else f"{p}_speech_{k}"
            plan.append(_played_take(bench, stem, tid, "speech", p, "speech", d))
        plan.append(_played_take(bench, anchor, f"{p}_anchor", "anchor", p, "anchor", d))
    if bubble_m is not None:
        if "bubble" not in by_prog:
            raise ValueError("the bench has no bubble program; pass bubble_m=None (--no-bubbles)")
        bp = pos_label(bubble_m) + "bub"
        plan.append(_noise_take(f"{bp}_noise", bp, bubble_m, bubble_noise_s, setup=BUBBLE_SETUP, bubbles=True))
        for k, stem in enumerate(by_prog["bubble"], start=1):
            tid = f"{bp}_speech" if len(by_prog["bubble"]) == 1 else f"{bp}_speech_{k}"
            plan.append(_played_take(bench, stem, tid, "speech", bp, "bubble", bubble_m, setup=BUBBLE_SETUP,
                                     bubbles=True))
    ids = [t.take_id for t in plan]
    if len(set(ids)) != len(ids):
        raise ValueError(f"take ids repeat: {ids}")
    return plan


def plan_from_meta(meta: dict) -> list[Take]:
    """The take plan stored in a session.json (pool or legacy)."""
    bench = resolve_path(meta["bench"])
    if meta.get("plan") == "pool":
        o = meta["pool"]
        return build_pool_plan(bench, o["distances_m"], o["ref_m"], o["pilot_m"], o["bubble_m"], o["noise_s"],
                               o["bubble_noise_s"], o["dry_run"])
    return build_plan(bench, noise_s=meta["noise_s"], extra_noise=meta["extra_noise"],
                      extra_noise_s=meta["extra_noise_s"], groups=meta.get("groups"))


def take_label(take_id: str, attempt: int = 1) -> str:
    """File label of a take attempt: ``p3_speech`` (attempt 1), ``p3_speech_r2`` (redo 2), ..."""
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return take_id if attempt == 1 else f"{take_id}_r{attempt}"


def raw_path(sdir: str | Path, session: int, take: Take | str, rec: str, attempt: int = 1,
             dryrun: bool | None = None) -> Path:
    """``<sdir>/raw/<rec>/s<N>_<label>_<rec>.wav``; dry-run takes live under ``<sdir>/dryrun/raw``."""
    tid = take if isinstance(take, str) else take.take_id
    if dryrun is None:
        dryrun = bool(getattr(take, "dryrun", False))
    base = Path(sdir) / ("dryrun" if dryrun else "") / "raw" / rec
    return base / f"s{session}_{take_label(tid, attempt)}_{rec}.wav"


def latest_attempt(sdir: str | Path, session: int, take: Take, recs: Sequence[str]) -> int:
    """Highest attempt with a file of any of ``recs`` on disk (1 if none)."""
    best = 1
    for rec in recs:
        d = raw_path(sdir, session, take, rec).parent
        if not d.exists():
            continue
        for p in d.glob(f"s{session}_{take.take_id}_r*_{rec}.wav"):
            k = p.name[len(f"s{session}_{take.take_id}_r"):-len(f"_{rec}.wav")]
            if k.isdigit():
                best = max(best, int(k))
    return best


def record_seconds(take: Take, lead_s: float = 20.0, tail_s: float = 10.0) -> int:
    """``record_daisy.py --seconds`` for a take: start the boards ``lead_s`` before playback."""
    return int(np.ceil(take.duration_s + lead_s + tail_s))


def daisy_commands(take: Take, cmap: "ChannelMap", session: int, linux_sdir: str, attempt: int = 1,
                   lead_s: float = 20.0, tail_s: float = 10.0, python: str = "python") -> list[str]:
    """Shell lines (Linux capture machine, project root) that record every Daisy board of a take at once.

    Both boards run in the background with ``--seconds``, so they stop on their own after
    the file; ``wait`` returns when both have written their WAV and sidecar JSON.
    """
    recs = [r for r in cmap.recorders.values() if r.board or r.name.lower().startswith("daisy")]
    if not recs:
        raise ValueError("the channel map has no Daisy recorder")
    secs = record_seconds(take, lead_s, tail_s)
    lines = []
    for r in recs:
        rel = raw_path("", session, take, r.name, attempt).as_posix().lstrip("/")
        out = f"{linux_sdir.rstrip('/')}/{rel}"
        board = f" --board {r.board}" if r.board else ""
        lines.append(f"{python} scripts/record_daisy.py{board} --rate {r.fs} --seconds {secs} --out {out} &")
    lines.append("wait")
    return lines


def verify_files(sdir: str | Path, session: int, take: Take, cmap: "ChannelMap", attempt: int = 1,
                 min_fraction: float = 1.0) -> dict[str, Any]:
    """Are the copied hydrophone-recorder files of a take all there and plausible?

    Per recorder with hydrophones: the WAV exists, has the map's rate and enough channels, lasts
    at least ``min_fraction`` x the played file, and its ``.wav.json`` sidecar (record_daisy.py)
    reports no USB frame drops. ``ok`` is False when any file is missing or short; frame drops
    only warn (QC in ``check`` decides).
    """
    import soundfile as sf

    out: dict[str, Any] = {"take_id": take.take_id, "attempt": attempt, "ok": True, "files": {}}
    for r in cmap.recorders.values():
        if not r.hydrophones:
            continue
        p = raw_path(sdir, session, take, r.name, attempt)
        row: dict[str, Any] = {"file": str(p), "ok": False}
        if not p.exists():
            row["error"] = "missing"
        else:
            info = sf.info(str(p))
            row.update({"fs": info.samplerate, "channels": info.channels, "seconds": round(info.duration, 2)})
            need_ch = 1 + max(list(r.hydrophones) + list(r.other))
            errs = []
            if info.samplerate != r.fs:
                errs.append(f"rate {info.samplerate} != {r.fs}")
            if info.channels < need_ch:
                errs.append(f"{info.channels} channels < {need_ch}")
            if info.duration < min_fraction * take.duration_s:
                errs.append(f"{info.duration:.1f} s shorter than the {take.duration_s:.1f} s take")
            side = p.with_suffix(p.suffix + ".json")
            if side.exists():
                meta = json.loads(side.read_text(encoding="utf-8"))
                row["dropped"] = int(meta.get("dropped", 0))
                if row["dropped"]:
                    row["warning"] = f"{row['dropped']} USB frame drops; QC decides"
            else:
                row["warning"] = "no record_daisy.py sidecar (.wav.json)"
            row["ok"] = not errs
            if errs:
                row["error"] = "; ".join(errs)
        out["files"][r.name] = row
        out["ok"] &= row["ok"]
    return out


def build_plan(bench: str | Path, noise_s: float = 60.0, extra_noise: Sequence[str] = EXTRA_NOISE,
               extra_noise_s: float = 90.0, groups: dict | None = None) -> list[Take]:
    """Session take order (TAKE_SHEET.md of ``make_session_audio.py``), one `Take` each.

    ``groups`` (see `extra_groups`) appends the optional E2/E3/E4 takes after the noise bank.
    """
    bench = Path(bench)
    sched = pl.read_schedule(bench / "schedule.json")
    positions = list(dict.fromkeys(e.position_id for e in sched if not e.anchor_repeat))
    anchors = list(dict.fromkeys(e.position_id for e in sched if e.anchor_repeat))
    sw_path = bench / "session" / "sweeps.json"
    sw = json.loads(sw_path.read_text(encoding="utf-8")) if sw_path.exists() else None
    plan: list[Take] = []
    for pos in positions:
        plan.append(Take(f"{pos}_noise", "noise", pos, None, noise_s,
                         cues=[Cue(0.0, "NOTHING PLAYING: noise-only take; thruster off")]))
        if any(e.thruster_state == "on" for e in sched if e.position_id == pos and not e.anchor_repeat):
            # noise power of the thruster-on SNR (measure_snr.py, method signed off 2026-11-08)
            plan.append(Take(f"{pos}_noise_on", "noise", pos, None, noise_s,
                             cues=[Cue(0.0, "NOTHING PLAYING: noise-only take; THRUSTER ON at the speech-take "
                                            "throttle and at steady speed BEFORE the recorders start"),
                                   Cue(round(noise_s, 2), "THRUSTER OFF after the recorders stop")]))
        if sw is not None:
            fs = sw["fs"]
            plan.append(Take(f"{pos}_sweeps", "sweeps", pos, "sweeps.wav",
                             audio_duration(bench / "session" / "sweeps.wav"),
                             chirps_s=[round(sw["chirp_start_sample"] / fs, 4)],
                             cues=[Cue(round(s / fs, 2), f"sweep {k + 1}")
                                   for k, s in enumerate(sw["sweep_start_samples"])]))
        plan.append(_speech_take(bench, pos, anchor=False))
    for pos in anchors:
        plan.append(_speech_take(bench, pos, anchor=True))
    for name in extra_noise:
        plan.append(Take(f"noise_{name}", "noise_bank", None, None, extra_noise_s,
                         cues=[Cue(0.0, f"NOTHING PLAYING: noise bank, {name.replace('_', ' ')}")]))
    if groups:
        plan += extra_groups(groups, sw, audio_duration(bench / "session" / "sweeps.wav"))
    return plan


# ------------------------------------------------------------------ optional take groups
#: defaults of the optional groups (RESEARCH_PLAN.md 6.4 steps 4-5 and 5.2); depths and grid points are rig-specific
GROUP_DEFAULTS = {
    "e2": {"depths_m": [1.0], "angles_deg": [0, 45, 90], "atten_db": [0.0, 6.0, 12.0]},
    "e3": {"points": [], "repeat_points": None, "pumps_point": None, "rig_point": None},
    "e4": {"a_throttles": list(range(10, 101, 10)), "a_distances_m": [0.5, 1.0, 2.0], "a_dirs": ["fwd", "rev"],
           "b_throttles": [20, 50, 80], "b_distance_m": 1.0, "emi_throttles": [20, 50, 80],
           "take_s": 60.0, "flow_s": 150.0},
}


def _num(x: float) -> str:
    """Filename-safe number: 1.0 -> 1, 0.5 -> 0p5, 45 -> 45."""
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def extra_groups(groups: dict, sw: dict | None, sweeps_s: float) -> list[Take]:
    """Optional characterisation takes, none imported; ``groups`` maps ``e2``/``e3``/``e4`` to overrides.

    - E2 (projector chain, 6.4 step 4): ``sweeps.wav`` at each source depth x orientation x drive
      level. Drive levels are digital attenuations of the played file, so the amplifier gain is never
      touched during the session.
    - E3 (pool IR grid, 6.4 step 5): ``sweeps.wav`` at each named grid point, a repeat at the repeat
      points (default the first two), pumps on vs off at one point and frame present vs absent at
      one point (default the first). The grid itself is set on the rig.
    - E4 (noise bank, 5.2): nothing played; thruster unit A throttles x distances x direction,
      unit B, the EMI check (dummy capacitive load in water; thruster in air) and flow noise.
    """
    out: list[Take] = []
    unknown = set(groups) - set(GROUP_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown take groups {sorted(unknown)}")

    def sweeps(take_id: str, setup: str, gain_db: float = 0.0) -> Take:
        if sw is None:
            raise ValueError(f"{take_id}: the bench has no session/sweeps.json")
        fs = sw["fs"]
        return Take(take_id, "sweeps", None, "sweeps.wav", sweeps_s, chirps_s=[round(sw["chirp_start_sample"] / fs, 4)],
                    cues=[Cue(round(s / fs, 2), f"sweep {k + 1}") for k, s in enumerate(sw["sweep_start_samples"])],
                    gain_db=gain_db, setup=setup)

    def thruster(take_id: str, dur: float, setup: str) -> Take:
        return Take(take_id, "noise_bank", None, None, dur, setup=setup,
                    cues=[Cue(0.0, f"NOTHING PLAYING: {setup}; THRUSTER ON and at steady speed BEFORE the recorders "
                                   "start"), Cue(round(dur, 2), "THRUSTER OFF after the recorders stop")])

    if "e2" in groups:
        g = {**GROUP_DEFAULTS["e2"], **(groups["e2"] or {})}
        for d in g["depths_m"]:
            for ang in g["angles_deg"]:
                for att in g["atten_db"]:
                    out.append(sweeps(f"e2_d{_num(d)}_a{_num(ang)}_att{_num(att)}",
                                      f"E2: projector at {d:g} m depth, rotated {ang:g} deg off-axis; "
                                      f"played {att:g} dB below the speech level", gain_db=-float(att)))
    if "e3" in groups:
        g = {**GROUP_DEFAULTS["e3"], **(groups["e3"] or {})}
        pts = list(g["points"])
        if not pts:
            raise ValueError("e3 needs at least one grid point label")
        for pt in pts:
            out.append(sweeps(f"e3_{pt}", f"E3: grid point {pt}"))
        for pt in (g["repeat_points"] if g["repeat_points"] is not None else pts[:2]):
            out.append(sweeps(f"e3_{pt}_rep", f"E3: grid point {pt}, repeat (rig re-set, same marks)"))
        p = g["pumps_point"] or pts[0]
        out.append(sweeps(f"e3_{p}_pumps_on", f"E3: grid point {p}, PUMPS ON"))
        out.append(sweeps(f"e3_{p}_pumps_off", f"E3: grid point {p}, PUMPS OFF"))
        p = g["rig_point"] or pts[0]
        out.append(sweeps(f"e3_{p}_noframe", f"E3: grid point {p}, rig-only test: frame ABSENT"))
    if "e4" in groups:
        g = {**GROUP_DEFAULTS["e4"], **(groups["e4"] or {})}
        t = float(g["take_s"])
        for dist in g["a_distances_m"]:
            for dr in g["a_dirs"]:
                for th in g["a_throttles"]:
                    out.append(thruster(f"e4_A_{dr}_t{_num(th)}_r{_num(dist)}", t,
                                        f"E4: unit A {dr} {th:g}% at {dist:g} m from the hydrophones"))
        for th in g["b_throttles"]:
            out.append(thruster(f"e4_B_t{_num(th)}_r{_num(g['b_distance_m'])}", t,
                                f"E4: unit B fwd {th:g}% at {g['b_distance_m']:g} m"))
        for th in g["emi_throttles"]:
            out.append(thruster(f"e4_emi_dummy_t{_num(th)}", t,
                                f"E4 EMI: thruster in water {th:g}%, one array channel on the dummy capacitive load"))
        for th in g["emi_throttles"]:
            out.append(thruster(f"e4_emi_air_t{_num(th)}", t,
                                f"E4 EMI: thruster IN AIR {th:g}%, hydrophones in water"))
        for name, what in (("axis", "jet axis pointed at the array"), ("90", "jet axis at 90 deg to the array")):
            out.append(thruster(f"e4_flow_{name}", float(g["flow_s"]), f"E4 flow noise: {what}"))
    return out


def utterance_rows(bench: str | Path, pos: str | None, anchor: bool = False,
                   played: str | None = None) -> list[dict[str, Any]]:
    """One row per utterance of a played file: where it is in the file and what is said."""
    fs, blocks = ingest.load_blocks(bench, pos or "", anchor=anchor, played=played)
    rows = []
    for b in blocks:
        for k, u in enumerate(b.timing["utterances"]):
            rows.append({"session_order_idx": b.session_order_idx, "block_id": b.block_id,
                         "content_block": b.content_block, "thruster_state": b.thruster_state, "k": k,
                         "item_id": u["item_id"], "start_s": round((b.start_sample + u["start"]) / fs, 3),
                         "dur_s": round((u["end"] - u["start"]) / fs, 3), "transcript": u.get("transcript", "")})
    return rows


# ------------------------------------------------------------------ levels
def file_levels(x: np.ndarray, fs: int, frame_s: float = 0.02, active_db: float = 30.0) -> dict[str, float]:
    """Peak and active RMS (frames within ``active_db`` of the loudest frame), in dBFS."""
    x = np.asarray(x, float)
    if x.ndim > 1:
        x = x[0]
    n = int(frame_s * fs)
    m = len(x) // n
    e = np.mean(x[: m * n].reshape(m, n) ** 2, axis=1) + 1e-30
    act = e >= e.max() * 10 ** (-active_db / 10)
    peak = float(np.max(np.abs(x))) + 1e-30
    return {"peak_dBFS": 20 * np.log10(peak), "active_rms_dBFS": 10 * np.log10(float(np.mean(e[act]))),
            "active_fraction": float(act.mean()), "duration_s": len(x) / fs}


def sl_from_vrms(v: float, sl_at_20vrms: float = LL916_SL_DB_AT_20VRMS) -> float:
    """Nominal LL916 source level (dB re 1 uPa @ 1 m, 1 kHz sensitivity) for a drive of ``v`` Vrms."""
    return float(sl_at_20vrms + 20 * np.log10(max(v, 1e-12) / 20.0))


def drive_estimate(rms_dbfs: float, vrms_at_0dbfs: float, amp_gain_db: float,
                   sl_at_20vrms: float = LL916_SL_DB_AT_20VRMS, peak_dbfs: float | None = None) -> dict[str, float]:
    """Projector drive (Vrms) and nominal source level for a digital (sample) RMS level.

    ``vrms_at_0dbfs`` is the playback interface's output Vrms for a full-scale 0 dBFS sine
    (peak 1.0, sample RMS `SINE_RMS_DBFS` = -3.01 dBFS), as a meter reads it;
    ``amp_gain_db`` the amplifier's voltage gain. A signal of sample RMS ``rms_dbfs``
    therefore drives ``vrms_at_0dbfs * 10^((rms_dbfs - SINE_RMS_DBFS + gain) / 20)``
    (fix 2026-11-19: this was 3.01 dB low). With ``peak_dbfs`` the peak drive voltage
    ``drive_peak_v`` is added. The source level uses the projector's nominal 1 kHz
    sensitivity, so it is an uncalibrated estimate.
    """
    k = vrms_at_0dbfs * 10 ** ((amp_gain_db - SINE_RMS_DBFS) / 20)  # volts per unit sample (RMS = RMS)
    v = k * 10 ** (rms_dbfs / 20)
    out = {"drive_vrms": v, "SL_est_dB": sl_from_vrms(v, sl_at_20vrms)}
    if peak_dbfs is not None:
        out["drive_peak_v"] = k * 10 ** (peak_dbfs / 20)
    return out


def tone_drive(rms_dbfs: float, peak_dbfs: float, tone_vrms: float, tone_rms_dbfs: float,
               sl_at_20vrms: float = LL916_SL_DB_AT_20VRMS) -> dict[str, float]:
    """Drive of a file from the measured Vrms of the level tone (``tone_1k.wav``, sample RMS ``tone_rms_dbfs``).

    The chain is linear, so volts per unit sample is ``tone_vrms / 10^(tone_rms_dbfs / 20)``; a file
    of active sample RMS ``rms_dbfs`` drives that times its RMS, and its sample peak maps to the peak
    drive voltage ``drive_peak_v`` (compare with the amplifier supply: `PEAK_SUPPLY_FRACTION`).
    """
    k = tone_vrms / 10 ** (tone_rms_dbfs / 20)
    v = k * 10 ** (rms_dbfs / 20)
    return {"drive_vrms": v, "drive_peak_v": k * 10 ** (peak_dbfs / 20), "SL_est_dB": sl_from_vrms(v, sl_at_20vrms)}


def tone_target_vrms(target_sl_db: float, speech_active_rms_dbfs: float, tone_rms_dbfs: float,
                     sl_at_20vrms: float = LL916_SL_DB_AT_20VRMS) -> float:
    """Tone Vrms to dial in so that speech at ``speech_active_rms_dbfs`` reaches ``target_sl_db`` (nominal)."""
    v_speech = 20.0 * 10 ** ((target_sl_db - sl_at_20vrms) / 20)
    return float(v_speech * 10 ** ((tone_rms_dbfs - speech_active_rms_dbfs) / 20))


def loopback_drive(rms_dbfs: float, in_dbfs_at_1vrms: float, atten_db: float) -> float:
    """Amplifier output Vrms from a loopback level through an ``atten_db`` attenuator."""
    return 10 ** ((rms_dbfs - in_dbfs_at_1vrms + atten_db) / 20)


# ------------------------------------------------------------------ channel map
@dataclass
class Recorder:
    name: str
    fs: int
    loopback: int | None  # file channel of the loopback, if this recorder has one
    hydrophones: dict[int, int]  # file channel -> benchmark channel id
    other: dict[int, str] = field(default_factory=dict)  # file channel -> role (e.g. monitor)
    note: str = ""
    board: str | None = None  # Daisy Seed USB serial (record_daisy.py --board)

    @property
    def records(self) -> bool:
        """Does this recorder capture any input (else it only plays, e.g. the interface without loopback)?"""
        return self.loopback is not None or bool(self.hydrophones) or bool(self.other)

    @property
    def has_sync(self) -> bool:
        """Can its files be chirp-aligned (a loopback or a hydrophone)?"""
        return self.loopback is not None or bool(self.hydrophones)


class ChannelMap:
    """Recorders and their channels (``configs/session_channels_*.json``).

    Each recorder writes one file per take. A file channel's role is
    ``loopback`` (amplifier output through an attenuator), ``hydrophone``
    (with its benchmark ``channel_id``, used in utt_ids) or anything else
    (recorded, not imported, e.g. ``monitor``).
    """

    def __init__(self, d: dict):
        self.raw = d
        self.recorders: dict[str, Recorder] = {}
        ids: list[int] = []
        for name, r in d["recorders"].items():
            lb, hyd, other = None, {}, {}
            for ch, c in r["channels"].items():
                ch = int(ch)
                if c["role"] == "loopback":
                    lb = ch
                elif c["role"] == "hydrophone":
                    hyd[ch] = int(c["channel_id"])
                else:
                    other[ch] = c["role"]
            ids += hyd.values()
            self.recorders[name] = Recorder(name, int(r["fs"]), lb, hyd, other, r.get("note", ""),
                                            str(r["board"]) if r.get("board") else None)
        if len(set(ids)) != len(ids):
            raise ValueError(f"hydrophone channel_ids repeat across recorders: {sorted(ids)}")
        self.scored_channel = d.get("scored_channel")

    @classmethod
    def load(cls, path: str | Path) -> "ChannelMap":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def sync_channel(self, rec: Recorder) -> int:
        """Loopback if recorded on this file (exact), else its first hydrophone (acoustic chirps)."""
        if rec.loopback is not None:
            return rec.loopback
        if not rec.hydrophones:
            raise ValueError(f"{rec.name} records neither a loopback nor a hydrophone")
        return min(rec.hydrophones)


# ------------------------------------------------------------------ QC and import
def qc_take(path: str | Path, bench: str | Path, position: str, anchor: bool, sync: int,
            psr_db: float = 6.0, max_drift_ppm: float = 200.0, played: str | None = None) -> dict[str, Any]:
    """Chirp alignment of a take on its sync channel; ``ok`` means every chirp passed.

    ``played`` names a pool-plan session file (`ingest.load_played`). Only the sync channel
    is loaded (block-wise, float32), so a 35 min two-channel 96 kHz take stays well under 1 GB.
    """
    fs, blocks = ingest.load_blocks(bench, position, anchor=anchor, played=played)
    import soundfile as sf

    n_ch = sf.info(str(path)).channels
    if sync >= n_ch:
        return {"ok": False, "error": f"{Path(path).name} has {n_ch} channels, sync {sync}"}
    x, fs_raw, bits, n_ch, peak = ingest.load_channel(path, sync, fs)
    lv = file_levels(x, fs)
    try:
        results, qc = ingest.align_take(x[None, :], fs, blocks, sync=0, max_drift_ppm=max_drift_ppm,
                                        psr_threshold_db=psr_db)
    except (RuntimeError, ValueError) as e:  # no chirp found / recording shorter than the file
        return {"ok": False, "error": str(e), "fs_raw": fs_raw, "n_channels": n_ch,
                "sync_peak_dBFS": round(lv["peak_dBFS"], 1)}
    flagged = [r.block.block_id for r in results if r.flags]
    clip = peak >= 0.999
    ok = qc["n_chirps_good"] == qc["n_chirps"] and not flagged and not clip
    return {"ok": ok, "fs_raw": fs_raw, "bits": bits, "n_channels": n_ch,
            "sync_peak_dBFS": round(lv["peak_dBFS"], 1), "clip": clip,
            "coarse_offset_s": round(qc["clock_offset"] / fs, 4), "take_drift_ppm": round(qc["take_drift_ppm"], 2),
            "n_chirps": qc["n_chirps"], "n_chirps_good": qc["n_chirps_good"],
            "min_psr_dB": round(min(c.psr_db for r in results for c in r.chirps), 1),
            "flagged_blocks": flagged}


IMPORT_SUBDIRS = {"chainref": "chain-reference cuts (ref_speech)", "pilot": "pilot cuts (pilot_speech)",
                  "bubble": "live-bubble cuts (p<d>bub_speech)"}


SUMS_FILE = "SHA256SUMS.txt"


def write_sha256sums(bench: str | Path) -> tuple[Path, str]:
    """Write ``<bench>/SHA256SUMS.txt`` and return ``(path, sha256 of that file)``.

    One ``<sha256>  <path>`` line per file under the bench (every file except the manifest
    itself), paths relative to the bench in posix form, sorted by path, UTF-8 without BOM,
    LF line ends: the same bytes on any machine, so the manifest's own hash (recorded in the
    frozen record) can be re-checked later with ``sha256sum -c SHA256SUMS.txt`` from the bench.
    """
    import hashlib

    bench = Path(bench)
    lines = []
    for f in sorted((q for q in bench.rglob("*") if q.is_file() and q.name != SUMS_FILE),
                    key=lambda q: q.relative_to(bench).as_posix()):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {f.relative_to(bench).as_posix()}\n")
    data = "".join(lines).encode("utf-8")
    out = bench / SUMS_FILE
    out.write_bytes(data)
    return out, hashlib.sha256(data).hexdigest()


def import_subdir(take: Take) -> str:
    """Sub-directory of the session's import dir for a pool take ("" = the main quiet dir).

    The main ``align.jsonl`` holds only the per-distance ``speech`` / ``anchor`` takes, because
    ``make_score_manifest.py`` and ``score_session.py`` score every row of it as the quiet
    condition. The chain reference, the pilot and the bubble pass each get their own dir.
    Legacy (thruster) takes have no program and always go to the main dir.
    """
    if take.take_id.startswith("ref_"):
        return "chainref"
    if take.program == "pilot":
        return "pilot"
    if take.program == "bubble" or take.bubbles:
        return "bubble"
    return ""


def import_argv(take: Take, rec: Recorder, rec_path: str | Path, cmap: ChannelMap, bench: str | Path,
                session: int, out: str | Path) -> list[str]:
    """``scripts/import_recording.py`` arguments for one recorder file of a speech or anchor take."""
    if not take.importable:
        raise ValueError(f"{take.take_id} ({take.kind}) is not imported")
    if not rec.hydrophones:
        raise ValueError(f"{rec.name} records no hydrophone")
    chans = sorted(rec.hydrophones)
    argv = ["--bench", str(bench), "--position", str(take.position_id), "--rec", str(rec_path),
            "--session", str(session), "--channels", ",".join(map(str, chans)),
            "--channel-ids", ",".join(str(rec.hydrophones[c]) for c in chans),
            "--sync-channel", str(cmap.sync_channel(rec)), "--out", str(out)]
    if take.played:
        argv += ["--played", take.played]
    elif take.kind == "anchor":
        argv.append("--anchor")
    return argv


def plan_json(plan: Sequence[Take]) -> list[dict]:
    return [asdict(t) for t in plan]
