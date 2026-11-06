"""Run-time support for a pool recording session (RESEARCH_PLAN.md 6.4-6.6, 6.8).

Everything here is hardware-free, so it can be tested on synthetic takes; the
audio I/O lives in ``scripts/run_session.py``.

- `build_plan`: the ordered take list for one session (noise, sweeps, speech per
  position, then the drift anchor and the extra noise-bank takes), with each take's
  chirp onsets and the thruster cues an operator must follow.
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
#: received-level rule (6.2): <= 150 dB re 1 uPa where anyone is in the water
RL_LIMIT_DB = 150.0
EXTRA_NOISE = ("quiet_pool", "thruster", "bubbles_splashing", "pumps")


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


def build_plan(bench: str | Path, noise_s: float = 60.0, extra_noise: Sequence[str] = EXTRA_NOISE,
               extra_noise_s: float = 90.0) -> list[Take]:
    """Session take order (TAKE_SHEET.md of ``make_session_audio.py``), one `Take` each."""
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
    return plan


def utterance_rows(bench: str | Path, pos: str, anchor: bool = False) -> list[dict[str, Any]]:
    """One row per utterance of a played file: where it is in the file and what is said."""
    fs, blocks = ingest.load_session(bench, pos, anchor=anchor)
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


def drive_estimate(rms_dbfs: float, vrms_at_0dbfs: float, amp_gain_db: float,
                   sl_at_20vrms: float = LL916_SL_DB_AT_20VRMS) -> dict[str, float]:
    """Projector drive (Vrms) and nominal source level for a digital RMS level.

    ``vrms_at_0dbfs`` is the playback interface's output at 0 dBFS (sine RMS),
    ``amp_gain_db`` the amplifier's voltage gain. The source level uses the
    projector's nominal 1 kHz sensitivity, so it is an uncalibrated estimate.
    """
    v = vrms_at_0dbfs * 10 ** ((rms_dbfs + amp_gain_db) / 20)
    return {"drive_vrms": v, "SL_est_dB": sl_at_20vrms + 20 * np.log10(v / 20.0)}


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
            self.recorders[name] = Recorder(name, int(r["fs"]), lb, hyd, other, r.get("note", ""))
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
            psr_db: float = 6.0, max_drift_ppm: float = 200.0) -> dict[str, Any]:
    """Chirp alignment of a take on its sync channel; ``ok`` means every chirp passed."""
    fs, blocks = ingest.load_session(bench, position, anchor=anchor)
    x, fs_raw, bits = ingest.load_take(path, fs)
    if sync >= x.shape[0]:
        return {"ok": False, "error": f"{Path(path).name} has {x.shape[0]} channels, sync {sync}"}
    lv = file_levels(x[sync], fs)
    results, qc = ingest.align_take(x[sync:sync + 1], fs, blocks, sync=0, max_drift_ppm=max_drift_ppm,
                                    psr_threshold_db=psr_db)
    flagged = [r.block.block_id for r in results if r.flags]
    clip = float(np.max(np.abs(x))) >= 0.999
    ok = qc["n_chirps_good"] == qc["n_chirps"] and not flagged and not clip
    return {"ok": ok, "fs_raw": fs_raw, "bits": bits, "n_channels": int(x.shape[0]),
            "sync_peak_dBFS": round(lv["peak_dBFS"], 1), "clip": clip,
            "coarse_offset_s": round(qc["clock_offset"] / fs, 4), "take_drift_ppm": round(qc["take_drift_ppm"], 2),
            "n_chirps": qc["n_chirps"], "n_chirps_good": qc["n_chirps_good"],
            "min_psr_dB": round(min(c.psr_db for r in results for c in r.chirps), 1),
            "flagged_blocks": flagged}


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
    if take.kind == "anchor":
        argv.append("--anchor")
    return argv


def plan_json(plan: Sequence[Take]) -> list[dict]:
    return [asdict(t) for t in plan]
