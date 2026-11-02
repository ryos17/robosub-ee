"""Join rendered playback blocks into one WAV per position, plus a sweep file and a take sheet.

Run after ``scripts/make_playlists.py``. The blocks stay the unit of analysis (each
carries its own chirps); these files only make a session easy to run from any
player: one file per position, played start to finish.

Usage::

    python scripts/make_session_audio.py --bench data/bench/v0_pilot

Writes, under ``<bench>/session/``:

- ``<pos>_speech.wav`` (+ ``.json``)  the schedule's blocks for that position, in order,
  separated by ``--gap-s`` of extra silence; the JSON lists each block's start sample.
- ``anchor_<pos>.wav``  the drift-anchor repeat (the session's first block), if scheduled.
- ``sweeps.wav`` (+ ``.json``)  sync chirp, then ``--n-sweeps`` exponential sweeps with
  a low-frequency shelf so the projector is not over-driven, each followed by silence.
- ``TAKE_SHEET.md``  what to play where, with a per-take log table.
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
from uwse.bench import playlist as pl  # noqa: E402
from uwse.irtools.ess import generate_ess  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Per-position session WAVs, sweep file and take sheet.")
    ap.add_argument("--bench", type=Path, required=True, help="output dir of make_playlists.py")
    ap.add_argument("--gap-s", type=float, default=1.0, help="extra silence between blocks (s)")
    ap.add_argument("--sweep-f1", type=float, default=150.0, help="sweep start (Hz), LL916 lower limit")
    ap.add_argument("--sweep-f2", type=float, default=20000.0)
    ap.add_argument("--sweep-T", type=float, default=10.0, help="sweep duration (s), plan 6.4: 10 s")
    ap.add_argument("--sweep-silence-s", type=float, default=5.0, help=">= 4 s (plan 6.4)")
    ap.add_argument("--n-sweeps", type=int, default=4)
    ap.add_argument("--sweep-peak-dbfs", type=float, default=-20.0,
                    help="sweep peak; its RMS (peak - 3 dB) sits near the speech level")
    ap.add_argument("--lf-shelf-hz", type=float, default=300.0)
    ap.add_argument("--lf-shelf-db", type=float, default=12.0)
    return ap.parse_args(argv)


def write_wav(path: Path, x: np.ndarray, fs: int) -> None:
    peak = float(np.max(np.abs(x)))
    if peak >= 1.0:
        raise ValueError(f"{path.name} clips (peak {peak:.3f})")
    audio.save(str(path), x, fs, subtype="PCM_24")


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    blocks_dir, out = a.bench / "blocks", a.bench / "session"
    out.mkdir(parents=True, exist_ok=True)
    sched = pl.read_schedule(a.bench / "schedule.json")
    fs = pl.FS_PLAYBACK
    gap = np.zeros(int(round(a.gap_s * fs)))

    positions = list(dict.fromkeys(e.position_id for e in sched if not e.anchor_repeat))
    summary = {}
    for pos in positions:
        entries = [e for e in sched if e.position_id == pos and not e.anchor_repeat]
        parts, index, n = [], [], 0
        for e in tqdm(entries, desc=f"join {pos}"):
            x, f_in = audio.load(str(blocks_dir / f"{e.content_block}.wav"))
            if f_in != fs:
                raise ValueError(f"{e.content_block}: {f_in} Hz, expected {fs}")
            index.append({"session_order_idx": e.session_order_idx, "content_block": e.content_block,
                          "thruster_state": e.thruster_state, "start_sample": n, "n_samples": len(x)})
            parts += [x, gap]
            n += len(x) + len(gap)
        sig = np.concatenate(parts)
        write_wav(out / f"{pos}_speech.wav", sig, fs)
        with open(out / f"{pos}_speech.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"position_id": pos, "fs": fs, "gap_s": a.gap_s, "blocks": index}, f, indent=1)
        summary[pos] = {"blocks": len(entries), "minutes": len(sig) / fs / 60}

    for e in (e for e in sched if e.anchor_repeat):
        x, _ = audio.load(str(blocks_dir / f"{e.content_block}.wav"))
        write_wav(out / f"anchor_{e.position_id}.wav", x, fs)
        summary[f"anchor_{e.position_id}"] = {"content_block": e.content_block, "minutes": len(x) / fs / 60}

    ess = generate_ess(a.sweep_f1, a.sweep_f2, a.sweep_T, fs, amplitude=10 ** (a.sweep_peak_dbfs / 20),
                       silence_s=a.sweep_silence_s, lf_shelf_hz=a.lf_shelf_hz, lf_shelf_db=a.lf_shelf_db)
    lay = pl.BlockLayout()
    ch = pl.lfm_chirp(fs, lay.chirp_s)
    lead = np.zeros(int(round(lay.lead_silence_s * fs)))
    post = np.zeros(int(round(1.0 * fs)))
    sweeps = np.concatenate([lead, ch, post] + [ess.signal] * a.n_sweeps)
    write_wav(out / "sweeps.wav", sweeps, fs)
    s0 = len(lead) + len(ch) + len(post)
    with open(out / "sweeps.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"fs": fs, "f1": ess.f1, "f2": ess.f2, "T": ess.T, "L": ess.L,
                   "peak_dBFS": a.sweep_peak_dbfs, "lf_shelf_hz": a.lf_shelf_hz,
                   "lf_shelf_db": a.lf_shelf_db, "synchronized": ess.synchronized,
                   "chirp_start_sample": len(lead),
                   "sweep_start_samples": [s0 + k * len(ess.signal) for k in range(a.n_sweeps)],
                   "sweep_n_samples": len(ess.excitation), "silence_s": a.sweep_silence_s}, f, indent=1)
    summary["sweeps"] = {"n": a.n_sweeps, "minutes": len(sweeps) / fs / 60}

    rows = "\n".join(f"| {k} | {v['minutes']:.1f} |" for k, v in summary.items())
    takes = "\n".join(f"| {p} | `{p}_speech.wav` | | | | | |" for p in positions)
    sheet = f"""# Pool session take sheet

Files in this folder, 48 kHz / 24-bit mono. Play each start to finish at ONE fixed
amplifier gain for the whole session (never touch it mid-session; write it down).

| file | minutes |
|---|---|
{rows}

## Order

1. Rig the projector at 1.0 m depth, the hydrophone(s) at 1.0 m depth, on-axis.
2. For each position (range = projector-to-hydrophone distance):
   1. Measure and log the range and depths.
   2. Record a 60 s noise-only take (nothing playing).
   3. Play `sweeps.wav`.
   4. Play `<pos>_speech.wav`.
3. Back at the first position, play `anchor_*.wav` (drift check).
4. Extra noise-only takes, 1-2 min each: quiet pool, thruster, bubbles/splashing, pumps.

Recorder: start BEFORE the file starts and stop AFTER it ends (keep both sync chirps).
If possible, record the amplifier output on a second input through an attenuator
(loopback), so alignment is exact.

## Log

| position | file | range (m) | proj depth (m) | hyd depth (m) | rec file name | notes |
|---|---|---|---|---|---|---|
{takes}
| noise | (none) | | | | | |

Session: date ______ pool ______ water temp ______ amp gain ______ rec gain ______
"""
    (out / "TAKE_SHEET.md").write_text(sheet, encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
