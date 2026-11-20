"""Join rendered playback blocks into session files, plus a sweep file, a level tone and a take sheet.

Run after ``scripts/make_playlists.py``. The blocks stay the unit of analysis (each
carries its own chirps); these files only make a session easy to run from any
player: one file per take, played start to finish.

Usage::

    python scripts/make_session_audio.py --bench data/bench/v1          # pool plan bench
    python scripts/make_session_audio.py --bench data/bench/v0_pilot    # legacy (per position)

Pool-plan bench (``make_playlists.py --plan pool``), under ``<bench>/session/``:

- ``pilot.wav``, ``speech.wav``, ``anchor.wav``, ``bubble.wav`` (+ ``.json`` block index each):
  one file per program, blocks in schedule order separated by ``--gap-s``. A program
  longer than ``--max-take-min`` (default 35) is split at block boundaries into
  ``<program>_1.wav``, ``<program>_2.wav``, ... so no take is longer than that. Files are
  written block by block, never held in memory whole.
- ``sweeps.wav`` (+ ``.json``) as below; ``tone_1k.wav`` (+ ``.json``): 1 kHz level-setting
  tone at ``--tone-rms-dbfs`` (default the -26 dBFS speech RMS target).
- ``programs.json``  the files above in play order; ``run_session.py plan`` reads it.
- ``TAKE_SHEET.md``  file list (the run sheet itself is per session, from ``run_session.py plan``).
- ``<bench>/SHA256SUMS.txt``  sha256 of every bench file (sorted posix paths, no BOM), written
  last; its own sha256 is printed for the frozen record. ``--sums-only`` rewrites just this
  manifest for a bench that is already built.

Legacy bench (thruster schedule with positions), under ``<bench>/session/``:

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
from uwse.bench import session as ss  # noqa: E402
from uwse.irtools.ess import generate_ess  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Session WAVs, sweep file, level tone and take sheet.")
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
    ap.add_argument("--max-take-min", type=float, default=35.0,
                    help="pool plan: split a program into parts so no file is longer (minutes)")
    ap.add_argument("--tone-s", type=float, default=60.0, help="pool plan: level-setting tone length (s)")
    ap.add_argument("--sums-only", action="store_true",
                    help="only (re)write <bench>/SHA256SUMS.txt of an already built bench and print its hash")
    ap.add_argument("--tone-hz", type=float, default=1000.0)
    ap.add_argument("--tone-rms-dbfs", type=float, default=pl.TARGET_RMS_DBFS,
                    help="tone RMS; run_session.py levels converts the measured tone Vrms to speech drive")
    a = ap.parse_args(argv)
    if a.max_take_min <= 0:
        ap.error("--max-take-min must be > 0")
    return a


def write_wav(path: Path, x: np.ndarray, fs: int) -> None:
    peak = float(np.max(np.abs(x)))
    if peak >= 1.0:
        raise ValueError(f"{path.name} clips (peak {peak:.3f})")
    audio.save(str(path), x, fs, subtype="PCM_24")


def write_sweeps(a: argparse.Namespace, out: Path, fs: int) -> float:
    """``sweeps.wav`` + ``sweeps.json``; returns the file length in minutes."""
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
    return len(sweeps) / fs / 60


def split_parts(lengths: list[int], max_n: int) -> list[list[int]]:
    """Contiguous groups of item indices: the fewest parts with every total <= ``max_n``, balanced.

    The part count k is that of a greedy fill at ``max_n`` (optimal for contiguous groups); the
    smallest capacity that still needs only k parts is then found by bisection, so the parts come
    out about equal. A single item longer than ``max_n`` gets a part of its own.
    """
    if not lengths:
        return []

    def greedy(cap: int) -> list[list[int]]:
        groups: list[list[int]] = [[]]
        acc = 0
        for i, n in enumerate(lengths):
            if groups[-1] and acc + n > cap:
                groups.append([])
                acc = 0
            groups[-1].append(i)
            acc += n
        return groups

    k = len(greedy(max_n))
    lo, hi = 1, max_n  # caps <= max_n keep every multi-item part within it
    while lo < hi:
        mid = (lo + hi) // 2
        if len(greedy(mid)) <= k:
            hi = mid
        else:
            lo = mid + 1
    return greedy(lo)


def main_pool(a: argparse.Namespace, sched: list[pl.ScheduleEntry]) -> int:
    """One file (or part) per program of a pool-plan bench; see the module docstring."""
    import soundfile as sf

    blocks_dir, out = a.bench / "blocks", a.bench / "session"
    out.mkdir(parents=True, exist_ok=True)
    fs = pl.FS_PLAYBACK
    gap_n = int(round(a.gap_s * fs))
    max_n = int(a.max_take_min * 60 * fs)
    files, summary = [], {}
    for prog in pl.POOL_PROGRAMS:
        entries = [e for e in sched if e.program == prog]
        if not entries:
            continue
        timings = [json.loads((blocks_dir / f"{e.content_block}.json").read_text(encoding="utf-8"))
                   for e in entries]
        parts = split_parts([int(t["n_samples"]) + gap_n for t in timings], max_n)
        for k, grp in enumerate(parts, start=1):
            stem = prog if len(parts) == 1 else f"{prog}_{k}"
            index, n, n_utts = [], 0, 0
            with sf.SoundFile(str(out / f"{stem}.wav"), "w", samplerate=fs, channels=1, subtype="PCM_24") as f:
                for i in tqdm(grp, desc=f"join {stem}"):
                    e = entries[i]
                    x, f_in = audio.load(str(blocks_dir / f"{e.content_block}.wav"))
                    if f_in != fs:
                        raise ValueError(f"{e.content_block}: {f_in} Hz, expected {fs}")
                    if float(np.max(np.abs(x))) >= 1.0:
                        raise ValueError(f"{e.content_block} clips")
                    index.append({"session_order_idx": e.session_order_idx, "block_id": e.block_id,
                                  "content_block": e.content_block, "thruster_state": e.thruster_state,
                                  "anchor_repeat": e.anchor_repeat, "start_sample": n, "n_samples": len(x)})
                    f.write(x)
                    f.write(np.zeros(gap_n))
                    n += len(x) + gap_n
                    n_utts += len(timings[i]["utterances"])
            with open(out / f"{stem}.json", "w", encoding="utf-8", newline="\n") as f:
                json.dump({"program": prog, "part": k, "n_parts": len(parts), "fs": fs, "gap_s": a.gap_s,
                           "blocks": index}, f, indent=1)
            files.append({"program": prog, "file": f"{stem}.wav", "index": f"{stem}.json", "part": k,
                          "n_parts": len(parts), "n_blocks": len(grp), "n_utts": n_utts,
                          "minutes": n / fs / 60})
            summary[stem] = {"blocks": len(grp), "utts": n_utts, "minutes": round(n / fs / 60, 2)}

    summary["sweeps"] = {"n": a.n_sweeps, "minutes": round(write_sweeps(a, out, fs), 2)}
    nt = int(round(a.tone_s * fs))
    tone = np.sqrt(2) * 10 ** (a.tone_rms_dbfs / 20) * np.sin(2 * np.pi * a.tone_hz * np.arange(nt) / fs)
    write_wav(out / "tone_1k.wav", audio.fade(tone, fs, 0.05, 0.05), fs)
    with open(out / "tone_1k.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"fs": fs, "freq_hz": a.tone_hz, "rms_dBFS": a.tone_rms_dbfs,
                   "peak_dBFS": a.tone_rms_dbfs + 20 * float(np.log10(np.sqrt(2))), "seconds": a.tone_s}, f,
                  indent=1)
    summary["tone_1k"] = {"minutes": round(a.tone_s / 60, 2)}
    with open(out / "programs.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"plan": "pool", "fs": fs, "max_take_min": a.max_take_min, "files": files,
                   "sweeps": "sweeps.wav", "tone": "tone_1k.wav"}, f, indent=1)
    rows = "".join(f"| {k} | {v['minutes']:.1f} |\n" for k, v in summary.items())
    (out / "TAKE_SHEET.md").write_text(
        "# Pool-plan session files\n\n48 kHz / 24-bit mono. The take order, distances and log columns are in "
        "the per-session\nRUN_SHEET.md written by `python scripts/run_session.py plan --bench <this bench> ...`.\n\n"
        f"| file | minutes |\n|---|---|\n{rows}", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    write_sums(a.bench)
    return 0


def write_sums(bench: Path) -> None:
    """Write ``<bench>/SHA256SUMS.txt`` (`uwse.bench.session.write_sha256sums`) and print its hash."""
    path, digest = ss.write_sha256sums(bench)
    n = len(path.read_text(encoding="utf-8").splitlines())
    print(f"{path}: {n} files; sha256 of the manifest {digest} (record it with the build's git hash)")


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    frozen = {(ss.PROJECT_ROOT / "data" / "bench" / n).resolve() for n in ("v0_s1", "v0_pilot")}
    if a.bench.resolve() in frozen:
        raise SystemExit(f"{a.bench} is a frozen, hashed bench; never rewrite its session files or SHA256SUMS.txt")
    sched = pl.read_schedule(a.bench / "schedule.json")
    if a.sums_only:
        write_sums(a.bench)
        return 0
    if pl.is_pool_schedule(sched):
        return main_pool(a, sched)
    blocks_dir, out = a.bench / "blocks", a.bench / "session"
    out.mkdir(parents=True, exist_ok=True)
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

    summary["sweeps"] = {"n": a.n_sweeps, "minutes": write_sweeps(a, out, fs)}

    rows = "\n".join(f"| {k} | {v['minutes']:.1f} |" for k, v in summary.items())
    takes = "\n".join(f"| {p} | `{p}_speech.wav` | | | | | |" for p in positions)
    sheet = f"""# Pool session take sheet (legacy per-position bench)

Files in this folder, 48 kHz / 24-bit mono. Play each start to finish at ONE fixed
amplifier gain for the whole session (never touch it mid-session; write it down).

| file | minutes |
|---|---|
{rows}

## Order

1. Rig the projector at 1.0 m depth, the hydrophone(s) at 1.0 m depth, on-axis.
2. For each position (range = projector-to-hydrophone distance):
   1. Measure and log the range and depths.
   2. Record a 60 s noise-only take (nothing playing). If the position has thruster-on
      blocks, record a second one with the thruster on at the speech-take throttle.
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
