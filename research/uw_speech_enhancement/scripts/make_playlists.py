"""Build the v0 (or v1) playback material: selection, blocks, ABBA schedule (RESEARCH_PLAN.md 6.5, 6.6).

Usage::

    python scripts/make_playlists.py --librispeech-root data/librispeech --out data/bench/v0
    python scripts/make_playlists.py --release v1 --positions p05 p1 ... --no-thruster

Position ids must match the utt_id grammar ``p[0-9a-z]+`` (uwse.bench.schema),
so 0.5 m is written ``p05`` and 4.5 m ``p4p5`` or similar, never ``p0.5``.

Writes, under ``--out``:

- ``manifests/<subset>.jsonl``  cached LibriSpeech manifests;
- ``selection/<split>.jsonl``   selected utterances per bench split;
- ``blocks/<split>_NNN.wav/.json``  48 kHz playback blocks + timing (unless --no-render);
- ``schedule.json``             session playback order (ABBA, drift anchor);
- ``budget.json``               playback time (plan formula and actual layout).

The split checker (5.4) runs before anything is rendered.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwse.bench import librispeech as ls  # noqa: E402
from uwse.bench import playlist as pl  # noqa: E402
from uwse.bench.splits import assert_disjoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build benchmark playback blocks and schedule.")
    ap.add_argument("--librispeech-root", type=Path, default=ROOT / "data" / "librispeech")
    ap.add_argument("--out", type=Path, default=None, help="default data/bench/<release>")
    ap.add_argument("--release", choices=("v0", "v1"), default="v0")
    ap.add_argument("--seed", type=int, default=ls.DEFAULT_SEED)
    ap.add_argument("--positions", nargs="+", default=["p1", "p3", "p6"],
                    help="position ids in playback order (v0 core: 1, 3, 6 m)")
    ap.add_argument("--v1-n", type=int, default=1000, help="v1-test size (>= 1000)")
    ap.add_argument("--f-lo", type=float, default=pl.CHIRP_F0_HZ, help="chirp start frequency (Hz)")
    ap.add_argument("--target-speech-s", type=float, default=60.0, help="speech per block (s)")
    ap.add_argument("--max-utts", type=int, default=None, help="cap on utterances per block")
    ap.add_argument("--first-state", choices=pl.STATES, default="off")
    ap.add_argument("--no-thruster", action="store_true", help="every block once, thruster off")
    ap.add_argument("--no-anchor", action="store_true", help="omit the drift-anchor repeat")
    ap.add_argument("--no-render", action="store_true", help="selection + schedule only")
    ap.add_argument("--min-dur", type=float, default=ls.MIN_DUR_S)
    ap.add_argument("--max-dur", type=float, default=ls.MAX_DUR_S)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = ROOT / "data" / "bench" / args.release
    if not 0 < args.f_lo < pl.CHIRP_F1_HZ:
        ap.error("--f-lo must be in (0, 8000)")
    if args.release == "v1" and args.v1_n < 1000:
        ap.error("--v1-n must be >= 1000")
    bad = [p for p in args.positions if not re.fullmatch(r"p[0-9a-z]+", p)]
    if bad or len(set(args.positions)) != len(args.positions):
        ap.error(f"--positions must be unique ids matching p[0-9a-z]+ (utt_id grammar), got {args.positions}")
    return args


def manifest(root: Path, subset: str, cache_dir: Path) -> list[ls.Utterance]:
    cache = cache_dir / f"{subset}.jsonl"
    if cache.exists():
        return ls.read_manifest(cache)
    m = ls.build_manifest(ls.find_subset_dir(root, subset), subset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ls.write_manifest(cache, m)
    return m


def interleave(main: list[str], extra: list[str]) -> list[str]:
    """Spread ``extra`` blocks evenly through ``main`` (dev among test blocks)."""
    if not extra:
        return list(main)
    out, step = [], (len(main) + 1) / (len(extra) + 1)
    slots = {int(round(step * (k + 1))) for k in range(len(extra))}
    it = iter(extra)
    for i in range(len(main) + 1):
        if i in slots:
            out.append(next(it))
        if i < len(main):
            out.append(main[i])
    out.extend(it)
    return out


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    out = a.out
    test = manifest(a.librispeech_root, "test-clean", out / "manifests")
    dev = manifest(a.librispeech_root, "dev-clean", out / "manifests")
    v0_test = ls.select_v0_test(test, seed=a.seed, min_s=a.min_dur, max_s=a.max_dur)
    v0_dev = ls.select_v0_dev(dev, seed=a.seed, min_s=a.min_dur, max_s=a.max_dur)
    if a.release == "v0":
        sel = {"v0-test": v0_test, "v0-dev": v0_dev}
    else:
        v1 = ls.select_v1_test(test, v0_test, n=a.v1_n, seed=a.seed, min_s=a.min_dur, max_s=a.max_dur)
        sel = {"v0-test": v0_test, "v1-test": v1}
    assert_disjoint(sel)
    (out / "selection").mkdir(parents=True, exist_ok=True)
    for name, utts in sel.items():
        ls.write_manifest(out / "selection" / f"{name}.jsonl", utts)
        print(f"{name}: {ls.summary(utts)}")

    play = {k: v for k, v in sel.items() if k.startswith(a.release) or k == "v0-dev"}
    blocks = {k: pl.pack_blocks(v, target_speech_s=a.target_speech_s, max_utts=a.max_utts,
                                seed=a.seed) for k, v in play.items()}
    names = {k: [f"{k}_{i:03d}" for i in range(len(b))] for k, b in blocks.items()}
    if not a.no_render:
        for k, b in blocks.items():
            pl.render_blocks(b, out / "blocks", prefix=k, f_lo=a.f_lo, bench_split=k)
    test_key = f"{a.release}-test"
    content = interleave(names[test_key], names.get("v0-dev", []) if a.release == "v0" else [])
    sched = pl.session_schedule(a.positions, content, a.first_state,
                                thruster=not a.no_thruster, anchor=not a.no_anchor)
    pl.write_schedule(out / "schedule.json", sched)

    n_test, n_dev = len(play[test_key]), len(play.get("v0-dev", []))
    all_utts = [u for v in play.values() for u in v]
    mean = sum(u.duration_s for u in all_utts) / len(all_utts)
    n_blocks = sum(len(b) for b in blocks.values())
    budget = pl.playback_budget(len(a.positions), 1 if a.no_thruster else 2, n_test, n_dev, mean,
                                utts_per_block=max(1, round(len(all_utts) / n_blocks)),
                                anchor_blocks=0 if a.no_anchor else 1)
    lay = pl.BlockLayout()
    durs = {cb: sum(u.duration_s for u in blk) + lay.overhead_s(len(blk))
            for k, bl in blocks.items() for cb, blk in zip(names[k], bl)}
    actual_s = sum(durs[e.content_block] for e in sched)
    report = {**asdict(budget), "schedule_entries": len(sched),
              "schedule_playback_h": actual_s / 3600.0, "mean_dur_s": mean}
    with open(out / "budget.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
