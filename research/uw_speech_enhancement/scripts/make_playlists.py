"""Build the v0 (or v1) playback material: selection, blocks, ABBA schedule (RESEARCH_PLAN.md 6.5, 6.6).

Usage::

    # pool plan (2026-11-19, no thruster): positions are chosen on the day by run_session.py plan
    python scripts/make_playlists.py --plan pool --librispeech-root data/librispeech --out data/bench/v1

    # legacy thruster plans (v0 / S1 benches; data/bench/v0_s1 is frozen)
    python scripts/make_playlists.py --librispeech-root data/librispeech --out data/bench/v0
    python scripts/make_playlists.py --release v1 --positions p05 p1 ... --no-thruster

``--plan pool`` selects v0-test (the 200 test utterances), v0-dev (selection only,
for the disjointness check) and v0-pilot (20 dev-clean utterances outside v0-dev,
`uwse.bench.librispeech.select_pilot`), renders the test and pilot blocks and writes a
program schedule (`uwse.bench.playlist.pool_schedule`): pilot, speech (every test block
once), anchor (first test block repeated) and bubble (an evenly spaced, deterministic
subset of test blocks of about ``--bubble-min`` minutes). ``selection/pilot_rule.json``
and ``selection/bubble_rule.json`` record the two choices.

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
    ap.add_argument("--plan", choices=("legacy", "pool"), default="legacy",
                    help="pool: the 2026-11-19 no-thruster pool plan (default out data/bench/v1)")
    ap.add_argument("--pilot-n", type=int, default=20, help="pool: pilot sentences (dev-clean, not v0-dev)")
    ap.add_argument("--bubble-min", type=float, default=10.0, help="pool: bubble-pass speech (minutes)")
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
        args.out = ROOT / "data" / "bench" / ("v1" if args.plan == "pool" else args.release)
    if args.plan == "pool" and args.release != "v0":
        ap.error("--plan pool plays the v0-test utterances; leave --release at v0")
    frozen = {(ROOT / "data" / "bench" / n).resolve() for n in ("v0_s1", "v0_pilot")}
    if args.out.resolve() in frozen:
        ap.error(f"{args.out} is a frozen, hashed bench; write a new bench (e.g. data/bench/v1)")
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


def main_pool(a: argparse.Namespace) -> int:
    """The no-thruster pool plan bench (see the module docstring)."""
    out = a.out
    test = manifest(a.librispeech_root, "test-clean", out / "manifests")
    dev = manifest(a.librispeech_root, "dev-clean", out / "manifests")
    v0_test = ls.select_v0_test(test, seed=a.seed, min_s=a.min_dur, max_s=a.max_dur)
    v0_dev = ls.select_v0_dev(dev, seed=a.seed, min_s=a.min_dur, max_s=a.max_dur)
    pilot = ls.select_pilot(dev, exclude=v0_dev, n=a.pilot_n, seed=a.seed, min_s=a.min_dur, max_s=a.max_dur)
    sel = {"v0-test": v0_test, "v0-dev": v0_dev, "v0-pilot": pilot}
    assert_disjoint(sel)
    if {u.utt_id for u in pilot} & {u.utt_id for u in v0_dev}:
        raise SystemExit("pilot overlaps v0-dev")  # select_pilot excludes it; belt and braces
    (out / "selection").mkdir(parents=True, exist_ok=True)
    for name, utts in sel.items():
        ls.write_manifest(out / "selection" / f"{name}.jsonl", utts)
        print(f"{name}: {ls.summary(utts)}")
    with open(out / "selection" / "pilot_rule.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"split": "v0-pilot", "rule": ls.PILOT_RULE, "seed": a.seed, "n": a.pilot_n,
                   "min_dur_s": a.min_dur, "max_dur_s": a.max_dur,
                   "utt_ids": [u.utt_id for u in pilot]}, f, indent=1)

    blocks = {k: pl.pack_blocks(sel[k], target_speech_s=a.target_speech_s, max_utts=a.max_utts, seed=a.seed)
              for k in ("v0-pilot", "v0-test")}
    names = {k: [f"{k}_{i:03d}" for i in range(len(b))] for k, b in blocks.items()}
    if not a.no_render:
        for k, b in blocks.items():
            pl.render_blocks(b, out / "blocks", prefix=k, f_lo=a.f_lo, bench_split=k)
    lay = pl.BlockLayout()
    durs = {cb: sum(u.duration_s for u in blk) + lay.overhead_s(len(blk))
            for k, bl in blocks.items() for cb, blk in zip(names[k], bl)}
    test_names = names["v0-test"]
    mean_block_s = sum(durs[c] for c in test_names) / len(test_names)
    bub = pl.bubble_subset(len(test_names), round(a.bubble_min * 60.0 / mean_block_s))
    with open(out / "selection" / "bubble_rule.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"rule": "evenly spaced test blocks, numpy.linspace(0, n_blocks - 1, k) rounded, "
                           "k = round(bubble_min * 60 / mean test block length)",
                   "bubble_min": a.bubble_min, "blocks": [test_names[i] for i in bub],
                   "utt_ids": [u.utt_id for i in bub for u in blocks["v0-test"][i]]}, f, indent=1)
    sched = pl.pool_schedule(names["v0-pilot"], test_names, bub, anchor=not a.no_anchor)
    pl.write_schedule(out / "schedule.json", sched)
    minutes: dict[str, float] = {}
    for e in sched:
        minutes[e.program] = minutes.get(e.program, 0.0) + durs[e.content_block] / 60.0
    report = {"plan": "pool", "schedule_entries": len(sched), "program_minutes_est": minutes,
              "n_blocks": {k: len(v) for k, v in names.items()},
              "mean_dur_s": {k: sum(u.duration_s for u in v) / len(v) for k, v in sel.items()},
              "note": "speech minutes exclude the inter-block gaps that make_session_audio.py adds; "
                      "it splits a program longer than --max-take-min into parts"}
    with open(out / "budget.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if a.plan == "pool":
        return main_pool(a)
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
