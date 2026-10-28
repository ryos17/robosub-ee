"""Generate the uwsim RIR bank (RESEARCH_PLAN.md section 5.3).

Writes N (scene, source->receiver RIR, thruster->receiver IR) triples as sharded
npz files plus ``scenes.jsonl`` and ``bank.json`` (layout: uwse/uwsim/dataset.py),
readable with ``uwse.uwsim.dataset.RIRBank``. The calibrated-pool vs generic split
is set by ``--generic-fraction`` (plan: 10,000 + 10,000, i.e. 0.5; the config's
training default is 0.2). Every RIR is seeded from (--seed, index), so a run can be
resumed shard by shard (``--resume`` skips shards whose npz already exists).

One RIR takes a few seconds on CPU (ISM + tail + modal low band), so a full bank is
a long job. Launch it detached with the logging convention used on this machine
(PowerShell, Option C)::

    Start-Process powershell -ArgumentList @(
        '-NoExit','-Command',
        "Set-Location 'C:\\Users\\ryota\\Documents\\repos\\robosub-ee\\research\\uw_speech_enhancement'; & 'C:\\Users\\ryota\\anaconda3\\shell\\condabin\\conda-hook.ps1'; conda activate uwse; python scripts/make_rir_bank.py --n 20000 --generic-fraction 0.5 --out data/rir_bank_v0 *>&1 | Tee-Object -FilePath `"$env:TEMP\\rirbank.log`""
    ) -WindowStyle Minimized

Inspect with ``Get-Content $env:TEMP\\rirbank.log -Wait``; stop by closing the window.
Run one bank job at a time (RAM).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwse.uwsim.config import SimConfig, load_config  # noqa: E402
from uwse.uwsim.dataset import RIRBank  # noqa: E402
from uwse.uwsim.simulate import FS_SIM, render_rir, render_thruster_rir, sample_scene  # noqa: E402


def is_under_datasets(path: str) -> bool:
    """True if ``path`` resolves under /datasets (read-only), also on Windows.

    On Windows ``os.path.abspath('/datasets/x')`` is ``C:/datasets/x`` (backslashes), so the drive is
    stripped before the check.
    """
    _, tail = os.path.splitdrive(os.path.abspath(path))
    tail = tail.replace("\\", "/").rstrip("/")
    return tail == "/datasets" or tail.startswith("/datasets/")


def _write_bank_json(out: str, n_done: int, n_total: int, shards: list[str], args, gf: float,
                     cfg: SimConfig) -> None:
    with open(os.path.join(out, "bank.json"), "w", encoding="utf-8") as f:
        json.dump({"n": n_done, "fs": FS_SIM, "shard_size": args.shard_size, "shards": shards,
                   "seed": args.seed, "generic_fraction": gf, "config": args.config,
                   "calibrated": cfg.calibrated, "complete": n_done == n_total}, f, indent=1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, required=True, help="number of RIRs")
    ap.add_argument("--out", required=True, help="output folder (must not be under /datasets)")
    ap.add_argument("--config", default="configs/uwsim_v0.yaml")
    ap.add_argument("--generic-fraction", type=float, default=None,
                    help="fraction of generic pools (default: config pool.generic_fraction)")
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rir-length", type=float, default=None, help="override acoustics.rir_length_s")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    if is_under_datasets(args.out):
        raise SystemExit("refusing to write under /datasets")
    cfg = load_config(args.config) if os.path.isfile(args.config) else SimConfig()
    if args.rir_length is not None:
        cfg.acoustics.rir_length_s = args.rir_length
    gf = cfg.pool.generic_fraction if args.generic_fraction is None else args.generic_fraction
    os.makedirs(args.out, exist_ok=True)

    n_generic = int(round(gf * args.n))
    # Deterministic, interleaved assignment of generic vs calibrated scenes.
    generic = np.zeros(args.n, bool)
    generic[np.random.default_rng(args.seed).permutation(args.n)[:n_generic]] = True

    shards: list[str] = []
    scenes_path = os.path.join(args.out, "scenes.jsonl")
    scene_lines: dict[int, str] = {}
    if args.resume and os.path.isfile(scenes_path):
        with open(scenes_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                scene_lines[i] = line.rstrip("\n")

    t0 = time.time()
    n_shards = (args.n + args.shard_size - 1) // args.shard_size
    pbar = tqdm(total=args.n, desc="rir bank", unit="rir")
    for k in range(n_shards):
        lo, hi = k * args.shard_size, min(args.n, (k + 1) * args.shard_size)
        name = f"shard_{k:05d}.npz"
        if args.resume and os.path.isfile(os.path.join(args.out, name)) and all(i in scene_lines for i in range(lo, hi)):
            shards.append(name)
            pbar.update(hi - lo)
            continue
        rirs, thrs = [], []
        for i in range(lo, hi):
            sc = sample_scene(cfg, np.random.default_rng((args.seed, i)), generic=bool(generic[i]))
            rirs.append(render_rir(sc).astype(np.float32))
            t = render_thruster_rir(sc)
            thrs.append(None if t is None else t.astype(np.float32))
            scene_lines[i] = json.dumps(sc.to_dict())
            pbar.update(1)
            pbar.set_postfix(generic=int(sc.is_generic), r=f"{sc.range_m:.1f}m")
        shards.append(RIRBank.write_shard(args.out, k, rirs, thrs, range(lo, hi)))
        # Rewrite metadata after every shard so an interrupted run is resumable.
        with open(scenes_path, "w", encoding="utf-8") as f:
            for i in sorted(scene_lines):
                f.write(scene_lines[i] + "\n")
        _write_bank_json(args.out, hi, args.n, shards, args, gf, cfg)
    pbar.close()
    # Final rewrite: shards skipped on --resume after the last regenerated one must count.
    _write_bank_json(args.out, args.n, args.n, shards, args, gf, cfg)
    print(f"wrote {args.n} RIRs in {len(shards)} shards to {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
