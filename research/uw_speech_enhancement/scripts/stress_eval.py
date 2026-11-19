"""Held-out mismatch stress suite on the uwsim E7 positions (exploratory).

Asks whether the gains seen on E7 survive conditions that no system trained on:
channel colouration, reverberation, level and noise-family mismatch of the kind
the first real pool data may bring. Every condition perturbs ONE factor of an
E7 cell (same 200 v0-test utterances, same 1 / 3 / 6 m scenes and seed as
`sim_baseline.py`), so each one reads against the matching E7 cell.

Conditions (``q_`` = quiet base, SNR 30 dB; ``n0_`` = noise0 base, SNR 0 dB with
the -3 dB/oct fallback noise). Training ranges are from configs/uwsim_v0_stat.yaml.

- ``tiltm6`` / ``tiltp6``: hydrophone tilt reaching -6 / +6 dB at 8 kHz and +6 / -6 dB at 125 Hz
  (training: within +/-2 dB). Applies to speech and noise (receive side).
- ``ripple9``: projector 1/3-octave ripple U(-9, 9) dB, one fixed draw per
  position (training: +/-3 dB). Speech only (transmit side).
- ``t60x2``: every octave T60 doubled, RIR 2 s (training scale 0.7-1.3).
- ``clip``: speech-active level -6 dBFS, so peaks hard-clip at the ADC
  (training: -35 to -15 dBFS).
- ``n0_slopem9`` / ``n0_slopep3``: stationary noise at -9 / +3 dB/oct, SNR 0
  (training slopes within [-6, 0]).
- ``n0_tonal``: harmonic hum (f0 120 Hz to 4 kHz, -3 dB/oct, wobble) over a
  -15 dB white floor, SNR 0. A noise family absent from training.
- ``n5``: the noise0 noise at SNR -5 dB (inside the training SNR range; a
  harder operating point, not a held-out factor).

Systems: C0 (``noisy``), the frozen C8 adapter (``c8``, noisy audio decoded with
runs/c8_whisper_lora_stat_s1/best_wer.pt), and DFN3-C2 (``dfn3_c2``, bubble-only
fine-tune) as the enhancement-front-end reference, on the quiet-base conditions
only. Every decode uses the frozen guarded policy (`--decode guard`).

Reading rule (fixed before the run, 2026-11-07):

- For each cell, paired speaker-cluster bootstrap dWER (2000 resamples) of
  c8 - noisy and dfn3_c2 - noisy.
- C8's E7 advantage "holds" under a condition if c8 - noisy has its CI below
  zero in every range of that condition; "is lost" if the CI includes zero in
  any range; "reverses" if the CI is above zero in any range. Same for dfn3_c2.
- The E7 reference deltas come from runs/sim_baseline/per_utt_guard.jsonl
  (labels ``c8stat1``, ``dfn3_c2``, ``noisy``) for the unperturbed base cells.
- Exploratory: nothing here changes the frozen protocol or the adapter.

    python scripts/stress_eval.py --out runs/stress_v0
    python scripts/stress_eval.py --out runs/stress_v0 --stages paired
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sim_baseline as sb  # noqa: E402

STRESS_SEED = 20261107
E7_SEED = 20261028  # sim_baseline.py default: the E7 scenes and per-item noise draws
C8_FROZEN = "runs/c8_whisper_lora_stat_s1/best_wer.pt"
C8_SHA256 = "5063f02fa14cea47064721878c2e21cd8cd8594c642769cc69ae288029f52f90"
DFN3_C2 = "runs/c2_dfn3_s0/best.pt"
E7_GUARD = ROOT / "runs" / "sim_baseline" / "per_utt_guard.jsonl"
E7_LABEL = {"noisy": "noisy", "c8": "c8stat1", "dfn3_c2": "dfn3_c2"}


def _tilt(db: float):
    from uwse.uwsim.chain import tilt_gains

    f, g = tilt_gains(db)
    return {"hyd_eq": [f.tolist(), g.tolist()]}


def _ripple(scene) -> dict:
    from uwse.uwsim.chain import random_third_octave_gains

    f, g = random_third_octave_gains(np.random.default_rng([STRESS_SEED, int(scene.range_m * 10)]), 9.0)
    return {"proj_eq": [f.tolist(), g.tolist()]}


def _t60x2(scene) -> dict:
    return {"t60_bands": {k: 2.0 * v for k, v in scene.t60_bands.items()}, "rir_length_s": 2.0}


def _stationary(slope: float) -> dict:
    return {"stationary_on": True, "stationary_slope_db": slope, "stationary_rel_db": 0.0}


CHANNEL = {"tiltm6": _tilt(-6.0), "tiltp6": _tilt(6.0), "ripple9": _ripple, "t60x2": _t60x2,
           "clip": {"level_dbfs": -6.0}}
BASES = {"q": 30.0, "n0": 0.0}
CONDITIONS = {f"{b}_{k}": (snr, over) for b, snr in BASES.items() for k, over in CHANNEL.items()}
CONDITIONS.update({
    "n0_slopem9": (0.0, _stationary(-9.0)),
    "n0_slopep3": (0.0, _stationary(3.0)),
    "n0_tonal": (0.0, {"tonal_on": True, "tonal_f0": 120.0, "tonal_rel_db": 0.0}),
    "n5": (-5.0, {}),
})
BASE_OF = {"q": "quiet", "n0": "noise0", "n5": "noise0"}


def _sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _by_cell(path: Path) -> dict:
    by = collections.defaultdict(dict)
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            by[(r["cond"], r["system"])][r["utt_id"]] = r
    return by


def _delta(A: dict, B: dict, n_boot: int):
    from uwse.evaluation.stats import paired_bootstrap_wer

    ids = sorted(set(A) & set(B))
    if not ids:
        return None
    ea = np.array([A[i]["errors"] for i in ids], float)
    eb = np.array([B[i]["errors"] for i in ids], float)
    n = np.array([A[i]["n_ref"] for i in ids], float)
    return paired_bootstrap_wer(ea, eb, n, [A[i]["speaker"] for i in ids], n_boot=n_boot, seed=0)


def _wer(A: dict) -> float:
    return 100 * sum(r["errors"] for r in A.values()) / max(sum(r["n_ref"] for r in A.values()), 1)


def _fmt(r) -> str:
    if r is None:
        return "n/a"
    sig = "*" if (r.ci_high < 0 or r.ci_low > 0) else ""
    return f"{100 * r.estimate:+.1f} [{100 * r.ci_low:+.1f}, {100 * r.ci_high:+.1f}]{sig}"


def _verdict(rs: list) -> str:
    rs = [r for r in rs if r is not None]
    if not rs:
        return "n/a"
    if any(r.ci_low > 0 for r in rs):
        return "reverses"
    return "holds" if all(r.ci_high < 0 for r in rs) else "lost"


def stage_paired(out: Path, n_boot: int) -> str:
    """Per-cell WER and paired dWER vs C0, the E7 reference for the base cell, and the verdicts."""
    by, e7 = _by_cell(out / "per_utt_guard.jsonl"), _by_cell(E7_GUARD)
    ranges = sorted({float(c.rsplit("_", 1)[1][:-1]) for c, _ in by})
    lines = ["| condition | range | C0 | C8 | DFN3-C2 | C8 - C0 | DFN3-C2 - C0 | E7 base: C8 - C0 | E7 base: DFN3-C2 - C0 |",
             "|---|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for cname in CONDITIONS:
        base = BASE_OF[cname.split("_")[0]]
        per = {"c8": [], "dfn3_c2": []}
        for r in ranges:
            cell = f"{cname}_{r:g}m"
            if (cell, "noisy") not in by:
                continue
            ref = f"{base}_{r:g}m"
            row = [cname, f"{r:g} m"] + [f"{_wer(by[(cell, s)]):.1f}" if (cell, s) in by else "n/a"
                                         for s in ("noisy", "c8", "dfn3_c2")]
            ds, e7d = [], []
            for s in ("c8", "dfn3_c2"):
                d = _delta(by[(cell, s)], by[(cell, "noisy")], n_boot) if (cell, s) in by else None
                per[s].append(d)
                ds.append(_fmt(d))
                a, b = e7.get((ref, E7_LABEL[s])), e7.get((ref, "noisy"))
                e7d.append(_fmt(_delta(a, b, n_boot)) if a and b else "n/a")
            lines.append("| " + " | ".join(row + ds + e7d) + " |")
        if per["c8"]:
            verdicts.append(f"| {cname} | {_verdict(per['c8'])} | {_verdict(per['dfn3_c2'])} |")
    txt = ("WER % (guarded decode) and paired dWER vs C0, points [95% CI], speaker-cluster bootstrap "
           f"{n_boot} resamples; * = CI excludes 0.\n\n" + "\n".join(lines) +
           "\n\nVerdict vs C0 across ranges (holds = CI < 0 in every range; lost = some CI includes 0; "
           "reverses = some CI > 0):\n\n| condition | C8 | DFN3-C2 |\n|---|---|---|\n" + "\n".join(verdicts) + "\n")
    (out / "stress_paired.md").write_text(txt, encoding="utf-8")
    print(txt)
    return txt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / "stress_v0"))
    ap.add_argument("--librispeech", default=str(ROOT / "data" / "librispeech"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "uwsim_v0.yaml"))
    ap.add_argument("--n-utts", type=int, default=None)
    ap.add_argument("--conditions", default=None, help="comma list (default: all)")
    ap.add_argument("--ranges", default="1,3,6")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stages", default="sim,asr,score,paired")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    out = Path(args.out)
    sb._check_out(out)
    out.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    conds = CONDITIONS if not args.conditions else {c: CONDITIONS[c] for c in args.conditions.split(",")}
    c8 = ROOT / C8_FROZEN
    if stages & {"asr"} and _sha256(c8) != C8_SHA256:
        raise SystemExit(f"{c8} does not match the frozen adapter's SHA-256")
    if stages & {"sim", "asr", "score"}:
        import torch

        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        utts = sb.select_utts(Path(args.librispeech), out, args.n_utts)
        print(f"{len(utts)} utterances, {len(conds)} conditions, device {device}")
    if "sim" in stages:
        ranges = tuple(float(r) for r in args.ranges.split(","))
        scenes = sb.make_scenes(args.config, E7_SEED, ranges)
        (out / "conditions.json").write_text(json.dumps(
            {c: {"snr_db": snr, "over": over(scenes[ranges[0]]) if callable(over) else over}
             for c, (snr, over) in conds.items()}, indent=1, default=float))
        # DFN3-C2 (bubble-only) is already at 80-90% WER on E7 noise0, so it runs on the quiet base only
        for se, cs in ((["dfn3_c2"], {c: v for c, v in conds.items() if c.startswith("q_")}),
                       ([], {c: v for c, v in conds.items() if not c.startswith("q_")})):
            if cs:
                sb.stage_sim_enhance(utts, scenes, se, out, device, E7_SEED, {"dfn3_c2": "dfn3"},
                                     {"dfn3_c2": str(ROOT / DFN3_C2)}, conditions=cs)
    if "asr" in stages:
        sb.stage_asr(utts, out, "large-v3", device, {"c8": str(c8)}, decode="guard",
                     only_systems={"noisy", "c8", "dfn3_c2"})
    if "score" in stages:
        sb.stage_score(utts, out, "large-v3", args.n_boot, decode="guard")
    if "paired" in stages:
        stage_paired(out, args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
