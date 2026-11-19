"""Select a C8 (Whisper LoRA) checkpoint by WER on a uwsim dev set, with guarded decoding.

Validation cross-entropy ranked the C8 seeds opposite to their WER (RESULTS_E7_SIM_DRYRUN.md,
"C8 seed 1"), so runs with ``train.keep_every`` keep step checkpoints and this script picks
one by dev WER instead.

The dev set is disjoint from the E7 test sets:

- Utterances: dev-clean (E7 uses test-clean), ``--n-utts`` chosen round-robin over speakers.
- Scenes: a different boundary-prior seed, ranges 1.5 / 3.5 / 5.5 m, 1.5 m depth, off the
  pool's long centreline (E7: 1 / 3 / 6 m, 1 m depth, centreline). Each utterance is
  rendered at one range (round-robin), in every condition. The script refuses to run if any
  dev scene matches an E7 scene.
- Conditions: quiet (30 dB), bubbles0 (as in E7), and stationary noise at 0 dB SNR with
  slopes -5 and -1 dB/oct. Those slopes are inside the training band; the held-out band
  (-4, -2) dB/oct around the noise0 test slope (-3) is never used, so selection does not
  see the test noise.

Decoding is `DecodeConfig.guarded()` throughout, after the same RMS-active level
normalization as E7. The selection score is the mean of the per-condition corpus WERs, so
the quiet cell counts as much as the noisy ones.

Usage::

    python scripts/dev_wer_select.py --run runs/c8_whisper_lora_stat_s1

Writes ``<dev-dir>/`` (rendered audio, shared by runs), and under ``<run>/dev_wer/``
``transcripts.jsonl`` (cache), ``summary.json``, ``summary.md``, ``selection.json``; then
copies the chosen checkpoint to ``<run>/best_wer.pt``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sim_baseline import FS_SCORE, FS_SIM, make_scenes  # noqa: E402

from uwse import audio  # noqa: E402

DEV_RANGES_M = (1.5, 3.5, 5.5)
DEV_DEPTH_M = 1.5
DEV_Y_FRAC = 0.35
# name -> (SNR dB, scene overrides)
DEV_CONDITIONS = {
    "quiet": (30.0, {}),
    "bubbles0": (0.0, {"bubble_rate": 10.0, "bubble_rel_db": 0.0}),
    "stat_m5": (0.0, {"stationary_on": True, "stationary_slope_db": -5.0, "stationary_rel_db": 0.0}),
    "stat_m1": (0.0, {"stationary_on": True, "stationary_slope_db": -1.0, "stationary_rel_db": 0.0}),
}
HELDOUT_SLOPE = (-4.0, -2.0)


def asr_key(model: str) -> str:
    return f"whisper-{model}+lnorm+guard"


def _check_disjoint(utts, scenes, e7_dir: Path) -> None:
    """Refuse to run if the dev set shares an utterance or a scene with E7."""
    for name, (_, over) in DEV_CONDITIONS.items():
        sl = over.get("stationary_slope_db")
        if sl is not None and HELDOUT_SLOPE[0] < sl < HELDOUT_SLOPE[1]:
            raise SystemExit(f"dev condition {name} uses a held-out slope ({sl} dB/oct)")
    test_ids = set()
    if (e7_dir / "v0_test.jsonl").exists():
        with open(e7_dir / "v0_test.jsonl", encoding="utf-8") as f:
            test_ids = {json.loads(line)["utt_id"] for line in f if line.strip()}
    shared = test_ids & {u.utt_id for u in utts}
    if shared:
        raise SystemExit(f"dev set shares {len(shared)} utterances with E7, e.g. {sorted(shared)[:3]}")
    if (e7_dir / "scenes.json").exists():
        e7 = json.loads((e7_dir / "scenes.json").read_text())
        e7_geo = {(tuple(np.round(s["src"], 3)), tuple(np.round(s["rx"], 3))) for s in e7.values()}
        for r, s in scenes.items():
            if (tuple(np.round(s.src, 3)), tuple(np.round(s.rx, 3))) in e7_geo:
                raise SystemExit(f"dev scene at {r} m has the same geometry as an E7 scene")


def render(utts, scenes, dev: Path, seed: int) -> list[dict]:
    """Render noisy + R0 (16 kHz) per utterance and condition; cached on disk."""
    from uwse.uwsim.simulate import render_rir, simulate

    rirs = {r: render_rir(s) for r, s in scenes.items()}
    items = []
    for k, u in enumerate(tqdm(utts, desc="render dev")):
        r = DEV_RANGES_M[k % len(DEV_RANGES_M)]
        s16 = None
        for ci, (cname, (snr, over)) in enumerate(DEV_CONDITIONS.items()):
            pn, p0 = (dev / "wav" / cname / d / f"{u.utt_id}.wav" for d in ("noisy", "R0"))
            items.append({"cond": cname, "utt_id": u.utt_id, "speaker": u.speaker, "range_m": r,
                          "ref": u.transcript, "noisy": str(pn), "R0": str(p0)})
            if pn.exists() and p0.exists():
                continue
            if s16 is None:
                s16, fs = sf.read(u.path, dtype="float64")
                s48 = audio.resample(s16, fs, FS_SIM)
            scene = dataclasses.replace(scenes[r], snr_db=snr, **over)
            sim = simulate(s48, scene, None, np.random.default_rng([seed, k, ci]), fs_model=FS_SIM, rir=rirs[r])
            for p, key in ((pn, "noisy"), (p0, "target_R0")):
                p.parent.mkdir(parents=True, exist_ok=True)
                y = audio.resample(sim[key].astype(np.float64), FS_SIM, FS_SCORE)
                sf.write(p, y.astype(np.float32), FS_SCORE, subtype="FLOAT")
    return items


def find_ckpts(run: Path) -> dict[str, Path | None]:
    """label -> checkpoint: the base model, every kept step checkpoint, and CE-best."""
    out: dict[str, Path | None] = {"base": None}
    for p in sorted(run.glob("ckpt_step*.pt")):
        out["step" + str(int(p.stem.removeprefix("ckpt_step")))] = p
    if (run / "best.pt").exists():
        out["best_ce"] = run / "best.pt"
    return out


def decode(items, ckpts: dict[str, Path | None], cache_path: Path, device: str, model: str) -> None:
    from uwse.evaluation.asr import DecodeConfig, TranscriptCache, WhisperASR
    from uwse.evaluation.normalize import normalize_cut

    cache = TranscriptCache(str(cache_path))
    key = asr_key(model)
    for label, ck in ckpts.items():
        todo = [it for it in items if cache.get(f"{it['cond']}/{label}", it["utt_id"], key) is None]
        if not todo:
            continue
        asr = WhisperASR(model, device=device, decode=DecodeConfig.guarded(),
                         lora_ckpt=None if ck is None else str(ck))
        for it in tqdm(todo, desc=f"ASR {label}"):
            x, _ = sf.read(it["noisy"])
            r0, _ = sf.read(it["R0"])
            y, _, _ = normalize_cut(x, r0, FS_SCORE)
            res = asr.transcribe(np.clip(y, -1, 1).astype(np.float32))
            cache.put({"system": f"{it['cond']}/{label}", "utt_id": it["utt_id"], "model": key,
                       "ckpt": None if ck is None else str(ck), **res})
        del asr
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


def score(items, ckpts: dict[str, Path | None], cache_path: Path, model: str) -> list[dict]:
    from uwse.evaluation.asr import TranscriptCache, utterance_counts

    cache = TranscriptCache(str(cache_path))
    rows = []
    for label, ck in ckpts.items():
        row = {"label": label, "ckpt": None if ck is None else str(ck), "cond": {}}
        for cname in DEV_CONDITIONS:
            its = [it for it in items if it["cond"] == cname]
            hyps = [cache.get(f"{cname}/{label}", it["utt_id"], asr_key(model)) for it in its]
            if any(h is None for h in hyps):
                raise SystemExit(f"missing transcripts for {cname}/{label}")
            cnt = utterance_counts([it["ref"] for it in its], [h["text"] for h in hyps])
            err, nref = sum(c.errors for c in cnt), sum(c.n_ref for c in cnt)
            row["cond"][cname] = {"wer": err / nref, "n": len(its),
                                  "n_loops": int(sum(c.insertions >= max(10, 2 * c.n_ref) for c in cnt))}
        row["mean_wer"] = float(np.mean([c["wer"] for c in row["cond"].values()]))
        row["n_loops"] = sum(c["n_loops"] for c in row["cond"].values())
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="training run dir with ckpt_step*.pt")
    ap.add_argument("--librispeech", default=str(ROOT / "data" / "librispeech"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "uwsim_v0.yaml"),
                    help="uwsim config for the scenes' boundary priors (E7 uses uwsim_v0.yaml)")
    ap.add_argument("--dev-dir", default=str(ROOT / "data" / "dev_wer_v0"))
    ap.add_argument("--e7-dir", default=str(ROOT / "runs" / "sim_baseline"), help="E7 run, for the disjointness check")
    ap.add_argument("--n-utts", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--asr-model", default="large-v3", help="tiny for smoke tests only")
    ap.add_argument("--stages", default="render,decode,score")
    args = ap.parse_args(argv)

    import torch

    from uwse.bench.librispeech import build_manifest, find_subset_dir, select_v0_dev

    run, dev = Path(args.run), Path(args.dev_dir)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    stages = set(args.stages.split(","))
    man = build_manifest(find_subset_dir(args.librispeech, "dev-clean"), "dev-clean")
    utts = select_v0_dev(man, n=args.n_utts)
    scenes = make_scenes(args.config, args.seed, ranges=DEV_RANGES_M, depth_m=DEV_DEPTH_M, y_frac=DEV_Y_FRAC)
    _check_disjoint(utts, scenes, Path(args.e7_dir))
    dev.mkdir(parents=True, exist_ok=True)
    (dev / "scenes.json").write_text(json.dumps({f"{r:g}m": s.to_dict() for r, s in scenes.items()},
                                                indent=1, default=float))
    items = render(utts, scenes, dev, args.seed)
    print(f"dev set: {len(utts)} utterances x {len(DEV_CONDITIONS)} conditions = {len(items)} items")
    if stages == {"render"}:
        return 0

    out = run / "dev_wer"
    out.mkdir(parents=True, exist_ok=True)
    ckpts = find_ckpts(run)
    print("checkpoints:", ", ".join(ckpts))
    if "decode" in stages:
        decode(items, ckpts, out / "transcripts.jsonl", device, args.asr_model)
    if "score" not in stages:
        return 0
    rows = score(items, ckpts, out / "transcripts.jsonl", args.asr_model)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    lines = ["| checkpoint | " + " | ".join(DEV_CONDITIONS) + " | mean WER | loops |",
             "|---|" + "---|" * (len(DEV_CONDITIONS) + 2)]
    for r in rows:
        lines.append(f"| {r['label']} | " + " | ".join(f"{100 * c['wer']:.1f}" for c in r["cond"].values())
                     + f" | {100 * r['mean_wer']:.2f} | {r['n_loops']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    cand = [r for r in rows if r["ckpt"] is not None]
    if not cand:
        print("no checkpoints to select from")
        return 0
    best = min(cand, key=lambda r: r["mean_wer"])
    sel = {"selected": best["label"], "ckpt": best["ckpt"], "mean_wer": best["mean_wer"],
           "criterion": "min mean per-condition corpus WER, guarded decode",
           "dev_dir": str(dev), "n_utts": len(utts), "conditions": list(DEV_CONDITIONS)}
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    shutil.copyfile(best["ckpt"], run / "best_wer.pt")
    print(f"selected {best['label']} (mean dev WER {100 * best['mean_wer']:.2f}%) -> {run / 'best_wer.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
