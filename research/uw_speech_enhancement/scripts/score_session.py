"""One command after a pool session: score C0, C8 and the enhancers per condition and distance.

Runs the existing scorer (``scripts/score.py``, in-process via ``score.main``) with the frozen
settings of RESULTS_E7_SIM_DRYRUN.md: ``--level-normalize`` (decode key
``whisper-large-v3+lnorm+guard``), C8 decoded with ``--asr-lora C8=<frozen ckpt>`` (sha256
checked before any decode), corpus WER CIs from ``--n-boot 10000`` and paired deltas from a
separate ``--n-boot 2000`` run (the 2026-11-10 bootstrap-count amendment). WER, flags and the
speaker-cluster bootstrap are score.py's; nothing is reimplemented.

Conditions:

- ``--condition NAME=MANIFEST`` (repeatable): a score.py manifest from
  scripts/make_score_manifest.py, e.g. ``quiet=data/real/v1_s2/score/manifest.jsonl`` and, for
  the live bubble pass, ``bubblelive=data/real/v1_s2/bubble/score/manifest.jsonl``.
- ``--added-noise-dir DIR``: scripts/mix_real_noise.py output; every ``DIR/<cond>/manifest.jsonl``
  (``pool0dB``, ``pool5dB``, ``bubble0dB``, ...) becomes a condition.

Each condition keeps its C0 rows, adds C8 rows (the same cut) if missing, and adds enhancer rows
either from ``--se-manifest COND=PATH`` (enhance_cuts.py output) or by running
``scripts/enhance_cuts.py --input-level sim`` on it with ``--enhance C1,C2,C10`` (long; GPU).
Rows can be restricted to bench splits (``--splits``, joined from ``--align`` by utt_id, since
make_score_manifest.py does not carry ``bench_split``).

    python scripts/score_session.py --out runs/score/v1_s2 \
        --condition quiet=data/real/v1_s2/score/manifest.jsonl \
        --added-noise-dir data/real/v1_s2/added --enhance C1,C2,C10 \
        --align data/real/v1_s2/align.jsonl --speaker-sex data/librispeech/LibriSpeech/SPEAKERS.TXT \
        --pilot data/real/v1_s2/pilot/pilot_wer.json --chainref data/real/v1_s2/chainref/chainref_wer.json \
        --device cuda

Writes under ``--out``:

- ``<cond>/manifest.jsonl``          the scored manifest (C0, C8, enhancers)
- ``<cond>/all/``                    score.py output over every distance, 10000 resamples
                                     (``summary.csv``: WER + CI per system and position)
- ``<cond>/paired_<subset>/``        score.py ``--baseline C0`` output, 2000 resamples, for
                                     ``ALL`` and each position (``paired.csv``: dWER + CI);
                                     the transcript cache of ``all/`` is reused, no re-decode
- ``index.json``                     conditions (each with its positions), the main distance series
                                     (positions of ``quiet``), distances (m), pilot / chain-reference
                                     records, the C8 checkpoint hash: read by make_paper_numbers.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

C8_CKPT = ROOT / "runs" / "c8_whisper_lora_stat_s1" / "best_wer.pt"
C8_SHA_PREFIX = "5063f02f"
N_BOOT_CI = 10000
N_BOOT_PAIRED = 2000
BASE_SYSTEMS = ("C0", "C8")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"uwse_{name}_cli", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCORE = _load("score")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def distance_m(position: str, overrides: dict[str, float]) -> float:
    """Distance of a position id: ``--distance POS=M`` or the number in the id (p3 -> 3, p0p5 -> 0.5,
    p3bub -> 3)."""
    if position in overrides:
        return overrides[position]
    m = re.fullmatch(r"p?(\d+)(?:[.p](\d+))?[A-Za-z_]*", position)
    if not m:
        raise SystemExit(f"cannot read a distance from position {position!r}; pass --distance {position}=M")
    return float(m.group(1) + ("." + m.group(2) if m.group(2) else ""))


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def base_rows(rows: list[dict], splits: set[str] | None, split_of: dict[str, str]) -> list[dict]:
    """C0 rows plus C8 rows (copies of C0 where the manifest has none); other systems dropped."""
    c0 = [r for r in rows if r["system"] == "C0"]
    if splits is not None:
        def split(r):
            return r.get("bench_split") or split_of.get(r["utt_id"])
        missing = [r["utt_id"] for r in c0 if split(r) is None]
        if missing:
            raise SystemExit(f"no bench_split for {len(missing)} rows (e.g. {missing[0]}); pass --align")
        c0 = [r for r in c0 if split(r) in splits]
    keep = {r["utt_id"] for r in c0}
    c8 = [r for r in rows if r["system"] == "C8" and r["utt_id"] in keep]
    if not c8:
        c8 = [{**r, "system": "C8"} for r in c0]
    elif {r["utt_id"] for r in c8} != keep:
        raise SystemExit("C8 rows do not cover the same cuts as C0")
    return c0 + c8


def run_enhance(c0_manifest: Path, out_dir: Path, systems: str, device: str) -> Path:
    """scripts/enhance_cuts.py --input-level sim (frozen C2 input level); returns its manifest."""
    _load("enhance_cuts").main(["--manifest", str(c0_manifest), "--out-dir", str(out_dir),
                                "--input-level", "sim", "--systems", systems, "--device", device])
    return out_dir / "manifest.jsonl"


def se_rows(path: Path, keep: set[str]) -> list[dict]:
    rows = [r for r in read_jsonl(path) if r["system"] not in BASE_SYSTEMS]
    by: dict[str, set[str]] = {}
    for r in rows:
        by.setdefault(r["system"], set()).add(r["utt_id"])
    bad = [s for s, u in by.items() if not keep <= u]
    if bad:
        raise SystemExit(f"{path}: systems {bad} do not cover every scored cut")
    return [r for r in rows if r["utt_id"] in keep]


def score_args(manifest: Path, out: Path, a, n_boot: int, baseline: bool) -> list[str]:
    argv = ["--manifest", str(manifest), "--out-dir", str(out), "--level-normalize",
            "--asr-lora", f"C8={a.c8_ckpt}", "--n-boot", str(n_boot), "--seed", str(a.seed),
            "--asr-model", a.asr_model]
    if baseline:
        argv += ["--baseline", "C0"]
    if a.device:
        argv += ["--device", a.device]
    if a.speaker_sex:
        argv += ["--speaker-sex", str(a.speaker_sex)]
    if a.no_metrics:
        argv += ["--no-metrics"]
    return argv


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--condition", action="append", default=[], metavar="NAME=MANIFEST")
    ap.add_argument("--added-noise-dir", type=Path, default=None, help="mix_real_noise.py --out")
    ap.add_argument("--enhance", default="", help="comma list of enhance_cuts.py systems to run (e.g. C1,C2,C10)")
    ap.add_argument("--se-manifest", action="append", default=[], metavar="COND=PATH",
                    help="existing enhance_cuts.py manifest for a condition (instead of --enhance)")
    ap.add_argument("--align", action="append", type=Path, default=[], help="align.jsonl to join bench_split")
    ap.add_argument("--splits", default=None, help="comma list of bench splits to keep (e.g. v1-test)")
    ap.add_argument("--distance", action="append", default=[], metavar="POS=M")
    ap.add_argument("--pilot", type=Path, default=None, help="pilot_wer.py json (role pilot)")
    ap.add_argument("--chainref", type=Path, default=None, help="pilot_wer.py json (role chainref)")
    ap.add_argument("--c8-ckpt", type=Path, default=C8_CKPT)
    ap.add_argument("--c8-sha256-prefix", default=C8_SHA_PREFIX)
    ap.add_argument("--speaker-sex", type=Path, default=None, metavar="SPEAKERS.TXT")
    ap.add_argument("--device", default=None)
    ap.add_argument("--asr-model", default="large-v3")
    ap.add_argument("--n-boot-ci", type=int, default=N_BOOT_CI)
    ap.add_argument("--n-boot-paired", type=int, default=N_BOOT_PAIRED)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-metrics", action="store_true", help="skip intrusive metrics (no R1 cuts)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    conds: dict[str, Path] = {}
    added: set[str] = set()
    for kv in a.condition:
        name, _, p = kv.partition("=")
        if not p or not re.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]*", name) or name in conds:
            raise SystemExit(f"--condition expects a unique NAME=MANIFEST, got {kv!r}")
        conds[name] = Path(p)
    if a.added_noise_dir:
        for m in sorted(a.added_noise_dir.glob("*/manifest.jsonl")):
            if m.parent.name in conds:
                raise SystemExit(f"condition {m.parent.name} given twice")
            conds[m.parent.name] = m
            added.add(m.parent.name)
    if not conds:
        raise SystemExit("no conditions: pass --condition and/or --added-noise-dir")
    se_given = dict(kv.split("=", 1) for kv in a.se_manifest)
    if set(se_given) - set(conds):
        raise SystemExit(f"--se-manifest for unknown conditions {sorted(set(se_given) - set(conds))}")
    overrides = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.distance)}
    splits = set(a.splits.split(",")) if a.splits else None
    split_of = {r["utt_id"]: r.get("bench_split") for p in a.align for r in read_jsonl(p)}

    sha = sha256_file(a.c8_ckpt)
    if not sha.startswith(a.c8_sha256_prefix):
        raise SystemExit(f"C8 checkpoint {a.c8_ckpt} sha256 {sha[:8]} != frozen {a.c8_sha256_prefix}")
    a.out.mkdir(parents=True, exist_ok=True)

    index = {"conditions": {}, "c8_ckpt": str(a.c8_ckpt), "c8_sha256": sha,
             "n_boot_ci": a.n_boot_ci, "n_boot_paired": a.n_boot_paired, "seed": a.seed,
             "asr_key": f"whisper-{a.asr_model}+lnorm+guard", "splits": sorted(splits) if splits else None}
    positions_all: set[str] = set()
    for cond, src in conds.items():
        d = a.out / cond
        rows = base_rows(read_jsonl(src), splits, split_of)
        keep = {r["utt_id"] for r in rows}
        if cond in se_given:
            rows += se_rows(Path(se_given[cond]), keep)
        elif a.enhance:
            c0m = d / "c0_manifest.jsonl"
            write_jsonl(c0m, [r for r in rows if r["system"] == "C0"])
            rows += se_rows(run_enhance(c0m, d / "enhance", a.enhance, a.device or "auto"), keep)
        man = d / "manifest.jsonl"
        write_jsonl(man, rows)
        print(f"[{cond}] {len(keep)} cuts, systems {sorted({r['system'] for r in rows})}")
        SCORE.main(score_args(man, d / "all", a, a.n_boot_ci, baseline=False))
        positions = sorted({str(r["position"]) for r in rows})
        positions_all |= set(positions)
        subsets = {"ALL": rows, **{p: [r for r in rows if str(r["position"]) == p] for p in positions}}
        for name, sub in subsets.items():
            pd = d / f"paired_{name}"
            pd.mkdir(parents=True, exist_ok=True)
            pm = pd / "manifest.jsonl"
            write_jsonl(pm, sub)
            shutil.copyfile(d / "all" / "transcripts.jsonl", pd / "transcripts.jsonl")  # reuse decodes
            SCORE.main(score_args(pm, pd, a, a.n_boot_paired, baseline=True))
        index["conditions"][cond] = {
            "source_manifest": str(src), "added_noise": cond in added,
            "positions": positions, "systems": sorted({r["system"] for r in rows}), "n_cuts": len(keep)}
    main_cond = "quiet" if "quiet" in index["conditions"] else next(iter(index["conditions"]))
    index["main_condition"] = main_cond
    # the distance series of the paper: the main (quiet) condition's positions; others (e.g. p3bub) per condition
    index["positions"] = sorted(index["conditions"][main_cond]["positions"], key=lambda p: distance_m(p, overrides))
    index["distances_m"] = {p: distance_m(p, overrides) for p in sorted(positions_all)}
    for key, p in (("pilot", a.pilot), ("chainref", a.chainref)):
        if p:
            shutil.copyfile(p, a.out / f"{key}.json")
            index[key] = f"{key}.json"
    (a.out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"-> {a.out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
