"""Score enhanced utterances: per-utterance metrics + ASR, summary table with CIs.

Implements the scoring entry point of RESEARCH_PLAN.md §8 (metrics §8.1/8.2,
Whisper protocol and level normalization §8.3, statistics §8.5).

Manifest: JSONL, one row per (system, utterance) with keys

    utt_id, system, wav, ref_wav (R1), transcript, speaker, position, thruster
    optional: r0_wav (aligned R0 cut; VAD source, defaults to ref_wav),
              kind ("speech" | "noise", default "speech")

Every system must use the same cut of each utterance (§8.3). Outputs in
``--out-dir``: per_utt.jsonl, per_utt.csv, summary.csv, summary.md,
paired.csv (with ``--baseline``), transcripts.jsonl (ASR cache), run_info.json.

Example::

    python scripts/score.py --manifest bench/v0_dev_manifest.jsonl \
        --out-dir runs/score/v0_dev --baseline C0 --n-boot 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwse.audio import load, resample  # noqa: E402
from uwse.evaluation import asr as A  # noqa: E402
from uwse.evaluation import metrics as M  # noqa: E402
from uwse.evaluation import normalize as N  # noqa: E402
from uwse.evaluation import stats as S  # noqa: E402

REQUIRED_KEYS = ("utt_id", "system", "wav", "speaker", "position", "thruster")
METRIC_KEYS = ("estoi", "estoi_r0", "stoi", "pesq_wb_r1", "pesq_wb_r0", "si_sdr", "ci_sdr")


def _load16(path: str) -> np.ndarray:
    x, fs = load(path)
    if x.ndim == 2:
        x = x[0]
    return resample(x, fs, M.SCORE_FS) if fs != M.SCORE_FS else x


def read_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                # speaker/position/thruster define the bootstrap clusters and strata (§8.5);
                # a missing label would silently merge clusters and shrink the CIs
                missing = [k for k in REQUIRED_KEYS if r.get(k) in (None, "")]
                if missing:
                    raise ValueError(f"{path}:{lineno}: missing manifest keys {missing}")
                r.setdefault("kind", "speech")
                r["thruster"] = str(r.get("thruster"))
                r["position"] = str(r.get("position"))
                rows.append(r)
    keys = {(r["system"], r["utt_id"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate (system, utt_id) rows in manifest")
    return rows


def level_normalize(rows: list[dict], vad: N.VADConfig) -> list[dict]:
    """§8.3 gains: speech rows from the R0 VAD, noise rows from the median rule."""
    for r in tqdm(rows, desc="level normalization"):
        if r["kind"] != "speech":
            continue
        x = _load16(r["wav"])
        r0 = _load16(r.get("r0_wav") or r["ref_wav"])
        _, g, _ = N.normalize_cut(x, r0, M.SCORE_FS, vad=vad)
        r["gain"] = g
    return N.assign_gains(rows, group_keys=("system", "position", "thruster"))


def score_metrics(rows: list[dict], f_lo: float, f_hi: float) -> None:
    for r in tqdm(rows, desc="metrics"):
        if r["kind"] != "speech" or not r.get("ref_wav"):
            continue
        est = _load16(r["wav"])
        r1 = _load16(r["ref_wav"])
        r0 = _load16(r["r0_wav"]) if r.get("r0_wav") else None
        r.update(M.compute_metrics(est, r1, M.SCORE_FS, ref_r0=r0, f_lo=f_lo, f_hi=f_hi))


def score_asr(rows: list[dict], asr: A.ASRBackend, cache_path: str, model_key: str | None = None) -> None:
    cache = A.TranscriptCache(cache_path)
    recs = A.transcribe_items(asr, rows, cache, model_key=model_key)
    for r, rec in zip(rows, recs):
        r["hyp"] = rec["text"]
        r["hyp_compression_ratio"] = rec["compression_ratio"]
    refs = [r.get("transcript", "") for r in rows]
    hyps = [r["hyp"] for r in rows]
    counts = A.utterance_counts(refs, hyps)
    flags = A.hallucination_flags(refs, hyps, raw_hyps=hyps)
    for r, c, fl in zip(rows, counts, flags):
        r.update(sub=c.substitutions, dele=c.deletions, ins=c.insertions, hits=c.hits, n_ref=c.n_ref,
                 errors=c.errors, halluc_flag=bool(fl))
        r["n_hyp_words"] = c.hits + c.substitutions + c.insertions


def _summ_group(rs: list[dict], n_boot: int, seed: int, have_asr: bool) -> dict:
    sp = [r for r in rs if r["kind"] == "speech"]
    out: dict = {"n_utts": len(sp)}
    if not sp:
        return out
    cl = [r.get("speaker") for r in sp]
    st = [f'{r["position"]}|{r["thruster"]}' for r in sp]
    kw = dict(n_boot=n_boot, seed=seed)
    if have_asr:
        e = np.array([r["errors"] for r in sp], float)
        n = np.array([r["n_ref"] for r in sp], float)
        flag = np.array([r["halluc_flag"] for r in sp])
        b = S.cluster_bootstrap_wer(e, n, cl, st, **kw)
        out.update(wer=b.estimate, wer_lo=b.ci_low, wer_hi=b.ci_high)
        for k in ("sub", "dele", "ins"):
            out[k] = int(sum(r[k] for r in sp))
        out["n_ref_words"] = int(n.sum())
        out["halluc_rate"] = float(flag.mean())
        keep = ~flag
        if keep.any():
            b2 = S.cluster_bootstrap_wer(e[keep], n[keep], np.asarray(cl, object)[keep], np.asarray(st, object)[keep], **kw)
            out.update(wer_unflagged=b2.estimate, wer_unflagged_lo=b2.ci_low, wer_unflagged_hi=b2.ci_high)
        noise = [r for r in rs if r["kind"] != "speech"]
        if noise:
            out["noise_n"] = len(noise)
            out["noise_insertions"] = int(sum(r["ins"] for r in noise))
            out["noise_nonempty_rate"] = float(np.mean([r["n_hyp_words"] > 0 for r in noise]))
    for m in METRIC_KEYS:
        v = np.array([r.get(m, np.nan) for r in sp], float)
        if np.isfinite(v).any():
            b = S.cluster_bootstrap_mean(v, cl, st, **kw)
            out.update({m: b.estimate, f"{m}_lo": b.ci_low, f"{m}_hi": b.ci_high})
    return out


def summarize(rows: list[dict], n_boot: int, seed: int, have_asr: bool) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["system"], "ALL", "ALL")].append(r)
        groups[(r["system"], r["position"], r["thruster"])].append(r)
    table = []
    for (sysname, pos, thr), rs in tqdm(sorted(groups.items()), desc="summary"):
        table.append({"system": sysname, "position": pos, "thruster": thr, **_summ_group(rs, n_boot, seed, have_asr)})
    return table


def paired(rows: list[dict], baseline: str, n_boot: int, seed: int, have_asr: bool) -> list[dict]:
    if baseline not in {r["system"] for r in rows}:
        raise ValueError(f"baseline system {baseline!r} not in manifest")
    by = {(r["system"], r["utt_id"]): r for r in rows if r["kind"] == "speech"}
    systems = sorted({r["system"] for r in rows} - {baseline})
    out = []
    for s in systems:
        uids = sorted(u for (sy, u) in by if sy == s and (baseline, u) in by)
        if not uids:
            continue
        a = [by[(s, u)] for u in uids]
        b = [by[(baseline, u)] for u in uids]
        cl = [r.get("speaker") for r in a]
        st = [f'{r["position"]}|{r["thruster"]}' for r in a]
        row = {"system": s, "baseline": baseline, "n_utts": len(uids)}
        if have_asr:
            ea = np.array([r["errors"] for r in a], float)
            eb = np.array([r["errors"] for r in b], float)
            n = np.array([r["n_ref"] for r in a], float)
            d = S.paired_bootstrap_wer(ea, eb, n, cl, st, n_boot=n_boot, seed=seed)
            row.update(d_wer=d.estimate, d_wer_lo=d.ci_low, d_wer_hi=d.ci_high, d_wer_p_boot=float(
                min(1.0, 2 * min(np.mean(d.replicates <= 0), np.mean(d.replicates >= 0)))))
            seg_a, seg_b = [], []
            for ra, rb in zip(a, b):
                ref = A.normalize_text(ra.get("transcript", ""))
                ea_w, ia = A.word_alignment(ref, A.normalize_text(ra["hyp"]))
                eb_w, ib = A.word_alignment(ref, A.normalize_text(rb["hyp"]))
                sa, sb = S.mapsswe_segments(ea_w, ia, eb_w, ib)
                seg_a.extend(sa)
                seg_b.extend(sb)
            mp = S.mapsswe(seg_a, seg_b)
            row.update(mapsswe_w=mp["w"], mapsswe_p=mp["p"], mapsswe_segments=mp["n_segments"])
        for m in ("estoi", "pesq_wb_r1", "ci_sdr"):
            va = np.array([r.get(m, np.nan) for r in a], float)
            vb = np.array([r.get(m, np.nan) for r in b], float)
            if np.isfinite(va).any() and np.isfinite(vb).any():
                d = S.paired_bootstrap_mean(va, vb, cl, st, n_boot=n_boot, seed=seed)
                row.update({f"d_{m}": d.estimate, f"d_{m}_lo": d.ci_low, f"d_{m}_hi": d.ci_high})
        out.append(row)
    if have_asr and out:
        p = [r.get("d_wer_p_boot", np.nan) for r in out]
        if np.all(np.isfinite(p)):
            rej, adj = S.holm(p)
            for r, a_, j in zip(out, adj, rej):
                r["d_wer_p_holm"] = float(a_)
                r["d_wer_reject_holm"] = bool(j)
    return out


def _write_csv(path: str, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fmt(v, lo=None, hi=None, pct=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    s = 100.0 if pct else 1.0
    if lo is None:
        return f"{v * s:.2f}"
    return f"{v * s:.2f} [{lo * s:.2f}, {hi * s:.2f}]"


def write_markdown(path: str, table: list[dict], have_asr: bool) -> None:
    cols = ["system", "position", "thruster", "n_utts"]
    head = cols + (["WER % [95% CI]", "S/D/I", "WER unflagged %", "halluc %"] if have_asr else []) + [
        "ESTOI", "STOI", "PESQ-WB(R1)", "CI-SDR dB"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in table:
        cells = [str(r.get(c, "")) for c in cols]
        if have_asr:
            cells += [
                _fmt(r.get("wer"), r.get("wer_lo"), r.get("wer_hi"), pct=True),
                f'{r.get("sub", "-")}/{r.get("dele", "-")}/{r.get("ins", "-")}',
                _fmt(r.get("wer_unflagged"), r.get("wer_unflagged_lo"), r.get("wer_unflagged_hi"), pct=True),
                _fmt(r.get("halluc_rate"), pct=True),
            ]
        for m in ("estoi", "stoi", "pesq_wb_r1", "ci_sdr"):
            cells.append(_fmt(r.get(m), r.get(f"{m}_lo"), r.get(f"{m}_hi")))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--asr-model", default="large-v3", help="openai-whisper model name")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--level-normalize", action="store_true",
                    help="apply §8.3 R0-VAD RMS normalization (-26 dBFS) before ASR")
    ap.add_argument("--f-lo", type=float, default=M.DEFAULT_F_LO, help="scoring band low edge (Hz)")
    ap.add_argument("--f-hi", type=float, default=M.DEFAULT_F_HI)
    ap.add_argument("--n-boot", type=int, default=S.DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default=None, help="system for paired differences")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_manifest(args.manifest)
    info: dict = {"args": vars(args), "n_rows": len(rows)}
    if args.level_normalize:
        rows = level_normalize(rows, N.VADConfig())
        info["vad"] = vars(N.VADConfig())
        info["target_dbfs"] = N.TARGET_DBFS
    if not args.no_metrics:
        score_metrics(rows, args.f_lo, args.f_hi)
    have_asr = not args.no_asr
    if have_asr:
        asr = A.WhisperASR(args.asr_model, device=args.device)
        # the cache key's model field also records whether level normalization was applied
        key = asr.name + ("+lnorm" if args.level_normalize else "")
        score_asr(rows, asr, os.path.join(args.out_dir, "transcripts.jsonl"), key)
        info["asr"] = asr.info()

    with open(os.path.join(args.out_dir, "per_utt.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: _jsonable(v) for k, v in r.items()}, ensure_ascii=False) + "\n")
    _write_csv(os.path.join(args.out_dir, "per_utt.csv"), rows)

    table = summarize(rows, args.n_boot, args.seed, have_asr)
    _write_csv(os.path.join(args.out_dir, "summary.csv"), table)
    write_markdown(os.path.join(args.out_dir, "summary.md"), table, have_asr)
    if args.baseline:
        pr = paired(rows, args.baseline, args.n_boot, args.seed, have_asr)
        _write_csv(os.path.join(args.out_dir, "paired.csv"), pr)
    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    print(open(os.path.join(args.out_dir, "summary.md"), encoding="utf-8").read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
