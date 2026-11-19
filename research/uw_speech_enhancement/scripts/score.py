"""Score enhanced utterances: per-utterance metrics + ASR, summary table with CIs.

Implements the scoring entry point of RESEARCH_PLAN.md §8 (metrics §8.1/8.2,
Whisper protocol and level normalization §8.3, statistics §8.5).

Manifest: JSONL, one row per (system, utterance) with keys

    utt_id, system, wav, ref_wav (R1), transcript, speaker, position, thruster
    optional: r0_wav (aligned R0 cut; VAD source, defaults to ref_wav),
              kind ("speech" | "noise", default "speech"),
              sex ("F" | "M"; or filled from ``--speaker-sex`` SPEAKERS.TXT),
              snr_db_500_8000, snr_db_50_8000 (measured upstream, or joined by utt_id
              from ``--snr`` = scripts/measure_snr.py's snr.jsonl), or
              snr_speech_wav + snr_noise_wav (speech-only and noise-only recordings
              through the same chain) from which both SNRs are measured here

Every system must use the same cut of each utterance (§8.3). Decoding follows the
frozen protocol (RESULTS_E7_SIM_DRYRUN.md): guarded decode by default, and C8-style
systems (the unprocessed audio decoded by a LoRA-adapted Whisper) via ``--asr-lora``.
Hallucination flags are the §8.3 flag and the substitution flag (signed off 2026-11-08);
"WER unflagged" is computed on one paired subset that drops an utterance flagged for any
system in the manifest. SNR is reported over both 500-8000 Hz and 50-8000 Hz, and WER
by speaker sex is read against both (reporting-only addition, 2026-11-08). Outputs in
``--out-dir``: per_utt.jsonl, per_utt.csv, summary.csv, summary.md,
paired.csv (with ``--baseline``), by_sex.csv/by_sex.md (with sex labels and ASR),
transcripts.jsonl (ASR cache), run_info.json.

Example::

    python scripts/score.py --manifest bench/v0_dev_manifest.jsonl \
        --out-dir runs/score/v0_dev --baseline C0 --n-boot 10000 \
        --asr-lora C8=runs/c8_whisper_lora_stat_s1/best_wer.pt
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
from uwse.uwsim.noise import band_snr_db, fit_length  # noqa: E402

REQUIRED_KEYS = ("utt_id", "system", "wav", "speaker", "position", "thruster")
METRIC_KEYS = ("estoi", "estoi_r0", "stoi", "pesq_wb_r1", "pesq_wb_r0", "si_sdr", "ci_sdr")
# speech-band SNR over speech-active samples, same definition as the simulator's mix_at_snr;
# both bands are reported because the 500 Hz edge favours talkers with energy below it
SNR_BANDS = {"snr_db_500_8000": (500.0, 8000.0), "snr_db_50_8000": (50.0, 8000.0)}


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
                if r.get("sex") not in (None, ""):
                    r["sex"] = str(r["sex"]).strip().upper()[:1]
                    if r["sex"] not in ("F", "M"):
                        raise ValueError(f"{path}:{lineno}: sex must be F or M, got {r['sex']!r}")
                r["thruster"] = str(r.get("thruster"))
                r["position"] = str(r.get("position"))
                rows.append(r)
    keys = {(r["system"], r["utt_id"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate (system, utt_id) rows in manifest")
    return rows


def read_speaker_sex(path: str) -> dict[str, str]:
    """LibriSpeech SPEAKERS.TXT -> {speaker_id: "F" | "M"}."""
    out = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(";"):
                continue
            fields = [x.strip() for x in line.split("|")]
            if len(fields) >= 2 and fields[1] in ("F", "M"):
                out[fields[0]] = fields[1]
    return out


def assign_sex(rows: list[dict], speaker_sex: dict[str, str] | None = None) -> None:
    """Fill missing ``sex`` from ``speaker_sex``; a speaker must have one sex throughout."""
    seen: dict[str, str] = {}
    for r in rows:
        if not r.get("sex") and speaker_sex:
            sx = speaker_sex.get(str(r["speaker"]))
            if sx:
                r["sex"] = sx
        if r.get("sex"):
            if seen.setdefault(str(r["speaker"]), r["sex"]) != r["sex"]:
                raise ValueError(f"speaker {r['speaker']!r} labelled both F and M")


def join_snr(rows: list[dict], path: str) -> int:
    """Fill both-band SNRs (and the snr_level_drift flag) from measure_snr.py's snr.jsonl."""
    with open(path, encoding="utf-8") as f:
        by = {r["utt_id"]: r for r in map(json.loads, filter(str.strip, f)) if "acoustic" in r}
    n = 0
    for r in rows:
        m = by.get(r["utt_id"])
        if m is None:
            continue
        for k in SNR_BANDS:
            r.setdefault(k, m[k])
        r.setdefault("snr_level_drift", "snr_level_drift" in m.get("flags", []))
        n += 1
    return n


def measure_snr(rows: list[dict]) -> int:
    """Measure both-band SNRs from snr_speech_wav/snr_noise_wav where not given; returns the
    number of rows measured. One measurement per (speech, noise) pair, shared across systems."""
    done: dict[tuple[str, str], dict[str, float]] = {}
    n = 0
    for r in tqdm(rows, desc="snr"):
        if r["kind"] != "speech" or all(k in r for k in SNR_BANDS):
            continue
        sp, nz = r.get("snr_speech_wav"), r.get("snr_noise_wav")
        if not (sp and nz):
            continue
        if (sp, nz) not in done:
            x = _load16(sp)
            w = fit_length(_load16(nz), x.size)
            done[(sp, nz)] = {k: band_snr_db(x, w, M.SCORE_FS, band) for k, band in SNR_BANDS.items()}
        for k, v in done[(sp, nz)].items():
            r.setdefault(k, v)
        n += 1
    return n


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


def score_asr(rows: list[dict], make_asr, cache_path: str, model_key: str,
              asr_lora: dict[str, str] | None = None) -> dict:
    """Decode every row. ``make_asr(lora_ckpt)`` builds the recognizer; systems named in
    ``asr_lora`` are decoded with that LoRA merged in (one model loaded at a time).
    Returns the base recognizer's ``info()``."""
    asr_lora = asr_lora or {}
    cache = A.TranscriptCache(cache_path)
    info: dict = {}
    for label in [None, *asr_lora]:
        part = [r for r in rows if (r["system"] == label if label else r["system"] not in asr_lora)]
        if not part:
            continue
        asr = make_asr(asr_lora.get(label))
        for r, rec in zip(part, A.transcribe_items(asr, part, cache, model_key=model_key)):
            r["hyp"] = rec["text"]
            r["hyp_compression_ratio"] = rec["compression_ratio"]
            r["hyp_avg_logprob"] = rec.get("avg_logprob")
        if label is None:
            info.update(asr.info())
        else:
            info.setdefault("asr_lora", {})[label] = asr_lora[label]
        del asr
    refs = [r.get("transcript", "") for r in rows]
    hyps = [r["hyp"] for r in rows]
    counts = A.utterance_counts(refs, hyps)
    flags = A.hallucination_flags(refs, hyps, raw_hyps=hyps)
    subst = A.substitution_flags(refs, hyps)
    for r, c, fl, sb in zip(rows, counts, flags, subst):
        r.update(sub=c.substitutions, dele=c.deletions, ins=c.insertions, hits=c.hits, n_ref=c.n_ref,
                 errors=c.errors, halluc_flag=bool(fl), subst_flag=bool(sb))
        r["n_hyp_words"] = c.hits + c.substitutions + c.insertions
    return info


def paired_keep(rows: list[dict]) -> set[str]:
    """utt_ids kept for "WER unflagged": no system flagged the utterance (§8.3 or substitution)."""
    sp = [r for r in rows if r["kind"] == "speech"]
    flagged = {r["utt_id"] for r in sp if r.get("halluc_flag") or r.get("subst_flag")}
    return {r["utt_id"] for r in sp} - flagged


def _summ_group(rs: list[dict], n_boot: int, seed: int, have_asr: bool, keep_ids: set[str] | None = None) -> dict:
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
        out["subst_rate"] = float(np.mean([r.get("subst_flag", False) for r in sp]))
        keep = np.array([r["utt_id"] in (keep_ids or set()) for r in sp])
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
    for k in SNR_BANDS:
        v = np.array([r.get(k, np.nan) for r in sp], float)
        if np.isfinite(v).any():
            out[f"{k}_median"] = float(np.nanmedian(v))
    return out


def _sex_gap(sp: list[dict], n_boot: int, seed: int) -> dict:
    """Corpus WER by sex, F - M with speakers resampled within sex, and the median SNR of
    each sex in both bands (the gap is read against both)."""
    out: dict = {}
    per: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))
    sx: dict[str, str] = {}
    for r in sp:
        per[r["speaker"]] += (r["errors"], r["n_ref"])
        sx[r["speaker"]] = r["sex"]
    by = {g: np.array([v for k, v in per.items() if sx[k] == g]).reshape(-1, 2) for g in ("F", "M")}
    for g in ("F", "M"):
        rg = [r for r in sp if r["sex"] == g]
        out[f"n_utts_{g}"] = len(rg)
        out[f"n_speakers_{g}"] = len(by[g])
        if len(by[g]):
            out[f"wer_{g}"] = float(by[g][:, 0].sum() / max(by[g][:, 1].sum(), 1.0))
        for k in SNR_BANDS:
            v = np.array([r.get(k, np.nan) for r in rg], float)
            if np.isfinite(v).any():
                out[f"{k}_median_{g}"] = float(np.nanmedian(v))
    if len(by["F"]) and len(by["M"]):
        F, Mm = by["F"], by["M"]
        out["wer_gap_f_m"] = out["wer_F"] - out["wer_M"]
        out["wer_ratio_f_m"] = out["wer_F"] / out["wer_M"] if out["wer_M"] > 0 else float("nan")
        rng = np.random.default_rng(seed)
        bs = np.empty(n_boot)
        for i in range(n_boot):
            f = F[rng.integers(0, len(F), len(F))]
            m = Mm[rng.integers(0, len(Mm), len(Mm))]
            bs[i] = f[:, 0].sum() / max(f[:, 1].sum(), 1.0) - m[:, 0].sum() / max(m[:, 1].sum(), 1.0)
        out["wer_gap_f_m_lo"], out["wer_gap_f_m_hi"] = (float(q) for q in np.percentile(bs, [2.5, 97.5]))
        for k in SNR_BANDS:
            if f"{k}_median_F" in out and f"{k}_median_M" in out:
                out[f"{k}_median_f_m"] = out[f"{k}_median_F"] - out[f"{k}_median_M"]
    return out


def by_sex(rows: list[dict], n_boot: int, seed: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["kind"] == "speech" and r.get("sex"):
            groups[(r["system"], "ALL", "ALL")].append(r)
            groups[(r["system"], r["position"], r["thruster"])].append(r)
    return [{"system": sy, "position": pos, "thruster": thr, **_sex_gap(rs, n_boot, seed)}
            for (sy, pos, thr), rs in tqdm(sorted(groups.items()), desc="by sex")]


def write_by_sex_markdown(path: str, table: list[dict]) -> None:
    head = ["system", "position", "thruster", "n F/M (spk)", "WER F / M %", "F / M ratio",
            "F - M pts [95% CI]", "SNR 500-8k F / M dB", "SNR 50-8k F / M dB"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in table:
        def pair(k, pct=False):
            return f"{_fmt(r.get(k + '_F'), pct=pct)} / {_fmt(r.get(k + '_M'), pct=pct)}"
        lines.append("| " + " | ".join([
            r["system"], r["position"], r["thruster"],
            f'{r.get("n_utts_F", 0)}/{r.get("n_utts_M", 0)} ({r.get("n_speakers_F", 0)}/{r.get("n_speakers_M", 0)})',
            pair("wer", pct=True), _fmt(r.get("wer_ratio_f_m")),
            _fmt(r.get("wer_gap_f_m"), r.get("wer_gap_f_m_lo"), r.get("wer_gap_f_m_hi"), pct=True),
            pair("snr_db_500_8000_median"), pair("snr_db_50_8000_median"),
        ]) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def summarize(rows: list[dict], n_boot: int, seed: int, have_asr: bool,
              keep_ids: set[str] | None = None) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["system"], "ALL", "ALL")].append(r)
        groups[(r["system"], r["position"], r["thruster"])].append(r)
    table = []
    for (sysname, pos, thr), rs in tqdm(sorted(groups.items()), desc="summary"):
        table.append({"system": sysname, "position": pos, "thruster": thr, **_summ_group(rs, n_boot, seed, have_asr, keep_ids)})
    return table


def paired(rows: list[dict], baseline: str, n_boot: int, seed: int, have_asr: bool,
           keep_ids: set[str] | None = None) -> list[dict]:
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
            k = np.array([u in (keep_ids or set()) for u in uids])
            if k.any():
                du = S.paired_bootstrap_wer(ea[k], eb[k], n[k], np.asarray(cl, object)[k],
                                            np.asarray(st, object)[k], n_boot=n_boot, seed=seed)
                row.update(n_unflagged=int(k.sum()), d_wer_unflagged=du.estimate,
                           d_wer_unflagged_lo=du.ci_low, d_wer_unflagged_hi=du.ci_high)
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
    head = cols + (["WER % [95% CI]", "S/D/I", "WER paired-unflagged %", "halluc %", "subst %"] if have_asr else []) + [
        "ESTOI", "STOI", "PESQ-WB(R1)", "CI-SDR dB", "SNR 500-8k dB", "SNR 50-8k dB"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in table:
        cells = [str(r.get(c, "")) for c in cols]
        if have_asr:
            cells += [
                _fmt(r.get("wer"), r.get("wer_lo"), r.get("wer_hi"), pct=True),
                f'{r.get("sub", "-")}/{r.get("dele", "-")}/{r.get("ins", "-")}',
                _fmt(r.get("wer_unflagged"), r.get("wer_unflagged_lo"), r.get("wer_unflagged_hi"), pct=True),
                _fmt(r.get("halluc_rate"), pct=True),
                _fmt(r.get("subst_rate"), pct=True),
            ]
        for m in ("estoi", "stoi", "pesq_wb_r1", "ci_sdr"):
            cells.append(_fmt(r.get(m), r.get(f"{m}_lo"), r.get(f"{m}_hi")))
        for k in SNR_BANDS:  # median over the group's speech rows
            cells.append(_fmt(r.get(f"{k}_median")))
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
    ap.add_argument("--decode", choices=("guard", "fixed"), default="guard",
                    help="guard: the frozen protocol's DecodeConfig.guarded (default); "
                         "fixed: the §8.3 policy, for sensitivity checks only")
    ap.add_argument("--asr-lora", action="append", default=[], metavar="SYSTEM=PATH",
                    help="decode SYSTEM's rows with this C8 LoRA checkpoint merged into Whisper; "
                         "its manifest wav should be the unprocessed cut")
    ap.add_argument("--speaker-sex", default=None, metavar="SPEAKERS.TXT",
                    help="LibriSpeech SPEAKERS.TXT, to fill rows without a sex label")
    ap.add_argument("--snr", default=None, metavar="SNR.JSONL",
                    help="scripts/measure_snr.py output; fills both-band SNRs by utt_id")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_manifest(args.manifest)
    info: dict = {"args": vars(args), "n_rows": len(rows)}
    assign_sex(rows, read_speaker_sex(args.speaker_sex) if args.speaker_sex else None)
    info["snr_bands_hz"] = SNR_BANDS
    if args.snr:
        info["n_rows_snr_joined"] = join_snr(rows, args.snr)
    info["n_rows_snr_measured"] = measure_snr(rows)
    sp_rows = [r for r in rows if r["kind"] == "speech"]
    info["n_speech_rows_without_sex"] = sum(not r.get("sex") for r in sp_rows)
    info["n_speech_rows_without_snr"] = sum(not all(k in r for k in SNR_BANDS) for r in sp_rows)
    if args.level_normalize:
        rows = level_normalize(rows, N.VADConfig())
        info["vad"] = vars(N.VADConfig())
        info["target_dbfs"] = N.TARGET_DBFS
    if not args.no_metrics:
        score_metrics(rows, args.f_lo, args.f_hi)
    have_asr = not args.no_asr
    if have_asr:
        dec = A.DecodeConfig.guarded() if args.decode == "guard" else A.DecodeConfig()
        lora = dict(kv.split("=", 1) for kv in args.asr_lora)
        unknown = set(lora) - {r["system"] for r in rows}
        if unknown:
            raise ValueError(f"--asr-lora systems not in manifest: {sorted(unknown)}")

        def make_asr(ckpt):
            return A.WhisperASR(args.asr_model, device=args.device, decode=dec, lora_ckpt=ckpt)

        # the cache key's model field records level normalization and the decode policy
        # (same format as sim_baseline.asr_key, e.g. whisper-large-v3+lnorm+guard)
        key = f"whisper-{args.asr_model}" + ("+lnorm" if args.level_normalize else "") + (
            "" if args.decode == "fixed" else f"+{args.decode}")
        info["asr"] = score_asr(rows, make_asr, os.path.join(args.out_dir, "transcripts.jsonl"), key, lora)
        info["decode"] = args.decode

    with open(os.path.join(args.out_dir, "per_utt.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: _jsonable(v) for k, v in r.items()}, ensure_ascii=False) + "\n")
    _write_csv(os.path.join(args.out_dir, "per_utt.csv"), rows)

    keep_ids = paired_keep(rows) if have_asr else None
    if keep_ids is not None:
        info["n_utts_paired_unflagged"] = len(keep_ids)
    table = summarize(rows, args.n_boot, args.seed, have_asr, keep_ids)
    _write_csv(os.path.join(args.out_dir, "summary.csv"), table)
    write_markdown(os.path.join(args.out_dir, "summary.md"), table, have_asr)
    if args.baseline:
        pr = paired(rows, args.baseline, args.n_boot, args.seed, have_asr, keep_ids)
        _write_csv(os.path.join(args.out_dir, "paired.csv"), pr)
    if have_asr and any(r.get("sex") for r in sp_rows):
        bs = by_sex(rows, args.n_boot, args.seed)
        _write_csv(os.path.join(args.out_dir, "by_sex.csv"), bs)
        write_by_sex_markdown(os.path.join(args.out_dir, "by_sex.md"), bs)
    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    print(open(os.path.join(args.out_dir, "summary.md"), encoding="utf-8").read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
