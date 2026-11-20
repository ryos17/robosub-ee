"""Pilot (or chain-reference) WER with C0 and the pre-registered quiet/added-noise decision.

Pool plan of 2026-11-19: the pilot take at 3 m plays ~20 pilot sentences that are not in
the 200-utterance test set. They are scored on site with C0 only: Whisper large-v3, the
frozen guarded decode (``DecodeConfig.guarded()``), no LoRA. The pre-registered rule:

    pilot C0 corpus WER >= 10%  ->  "QUIET primary"        (quiet recordings primary,
                                                            added noise secondary)
    pilot C0 corpus WER <  10%  ->  "ADDED-NOISE primary"  (added real noise at 0 dB SNR
                                                            primary, quiet secondary)

The same script scores the chain-reference take (0.5-1 m) with ``--role chainref``; it then
prints the WER only and makes no decision.

Input (exactly one):

    --manifest  score.py manifest from scripts/make_score_manifest.py (its C0 rows; aligned
                R0 cuts present, so the frozen ``+lnorm+guard`` decode is used: score.py's
                R0-VAD level normalization to -26 dBFS before decoding). Preferred.
    --align     align.jsonl from scripts/import_recording.py, when no bench/R0 is at hand on
                site. No level normalization (decode key ``+guard``); recorded in the json.

Decoding, counting and the bootstrap are score.py's own functions (``read_manifest``,
``level_normalize``, ``score_asr``, ``_summ_group``); nothing is reimplemented here.

    python scripts/pilot_wer.py --manifest data/real/v1_s2/pilot/score/manifest.jsonl \
        --out data/real/v1_s2/pilot/pilot_wer.json --device cuda

Writes ``--out`` (json: timestamp, WER with speaker-cluster 95% CI, S/D/I, counts, the
decision, ASR info, input sha256, per-utterance ref/hyp) and ``<out stem>_transcripts.jsonl``
(transcript cache). An existing ``--out`` is never overwritten (the decision is made once);
pass ``--overwrite`` only to redo a run that failed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

THRESHOLD = 0.10  # pre-registered 2026-11-19
DECISION_QUIET = "QUIET primary"
DECISION_NOISE = "ADDED-NOISE primary"


def load_score():
    """scripts/score.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("uwse_score_cli", ROOT / "scripts" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCORE = load_score()


def decide(wer: float, threshold: float = THRESHOLD) -> str:
    """The pre-registered rule: >= threshold -> quiet primary, else added noise primary."""
    return DECISION_QUIET if wer >= threshold else DECISION_NOISE


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_from_align(path: Path, channel: int) -> list[dict]:
    """C0 score rows from import_recording.py's align.jsonl (scored channel, no anchor repeats)."""
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["channel"]) != channel or r.get("anchor_repeat"):
            continue
        rows.append({"utt_id": r["utt_id"], "system": "C0", "wav": r["wav"],
                     "transcript": r.get("transcript") or "",
                     "speaker": str(r["librispeech_id"]).split("-")[0],
                     "position": str(r.get("position_id")), "thruster": str(r.get("thruster_state") or "none"),
                     "kind": "speech", "flags": list(r.get("flags", []))})
    return rows


def rows_from_manifest(path: Path) -> list[dict]:
    rows = [r for r in SCORE.read_manifest(str(path)) if r["system"] == "C0" and r["kind"] == "speech"]
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="make_score_manifest.py manifest (C0 rows used)")
    src.add_argument("--align", type=Path, help="import_recording.py align.jsonl (no level normalization)")
    ap.add_argument("--channel", type=int, default=0, help="scored channel id (--align only)")
    ap.add_argument("--out", type=Path, required=True, help="output json")
    ap.add_argument("--role", choices=("pilot", "chainref"), default="pilot")
    ap.add_argument("--device", choices=("cuda", "cpu"), default=None)
    ap.add_argument("--asr-model", default="large-v3")
    ap.add_argument("--n-boot", type=int, default=SCORE.S.DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="pre-registered 0.10; changing it makes the decision post hoc")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.out.exists() and not a.overwrite:
        raise SystemExit(f"{a.out} exists; the {a.role} result is written once (use --overwrite to redo)")
    src = a.manifest or a.align
    rows = rows_from_manifest(a.manifest) if a.manifest else rows_from_align(a.align, a.channel)
    if not rows:
        raise SystemExit(f"no C0 speech rows in {src}")
    lnorm = bool(a.manifest) and all(r.get("r0_wav") for r in rows)
    if lnorm:
        rows = SCORE.level_normalize(rows, SCORE.N.VADConfig())
    dec = SCORE.A.DecodeConfig.guarded()

    def make_asr(ckpt):
        assert ckpt is None  # C0 only: never a LoRA here
        return SCORE.A.WhisperASR(a.asr_model, device=a.device, decode=dec, lora_ckpt=None)

    key = f"whisper-{a.asr_model}" + ("+lnorm" if lnorm else "") + "+guard"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cache = a.out.with_name(a.out.stem + "_transcripts.jsonl")
    asr_info = SCORE.score_asr(rows, make_asr, str(cache), key, None)
    summ = SCORE._summ_group(rows, a.n_boot, a.seed, True, SCORE.paired_keep(rows))
    wer = float(summ["wer"])
    rec = {
        "role": a.role,
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "system": "C0",
        "asr_key": key,
        "level_normalize": lnorm,
        "input": str(src), "input_sha256": sha256_file(src),
        "n_utts": summ["n_utts"], "n_speakers": len({r["speaker"] for r in rows}),
        "n_ref_words": summ["n_ref_words"],
        "wer": wer, "wer_lo": summ["wer_lo"], "wer_hi": summ["wer_hi"],
        "sub": summ["sub"], "dele": summ["dele"], "ins": summ["ins"],
        "halluc_rate": summ["halluc_rate"], "subst_rate": summ["subst_rate"],
        "positions": sorted({r["position"] for r in rows}),
        "n_boot": a.n_boot, "seed": a.seed,
        "asr": asr_info,
        "per_utt": [{"utt_id": r["utt_id"], "ref": r.get("transcript", ""), "hyp": r["hyp"],
                     "errors": r["errors"], "n_ref": r["n_ref"], "flags": r.get("flags", [])} for r in rows],
    }
    if a.role == "pilot":
        rec.update(threshold=a.threshold, threshold_preregistered=a.threshold == THRESHOLD,
                   decision=decide(wer, a.threshold),
                   primary_condition="quiet" if wer >= a.threshold else "added_noise_0dB")
    a.out.write_text(json.dumps(rec, indent=1, default=SCORE._jsonable), encoding="utf-8")
    print(f"{a.role}: C0 corpus WER {100 * wer:.2f}% [{100 * summ['wer_lo']:.2f}, {100 * summ['wer_hi']:.2f}] "
          f"over {summ['n_utts']} utts / {rec['n_speakers']} speakers ({summ['n_ref_words']} words, "
          f"S/D/I {summ['sub']}/{summ['dele']}/{summ['ins']}, decode {key})")
    if a.role == "pilot":
        print(f"DECISION ({'>=' if wer >= a.threshold else '<'} {100 * a.threshold:g}%): {rec['decision']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
