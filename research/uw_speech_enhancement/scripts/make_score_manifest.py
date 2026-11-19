"""Build the scripts/score.py manifest from imported pool takes, with aligned R0 cuts.

R0 (RESEARCH_PLAN.md 8.3) is the played utterance itself: it is cut from the rendered
block WAV with the same ``pad_s`` as the hydrophone cut, then shifted by the fine lag
of the cut against it (6.5 step 3, ``align.fine_lag`` within +-20 ms, sub-sample), so
its VAD marks the speech in the cut. score.py's ``--level-normalize`` reads it as
``r0_wav``. R1 (the chain-only reference) is not made here; pass ``--r1-dir`` once it
exists, otherwise rows have no ``ref_wav`` and score.py skips the intrusive metrics.

    python scripts/make_score_manifest.py --align data/real/v0_s2/align.jsonl \
        --bench data/bench/v0 --out data/real/v0_s2/score \
        --snr data/real/v0_s2/snr.jsonl --speaker-sex data/librispeech/LibriSpeech/SPEAKERS.TXT \
        --c8 --se C2=runs/enhance/dfn3_c2/v0_s2

Rows: one per (system, cut) for the scored channel (``--channel``, default 0), anchor
repeats excluded unless ``--keep-anchor``. Systems: C0 (the cut), C8 with ``--c8`` (the
same cut; decode with ``score.py --asr-lora C8=...``), and each ``--se NAME=DIR`` whose
``DIR/<utt_id>.wav`` must exist. Writes ``<out>/manifest.jsonl``, ``<out>/r0/<utt_id>.wav``
and ``<out>/r0_lags.jsonl`` (lag, PSR, low-PSR flag per cut).
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
from uwse.evaluation import align as AL  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--align", type=Path, required=True, help="align.jsonl from import_recording.py")
    ap.add_argument("--bench", type=Path, required=True, help="bench dir (blocks/<content_block>.wav/.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channel", type=int, default=0, help="scored hydrophone channel id")
    ap.add_argument("--keep-anchor", action="store_true", help="also score the drift-anchor repeats")
    ap.add_argument("--snr", type=Path, default=None, help="measure_snr.py snr.jsonl")
    ap.add_argument("--speaker-sex", type=Path, default=None, metavar="SPEAKERS.TXT")
    ap.add_argument("--c8", action="store_true", help="add C8 rows (the C0 cut, decoded with the LoRA)")
    ap.add_argument("--se", action="append", default=[], metavar="NAME=DIR", help="enhanced cuts DIR/<utt_id>.wav")
    ap.add_argument("--r1-dir", type=Path, default=None, help="R1 cuts DIR/<utt_id>.wav, if made")
    ap.add_argument("--max-lag-s", type=float, default=0.020)
    return ap.parse_args(argv)


def r0_cut(block: np.ndarray, timing: dict, k: int, pad_s: float, n_out: int, fs: int) -> np.ndarray:
    """Utterance k of the rendered block with ``pad_s`` either side, zero-filled, ``n_out`` long."""
    u = timing["utterances"][k]
    a = int(u["start"]) - int(round(pad_s * fs))
    out = np.zeros(n_out)
    src0, src1 = max(0, a), min(block.size, a + n_out)
    if src1 > src0:
        out[src0 - a: src1 - a] = block[src0:src1]
    return out


def _speaker_sex(path: Path | None) -> dict[str, str]:
    out = {}
    if path:
        for line in open(path, encoding="utf-8", errors="ignore"):
            f = [x.strip() for x in line.split("|")]
            if not line.startswith(";") and len(f) >= 2 and f[1] in ("F", "M"):
                out[f[0]] = f[1]
    return out


def main(argv=None) -> int:
    a = parse_args(argv)
    se = {}
    for kv in a.se:
        if "=" not in kv:
            raise SystemExit(f"--se expects NAME=DIR, got {kv!r}")
        name, d = kv.split("=", 1)
        if name in ("C0", "C8") or name in se:
            raise SystemExit(f"--se name {name!r} is reserved or repeated")
        se[name] = Path(d)
    rows = [json.loads(l) for l in open(a.align, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["channel"] == a.channel and (a.keep_anchor or not r.get("anchor_repeat"))]
    if not rows:
        raise SystemExit(f"no cuts for channel {a.channel} in {a.align}")
    snr = {}
    if a.snr:
        snr = {r["utt_id"]: r for r in map(json.loads, filter(str.strip, open(a.snr, encoding="utf-8")))
               if "acoustic" in r}
    sex = _speaker_sex(a.speaker_sex)
    (a.out / "r0").mkdir(parents=True, exist_ok=True)
    blocks: dict[str, tuple[np.ndarray, dict]] = {}
    manifest, lags = [], []
    for r in tqdm(rows, desc="r0 + manifest"):
        cb = r["content_block"]
        if cb not in blocks:
            x, fsb = audio.load(str(a.bench / "blocks" / f"{cb}.wav"))
            t = json.loads((a.bench / "blocks" / f"{cb}.json").read_text(encoding="utf-8"))
            if fsb != int(r["fs"]):
                raise SystemExit(f"{cb}: block {fsb} Hz, cut {r['fs']} Hz")
            blocks[cb] = (x if x.ndim == 1 else x[0], t)
        block, timing = blocks[cb]
        fs = int(r["fs"])
        y, _ = audio.load(r["wav"])
        k = next(i for i, u in enumerate(timing["utterances"]) if u["item_id"] == r["librispeech_id"])
        ref = r0_cut(block, timing, k, float(r.get("pad_s", 0.25)), y.size, fs)
        lag = AL.fine_lag(y, ref, fs, max_lag_s=a.max_lag_s, subsample=True)
        r0 = AL.shift_signal(ref, -lag.lag, n_out=y.size)
        r0p = a.out / "r0" / f"{r['utt_id']}.wav"
        audio.save(str(r0p), r0, fs, subtype="FLOAT")
        lags.append({"utt_id": r["utt_id"], "lag_samples": lag.lag, "lag_ms": 1e3 * lag.lag / fs,
                     "psr_dB": lag.psr_db, "low_psr": lag.flagged, "fs": fs})
        spk = r["librispeech_id"].split("-")[0]
        base = {"utt_id": r["utt_id"], "r0_wav": str(r0p), "transcript": r.get("transcript") or "",
                "speaker": spk, "position": r["position_id"], "thruster": r["thruster_state"],
                "kind": "speech", "flags": list(r.get("flags", [])) + (["r0_low_psr"] if lag.flagged else [])}
        if spk in sex:
            base["sex"] = sex[spk]
        if a.r1_dir:
            base["ref_wav"] = str(a.r1_dir / f"{r['utt_id']}.wav")
        m = snr.get(r["utt_id"])
        if m:
            base.update(snr_db_500_8000=m["snr_db_500_8000"], snr_db_50_8000=m["snr_db_50_8000"],
                        snr_level_drift="snr_level_drift" in m.get("flags", []))
        systems = {"C0": r["wav"], **({"C8": r["wav"]} if a.c8 else {}),
                   **{n: str(d / f"{r['utt_id']}.wav") for n, d in se.items()}}
        for name, wav in systems.items():
            if not Path(wav).exists():
                raise SystemExit(f"{name}: missing {wav}")
            manifest.append({**base, "system": name, "wav": wav})
    with open(a.out / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(m) + "\n" for m in manifest)
    with open(a.out / "r0_lags.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(m) + "\n" for m in lags)
    ms = np.array([x["lag_ms"] for x in lags])
    print(f"{len(rows)} cuts x {len(manifest) // len(rows)} systems -> {a.out / 'manifest.jsonl'}; "
          f"R0 lag median {np.median(ms):+.2f} ms [{ms.min():+.2f}, {ms.max():+.2f}], "
          f"{sum(x['low_psr'] for x in lags)} low-PSR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
