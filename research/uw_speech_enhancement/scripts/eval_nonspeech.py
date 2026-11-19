"""Invented words on non-speech audio: C8 (guarded decode) vs C8g (C8 + no-speech gate).

Decodes every segment of the simulated non-speech set (``scripts/make_nonspeech_set.py``) with
C8 under ``DecodeConfig.guarded()``, records each 30 s window's ``p_ns`` and the long-form
segments, and reports how many segments get any output word, with and without the gate
(``uwse.evaluation.nospeech_gate``). Each segment is level-normalized with its frozen
``gain_median`` (the median speech gain of its noise type and range group).

    python scripts/make_nonspeech_set.py --split dev
    python scripts/eval_nonspeech.py --split dev            # GPU, about 15 min for 360 segments
    python scripts/eval_nonspeech.py --split dev --report   # CPU, from the stored decode

Outputs in ``--out`` (default runs/nonspeech/<split>): decode.jsonl (resumable) and
summary.json. The test split is held out: decode it once, with nothing chosen from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uwse.evaluation import nospeech_gate as G  # noqa: E402

LORA = ROOT / "runs" / "c8_whisper_lora_stat_s1" / "best_wer.pt"
LORA_SHA_PREFIX = "5063f02f"


def load_manifest(data: Path, split: str) -> list[dict]:
    with open(data / split / "manifest.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def segment_audio(data: Path, split: str, r: dict) -> np.ndarray:
    import soundfile as sf

    x, fs = sf.read(data / split / r["wav"], dtype="float64")
    assert fs == G.FS, (r["seg_id"], fs)
    return (x * float(r["gain_median"])).astype(np.float32)


def n_words(text: str) -> int:
    from uwse.evaluation import asr as A

    return len(A.normalize_text(text).split())


def summarize(decoded: list[dict], tau: float = G.TAU) -> dict:
    """Segments with any (normalized) output word, overall and per noise type, for C8 and C8g."""
    rows = []
    for d in decoded:
        gated, _ = G.gate_text(d["text"], d.get("segments") or [], d["p_ns"], tau)
        rows.append({"noise_type": d["noise_type"], "C8": n_words(d["text"]), "C8g": n_words(gated)})

    def block(rs: list[dict]) -> dict:
        out = {"n": len(rs)}
        for s in ("C8", "C8g"):
            k = sum(r[s] > 0 for r in rs)
            out[s] = {"with_output": k, "rate": k / len(rs) if rs else float("nan"),
                      "wilson95": G.wilson(k, len(rs)), "words": sum(r[s] for r in rs)}
        return out

    types = sorted({r["noise_type"] for r in rows})
    return {"tau": tau, "overall": block(rows), "by_noise_type": {t: block([r for r in rows if r["noise_type"] == t])
                                                                 for t in types}}


def decode(rows: list[dict], data: Path, split: str, out: Path, device: str) -> None:
    from tqdm import tqdm

    from uwse.evaluation import asr as A

    sha = hashlib.sha256(LORA.read_bytes()).hexdigest()
    if not sha.startswith(LORA_SHA_PREFIX):
        raise SystemExit(f"{LORA}: sha256 {sha[:8]} != {LORA_SHA_PREFIX} (not the frozen C8 adapter)")
    path = out / "decode.jsonl"
    done = {json.loads(x)["seg_id"] for x in open(path, encoding="utf-8") if x.strip()} if path.exists() else set()
    todo = [r for r in rows if r["seg_id"] not in done]
    if not todo:
        return
    asr = A.WhisperASR("large-v3", device=device, decode=A.DecodeConfig.guarded(), lora_ckpt=str(LORA))
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for r in tqdm(todo, desc=f"decode {split}"):
            x = segment_audio(data, split, r)
            p = G.window_p_ns(asr, x)
            res = G.decode_with_segments(asr, x)
            f.write(json.dumps({"seg_id": r["seg_id"], "noise_type": r["noise_type"], "range_m": r["range_m"],
                                "p_ns": p, "text": res["text"], "segments": res["segments"]},
                               ensure_ascii=False) + "\n")
            f.flush()
    (out / "run_info.json").write_text(json.dumps({"lora_sha256": sha, "asr": asr.info(), "device": device,
                                                   "decode_wall_s": time.time() - t0}, indent=1, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "nonspeech_v0")
    ap.add_argument("--out", type=Path, default=None, help="default runs/nonspeech/<split>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", action="store_true", help="only summarize the stored decode (CPU)")
    a = ap.parse_args(argv)
    out = a.out or ROOT / "runs" / "nonspeech" / a.split
    out.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(a.data, a.split)
    if not a.report:
        decode(rows, a.data, a.split, out, a.device)
    decoded = [json.loads(x) for x in open(out / "decode.jsonl", encoding="utf-8") if x.strip()]
    if len(decoded) != len(rows):
        print(f"warning: {len(decoded)} of {len(rows)} segments decoded; summary covers those only")
    s = summarize(decoded)
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    o = s["overall"]
    for sysname in ("C8", "C8g"):
        lo, hi = o[sysname]["wilson95"]
        print(f"{sysname:4s} {o[sysname]['with_output']}/{o['n']} segments with output "
              f"({100 * o[sysname]['rate']:.1f}%, Wilson 95% [{100 * lo:.2f}, {100 * hi:.2f}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
