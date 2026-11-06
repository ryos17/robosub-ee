"""Average C8 Whisper LoRA checkpoints (e.g. two seeds) into one exact-average adapter.

The output is a rank-K*r LoRA checkpoint whose merged delta is the mean of the
inputs' merged deltas (`uwse.evaluation.asr.average_lora_checkpoints`). It loads
anywhere a C8 checkpoint does (``sim_baseline.py --asr-lora``, ``dev_wer_select.py``).

    python scripts/average_lora.py --out runs/c8_whisper_lora_stat_avg/ckpt_step030000.pt \
        runs/c8_whisper_lora_stat_s0/last.pt runs/c8_whisper_lora_stat_s1/ckpt_step030000.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uwse.evaluation.asr import average_lora_checkpoints  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpts", nargs="+", help="C8 LoRA checkpoints with the same r, alpha and targets")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if len(args.ckpts) < 2:
        ap.error("need at least two checkpoints")
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in args.ckpts]
    avg = average_lora_checkpoints(states)
    avg["sources"] = [{"path": p, "step": s.get("step")} for p, s in zip(args.ckpts, states)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(avg, args.out)
    print(f"averaged {len(states)} checkpoints (r {avg['lora']['r']}, alpha {avg['lora']['alpha']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
