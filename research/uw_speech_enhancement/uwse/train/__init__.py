"""Model-agnostic fine-tuning (RESEARCH_PLAN.md §7.2-7.4; conditions C2, C3, C7-C10).

loop          `train(adapter, train_ds, val_ds, cfg)` / generic `fit`; AdamW, warm-up +
              cosine, clipping, audio-seconds batching, early stopping, resume,
              JSONL log with measured throughput (W4 smoke test)
datasets      uwsim (C2), in-air control (C7), measured-IR replay (C9), synthetic,
              cropping/resampling wrappers, frozen sim-val builder
whisper_lora  C8: Whisper large-v3 + LoRA on uwsim audio (transformers + peft,
              imported lazily)

CLI: ``python scripts/train.py --config configs/c2_dfn3.yaml``.
"""

from .datasets import (
    InAirPairs,
    ReplayPairs,
    Resampled,
    SegmentDataset,
    SimPairsDataset,
    SyntheticPairs,
    WavPairs,
    build_val_set,
    collate_pairs,
    list_audio_files,
)
from .loop import (
    AdapterTrainable,
    StepBatchSampler,
    TrainConfig,
    TrainResult,
    evaluate,
    fit,
    lr_at,
    lr_factor,
    seed_everything,
    train,
)

__all__ = [
    "AdapterTrainable", "StepBatchSampler", "TrainConfig", "TrainResult", "evaluate", "fit",
    "lr_at", "lr_factor", "seed_everything", "train",
    "InAirPairs", "ReplayPairs", "Resampled", "SegmentDataset", "SimPairsDataset",
    "SyntheticPairs", "WavPairs", "build_val_set", "collate_pairs", "list_audio_files",
]
