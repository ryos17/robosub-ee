"""Rank-stacked LoRA averaging: merged delta equals the mean of the inputs' merged deltas."""
from __future__ import annotations

import pytest
import torch

from uwse.evaluation.asr import average_lora_checkpoints, lora_deltas_openai

KEYS = [
    "base_model.model.model.encoder.layers.0.self_attn.q_proj.lora_{}.weight",
    "base_model.model.model.decoder.layers.1.encoder_attn.v_proj.lora_{}.weight",
]


def _ckpt(seed: int, r: int = 4, alpha: float = 16.0, d: int = 12) -> dict:
    g = torch.Generator().manual_seed(seed)
    model = {}
    for k in KEYS:
        model[k.format("A")] = torch.randn(r, d, generator=g)
        model[k.format("B")] = torch.randn(d, r, generator=g)
    return {"model": model, "adapter": "whisper_lora", "base": "x",
            "lora": {"r": r, "alpha": alpha, "dropout": 0.05, "target_modules": ("q_proj", "v_proj")}}


def _deltas(c: dict) -> dict:
    return lora_deltas_openai(c["model"], int(c["lora"]["r"]), float(c["lora"]["alpha"]))


@pytest.mark.parametrize("k", [2, 3])
def test_average_is_mean_of_merged_deltas(k):
    cs = [_ckpt(s) for s in range(k)]
    avg = average_lora_checkpoints(cs)
    assert avg["lora"]["r"] == 4 * k and avg["lora"]["alpha"] == 16.0 * k
    got = _deltas(avg)
    parts = [_deltas(c) for c in cs]
    for name, d in got.items():
        ref = sum(p[name] for p in parts) / k
        torch.testing.assert_close(d, ref, rtol=1e-5, atol=1e-5)


def test_average_of_identical_checkpoints_is_unchanged():
    c = _ckpt(0)
    got, ref = _deltas(average_lora_checkpoints([c, c])), _deltas(c)
    for name in ref:
        torch.testing.assert_close(got[name], ref[name], rtol=1e-5, atol=1e-5)


def test_rejects_mismatched_rank():
    with pytest.raises(ValueError):
        average_lora_checkpoints([_ckpt(0, r=4), _ckpt(1, r=8)])


def test_rejects_mismatched_keys():
    a, b = _ckpt(0), _ckpt(1)
    b["model"].pop(KEYS[0].format("A"))
    b["model"].pop(KEYS[0].format("B"))
    with pytest.raises(ValueError):
        average_lora_checkpoints([a, b])
