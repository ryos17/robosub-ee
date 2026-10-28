"""Tests for uwse.train (RESEARCH_PLAN.md §7.2-7.4): schedule, loop, resume, early
stopping, clipping, datasets (C7 in-air, C9 replay), CLI smoke run, Whisper-LoRA (C8)."""

from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pytest
import torch
from torch import nn

from uwse.adapters.base import SEAdapter, register
from uwse.train.datasets import (
    InAirPairs,
    ReplayPairs,
    SegmentDataset,
    SyntheticPairs,
    build_val_set,
    random_band_gain,
    time_stretch_ir,
)
from uwse.train.loop import TrainConfig, lr_at, lr_factor, train

FS = 8000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@register("_test_fir")
class ToyFIRAdapter(SEAdapter):
    """A learnable 33-tap FIR, zero-initialised; MSE to the target (x `scale`)."""

    sample_rate = FS

    def __init__(self, device: str = "cpu", scale: float = 1.0) -> None:
        super().__init__(device)
        self.scale = scale

    def load_pretrained(self) -> None:
        torch.manual_seed(0)
        self.model = nn.Conv1d(1, 1, 33, padding=16, bias=False)
        nn.init.zeros_(self.model.weight)
        self.model.to(self.device)

    def enhance(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model(wav.unsqueeze(1)).squeeze(1)

    def loss(self, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.mean((self.forward_train(noisy) - clean) ** 2)


def _adapter(scale: float = 1.0) -> ToyFIRAdapter:
    a = ToyFIRAdapter(scale=scale)
    a.load_pretrained()
    return a


def _cfg(tmp_path, **kw) -> TrainConfig:
    base = dict(run_name="t", out_dir=str(tmp_path), lr=1e-2, warmup_steps=3, max_steps=30,
                segment_s=0.25, batch_size=4, audio_s_per_step=2.0, eval_every=10**6,
                log_every=1, ckpt_every=10**6, device="cpu", seed=0)
    base.update(kw)
    return TrainConfig(**base)


def _ds(n: int = 256, seed: int = 0) -> SyntheticPairs:
    return SyntheticPairs(n, FS, dur_s=0.5, seed=seed, snr_range=(5.0, 15.0), reverb=False)


def _log(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, "train_log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
def test_schedule_values_exact():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == pytest.approx(1e-4, rel=1e-12)
    assert lr_at(4, cfg) == pytest.approx(5e-4, rel=1e-12)
    assert lr_at(9, cfg) == pytest.approx(1e-3, rel=1e-12)
    assert lr_at(10, cfg) == pytest.approx(1e-3, rel=1e-12)
    assert lr_at(35, cfg) == pytest.approx(1e-3 * 0.5 * (1 + math.cos(math.pi / 4)), rel=1e-12)
    assert lr_at(60, cfg) == pytest.approx(5e-4, rel=1e-12)
    assert lr_at(110, cfg) == pytest.approx(0.0, abs=1e-18)
    assert lr_factor(110, 10, 110, min_ratio=0.1) == pytest.approx(0.1)
    assert lr_factor(5, 0, 10) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))


def test_accum_from_audio_seconds():
    cfg = TrainConfig(audio_s_per_step=64.0, batch_size=4, segment_s=4.0)
    assert cfg.resolved_accum() == 4
    assert TrainConfig(audio_s_per_step=64.0, batch_size=6, segment_s=4.0).resolved_accum() == 3
    assert TrainConfig(param_mode="lora").resolved_weight_decay() == 0.0
    assert TrainConfig().resolved_weight_decay() == 1e-4
    with pytest.raises(KeyError):
        TrainConfig.from_dict({"lr": 1e-4, "typo_key": 1})


def test_loss_decreases_and_logged_lr_matches_schedule(tmp_path):
    cfg = _cfg(tmp_path)
    res = train(_adapter(), _ds(), None, cfg)
    assert res.step == 30
    rows = [r for r in _log(res.run_dir) if "loss" in r]
    assert [r["step"] for r in rows] == list(range(1, 31))
    first = np.mean([r["loss"] for r in rows[:3]])
    last = np.mean([r["loss"] for r in rows[-3:]])
    assert last < 0.7 * first, (first, last)
    for r in rows:  # logged LR is the LR used for update `step - 1`
        assert r["lr"] == pytest.approx(lr_at(r["step"] - 1, cfg), rel=1e-12)
    assert res.throughput_compute > 0 and "throughput_audio_s_per_compute_s" in rows[0]
    assert rows[0]["audio_s"] == pytest.approx(2.0)  # 2 micro-batches x 4 x 0.25 s
    for name in ("last.pt", "last_state.pt", "best.pt", "summary.json", "config.json"):
        assert os.path.isfile(os.path.join(res.run_dir, name)), name


def test_checkpoint_resume_round_trip(tmp_path):
    full = train(_adapter(), _ds(), None, _cfg(tmp_path / "a", max_steps=12))
    a_full = ToyFIRAdapter()
    a_full.load_pretrained()
    a_full.load_checkpoint(os.path.join(full.run_dir, "last.pt"))

    part = train(_adapter(), _ds(), None, _cfg(tmp_path / "b", max_steps=12, stop_at=6))
    assert part.step == 6
    st = torch.load(os.path.join(part.run_dir, "last_state.pt"), weights_only=False)
    assert st["step"] == 6
    resumed_adapter = _adapter()  # fresh weights: must be overwritten by last.pt
    res = train(resumed_adapter, _ds(), None, _cfg(tmp_path / "b", max_steps=12))
    assert res.step == 12
    rows = _log(res.run_dir)
    steps = [r["step"] for r in rows if "loss" in r]
    assert steps == list(range(1, 13))  # continues from 7, no duplicates
    assert any(r.get("event") == "resume" and r["step"] == 6 for r in rows)
    torch.testing.assert_close(resumed_adapter.model.weight, a_full.model.weight,
                               rtol=0, atol=1e-7)


def test_early_stopping_triggers(tmp_path):
    cfg = _cfg(tmp_path, max_steps=100, eval_every=2, patience=3, min_delta=1e9)
    res = train(_adapter(), _ds(), _ds(4, seed=99), cfg)
    # first eval sets best (from +inf); the next 3 cannot improve by 1e9 -> stop at 2 * 4
    assert res.stopped_early and res.step == 8
    assert res.best_step == 2
    rows = _log(res.run_dir)
    assert any(r.get("event") == "early_stop" for r in rows)
    assert len([r for r in rows if "val_loss" in r]) == 4
    assert os.path.isfile(os.path.join(res.run_dir, "best.pt"))


def test_grad_clipping_applied(tmp_path):
    seen: list[float] = []

    def hook(opt, args, kwargs):
        g = [p.grad for grp in opt.param_groups for p in grp["params"] if p.grad is not None]
        seen.append(float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in g]))))

    from torch.optim.optimizer import register_optimizer_step_pre_hook

    h = register_optimizer_step_pre_hook(hook)
    try:
        res = train(_adapter(scale=1e4), _ds(), None, _cfg(tmp_path, max_steps=5, grad_clip=0.5))
    finally:
        h.remove()
    pre = [r["grad_norm"] for r in _log(res.run_dir) if "grad_norm" in r]
    assert len(seen) == 5 and min(pre) > 0.5  # unclipped norms exceed the threshold
    assert max(seen) <= 0.5 * (1 + 1e-5)  # the optimizer only ever saw clipped gradients


def test_segment_dataset_crops_deterministic():
    seg = SegmentDataset(_ds(8), 1000)
    a, b = seg[(3, 11)], seg[(3, 11)]
    torch.testing.assert_close(a[0], b[0])
    assert a[0].shape == (1000,) and a[1].shape == (1000,)
    c = seg[(3, 12)]
    assert c[2]["crop_offset"] != a[2]["crop_offset"] or not torch.equal(c[0], a[0])
    torch.testing.assert_close(seg[5][0], seg[5][0])  # validation keys: fixed crop
    pad = SegmentDataset(_ds(2), 6000)[0]
    assert pad[0].shape == (6000,) and float(pad[0][-100:].abs().sum()) == 0.0


def _clean_arrays(rng: np.random.Generator, n: int = 2, dur: float = 1.0) -> list[np.ndarray]:
    t = np.arange(int(dur * FS)) / FS
    return [np.sin(2 * np.pi * rng.uniform(100, 300) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
            for _ in range(n)]


def test_inair_pairs_finite_small_room(rng):
    pytest.importorskip("pyroomacoustics")
    ds = InAirPairs(_clean_arrays(rng), None, FS, fs_clean=FS, n_items=3, seed=1, segment_s=0.5,
                    room_dim_range=((4.0, 5.0), (3.0, 4.0), (2.5, 3.0)), t60_range=(0.3, 0.4),
                    distance_range=(0.5, 2.0), max_order=6, n_rirs=2)
    for i in range(3):
        noisy, target, meta = ds[i]
        assert noisy.shape == target.shape == (int(0.5 * FS),)
        assert torch.isfinite(noisy).all() and torch.isfinite(target).all()
        assert float(noisy.abs().max()) <= 0.99 + 1e-6 and float(target.abs().max()) > 0
        assert not torch.allclose(noisy, target)
        assert 0.3 <= meta["t60"] <= 0.4 and meta["c"] == 343.0
    torch.testing.assert_close(ds[1][0], ds[1][0])  # deterministic per index


def test_replay_pairs_and_ir_perturbations(rng):
    n = int(0.4 * FS)
    t = np.arange(n) / FS
    irs = []
    for _ in range(2):
        h = rng.standard_normal(n) * np.exp(-6.91 * t / 0.3) * 0.2
        h[40] = 1.0
        irs.append(h)
    h2 = time_stretch_ir(irs[0], 1.1)
    assert abs(h2.size - 1.1 * n) <= 2
    g = random_band_gain(irs[0], FS, rng, 2.0)
    ratio_db = 20 * np.log10(np.abs(np.fft.rfft(g, 8192)) / np.abs(np.fft.rfft(irs[0], 8192)) + 1e-12)
    assert np.percentile(np.abs(ratio_db), 90) < 2.5
    ds = ReplayPairs(_clean_arrays(rng), None, FS, irs=irs, fs_clean=FS, n_items=4, seed=0,
                     segment_s=0.5, p_tail=1.0)
    for i in range(4):
        noisy, target, meta = ds[i]
        assert noisy.shape == target.shape == (int(0.5 * FS),)
        assert torch.isfinite(noisy).all() and meta["tail_resynth"]
        assert 0.9 <= meta["stretch"] <= 1.1


def test_build_val_set_frozen(tmp_path):
    ds = _ds(3, seed=5)
    va = build_val_set(ds, 3, out_dir=str(tmp_path / "val"), fs=FS)
    assert len(va) == 3
    x, y, meta = va[1]
    torch.testing.assert_close(x, ds[1][0], atol=1e-6, rtol=0)  # FLOAT wav: lossless
    assert meta["index"] == 1
    again = build_val_set(ds, 3, out_dir=str(tmp_path / "val"), fs=FS)  # reused
    assert len(again) == 3


def test_cli_smoke(tmp_path):
    spec = importlib.util.spec_from_file_location("train_script", os.path.join(ROOT, "scripts", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {
        "run_name": "toy", "task": "se", "adapter": {"name": "_test_fir"},
        "data": {"kind": "synthetic"},
        "train": {"out_dir": str(tmp_path), "lr": 1e-2, "warmup_steps": 2, "max_steps": 100,
                  "segment_s": 0.25, "batch_size": 2, "audio_s_per_step": 1.0, "device": "cpu"},
        "seeds": [3, 4],
    }
    p = tmp_path / "toy.yaml"
    import yaml
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = mod.main(["--config", str(p), "--smoke", "--steps", "4", "--set", "train.lr=0.005"])
    assert len(out) == 1 and out[0]["step"] == 4
    assert out[0]["run_dir"].endswith("toy_s3_smoke")
    summ = json.load(open(os.path.join(out[0]["run_dir"], "summary.json"), encoding="utf-8"))
    assert summ["throughput_compute"] > 0 and summ["accum_steps"] == 2


@pytest.mark.network
@pytest.mark.slow
def test_whisper_lora_tiny(tmp_path):
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from torch.utils.data import Dataset

    from uwse.train.whisper_lora import LoRASpec, WhisperLoRATrainable, build_whisper_lora, train_whisper_lora

    class Toy(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, key):
            i = key[0] if isinstance(key, tuple) else key
            g = torch.Generator().manual_seed(i)
            return 0.05 * torch.randn(16000 * 2, generator=g), ["Hello world.", "Testing one two.",
                                                               "Under the sea.", "Speech in water."][i]

    model, proc = build_whisper_lora("openai/whisper-tiny", LoRASpec(), "cpu")
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_lora > 0 and all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    cfg = TrainConfig(run_name="wl", out_dir=str(tmp_path), lr=1e-3, warmup_steps=1, max_steps=3,
                      batch_size=2, accum_steps=1, eval_every=3, log_every=1, device="cpu")
    res = train_whisper_lora(Toy(), Toy(), cfg, model_name="openai/whisper-tiny",
                             model_and_processor=(model, proc))
    assert res.step == 3 and math.isfinite(res.best_val)
    changed = any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if p.requires_grad)
    assert changed
    tr = WhisperLoRATrainable(model, "openai/whisper-tiny")
    tr.load(os.path.join(res.run_dir, "best.pt"))
    assert os.path.isdir(os.path.join(res.run_dir, "lora_adapter_best"))
