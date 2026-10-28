"""Adversarial-review regression tests for uwse.train (RESEARCH_PLAN.md §5.3, §7.2-7.4).

Each test pins a defect found in review: resume after early stopping, best-weights
restore, seeding before model init (C10 from scratch), Windows-spawn picklability of
the C7/C9 datasets, uwsim-consistent padding/SNR for C7/C9, LoRA weight decay 0 for
C3, benchmark-IR exclusion relative to the IR root, and steady-state throughput.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from uwse.adapters.base import SEAdapter, register
from uwse.train.datasets import InAirPairs, ReplayPairs, SyntheticPairs
from uwse.train.loop import TrainConfig, train

FS = 8000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@register("_review_randinit")
class RandInitAdapter(SEAdapter):
    """Learnable FIR with RANDOM init from the global torch RNG (like a from-scratch C10)."""

    sample_rate = FS

    def load_pretrained(self) -> None:
        self.model = nn.Conv1d(1, 1, 17, padding=8, bias=False).to(self.device)

    def enhance(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model(wav.unsqueeze(1)).squeeze(1)

    def loss(self, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        return torch.mean((self.forward_train(noisy) - clean) ** 2)


def _script():
    spec = importlib.util.spec_from_file_location("train_script_review",
                                                  os.path.join(ROOT, "scripts", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ds(n: int = 64, seed: int = 0) -> SyntheticPairs:
    return SyntheticPairs(n, FS, dur_s=0.5, seed=seed, snr_range=(5.0, 15.0), reverb=False)


def _cfg(tmp_path, **kw) -> TrainConfig:
    base = dict(run_name="r", out_dir=str(tmp_path), lr=1e-2, warmup_steps=2, max_steps=40,
                segment_s=0.25, batch_size=4, audio_s_per_step=1.0, eval_every=10**6,
                log_every=1, ckpt_every=10**6, device="cpu", seed=0)
    base.update(kw)
    return TrainConfig(**base)


def _adapter() -> RandInitAdapter:
    torch.manual_seed(123)
    a = RandInitAdapter()
    a.load_pretrained()
    return a


# ---------------------------------------------------------------------------
def test_resume_after_early_stop_does_not_train_further(tmp_path):
    cfg = _cfg(tmp_path, eval_every=2, patience=2, min_delta=1e9)
    res = train(_adapter(), _ds(), _ds(4, seed=9), cfg)
    assert res.stopped_early and res.step == 6
    # relaunching the same command (e.g. a chunked job) must not continue past the stop
    again = train(_adapter(), _ds(), _ds(4, seed=9), _cfg(tmp_path, eval_every=2, patience=2,
                                                          min_delta=1e9))
    assert again.step == 6 and again.stopped_early


def test_train_restores_best_weights(tmp_path):
    cfg = _cfg(tmp_path, eval_every=2, patience=2, min_delta=1e9)
    ad = _adapter()
    res = train(ad, _ds(), _ds(4, seed=9), cfg)
    best = torch.load(os.path.join(res.run_dir, "best.pt"), weights_only=False)
    assert best["step"] == 2 and res.step == 6
    torch.testing.assert_close(ad.model.weight, best["model"]["weight"], rtol=0, atol=0)


def test_steady_throughput_in_summary(tmp_path):
    res = train(_adapter(), _ds(), None, _cfg(tmp_path, max_steps=6))
    summ = json.load(open(os.path.join(res.run_dir, "summary.json"), encoding="utf-8"))
    assert summ["throughput_compute_steady"] > 0
    assert summ["steady_from_step"] >= 1


def test_cli_seed_controls_random_init(tmp_path):
    mod = _script()
    cfg = {"run_name": "ri", "task": "se", "adapter": {"name": "_review_randinit"},
           "data": {"kind": "synthetic"},
           "train": {"out_dir": str(tmp_path), "lr": 1e-2, "warmup_steps": 1, "max_steps": 3,
                     "segment_s": 0.25, "batch_size": 2, "audio_s_per_step": 0.5,
                     "device": "cpu", "num_workers": 0}}
    p = tmp_path / "ri.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def weights(seed: int, sub: str) -> torch.Tensor:
        torch.manual_seed(1000 + len(sub))  # perturb the global RNG between runs
        out = mod.main(["--config", str(p), "--smoke", "--steps", "2", "--seed", str(seed),
                        "--set", f"train.out_dir={tmp_path / sub}"])
        st = torch.load(os.path.join(out[0]["run_dir"], "last.pt"), weights_only=False)
        return st["model"]["weight"]

    a, b, c = weights(5, "a"), weights(5, "bb"), weights(6, "ccc")
    torch.testing.assert_close(a, b, rtol=0, atol=0)  # same seed -> same init and run
    assert not torch.equal(a, c)


def _clean(n_s: float = 0.3) -> list[np.ndarray]:
    t = np.arange(int(n_s * FS)) / FS
    return [np.sin(2 * np.pi * 200 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))]


def _irs(rng) -> list[np.ndarray]:
    n = int(0.2 * FS)
    h = rng.standard_normal(n) * np.exp(-6.91 * np.arange(n) / FS / 0.15) * 0.2
    h[20] = 1.0
    return [h]


def test_pair_datasets_picklable_with_chain_fir(rng):
    """Windows DataLoader workers use spawn: the dataset (with a measured h_chain) must pickle."""
    chain = np.r_[1.0, 0.5, 0.25]
    ds = ReplayPairs(_clean(1.0), None, FS, irs=_irs(rng), fs_clean=FS, n_items=2,
                     segment_s=0.25, chain=chain, p_tail=0.0)
    x0 = ds[0]
    ds2 = pickle.loads(pickle.dumps(ds))  # after use too (lazy noise fn already built)
    torch.testing.assert_close(ds2[0][0], x0[0])
    torch.testing.assert_close(ds2[0][1], x0[1])


def test_short_utterance_padded_before_mixing_like_uwsim(rng):
    """uwsim zero-pads the clean speech to the segment and simulates noise over all of it;
    C7/C9 must do the same, not return a short item that is later padded with digital zeros."""
    ds = ReplayPairs(_clean(0.3), None, FS, irs=_irs(rng), fs_clean=FS, n_items=2,
                     segment_s=0.5, p_tail=0.0)
    noisy, target, _ = ds[0]
    assert noisy.shape == target.shape == (int(0.5 * FS),)
    assert float(noisy[-200:].abs().mean()) > 0  # noise continues past the speech


def test_c3_lora_override_gets_zero_weight_decay():
    with open(os.path.join(ROOT, "configs", "c2_dfn3.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mod = _script()
    lora = TrainConfig.from_dict(mod.apply_overrides(cfg, ["train.param_mode=lora"])["train"])
    full = TrainConfig.from_dict(cfg["train"])
    assert lora.resolved_weight_decay() == 0.0  # §7.4 LoRA column
    assert full.resolved_weight_decay() == 1e-4


def test_inair_replay_snr_and_level_follow_uwsim_config():
    """'same noise, same SNR rule' (C7): ranges are taken from the uwsim config."""
    mod = _script()
    from uwse.uwsim.config import load_config

    sc = load_config(os.path.join(ROOT, "configs", "uwsim_v0.yaml"))
    nc = sc.noise
    lo = max(nc.snr_measured[0] - nc.snr_widen_db, nc.snr_clip[0])
    hi = min(nc.snr_measured[1] + nc.snr_widen_db, nc.snr_clip[1])
    for name in ("c7_inair_dfn3.yaml", "c9_replay_dfn3.yaml"):
        with open(os.path.join(ROOT, "configs", name), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        params = mod.sim_matched_params(cfg["data"])
        assert tuple(params["snr_range"]) == (lo, hi), name
        assert tuple(params["level_dbfs_range"]) == tuple(sc.chain.level_dbfs), name


def test_replay_exclusion_is_relative_to_ir_root(tmp_path, rng):
    from uwse import audio

    root = tmp_path / "benchmarks_project" / "irs"  # an ancestor dir contains "bench"
    (root / "calib").mkdir(parents=True)
    (root / "bench_P1").mkdir()
    h = _irs(rng)[0]
    audio.save(str(root / "calib" / "a.wav"), h * 0.5, FS, "FLOAT")
    audio.save(str(root / "bench_P1" / "b.wav"), h * 0.5, FS, "FLOAT")
    ds = ReplayPairs(_clean(1.0), None, FS, irs=str(root), fs_clean=FS, n_items=1,
                     segment_s=0.25, p_tail=0.0)
    assert len(ds.irs) == 1 and "calib" in ds.ir_names[0]


def test_check_not_datasets_windows_drive():
    mod = _script()
    with pytest.raises(ValueError):
        mod._check_not_datasets("/datasets/x")
    with pytest.raises(ValueError):
        mod._check_not_datasets("C:\\datasets\\x")
    mod._check_not_datasets(os.path.join(ROOT, "runs"))
