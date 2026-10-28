"""Tests for uwse.adapters.kiel_unet (C10) and uwse.adapters.lora (§7.2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import resample_poly
from torch import nn

from uwse.adapters.base import build
from uwse.adapters.kiel_unet import KielUNet, KielUNetAdapter, ideal_ratio_mask, resample_torch
from uwse.adapters.lora import (
    LoRAGRU,
    apply_lora,
    count_trainable,
    load_lora,
    lora_param_count,
    lora_state_dict,
    merge_lora,
    save_lora,
    unmerge_lora,
)


# ---------------------------------------------------------------------------
# Kiel U-Net
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("t", [1, 17, 100, 128, 255])
def test_unet_shape_roundtrip(t):
    torch.manual_seed(0)
    net = KielUNet(base_channels=4).eval()
    x = torch.randn(2, 1, 256, t)
    with torch.no_grad():
        m = net(x)
    assert m.shape == x.shape
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0


def test_irm_range_and_limits():
    torch.manual_seed(0)
    s = torch.rand(3, 257, 40)
    n = torch.rand(3, 257, 40)
    irm = ideal_ratio_mask(s, n)
    assert float(irm.min()) >= 0.0 and float(irm.max()) <= 1.0
    assert torch.allclose(ideal_ratio_mask(s + 0.1, torch.zeros_like(s)), torch.ones_like(s))
    assert torch.allclose(ideal_ratio_mask(torch.zeros_like(n), n + 0.1), torch.zeros_like(n))


def test_adapter_irm_in_unit_interval(rng):
    ad = KielUNetAdapter(base_channels=4)
    ad.load_pretrained()
    clean = torch.from_numpy(rng.standard_normal((2, 8000))).float() * 0.1
    noisy = clean + torch.from_numpy(rng.standard_normal((2, 8000))).float() * 0.05
    _, _, irm = ad.irm_target(noisy, clean)
    assert irm.shape[1] == 257
    assert float(irm.min()) >= 0.0 and float(irm.max()) <= 1.0


@pytest.mark.parametrize("fs_in,fs_out", [(16000, 12000), (12000, 16000)])
def test_resample_matches_scipy(rng, fs_in, fs_out):
    x = rng.standard_normal(1001)
    ref = resample_poly(x, *(3, 4) if fs_out < fs_in else (4, 3))
    got = resample_torch(torch.from_numpy(x), fs_in, fs_out).numpy()
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("t", [100, 4000, 16007])
def test_enhance_returns_length_t(rng, t):
    ad = build("kiel_unet", base_channels=4)
    ad.load_pretrained()
    ad.eval()
    x = torch.from_numpy(rng.standard_normal((2, t))).float() * 0.1
    y = ad.enhance(x)
    assert y.shape == (2, t)
    assert torch.isfinite(y).all()
    assert ad.enhance(x[0]).shape == (t,)


def test_one_step_reduces_loss(rng):
    torch.manual_seed(0)
    ad = KielUNetAdapter(base_channels=4, crop_frames=None, lambda_mag=0.1)
    ad.load_pretrained()
    ad.train()
    t = 6000
    clean = torch.from_numpy(rng.standard_normal((2, t))).float() * 0.1
    noisy = clean + torch.from_numpy(rng.standard_normal((2, t))).float() * 0.1
    opt = torch.optim.Adam(ad.trainable_params(), lr=1e-3)
    l0 = ad.loss(noisy, clean)
    opt.zero_grad()
    l0.backward()
    opt.step()
    l1 = ad.loss(noisy, clean)
    assert torch.isfinite(l0) and l1.item() < l0.item()


def test_loss_with_crop_and_checkpoint_roundtrip(rng, tmp_path):
    torch.manual_seed(0)
    ad = KielUNetAdapter(base_channels=4, crop_frames=128)
    ad.load_pretrained()
    ad.train()
    x = torch.from_numpy(rng.standard_normal((1, 16000))).float() * 0.1
    assert torch.isfinite(ad.loss(x, 0.5 * x))
    ad.eval()
    path = str(tmp_path / "unet.pt")
    ad.save_checkpoint(path)
    ad2 = KielUNetAdapter(base_channels=4, checkpoint=path)
    ad2.load_pretrained()
    ad2.eval()
    assert torch.allclose(ad.enhance(x), ad2.enhance(x))


def test_output_band_limited_to_6khz():
    ad = KielUNetAdapter(base_channels=4)
    ad.load_pretrained()
    ad.eval()
    t = torch.arange(16000) / 16000
    x = (0.3 * torch.sin(2 * np.pi * 7000 * t)).unsqueeze(0)  # above 6 kHz
    y = ad.enhance(x)
    assert float(y[:, 2000:-2000].abs().max()) < 1e-3


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(8, 16)
        self.conv = nn.Conv1d(16, 16, 3, padding=1)
        self.gru = nn.GRU(16, 12, num_layers=2, batch_first=True)
        self.out = nn.Linear(12, 4)

    def forward(self, x):  # (B, T, 8)
        h = torch.relu(self.inp(x))
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.out(h)


PATTERNS = [r"^inp$", r"^conv$", r"^gru$", r"^out$"]
R = 4


def _toy_with_lora(seed: int = 0):
    torch.manual_seed(seed)
    model = Toy().eval()
    base = Toy().eval()
    base.load_state_dict(model.state_dict())
    # Freeze the reference too: on CPU, torch picks a different (oneDNN) GRU
    # kernel when no weight requires grad, which differs in the last ulp.
    # Same kernel on both sides -> the comparison tests LoRA, not kernel choice.
    for p in base.parameters():
        p.requires_grad_(False)
    params = apply_lora(model, PATTERNS, r=R, alpha=8, dropout=0.0)
    return base, model, params


def test_lora_identical_at_init():
    base, model, _ = _toy_with_lora()
    assert isinstance(model.gru, LoRAGRU)
    x = torch.randn(3, 10, 8)
    with torch.no_grad():
        assert torch.equal(model(x), base(x))


def test_lora_gru_alone_bit_identical():
    torch.manual_seed(1)
    gru = nn.GRU(5, 7, num_layers=2, bidirectional=True)
    lg = LoRAGRU.from_gru(gru, r=3)
    x = torch.randn(9, 2, 5)
    with torch.no_grad():
        a, ha = gru(x)
        b, hb = lg(x)
    assert torch.equal(a, b) and torch.equal(ha, hb)


def test_only_lora_trainable_and_count():
    base, model, params = _toy_with_lora()
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert len(params) == len(names)
    expected = sum(lora_param_count(getattr(base, n), R) for n in ("inp", "conv", "gru", "out"))
    # r*(in+out) for the Linear layers, spelled out
    assert lora_param_count(base.inp, R) == R * (8 + 16)
    assert count_trainable(model) == expected


def test_lora_gradients_flow():
    _, model, params = _toy_with_lora()
    model.train()
    x = torch.randn(2, 6, 8)
    model(x).square().mean().backward()
    assert all(p.grad is not None for p in params)
    named = dict(model.named_parameters())
    b_grads = [named[n].grad for n in named if "lora_B" in n]
    assert all(float(g.abs().sum()) > 0 for g in b_grads)
    assert all(not p.requires_grad and p.grad is None
               for n, p in model.named_parameters() if "lora_" not in n)


def _randomize_lora(model: nn.Module) -> None:
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))


def test_merge_equals_unmerged():
    _, model, _ = _toy_with_lora()
    _randomize_lora(model)
    x = torch.randn(3, 10, 8)
    with torch.no_grad():
        y_unmerged = model(x)
        merge_lora(model)
        y_merged = model(x)
        unmerge_lora(model)
        y_back = model(x)
    assert not torch.equal(y_unmerged, _toy_with_lora()[0](x))  # LoRA active
    torch.testing.assert_close(y_merged, y_unmerged, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(y_back, y_unmerged, rtol=1e-5, atol=1e-6)


def test_lora_state_dict_roundtrip(tmp_path):
    _, model, _ = _toy_with_lora()
    _randomize_lora(model)
    sd = lora_state_dict(model)
    assert sd and all("lora_" in k for k in sd)
    path = str(tmp_path / "lora.pt")
    save_lora(model, path, r=R)
    _, fresh, _ = _toy_with_lora()
    load_lora(fresh, path)
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        assert torch.equal(model(x), fresh(x))
    with pytest.raises(KeyError):
        load_lora(fresh, {"bogus.lora_A": torch.zeros(1)})


def test_lora_on_unet_conv2d():
    torch.manual_seed(0)
    ad = KielUNetAdapter(base_channels=4)
    ad.load_pretrained()
    ad.eval()
    x = torch.randn(1, 4000) * 0.1
    y0 = ad.enhance(x)
    apply_lora(ad.model, [r"^dec\.\d+\.[03]$"], r=2, alpha=4)
    params = list(ad.trainable_params("lora"))
    assert params and count_trainable(ad.model) == sum(p.numel() for p in params)
    assert torch.equal(ad.enhance(x), y0)


def test_apply_lora_no_match_raises():
    with pytest.raises(ValueError):
        apply_lora(Toy(), [r"^nothing$"])
