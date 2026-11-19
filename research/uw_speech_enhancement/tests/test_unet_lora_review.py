"""Adversarial-review tests for uwse.adapters.kiel_unet and uwse.adapters.lora."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import resample_poly
from torch import nn

from uwse.adapters.kiel_unet import KielUNetAdapter
from uwse.adapters.lora import LoRAGRU, apply_lora, load_lora, lora_state_dict, save_lora


# ---------------------------------------------------------------------------
# Kiel U-Net signal path
# ---------------------------------------------------------------------------
class _OnesNet(nn.Module):
    def forward(self, x):
        return torch.ones_like(x)


def test_unit_mask_reconstructs_lowpassed_input(rng):
    """With mask = 1 the path must be a pure 0-6 kHz low-pass: no delay, no gain error."""
    ad = KielUNetAdapter()
    ad.model = _OnesNet()
    x = rng.standard_normal(16000).astype(np.float32) * 0.1
    y = ad.enhance(torch.from_numpy(x)[None])[0].numpy()
    # reference: the same 16 -> 12 -> 16 kHz round trip in scipy
    ref = resample_poly(resample_poly(x.astype(np.float64), 3, 4), 4, 3)[: x.size]
    core = slice(1000, -1000)
    rel = lambda a: np.sqrt(np.mean((a[core] - ref[core]) ** 2) / np.mean(ref[core] ** 2))  # noqa: E731
    # the only deviation is the documented zero mask on the 6 kHz (Nyquist) bin
    assert rel(y) < 0.03, rel(y)
    ad.estimate_mask = lambda spec: torch.ones(spec.shape)  # also pass the Nyquist bin
    y_all = ad.enhance(torch.from_numpy(x)[None])[0].numpy()
    assert rel(y_all) < 1e-5, rel(y_all)
    ad = KielUNetAdapter()
    ad.model = _OnesNet()
    # and a 1 kHz tone passes with unit gain and zero lag
    t = np.arange(16000) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    yt = ad.enhance(torch.from_numpy(tone))
    np.testing.assert_allclose(yt.numpy()[core], tone[core], atol=3e-3)


def test_irm_target_is_beta_half_on_known_mix():
    """IRM of a tone at 1 kHz with noise in a disjoint band is ~1 at 1 kHz, ~0 at noise band."""
    ad = KielUNetAdapter()
    fs = 16000
    t = torch.arange(fs) / fs
    clean = 0.3 * torch.sin(2 * np.pi * 1000 * t)
    noise = 0.3 * torch.sin(2 * np.pi * 4000 * t)
    _, _, irm = ad.irm_target((clean + noise)[None], clean[None])
    hz_per_bin = ad.internal_sr / ad.n_fft
    k1, k4 = round(1000 / hz_per_bin), round(4000 / hz_per_bin)
    mid = irm.shape[-1] // 2
    assert float(irm[0, k1, mid]) > 0.99
    assert float(irm[0, k4, mid]) < 0.01


# ---------------------------------------------------------------------------
# LoRA: BatchNorm running statistics must not drift while "frozen"
# ---------------------------------------------------------------------------
def test_lora_checkpoint_reproduces_trained_unet(tmp_path):
    """Base weights + saved LoRA factors must reproduce the LoRA-trained model.

    A LoRA run in train mode must not silently change base state (BatchNorm
    running statistics) that `lora_state_dict` does not save.
    """
    torch.manual_seed(0)
    ad = KielUNetAdapter(base_channels=4, crop_frames=None)
    ad.load_pretrained()
    base_sd = {k: v.clone() for k, v in ad.model.state_dict().items()}
    apply_lora(ad.model, [r"^dec\.\d+\.[03]$"], r=2, alpha=4)
    ad.train()
    opt = torch.optim.Adam(ad.trainable_params("lora"), lr=1e-2)
    x = torch.randn(2, 6000) * 0.1
    for _ in range(3):
        opt.zero_grad()
        ad.loss(x + 0.3 * torch.randn_like(x), x).backward()
        opt.step()
    ad.eval()
    y_trained = ad.enhance(x)
    path = str(tmp_path / "lora.pt")
    save_lora(ad.model, path)

    ad2 = KielUNetAdapter(base_channels=4)
    ad2.load_pretrained()
    ad2.model.load_state_dict(base_sd)
    apply_lora(ad2.model, [r"^dec\.\d+\.[03]$"], r=2, alpha=4)
    load_lora(ad2.model, path)
    ad2.eval()
    torch.testing.assert_close(ad2.enhance(x), y_trained)


def test_load_lora_accepts_loaded_save_dict(tmp_path):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 4))
    apply_lora(m, [r"^0$"], r=2)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_()
    path = str(tmp_path / "l.pt")
    save_lora(m, path, r=2)
    torch.manual_seed(0)
    m2 = nn.Sequential(nn.Linear(4, 4))
    apply_lora(m2, [r"^0$"], r=2)
    load_lora(m2, torch.load(path, weights_only=True))  # the dict save_lora wrote
    for k, v in lora_state_dict(m).items():
        assert torch.equal(lora_state_dict(m2)[k], v)


def test_apply_lora_warns_on_skipped_leaf():
    m = nn.Sequential(nn.Linear(4, 4), nn.LSTM(4, 4))
    with pytest.warns(UserWarning, match="LSTM"):
        apply_lora(m, [r"^\d$"], r=2)


# ---------------------------------------------------------------------------
# LoRAGRU on CUDA (cuDNN path differs from CPU)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("targets", [("ih", "hh"), ("ih",)])
def test_lora_gru_cuda_forward_backward(targets):
    torch.manual_seed(0)
    gru = nn.GRU(6, 8, num_layers=2, batch_first=True).cuda()
    for p in gru.parameters():
        p.requires_grad_(False)
    lg = LoRAGRU.from_gru(gru, r=2, targets=targets)
    x = torch.randn(3, 11, 6, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(lg(x)[0], gru(x)[0])
    for _ in range(2):  # second iteration catches state left over by the first
        out, _ = lg(x)
        out.square().mean().backward()
    assert all(getattr(lg, f"lora_B_{s}").grad is not None for s in lg._lora_map.values())


# ---------------------------------------------------------------------------
# AMP: uwse.train.loop wraps adapter.loss in torch.autocast when cfg.amp is set
# ---------------------------------------------------------------------------
def test_dsp_path_unaffected_by_autocast(rng):
    """Resampling/STFT/IRM must stay float32 under autocast; only the net may drop precision."""
    torch.manual_seed(0)
    ad = KielUNetAdapter(base_channels=4, crop_frames=None)
    ad.load_pretrained()
    ad.eval()
    clean = torch.from_numpy(rng.standard_normal((1, 8000))).float() * 0.5
    noisy = clean + torch.from_numpy(rng.standard_normal((1, 8000))).float() * 0.5
    spec_ref, _, irm_ref = ad.irm_target(noisy, clean)
    l_ref = ad.loss(noisy, clean)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        spec, _, irm = ad.irm_target(noisy, clean)
        loss = ad.loss(noisy, clean)
        y = ad.enhance(noisy)
    assert spec.dtype == torch.complex64 and irm.dtype == torch.float32
    torch.testing.assert_close(irm, irm_ref)
    assert y.dtype == torch.float32 and y.shape == noisy.shape
    assert torch.isfinite(loss) and abs(loss.item() - l_ref.item()) < 0.05 * l_ref.item()


def test_enhance_in_train_mode_does_not_touch_batchnorm_stats(rng):
    """enhance() is inference: it must not update BN running stats or depend on batch mates."""
    torch.manual_seed(0)
    ad = KielUNetAdapter(base_channels=4)
    ad.load_pretrained()
    ad.train()
    before = {k: v.clone() for k, v in ad.model.state_dict().items() if "running" in k}
    x = torch.from_numpy(rng.standard_normal((2, 4000))).float() * 0.1
    y_pair = ad.enhance(x)
    after = {k: v for k, v in ad.model.state_dict().items() if "running" in k}
    assert all(torch.equal(before[k], after[k]) for k in before)
    torch.testing.assert_close(ad.enhance(x[:1]), y_pair[:1])
    assert ad.model.training  # mode restored
