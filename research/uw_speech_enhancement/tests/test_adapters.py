"""Tests for uwse.adapters (RESEARCH_PLAN.md §7.1-7.3).

Fast tests: registry, C0 identity, WPE shapes and dereverberation, helpers.
Network tests (`-m network`) download each pretrained model, enhance 2 s of
noise + tone, and check that forward_train + loss backpropagates.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

import uwse.adapters as A
from uwse.adapters.identity import chunked_apply, fit_length, si_snr

FS = 16000
EXPECTED = {
    "identity",
    "wpe1",
    "wpe4",
    "metricganplus",
    "sepformer_wham16k",
    "sepformer_whamr",
    "dfn3",
    "tfgridnet_urgent25",
}


def test_registry_lists_adapters():
    names = set(A.available())
    assert EXPECTED <= names, (names, A.import_errors)
    with pytest.raises(KeyError):
        A.build("no_such_model")


def test_identity_shapes(rng):
    x = torch.from_numpy(rng.standard_normal((2, FS)).astype(np.float32))
    se = A.build("identity")
    assert torch.equal(se.enhance(x), x)
    mc = torch.from_numpy(rng.standard_normal((2, 4, FS)).astype(np.float32))
    se = A.build("identity", scored_channel=2)
    y = se.enhance(mc)
    assert y.shape == (2, FS) and torch.equal(y, mc[:, 2])
    assert se.enhance(x[0]).shape == (FS,)
    with pytest.raises(NotImplementedError):
        se.loss(x, x)
    assert list(se.trainable_params()) == []


@pytest.mark.parametrize("T", [FS, FS + 77])
def test_wpe_shapes(rng, T):
    x1 = torch.from_numpy(rng.standard_normal((2, T)).astype(np.float32))
    y1 = A.build("wpe1", taps=10).enhance(x1)
    assert y1.shape == (2, T) and y1.dtype == torch.float32 and torch.isfinite(y1).all()
    x4 = torch.from_numpy(rng.standard_normal((2, 4, T)).astype(np.float32))
    se4 = A.build("wpe4", taps=10, scored_channel=1)
    assert se4.multichannel
    y4 = se4.enhance(x4)
    assert y4.shape == (2, T) and torch.isfinite(y4).all()
    with pytest.raises(ValueError):
        se4.enhance(x4[:, :3])


def _reverb_scene(rng, n_ch: int, seconds: float = 6.0, t60: float = 0.6):
    """Noise bursts (0.25 s on, 0.5 s off) through exponentially decaying RIRs.

    Returns (reverberant (C, T), mask of samples in the late tail after each
    burst ends, excluding the first 50 ms which WPE keeps by design (delay)).
    """
    T = int(seconds * FS)
    src = np.zeros(T)
    tail = np.zeros(T, dtype=bool)
    on, off = int(0.25 * FS), int(0.5 * FS)
    start = 0
    while start + on + off <= T:
        src[start : start + on] = rng.standard_normal(on)
        tail[start + on + int(0.05 * FS) : start + on + off] = True
        start += on + off
    L = int(t60 * FS)
    t = np.arange(L) / FS
    env = 10 ** (-3 * t / t60)  # -60 dB at t60
    out = []
    for _ in range(n_ch):
        h = 0.1 * rng.standard_normal(L) * env
        h[0] = 0.0
        h[int(rng.integers(0, 20))] = 4.0  # direct path dominates early part
        out.append(np.convolve(src, h)[:T])
    y = np.stack(out)
    return y / np.abs(y).max(), tail


# Single-channel WPE on a dense, diffuse synthetic tail removes only ~1 dB of
# late energy (measured: 0.8 dB at T60 0.6 s, 30 taps); 4-channel WPE removes
# several dB even with 10 taps (4.3 dB). Thresholds sit at about half of that.
@pytest.mark.parametrize("name,n_ch,taps,min_gain_db", [("wpe1", 1, 30, 0.4), ("wpe4", 4, 10, 2.0)])
def test_wpe_reduces_reverb_tail(rng, name, n_ch, taps, min_gain_db):
    y, tail = _reverb_scene(rng, n_ch, seconds=4.0)
    x = torch.from_numpy(y.astype(np.float32))
    x = x.unsqueeze(0) if name == "wpe4" else x[:1]
    z = A.build(name, taps=taps, delay=3, iterations=3).enhance(x)[0].numpy()
    ref = y[0]
    # energy of the late tail relative to the whole signal, in dB
    e_in = 10 * np.log10(np.sum(ref[tail] ** 2) / np.sum(ref**2))
    e_out = 10 * np.log10(np.sum(z[tail] ** 2) / np.sum(z**2))
    assert e_out < e_in - min_gain_db, (e_in, e_out)


def test_chunked_apply_reconstructs(rng):
    x = torch.from_numpy(rng.standard_normal((2, 10_000)).astype(np.float32))
    y = chunked_apply(lambda s: s * 1.0, x, chunk=3000, overlap=500)
    assert torch.allclose(y, x, atol=1e-5)
    y = chunked_apply(lambda s: s, x, chunk=20_000, overlap=500)
    assert torch.equal(y, x)
    # scale-invariant models: arbitrary gain and sign per call
    g = iter([1.0, -3.0, 0.5, 2.0, -1.0, 4.0])
    y = chunked_apply(lambda s: s * next(g), x, chunk=3000, overlap=500, match_gain=True)
    assert torch.allclose(y, x, atol=1e-4)


def test_helpers(rng):
    x = torch.randn(3, 100)
    assert fit_length(x, 80).shape == (3, 80) and fit_length(x, 120).shape == (3, 120)
    s = torch.from_numpy(rng.standard_normal((1, 8000)).astype(np.float32))
    assert si_snr(3.0 * s, s).item() > 60  # scale invariant
    noisy = s + torch.from_numpy(rng.standard_normal((1, 8000)).astype(np.float32))
    assert abs(si_snr(noisy, s).item()) < 1.0  # 0 dB SNR


# --------------------------------------------------------------------------
# network: pretrained models
# --------------------------------------------------------------------------
def _noise_tone(fs: int, seconds: float = 2.0, batch: int = 2, seed: int = 0):
    g = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    clean = np.stack([tone * (1 + 0.1 * k) for k in range(batch)])
    noisy = clean + 0.05 * g.standard_normal(clean.shape)
    return torch.from_numpy(noisy.astype(np.float32)), torch.from_numpy(clean.astype(np.float32))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _grad_step(se, mode: str) -> int:
    """Loss + backward on noise+tone; returns #params of `mode` with nonzero grad."""
    noisy, clean = _noise_tone(se.sample_rate)
    se.train()
    params = list(se.trainable_params(mode))
    assert params, f"{se.name}: no trainable params for mode {mode!r}"
    loss = se.loss(noisy.to(se.device), clean.to(se.device))
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    se.eval()
    return sum(1 for p in params if p.grad is not None and p.grad.abs().sum() > 0)


def _check_model(name: str, lora_modes: tuple[str, ...] = ("lora",), **kw):
    try:
        se = A.build(name, device=_device(), **kw)
        se.load_pretrained()
    except ImportError as e:
        pytest.skip(f"{name}: {e}")
    fs = se.sample_rate
    noisy, clean = _noise_tone(fs)
    y = se.enhance(noisy)
    assert y.shape == noisy.shape
    assert torch.isfinite(y).all()
    assert _grad_step(se, "full") > 0, f"{name}: no parameter received a gradient"
    for mode in lora_modes:
        se_l = A.build(name, device=_device(), **kw)
        se_l.load_pretrained()
        assert _grad_step(se_l, mode) > 0, f"{name}/{mode}: no LoRA parameter received a gradient"
        del se_l
    del se
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.network
@pytest.mark.parametrize("loss_name", ["metricgan", "spectral_mse"])
def test_metricganplus(loss_name):
    _check_model("metricganplus", loss_name=loss_name)


@pytest.mark.network
def test_sepformer_wham16k():
    _check_model("sepformer_wham16k")


@pytest.mark.network
def test_sepformer_whamr():
    _check_model("sepformer_whamr")


@pytest.mark.network
def test_dfn3():
    _check_model("dfn3", lora_modes=("lora", "lora_nogru"))


@pytest.mark.network
def test_tfgridnet_zero_shot():
    try:
        se = A.build("tfgridnet_urgent25", device=_device())
        se.load_pretrained()
    except ImportError as e:
        pytest.skip(str(e))
    noisy, _ = _noise_tone(se.sample_rate)
    y = se.enhance(noisy)
    assert y.shape == noisy.shape and torch.isfinite(y).all()
