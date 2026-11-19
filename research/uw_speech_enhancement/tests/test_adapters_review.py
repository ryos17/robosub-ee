"""Adversarial review tests for uwse.adapters (RESEARCH_PLAN.md §7.1-7.3)."""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

import uwse.adapters as A
from uwse.adapters.identity import chunked_apply, multires_stft_loss

FS = 16000


# --------------------------------------------------------------------------
# chunked_apply: a silent overlap must not zero the rest of the file
# --------------------------------------------------------------------------
def test_chunked_apply_match_gain_silent_overlap(rng):
    x = torch.from_numpy(rng.standard_normal((1, 10_000)).astype(np.float32))
    # chunk=3000, overlap=500 -> hop 2500; first overlap is [2500, 3000)
    x[:, 2500:3000] = 0.0
    y = chunked_apply(lambda s: s.clone(), x, chunk=3000, overlap=500, match_gain=True)
    assert torch.allclose(y, x, atol=1e-5), float((y - x).abs().max())


# --------------------------------------------------------------------------
# WPE: digital silence (all-zero item, e.g. batch padding) must stay finite
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,shape", [("wpe1", (2, FS)), ("wpe4", (2, 4, FS))])
def test_wpe_silent_input_is_finite(rng, name, shape):
    x = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
    x[1] = 0.0  # one all-zero item in the batch
    y = A.build(name, taps=10).enhance(x)
    assert torch.isfinite(y).all()
    assert torch.equal(y[1], torch.zeros_like(y[1]))
    assert y[0].abs().sum() > 0


# --------------------------------------------------------------------------
# multires_stft_loss: gradient must stay finite on exactly-zero output bins
# --------------------------------------------------------------------------
def test_multires_loss_finite_grad_on_silence(rng):
    est = torch.zeros(1, 4096, requires_grad=True)
    ref = torch.from_numpy(rng.standard_normal((1, 4096)).astype(np.float32))
    loss = multires_stft_loss(est, ref)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(est.grad).all()


# --------------------------------------------------------------------------
# MetricGAN+: the generator must be scored by the same discriminator that was
# trained (train-mode BatchNorm, as in the SpeechBrain recipe), not by the
# eval-mode running statistics that lag behind (BN momentum 0.01).
# --------------------------------------------------------------------------
class _FakeMGAdapter:
    """Minimal stand-in exposing what MetricGANTrainer uses (no download)."""

    sample_rate = FS
    device = torch.device("cpu")

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        win = torch.hamming_window(512)
        X = torch.stft(wav, 512, 256, window=win, return_complex=True)
        return torch.log1p(X.abs()).transpose(1, 2)  # (B, frames, F)

    def _forward(self, noisy):
        return 0.5 * noisy, self.features(0.5 * noisy)


def test_metricgan_generator_loss_uses_trained_discriminator_mode(rng):
    pytest.importorskip("speechbrain")
    from uwse.adapters.speechbrain_models import MetricGANTrainer

    torch.manual_seed(0)
    tr = MetricGANTrainer(_FakeMGAdapter())
    clean = torch.from_numpy(0.3 * rng.standard_normal((2, FS)).astype(np.float32))
    enh_spec = _FakeMGAdapter().features(0.5 * clean)
    clean_spec = _FakeMGAdapter().features(clean)
    tr.discriminator.train()
    with torch.no_grad():
        expected = tr._d(enh_spec, clean_spec)
    got_loss = tr.generator_loss(enh_spec, clean)
    exp_loss = torch.nn.functional.mse_loss(expected, torch.ones_like(expected))
    # train-mode spectral_norm does one power iteration per forward, so two
    # calls differ slightly (~1 %); eval-mode BN differs by ~25 % here
    assert torch.allclose(got_loss, exp_loss, rtol=0.03), (float(got_loss), float(exp_loss))


# --------------------------------------------------------------------------
# DFN3: no state leakage between calls / batch items (libdf is stateful)
# --------------------------------------------------------------------------
@pytest.mark.network
def test_dfn3_stateless_and_batch_consistent():
    try:
        se = A.build("dfn3", device="cpu")
        se.load_pretrained()
    except ImportError as e:
        pytest.skip(str(e))
    g = np.random.default_rng(1)
    x = torch.from_numpy((0.1 * g.standard_normal((2, 48000))).astype(np.float32))
    y_b = se.enhance(x)
    y0 = se.enhance(x[0])
    y0_again = se.enhance(x[0])
    y1 = se.enhance(x[1])
    assert torch.allclose(y0, y0_again, atol=1e-6)
    assert torch.allclose(y_b[0], y0, atol=1e-5) and torch.allclose(y_b[1], y1, atol=1e-5)
    del se
    gc.collect()
