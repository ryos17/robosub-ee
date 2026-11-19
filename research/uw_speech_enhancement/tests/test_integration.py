"""Cross-module integration checks added by the integrator.

- the playback chirp (uwse.bench.playlist) and the alignment template
  (uwse.evaluation.align) are the same function (RESEARCH_PLAN §6.5);
- a bench timing JSON converts into the alignment layout exactly;
- kiel_unet LoRA mode inserts its own LoRA factors; a LoRA checkpoint cannot be
  silently half-loaded into a plain model (§7.2);
- C7 in-air pairs use the geometric direct-path index, as uwsim does (§8.2).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from uwse.bench import playlist as pl
from uwse.evaluation import align


def test_bench_chirp_is_align_chirp():
    a = pl.lfm_chirp()
    b = align.lfm_chirp(48000, amplitude=align.DEFAULT_CHIRP_AMPLITUDE)
    assert np.array_equal(a, b)
    assert abs(np.max(np.abs(a)) - 0.1) < 1e-3  # -20 dBFS peak
    x, _ = align.assemble_block([np.zeros(4800)], 48000)
    i = int(round(0.5 * 48000))
    assert np.array_equal(x[i : i + a.size], a)


def test_layout_from_timing_matches_bench(rng):
    items = [(f"u{k}", 0.05 * rng.standard_normal(16000 * (1 + k)), 16000) for k in range(3)]
    sig, t = pl.render_block(items)
    lay = align.layout_from_timing(t)
    fs = t["fs"]
    assert round(lay.chirp1_start_s * fs) == t["chirps"][0]["start"]
    assert round(lay.chirp2_start_s * fs) == t["chirps"][1]["start"]
    assert round(lay.total_s * fs) == t["n_samples"]
    for s0, u in zip(lay.utt_starts_s, t["utterances"]):
        assert round(s0 * fs) == u["start"]
    dets = align.detect_chirps(sig, fs, f_lo=t["chirp"]["f0_hz"], f_hi=t["chirp"]["f1_hz"])
    assert [d.index for d in dets] == [c["start"] for c in t["chirps"]]


def test_kiel_lora_mode_inserts_factors(tmp_path):
    pytest.importorskip("peft")
    from uwse.adapters import build

    se = build("kiel_unet")
    se.load_pretrained()
    n_full = sum(p.numel() for p in se.trainable_params("full"))
    ps = se.trainable_params("lora")
    n_lora = sum(p.numel() for p in ps)
    assert 0 < n_lora < 0.2 * n_full
    assert all(p.requires_grad for p in ps)
    # calling again must not wrap twice
    assert sum(p.numel() for p in se.trainable_params("lora")) == n_lora
    path = tmp_path / "lora.pt"
    se.save_checkpoint(str(path))
    fresh = build("kiel_unet")
    fresh.load_pretrained()
    with pytest.raises(RuntimeError):
        fresh.load_checkpoint(str(path))  # strict Kiel load: key mismatch is an error
    fresh.trainable_params("lora")
    fresh.load_checkpoint(str(path))


def test_base_load_checkpoint_rejects_lora_keys(tmp_path):
    from uwse.adapters.base import SEAdapter

    class Tiny(SEAdapter):
        name = "tiny"
        sample_rate = 16000

        def load_pretrained(self):
            self.model = torch.nn.Sequential(torch.nn.Linear(4, 4))

        def enhance(self, wav):
            return wav

        def loss(self, noisy, clean):
            return torch.zeros(())

    a = Tiny("cpu")
    a.load_pretrained()
    sd = {"0.base_layer.weight": torch.zeros(4, 4), "0.lora_A.default.weight": torch.zeros(2, 4)}
    torch.save({"model": sd}, tmp_path / "x.pt")
    with pytest.raises(RuntimeError, match="LoRA"):
        a.load_checkpoint(str(tmp_path / "x.pt"))


def test_inair_target_aligned_to_geometric_direct_path(rng):
    pytest.importorskip("pyroomacoustics")
    from uwse.train.datasets import InAirPairs

    fs = 16000
    clean = [0.1 * rng.standard_normal(fs)]
    ds = InAirPairs(clean, None, fs, fs_clean=fs, n_items=2, seed=3, segment_s=None,
                    room_dim_range=((6.0, 7.0), (5.0, 6.0), (3.0, 3.5)), t60_range=(0.3, 0.4),
                    distance_range=(2.0, 4.0), max_order=8, n_rirs=1, synth_tail=False)
    h, meta = ds._room_ir(np.random.default_rng(0), 0)
    p = meta["direct_index"]
    assert abs(p - (meta["distance"] / meta["c"] * fs + 40)) <= 0.5
    # the direct-path peak sits at the geometric index (pra fractional-delay centre)
    assert abs(int(np.argmax(np.abs(h[: p + 3]))) - p) <= 1
