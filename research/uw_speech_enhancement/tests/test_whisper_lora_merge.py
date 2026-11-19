"""C8 eval path: a peft LoRA on HF Whisper, merged into openai-whisper, gives the same logits."""
from __future__ import annotations

import os

import pytest
import torch

whisper = pytest.importorskip("whisper")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from uwse.evaluation.asr import WhisperASR, lora_deltas_openai  # noqa: E402

OPENAI_TINY = os.path.expanduser("~/.cache/whisper/tiny.pt")


def _hf_tiny():
    from transformers import WhisperForConditionalGeneration

    try:
        return WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", local_files_only=True)
    except OSError:
        pytest.skip("openai/whisper-tiny not in the HF cache")


@pytest.fixture(scope="module")
def models():
    if not os.path.isfile(OPENAI_TINY):
        pytest.skip("openai-whisper tiny.pt not cached")
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    torch.manual_seed(0)
    hf = get_peft_model(_hf_tiny().eval(), LoraConfig(r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
    for n, p in hf.named_parameters():  # make B non-zero so the delta matters
        if "lora_B" in n:
            p.data.normal_(0, 0.02)
    hf.eval()
    state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(hf).items()}
    return hf, state


def _logits_openai(m, mel, tokens):
    with torch.no_grad():
        return m.logits(tokens, m.embed_audio(mel))


def test_deltas_cover_every_lora_pair(models):
    _, state = models
    d = lora_deltas_openai(state, 4, 16)
    assert len(d) == sum("lora_A" in k for k in state)
    assert any(".cross_attn.query." in k for k in d) and any("encoder.blocks.0.attn.value" in k for k in d)


def test_merged_openai_matches_hf_peft_logits(models, tmp_path):
    hf, state = models
    mel = torch.randn(1, 80, 3000)
    tokens = torch.tensor([[50258, 50259, 50359, 50363, 440, 2068]])
    with torch.no_grad():
        ref = hf(input_features=mel, decoder_input_ids=tokens).logits
    ck = tmp_path / "lora.pt"
    torch.save({"model": state, "lora": {"r": 4, "alpha": 16}}, ck)
    asr = WhisperASR("tiny", device="cpu", fp16=False, lora_ckpt=str(ck))
    got = _logits_openai(asr.model, mel, tokens)
    base = _logits_openai(whisper.load_model("tiny", device="cpu"), mel, tokens)
    assert (ref - base).abs().max() > 1e-2  # the LoRA changes the output
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=2e-3)
