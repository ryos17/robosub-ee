"""URGENT 2025 TF-GridNet baseline (zero-shot only).

Implements the "universal" C1 row of RESEARCH_PLAN.md §7.1/§7.3:
``kohei0209/tfgridnet_urgent25`` (8.5M parameters, CC BY 4.0, arXiv 2505.23212),
loaded with ESPnet's ``SeparateSpeech.from_pretrained``. ESPnet is a heavy,
optional dependency and is not part of the ``uwse`` env, so it is imported
lazily and a clear ImportError is raised when it is missing. The model is
sampling-frequency independent; we feed it 16 kHz (the plan's scoring rate).
No fine-tuning (zero-shot only in phase 1), so ``loss`` is not implemented.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .base import SEAdapter, register
from .identity import as_batch, chunked_apply, fit_length, pretrained_dir

_INSTALL_HINT = (
    "tfgridnet_urgent25 needs ESPnet: `pip install espnet espnet_model_zoo` "
    "(ESPnet pins its own torch range; prefer a separate env)."
)


@register("tfgridnet_urgent25")
class TFGridNetURGENTAdapter(SEAdapter):
    """Zero-shot TF-GridNet (URGENT 2025 challenge baseline)."""

    source = "kohei0209/tfgridnet_urgent25"
    sample_rate = 16000

    def __init__(
        self,
        device: str | torch.device = "cpu",
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 1.0,
    ) -> None:
        super().__init__(device)
        self.separator = None
        self.chunk = int(chunk_seconds * self.sample_rate)
        self.overlap = int(overlap_seconds * self.sample_rate)

    def load_pretrained(self) -> None:
        try:
            from espnet2.bin.enh_inference import SeparateSpeech
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e
        import os

        # keep the HF download under pretrained_models/ (no symlinks needed)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        cache = str(pretrained_dir(self.name))
        try:
            self.separator = SeparateSpeech.from_pretrained(
                self.source, device=str(self.device), normalize_output_wav=True, cache_dir=cache
            )
        except TypeError:
            self.separator = SeparateSpeech.from_pretrained(
                self.source, device=str(self.device), normalize_output_wav=True
            )
        self.model = self.separator.enh_model
        self.model.eval()

    def to(self, device):
        super().to(device)
        if self.separator is not None:
            self.separator.device = str(self.device)
        return self

    def _run(self, wav: Tensor) -> Tensor:
        out = self.separator(wav.detach().cpu().numpy(), fs=self.sample_rate)  # list[(B, T)]
        y = torch.as_tensor(np.asarray(out[0]), dtype=torch.float32)
        return fit_length(y, wav.shape[-1])

    def enhance(self, wav: Tensor) -> Tensor:
        if self.separator is None:
            self.load_pretrained()
        wav, was_1d = as_batch(wav)
        with torch.no_grad():
            # normalize_output_wav peak-normalizes every call, so chunk gains differ
            y = chunked_apply(self._run, wav.float().cpu(), self.chunk, self.overlap, match_gain=True)
        y = y.to(wav.device)
        return y[0] if was_1d else y

    def trainable_params(self, mode: str = "full"):
        raise NotImplementedError(f"{self.name} is zero-shot only (RESEARCH_PLAN.md §7.1)")
