"""The one interface every speech-enhancement model is wrapped behind.

`train` and `evaluation` only ever see an `SEAdapter`, so adding a model means
writing one subclass and registering it; nothing else changes.

Conventions
-----------
- Audio tensors are float32, shape (B, T), mono, at `sample_rate`, roughly in
  [-1, 1]. Multichannel adapters (WPE-4ch) take (B, C, T) and return (B, T).
- `enhance` must return exactly T samples (pad/trim internally).
- `loss` returns a scalar tensor to minimize; it should be the model's native
  training loss where practical (RESEARCH_PLAN.md §7.2).
- Adapters without trainable parameters (identity, WPE) raise
  NotImplementedError from `loss` and return an empty iterable from
  `trainable_params`.
"""

from __future__ import annotations

import abc
from typing import Callable, Iterable

import torch
from torch import Tensor, nn


class SEAdapter(abc.ABC):
    """Base class for all enhancement systems (neural or classical)."""

    #: registry key, e.g. "dfn3", "metricganplus"
    name: str = ""
    #: model's native sample rate in Hz
    sample_rate: int = 16000
    #: True if `enhance` expects (B, C, T) multichannel input
    multichannel: bool = False

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model: nn.Module | None = None

    # -- lifecycle ---------------------------------------------------------
    @abc.abstractmethod
    def load_pretrained(self) -> None:
        """Download/load the pretrained checkpoint into `self.model`."""

    def load_checkpoint(self, path: str) -> None:
        """Load fine-tuned weights saved by `save_checkpoint`."""
        assert self.model is not None, "call load_pretrained() first"
        state = torch.load(path, map_location=self.device)
        res = self.model.load_state_dict(state["model"], strict=False)
        lora_keys = [k for k in res.unexpected_keys if "lora_" in k or "base_layer" in k]
        if lora_keys:
            # a LoRA-trained checkpoint: the LoRA factors must exist before loading
            raise RuntimeError(
                f"{self.name}: checkpoint {path} holds LoRA keys (e.g. {lora_keys[0]!r}); "
                "call trainable_params('lora') before load_checkpoint")
        if res.missing_keys or res.unexpected_keys:
            import warnings

            warnings.warn(
                f"{self.name}: load_checkpoint({path}) missing={res.missing_keys[:5]} "
                f"unexpected={res.unexpected_keys[:5]} (strict=False)", stacklevel=2)

    def save_checkpoint(self, path: str, **extra) -> None:
        assert self.model is not None
        torch.save({"model": self.model.state_dict(), "adapter": self.name, **extra}, path)

    def to(self, device: str | torch.device) -> "SEAdapter":
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def train(self, mode: bool = True) -> "SEAdapter":
        if self.model is not None:
            self.model.train(mode)
        return self

    def eval(self) -> "SEAdapter":
        return self.train(False)

    # -- inference / training ---------------------------------------------
    @abc.abstractmethod
    def enhance(self, wav: Tensor) -> Tensor:
        """(B, T) -> (B, T) at `sample_rate` (or (B, C, T) -> (B, T))."""

    def forward_train(self, noisy: Tensor) -> Tensor:
        """Differentiable forward used inside `loss`. Defaults to `enhance`."""
        return self.enhance(noisy)

    def loss(self, noisy: Tensor, clean: Tensor) -> Tensor:
        """Scalar training loss for a batch of (noisy, clean) at `sample_rate`."""
        raise NotImplementedError(f"{self.name} is not trainable")

    def trainable_params(self, mode: str = "full") -> Iterable[nn.Parameter]:
        """Parameters to optimize. mode: "full" | "lora"."""
        if self.model is None:
            return []
        if mode == "full":
            return [p for p in self.model.parameters() if p.requires_grad]
        raise NotImplementedError(f"{self.name}: mode {mode!r} not supported")


# -- registry ---------------------------------------------------------------
_REGISTRY: dict[str, Callable[..., SEAdapter]] = {}


def register(name: str):
    """Class decorator: `@register("dfn3")`."""

    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def build(name: str, **kwargs) -> SEAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
