"""LoRA utilities shared by every trainable adapter.

Implements the LoRA insertion plan of RESEARCH_PLAN.md §7.2 (column "LoRA
insertion") used by conditions C3 (sim LoRA, r in {4, 8, 16}) and C8, with the
§7.4 starting points (AdamW, wd 0, lr 1e-3 at r = 8).

- ``nn.Linear`` / ``nn.Conv1d`` / ``nn.Conv2d``: ``peft`` LoRA layers, injected
  with ``peft.inject_adapter_in_model`` (no PeftModel wrapper, so adapters keep
  calling the model exactly as before). Adapter name is ``"default"``.
- ``nn.GRU``: ``peft`` has no recurrent LoRA, so modules matching a pattern are
  replaced by :class:`LoRAGRU`, a ``nn.GRU`` subclass that adds
  ``scaling * B @ A`` to each targeted ``weight_ih_l*`` / ``weight_hh_l*``
  before the fused (cuDNN) GRU kernel runs. The base parameter names are
  unchanged, so base checkpoints still load. With B = 0 at init the effective
  weights are exactly W + 0, i.e. bit-identical to the base GRU.

Design note: we re-implement the delta instead of using
``torch.nn.utils.parametrize`` because parametrize renames the base weights
(``parametrizations.weight_ih_l0.original``), which would break loading the
pretrained state dicts. LoRA dropout acts on the layer *input* and therefore
cannot be expressed as a weight delta; it is applied to the peft layers only
and ignored for GRUs (documented, not silently different from peft).

Parameter counts (groups = 1): Linear r*(in+out); Conv r*(in*prod(k)+out);
GRU per layer and direction r*(in+3H) for weight_ih and r*(H+3H) for weight_hh.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "LoRAGRU",
    "apply_lora",
    "count_trainable",
    "count_params",
    "merge_lora",
    "unmerge_lora",
    "lora_state_dict",
    "save_lora",
    "load_lora",
    "lora_param_count",
]

_PEFT_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d)


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------
class LoRAGRU(nn.GRU):
    """nn.GRU with low-rank deltas on its input/hidden weight matrices.

    Effective weight: W + (alpha / r) * B @ A, A in R^{r x in}, B in R^{out x r}.
    A is Kaiming-uniform initialised (as peft), B = 0.
    """

    def __init__(
        self,
        *args,
        r: int = 8,
        lora_alpha: float = 16.0,
        targets: Sequence[str] = ("ih", "hh"),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if r <= 0:
            raise ValueError("r must be positive")
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False
        self._lora_active = False
        #: base weight name -> LoRA suffix, e.g. "weight_ih_l0" -> "ih_l0"
        self._lora_map: dict[str, str] = {}
        for name in self._flat_weights_names:
            if not name.startswith("weight_"):
                continue
            suffix = name[len("weight_"):]
            if suffix.split("_")[0] not in targets:
                continue
            w = getattr(self, name)
            out_f, in_f = w.shape
            self.register_parameter(f"lora_A_{suffix}", nn.Parameter(w.new_empty(r, in_f)))
            self.register_parameter(f"lora_B_{suffix}", nn.Parameter(w.new_zeros(out_f, r)))
            self._lora_map[name] = suffix
        self.reset_lora_parameters()

    @classmethod
    def from_gru(cls, gru: nn.GRU, r: int = 8, lora_alpha: float = 16.0,
                 targets: Sequence[str] = ("ih", "hh")) -> "LoRAGRU":
        """Wrap an existing GRU, copying its weights, device and dtype."""
        w = gru.weight_ih_l0
        new = cls(
            gru.input_size, gru.hidden_size, num_layers=gru.num_layers, bias=gru.bias,
            batch_first=gru.batch_first, dropout=gru.dropout, bidirectional=gru.bidirectional,
            r=r, lora_alpha=lora_alpha, targets=targets, device=w.device, dtype=w.dtype,
        )
        missing, unexpected = new.load_state_dict(gru.state_dict(), strict=False)
        assert not unexpected and all(k.startswith("lora_") for k in missing)
        new.train(gru.training)
        return new

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if getattr(self, "_lora_map", None):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        for suffix in self._lora_map.values():
            nn.init.kaiming_uniform_(getattr(self, f"lora_A_{suffix}"), a=math.sqrt(5))
            nn.init.zeros_(getattr(self, f"lora_B_{suffix}"))

    def delta(self, name: str) -> Tensor:
        """scaling * B @ A for base weight `name`."""
        s = self._lora_map[name]
        return (getattr(self, f"lora_B_{s}") @ getattr(self, f"lora_A_{s}")) * self.scaling

    def _effective(self, name: str) -> Tensor:
        w = getattr(self, name)
        if self.merged or name not in self._lora_map:
            return w
        return w + self.delta(name)

    def _update_flat_weights(self) -> None:
        # nn.GRU.forward calls this first and then hands self._flat_weights to
        # the fused kernel; during our forward we substitute W + delta.
        if self._lora_active:
            self._flat_weights = [self._effective(n) for n in self._flat_weights_names]
        else:
            super()._update_flat_weights()

    def forward(self, input, hx=None):  # noqa: A002 - mirror nn.GRU
        self._lora_active = True
        try:
            with warnings.catch_warnings():
                # W + delta are separate tensors, so cuDNN warns about
                # non-contiguous weights; expected and harmless.
                warnings.filterwarnings("ignore", message=".*contiguous chunk of memory.*")
                return super().forward(input, hx)
        finally:
            self._lora_active = False
            self._init_flat_weights()  # drop graph-carrying tensors

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        for name in self._lora_map:
            getattr(self, name).add_(self.delta(name))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        for name in self._lora_map:
            getattr(self, name).sub_(self.delta(name))
        self.merged = False

    def extra_repr(self) -> str:
        return super().extra_repr() + f", lora_r={self.r}, lora_alpha={self.lora_alpha}"


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------
def _match(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def _set_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def apply_lora(
    model: nn.Module,
    target_patterns: Sequence[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    gru_targets: Sequence[str] = ("ih", "hh"),
    freeze_norm_stats: bool = True,
) -> list[nn.Parameter]:
    """Insert LoRA into `model` in place, freeze everything else.

    Parameters
    ----------
    model : the module to modify (in place).
    target_patterns : regexes, ``re.search``-ed against names from
        ``model.named_modules()``. Matching Linear/Conv1d/Conv2d get peft LoRA,
        matching GRUs become :class:`LoRAGRU`; other matched types are ignored.
    r, alpha, dropout : LoRA rank, alpha (scaling = alpha / r) and input
        dropout (peft layers only; see module docstring).
    gru_targets : which GRU matrices get a delta: "ih", "hh" or both.
    freeze_norm_stats : set ``momentum = 0`` on every BatchNorm module so its
        running mean/var stay at the base values in train mode. Without this a
        "frozen" base still drifts through its BN buffers, which
        :func:`lora_state_dict` does not save, so base + LoRA file would not
        reproduce the trained model and the forgetting test (§7.2) would be
        confounded. Train-mode BN still normalises with batch statistics.

    Returns
    -------
    The trainable (LoRA) parameters, ready for an optimizer.
    """
    selected = [(n, m) for n, m in model.named_modules() if n and _match(n, target_patterns)]
    peft_names = [n for n, m in selected if isinstance(m, _PEFT_TYPES)]
    gru_names = [n for n, m in selected if type(m) is nn.GRU]
    skipped = [
        f"{n} ({type(m).__name__})"
        for n, m in selected
        if n not in peft_names and n not in gru_names
        and any(True for _ in m.parameters(recurse=False))
        and not isinstance(m, LoRAGRU) and not _is_peft_layer(m)
    ]
    if skipped:
        warnings.warn(f"apply_lora: matched modules of unsupported type left frozen: {skipped}",
                      stacklevel=2)
    if not peft_names and not gru_names:
        raise ValueError(f"no Linear/Conv1d/Conv2d/GRU module matches {list(target_patterns)}")

    for p in model.parameters():
        p.requires_grad_(False)
    if freeze_norm_stats:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
                # running = (1 - momentum) * running + momentum * batch -> unchanged
                m.momentum = 0.0

    for n in gru_names:
        _set_submodule(model, n, LoRAGRU.from_gru(model.get_submodule(n), r, alpha, gru_targets))

    if peft_names:
        from peft import LoraConfig, inject_adapter_in_model

        # a single anchored regex string -> peft uses re.fullmatch: exact names only
        target = "^(?:" + "|".join(re.escape(n) for n in peft_names) + ")$"
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=target, bias="none")
        inject_adapter_in_model(cfg, model, adapter_name="default")

    trainable = []
    for n, p in model.named_parameters():
        is_lora = "lora_" in n
        p.requires_grad_(is_lora)
        if is_lora:
            trainable.append(p)
    return trainable


# ---------------------------------------------------------------------------
# counting, merging, (de)serialisation
# ---------------------------------------------------------------------------
def count_trainable(model: nn.Module) -> int:
    """Number of scalar parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) scalar parameter counts."""
    return count_trainable(model), sum(p.numel() for p in model.parameters())


def lora_param_count(module: nn.Module, r: int) -> int:
    """Expected LoRA parameter count for a *base* Linear/Conv1d/Conv2d/GRU module.

    GRU count assumes both "ih" and "hh" are targeted.
    """
    if isinstance(module, nn.Linear):
        return r * (module.in_features + module.out_features)
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        k = math.prod(module.kernel_size)
        return r * (module.in_channels // module.groups * k + module.out_channels)
    if isinstance(module, nn.GRU):
        h = module.hidden_size
        d = 2 if module.bidirectional else 1
        total = 0
        for layer in range(module.num_layers):
            i = module.input_size if layer == 0 else h * d
            total += d * (r * (i + 3 * h) + r * (h + 3 * h))
        return total
    raise TypeError(f"unsupported module type {type(module).__name__}")


def _is_peft_layer(m: nn.Module) -> bool:
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:  # pragma: no cover
        return False
    return isinstance(m, LoraLayer)


def _lora_layers(model: nn.Module) -> Iterable[nn.Module]:
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:  # pragma: no cover - peft is a hard dependency of the env
        LoraLayer = ()  # type: ignore[assignment]
    for m in model.modules():
        if isinstance(m, LoRAGRU) or (LoraLayer and isinstance(m, LoraLayer)):
            yield m


def merge_lora(model: nn.Module) -> None:
    """Fold every LoRA delta into its base weight (inference speed-up)."""
    for m in _lora_layers(model):
        if not getattr(m, "merged", False):
            m.merge()


def unmerge_lora(model: nn.Module) -> None:
    """Undo :func:`merge_lora`."""
    for m in _lora_layers(model):
        if getattr(m, "merged", False):
            m.unmerge()


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Only the LoRA factors (keys containing "lora_"), detached on CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def save_lora(model: nn.Module, path: str, **meta) -> None:
    torch.save({"lora": lora_state_dict(model), **meta}, path)


def load_lora(model: nn.Module, src: str | dict[str, Tensor], strict: bool = True) -> None:
    """Load LoRA factors from a path written by :func:`save_lora` or a dict.

    `model` must already have had :func:`apply_lora` applied with the same
    patterns and rank. With `strict`, every LoRA key must match both ways.
    """
    if isinstance(src, str):
        src = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(src.get("lora"), dict):  # the {"lora": ..., **meta} written by save_lora
        src = src["lora"]
    own = {k for k in model.state_dict() if "lora_" in k}
    if strict:
        missing, extra = own - set(src), set(src) - own
        if missing or extra:
            raise KeyError(f"LoRA key mismatch: missing={sorted(missing)} unexpected={sorted(extra)}")
    model.load_state_dict(src, strict=False)
