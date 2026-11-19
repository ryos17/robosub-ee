"""DeepFilterNet3 adapter (48 kHz), the primary fine-tuning model.

Implements the DFN3 rows of RESEARCH_PLAN.md §7.1 (C1 zero-shot, C2/C7/C9
fine-tuning) and its §7.2 recipe, via the ``deepfilternet`` package
(``df.enhance.init_df``). ``sample_rate = 48000``: the caller resamples.

Differentiable path. ``df_features`` (libdf STFT analysis, ERB and unit-norm
features) runs in NumPy/Rust on the *noisy input only*, which needs no
gradient. The network (``DfNet``) is then run in torch on those features, and
its enhanced complex spectrum is either synthesized with a torch iSTFT
(``forward_train``) or compared to the clean spectrum in the loss, so gradients
reach every network parameter.

Loss (``loss_name="native"``, default). ``df.loss.Loss`` built from the loaded
checkpoint's ``config.ini``, exactly as ``df/train.py`` builds it
(``Loss(state, Istft(fft_size, hop, fft_window))``). For the released DFN3
config this is MultiResSpecLoss (FFT 256/512/1024/2048, gamma 0.3, magnitude
and complex factors 500) on the iSTFT waveforms + LocalSnrLoss (factor 1e-3,
target from clean vs noisy); the MaskLoss, SpectralLoss, SDR, DF-alpha and ASR
terms have factor 0 there. If ``df.loss.Loss`` cannot be built, the fallback is
``df.loss.MultiResSpecLoss`` with those same settings; ``loss_name="multires"``
(or an unusable ``df.loss``) uses ``uwse.adapters.identity.multires_stft_loss``.
``self.loss_impl`` records which is active ("df.loss.Loss",
"df.loss.MultiResSpecLoss" or "uwse.multires_stft").

Inference parity. ``enhance`` reproduces ``df.enhance.enhance(pad=True)``
bit-exactly (max abs difference 0.0 measured), including libdf's streaming
synthesis; ``forward_train`` replaces that synthesis by a torch iSTFT with the
same output (difference 2e-9). Weights are downloaded by deepfilternet to its
own cache (``%LOCALAPPDATA%/DeepFilterNet``), not ``pretrained_models/``.

torchaudio >= 2.1 removed ``torchaudio.backend.common``, which ``df.io``
imports; ``_install_torchaudio_shim`` registers a tiny stand-in module in
``sys.modules`` before importing ``df`` (site-packages is not modified).
"""

from __future__ import annotations

import logging
import sys
import types

import torch
from torch import Tensor, nn

from .base import SEAdapter, register
from .identity import as_batch, eval_mode, fit_length, multires_stft_loss, pretrained_dir

log = logging.getLogger(__name__)

def _install_torchaudio_shim() -> None:
    """Provide `torchaudio.backend.common.AudioMetaData` if torchaudio lacks it."""
    try:
        import torchaudio.backend.common  # noqa: F401

        return
    except Exception:
        pass
    try:
        import torchaudio

        meta = getattr(torchaudio, "AudioMetaData", None)
    except Exception:
        torchaudio, meta = None, None
    if meta is None:
        from dataclasses import dataclass

        @dataclass
        class AudioMetaData:  # same fields as torchaudio's
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        meta = AudioMetaData
    backend = sys.modules.get("torchaudio.backend") or types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = meta
    backend.common = common
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules["torchaudio.backend.common"] = common
    if torchaudio is not None and not hasattr(torchaudio, "backend"):
        torchaudio.backend = backend

def import_df():
    """Import the deepfilternet modules we need, with the torchaudio shim."""
    _install_torchaudio_shim()
    import importlib

    # `import df.enhance as dfe` would bind the *function* df.enhance, which
    # df/__init__.py re-exports under the same name as the submodule
    return importlib.import_module("df.enhance")

@register("dfn3")
class DFN3Adapter(SEAdapter):
    """DeepFilterNet3 (ERB gains + deep filtering), 48 kHz."""

    sample_rate = 48000

    def __init__(
        self,
        device: str | torch.device = "cpu",
        model_base_dir: str | None = None,
        post_filter: bool = False,
        atten_lim_db: float | None = None,
        loss_name: str = "native",
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__(device)
        self.lora_cfg = dict(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        if loss_name not in ("native", "multires"):
            raise ValueError("loss_name must be 'native' or 'multires'")
        self.model_base_dir = model_base_dir
        self.post_filter = post_filter
        self.atten_lim_db = atten_lim_db
        self.loss_name = loss_name
        self.df_state = None
        self.nb_df: int | None = None
        self._istft: nn.Module | None = None
        self._loss_fn = None
        self.loss_impl: str | None = None

    # -- lifecycle -------------------------------------------------------------
    def load_pretrained(self) -> None:
        dfe = import_df()
        kw = dict(post_filter=self.post_filter, log_level="WARNING", log_file=None)
        if self.model_base_dir is not None:
            kw["model_base_dir"] = self.model_base_dir
        try:
            model, df_state, _ = dfe.init_df(**kw)
        except TypeError:  # older signatures
            kw.pop("log_file", None)
            model, df_state, _ = dfe.init_df(**kw)
        self.model = model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()
        self.df_state = df_state
        from df.model import ModelParams

        self.nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
        self.sample_rate = int(df_state.sr())
        self._istft = self._make_istft().to(self.device)

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load_pretrained()

    def to(self, device):
        super().to(device)
        if self._istft is not None:
            self._istft.to(self.device)
        if self._loss_fn is not None and isinstance(self._loss_fn, nn.Module):
            self._loss_fn.to(self.device)
        return self

    def _make_istft(self) -> nn.Module:
        from df.loss import Istft

        st = self.df_state
        return Istft(st.fft_size(), st.hop_size(), torch.as_tensor(st.fft_window().copy()))

    # -- maths ---------------------------------------------------------------
    def _spec_features(self, wav: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """libdf analysis + DFN features of (B, T) audio (no gradient needed)."""
        df_features = import_df().df_features

        if hasattr(self.df_state, "reset"):
            self.df_state.reset()  # analysis is stateful across calls
        x = wav.detach().float().cpu().contiguous()
        spec, erb_feat, spec_feat = df_features(x, self.df_state, self.nb_df, device=self.device)
        return spec, erb_feat, spec_feat

    def _pad(self, wav: Tensor) -> Tensor:
        # as df.enhance.enhance(pad=True): pad n_fft so the last frames are flushed
        return torch.nn.functional.pad(wav, (0, self.df_state.fft_size()))

    def _unpad(self, wav: Tensor, T: int) -> Tensor:
        # libdf's streaming synthesis delays the output by n_fft - hop samples
        d = self.df_state.fft_size() - self.df_state.hop_size()
        return fit_length(wav[..., d:], T)

    def _reset_h0(self, batch: int) -> None:
        if hasattr(self.model, "reset_h0"):
            self.model.reset_h0(batch_size=batch, device=self.device)

    def _forward_spec(self, noisy: Tensor) -> tuple[Tensor, tuple]:
        """noisy (B, T) padded -> (enhanced spec (B,1,F',Fbins,2), model outputs)."""
        spec, erb_feat, spec_feat = self._spec_features(noisy)
        self._reset_h0(noisy.shape[0])
        out = self.model(spec.clone(), erb_feat, spec_feat)
        return out[0], out

    def _synth(self, spec: Tensor, T: int) -> Tensor:
        """(B, 1, frames, F, 2) real -> (B, T) waveform, differentiable.

        `df.loss.Istft` (torch.istft, normalized=True) on a libdf spectrum is
        aligned with the input (no streaming delay) but scaled by 1/sqrt(n_fft)
        relative to libdf's synthesis (measured: lag 0, gain 0.03227 for
        n_fft = 960, residual 3e-9), so the gain is undone here.
        """
        y = self._istft(spec.squeeze(1)) * float(self.df_state.fft_size()) ** 0.5
        return fit_length(y, T)

    def _synth_libdf(self, spec: Tensor, T: int) -> Tensor:
        """Exact libdf synthesis as in `df.enhance.enhance` (no gradient)."""
        from df.utils import as_complex

        if hasattr(self.df_state, "reset"):
            self.df_state.reset()
        z = as_complex(spec.squeeze(1).detach().float().cpu())
        y = torch.as_tensor(self.df_state.synthesis(z.numpy()))
        return self._unpad(y, T)

    def enhance(self, wav: Tensor) -> Tensor:
        """Same computation as `df.enhance.enhance(pad=True)`, batched over B."""
        self._ensure_loaded()
        wav, was_1d = as_batch(wav)
        T = wav.shape[-1]
        with eval_mode(self.model):
            enh, _ = self._forward_spec(self._pad(wav.float()))
            if self.atten_lim_db is not None and abs(self.atten_lim_db) > 0:
                # same mixing rule as df.enhance: limit attenuation to atten_lim_db
                lim = 10 ** (-abs(self.atten_lim_db) / 20)
                spec, _, _ = self._spec_features(self._pad(wav.float()))
                enh = spec * lim + enh * (1 - lim)
            y = self._synth_libdf(enh, T).to(device=wav.device, dtype=torch.float32)
        return y[0] if was_1d else y

    def forward_train(self, noisy: Tensor) -> Tensor:
        """Differentiable enhanced waveform (B, T) via torch iSTFT."""
        self._ensure_loaded()
        T = noisy.shape[-1]
        enh, _ = self._forward_spec(self._pad(noisy.float()))
        return self._synth(enh, T)

    # -- parameters ------------------------------------------------------------
    #: §7.2: peft LoRA on the Linear layers; GRUs either get a low-rank wrapper
    #: ("lora") or stay frozen ("lora_nogru"); both variants are to be logged.
    #: The GroupedLinearEinsum layers are custom modules that peft cannot wrap.
    LORA_LINEAR = (r"\.(lsnr_fc|df_fc_a)\.0$",)
    LORA_GRU = (r"\.gru$",)

    def trainable_params(self, mode: str = "full"):
        self._ensure_loaded()
        if mode == "full":
            return [p for p in self.model.parameters() if p.requires_grad]
        if mode in ("lora", "lora_nogru"):
            from .lora import apply_lora

            pats = list(self.LORA_LINEAR) + (list(self.LORA_GRU) if mode == "lora" else [])
            return list(apply_lora(self.model, pats, **self.lora_cfg))
        raise NotImplementedError(f"{self.name}: mode {mode!r} not supported")

    # -- loss ----------------------------------------------------------------
    def _build_loss(self) -> None:
        if self.loss_name == "native":
            try:
                from df.loss import Loss

                self._loss_fn = Loss(self.df_state, self._istft).to(self.device)
                self.loss_impl = "df.loss.Loss"
                return
            except Exception as e:  # noqa: BLE001
                log.warning("df.loss.Loss unavailable (%r); falling back", e)
            try:
                from df.loss import MultiResSpecLoss

                # DFN3 config.ini [multiresspecloss]
                self._loss_fn = MultiResSpecLoss(
                    n_ffts=[256, 512, 1024, 2048], gamma=0.3, factor=500, f_complex=500
                ).to(self.device)
                self.loss_impl = "df.loss.MultiResSpecLoss"
                return
            except Exception as e:  # noqa: BLE001
                log.warning("df.loss terms unavailable (%r); using uwse multires_stft_loss", e)
        self._loss_fn = None
        self.loss_impl = "uwse.multires_stft"

    def loss(self, noisy: Tensor, clean: Tensor) -> Tensor:
        self._ensure_loaded()
        if self.loss_impl is None:
            self._build_loss()
        noisy = noisy.float().to(self.device)
        clean = clean.float().to(self.device)
        T = noisy.shape[-1]
        enh_spec, out = self._forward_spec(self._pad(noisy))
        if self.loss_impl == "uwse.multires_stft":
            return multires_stft_loss(self._synth(enh_spec, T), clean, fft_sizes=(256, 512, 1024, 2048))
        if self.loss_impl == "df.loss.Loss":
            clean_spec, _, _ = self._spec_features(self._pad(clean))
            noisy_spec, _, _ = self._spec_features(self._pad(noisy))
            mask, lsnr = out[1], out[2]
            # train.py passes the mixture SNR per item; Loss uses it only for
            # logging summaries (store_losses), not in the returned value
            snrs = _mixture_snr_db(clean, noisy)
            return self._loss_fn(clean_spec, noisy_spec, enh_spec, mask, lsnr, snrs)
        return self._loss_fn(self._synth(enh_spec, T), clean)


def _mixture_snr_db(clean: Tensor, noisy: Tensor) -> Tensor:
    noise = noisy - clean
    return 10 * torch.log10(clean.pow(2).sum(-1) / noise.pow(2).sum(-1).clamp_min(1e-10))
