"""Model-agnostic fine-tuning loop (RESEARCH_PLAN.md §7.2-7.4; conditions C2, C3, C7-C10).

One loop serves every trainable system. It only sees a `Trainable` (parameters,
`batch_loss`, save/load); `AdapterTrainable` wraps any `SEAdapter`, and
`uwse.train.whisper_lora.WhisperLoRATrainable` wraps Whisper + LoRA (C8).

Recipe (§7.4 "Full FT" / "LoRA" columns)
---------------------------------------
- AdamW, betas (0.9, 0.999); weight decay 1e-4 for full FT, 0 for LoRA.
- Peak LR per model from the config; linear warm-up then cosine decay.
- Gradient clipping at 5.0 (global L2 norm, measured before clipping and logged).
- A fixed amount of *audio* per optimizer step (64 s by default), reached by
  gradient accumulation over micro-batches: accum = ceil(64 / (B * segment_s)).
- Validation every `eval_every` steps (2k), early stopping with patience 5
  evaluations, best + last checkpoints under runs/<run_name>/.
- Exact resume: the micro-batch index stream is a pure function of
  (seed, micro-batch number), so a resumed run sees the same data as an
  uninterrupted one; optimizer, scaler and RNG states are restored too.
- Throughput (audio seconds per device-compute second) is measured and logged;
  this is the number the W4 smoke test needs to cap the step budget (§7.4).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from uwse.adapters.base import SEAdapter

__all__ = [
    "TrainConfig",
    "TrainResult",
    "Trainable",
    "AdapterTrainable",
    "StepBatchSampler",
    "lr_factor",
    "lr_at",
    "seed_everything",
    "make_optimizer",
    "fit",
    "train",
    "evaluate",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Hyperparameters of one run. Defaults are the §7.4 full-FT column for DFN3."""

    run_name: str = "run"
    out_dir: str = "runs"
    # optimizer / schedule
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float | None = None  # None -> 1e-4 (full) / 0.0 (lora)
    warmup_steps: int = 1000
    max_steps: int = 50_000
    min_lr_ratio: float = 0.0
    grad_clip: float = 5.0
    param_mode: str = "full"  # "full" | "lora"
    # batching
    audio_s_per_step: float = 64.0
    segment_s: float = 4.0
    batch_size: int = 8  # items per micro-batch
    accum_steps: int | None = None  # None -> derived from audio_s_per_step
    val_segment_s: float | None = None  # None -> segment_s
    val_batch_size: int | None = None  # None -> batch_size
    max_val_batches: int | None = None
    num_workers: int = 0
    # precision / device
    amp: bool = False
    amp_dtype: str = "fp16"  # "fp16" (GradScaler) | "bf16"
    device: str = "auto"
    # evaluation / early stopping / checkpoints
    eval_every: int = 2000
    patience: int = 5
    min_delta: float = 0.0
    log_every: int = 50
    ckpt_every: int = 1000
    resume: bool = True
    stop_at: int | None = None  # stop (resumably) after this many steps; for chunked jobs
    # reproducibility
    seed: int = 0
    deterministic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)  # free-form, copied into config.json

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "TrainConfig":
        """Build from a (YAML) dict; unknown keys raise so typos are caught."""
        d = dict(d or {})
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise KeyError(f"unknown TrainConfig keys: {unknown}")
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    # derived quantities ----------------------------------------------------
    def resolved_weight_decay(self) -> float:
        if self.weight_decay is not None:
            return float(self.weight_decay)
        return 0.0 if self.param_mode == "lora" else 1e-4

    def resolved_accum(self) -> int:
        if self.accum_steps is not None:
            return max(1, int(self.accum_steps))
        per_micro = self.batch_size * self.segment_s
        return max(1, math.ceil(self.audio_s_per_step / per_micro - 1e-9))

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_dir, self.run_name)


@dataclass
class TrainResult:
    run_dir: str
    step: int
    best_val: float
    best_step: int
    stopped_early: bool
    throughput_compute: float  # audio s per device-compute s (whole run)
    throughput_wall: float  # audio s per wall s, data loading included
    audio_s_seen: float
    # steady state: excludes the first `STEADY_SKIP` steps of this process (CUDA context,
    # cuDNN autotune, allocator and worker start-up), which bias a 20-step W4 smoke run
    throughput_compute_steady: float = float("nan")
    steady_from_step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# schedule, seeding, optimizer
# ---------------------------------------------------------------------------
STEADY_SKIP = 2  # steps per process excluded from the steady-state throughput


def lr_factor(step: int, warmup_steps: int, max_steps: int, min_ratio: float = 0.0) -> float:
    """Multiplier of the peak LR for the update with 0-based index `step`.

    Warm-up: (step + 1) / warmup for step < warmup (so the first update is
    not wasted at LR 0). Then cosine from 1 at step == warmup down to
    `min_ratio` at step == max_steps: min + (1 - min) * (1 + cos(pi * p)) / 2,
    p = (step - warmup) / (max_steps - warmup).
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(1, max_steps - warmup_steps)
    p = min(1.0, max(0.0, (step - warmup_steps) / span))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))


def lr_at(step: int, cfg: TrainConfig) -> float:
    return cfg.lr * lr_factor(step, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_optimizer(params: Sequence[nn.Parameter], cfg: TrainConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps,
        weight_decay=cfg.resolved_weight_decay(),
    )


# ---------------------------------------------------------------------------
# what the loop trains
# ---------------------------------------------------------------------------
class Trainable(Protocol):
    """Minimal interface `fit` needs."""

    def parameters(self) -> list[nn.Parameter]: ...

    def batch_loss(self, batch: Any, device: torch.device) -> tuple[Tensor, float]:
        """Return (scalar loss, seconds of audio in the batch)."""
        ...

    def set_train(self, mode: bool) -> None: ...

    def save(self, path: str, **extra: Any) -> None: ...

    def load(self, path: str) -> None: ...


class AdapterTrainable:
    """`Trainable` view of an `SEAdapter`; batches are (noisy[B,T], target[B,T], metas)."""

    def __init__(self, adapter: SEAdapter, param_mode: str = "full") -> None:
        self.adapter = adapter
        self.param_mode = param_mode
        self._params = [p for p in adapter.trainable_params(param_mode) if p.requires_grad]
        if not self._params:
            raise ValueError(f"adapter {adapter.name!r} has no trainable parameters "
                             f"in mode {param_mode!r}")

    def parameters(self) -> list[nn.Parameter]:
        return self._params

    def batch_loss(self, batch: Any, device: torch.device) -> tuple[Tensor, float]:
        noisy, target = batch[0], batch[1]
        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        loss = self.adapter.loss(noisy, target)
        audio_s = noisy.shape[0] * noisy.shape[-1] / float(self.adapter.sample_rate)
        return loss, audio_s

    def set_train(self, mode: bool) -> None:
        self.adapter.train(mode)

    def save(self, path: str, **extra: Any) -> None:
        self.adapter.save_checkpoint(path, **extra)

    def load(self, path: str) -> None:
        self.adapter.load_checkpoint(path)


# ---------------------------------------------------------------------------
# data stream
# ---------------------------------------------------------------------------
class StepBatchSampler(Sampler):
    """Deterministic, resumable stream of micro-batches of dataset keys.

    Micro-batch k holds B keys (index, crop_seed) drawn with replacement from
    `np.random.default_rng([seed, k])`, so the stream from micro-batch k0 on
    does not depend on what came before (exact resume, worker-safe).
    """

    def __init__(self, n_items: int, batch_size: int, seed: int, start: int = 0,
                 stop: int | None = None) -> None:
        if n_items <= 0:
            raise ValueError("empty dataset")
        self.n, self.b, self.seed, self.start, self.stop = n_items, batch_size, seed, start, stop

    def batch(self, k: int) -> list[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, k])
        idx = rng.integers(0, self.n, self.b)
        crop = rng.integers(0, 2**31 - 1, self.b)
        return [(int(i), int(c)) for i, c in zip(idx, crop)]

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        k = self.start
        while self.stop is None or k < self.stop:
            yield self.batch(k)
            k += 1

    def __len__(self) -> int:
        if self.stop is None:
            raise TypeError("infinite sampler")
        return max(0, self.stop - self.start)


def _default_collate(items: list[Any]) -> Any:
    """Stack (noisy, target, meta) items of equal length."""
    noisy = torch.stack([torch.as_tensor(it[0]) for it in items]).float()
    target = torch.stack([torch.as_tensor(it[1]) for it in items]).float()
    metas = [it[2] if len(it) > 2 else {} for it in items]
    return noisy, target, metas


def _amp_ctx(device: torch.device, cfg: TrainConfig):
    if not cfg.amp:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.bfloat16  # CPU autocast supports bf16 only
    return torch.autocast(device_type=device.type, dtype=dtype)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(trainable: Trainable, loader: Iterable[Any], device: torch.device,
             cfg: TrainConfig | None = None, max_batches: int | None = None) -> float:
    """Mean validation loss, weighted by audio seconds per batch."""
    cfg = cfg or TrainConfig()
    trainable.set_train(False)
    tot, w = 0.0, 0.0
    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            with _amp_ctx(device, cfg):
                loss, sec = trainable.batch_loss(batch, device)
            if torch.isfinite(loss):
                tot += float(loss) * sec
                w += sec
    finally:
        trainable.set_train(True)
    return tot / w if w > 0 else float("nan")


def _atomic_save(save_fn: Callable[[str], None], path: str) -> None:
    tmp = path + ".tmp"
    save_fn(tmp)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.replace(tmp, path)


class _JsonlLog:
    def __init__(self, path: str, truncate_after_step: int | None) -> None:
        self.path = path
        if truncate_after_step is not None and os.path.exists(path):
            # drop entries from a crashed tail so a resumed log has no duplicates
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            rows = [r for r in rows if r.get("step", 0) <= truncate_after_step]
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    def write(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def fit(trainable: Trainable, train_ds: Dataset, val_ds: Dataset | None, cfg: TrainConfig,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        val_collate_fn: Callable[[list[Any]], Any] | None = None,
        device: torch.device | None = None) -> TrainResult:
    """Train `trainable` on keys streamed by `StepBatchSampler` (see module docstring).

    `train_ds[(index, crop_seed)]` must return one item; `val_ds[i]` (int
    keys) is iterated in order. Nothing here knows about audio models: the
    loss and the audio duration of a batch come from `trainable.batch_loss`.
    """
    device = device or cfg.resolved_device()
    collate_fn = collate_fn or _default_collate
    val_collate_fn = val_collate_fn or collate_fn
    os.makedirs(cfg.run_dir, exist_ok=True)
    last_path = os.path.join(cfg.run_dir, "last.pt")
    state_path = os.path.join(cfg.run_dir, "last_state.pt")
    best_path = os.path.join(cfg.run_dir, "best.pt")
    accum = cfg.resolved_accum()

    seed_everything(cfg.seed, cfg.deterministic)
    params = trainable.parameters()
    opt = make_optimizer(params, cfg)
    use_scaler = cfg.amp and cfg.amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    step, best_val, best_step, bad_evals = 0, float("inf"), -1, 0
    audio_seen, compute_total, wall_total = 0.0, 0.0, 0.0
    resumed = False
    if cfg.resume and os.path.exists(state_path) and os.path.exists(last_path):
        trainable.load(last_path)
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        step, best_val, best_step = st["step"], st["best_val"], st["best_step"]
        bad_evals = st["bad_evals"]
        audio_seen, compute_total, wall_total = st["audio_seen"], st["compute_s"], st["wall_s"]
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        resumed = True
    with open(os.path.join(cfg.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**cfg.to_dict(), "accum_steps_resolved": accum,
                   "weight_decay_resolved": cfg.resolved_weight_decay(),
                   "device_resolved": str(device)}, f, indent=2, default=str)
    log = _JsonlLog(os.path.join(cfg.run_dir, "train_log.jsonl"), step if resumed else 0)
    if resumed:
        log.write({"event": "resume", "step": step})

    def save_state() -> None:
        _atomic_save(lambda p: trainable.save(p, step=step, val_loss=best_val), last_path)
        st = {
            "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
            "best_val": best_val, "best_step": best_step, "bad_evals": bad_evals,
            "audio_seen": audio_seen, "compute_s": compute_total, "wall_s": wall_total,
            "py_rng": random.getstate(), "np_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": cfg.to_dict(),
        }
        _atomic_save(lambda p: torch.save(st, p), state_path)

    sampler = StepBatchSampler(len(train_ds), cfg.batch_size, cfg.seed,
                               start=step * accum, stop=cfg.max_steps * accum)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=cfg.num_workers > 0)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=cfg.val_batch_size or cfg.batch_size,
                                shuffle=False, collate_fn=val_collate_fn,
                                num_workers=cfg.num_workers)

    stop_at = cfg.max_steps if cfg.stop_at is None else min(cfg.max_steps, cfg.stop_at)
    # a run that already early-stopped stays stopped when the same command is relaunched
    stopped_early = bool(resumed and val_loader is not None and bad_evals >= cfg.patience)
    start_step = step
    steady: tuple[int, float, float] | None = None  # (step, audio_seen, compute_total)
    history: list[dict[str, Any]] = []
    win = {"loss": 0.0, "n": 0, "audio": 0.0, "compute": 0.0, "data": 0.0, "gn": 0.0}
    n_bad_loss = 0
    trainable.set_train(True)
    it = iter(loader)
    pbar = tqdm(total=cfg.max_steps, initial=step, desc=cfg.run_name, dynamic_ncols=True)
    try:
        while step < stop_at and not stopped_early:
            lr = lr_at(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            step_loss, step_audio, step_compute, step_data, finite = 0.0, 0.0, 0.0, 0.0, True
            for _ in range(accum):
                t0 = time.perf_counter()
                batch = next(it)
                t1 = time.perf_counter()
                with _amp_ctx(device, cfg):
                    loss, sec = trainable.batch_loss(batch, device)
                if not torch.isfinite(loss):
                    finite = False
                else:
                    scaler.scale(loss / accum).backward()
                    step_loss += float(loss.detach()) / accum
                _sync(device)
                t2 = time.perf_counter()
                step_audio += sec
                step_data += t1 - t0
                step_compute += t2 - t1
            t3 = time.perf_counter()
            if finite:
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
                scaler.step(opt)  # skips the update itself if fp16 grads overflowed
                scaler.update()
                n_bad_loss = 0
            else:
                gn = float("nan")
                opt.zero_grad(set_to_none=True)
                n_bad_loss += 1
                log.write({"event": "nonfinite_loss", "step": step + 1})
                if n_bad_loss >= 10:
                    raise FloatingPointError("10 consecutive non-finite losses")
            _sync(device)
            step_compute += time.perf_counter() - t3
            step += 1
            audio_seen += step_audio
            compute_total += step_compute
            wall_total += step_compute + step_data
            if finite:
                win["loss"] += step_loss
                win["gn"] += gn
                win["n"] += 1
            win["audio"] += step_audio
            win["compute"] += step_compute
            win["data"] += step_data
            if steady is None and step - start_step == STEADY_SKIP:
                steady = (step, audio_seen, compute_total)
            pbar.update(1)
            pbar.set_postfix(loss=f"{step_loss:.4g}", lr=f"{lr:.2e}")

            if step % cfg.log_every == 0 or step == stop_at:
                n = max(1, win["n"])
                row = {
                    "step": step, "loss": win["loss"] / n, "lr": lr, "grad_norm": win["gn"] / n,
                    "audio_s": win["audio"],
                    "throughput_audio_s_per_compute_s": win["audio"] / max(win["compute"], 1e-9),
                    "throughput_audio_s_per_wall_s":
                        win["audio"] / max(win["compute"] + win["data"], 1e-9),
                    "data_frac": win["data"] / max(win["compute"] + win["data"], 1e-9),
                }
                log.write(row)
                history.append(row)
                win = dict.fromkeys(win, 0.0)
                win["n"] = 0

            evaluated = False
            if val_loader is not None and (step % cfg.eval_every == 0 or step == cfg.max_steps):
                evaluated = True
                val = evaluate(trainable, val_loader, device, cfg, cfg.max_val_batches)
                improved = val < best_val - cfg.min_delta
                if improved:
                    best_val, best_step, bad_evals = val, step, 0
                    _atomic_save(lambda p: trainable.save(p, step=step, val_loss=val), best_path)
                else:
                    bad_evals += 1
                row = {"step": step, "val_loss": val, "best_val": best_val,
                       "best_step": best_step, "bad_evals": bad_evals}
                log.write(row)
                history.append(row)
                tqdm.write(f"[{cfg.run_name}] step {step} val {val:.5g} best {best_val:.5g}"
                           f" ({bad_evals}/{cfg.patience})")
                if bad_evals >= cfg.patience:
                    stopped_early = True
            if evaluated or step % cfg.ckpt_every == 0 or step == stop_at or stopped_early:
                save_state()
            if stopped_early:
                log.write({"event": "early_stop", "step": step, "best_step": best_step})
                break
    finally:
        pbar.close()

    if val_loader is None and os.path.isfile(last_path):
        shutil.copyfile(last_path, best_path)  # no validation: best == last
    if steady is not None and step > steady[0]:
        tp_steady = (audio_seen - steady[1]) / max(compute_total - steady[2], 1e-9)
        steady_from = steady[0]
    else:  # too few steps in this process: fall back to the whole-run figure
        tp_steady, steady_from = audio_seen / max(compute_total, 1e-9), start_step
    result = TrainResult(
        run_dir=cfg.run_dir, step=step, best_val=best_val, best_step=best_step,
        stopped_early=stopped_early,
        throughput_compute=audio_seen / max(compute_total, 1e-9),
        throughput_wall=audio_seen / max(wall_total, 1e-9),
        audio_s_seen=audio_seen, history=history,
        throughput_compute_steady=tp_steady, steady_from_step=steady_from,
    )
    summary = {k: v for k, v in dataclasses.asdict(result).items() if k != "history"}
    summary.update(device=str(device), accum_steps=accum,
                   gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   max_mem_gb=(torch.cuda.max_memory_allocated(device) / 2**30
                               if device.type == "cuda" else None))
    with open(os.path.join(cfg.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return result


def train(adapter: SEAdapter, train_ds: Dataset, val_ds: Dataset | None,
          cfg: TrainConfig | dict[str, Any]) -> TrainResult:
    """Fine-tune an `SEAdapter` on (noisy, target, meta) pair datasets.

    On return the adapter holds the best-validation weights (best.pt; equal to
    the last weights when there is no validation set).

    `train_ds`/`val_ds` yield variable-length pairs at `adapter.sample_rate`;
    they are wrapped in `SegmentDataset` (random crops of `segment_s` for
    training, deterministic crops of `val_segment_s` for validation) unless
    they already are one.
    """
    from uwse.train.datasets import SegmentDataset  # local: avoid import cycle

    cfg = cfg if isinstance(cfg, TrainConfig) else TrainConfig.from_dict(cfg)
    device = cfg.resolved_device()
    adapter.to(device)
    fs = adapter.sample_rate
    if not isinstance(train_ds, SegmentDataset):
        train_ds = SegmentDataset(train_ds, int(round(cfg.segment_s * fs)))
    if val_ds is not None and not isinstance(val_ds, SegmentDataset):
        val_ds = SegmentDataset(val_ds, int(round((cfg.val_segment_s or cfg.segment_s) * fs)))
    trainable = AdapterTrainable(adapter, cfg.param_mode)
    res = fit(trainable, train_ds, val_ds, cfg, device=device)
    # leave the adapter holding the selected (best-on-sim-val) weights, as C8 does;
    # last.pt keeps the final weights for resuming
    best = os.path.join(res.run_dir, "best.pt")
    if os.path.isfile(best):
        trainable.load(best)
    return res
