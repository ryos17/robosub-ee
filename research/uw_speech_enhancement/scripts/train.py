"""Fine-tuning entry point (RESEARCH_PLAN.md §7.2-7.4; conditions C2, C3, C7, C8, C9, C10).

Builds the adapter and the datasets named in a condition YAML (configs/*.yaml) and
runs `uwse.train.loop.train` (or `uwse.train.whisper_lora.train_whisper_lora` for C8).

Usage (from the project root)::

    python scripts/train.py --config configs/c2_dfn3.yaml                 # first seed
    python scripts/train.py --config configs/c2_dfn3.yaml --all-seeds     # seeds in sequence
    python scripts/train.py --config configs/c2_dfn3.yaml --smoke         # W4 throughput test
    python scripts/train.py --config configs/c2_dfn3.yaml --set train.param_mode=lora \\
        --set run_name=c3_dfn3_lora_r8                                      # C3
    python scripts/train.py --config configs/c2_dfn3.yaml --dry-run       # build only

``--smoke`` trains ~20 steps on `SyntheticPairs` (no corpora needed) with the
config's real batching (64 s of audio per step) and writes the measured
throughput (audio s per device-compute s) to runs/<run_name>_smoke/summary.json.

Training jobs are long: launch them detached in their own window, logging to
$env:TEMP\\<name>.log (PowerShell; one heavy job at a time on the 8 GB laptop GPU)::

    Start-Process powershell -ArgumentList @(
        '-NoExit','-Command',
        "Set-Location 'C:\\Users\\ryota\\Documents\\repos\\robosub-ee\\research\\uw_speech_enhancement'; & 'C:\\Users\\ryota\\anaconda3\\shell\\condabin\\conda-hook.ps1'; conda activate uwse; python scripts/train.py --config configs/c2_dfn3.yaml *>&1 | Tee-Object -FilePath `"$env:TEMP\\c2_dfn3.log`""
    ) -WindowStyle Minimized

Inspect with ``Get-Content $env:TEMP\\c2_dfn3.log -Wait``; the run resumes from
runs/<run_name>/last.pt if relaunched with the same command.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from uwse.train.datasets import (  # noqa: E402
    InAirPairs,
    ReplayPairs,
    SimPairsDataset,
    SyntheticPairs,
    build_val_set,
    list_audio_files,
)
from uwse.train.loop import TrainConfig, seed_everything, train  # noqa: E402


# ---------------------------------------------------------------------------
# config handling
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: dict[str, Any], sets: list[str]) -> dict[str, Any]:
    """``--set a.b.c=value`` (value parsed as YAML) into a nested dict copy."""
    cfg = copy.deepcopy(cfg)
    for s in sets:
        key, _, val = s.partition("=")
        if not _:
            raise ValueError(f"--set expects key=value, got {s!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def _path(p: str | None) -> str | None:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _check_not_datasets(path: str | None) -> None:
    # /datasets/ is read-only by user policy: never write frozen sets or runs there.
    # On Windows "/datasets" resolves to "<drive>:\datasets", so compare without the drive.
    if not path:
        return
    p = os.path.splitdrive(os.path.abspath(path))[1].replace("\\", "/")
    if p == "/datasets" or p.startswith("/datasets/"):
        raise ValueError(f"refusing to write under /datasets/: {path}")


def _load_fir(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = _path(path)
    if p.endswith(".npy"):
        return np.load(p)
    from uwse import audio
    x, _ = audio.load(p)
    return x


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def sim_matched_params(d: dict[str, Any]) -> dict[str, Any]:
    """`data.params` for C7/C9 with the SNR and level ranges of the uwsim config.

    C7 must differ from C2 only in the room response ("same noise, same SNR rule",
    §7.3), so unless `params` sets them explicitly, ``snr_range`` is uwsim's
    U(snr_measured -/+ snr_widen_db) clipped to snr_clip, and ``level_dbfs_range`` is
    uwsim's chain.level_dbfs, both read from ``data.sim_config``.
    """
    params = dict(d.get("params") or {})
    if d.get("sim_config"):
        from uwse.uwsim.config import load_config as load_sim

        sc = load_sim(_path(d["sim_config"]))
        nc = sc.noise
        lo = max(nc.snr_measured[0] - nc.snr_widen_db, nc.snr_clip[0])
        hi = min(nc.snr_measured[1] + nc.snr_widen_db, nc.snr_clip[1])
        params.setdefault("snr_range", (float(lo), float(hi)))
        params.setdefault("level_dbfs_range", tuple(float(v) for v in sc.chain.level_dbfs))
    return params


def build_datasets(cfg: dict[str, Any], fs_model: int, seed: int, segment_s: float | None,
                   val_segment_s: float | None):
    """(train_ds, val_ds) for the data section of a condition config."""
    d = cfg["data"]
    kind = d["kind"]
    n_train = int(d.get("n_train_items", 1_000_000))
    n_val = int(d.get("n_val_items", 900))
    val_seed = int(d.get("val_seed", 20261027))
    params = dict(d.get("params") or {})

    if kind == "synthetic":
        dur = (segment_s or 4.0) + 1.0
        tr = SyntheticPairs(n_train, fs_model, dur, seed=seed)
        va = SyntheticPairs(n_val, fs_model, dur, seed=val_seed)
        return tr, va

    train_clean = list_audio_files(_path(d["train_clean"]))
    val_clean = list_audio_files(_path(d["val_clean"]))
    if not train_clean or not val_clean:
        raise FileNotFoundError(f"no clean speech under {d['train_clean']!r} / {d['val_clean']!r}")

    if kind == "uwsim":
        common = dict(noise_root=_path(d.get("noise_root")), sim_config=_path(d["sim_config"]),
                      fs_model=fs_model, rir_bank=_path(d.get("rir_bank")),
                      target=d.get("target", "R1"), **params)
        tr = SimPairsDataset(train_clean, segment_s=segment_s, seed=seed, n_items=n_train, **common)
        va = SimPairsDataset(val_clean, segment_s=val_segment_s, seed=val_seed, n_items=n_val, **common)
    elif kind in ("inair", "replay"):
        from uwse.uwsim.noise import NoiseBank

        fs_sim = int(d.get("fs_sim", fs_model))
        noise = NoiseBank(_path(d["noise_root"]), fs=fs_sim) if d.get("noise_root") else None
        common = dict(fs_sim=fs_sim, chain=_load_fir(d.get("chain_fir")), **sim_matched_params(d))
        cls = InAirPairs if kind == "inair" else ReplayPairs
        if kind == "replay":
            common["irs"] = _path(d["ir_dir"])
        tr = cls(train_clean, noise, fs_model, n_items=n_train, seed=seed, segment_s=segment_s,
                 noise_split="train", **common)
        va = cls(val_clean, noise, fs_model, n_items=n_val, seed=val_seed, segment_s=val_segment_s,
                 noise_split="train", **common)
    else:
        raise ValueError(f"unknown data.kind {kind!r}")

    val_dir = _path(d.get("val_dir"))
    if val_dir:
        _check_not_datasets(val_dir)
        va = build_val_set(va, n_val, out_dir=val_dir, fs=fs_model)
    return tr, va


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
def run_se(cfg: dict[str, Any], seed: int, args: argparse.Namespace) -> dict[str, Any]:
    from uwse.adapters import build

    tcfg = TrainConfig.from_dict(cfg.get("train", {}))
    tcfg.seed = seed
    tcfg.run_name = f"{cfg.get('run_name', 'run')}_s{seed}"
    tcfg.out_dir = _path(tcfg.out_dir)
    if args.device:
        tcfg.device = args.device
    if args.steps:
        tcfg.max_steps = args.steps
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        cfg["data"] = {"kind": "synthetic", "n_train_items": 4096, "n_val_items": 8}
        tcfg.run_name += "_smoke"
        tcfg.max_steps = args.steps or 20
        tcfg.warmup_steps = min(tcfg.warmup_steps, 5)
        tcfg.eval_every = tcfg.max_steps
        tcfg.log_every = 5
        tcfg.resume = False
        tcfg.patience = 10**9
    _check_not_datasets(tcfg.out_dir)

    a = cfg["adapter"]
    # seed BEFORE the model exists: from-scratch init (C10) and LoRA init depend on it
    seed_everything(seed, tcfg.deterministic)
    adapter = build(a["name"], device=str(tcfg.resolved_device()), **(a.get("kwargs") or {}))
    adapter.load_pretrained()
    fs = int(adapter.sample_rate)
    train_ds, val_ds = build_datasets(cfg, fs, seed, tcfg.segment_s,
                                      tcfg.val_segment_s or tcfg.segment_s)
    print(f"[train] {tcfg.run_name}: adapter={a['name']} fs={fs} data={cfg['data']['kind']} "
          f"accum={tcfg.resolved_accum()} device={tcfg.resolved_device()}", flush=True)
    if args.dry_run:
        item = train_ds[(0, 0)]
        print(f"[dry-run] first item: noisy {tuple(item[0].shape)} target {tuple(item[1].shape)}")
        return {"dry_run": True}
    res = train(adapter, train_ds, val_ds, tcfg)
    out = {"run_dir": res.run_dir, "step": res.step, "best_val": res.best_val,
           "best_step": res.best_step, "stopped_early": res.stopped_early,
           "throughput_audio_s_per_compute_s": res.throughput_compute,
           "throughput_audio_s_per_wall_s": res.throughput_wall}
    with open(os.path.join(res.run_dir, "condition.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def run_whisper(cfg: dict[str, Any], seed: int, args: argparse.Namespace) -> dict[str, Any]:
    from uwse.train.whisper_lora import (LoRASpec, TranscribedAudio, load_librispeech_transcripts,
                                         train_whisper_lora)

    tcfg = TrainConfig.from_dict(cfg.get("train", {}))
    tcfg.seed = seed
    tcfg.run_name = f"{cfg.get('run_name', 'run')}_s{seed}"
    tcfg.out_dir = _path(tcfg.out_dir)
    if args.device:
        tcfg.device = args.device
    if args.steps:
        tcfg.max_steps = args.steps
    _check_not_datasets(tcfg.out_dir)
    w = cfg.get("whisper", {})
    lora = LoRASpec(**(w.get("lora") or {}))
    if "target_modules" in (w.get("lora") or {}):
        lora.target_modules = tuple(lora.target_modules)
    if args.smoke:
        raise SystemExit("--smoke is not defined for C8 (needs transcripts); use --steps 20")
    seed_everything(seed, tcfg.deterministic)
    train_ds, val_ds = build_datasets(cfg, 16000, seed, None, None)
    transcripts = load_librispeech_transcripts(_path(cfg["data"]["transcripts_root"]))
    train_ds, val_ds = TranscribedAudio(train_ds, transcripts), TranscribedAudio(val_ds, transcripts)
    if args.dry_run:
        wav, text = train_ds[(0, 0)]
        print(f"[dry-run] first item: audio {tuple(wav.shape)} text {text[:60]!r}")
        return {"dry_run": True}
    res = train_whisper_lora(train_ds, val_ds, tcfg, model_name=w.get("model", "openai/whisper-large-v3"),
                             lora=lora, gradient_checkpointing=bool(w.get("gradient_checkpointing", True)))
    return {"run_dir": res.run_dir, "step": res.step, "best_val": res.best_val,
            "throughput_audio_s_per_compute_s": res.throughput_compute}


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override train.max_steps")
    ap.add_argument("--smoke", action="store_true", help="~20 steps on synthetic data; throughput")
    ap.add_argument("--seed", type=int, default=None, help="one seed (default: first of seeds)")
    ap.add_argument("--all-seeds", action="store_true", help="run every seed in sequence")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dry-run", action="store_true", help="build adapter + data, then exit")
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(_path(args.config)), args.set)
    seeds = list(cfg.get("seeds") or [0])
    if args.seed is not None:
        seeds = [args.seed]
    elif not args.all_seeds:
        seeds = seeds[:1]
    task = cfg.get("task", "se")
    results = []
    for seed in seeds:
        fn = run_whisper if task == "whisper_lora" else run_se
        r = fn(cfg, int(seed), args)
        print("[train] result " + json.dumps(r), flush=True)
        results.append(r)
    return results


if __name__ == "__main__":
    main()
