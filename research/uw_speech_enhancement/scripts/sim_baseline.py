"""Sim-only baseline: C0 and zero-shot systems on uwsim twins of v0-test.

This is a *pre-pool* dry run of the E7 baseline table (RESEARCH_PLAN.md §9 E7,
§7.3 C0/C1, §8.1-8.3) on simulated audio only. It exercises the whole scoring
path on real speech and gives first sim numbers; it is NOT a result for the
paper, because every simulator parameter is still an uncalibrated placeholder
and no noise bank has been recorded (the noisy condition uses synthetic
-3 dB/oct noise, labelled as such).

Design
------
- Utterances: v0-test selection (200 test-clean, 5 x 40 speakers), §5.4.
- Positions: the planned v0 core, ranges 1 / 3 / 6 m in the Avery pool at
  nominal dimensions, projector and receiver at 1 m depth, on-axis (§6.6).
  One fixed RIR per position (the real pool has one channel per position).
- Conditions: "quiet" (SNR 30 dB) and "noise0" (SNR 0 dB, synthetic noise).
- Systems: identity (C0), wpe1, metricganplus, sepformer_wham16k,
  sepformer_whamr, dfn3; oracles R1 (chain reference) and clean (R0 digital).
- ASR: Whisper large-v3 with the §8.3 protocol, after RMS-active level
  normalization using the VAD of R0.
- Statistics: corpus WER and mean ESTOI with cluster-bootstrap CIs over
  speakers (one stratum per position x condition).

Stages are cached under ``--out`` so a run can be resumed:
  sim+enhance -> <out>/wav/<cond>/<system>/<utt>.wav (16 kHz)
  asr         -> <out>/transcripts.jsonl
  score       -> <out>/per_utt.jsonl, <out>/summary.json, <out>/summary.md

Usage (long: launch detached per the repo convention)::

    Start-Process powershell -ArgumentList @('-NoExit','-Command',
      "Set-Location '<project>'; & 'C:\\Users\\ryota\\anaconda3\\shell\\condabin\\conda-hook.ps1';
       conda activate uwse; python scripts/sim_baseline.py *>&1 |
       Tee-Object -FilePath `"$env:TEMP\\sim_baseline.log`"") -WindowStyle Minimized

Quick check: ``python scripts/sim_baseline.py --n-utts 4 --systems identity,wpe1
--asr-model tiny --out runs/sim_baseline_quick``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uwse import audio  # noqa: E402

FS_SIM = 48000
FS_SCORE = 16000
RANGES_M = (1.0, 3.0, 6.0)
DEPTH_M = 1.0
# name -> (SNR dB, scene overrides). "noise0" gets the simulator's stationary
# -3 dB/oct fallback noise; "bubbles0" gets Minnaert bubble trains, the only noise
# training sees while no recorded bank exists (matched-noise check).
CONDITIONS = {"quiet": (30.0, {}), "noise0": (0.0, {}),
              "bubbles0": (0.0, {"bubble_rate": 10.0, "bubble_rel_db": 0.0})}
SYSTEMS = ("wpe1", "metricganplus", "sepformer_wham16k", "sepformer_whamr", "dfn3")
ORACLES = ("R1", "clean")


def _check_out(p: Path) -> None:
    if p.resolve().as_posix().split(":", 1)[-1].startswith("/datasets"):
        raise SystemExit("refusing to write under /datasets")


def select_utts(ls_root: Path, out: Path, n: int | None):
    from uwse.bench.librispeech import build_manifest, find_subset_dir, read_manifest, select_v0_test, write_manifest

    man_path = out / "manifest_test-clean.jsonl"
    if man_path.exists():
        man = read_manifest(man_path)
    else:
        man = build_manifest(find_subset_dir(ls_root, "test-clean"), "test-clean")
        write_manifest(man_path, man)
    sel = select_v0_test(man)
    write_manifest(out / "v0_test.jsonl", sel)
    return sel[:n] if n else sel


def make_scenes(cfg_path: str, seed: int) -> dict[float, object]:
    """One fixed scene per range: nominal Avery pool, 1 m depths, on-axis, no randomization
    beyond the (uncalibrated) boundary priors drawn once per position."""
    from uwse.uwsim.config import load_config
    from uwse.uwsim.geometry import avery_competition_pool
    from uwse.uwsim.simulate import sample_scene

    cfg = load_config(cfg_path)
    pool = avery_competition_pool()
    pdict = {"L": pool.L, "W": pool.W, "H0": pool.H0, "H1": pool.H1}
    scenes = {}
    for i, r in enumerate(RANGES_M):
        base = sample_scene(cfg, seed + i, generic=False)
        x0, y0 = 0.5 * pool.L - 0.5 * r, 0.5 * pool.W
        src, rx = [x0, y0, DEPTH_M], [x0 + r, y0, DEPTH_M]
        f_s = base.c / (4.0 * max(pool.H0, pool.H1))
        f_x = max(f_s, cfg.acoustics.f_x_min)
        scenes[r] = dataclasses.replace(
            base, pool=pdict, src=src, rx=rx, range_m=r, orientation_deg=0.0, theta_deg=0.0,
            eps=0.0, ir_perturb=None, thruster_on=False, thruster_pos=None, ambient_on=False,
            bubble_rate=0.0, proj_eq=[base.proj_eq[0], [0.0] * len(base.proj_eq[1])],
            hyd_eq=[base.hyd_eq[0], [0.0] * len(base.hyd_eq[1])], drive_db=0.0,
            resonance_factor=1.0, f_x=float(f_x), modal_f_max=float(f_x * cfg.acoustics.modal_f_max_factor),
            level_dbfs=-26.0, extra={"sim_baseline": True},
        )
    return scenes


def _load_system(label: str, device: str, adapters: dict[str, str], ckpts: dict[str, str]):
    """Build the adapter behind system `label` (`adapters` maps label -> registry
    name, default the label itself) and load its pretrained weights, then the
    fine-tuned checkpoint if `ckpts` names one."""
    from uwse.adapters import build

    se = build(adapters.get(label, label), device=device)
    se.load_pretrained()
    if label in ckpts:
        se.load_checkpoint(ckpts[label])
    se.eval()
    return se


def _enhance_one(se, x: np.ndarray, fs_in: int, n_out: int, device: str) -> np.ndarray:
    """Resample `x` (fs_in) to the adapter rate, enhance, return n_out samples at 16 kHz."""
    import torch

    xa = audio.resample(x.astype(np.float64), fs_in, se.sample_rate)
    with torch.no_grad():
        y = se.enhance(torch.as_tensor(xa, dtype=torch.float32)[None].to(device))
    y16 = audio.resample(y.detach().float().cpu().numpy().reshape(-1).astype(np.float64), se.sample_rate, FS_SCORE)
    return np.pad(y16, (0, max(0, n_out - y16.size)))[:n_out]


def stage_sim_enhance(utts, scenes, systems, out: Path, device: str, seed: int,
                      adapters: dict[str, str] | None = None, ckpts: dict[str, str] | None = None) -> None:
    import torch

    from uwse.adapters import import_errors
    from uwse.uwsim.simulate import render_rir, simulate

    rirs = {r: render_rir(s) for r, s in scenes.items()}
    for (cname, (snr, over)), r in [(c, r) for c in CONDITIONS.items() for r in RANGES_M]:
        cond = f"{cname}_{r:g}m"
        scene = dataclasses.replace(scenes[r], snr_db=snr, **over)
        cdir = out / "wav" / cond
        todo = [s for s in (*ORACLES, "noisy", *systems)
                if not all((cdir / s / f"{u.utt_id}.wav").exists() for u in utts)]
        if not todo:
            continue
        if not set(todo) & {*ORACLES, "noisy"} and (cdir / "R0").is_dir():
            # References are cached: add the missing systems by re-simulating one
            # item at a time (same per-item RNG, so the same 48 kHz `noisy` up to
            # the last ADC step) without rewriting the cached wavs.
            _sim_enhance_stream(utts, scene, rirs[r], todo, cdir, device, seed, r, snr,
                                adapters or {}, ckpts or {})
            continue
        sims = {}
        for k, u in enumerate(tqdm(utts, desc=f"sim {cond}", leave=False)):
            s16, fs = sf.read(u.path, dtype="float64")
            s48 = audio.resample(s16, fs, FS_SIM)
            rng = np.random.default_rng([seed, k, int(r * 10), int(snr + 100)])
            sim = simulate(s48, scene, None, rng, fs_model=FS_SIM, rir=rirs[r])
            sims[u.utt_id] = ({k: sim[k] for k in ("target_R1", "target_R0", "noisy")}, s16)
        for name in ("R1", "clean", "noisy"):
            d = cdir / name
            d.mkdir(parents=True, exist_ok=True)
            for uid, (sim, s16) in sims.items():
                x = {"R1": sim["target_R1"], "noisy": sim["noisy"]}.get(name)
                y = s16 if name == "clean" else audio.resample(x.astype(np.float64), FS_SIM, FS_SCORE)
                sf.write(d / f"{uid}.wav", y.astype(np.float32), FS_SCORE, subtype="FLOAT")
            # R0 (band-limited, aligned) is the VAD/level reference for §8.3
        d = cdir / "R0"
        d.mkdir(parents=True, exist_ok=True)
        for uid, (sim, _) in sims.items():
            y = audio.resample(sim["target_R0"].astype(np.float64), FS_SIM, FS_SCORE)
            sf.write(d / f"{uid}.wav", y.astype(np.float32), FS_SCORE, subtype="FLOAT")
        for name in systems:
            d = cdir / name
            if all((d / f"{u.utt_id}.wav").exists() for u in utts):
                continue
            if (adapters or {}).get(name, name) in import_errors:
                print(f"[skip] {name}: {import_errors[(adapters or {}).get(name, name)]}")
                continue
            d.mkdir(parents=True, exist_ok=True)
            dev = "cpu" if (adapters or {}).get(name, name) in ("identity", "wpe1") else device
            se = _load_system(name, dev, adapters or {}, ckpts or {})
            fs_a = se.sample_rate
            t0, n_audio = time.perf_counter(), 0.0
            for uid, (sim, _) in tqdm(sims.items(), desc=f"{name} {cond}", leave=False):
                x = audio.resample(sim["noisy"].astype(np.float64), FS_SIM, fs_a)
                with torch.no_grad():
                    y = se.enhance(torch.as_tensor(x, dtype=torch.float32)[None].to(dev))
                y = y.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
                y16 = audio.resample(y, fs_a, FS_SCORE)
                n16 = int(round(sim["noisy"].size * FS_SCORE / FS_SIM))
                y16 = np.pad(y16, (0, max(0, n16 - y16.size)))[:n16]
                sf.write(d / f"{uid}.wav", y16.astype(np.float32), FS_SCORE, subtype="FLOAT")
                n_audio += x.size / fs_a
            print(f"{name} {cond}: RTF {(time.perf_counter() - t0) / max(n_audio, 1e-9):.3f}")
            del se
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()


def _sim_enhance_stream(utts, scene, rir, systems, cdir: Path, device: str, seed: int, r: float, snr: float,
                        adapters: dict[str, str], ckpts: dict[str, str]) -> None:
    from uwse.uwsim.simulate import simulate

    ses = {}
    for name in systems:
        (cdir / name).mkdir(parents=True, exist_ok=True)
        dev = "cpu" if adapters.get(name, name) in ("identity", "wpe1") else device
        ses[name] = (_load_system(name, dev, adapters, ckpts), dev)
    for k, u in enumerate(tqdm(utts, desc=f"sim+{','.join(systems)} {cdir.name}", leave=False)):
        if all((cdir / name / f"{u.utt_id}.wav").exists() for name in systems):
            continue
        s16, fs = sf.read(u.path, dtype="float64")
        rng = np.random.default_rng([seed, k, int(r * 10), int(snr + 100)])
        noisy = simulate(audio.resample(s16, fs, FS_SIM), scene, None, rng, fs_model=FS_SIM, rir=rir)["noisy"]
        n16 = int(round(noisy.size * FS_SCORE / FS_SIM))
        for name, (se, dev) in ses.items():
            y16 = _enhance_one(se, noisy, FS_SIM, n16, dev)
            sf.write(cdir / name / f"{u.utt_id}.wav", y16.astype(np.float32), FS_SCORE, subtype="FLOAT")
    del ses
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


def stage_enhance_cached(utts, systems, out: Path, device: str, ckpts: dict[str, str],
                         adapters: dict[str, str] | None = None) -> None:
    """Enhance the cached 16 kHz `noisy` wavs with (trained) adapters.

    Used to add systems to an existing run without re-simulating: the cached
    `noisy` is the same 48 -> 16 kHz resample the sim stage feeds 16 kHz
    adapters, so for those this matches `stage_sim_enhance` up to float32
    storage round-off. Adapters at other rates resample from 16 kHz and lose
    the 8-24 kHz band of the sim input.
    """
    import torch

    for cond in sorted(p.name for p in (out / "wav").iterdir()):
        cdir = out / "wav" / cond
        for name in systems:
            d = cdir / name
            if all((d / f"{u.utt_id}.wav").exists() for u in utts):
                continue
            d.mkdir(parents=True, exist_ok=True)
            se = _load_system(name, device, adapters or {}, ckpts)
            for u in tqdm(utts, desc=f"{name} {cond}", leave=False):
                x16, _ = sf.read(cdir / "noisy" / f"{u.utt_id}.wav", dtype="float64")
                y16 = _enhance_one(se, x16, FS_SCORE, x16.size, device)
                sf.write(d / f"{u.utt_id}.wav", y16.astype(np.float32), FS_SCORE, subtype="FLOAT")
            del se
            if device.startswith("cuda"):
                torch.cuda.empty_cache()


def _systems_on_disk(out: Path, cond: str) -> list[str]:
    return sorted(p.name for p in (out / "wav" / cond).iterdir() if p.is_dir() and p.name != "R0")


def stage_asr(utts, out: Path, model: str, device: str) -> None:
    from uwse.evaluation.asr import TranscriptCache, WhisperASR
    from uwse.evaluation.normalize import normalize_cut

    asr = WhisperASR(model, device=device)
    cache = TranscriptCache(str(out / "transcripts.jsonl"))
    key = f"whisper-{model}+lnorm"
    # Build only the uncached work list (paths, no audio), then load, normalize
    # and decode one item at a time: holding every normalized cut in RAM cost
    # ~4.5 GB for the full 9600-item run.
    todo = []
    for cond in sorted(p.name for p in (out / "wav").iterdir()):
        for sysname in _systems_on_disk(out, cond):
            for u in utts:
                p = out / "wav" / cond / sysname / f"{u.utt_id}.wav"
                if p.exists() and cache.get(f"{cond}/{sysname}", u.utt_id, key) is None:
                    todo.append((cond, sysname, u.utt_id, p))
    print(f"ASR: {len(todo)} uncached items, {len(cache)} cached, device {asr.device}, fp16 {asr.fp16}")
    r0_cond, r0 = None, {}
    for cond, sysname, uid, p in tqdm(todo, desc=f"ASR {key}"):
        if cond != r0_cond:
            r0_cond = cond
            r0 = {u.utt_id: sf.read(out / "wav" / cond / "R0" / f"{u.utt_id}.wav")[0] for u in utts}
        x, _ = sf.read(p)
        ref = x if sysname == "clean" else r0[uid]
        y, _, _ = normalize_cut(x, ref, FS_SCORE)
        res = asr.transcribe(np.clip(y, -1, 1).astype(np.float32))
        cache.put({"system": f"{cond}/{sysname}", "utt_id": uid, "model": key, **res})
    (out / "asr_info.json").write_text(json.dumps(asr.info(), indent=1))


def stage_score(utts, out: Path, model: str, n_boot: int) -> None:
    from uwse.evaluation.asr import TranscriptCache, hallucination_flags, utterance_counts
    from uwse.evaluation.metrics import compute_metrics
    from uwse.evaluation.stats import cluster_bootstrap_mean, cluster_bootstrap_wer

    cache = TranscriptCache(str(out / "transcripts.jsonl"))
    key = f"whisper-{model}+lnorm"
    refs = {u.utt_id: u.transcript for u in utts}
    spk = {u.utt_id: u.speaker for u in utts}
    rows, summary = [], []
    for cond in sorted(p.name for p in (out / "wav").iterdir()):
        R1 = {u.utt_id: sf.read(out / "wav" / cond / "R1" / f"{u.utt_id}.wav")[0] for u in utts}
        R0 = {u.utt_id: sf.read(out / "wav" / cond / "R0" / f"{u.utt_id}.wav")[0] for u in utts}
        for sysname in _systems_on_disk(out, cond):
            ids, hyps, raw = [], [], []
            for u in utts:
                rec = cache.get(f"{cond}/{sysname}", u.utt_id, key)
                if rec is not None:
                    ids.append(u.utt_id)
                    hyps.append(rec["text"])
            if not ids:
                continue
            cnt = utterance_counts([refs[i] for i in ids], hyps)
            flags = hallucination_flags([refs[i] for i in ids], hyps, raw_hyps=hyps)
            met = []
            for i in tqdm(ids, desc=f"metrics {cond}/{sysname}", leave=False):
                x, _ = sf.read(out / "wav" / cond / sysname / f"{i}.wav")
                m = compute_metrics(x, R1[i], FS_SCORE, ref_r0=R0[i], metrics=("estoi", "si_sdr", "ci_sdr"))
                met.append(m)
            for i, h, c, f, m in zip(ids, hyps, cnt, flags, met):
                rows.append({"cond": cond, "system": sysname, "utt_id": i, "speaker": spk[i], "hyp": h,
                             **c.as_dict(), "flag": bool(f), **m})
            err = np.array([c.errors for c in cnt], float)
            nref = np.array([c.n_ref for c in cnt], float)
            cl = [spk[i] for i in ids]
            w = cluster_bootstrap_wer(err, nref, cl, n_boot=n_boot)
            e = cluster_bootstrap_mean([m["estoi"] for m in met], cl, n_boot=n_boot)
            ci = cluster_bootstrap_mean([m["ci_sdr"] for m in met], cl, n_boot=n_boot)
            summary.append({"cond": cond, "system": sysname, "n": len(ids),
                            "wer": w.estimate, "wer_lo": w.ci_low, "wer_hi": w.ci_high,
                            "S": int(sum(c.substitutions for c in cnt)), "D": int(sum(c.deletions for c in cnt)),
                            "I": int(sum(c.insertions for c in cnt)), "n_flagged": int(flags.sum()),
                            "estoi": e.estimate, "estoi_lo": e.ci_low, "estoi_hi": e.ci_high,
                            "ci_sdr": ci.estimate})
    with open(out / "per_utt.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    lines = ["| condition | system | n | WER % [95% CI] | S/D/I | flagged | ESTOI vs R1 [95% CI] | CI-SDR dB |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['cond']} | {s['system']} | {s['n']} | {100 * s['wer']:.1f} "
                     f"[{100 * s['wer_lo']:.1f}, {100 * s['wer_hi']:.1f}] | {s['S']}/{s['D']}/{s['I']} | "
                     f"{s['n_flagged']} | {s['estoi']:.3f} [{s['estoi_lo']:.3f}, {s['estoi_hi']:.3f}] | "
                     f"{s['ci_sdr']:.1f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--librispeech", default=str(ROOT / "data" / "librispeech"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "sim_baseline"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "uwsim_v0.yaml"))
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--n-utts", type=int, default=None)
    ap.add_argument("--asr-model", default="large-v3")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--stages", default="sim,asr,score",
                    help="comma list of sim, enhance (cached noisy -> --systems), asr, score")
    ap.add_argument("--ckpt", action="append", default=[], metavar="SYSTEM=PATH",
                    help="fine-tuned checkpoint for a system, e.g. kiel_unet=runs/c10_kiel_unet_s0/best.pt")
    ap.add_argument("--adapter", action="append", default=[], metavar="SYSTEM=ADAPTER",
                    help="registry adapter behind a system label, e.g. dfn3_c2=dfn3 (default: the label)")
    args = ap.parse_args(argv)

    import torch

    out = Path(args.out)
    _check_out(out)
    out.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    utts = select_utts(Path(args.librispeech), out, args.n_utts)
    print(f"{len(utts)} utterances, {len({u.speaker for u in utts})} speakers, device {device}")
    stages = set(args.stages.split(","))
    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    adapters = dict(a.split("=", 1) for a in args.adapter)
    if "sim" in stages:
        scenes = make_scenes(args.config, args.seed)
        (out / "scenes.json").write_text(json.dumps({f"{r:g}m": s.to_dict() for r, s in scenes.items()},
                                                    indent=1, default=float))
        stage_sim_enhance(utts, scenes, [s for s in args.systems.split(",") if s], out, device, args.seed,
                          adapters, ckpts)
    if "enhance" in stages:
        stage_enhance_cached(utts, [s for s in args.systems.split(",") if s], out, device, ckpts, adapters)
    if "asr" in stages:
        stage_asr(utts, out, args.asr_model, device)
    if "score" in stages:
        stage_score(utts, out, args.asr_model, args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
