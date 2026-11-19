"""Generate the frozen simulated non-speech set nonspeech_v0 (C8g gate evaluation, FINDINGS.md).

Each segment is a uwsim v0_stat scene rendered exactly like a speech scene (speech through the
projector -> pool RIR -> noise at an SNR drawn from the config's measured range -> receive chain ->
gain -> ADC floor); the received *speech is then dropped* and the received noise is passed on
with the same gain and an independent ADC-floor realisation. Noise types: Minnaert bubbles,
stationary coloured Gaussian (slopes inside the training band, ``stat_in``, and inside the
held-out -4..-2 dB/oct band, ``stat_held``), thruster-like harmonic hum, and ``near_silent``
(stationary scene whose SNR is raised by 30-40 dB, i.e. noise that far below the stationary level,
leaving mostly the codec floor).

Per segment the manifest also stores ``speech_gain``: the frozen level-normalization gain
(scripts/score.py ``level_normalize``: R0-VAD RMS gain, ``normalize_cut``) of the *paired speech
rendering* of the same scene; ``gain_median`` is the median of those over the segment's
(noise type, range) group, assigned with ``normalize.assign_gains`` (the section 8.3 noise-row rule).

Usage::

    python scripts/make_nonspeech_set.py --split dev  --out data/nonspeech_v0
    python scripts/make_nonspeech_set.py --split dev  --check-regen 5   # byte-identical check
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FS_SIM, FS_SCORE = 48000, 16000


# ---------------------------------------------------------------------------
# Planning (pure; no audio)
# ---------------------------------------------------------------------------
def load_recipe(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _draw_slope(rng: np.random.Generator, bands: list[list[float]]) -> float:
    """Uniform over the union of intervals."""
    lens = np.array([b[1] - b[0] for b in bands])
    i = int(rng.choice(len(bands), p=lens / lens.sum()))
    return float(rng.uniform(*bands[i]))


def cell_params(recipe: dict, noise_type: str, rng: np.random.Generator) -> dict:
    """Draw the noise-dependent scene overrides and SNR of one segment."""
    snr = float(rng.uniform(*recipe["snr_db"]))
    if noise_type == "bubbles":
        c = recipe["bubbles"]
        return {"snr_db": snr, "over": {"bubble_rate": float(rng.uniform(*c["rate"])), "bubble_rel_db": c["rel_db"]}}
    if noise_type in ("stat_in", "stat_held", "near_silent"):
        c = recipe["stat_held" if noise_type == "stat_held" else "stat_in"]
        over = {"stationary_on": True, "stationary_slope_db": _draw_slope(rng, c["slope_db_per_oct"]),
                "stationary_rel_db": c["rel_db"]}
        att = 0.0
        if noise_type == "near_silent":
            att = float(rng.uniform(*recipe["near_silent_extra_att_db"]))
        return {"snr_db": snr + att, "over": over, "att_db": att, "snr_base_db": snr}
    if noise_type == "thruster":
        c = recipe["thruster"]
        return {"snr_db": snr, "over": {"tonal_on": True, "tonal_f0": float(rng.uniform(*c["f0_hz"])),
                                        "tonal_rel_db": c["rel_db"]}}
    raise ValueError(noise_type)


def plan_segments(recipe: dict, split: str, utts: list, ref_durs: list[float],
                  used: set[str] | None = None) -> list[dict]:
    """Balanced plan: ``n_per_cell`` segments per (noise type x range); each gets a distinct
    LibriSpeech utterance whose duration is nearest to a duration drawn from ``ref_durs``."""
    sp = recipe["splits"][split]
    rng = np.random.default_rng([sp["seed"], 1])
    used = set() if used is None else used
    pool = sorted(utts, key=lambda u: u.utt_id)
    durs = np.array([u.duration_s for u in pool])
    segs = []
    k = 0
    for nt in recipe["noise_types"]:
        for r in recipe["ranges_m"]:
            for j in range(recipe["n_per_cell"]):
                d = float(rng.choice(ref_durs))
                order = np.argsort(np.abs(durs - d), kind="stable")
                u = next(pool[i] for i in order if pool[i].utt_id not in used)
                used.add(u.utt_id)
                prm = cell_params(recipe, nt, np.random.default_rng([sp["seed"], 2, k]))
                kind = "near_silent" if nt == "near_silent" else "noise_only"
                segs.append({"seg_id": f"{split}_{nt}_{r:g}m_{j:03d}", "split": split, "idx": k, "noise_type": nt,
                             "range_m": r, "kind": kind, "utt_id": u.utt_id, "utt_path": u.path,
                             "dur_s": u.duration_s, "snr_db": prm["snr_db"], "over": prm["over"],
                             "att_db": prm.get("att_db", 0.0), "group": f"{nt}|{r:g}"})
                k += 1
    return segs


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def make_chain_cls():
    from uwse.uwsim.chain import TransducerChain

    class CapChain(TransducerChain):
        """Records the batched receive output (speech, noise, h_chain rows) of simulate()."""

        last = None

        def receive(self, p, hyd_eq=None):
            out = super().receive(p, hyd_eq)
            o = np.asarray(out)
            if o.ndim == 2 and o.shape[0] == 3:  # the batched (speech, noise, h_chain) call only
                self.last = o.copy()
            return out

    return CapChain


def build_scenes(recipe: dict, split: str):
    from sim_baseline import make_scenes

    sp = recipe["splits"][split]
    return make_scenes(str(ROOT / recipe["sim_config"]), sp["seed"], ranges=tuple(recipe["ranges_m"]),
                       depth_m=sp["scene_depth_m"], y_frac=sp["scene_y_frac"])


def render_segment(seg: dict, scenes: dict, rirs: dict, recipe: dict) -> dict:
    """Render one segment. Returns ``nonspeech`` (16 kHz float32), plus the paired speech
    rendering's ``noisy`` and ``R0`` (16 kHz) for the level gain. Fully determined by ``seg``."""
    from uwse import audio
    from uwse.uwsim.chain import Directivity, adc
    from uwse.uwsim.simulate import simulate

    sp = recipe["splits"][seg["split"]]
    scene = dataclasses.replace(scenes[seg["range_m"]], snr_db=seg["snr_db"], **seg["over"])
    x, fs = sf.read(seg["utt_path"], dtype="float64")
    s48 = audio.resample(x, fs, FS_SIM)
    chain = make_chain_cls()(fs=FS_SIM, directivity=Directivity.piston(c=scene.c))
    sim = simulate(s48, scene, None, np.random.default_rng([sp["seed"], seg["idx"], 0]), fs_model=FS_SCORE,
                   chain=chain, rir=rirs[seg["range_m"]])
    n_rx = chain.last[1]
    gain = 10.0 ** (sim["meta"]["gain_db"] / 20.0)
    ns, _ = adc(n_rx * gain, np.random.default_rng([sp["seed"], seg["idx"], 99]),
                scene.adc_floor_dbfs, scene.adc_bits)
    n_out = len(sim["noisy"])
    ns16 = audio.resample(ns, FS_SIM, FS_SCORE)
    ns16 = np.pad(ns16, (0, max(0, n_out - len(ns16))))[:n_out]
    return {"nonspeech": ns16.astype(np.float32), "noisy": sim["noisy"], "R0": sim["target_R0"],
            "clipped_nonspeech": bool(np.any(np.abs(ns) >= 1.0))}


def speech_gain(noisy: np.ndarray, r0: np.ndarray) -> float:
    from uwse.evaluation import normalize as N

    _, g, _ = N.normalize_cut(noisy.astype(np.float64), r0.astype(np.float64), FS_SCORE, vad=N.VADConfig())
    return float(g)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def select_utts(recipe: dict, ls_root: str):
    from uwse.bench.librispeech import build_manifest, filter_duration, find_subset_dir, select_v0_dev

    man = build_manifest(find_subset_dir(ls_root, recipe["librispeech_subset"]), recipe["librispeech_subset"],
                         progress=False)
    dev120 = select_v0_dev(man, n=120)
    ref_durs = [u.duration_s for u in dev120]
    excl = {u.utt_id for u in dev120} if recipe.get("exclude_dev_wer_v0") else set()
    return [u for u in filter_duration(man) if u.utt_id not in excl], ref_durs, excl


def plan_all(recipe: dict, ls_root: str) -> dict[str, list[dict]]:
    utts, ref_durs, _ = select_utts(recipe, ls_root)
    used: set[str] = set()
    plans = {}
    for split in ("dev", "test"):  # fixed order: test never reuses a dev utterance
        plans[split] = plan_segments(recipe, split, utts, ref_durs, used)
    return plans


def generate(recipe: dict, split: str, out: Path, ls_root: str, only: list[str] | None = None) -> list[dict]:
    from uwse.uwsim.simulate import render_rir

    segs = plan_all(recipe, ls_root)[split]
    scenes = build_scenes(recipe, split)
    rirs = {r: render_rir(s) for r, s in scenes.items()}
    wav_dir = out / split / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg in tqdm(segs, desc=f"render {split}"):
        if only is not None and seg["seg_id"] not in only:
            continue
        res = render_segment(seg, scenes, rirs, recipe)
        p = wav_dir / f'{seg["seg_id"]}.wav'
        sf.write(p, res["nonspeech"], FS_SCORE, subtype="FLOAT")
        rms = float(np.mean(res["nonspeech"].astype(np.float64) ** 2))
        rows.append({**seg, "wav": str(p.relative_to(out)).replace("\\", "/"), "n_samples": len(res["nonspeech"]),
                     "speech_gain": speech_gain(res["noisy"], res["R0"]), "clipped": res["clipped_nonspeech"],
                     "wav_sha256": sha256_file(p), "nonspeech_rms_dbfs": float(10 * np.log10(rms + 1e-30))})
    if only is None:
        rows = _assign_group_gains(rows)
        with open(out / split / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out / split / "manifest.sha256").write_text(sha256_file(out / split / "manifest.jsonl") + "\n")
    return rows


def _assign_group_gains(rows: list[dict]) -> list[dict]:
    """Frozen median rule: each non-speech row gets the median speech gain of its group."""
    from uwse.evaluation import normalize as N

    sp = [{"group": r["group"], "kind": "speech", "gain": r["speech_gain"]} for r in rows]
    ns = [{"group": r["group"], "kind": "noise", "seg_id": r["seg_id"]} for r in rows]
    filled = N.assign_gains(sp + ns, group_keys=("group",))[len(sp):]
    gm = {r["seg_id"]: r["gain"] for r in filled}
    return [dict(r, gain_median=gm[r["seg_id"]]) for r in rows]


def check_regen(recipe: dict, split: str, out: Path, ls_root: str, n: int = 5) -> bool:
    """Re-render ``n`` segments (spread over the plan) into a scratch dir; compare decoded samples."""
    man = [json.loads(line) for line in open(out / split / "manifest.jsonl", encoding="utf-8")]
    pick = [man[i]["seg_id"] for i in np.linspace(0, len(man) - 1, n).astype(int)]
    ref = {r["seg_id"]: r for r in man}
    ok = True
    # NB: float WAVs carry a libsndfile PEAK chunk with a creation timestamp, so file bytes differ between
    # runs (manifest wav_sha256 is of the file bytes); compare the decoded samples bit-for-bit instead.
    with tempfile.TemporaryDirectory(dir=str(out)) as td:
        rows = generate(recipe, split, Path(td), ls_root, only=pick)
        ok &= len(rows) == n
        for r in rows:
            a, _ = sf.read(out / ref[r["seg_id"]]["wav"], dtype="float32")
            b, _ = sf.read(Path(td) / r["wav"], dtype="float32")
            same = a.shape == b.shape and bool(np.array_equal(a, b))
            ok &= same
            d = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
            print(f'{r["seg_id"]}: regen samples vs stored max|diff|={d:.3g} -> {"IDENTICAL" if same else "DIFFERENT"}')
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recipe", default=str(ROOT / "configs" / "nonspeech_v0.yaml"))
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "nonspeech_v0"))
    ap.add_argument("--librispeech", default=str(ROOT / "data" / "librispeech"))
    ap.add_argument("--check-regen", type=int, default=0, help="re-render N stored segments and compare hashes")
    a = ap.parse_args(argv)
    recipe, out = load_recipe(a.recipe), Path(a.out)
    if "datasets" in out.resolve().parts[:3]:
        raise SystemExit("refusing to write under the protected datasets directory")
    if a.check_regen:
        return 0 if check_regen(recipe, a.split, out, a.librispeech, a.check_regen) else 1
    rows = generate(recipe, a.split, out, a.librispeech)
    print(f"{a.split}: {len(rows)} segments; manifest sha256 "
          f"{(out / a.split / 'manifest.sha256').read_text().strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
