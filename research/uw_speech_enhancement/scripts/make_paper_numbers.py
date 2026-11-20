"""Paper numbers from a score_session.py results dir: LaTeX macros + the real-data figure.

    python scripts/make_paper_numbers.py --results runs/score/v1_s2
    # -> paper/results_real.tex  and  paper/fig_real.pdf

The uwse env has no matplotlib: there the figure step exits 3 and prints the command to make
it with a python that has matplotlib (the script needs only the stdlib, numpy and matplotlib),
as paper/fig_wer.py is built::

    C:/Users/ryota/anaconda3/python.exe scripts/make_paper_numbers.py --results runs/score/v1_s2 --fig-only

In the paper: ``\\input{results_real}`` in the preamble, then e.g. ``\\PrimWerCZeroThree``.
Every number comes from score.py's own outputs (``<cond>/all/summary.csv`` for WER and its
speaker-cluster 95% CI at 10000 resamples, ``<cond>/paired_<subset>/paired.csv`` for the paired
delta at 2000 resamples, ``<cond>/all/per_utt.jsonl`` for utterance and speaker counts) and the
pilot / chain-reference records of scripts/pilot_wer.py. Nothing is recomputed.

Macro names (LaTeX names cannot hold digits, so numbers are spelled: C0 -> CZero, C8 -> CEight,
C10 -> CTen, 3 m -> Three, 0.5 m -> ZeroPtFive, pool0dB -> PoolZero, quiet -> Quiet; ``All`` is
every distance pooled). ``<S>`` system, ``<C>`` condition, ``<D>`` distance:

    \\Wer<S><C><D>[Lo|Hi|CI]     corpus WER %, its 95% CI bounds, "[lo, hi]"
    \\DWer<S><C><D>[Lo|Hi|CI]    paired WER difference S - C0, percentage points (signed)
    \\RelDWer<S><C><D>           relative difference, % of the C0 WER (signed)
    \\NUtts<C><D>  \\NSpk<C><D>   scored utterances and speakers
    \\PrimCond \\PrimCondName \\SecCond \\SecCondName      primary / secondary condition (pilot rule)
    \\PrimWer<S><D>[Lo|Hi|CI] for S in CZero, CEight; \\PrimDWer<D>[Lo|Hi|CI]; \\PrimRelDWer<D>
    \\SecWer...  \\SecDWer...  \\SecRelDWer...                same for the secondary condition
    \\PilotWer[Lo|Hi] \\PilotN \\PilotNSpk \\PilotThreshold \\PilotDecision \\PilotDist
    \\ChainRefWer[Lo|Hi] \\ChainRefN \\ChainRefNSpk \\ChainRefDist
    \\NDist \\DistList \\NUttsPerDist \\NSpk \\NBootCI \\NBootPaired \\CEightSha

The full list with values is written at the top of the generated file. A number that is not
available yet is typeset as a bold ``??``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MISSING = r"\textbf{??}"
WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven",
         "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty"]
SYSTEM_ORDER = ["C0", "C1", "C2", "C10", "C8"]
SYSTEM_LABEL = {"C0": "Unprocessed", "C1": "WPE", "C2": "DFN3 fine-tuned", "C10": "U-Net", "C8": "Whisper + LoRA"}
# validated default categorical palette, fixed order (dataviz validator: all checks pass, light)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"]


# ----------------------------------------------------------------------------- naming
def numword(s: str | float) -> str:
    """3 -> Three, 0.5 -> ZeroPtFive, 12 -> Twelve, 25 -> TwoFive."""
    if isinstance(s, float):
        s = f"{s:g}"
    s = str(s)
    if s.isdigit() and int(s) < len(WORDS):
        return WORDS[int(s)]
    return "".join(WORDS[int(c)] if c.isdigit() else "Pt" if c == "." else "Minus" if c == "-" else "" for c in s)


def camel(name: str) -> str:
    """Condition / system name -> letters only: pool0dB -> PoolZero, C10 -> CTen, bubble-5dB -> BubbleMinusFive."""
    name = re.sub(r"dB$", "", name)
    out = []
    for tok in re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|-", name):
        out.append(numword(tok) if not tok.isalpha() else tok[0].upper() + tok[1:])
    res = "".join(out)
    if not res.isalpha():
        raise ValueError(f"cannot make a macro name from {name!r}")
    return res


def dist_word(pos: str, distances: dict[str, float]) -> str:
    return "All" if pos == "ALL" else numword(float(distances[pos]))


# ----------------------------------------------------------------------------- reading
def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def _csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def collect(results: Path) -> dict:
    """Everything the macros and the figure need, from a score_session.py results dir."""
    idx = json.loads((results / "index.json").read_text(encoding="utf-8"))
    out = {"index": idx, "positions": idx["positions"], "distances": idx["distances_m"], "conds": {}}
    for cond, ci in idx["conditions"].items():
        d = results / cond
        wer, dwer, n_utts, n_spk = {}, {}, {}, {}
        for r in _csv(d / "all" / "summary.csv"):
            pos = r["position"]
            key = (r["system"], pos)
            if key in wer:
                raise SystemExit(f"{cond}: several thruster states at {pos}; not a v1 pool session")
            wer[key] = {"wer": _f(r.get("wer")), "lo": _f(r.get("wer_lo")), "hi": _f(r.get("wer_hi")),
                        "n": int(_f(r.get("n_utts")) or 0)}
        for sub in ["ALL", *ci["positions"]]:
            for r in _csv(d / f"paired_{sub}" / "paired.csv"):
                if r.get("baseline") == "C0":
                    dwer[(r["system"], sub)] = {"d": _f(r.get("d_wer")), "lo": _f(r.get("d_wer_lo")),
                                                "hi": _f(r.get("d_wer_hi")), "n": int(_f(r.get("n_utts")) or 0)}
        per = d / "all" / "per_utt.jsonl"
        if per.exists():
            c0 = [json.loads(l) for l in open(per, encoding="utf-8") if l.strip()]
            c0 = [r for r in c0 if r["system"] == "C0" and r.get("kind", "speech") == "speech"]
            for sub in ["ALL", *ci["positions"]]:
                rs = [r for r in c0 if sub == "ALL" or str(r["position"]) == sub]
                n_utts[sub] = len(rs)
                n_spk[sub] = len({str(r["speaker"]) for r in rs})
        out["conds"][cond] = {"wer": wer, "dwer": dwer, "n_utts": n_utts, "n_spk": n_spk,
                              "positions": sorted(ci["positions"], key=lambda p: idx["distances_m"][p]),
                              "added_noise": ci.get("added_noise", False), "systems": ci["systems"]}
    for key in ("pilot", "chainref"):
        out[key] = json.loads((results / idx[key]).read_text(encoding="utf-8")) if idx.get(key) else None
    return out


def primary_secondary(nums: dict, noise_kind: str = "pool") -> tuple[str | None, str | None]:
    """Pre-registered rule (pilot C0 WER >= 10% -> quiet primary, else <kind>0dB primary)."""
    added = f"{noise_kind}0dB"
    p = nums.get("pilot")
    if not p or "decision" not in p:
        return None, None
    if p["decision"].startswith("QUIET"):
        return "quiet", added
    return added, "quiet"


def cond_text(cond: str | None) -> str:
    if cond is None:
        return MISSING
    if cond == "quiet":
        return "quiet"
    m = re.fullmatch(r"([A-Za-z]+)(-?\d+(?:\.\d+)?)dB", cond)
    if m:
        return f"added recorded {m.group(1)} noise at {m.group(2)}~dB SNR"
    return cond


def short_title(cond: str) -> str:
    """Figure panel title: Quiet, Pool noise added, 0 dB SNR, ..."""
    if cond == "quiet":
        return "Quiet"
    m = re.fullmatch(r"([A-Za-z]+)(-?\d+(?:\.\d+)?)dB", cond)
    return f"{m.group(1).capitalize()} noise added, {m.group(2)} dB SNR" if m else cond


# ----------------------------------------------------------------------------- formatting
def pct(v: float) -> str:
    return MISSING if not math.isfinite(v) else f"{100 * v:.1f}"


def spct(v: float, digits: int = 1) -> str:
    if not math.isfinite(v):
        return MISSING
    s = f"{100 * v:+.{digits}f}"
    return rf"\ensuremath{{{s}}}"


def ci(lo: str, hi: str) -> str:
    return f"[{lo}, {hi}]"


def build_macros(nums: dict, noise_kind: str = "pool") -> list[tuple[str, str, str]]:
    """[(macro name without backslash, value, description)]."""
    m: list[tuple[str, str, str]] = []
    dist = nums["distances"]
    subsets = ["ALL", *nums["positions"]]

    def add(name, value, desc):
        m.append((name, value, desc))

    def wer_block(prefix, cond, sysname, pos, desc):
        w = nums["conds"].get(cond, {}).get("wer", {}).get((sysname, pos))
        v = w or {"wer": math.nan, "lo": math.nan, "hi": math.nan}
        add(prefix, pct(v["wer"]), f"{desc}: corpus WER %")
        add(prefix + "Lo", pct(v["lo"]), f"{desc}: WER 95% CI low")
        add(prefix + "Hi", pct(v["hi"]), f"{desc}: WER 95% CI high")
        add(prefix + "CI", ci(pct(v["lo"]), pct(v["hi"])), f"{desc}: WER 95% CI")

    def dwer_block(prefix, rel, cond, sysname, pos, desc):
        c = nums["conds"].get(cond, {})
        d = c.get("dwer", {}).get((sysname, pos)) or {"d": math.nan, "lo": math.nan, "hi": math.nan}
        base = c.get("wer", {}).get(("C0", pos), {}).get("wer", math.nan)
        add(prefix, spct(d["d"]), f"{desc}: WER {sysname} - C0, points")
        add(prefix + "Lo", spct(d["lo"]), f"{desc}: delta 95% CI low")
        add(prefix + "Hi", spct(d["hi"]), f"{desc}: delta 95% CI high")
        add(prefix + "CI", ci(spct(d["lo"]), spct(d["hi"])), f"{desc}: delta 95% CI")
        relv = d["d"] / base if math.isfinite(d["d"]) and math.isfinite(base) and base > 0 else math.nan
        add(rel, spct(relv, 0), f"{desc}: delta relative to the C0 WER, %")

    for cond, c in nums["conds"].items():
        cw = camel(cond)
        systems = sorted({s for s, _ in c["wer"]}, key=lambda s: (SYSTEM_ORDER + [s]).index(s))
        for pos in ["ALL", *c.get("positions", nums["positions"])]:
            dw = dist_word(pos, dist)
            where = "all distances" if pos == "ALL" else f"{dist[pos]:g} m"
            for s in systems:
                sw = camel(s)
                wer_block(f"Wer{sw}{cw}{dw}", cond, s, pos, f"{s}, {cond}, {where}")
                if s != "C0":
                    dwer_block(f"DWer{sw}{cw}{dw}", f"RelDWer{sw}{cw}{dw}", cond, s, pos, f"{s}, {cond}, {where}")
            add(f"NUtts{cw}{dw}", str(c["n_utts"].get(pos, MISSING)), f"{cond}, {where}: scored utterances")
            add(f"NSpk{cw}{dw}", str(c["n_spk"].get(pos, MISSING)), f"{cond}, {where}: speakers")

    prim, sec = primary_secondary(nums, noise_kind)
    for tag, cond in (("Prim", prim), ("Sec", sec)):
        name = "primary" if tag == "Prim" else "secondary"
        add(f"{tag}Cond", cond_text(cond), f"{name} condition (pre-registered pilot rule), text")
        add(f"{tag}CondName", cond or MISSING, f"{name} condition, results-dir name")
        for pos in subsets:
            dw = dist_word(pos, dist)
            where = "all distances" if pos == "ALL" else f"{dist[pos]:g} m"
            for s in ("C0", "C8"):
                wer_block(f"{tag}Wer{camel(s)}{dw}", cond, s, pos, f"{name}, {s}, {where}")
            dwer_block(f"{tag}DWer{dw}", f"{tag}RelDWer{dw}", cond, "C8", pos, f"{name}, C8, {where}")

    p = nums.get("pilot") or {}
    add("PilotWer", pct(_f(p.get("wer"))), "pilot C0 corpus WER %")
    add("PilotWerLo", pct(_f(p.get("wer_lo"))), "pilot WER 95% CI low")
    add("PilotWerHi", pct(_f(p.get("wer_hi"))), "pilot WER 95% CI high")
    add("PilotN", str(p.get("n_utts", MISSING)), "pilot utterances")
    add("PilotNSpk", str(p.get("n_speakers", MISSING)), "pilot speakers")
    add("PilotThreshold", f"{100 * p['threshold']:g}" if "threshold" in p else "10", "decision threshold, % WER")
    add("PilotDecision", {"QUIET primary": "quiet primary", "ADDED-NOISE primary": "added noise primary"}.get(
        p.get("decision"), MISSING), "pilot decision, text")
    add("PilotDist", _pos_dist(p, dist), "pilot distance, m")
    c = nums.get("chainref") or {}
    add("ChainRefWer", pct(_f(c.get("wer"))), "chain-reference take, C0 corpus WER %")
    add("ChainRefWerLo", pct(_f(c.get("wer_lo"))), "chain-reference WER 95% CI low")
    add("ChainRefWerHi", pct(_f(c.get("wer_hi"))), "chain-reference WER 95% CI high")
    add("ChainRefN", str(c.get("n_utts", MISSING)), "chain-reference utterances")
    add("ChainRefNSpk", str(c.get("n_speakers", MISSING)), "chain-reference speakers")
    add("ChainRefDist", _pos_dist(c, dist), "chain-reference distance, m")

    ds = [f"{dist[p_]:g}" for p_ in nums["positions"]]
    add("NDist", str(len(ds)), "number of distances")
    add("DistList", ", ".join(ds[:-1]) + (" and " if len(ds) > 1 else "") + ds[-1] if ds else MISSING,
        "distances in m, e.g. '1, 3 and 6'")
    q = nums["conds"].get("quiet", {})
    pc = nums["conds"].get(prim or "quiet", q)  # the primary condition's scored set (quiet if unknown)
    per = [pc["n_utts"][p_] for p_ in nums["positions"] if p_ in pc.get("n_utts", {})]
    add("NUttsPerDist", (str(per[0]) if min(per) == max(per) else f"{min(per)}--{max(per)}") if per else MISSING,
        "primary-condition test utterances per distance (a min--max range if they differ)")
    add("NSpk", str(q.get("n_spk", {}).get("ALL", MISSING)), "quiet test speakers, all distances")
    idx = nums["index"]
    add("NBootCI", f"{idx.get('n_boot_ci', MISSING)}", "bootstrap resamples for WER CIs")
    add("NBootPaired", f"{idx.get('n_boot_paired', MISSING)}", "bootstrap resamples for paired deltas")
    add("CEightSha", str(idx.get("c8_sha256", ""))[:8] or MISSING, "C8 checkpoint sha256 prefix")
    names = [n for n, _, _ in m]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"macro name collision: {sorted(dup)}")
    return m


def _pos_dist(rec: dict, dist: dict[str, float]) -> str:
    ps = rec.get("positions") or []
    if len(ps) != 1:
        return MISSING
    p_ = ps[0]
    if p_ in dist:
        return f"{dist[p_]:g}"
    mm = re.fullmatch(r"p?(\d+)(?:[.p](\d+))?[A-Za-z_]*", p_)
    return (mm.group(1) + ("." + mm.group(2) if mm.group(2) else "")) if mm else MISSING


def write_tex(path: Path, macros: list[tuple[str, str, str]], source: Path) -> None:
    lines = ["% Generated by scripts/make_paper_numbers.py from " + source.as_posix() + "; do not edit.",
             "% Real pool data (pool plan 2026-11-19). WER in %, deltas in percentage points (S - C0).",
             "% Macros:"]
    lines += [f"%   \\{n:<40s} {desc}" for n, _, desc in macros]
    lines.append("")
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v, _ in macros]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- figure
def make_figure(nums: dict, path: Path, conds: list[str] | None = None, noise_kind: str = "pool") -> Path:
    """WER by distance per system with 95% CIs, one panel per condition (default: quiet and the
    added-noise primary candidate); the chain-reference WER as a dashed line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if conds is None:
        conds = [c for c in ("quiet", f"{noise_kind}0dB") if c in nums["conds"]] or list(nums["conds"])[:2]
    pos = nums["positions"]
    allsys = list(dict.fromkeys(s for c in conds for s, _ in nums["conds"][c]["wer"]))
    systems = sorted(allsys, key=lambda s: ((SYSTEM_ORDER + allsys).index(s)))
    # colour follows the system, never its rank; C8 keeps slot 5 whatever subset is shown
    col = {s: COLORS[(SYSTEM_ORDER + [x for x in allsys if x not in SYSTEM_ORDER]).index(s) % len(COLORS)]
           for s in systems}
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(conds), figsize=(3.45, 1.9), constrained_layout=True, squeeze=False)
    x = np.arange(len(pos))
    w = 0.8 / max(len(systems), 1)
    chain = (nums.get("chainref") or {}).get("wer")
    for ax, cond in zip(axes[0], conds):
        wer = nums["conds"][cond]["wer"]
        top = 0.0
        for k, s in enumerate(systems):
            v = [wer.get((s, p), {}) for p in pos]
            y = np.array([100 * r.get("wer", np.nan) for r in v])
            lo = np.array([100 * r.get("lo", np.nan) for r in v])
            hi = np.array([100 * r.get("hi", np.nan) for r in v])
            xs = x + (k - (len(systems) - 1) / 2) * w
            ax.bar(xs, y, w * 0.9, color=col[s], linewidth=0)
            ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            ax.errorbar(xs[ok], y[ok], yerr=[y[ok] - lo[ok], hi[ok] - y[ok]], fmt="none", ecolor="#52514e",
                        elinewidth=0.5, capsize=1)
            if np.isfinite(hi).any():
                top = max(top, np.nanmax(hi))
        if chain is not None and math.isfinite(chain):
            ax.axhline(100 * chain, color="#52514e", linewidth=0.6, linestyle="--")
            top = max(top, 100 * chain)
        ax.set_xticks(x, [f"{nums['distances'][p]:g} m" for p in pos])
        ax.set_title(short_title(cond), fontsize=7)
        ax.tick_params(length=2)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(top * 1.08, 1.0))
    axes[0][0].set_ylabel("WER (%)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Patch(color=col[s], linewidth=0) for s in systems]
    labels = [SYSTEM_LABEL.get(s, s) for s in systems]
    if chain is not None and math.isfinite(chain):
        handles.append(Line2D([], [], color="#52514e", linewidth=0.6, linestyle="--"))
        labels.append("Chain reference")
    fig.legend(handles, labels, loc="outside upper center", ncol=min(len(labels), 3), fontsize=5.5,
               frameon=False, handlelength=1, columnspacing=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="score_session.py --out")
    ap.add_argument("--tex", type=Path, default=ROOT / "paper" / "results_real.tex")
    ap.add_argument("--fig", type=Path, default=ROOT / "paper" / "fig_real.pdf")
    ap.add_argument("--fig-conds", default=None, help="comma list of conditions to plot")
    ap.add_argument("--no-fig", action="store_true", help="macros only")
    ap.add_argument("--fig-only", action="store_true", help="figure only (e.g. from a python with matplotlib)")
    a = ap.parse_args(argv)
    nums = collect(a.results)
    # the added-noise primary candidate is pool noise at 0 dB (pre-registered; no option to change it)
    prim, _ = primary_secondary(nums)
    if not a.fig_only:
        macros = build_macros(nums)
        write_tex(a.tex, macros, a.results)
        print(f"{len(macros)} macros -> {a.tex}; primary condition: {prim or 'unknown (no pilot)'}")
    if a.no_fig:
        return 0
    try:
        make_figure(nums, a.fig, a.fig_conds.split(",") if a.fig_conds else None)
    except ImportError:
        # the uwse env has no matplotlib; this script needs only the stdlib, numpy and matplotlib
        print(f"[error] no matplotlib here; make the figure with a python that has it, e.g.\n"
              f"  C:/Users/ryota/anaconda3/python.exe scripts/make_paper_numbers.py --results {a.results} "
              f"--fig {a.fig} --fig-only")
        return 3
    print(f"figure -> {a.fig}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
