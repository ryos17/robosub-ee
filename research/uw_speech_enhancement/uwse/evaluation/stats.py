"""Statistics for the benchmark: cluster bootstrap, paired differences, TOST,
Holm, MAPSSWE and Kendall tau.

Implements RESEARCH_PLAN.md §8.5 "Statistics":

- **Unit of analysis (v0).** Cluster bootstrap over speakers within
  position x thruster strata, 10,000 replicates: clusters are resampled with
  replacement *within* each stratum, every utterance of a drawn cluster comes
  along. The CIs are valid only for the recorded positions and session.
- **Corpus WER** is a ratio estimator, sum(errors) / sum(reference words), and
  is bootstrapped as such (never the mean of per-utterance WERs). Mean
  metrics are the same estimator with denominator 1 per utterance.
- **Paired differences** reuse the same resampled clusters for both systems.
- **Nulls** via TOST equivalence / non-inferiority with pre-registered margins.
- **Multiplicity** via Holm.
- **WER significance cross-check** via MAPSSWE (Gillick & Cox 1989; NIST
  sc_stats; Bisani & Ney 2004).
- **Correlations (RQ8)** via Kendall tau with a cluster-bootstrap CI
  (clusters = systems).

All resampling is seeded (``numpy.random.default_rng(seed)``) and
vectorized as a (B x n_clusters) multiplicity matrix.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats as sps

__all__ = [
    "BootstrapResult",
    "ClusterDesign",
    "cluster_design",
    "bootstrap_weights",
    "corpus_ratio",
    "cluster_bootstrap_ratio",
    "cluster_bootstrap_wer",
    "cluster_bootstrap_mean",
    "paired_cluster_bootstrap_ratio",
    "paired_bootstrap_wer",
    "paired_bootstrap_mean",
    "tost",
    "tost_normal",
    "noninferiority",
    "holm",
    "mapsswe",
    "mapsswe_segments",
    "kendall_tau_bootstrap",
]

DEFAULT_N_BOOT = 10_000


@dataclass
class BootstrapResult:
    """Point estimate, percentile CI and bootstrap SE."""

    estimate: float
    ci_low: float
    ci_high: float
    se: float
    n_boot: int
    alpha: float
    replicates: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "se": self.se,
            "n_boot": self.n_boot,
            "alpha": self.alpha,
        }


@dataclass
class ClusterDesign:
    """Utterance -> cluster map with each cluster's stratum."""

    cluster_of_utt: np.ndarray  # (n_utts,) int in [0, n_clusters)
    stratum_of_cluster: np.ndarray  # (n_clusters,) int in [0, n_strata)
    n_clusters: int
    n_strata: int

    def cluster_sums(self, v: np.ndarray) -> np.ndarray:
        return np.bincount(self.cluster_of_utt, weights=np.asarray(v, np.float64), minlength=self.n_clusters)


def cluster_design(clusters: Sequence, strata: Sequence | None = None) -> ClusterDesign:
    """Build the design. A cluster is a (stratum, cluster label) pair, so the
    same speaker in two position x thruster strata forms two clusters."""
    # str() per element so tuple labels such as (position, thruster) stay 1-D
    c_str = np.array([str(c) for c in clusters], dtype=str)
    s_str = np.array(["0"] * len(c_str) if strata is None else [str(s) for s in strata], dtype=str)
    if len(s_str) != len(c_str):
        raise ValueError("clusters and strata differ in length")
    s_lab, s_idx = np.unique(s_str, return_inverse=True)
    pair = np.char.add(np.char.add(s_idx.astype(str), "\x1f"), c_str)
    c_lab, c_idx = np.unique(pair, return_inverse=True)
    stratum_of_cluster = np.zeros(len(c_lab), dtype=np.int64)
    stratum_of_cluster[c_idx] = s_idx
    return ClusterDesign(c_idx.astype(np.int64), stratum_of_cluster, len(c_lab), len(s_lab))


def bootstrap_weights(design: ClusterDesign, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_clusters) multiplicities: clusters drawn with replacement within strata."""
    W = np.zeros((n_boot, design.n_clusters), dtype=np.float64)
    n_single = 0
    for s in range(design.n_strata):
        members = np.flatnonzero(design.stratum_of_cluster == s)
        k = len(members)
        if k == 0:
            continue
        n_single += k == 1
        draw = rng.integers(0, k, size=(n_boot, k))
        flat = (draw + np.arange(n_boot)[:, None] * k).ravel()
        counts = np.bincount(flat, minlength=n_boot * k).reshape(n_boot, k)
        W[:, members] = counts
    if n_single:
        # a stratum with one cluster contributes zero bootstrap variance: the CI is too narrow
        warnings.warn(
            f"{n_single} of {design.n_strata} strata contain a single cluster; their between-cluster "
            "variance is not represented and the CI is too narrow (missing speaker labels?)"
        )
    return W


def corpus_ratio(num: np.ndarray, den: np.ndarray) -> float:
    """sum(num) / sum(den), e.g. corpus WER = summed errors / summed ref words."""
    d = float(np.sum(den))
    return float(np.sum(num)) / d if d > 0 else float("nan")


def _summarize(est: float, reps: np.ndarray, alpha: float, keep: bool) -> BootstrapResult:
    reps = reps[np.isfinite(reps)]
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2]) if reps.size else (np.nan, np.nan)
    return BootstrapResult(
        float(est), float(lo), float(hi), float(np.std(reps, ddof=1)) if reps.size > 1 else float("nan"),
        int(reps.size), alpha, reps if keep else np.empty(0),
    )


def _clean(*arrays: np.ndarray) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(np.asarray(a, np.float64))
    return ok


def cluster_bootstrap_ratio(
    num: Sequence[float],
    den: Sequence[float],
    clusters: Sequence,
    strata: Sequence | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    keep_replicates: bool = False,
) -> BootstrapResult:
    """Stratified cluster bootstrap of the ratio estimator sum(num)/sum(den).

    Utterances with non-finite num or den are dropped first.
    """
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    ok = _clean(num, den)
    d = cluster_design(np.asarray(clusters, object)[ok], None if strata is None else np.asarray(strata, object)[ok])
    cn, cd = d.cluster_sums(num[ok]), d.cluster_sums(den[ok])
    W = bootstrap_weights(d, n_boot, np.random.default_rng(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (W @ cn) / (W @ cd)
    return _summarize(corpus_ratio(cn, cd), reps, alpha, keep_replicates)


def cluster_bootstrap_wer(errors, n_ref, clusters, strata=None, **kw) -> BootstrapResult:
    """Corpus WER CI. ``errors`` = S+D+I per utterance, ``n_ref`` = reference words."""
    return cluster_bootstrap_ratio(errors, n_ref, clusters, strata, **kw)


def cluster_bootstrap_mean(values, clusters, strata=None, **kw) -> BootstrapResult:
    """CI of the utterance-weighted mean of a per-utterance metric (NaNs dropped)."""
    v = np.asarray(values, np.float64)
    return cluster_bootstrap_ratio(v, np.ones_like(v), clusters, strata, **kw)


def paired_cluster_bootstrap_ratio(
    num_a: Sequence[float],
    den_a: Sequence[float],
    num_b: Sequence[float],
    den_b: Sequence[float],
    clusters: Sequence,
    strata: Sequence | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    keep_replicates: bool = True,
) -> BootstrapResult:
    """CI of ratio_A - ratio_B with the same resampled clusters for both systems.

    Inputs are aligned per utterance (same utterance, two systems). Utterances
    non-finite in either system are dropped from both.
    """
    arrs = [np.asarray(a, np.float64) for a in (num_a, den_a, num_b, den_b)]
    ok = _clean(*arrs)
    d = cluster_design(np.asarray(clusters, object)[ok], None if strata is None else np.asarray(strata, object)[ok])
    na, da, nb, db = (d.cluster_sums(a[ok]) for a in arrs)
    W = bootstrap_weights(d, n_boot, np.random.default_rng(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (W @ na) / (W @ da) - (W @ nb) / (W @ db)
    est = corpus_ratio(na, da) - corpus_ratio(nb, db)
    return _summarize(est, reps, alpha, keep_replicates)


def paired_bootstrap_wer(err_a, err_b, n_ref, clusters, strata=None, **kw) -> BootstrapResult:
    """CI of WER_A - WER_B (same references, paired clusters)."""
    return paired_cluster_bootstrap_ratio(err_a, n_ref, err_b, n_ref, clusters, strata, **kw)


def paired_bootstrap_mean(val_a, val_b, clusters, strata=None, **kw) -> BootstrapResult:
    """CI of mean_A - mean_B of a per-utterance metric (paired clusters)."""
    a, b = np.asarray(val_a, np.float64), np.asarray(val_b, np.float64)
    one = np.ones_like(a)
    return paired_cluster_bootstrap_ratio(a, one, b, one, clusters, strata, **kw)


# ---------------------------------------------------------------------------
# Equivalence / non-inferiority
# ---------------------------------------------------------------------------
def tost(
    replicates: np.ndarray,
    estimate: float,
    low: float,
    high: float,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap TOST: equivalent iff the (1 - 2 alpha) percentile CI lies in (low, high).

    One-sided bootstrap p-values: p_low = P*(diff <= low), p_high =
    P*(diff >= high); p = max(p_low, p_high). With a percentile CI, p < alpha
    is equivalent to the CI criterion.
    """
    r = np.asarray(replicates, np.float64)
    r = r[np.isfinite(r)]
    lo, hi = np.quantile(r, [alpha, 1 - alpha])
    p_low = float(np.mean(r <= low))
    p_high = float(np.mean(r >= high))
    return {
        "estimate": float(estimate),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": 1 - 2 * alpha,
        "margin_low": low,
        "margin_high": high,
        "p_low": p_low,
        "p_high": p_high,
        "p": max(p_low, p_high),
        "equivalent": bool(lo > low and hi < high),
    }


def tost_normal(estimate: float, se: float, low: float, high: float, alpha: float = 0.05, df: float | None = None) -> dict:
    """Normal-theory (or t with ``df``) TOST on an estimate and its standard error."""
    dist = sps.t(df) if df is not None else sps.norm()
    t_low = (estimate - low) / se
    t_high = (estimate - high) / se
    p_low = float(dist.sf(t_low))  # H0: diff <= low
    p_high = float(dist.cdf(t_high))  # H0: diff >= high
    q = dist.ppf(1 - alpha)
    return {
        "estimate": estimate,
        "ci_low": estimate - q * se,
        "ci_high": estimate + q * se,
        "ci_level": 1 - 2 * alpha,
        "p_low": p_low,
        "p_high": p_high,
        "p": max(p_low, p_high),
        "equivalent": bool(max(p_low, p_high) < alpha),
    }


def noninferiority(
    replicates: np.ndarray,
    estimate: float,
    margin: float,
    alpha: float = 0.05,
    lower_is_better: bool = True,
) -> dict:
    """One-sided non-inferiority of diff = new - baseline with margin > 0.

    lower_is_better (WER): non-inferior iff the upper (1 - alpha) bound < margin.
    higher_is_better (ESTOI): non-inferior iff the lower bound > -margin.
    """
    r = np.asarray(replicates, np.float64)
    r = r[np.isfinite(r)]
    if lower_is_better:
        bound = float(np.quantile(r, 1 - alpha))
        p = float(np.mean(r >= margin))
        ok = bound < margin
    else:
        bound = float(np.quantile(r, alpha))
        p = float(np.mean(r <= -margin))
        ok = bound > -margin
    return {"estimate": float(estimate), "bound": bound, "margin": margin, "p": p, "noninferior": bool(ok)}


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down correction. Returns (reject, adjusted p) in input order."""
    p = np.asarray(pvalues, np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= alpha, adj


# ---------------------------------------------------------------------------
# MAPSSWE
# ---------------------------------------------------------------------------
def mapsswe_segments(
    ref_err_a: np.ndarray,
    ins_a: np.ndarray,
    ref_err_b: np.ndarray,
    ins_b: np.ndarray,
    guard: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one utterance into MAPSSWE segments; return per-segment errors (A, B).

    Inputs per reference word: ``ref_err_*`` (substitution or deletion) and
    per gap (len + 1): ``ins_*`` (insertions before word g; gap N is after the
    last word), e.g. from :func:`uwse.evaluation.asr.word_alignment`.

    Following NIST sc_stats, segments are separated by runs of at least
    ``guard`` reference words that both systems recognized correctly with no
    insertion between them; error-free segments are dropped.
    """
    ea, eb = np.asarray(ref_err_a, bool), np.asarray(ref_err_b, bool)
    ia, ib = np.asarray(ins_a, np.int64), np.asarray(ins_b, np.int64)
    if guard < 2:
        raise ValueError("guard must be >= 2 words")
    n = len(ea)
    ok = ~ea & ~eb
    # inner gap g (1..n-1) sits between words g-1 and g; it is "clean" if both
    # neighbours are correct for both systems and neither system inserts there.
    # `guard` consecutive clean words <=> `guard - 1` consecutive clean gaps.
    clean = np.zeros(n + 1, dtype=bool)
    if n > 1:
        g = np.arange(1, n)
        clean[g] = ok[g - 1] & ok[g] & (ia[g] == 0) & (ib[g] == 0)
    cut_gaps = []  # one boundary per qualifying run of clean gaps (its first gap)
    need = guard - 1
    g = 1
    while g < n:
        if not clean[g]:
            g += 1
            continue
        h = g
        while h < n and clean[h]:
            h += 1
        if h - g >= need:
            cut_gaps.append(g)
        g = h
    bounds = [0] + cut_gaps + [n + 1]  # gap ranges [bounds[i], bounds[i+1])
    seg_a, seg_b = [], []
    for s, t in zip(bounds[:-1], bounds[1:]):
        w0, w1 = s, min(t, n)  # words s .. t-1 belong to the segment
        na = int(ea[w0:w1].sum() + ia[s:t].sum())
        nb = int(eb[w0:w1].sum() + ib[s:t].sum())
        if na or nb:
            seg_a.append(na)
            seg_b.append(nb)
    return np.asarray(seg_a, np.int64), np.asarray(seg_b, np.int64)


def mapsswe(seg_err_a: Sequence[int], seg_err_b: Sequence[int]) -> dict:
    """Matched-pairs sentence-segment word error test (Gillick & Cox 1989).

    Z_i = N_A,i - N_B,i over segments; W = mean(Z) / (sd(Z)/sqrt(n)) is
    approximately standard normal under H0 (equal expected errors); two-sided
    p-value. Segments may come from :func:`mapsswe_segments` or be whole
    utterances.
    """
    z = np.asarray(seg_err_a, np.float64) - np.asarray(seg_err_b, np.float64)
    n = len(z)
    if n < 2:
        return {"n_segments": n, "mean_diff": float(z.mean()) if n else float("nan"), "w": float("nan"), "p": float("nan")}
    mu, sd = z.mean(), z.std(ddof=1)
    if sd == 0:
        w = 0.0 if mu == 0 else np.sign(mu) * np.inf
    else:
        w = mu / (sd / np.sqrt(n))
    p = float(2 * sps.norm.sf(abs(w))) if np.isfinite(w) else 0.0
    return {"n_segments": n, "mean_diff": float(mu), "w": float(w), "p": p}


# ---------------------------------------------------------------------------
# Kendall tau
# ---------------------------------------------------------------------------
def kendall_tau_bootstrap(
    x: Sequence[float],
    y: Sequence[float],
    clusters: Sequence,
    strata: Sequence | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
    progress: bool = False,
) -> BootstrapResult:
    """Kendall tau-b of (x, y) with a cluster-bootstrap percentile CI.

    For RQ8 the clusters are systems (§8.5); every replicate resamples whole
    clusters and recomputes tau on the pooled points.
    """
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = _clean(x, y)
    x, y = x[ok], y[ok]
    d = cluster_design(np.asarray(clusters, object)[ok], None if strata is None else np.asarray(strata, object)[ok])
    est = sps.kendalltau(x, y).statistic
    W = bootstrap_weights(d, n_boot, np.random.default_rng(seed)).astype(np.int64)
    members = [np.flatnonzero(d.cluster_of_utt == c) for c in range(d.n_clusters)]
    reps = np.empty(n_boot)
    it = range(n_boot)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, desc="kendall bootstrap")
    for b in it:
        idx = np.concatenate([np.repeat(members[c], W[b, c]) for c in np.flatnonzero(W[b])])
        reps[b] = sps.kendalltau(x[idx], y[idx]).statistic
    return _summarize(float(est), reps, alpha, False)
