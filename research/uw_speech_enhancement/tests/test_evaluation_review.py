"""Adversarial-review regression tests for uwse.evaluation (RESEARCH_PLAN §8.3, §8.5)."""

from __future__ import annotations

import importlib.util
import json
import os
import warnings

import numpy as np
import pytest

from uwse.evaluation import asr, metrics, stats


def test_wer_counts_consistent_with_irregular_whitespace():
    # hits + S + D must equal the number of reference words even without normalization
    c = asr.corpus_wer(["a\tb  c"], ["a b c"], normalize=False)
    assert c.errors == 0 and c.hits == 3 and c.n_ref == 3
    err, ins = asr.word_alignment("a\tb c", "a b c")
    assert not err.any() and ins.sum() == 0


def test_cluster_design_accepts_tuple_labels():
    d = stats.cluster_design(["s1", "s2", "s1"], [("p1", "off"), ("p1", "off"), ("p3", "on")])
    assert d.n_strata == 2 and d.n_clusters == 3


def test_mapsswe_guard_three_requires_three_clean_words():
    # words 1..3 correct for both, but an insertion sits between words 1 and 2:
    # there is no run of 3 correct words free of insertions -> one segment only
    ea = np.array([1, 0, 0, 0, 1], bool)
    eb = np.zeros(5, bool)
    ins_a = np.zeros(6, int)
    ins_a[2] = 1
    sa, sb = stats.mapsswe_segments(ea, ins_a, eb, np.zeros(6, int), guard=3)
    assert sa.tolist() == [3] and sb.tolist() == [0]
    # guard=2 may cut between words 2 and 3
    sa, sb = stats.mapsswe_segments(ea, ins_a, eb, np.zeros(6, int), guard=2)
    assert sa.tolist() == [2, 1]
    # an insertion-free run of exactly 3 correct words does cut with guard=3
    sa, _ = stats.mapsswe_segments(ea, np.zeros(6, int), eb, np.zeros(6, int), guard=3)
    assert sa.tolist() == [1, 1]


def test_single_cluster_stratum_warns():
    e = np.array([1, 2, 3, 4.0])
    n = np.full(4, 10.0)
    with pytest.warns(UserWarning, match="single cluster"):
        b = stats.cluster_bootstrap_wer(e, n, ["a", "a", "b", "b"], ["x", "x", "y", "y"], n_boot=50)
    assert b.ci_low == b.ci_high  # zero-width CI is exactly what the warning is about


def test_pesq_unexpected_error_warns(rng):
    pytest.importorskip("pesq")
    with pytest.warns(UserWarning, match="PESQ"):
        v = metrics.pesq_wb(rng.standard_normal(100), rng.standard_normal(100))
    assert np.isnan(v)


def _load_score():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("score_cli_rev", os.path.join(root, "scripts", "score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_score_manifest_requires_cluster_keys(tmp_path):
    score = _load_score()
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"utt_id": "u", "system": "A", "wav": "x.wav", "ref_wav": "r.wav",
                             "transcript": "a", "position": "p1", "thruster": "off"}), encoding="utf-8")
    with pytest.raises(ValueError, match="speaker"):
        score.read_manifest(str(p))


def test_score_unknown_baseline_raises():
    score = _load_score()
    rows = [{"system": "A", "utt_id": "u", "kind": "speech", "speaker": "s", "position": "p", "thruster": "off"}]
    with pytest.raises(ValueError, match="baseline"):
        score.paired(rows, "C0", 10, 0, False)
