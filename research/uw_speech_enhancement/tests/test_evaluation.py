"""Tests for uwse.evaluation (RESEARCH_PLAN §6.5, §8.1-8.3, §8.5)."""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest
from scipy.signal import butter, lfilter, sosfilt

from uwse.evaluation import align, asr, metrics, normalize, stats

FS16 = 16000


def speechlike(rng, seconds, fs=FS16, band=(300.0, 4000.0), rate_hz=4.0):
    """Band-limited noise with syllable-rate on/off modulation (no real speech needed)."""
    n = int(seconds * fs)
    sos = butter(6, band, btype="bandpass", fs=fs, output="sos")
    x = sosfilt(sos, rng.standard_normal(n))
    t = np.arange(n) / fs
    env = np.clip(np.sin(2 * np.pi * rate_hz * t + rng.uniform(0, 6)), 0, None) ** 2
    env *= np.repeat(rng.uniform(0.3, 1.0, n // fs * 4 + 4), fs // 4)[:n]
    y = x * env
    return 0.1 * y / np.sqrt(np.mean(y**2))


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------
def test_chirp_detection_exact_with_noise(rng):
    fs = 48000
    c = align.lfm_chirp(fs, amplitude=0.5)
    assert len(c) == int(0.25 * fs)
    x = 0.3 * rng.standard_normal(6 * fs)  # per-sample SNR about -9 dB
    onsets = [24007, 4 * fs + 3]
    for o in onsets:
        x[o : o + len(c)] += c
    dets = align.detect_chirps(x, fs, n_chirps=2)
    assert [d.index for d in dets] == onsets
    assert all(abs(d.frac_index - o) < 0.3 for d, o in zip(dets, onsets))
    assert all(d.psr_db > 10 for d in dets)


def test_drift_30ppm_recovered(rng):
    fs, d_true = 48000, 30.0
    t1, t2 = 0.5, 30.5  # nominal chirp onsets on the reference clock (s)
    n = int(31.5 * fs)
    t_ref = np.arange(n) / (fs * (1 + d_true * 1e-6))  # device sample m -> reference time
    x = align.lfm_chirp_at(t_ref - t1) * 0.5 + align.lfm_chirp_at(t_ref - t2) * 0.5
    x += 0.1 * rng.standard_normal(n)
    dets = align.detect_chirps(x, fs)
    est = align.estimate_drift_ppm(dets[0].frac_index, dets[1].frac_index, (t2 - t1) * fs)
    assert abs(est - d_true) < 1.0


def test_correct_drift_restores_reference_clock():
    fs, d = 16000, 200.0
    n = 3 * fs
    t_dev = np.arange(n) / (fs * (1 + d * 1e-6))
    f = lambda t: np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 2100 * t)  # noqa: E731
    y = align.correct_drift(f(t_dev), d)
    ref = f(np.arange(len(y)) / fs)
    core = slice(100, len(y) - 100)
    err = np.max(np.abs(y[core] - ref[core]))
    assert err < 1e-3


def test_fine_lag_exact_and_flag(rng):
    ref = speechlike(rng, 2.0)
    for lag in (37, -113, 0):
        y = 0.7 * align.shift_signal(ref, -lag) + 0.01 * rng.standard_normal(len(ref))
        r = align.fine_lag(y, ref, FS16)
        assert r.lag_int == lag and r.lag == lag
        assert not r.flagged and r.psr_db > 6
    # polarity inversion still found
    r = align.fine_lag(-align.shift_signal(ref, -5), ref, FS16)
    assert r.lag_int == 5 and r.peak < 0
    # unrelated signal -> low PSR flag
    r = align.fine_lag(rng.standard_normal(len(ref)), ref, FS16)
    assert r.flagged


def test_fine_lag_subsample(rng):
    sos = butter(8, 3000, fs=FS16, output="sos")
    ref = sosfilt(sos, rng.standard_normal(2 * FS16))
    y = align.shift_signal(ref, -12.3)  # y[n] = ref[n - 12.3]
    r = align.fine_lag(y, ref, FS16, subsample=True)
    assert r.lag_int == 12 and abs(r.lag - 12.3) < 0.15


def test_block_cut_roundtrip(rng):
    fs = 48000
    utts = [speechlike(rng, s, fs=fs) for s in (1.3, 2.0, 0.7)]
    block, layout = align.assemble_block(utts, fs)
    assert abs(len(block) / fs - layout.total_s) < 1e-3
    res = align.align_block(block, fs, layout, resample=False)
    assert abs(res["drift_ppm"]) < 1.0
    pad = int(0.25 * fs)
    for u, cut in zip(utts, res["cuts"]):
        assert len(cut) == len(u) + 2 * pad
        np.testing.assert_allclose(cut[pad : pad + len(u)], u, atol=1e-12)
        assert np.all(cut[:pad] == 0)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------
def test_normalization_hits_target_on_active_region(rng):
    fs = FS16
    sil = lambda s: np.zeros(int(s * fs))  # noqa: E731
    r0 = np.concatenate([sil(0.25), speechlike(rng, 1.0), sil(0.8), speechlike(rng, 1.2), sil(0.25)])
    sysout = 0.03 * r0 + 3e-4 * rng.standard_normal(len(r0))
    y, g, mask = normalize.normalize_cut(sysout, r0, fs)
    assert abs(normalize.active_rms_dbfs(y, mask) - normalize.TARGET_DBFS) < 1e-9
    # mask follows R0: leading padding and the middle of the long pause are inactive
    assert not mask[: int(0.15 * fs)].any()
    mid = int((0.25 + 1.0 + 0.6) * fs)
    assert not mask[mid]
    assert mask[int(0.75 * fs)]
    # the residual noise is not blown up to full scale
    assert np.max(np.abs(y[: int(0.2 * fs)])) < 0.1


def test_silence_not_amplified(rng):
    x = 1e-4 * rng.standard_normal(FS16)
    y, g, mask = normalize.normalize_cut(x, np.zeros(FS16), FS16)
    assert np.isnan(g) and not mask.any()
    np.testing.assert_array_equal(y, x)


def test_noise_only_gain_rule():
    rows = [
        {"system": "A", "position": "p1", "condition": "on", "kind": "speech", "gain": 1.0},
        {"system": "A", "position": "p1", "condition": "on", "kind": "speech", "gain": 3.0},
        {"system": "A", "position": "p1", "condition": "on", "kind": "speech", "gain": 2.0},
        {"system": "A", "position": "p1", "condition": "off", "kind": "speech", "gain": 10.0},
        {"system": "B", "position": "p1", "condition": "on", "kind": "speech", "gain": 7.0},
        {"system": "A", "position": "p1", "condition": "on", "kind": "noise"},
        {"system": "B", "position": "p1", "condition": "on", "kind": "noise"},
    ]
    out = normalize.assign_gains(rows)
    assert out[5]["gain"] == 2.0 and out[6]["gain"] == 7.0
    assert "gain" not in rows[5]
    with pytest.raises(ValueError):
        normalize.assign_gains([{"system": "C", "position": "p", "condition": "x", "kind": "noise"}])


# ---------------------------------------------------------------------------
# asr / WER
# ---------------------------------------------------------------------------
def test_wer_sdi_counts():
    c = asr.corpus_wer(["a b c d"], ["a x c d e"], normalize=False)
    assert (c.substitutions, c.deletions, c.insertions, c.n_ref) == (1, 0, 1, 4)
    c = asr.corpus_wer(["a b c d"], ["a c d"], normalize=False)
    assert (c.substitutions, c.deletions, c.insertions) == (0, 1, 0)
    c = asr.corpus_wer(["a b", ""], ["", "z"], normalize=False)  # empty hyp / empty ref handled
    assert (c.deletions, c.insertions, c.n_ref) == (2, 1, 2)


def test_corpus_wer_is_not_mean_wer():
    refs = ["a b c d e f g h i j", "a"]
    hyps = ["a b c d e f g h i j", "b"]
    per = asr.utterance_counts(refs, hyps, normalize=False)
    mean_wer = np.mean([p.wer for p in per])
    c = asr.corpus_wer(refs, hyps, normalize=False)
    assert c.wer == pytest.approx(1 / 11) and mean_wer == pytest.approx(0.5)
    # bootstrap ratio estimator gives the corpus value, not the mean
    b = stats.cluster_bootstrap_wer([p.errors for p in per], [p.n_ref for p in per], ["s1", "s2"], n_boot=50)
    assert b.estimate == pytest.approx(1 / 11)


def test_whisper_normalizer_both_sides():
    pytest.importorskip("whisper")
    c = asr.corpus_wer(["Mr. Smith has 2 cats."], ["mister smith has two cats"])
    assert c.errors == 0


def test_hallucination_flags_and_report():
    refs = ["hello world", "the quick brown fox jumps", "one two three"]
    hyps = ["hello world hello world hello", "the quick brown fox jumps", "one two three " + "la " * 40]
    f = asr.hallucination_flags(refs, hyps, normalize=False)
    assert f.tolist() == [True, False, True]
    assert asr.compression_ratio("la " * 40) > 2.4
    rep = asr.wer_report(refs, hyps, normalizer=lambda s: s.lower())
    assert rep["n_flagged"] == 2 and rep["unflagged"]["wer"] == 0.0 and rep["all"]["wer"] > 1


def test_word_alignment():
    err, ins = asr.word_alignment("a b c d", "a x c d e")
    assert err.tolist() == [False, True, False, False] and ins.tolist() == [0, 0, 0, 0, 1]


def test_transcript_cache_and_batch(tmp_path):
    class Fake:
        name = "fake"
        calls = 0

        def transcribe(self, audio):
            Fake.calls += 1
            return {"text": f"n{len(audio)}", "compression_ratio": 1.0}

    path = str(tmp_path / "t.jsonl")
    items = [{"system": "A", "utt_id": f"u{i}", "audio": np.zeros(160 * (i + 1)), "fs": FS16} for i in range(3)]
    out = asr.transcribe_items(Fake(), items, asr.TranscriptCache(path), progress=False)
    assert [o["text"] for o in out] == ["n160", "n320", "n480"] and Fake.calls == 3
    out2 = asr.transcribe_items(Fake(), items, asr.TranscriptCache(path), progress=False)
    assert Fake.calls == 3 and out2[1]["text"] == "n320"


def test_parakeet_missing_nemo_raises():
    if importlib.util.find_spec("nemo") is not None:
        pytest.skip("nemo installed")
    with pytest.raises(ImportError, match="NeMo"):
        asr.ParakeetASR().model


@pytest.mark.network
@pytest.mark.slow
def test_whisper_tiny_protocol(rng):
    w = asr.WhisperASR("tiny", device="cpu")
    out = w.transcribe(np.zeros(FS16 * 2, np.float32))
    assert "text" in out and "compression_ratio" in out
    info = w.info()
    assert len(info["model_sha256"]) == 64 and info["decode"]["beam_size"] == 5
    assert info["decode"]["temperature"] == 0.0 and info["decode"]["condition_on_previous_text"] is False


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_ci_sdr_invariant_to_short_fir(rng):
    ref = speechlike(rng, 2.0, band=(100, 7500))
    g = rng.standard_normal(64) * np.exp(-np.arange(64) / 10.0)
    est = lfilter(g, [1.0], ref)
    assert metrics.ci_sdr(est, ref) >= 30.0
    assert metrics.si_sdr(est, ref) < 10.0
    noisy = ref + 0.1 * np.sqrt(np.mean(ref**2)) * rng.standard_normal(len(ref))
    assert metrics.ci_sdr(noisy, ref) == pytest.approx(20.0, abs=1.5)
    assert metrics.si_sdr(noisy, ref) == pytest.approx(20.0, abs=0.5)


def test_stoi_banded_matches_pystoi(rng):
    from pystoi import stoi as pystoi_stoi

    ref = speechlike(rng, 3.0)
    est = ref + 0.05 * rng.standard_normal(len(ref))
    for ext in (False, True):
        a = metrics.stoi_banded(ref, est, FS16, ext, None)
        b = pystoi_stoi(ref, est, FS16, extended=ext)
        assert a == pytest.approx(b, abs=1e-10)


def test_estoi_band_limiting_applied(rng):
    ref = metrics.bandlimit(speechlike(rng, 3.0, band=(300, 7000)), FS16, 500, 8000)
    lowband = sosfilt(butter(6, [150, 400], btype="bandpass", fs=FS16, output="sos"), rng.standard_normal(len(ref)))
    est = ref + 3.0 * np.sqrt(np.mean(ref**2)) * lowband / np.sqrt(np.mean(lowband**2))
    full = metrics.estoi(est, ref, FS16, f_lo=None, f_hi=None, drop_bands=False)
    banded = metrics.estoi(est, ref, FS16, f_lo=500, f_hi=8000)
    # filtering alone does not help: (E)STOI renormalizes every band envelope
    filt_only = metrics.estoi(est, ref, FS16, f_lo=500, f_hi=8000, drop_bands=False)
    assert banded > 0.9 and full < banded - 0.3 and filt_only < banded - 0.3
    mask = metrics.third_octave_mask(500, 8000)
    assert mask.sum() == 9 and not mask[:6].any()


def test_pesq_no_utterance_is_nan():
    pytest.importorskip("pesq")
    assert np.isnan(metrics.pesq_wb(np.zeros(FS16), np.zeros(FS16)))


def test_compute_metrics_dict(rng):
    ref = speechlike(rng, 2.0)
    m = metrics.compute_metrics(ref + 0.01 * rng.standard_normal(len(ref)), ref, ref_r0=ref,
                                metrics=("estoi", "stoi", "si_sdr", "ci_sdr"))
    assert set(m) == {"estoi", "estoi_r0", "stoi", "si_sdr", "ci_sdr"}
    assert m["estoi"] > 0.9


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def _sim_wer_data(rng, n_strata=3, n_spk=12, p=0.2):
    errs, nref, spk, strat = [], [], [], []
    for s in range(n_strata):
        for k in range(n_spk):
            pk = rng.beta(4 * p / (1 - p), 4)  # mean p, speaker heterogeneity
            for _ in range(5):
                n = rng.integers(8, 30)
                errs.append(rng.binomial(n, pk))
                nref.append(n)
                spk.append(f"s{k}")
                strat.append(f"st{s}")
    return np.array(errs), np.array(nref), spk, strat


def test_bootstrap_coverage_sanity():
    rng = np.random.default_rng(1)
    hits, n_sim = 0, 40
    for i in range(n_sim):
        e, n, spk, st = _sim_wer_data(rng)
        b = stats.cluster_bootstrap_wer(e, n, spk, st, n_boot=300, seed=i)
        hits += b.ci_low <= 0.2 <= b.ci_high
    assert hits / n_sim >= 0.8


def test_bootstrap_resamples_within_strata():
    d = stats.cluster_design(["a", "b", "a", "c"], ["x", "x", "y", "y"])
    assert d.n_clusters == 4 and d.n_strata == 2
    W = stats.bootstrap_weights(d, 200, np.random.default_rng(0))
    for s in range(2):
        assert np.all(W[:, d.stratum_of_cluster == s].sum(1) == 2)


def test_paired_identical_systems_zero_ci(rng):
    e, n, spk, st = _sim_wer_data(rng)
    b = stats.paired_bootstrap_wer(e, e, n, spk, st, n_boot=500)
    assert (b.estimate, b.ci_low, b.ci_high) == (0.0, 0.0, 0.0)
    v = rng.random(len(e))
    m = stats.paired_bootstrap_mean(v, v, spk, st, n_boot=200)
    assert (m.ci_low, m.ci_high) == (0.0, 0.0)


def test_tost_basic():
    assert stats.tost_normal(0.0, 0.01, -0.05, 0.05)["equivalent"]
    r = stats.tost_normal(0.04, 0.01, -0.05, 0.05)
    assert not r["equivalent"] and r["ci_high"] > 0.05
    reps = np.random.default_rng(0).normal(0.0, 0.01, 5000)
    assert stats.tost(reps, 0.0, -0.05, 0.05)["equivalent"]
    assert not stats.tost(reps + 0.045, 0.045, -0.05, 0.05)["equivalent"]
    ni = stats.noninferiority(reps + 0.01, 0.01, margin=0.05, lower_is_better=True)
    assert ni["noninferior"]


def test_holm_known_example():
    rej, adj = stats.holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert rej.tolist() == [True, False, False, True]


def test_mapsswe():
    ea = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0], bool)
    eb = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1], bool)
    ins0 = np.zeros(10, int)
    sa, sb = stats.mapsswe_segments(ea, ins0, eb, ins0)
    assert sa.tolist() == [1, 1] and sb.tolist() == [0, 2]
    assert stats.mapsswe([1, 2, 0], [1, 2, 0])["w"] == 0.0
    r = stats.mapsswe([3, 2, 4, 3, 5, 2, 3, 4] * 5, [1, 1, 2, 1, 2, 0, 1, 2] * 5)
    assert r["p"] < 1e-6 and r["w"] > 0


def test_kendall_tau_bootstrap():
    x = np.arange(30.0)
    b = stats.kendall_tau_bootstrap(x, 2 * x + 1, [i % 6 for i in range(30)], n_boot=100)
    assert b.estimate == pytest.approx(1.0) and b.ci_low == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# scripts/score.py end-to-end (fake ASR, small bootstrap)
# ---------------------------------------------------------------------------
def test_score_cli_end_to_end(tmp_path, rng, monkeypatch):
    from uwse.audio import save

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("score_cli", os.path.join(root, "scripts", "score.py"))
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)

    class FakeASR:
        def __init__(self, model_name, device=None):
            self.name = "fake"

        def transcribe(self, audio):
            return {"text": "a b c", "compression_ratio": 1.0}

        def info(self):
            return {"backend": "fake"}

    monkeypatch.setattr(score.A, "WhisperASR", FakeASR)
    rows = []
    for i in range(6):
        ref = speechlike(rng, 1.0)
        rp = str(tmp_path / f"r{i}.wav")
        save(rp, ref, FS16, subtype="FLOAT")
        for sysname, noise in (("C0", 0.05), ("C1", 0.01)):
            wp = str(tmp_path / f"{sysname}_{i}.wav")
            save(wp, ref + noise * rng.standard_normal(len(ref)), FS16, subtype="FLOAT")
            rows.append({"utt_id": f"u{i}", "system": sysname, "wav": wp, "ref_wav": rp,
                         "transcript": "a b c d" if i % 2 else "a b c", "speaker": f"s{i % 3}",
                         "position": "p1", "thruster": "off" if i < 3 else "on"})
    man = tmp_path / "m.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "out"
    assert score.main(["--manifest", str(man), "--out-dir", str(out), "--n-boot", "50",
                       "--baseline", "C0", "--level-normalize"]) == 0
    for f in ("per_utt.jsonl", "summary.csv", "summary.md", "paired.csv", "run_info.json", "transcripts.jsonl"):
        assert (out / f).exists()
    per = [json.loads(line) for line in (out / "per_utt.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(np.isfinite(r["gain"]) for r in per)
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "C1" in md and "WER" in md
