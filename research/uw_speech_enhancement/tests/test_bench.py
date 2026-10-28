"""Tests for uwse.bench (RESEARCH_PLAN.md 5.1, 5.4, 6.5, 6.6, 6.8)."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from uwse.bench import librispeech as ls
from uwse.bench import playlist as pl
from uwse.bench import schema
from uwse.bench import splits

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ schema
def test_schema_example_valid_and_roundtrip(tmp_path):
    row = schema.example_row()
    schema.validate(row)
    r2 = copy.deepcopy(row)
    r2["utt_id"] = "v0_s2_p3_b07_ch1_1089-134686-0000"
    r2["channel"] = 1
    r2["rx_xyz_m"] = {"ch0": [0, 0, 0], "ch1": [0.1, 0, 1.0]}
    r2["acoustic"]["drr_dB"] = 3.5
    p = tmp_path / "metadata.jsonl"
    assert schema.write_jsonl(p, [row, schema.from_dict(r2)]) == 2
    back = schema.read_jsonl(p)
    assert isinstance(back[0], schema.MetadataRow)
    assert isinstance(back[0].thruster, schema.Thruster)
    assert back[0].acoustic.drr_dB is None and back[1].acoustic.drr_dB == 3.5
    assert back[0].to_dict() == row
    assert list(back[1].to_dict()) == list(row)  # key order as in 6.8


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r.update(material="podcast"), "material"),
    (lambda r: r.update(corpus_split="train-clean-100"), "corpus_split"),
    (lambda r: r.update(bench_split="v2-test"), "bench_split"),
    (lambda r: r["thruster"].update(state="idle"), "thruster.state"),
    (lambda r: r["thruster"].update(unit="C"), "thruster.unit"),
    (lambda r: r.update(pumps="maybe"), "pumps"),
    (lambda r: r.update(SL_abs_calibrated="yes"), "SL_abs_calibrated"),
    (lambda r: r.update(flags=["clip", "bad"]), "flags[1]"),
    (lambda r: r.update(utt_id="v0-s2-p3-b07-ch0-1089-134686-0000"), "utt_id"),
    (lambda r: r.update(utt_id="v0_s2_p3_b08_ch0_1089-134686-0000"), "block"),
    (lambda r: r.update(utt_id="v1_s2_p3_b07_ch0_1089-134686-0000"), "release"),
    (lambda r: r.update(speaker_id="1090"), "speaker_id"),
    (lambda r: r["acoustic"].update(c50_dB=None), "acoustic.c50_dB"),
    (lambda r: r.update(block_id="7"), "block_id"),
    (lambda r: r.update(scored_channel=1), "scored_channel"),
    (lambda r: r.update(bench_split="v0-dev"), "corpus_split"),
    (lambda r: r.pop("notes"), "notes: missing"),
    (lambda r: r.update(extra=1), "extra: unknown"),
    (lambda r: r["pool"].pop("L_m"), "pool.L_m"),
])
def test_schema_rejects(mutate, needle):
    row = schema.example_row()
    mutate(row)
    with pytest.raises(schema.SchemaError) as e:
        schema.validate(row)
    assert any(needle in m for m in e.value.errors), e.value.errors


def test_schema_newprompt_nulls_and_utt_id_helpers():
    row = schema.example_row()
    row.update(material="newprompt_tts", librispeech_id=None, corpus_split=None,
               bench_split="v1-test", utt_id="v1_s4_p10_b03_ch0_np0042",
               session_id="2027-01-20_s4")
    row["block_id"] = 3
    row["position_id"] = "p10"
    schema.validate(row)
    uid = schema.make_utt_id("v0", 2, "p3", 7, 0, "1089-134686-0000")
    assert uid == "v0_s2_p3_b07_ch0_1089-134686-0000"
    assert schema.parse_utt_id(uid)["block_id"] == 7


# ------------------------------------------------------------------ librispeech
def _fake_tree(root: Path, subset: str, speakers: list[str], n_per: int, fs: int = 16000):
    """Tiny LibriSpeech tree: short tone-burst FLACs + transcripts."""
    rng = np.random.default_rng(1)
    for s in speakers:
        ch = f"{int(s) * 10 + 1}"
        d = root / "LibriSpeech" / subset / s / ch
        d.mkdir(parents=True)
        lines = []
        for k in range(n_per):
            uid = f"{s}-{ch}-{k:04d}"
            dur = 2.0 + 0.25 * (k % 5)  # 2.0 .. 3.0 s, all pass the 2-15 s filter
            x = 0.1 * rng.standard_normal(int(dur * fs)).astype(np.float32)
            sf.write(d / f"{uid}.flac", x, fs)
            lines.append(f"{uid} WORDS OF {s} NUMBER {k}")
        (d / f"{s}-{ch}.trans.txt").write_text("\n".join(lines) + "\n")


def test_manifest_and_v0_selection(tmp_path):
    test_spk = [str(1000 + i) for i in range(42)]
    dev_spk = [str(2000 + i) for i in range(40)]
    _fake_tree(tmp_path, "test-clean", test_spk, 6)
    _fake_tree(tmp_path, "dev-clean", dev_spk, 2)
    test = ls.build_manifest(ls.find_subset_dir(tmp_path, "test-clean"), progress=False)
    dev = ls.build_manifest(ls.find_subset_dir(tmp_path / "LibriSpeech", "dev-clean"), progress=False)
    assert len(test) == 42 * 6 and len(dev) == 80
    u = test[0]
    assert u.subset == "test-clean" and u.speaker == "1000" and u.transcript.startswith("WORDS")
    assert abs(u.duration_s - 2.0) < 1e-6

    v0t = ls.select_v0_test(test, seed=3)
    assert len(v0t) == 200
    per = {}
    for x in v0t:
        per[x.speaker] = per.get(x.speaker, 0) + 1
    assert len(per) == 40 and set(per.values()) == {5}
    assert [x.utt_id for x in ls.select_v0_test(test, seed=3)] == [x.utt_id for x in v0t]
    assert [x.utt_id for x in ls.select_v0_test(test, seed=4)] != [x.utt_id for x in v0t]

    v0d = ls.select_v0_dev(dev, seed=3)
    assert len(v0d) == 40 and len({x.speaker for x in v0d}) == 40

    with pytest.raises(ValueError):
        ls.select_v0_test(dev)  # wrong subset
    p = tmp_path / "m.jsonl"
    ls.write_manifest(p, v0t)
    assert ls.read_manifest(p) == v0t


def test_v1_selection_disjoint_from_v0():
    test = [ls.Utterance(f"{s}-1-{k:04d}", str(s), "1", "x", f"/x/{s}-{k}.flac", 5.0, "test-clean")
            for s in range(100, 140) for k in range(40)]
    v0t = ls.select_v0_test(test)
    v1 = ls.select_v1_test(test, v0t, n=1000)
    assert len(v1) == 1000 and len({u.utt_id for u in v1}) == 1000
    assert not {u.utt_id for u in v1} & {u.utt_id for u in v0t}
    counts = np.bincount([int(u.speaker) - 100 for u in v1])
    assert counts.max() - counts.min() <= 1  # balanced round robin
    with pytest.raises(ValueError):
        ls.select_v1_test(test, v0t, n=999)
    splits.assert_disjoint({"v0-test": v0t, "v1-test": v1})


# ------------------------------------------------------------------ splits
def _u(uid, subset):
    return ls.Utterance(uid, uid.split("-")[0], "1", "", "", 5.0, subset)


def test_disjointness_checker():
    train = [_u("10-1-0000", "train-clean-100"), _u("11-1-0000", "train-clean-100")]
    dev = [_u("20-1-0000", "dev-clean")]
    test = [_u("30-1-0000", "test-clean"), _u("30-1-0001", "test-clean")]
    splits.assert_disjoint({"sim-train": train, "sim-val": dev, "v0-dev": dev,
                            "real-train": dev, "v0-test": test})
    # speaker leak across groups (train-clean speaker 30 does not exist, but the id collides)
    with pytest.raises(splits.SplitLeakError, match="speaker leak"):
        splits.assert_disjoint({"sim-train": train + [_u("30-9-0000", "train-clean-100")],
                                "v0-test": test})
    # utterance overlap between v0-test and v1-test
    with pytest.raises(splits.SplitLeakError, match="utterance overlap"):
        splits.assert_disjoint({"v0-test": test, "v1-test": test[:1]})
    # sim twin may mirror v0-test
    splits.assert_disjoint({"v0-test": test, "sim-twin-v0": test})
    # wrong corpus subset for a role
    with pytest.raises(splits.SplitLeakError, match="not allowed"):
        splits.assert_disjoint({"v0-dev": [_u("40-1-0000", "test-clean")]})
    # metadata-style dicts work too
    rows = [{"librispeech_id": "30-1-0000", "speaker_id": "30", "corpus_split": "test-clean"}]
    with pytest.raises(splits.SplitLeakError):
        splits.assert_disjoint({"v0-dev": [{"utt_id": "30-1-0005", "speaker": "30",
                                            "subset": "dev-clean"}], "v0-test": rows})


def test_noise_and_positions():
    splits.assert_noise_split(["A"], ["B"])
    with pytest.raises(splits.SplitLeakError):
        splits.assert_noise_split(["A", "B"], ["B"])
    with pytest.raises(splits.SplitLeakError):
        splits.check_noise_unit("test", "A")
    splits.assert_positions_disjoint(["p2", "p4.5"], ["p1", "p3"])
    with pytest.raises(splits.SplitLeakError):
        splits.assert_positions_disjoint(["p3"], ["p1", "p3"])


# ------------------------------------------------------------------ playlist
def test_block_timing_exact():
    fs = 48000
    lens = [48000 * 3, 100001, 7]
    t = pl.block_timing(lens, fs)
    c, u = t["chirps"], t["utterances"]
    assert c[0] == {"start": 24000, "end": 36000}
    assert u[0]["start"] == 36000 + 24000
    assert u[1]["start"] == u[0]["end"] + 48000
    assert u[2]["start"] == u[1]["end"] + 48000 and u[2]["end"] - u[2]["start"] == 7
    assert c[1]["start"] == u[2]["end"] + 24000
    assert t["n_samples"] == c[1]["end"] + 96000
    assert t["n_samples"] == sum(lens) + round(pl.BlockLayout().overhead_s(3) * fs)


def test_render_block(rng):
    items = [(f"u{k}", 0.05 * rng.standard_normal(16000 * (2 + k)), 16000) for k in range(3)]
    sig, t = pl.render_block(items)
    assert len(sig) == t["n_samples"]
    for k, u in enumerate(t["utterances"]):
        assert u["end"] - u["start"] == 3 * 16000 * (2 + k)  # 16 -> 48 kHz exact
        assert u["item_id"] == f"u{k}"
        assert abs(u["rms_dBFS"] + 26.0) < 1.0  # LPF/fade barely change the level
        assert abs(u["gain_dB"]) < 60
        seg = sig[u["start"]:u["end"]]
        assert np.allclose(seg[:5], 0.0, atol=1e-3)
    ch = pl.lfm_chirp()
    for c in t["chirps"]:
        assert np.array_equal(sig[c["start"]:c["end"]], ch)
    # silence between items is exactly zero
    assert not np.any(sig[: t["chirps"][0]["start"]])
    assert not np.any(sig[t["utterances"][0]["end"]:t["utterances"][1]["start"]])
    # chirp energy sits in 500 Hz .. 8 kHz
    spec = np.abs(np.fft.rfft(ch, 1 << 15)) ** 2
    f = np.fft.rfftfreq(1 << 15, 1 / 48000)
    assert spec[(f >= 500) & (f <= 8000)].sum() / spec.sum() > 0.97


def test_render_blocks_files(tmp_path, rng):
    for k in range(2):
        sf.write(tmp_path / f"a{k}.flac", (0.05 * rng.standard_normal(16000)).astype(np.float32), 16000)
    utts = [ls.Utterance(f"1-1-{k:04d}", "1", "1", "T", str(tmp_path / f"a{k}.flac"), 1.0, "dev-clean")
            for k in range(2)]
    stems = pl.render_blocks([utts], tmp_path / "out", prefix="v0-dev", bench_split="v0-dev")
    info = sf.info(str(tmp_path / "out" / f"{stems[0]}.wav"))
    t = json.loads((tmp_path / "out" / f"{stems[0]}.json").read_text())
    assert info.samplerate == 48000 and info.frames == t["n_samples"]
    assert t["utterances"][1]["item_id"] == "1-1-0001" and t["chirp"]["f0_hz"] == 500.0


def test_abba_and_anchor():
    assert pl.abba_states(8) == ["off", "on", "on", "off", "off", "on", "on", "off"]
    sch = pl.session_schedule(["p1", "p3"], ["b0", "b1", "b2"])
    assert len(sch) == 2 * 3 * 2 + 1
    for pos in ("p1", "p3"):
        e = [x for x in sch if x.position_id == pos and not x.anchor_repeat]
        assert [x.thruster_state for x in e] == pl.abba_states(6)
        for cb in ("b0", "b1", "b2"):
            assert sorted(x.thruster_state for x in e if x.content_block == cb) == ["off", "on"]
    last = sch[-1]
    assert last.anchor_repeat and last.content_block == sch[0].content_block
    assert (last.position_id, last.thruster_state) == (sch[0].position_id, sch[0].thruster_state)
    assert [x.session_order_idx for x in sch] == list(range(len(sch)))


def test_pack_blocks():
    items = [ls.Utterance(str(k), "1", "1", "", "", 7.4, "test-clean") for k in range(200)]
    blocks = pl.pack_blocks(items, seed=0)
    assert sum(len(b) for b in blocks) == 200
    assert {len(b) for b in blocks[:-1]} == {8}
    assert len({u.utt_id for b in blocks for u in b}) == 200


def test_overhead_and_budget():
    frac = pl.overhead_fraction(8, 7.4)
    assert abs(frac - 11.0 / 70.2) < 1e-12
    assert round(100 * frac) == 16  # plan 6.5: "about 16 %"
    b = pl.playback_budget()
    assert round(b.test_speech_h, 2) == 2.47  # plan 6.6
    assert round(b.total_plan_h, 2) == 3.43   # plan 6.6, speech x 1.16
    # the actual layout implies 18.6 % of speech time, i.e. ~3.51 h, not 3.43 h
    assert abs(b.overhead_ratio - 11.0 / 59.2) < 1e-12
    assert round(b.total_layout_h, 2) == 3.51 and b.notes


# ------------------------------------------------------------------ scripts
def test_fetch_args():
    fl = _load_script("fetch_librispeech")
    a = fl.parse_args(["dev-clean", "test-clean", "dev-clean", "--dry-run"])
    assert a.subsets == ["dev-clean", "test-clean"] and a.dry_run
    assert fl.url_for("dev-clean") == "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    with pytest.raises(SystemExit):
        fl.parse_args(["train-other-500"])
    assert fl.main(["dev-clean", "--dry-run"]) == 0


def test_make_playlists_args_and_interleave():
    mp = _load_script("make_playlists")
    a = mp.parse_args(["--positions", "p1", "p3", "--no-render"])
    assert a.positions == ["p1", "p3"] and a.out.name == "v0"
    with pytest.raises(SystemExit):
        mp.parse_args(["--release", "v1", "--v1-n", "10"])
    out = mp.interleave([f"t{i}" for i in range(25)], [f"d{i}" for i in range(5)])
    assert len(out) == 30 and sorted(out) == sorted([f"t{i}" for i in range(25)] + [f"d{i}" for i in range(5)])
    assert out[0].startswith("t") and out[-1].startswith("t")
