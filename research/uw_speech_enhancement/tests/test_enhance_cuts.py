"""scripts/enhance_cuts.py: manifest logic, resume, lengths and rates with fake adapters (no GPU)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from uwse import audio

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolves annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


E = _script("enhance_cuts")
FAKE = E.SystemSpec("C1", "identity")
FAKE48 = E.SystemSpec("C1", "identity", adapter_fs=48000)


class FakeSE:
    """Identity at ``sample_rate``, recording what it is fed."""

    name = "fake"

    def __init__(self, sample_rate=16000, gain=1.0, fail_after=None):
        self.sample_rate, self.gain, self.fail_after = sample_rate, gain, fail_after
        self.fed = []

    def enhance(self, wav):
        if self.fail_after is not None and len(self.fed) >= self.fail_after:
            raise RuntimeError("boom")
        self.fed.append(wav.detach().cpu().numpy().copy())
        return wav * self.gain


class Loader:
    def __init__(self, *a, **kw):
        self.se = FakeSE(*a, **kw)
        self.n_loads = 0

    def __call__(self, spec, device):
        self.n_loads += 1
        return self.se


def _rows(tmp_path, n=3, fs=16000, dur=0.5, seed=0):
    rng = np.random.default_rng(seed)
    (tmp_path / "cuts").mkdir(exist_ok=True)
    rows = []
    for k in range(n):
        p = tmp_path / "cuts" / f"utt{k}.wav"
        audio.save(str(p), 0.1 * rng.standard_normal(int(dur * fs) + k), fs, subtype="FLOAT")
        rows.append({"utt_id": f"utt{k}", "system": "C0", "wav": str(p), "r0_wav": str(p), "ref_wav": str(p),
                     "transcript": "a b", "speaker": str(k % 2), "position": "p3", "thruster": "off",
                     "kind": "speech", "flags": ["x"]})
    return rows


def _write_manifest(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Lengths and rates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fs", [16000, 44100, 48000, 96000])
@pytest.mark.parametrize("frames", [1000, 1001, 1002, 48001, 96005])
def test_n_out_is_the_length_score_py_loads(fs, frames):
    assert E.n_out_16k(frames, fs) == audio.resample(np.zeros(frames), fs, 16000).size


def test_16k_cut_is_preserved_sample_for_sample(tmp_path):
    rows = _rows(tmp_path)
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=Loader())
    for r in rows:
        y, fs = sf.read(E.out_path(out, "C1", r["utt_id"]), dtype="float64")
        x, _ = audio.load(r["wav"])
        assert fs == 16000 and y.size == x.size
        np.testing.assert_allclose(y, x, atol=1e-7)


def test_48k_cut_fed_natively_to_48k_adapter_and_output_matches_c0(tmp_path):
    rows = _rows(tmp_path, n=2, fs=48000, dur=0.4)
    load = Loader(sample_rate=48000)
    out = tmp_path / "out"
    E.run(rows, [FAKE48], out, "cpu", loader=load)
    score = _script("score")
    for r, fed in zip(rows, load.se.fed):
        x, _ = audio.load(r["wav"])
        np.testing.assert_allclose(fed[0], x, atol=1e-7)  # no resampling on the way in
        c0 = score._load16(r["wav"])
        y = score._load16(str(E.out_path(out, "C1", r["utt_id"])))
        assert y.size == c0.size
        np.testing.assert_allclose(y, c0, atol=1e-6)


def test_48k_cut_to_16k_adapter_is_the_audio_resample_of_the_simulation(tmp_path):
    rows = _rows(tmp_path, n=1, fs=48000, dur=0.4)
    load = Loader(sample_rate=16000)
    E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=load)
    x, _ = audio.load(rows[0]["wav"])
    np.testing.assert_allclose(load.se.fed[0][0], audio.resample(x, 48000, 16000), atol=1e-6)


def test_16k_cut_for_48k_adapter_refused_by_default(tmp_path):
    rows = _rows(tmp_path, n=1, fs=16000)
    load = Loader(sample_rate=48000)
    with pytest.raises(SystemExit, match="allow-upsampled-input"):
        E.run(rows, [FAKE48], tmp_path / "out", "cpu", loader=load)
    assert load.n_loads == 0


def test_frozen_c2_refuses_16k_cuts_before_any_system_runs(tmp_path):
    rows = _rows(tmp_path, n=1, fs=16000)
    load = Loader()
    c2 = E.SystemSpec("C2", "identity", adapter_fs=E.FROZEN["C2"].adapter_fs)
    with pytest.raises(SystemExit, match="allow-upsampled-input"):
        E.run(rows, [FAKE, c2], tmp_path / "out", "cpu", loader=load)
    assert load.n_loads == 0


def test_allowed_upsampling_is_labelled_and_recorded(tmp_path):
    rows = _rows(tmp_path, n=1, fs=16000)
    load = Loader(sample_rate=48000)
    out = tmp_path / "out"
    path, done = E.run(rows, [FAKE48], out, "cpu", loader=load, allow_upsampled=True)
    assert load.se.fed[0].shape[-1] == 3 * audio.load(rows[0]["wav"])[0].size
    assert done == ["C1_16kin"] and {r["system"] for r in _read_jsonl(path)} == {"C0", "C1_16kin"}
    assert (out / "C1_16kin" / "utt0.wav").is_file() and not (out / "C1").exists()
    info = json.loads((out / "run_info.json").read_text())
    assert info["input_level"]["fs_in"] == [16000]
    assert info["systems"]["C1_16kin"]["adapter_sample_rate"] == 48000
    assert info["systems"]["C1_16kin"]["upsampled_input"] is True


def test_48k_cut_for_48k_adapter_keeps_the_frozen_label(tmp_path):
    rows = _rows(tmp_path, n=1, fs=48000)
    _, done = E.run(rows, [FAKE48], tmp_path / "out", "cpu", loader=Loader(sample_rate=48000), allow_upsampled=True)
    assert done == ["C1"]


def test_declared_adapter_rate_is_checked_on_the_loaded_model(tmp_path):
    rows = _rows(tmp_path, n=1, fs=48000)
    with pytest.raises(SystemExit, match="declared"):
        E.run(rows, [FAKE48], tmp_path / "out", "cpu", loader=Loader(sample_rate=16000))


def test_cpu_rule_matches_sim_baseline():
    assert E.resolve_device("wpe1", "cuda") == "cpu"
    assert E.resolve_device("dfn3", "cuda") == "cuda"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def test_manifest_rows_copy_c0_and_change_only_system_and_wav(tmp_path):
    rows = _rows(tmp_path)
    out = tmp_path / "out"
    specs = [E.SystemSpec("C1", "identity"), E.SystemSpec("C2", "identity"), E.SystemSpec("C10", "identity")]
    path, done = E.run(rows, specs, out, "cpu", loader=Loader())
    assert done == ["C1", "C2", "C10"]
    got = _read_jsonl(path)
    assert len(got) == 4 * len(rows)
    by = {(r["system"], r["utt_id"]): r for r in got}
    assert len(by) == len(got)
    for r in rows:
        assert by[("C0", r["utt_id"])] == r
        for s in done:
            e = by[(s, r["utt_id"])]
            assert Path(e["wav"]).is_file() and Path(e["wav"]).parent.name == s
            assert {k: v for k, v in e.items() if k not in ("system", "wav")} == \
                   {k: v for k, v in r.items() if k not in ("system", "wav")}


def test_non_c0_rows_are_dropped_and_counted(tmp_path):
    rows = _rows(tmp_path, n=2)
    m = tmp_path / "m.jsonl"
    _write_manifest(m, rows + [dict(rows[0], system="C8")])
    c0, dropped = E.read_c0_rows(str(m))
    assert c0 == rows and dropped == {"C8": 1}


def test_manifest_without_c0_refused(tmp_path):
    rows = _rows(tmp_path, n=1)
    m = tmp_path / "m.jsonl"
    _write_manifest(m, [dict(rows[0], system="C8")])
    with pytest.raises(SystemExit, match="no C0 rows"):
        E.read_c0_rows(str(m))


def test_score_py_runs_on_the_output_with_paired_differences(tmp_path):
    rows = _rows(tmp_path, n=4, dur=1.0)
    path, _ = E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader(gain=0.5))
    d = tmp_path / "score"
    assert _script("score").main(["--manifest", str(path), "--out-dir", str(d), "--no-asr", "--no-metrics",
                                  "--level-normalize", "--baseline", "C0", "--n-boot", "10"]) == 0
    assert (d / "paired.csv").exists() and (d / "summary.csv").exists()
    per = _read_jsonl(d / "per_utt.jsonl")
    assert {r["system"] for r in per} == {"C0", "C1"} and len(per) == 8


def test_separate_calls_accumulate_in_one_manifest(tmp_path):
    rows = _rows(tmp_path, n=2)
    out = tmp_path / "out"
    E.run(rows, [E.SystemSpec("C1", "identity")], out, "cpu", loader=Loader())
    path, done = E.run(rows, [E.SystemSpec("C2", "identity")], out, "cpu", loader=Loader())
    assert done == ["C1", "C2"]
    assert {r["system"] for r in _read_jsonl(path)} == {"C0", "C1", "C2"}


# ---------------------------------------------------------------------------
# Resume and refusals
# ---------------------------------------------------------------------------
def test_resume_skips_finished_outputs_without_loading(tmp_path):
    rows = _rows(tmp_path)
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=Loader())
    p = E.out_path(out, "C1", "utt0")
    before = p.read_bytes()
    load = Loader(gain=0.5)
    E.run(rows, [FAKE], out, "cpu", loader=load)
    assert load.n_loads == 0 and p.read_bytes() == before


def test_resume_makes_only_the_missing_output(tmp_path):
    rows = _rows(tmp_path)
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=Loader())
    os.remove(E.out_path(out, "C1", "utt1"))
    load = Loader()
    E.run(rows, [FAKE], out, "cpu", loader=load)
    assert load.n_loads == 1 and len(load.se.fed) == 1
    assert json.loads((out / "run_info.json").read_text())["systems"]["C1"]["n_skipped"] == 2


def test_failed_cuts_are_named_and_the_job_resumes(tmp_path):
    rows = _rows(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=r"2 cuts failed[\s\S]*utt1[\s\S]*boom"):
        E.run(rows, [FAKE], out, "cpu", loader=Loader(fail_after=1))
    assert sorted(p.name for p in (out / "C1").iterdir()) == ["utt0.wav"]
    assert {r["system"] for r in _read_jsonl(out / "manifest.jsonl")} == {"C0"}  # the partial system stays out
    info = json.loads((out / "run_info.json").read_text())["systems"]["C1"]
    assert info["status"] == "failed_cuts" and [f["utt_id"] for f in info["failures"]] == ["utt1", "utt2"]
    load = Loader()
    _, done = E.run(rows, [FAKE], out, "cpu", loader=load)
    assert done == ["C1"] and len(load.se.fed) == 2
    info = json.loads((out / "run_info.json").read_text())["systems"]["C1"]
    assert info["status"] == "done" and info["failures"] == []


class _PerSystemLoader:
    """A system-specific loader: C2 fails on its second cut, the others work."""

    def __call__(self, spec, device):
        return FakeSE(fail_after=1 if spec.label == "C2" else None)


def test_one_failing_cut_does_not_stop_later_systems_or_the_manifest(tmp_path):
    rows = _rows(tmp_path)
    out = tmp_path / "out"
    specs = [E.SystemSpec(s, "identity") for s in ("C1", "C2", "C10")]
    with pytest.raises(SystemExit, match="C2: utt1"):
        E.run(rows, specs, out, "cpu", loader=_PerSystemLoader())
    assert {r["system"] for r in _read_jsonl(out / "manifest.jsonl")} == {"C0", "C1", "C10"}


def test_fatal_stop_still_writes_the_manifest_of_finished_systems(tmp_path):
    rows = _rows(tmp_path)
    out = tmp_path / "out"

    def loader(spec, device):
        if spec.label == "C2":
            raise SystemExit("cannot load C2")
        return FakeSE()

    with pytest.raises(SystemExit, match="cannot load C2"):
        E.run(rows, [E.SystemSpec("C1", "identity"), E.SystemSpec("C2", "identity")], out, "cpu", loader=loader)
    assert {r["system"] for r in _read_jsonl(out / "manifest.jsonl")} == {"C0", "C1"}


def test_partial_system_stays_out_of_the_manifest(tmp_path):
    rows = _rows(tmp_path)
    out = tmp_path / "out"
    E.run(rows[:2], [FAKE], out, "cpu", loader=Loader())  # superset below has no C1 for utt2 yet
    path, done = E.run(rows, [], out, "cpu")
    assert done == [] and {r["system"] for r in _read_jsonl(path)} == {"C0"}


def test_changed_cut_refused_on_resume(tmp_path):
    rows = _rows(tmp_path)
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=Loader())
    audio.save(rows[0]["wav"], np.zeros(8000), 16000, subtype="FLOAT")
    with pytest.raises(SystemExit, match="differ"):
        E.run(rows, [FAKE], out, "cpu", loader=Loader())


def test_changed_checkpoint_refused_on_resume(tmp_path):
    rows = _rows(tmp_path, n=1)
    out = tmp_path / "out"
    ck = tmp_path / "w.pt"
    ck.write_bytes(b"one")
    spec = E.SystemSpec("C2", "identity", str(ck))
    E.run(rows, [spec], out, "cpu", loader=Loader())
    info = json.loads((out / "run_info.json").read_text())
    assert info["systems"]["C2"]["ckpt_sha256"] == E.sha256(ck)
    ck.write_bytes(b"two")
    with pytest.raises(SystemExit, match="sha256"):
        E.run(rows, [spec], out, "cpu", loader=Loader())


def test_wavs_with_no_recorded_checkpoint_refused(tmp_path):
    rows = _rows(tmp_path, n=1)
    out = tmp_path / "out" / "C1"
    out.mkdir(parents=True)
    sf.write(out / "utt0.wav", np.zeros(8000), 16000)
    with pytest.raises(SystemExit, match="no recorded checkpoint"):
        E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader())


def test_existing_output_of_wrong_length_refused(tmp_path):
    rows = _rows(tmp_path, n=1)
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=Loader())
    sf.write(E.out_path(out, "C1", "utt0"), np.zeros(10), 16000, subtype="FLOAT")
    with pytest.raises(SystemExit, match="expected mono.*Delete this file"):
        E.run(rows, [FAKE], out, "cpu", loader=Loader())


@pytest.mark.parametrize("content", [b"", b"RIFF", b"garbage" * 100])
def test_unreadable_output_named_with_recovery_and_rerun_fixes_it(tmp_path, content):
    rows = _rows(tmp_path, n=2)
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=Loader())
    p = E.out_path(out, "C1", "utt0")
    p.write_bytes(content)
    with pytest.raises(SystemExit, match="unreadable.*Delete this file") as e:
        E.run(rows, [FAKE], out, "cpu", loader=Loader())
    assert "utt0.wav" in str(e.value)
    os.remove(p)
    load = Loader()
    _, done = E.run(rows, [FAKE], out, "cpu", loader=load)
    assert done == ["C1"] and len(load.se.fed) == 1


def test_damaged_output_of_another_system_stays_out_of_the_manifest(tmp_path, capsys):
    rows = _rows(tmp_path)
    out = tmp_path / "out"
    E.run(rows, [E.SystemSpec("C1", "identity")], out, "cpu", loader=Loader())
    sf.write(E.out_path(out, "C1", "utt1"), np.zeros(100), 16000, subtype="FLOAT")
    path, done = E.run(rows, [E.SystemSpec("C2", "identity")], out, "cpu", loader=Loader())
    assert done == ["C2"] and {r["system"] for r in _read_jsonl(path)} == {"C0", "C2"}
    assert "C1 left out of the manifest" in capsys.readouterr().out


def test_missing_checkpoint_refused_before_any_work(tmp_path):
    rows = _rows(tmp_path, n=1)
    load = Loader()
    spec = E.SystemSpec("C2", "dfn3", str(tmp_path / "nope.pt"))
    with pytest.raises(SystemExit, match="does not exist"):
        E.run(rows, [spec], tmp_path / "out", "cpu", loader=load)
    assert load.n_loads == 0 and not (tmp_path / "out").exists()


def test_missing_checkpoint_refused_by_the_cli(tmp_path):
    rows = _rows(tmp_path, n=1)
    m = tmp_path / "m.jsonl"
    _write_manifest(m, rows)
    with pytest.raises(SystemExit, match="does not exist"):
        E.main(["--manifest", str(m), "--out-dir", str(tmp_path / "out"), "--systems", "C10", "--input-level", "none",
                "--ckpt", f"C10={tmp_path / 'nope.pt'}", "--device", "cpu"])


def test_cli_requires_an_explicit_input_level(tmp_path, capsys):
    rows = _rows(tmp_path, n=1)
    m = tmp_path / "m.jsonl"
    _write_manifest(m, rows)
    with pytest.raises(SystemExit) as e:
        E.main(["--manifest", str(m), "--out-dir", str(tmp_path / "out"), "--systems", "C1", "--device", "cpu"])
    assert e.value.code == 2 and "--input-level" in capsys.readouterr().err


class _Net:
    name = "fake"

    def __init__(self, sd):
        self.model = torch.nn.Linear(1, 1)
        self.model.load_state_dict(sd)


def test_verify_checkpoint_catches_weights_that_did_not_load(tmp_path):
    good = torch.nn.Linear(1, 1).state_dict()
    p = tmp_path / "ck.pt"
    torch.save({"model": good, "adapter": "x"}, p)
    E.verify_checkpoint_loaded(_Net(good), str(p))  # exact match passes
    other = {k: v + 1 for k, v in good.items()}
    with pytest.raises(SystemExit, match="differs"):
        E.verify_checkpoint_loaded(_Net(other), str(p))
    torch.save({"model": {"weight": good["weight"]}}, p)  # strict=False would have left `bias` alone
    with pytest.raises(SystemExit, match="differs"):
        E.verify_checkpoint_loaded(_Net(good), str(p))


def test_non_finite_output_refused(tmp_path):
    rows = _rows(tmp_path, n=1)
    with pytest.raises(SystemExit, match="utt0: ValueError: non-finite"):
        E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader(gain=float("nan")))
    assert not E.out_path(tmp_path / "out", "C1", "utt0").exists()


def test_empty_cut_refused_by_name(tmp_path):
    rows = _rows(tmp_path, n=1)
    sf.write(rows[0]["wav"], np.zeros(0), 16000, subtype="FLOAT")
    with pytest.raises(SystemExit, match="utt0: empty cut"):
        E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader())


def test_multichannel_and_missing_cuts_refused(tmp_path):
    rows = _rows(tmp_path, n=1)
    audio.save(rows[0]["wav"], np.zeros((2, 8000)), 16000, subtype="FLOAT")
    with pytest.raises(SystemExit, match="mono"):
        E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader())
    os.remove(rows[0]["wav"])
    with pytest.raises(SystemExit, match="missing cut"):
        E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader())


def test_unsafe_utt_id_refused(tmp_path):
    with pytest.raises(SystemExit, match="unsafe"):
        E.out_path(tmp_path, "C1", "../../evil")


@pytest.mark.skipif(os.name != "nt", reason="drive-relative paths are a Windows thing")
def test_drive_relative_utt_id_refused(tmp_path):
    with pytest.raises(SystemExit, match="unsafe"):
        E.out_path(tmp_path, "C1", "C:foo")


def _renamed(rows, ids):
    return [dict(r, utt_id=i) for r, i in zip(rows, ids)]


def test_ids_that_share_an_output_file_refused(tmp_path):
    rows = _renamed(_rows(tmp_path, n=2), ["a", "a.wav"])
    load = Loader()
    with pytest.raises(SystemExit, match="collide"):
        E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=load)
    assert load.n_loads == 0


@pytest.mark.skipif(os.name != "nt", reason="needs a case-insensitive filesystem")
def test_ids_differing_only_in_case_refused(tmp_path):
    rows = _renamed(_rows(tmp_path, n=2), ["Foo_u1", "foo_u1"])
    with pytest.raises(SystemExit, match="collide"):
        E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader())


def test_second_process_on_one_out_dir_refused(tmp_path):
    rows = _rows(tmp_path, n=1)
    out = tmp_path / "out"
    out.mkdir()
    with E.out_dir_lock(out):
        with pytest.raises(SystemExit, match="sequential"):
            E.run(rows, [FAKE], out, "cpu", loader=Loader())
    E.run(rows, [FAKE], out, "cpu", loader=Loader())  # released


def test_lock_is_released_after_a_failing_run(tmp_path):
    rows = _rows(tmp_path, n=1)
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        E.run(rows, [FAKE], out, "cpu", loader=Loader(fail_after=0))
    with E.out_dir_lock(out):
        pass


# ---------------------------------------------------------------------------
# Input level
# ---------------------------------------------------------------------------
def _speech_rows(tmp_path, level_db, n=2, fs=48000, thruster="off"):
    """Bursts of tone over a quiet floor: the energy VAD finds them, the level is known."""
    (tmp_path / "cuts").mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for k in range(n):
        t = np.arange(fs) / fs
        x = np.where(np.sin(2 * np.pi * 2 * t) > 0, 0.1 * np.sin(2 * np.pi * 440 * t), 0.0)
        x = x + 1e-5 * rng.standard_normal(t.size)
        p = tmp_path / "cuts" / f"u{thruster}{k}.wav"
        audio.save(str(p), x * 10 ** (level_db / 20), fs, subtype="FLOAT")
        rows.append({"utt_id": f"u{thruster}{k}", "system": "C0", "wav": str(p), "r0_wav": str(p), "transcript": "a",
                     "speaker": "s", "position": "p3", "thruster": thruster, "kind": "speech"})
    return rows


def _score_gain_db(row):
    score = _script("score")
    from uwse.evaluation import normalize as N

    r = score.level_normalize([dict(row)], N.VADConfig())[0]
    return 20 * np.log10(r["gain"])


def test_sim_level_feeds_scores_c0_gain_and_undoes_it_on_the_output(tmp_path):
    rows = _speech_rows(tmp_path, -12.0)
    load = Loader(sample_rate=48000)
    out = tmp_path / "out"
    E.run(rows, [FAKE48], out, "cpu", loader=load, input_level="sim")
    cuts = json.loads((out / "cuts.json").read_text())
    score = _script("score")
    for r, fed in zip(rows, load.se.fed):
        want = _score_gain_db(r)
        assert cuts[r["utt_id"]]["input_gain_db"] == pytest.approx(want, abs=1e-9)
        x, _ = audio.load(r["wav"])
        np.testing.assert_allclose(fed[0], x * 10 ** (want / 20), atol=1e-6)
        c0 = score._load16(r["wav"])
        y = score._load16(str(E.out_path(out, "C1", r["utt_id"])))
        np.testing.assert_allclose(y, c0, atol=1e-6)  # identity output is back on the C0 scale
    info = json.loads((out / "run_info.json").read_text())
    assert info["input_level_policy"] == "sim"


def test_sim_level_brings_active_level_to_minus_26(tmp_path):
    rows = _speech_rows(tmp_path, -12.0, fs=16000)
    load = Loader()
    E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=load, input_level="sim")
    from uwse.evaluation import normalize as N

    fed = load.se.fed[0][0].astype(float)
    r0, _ = audio.load(rows[0]["r0_wav"])
    assert N.active_rms_dbfs(fed, N.energy_vad(r0, 16000, N.VADConfig())) == pytest.approx(-26.0, abs=1e-3)


def test_none_level_feeds_the_cut_unchanged_and_records_zero_gain(tmp_path):
    rows = _speech_rows(tmp_path, -12.0, fs=16000)
    load = Loader()
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=load, input_level="none")
    x, _ = audio.load(rows[0]["wav"])
    np.testing.assert_allclose(load.se.fed[0][0], x, atol=1e-7)
    c = json.loads((out / "cuts.json").read_text())[rows[0]["utt_id"]]
    assert c["input_gain_db"] == 0.0 and c["score_gain_db"] != 0.0


def test_level_warning_uses_distance_from_simulator_level_per_thruster_state(tmp_path, capsys):
    rows = _speech_rows(tmp_path, 0.0, fs=16000, thruster="off") + _speech_rows(tmp_path, -12.0, fs=16000, thruster="on")
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=Loader(), input_level="none")
    txt = capsys.readouterr().out
    info = json.loads((out / "run_info.json").read_text())["input_level"]["active_dbfs_by_thruster"]
    assert set(info) == {"off", "on"} and abs(info["off"]["median"] - info["on"]["median"] - 12.0) < 0.5
    assert abs(info["off"]["median"] + 26) <= E.LEVEL_WARN_DB < abs(info["on"]["median"] + 26)
    assert "[warn] thruster on" in txt and "[warn] thruster off" not in txt


def test_sim_level_does_not_warn_about_level(tmp_path, capsys):
    rows = _speech_rows(tmp_path, 0.0, fs=16000)
    E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader(), input_level="sim")
    assert "[warn] thruster" not in capsys.readouterr().out


def test_sim_level_needs_a_vad_source(tmp_path):
    rows = _speech_rows(tmp_path, -12.0, n=1, fs=16000)
    del rows[0]["r0_wav"]
    with pytest.raises(SystemExit, match="no r0_wav"):
        E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader(), input_level="sim")


def test_changed_level_policy_refused_on_resume(tmp_path):
    rows = _speech_rows(tmp_path, -12.0, n=1, fs=16000)
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=Loader(), input_level="none")
    with pytest.raises(SystemExit, match="--input-level none"):
        E.run(rows, [FAKE], out, "cpu", loader=Loader(), input_level="sim")


def test_changed_gain_refused_on_resume(tmp_path):
    rows = _speech_rows(tmp_path, -12.0, n=1, fs=16000)
    out = tmp_path / "out"
    E.run(rows, [FAKE], out, "cpu", loader=Loader(), input_level="sim")
    r0 = tmp_path / "other_r0.wav"
    x, _ = audio.load(rows[0]["r0_wav"])
    x = x.copy()
    x[: x.size // 2] = 0.0  # a different VAD mask, so a different gain from the same cut
    audio.save(str(r0), x, 16000, subtype="FLOAT")
    with pytest.raises(SystemExit, match="differ"):
        E.run([dict(rows[0], r0_wav=str(r0))], [FAKE], out, "cpu", loader=Loader(), input_level="sim")


# ---------------------------------------------------------------------------
# Unlabeled clips
# ---------------------------------------------------------------------------
def _clip_root(tmp_path):
    root = tmp_path / "root"
    (root / "s1").mkdir(parents=True)
    for n in (0, 1):
        audio.save(str(root / "s1" / f"{n}.wav"), 0.1 * np.ones(8000), 16000, subtype="PCM_16")
    return root


def test_wav_list_mode_writes_session_tree(tmp_path):
    root = _clip_root(tmp_path)
    lst = tmp_path / "l.txt"
    lst.write_text("s1/0.wav\ns1/1.wav\n", encoding="utf-8")
    rows = E.rows_from_wav_list(str(lst), str(root))
    path, _ = E.run(rows, [FAKE], tmp_path / "out", "cpu", loader=Loader())
    assert (tmp_path / "out" / "C1" / "s1" / "0.wav").is_file()
    got = _read_jsonl(path)
    assert len(got) == 4 and all(r["r0_wav"].endswith(".wav") and "out" not in Path(r["r0_wav"]).parts for r in got)


def test_wav_list_mode(tmp_path):
    root = _clip_root(tmp_path)
    lst = tmp_path / "l.txt"
    lst.write_text("s1/0.wav\ns1/1.wav\n", encoding="utf-8")
    rows = E.rows_from_wav_list(str(lst), str(root))
    assert [r["utt_id"] for r in rows] == ["s1/0.wav", "s1/1.wav"]
    lst.write_text(f"{root / 's1' / '0.wav'}\n{root / 's1' / '0.wav'}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate"):
        E.rows_from_wav_list(str(lst), None)


# ---------------------------------------------------------------------------
# Real systems on very short synthetic audio (CPU)
# ---------------------------------------------------------------------------
def _speechy(tmp_path, fs=16000, dur=1.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(dur * fs)) / fs
    x = 0.05 * np.sin(2 * np.pi * 440 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.005 * rng.standard_normal(t.size)
    p = tmp_path / "c.wav"
    audio.save(str(p), x, fs, subtype="FLOAT")
    return [{"utt_id": "c", "system": "C0", "wav": str(p), "speaker": "0", "position": "p", "thruster": "off"}]


def _real(tmp_path, label, fs=16000):
    rows = _speechy(tmp_path, fs=fs)
    out = tmp_path / "out"
    _, done = E.run(rows, [E.FROZEN[label]], out, "cpu")
    y, fsy = sf.read(E.out_path(out, label, "c"), dtype="float64")
    assert done == [label] and fsy == 16000 and y.size == E.n_out_16k(audio.load(rows[0]["wav"])[0].size, fs)
    assert np.all(np.isfinite(y)) and np.sqrt(np.mean(y**2)) > 1e-5  # no gain here (--input-level none path)
    return json.loads((out / "run_info.json").read_text())["systems"][label]


def test_wpe1_real_run(tmp_path):
    pytest.importorskip("nara_wpe")
    info = _real(tmp_path, "C1")
    assert info["ckpt_sha256"] is None and info["device"] == "cpu" and info["adapter_sample_rate"] == 16000


@pytest.mark.skipif(not Path(E.FROZEN["C10"].ckpt).exists(), reason="C10 weights absent")
def test_c10_real_checkpoint_cpu(tmp_path):
    info = _real(tmp_path, "C10")
    assert info["ckpt_sha256"] == E.sha256(E.FROZEN["C10"].ckpt) and info["adapter_sample_rate"] == 16000


@pytest.mark.skipif(not Path(E.FROZEN["C2"].ckpt).exists()
                    or not (Path(os.environ.get("LOCALAPPDATA", "/nonexistent")) / "DeepFilterNet").exists(),
                    reason="C2 or DeepFilterNet zero-shot weights absent")
def test_c2_real_checkpoint_cpu_at_48k(tmp_path):
    info = _real(tmp_path, "C2", fs=48000)
    assert info["ckpt_sha256"] == E.sha256(E.FROZEN["C2"].ckpt) and info["adapter_sample_rate"] == 48000
