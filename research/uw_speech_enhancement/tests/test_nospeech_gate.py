import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from uwse.evaluation import nospeech_gate as G  # noqa: E402
import eval_nonspeech as E  # noqa: E402


def test_split_windows():
    assert [len(w) for w in G.split_windows(np.zeros(10))] == [10]
    assert [len(w) for w in G.split_windows(np.zeros(G.WIN))] == [G.WIN]
    assert [len(w) for w in G.split_windows(np.zeros(G.WIN + 1))] == [G.WIN, 1]
    x = np.arange(2 * G.WIN + 5, dtype=np.float32)
    ws = G.split_windows(x)
    assert len(ws) == 3 and np.array_equal(np.concatenate(ws), x)


def test_each_window_from_own_audio():
    def fn(asr, w):
        return float(w.mean())

    x = np.concatenate([np.full(G.WIN, 1.0), np.full(G.WIN, 3.0), np.full(7, 5.0)]).astype(np.float32)
    assert G.window_p_ns(None, x, fn=fn) == [1.0, 3.0, 5.0]
    short = np.full(100, 2.0, np.float32)
    assert G.window_p_ns(None, short, fn=fn) == [2.0]


def test_segment_window_by_start():
    assert G.seg_window(0.0) == 0 and G.seg_window(29.99) == 0 and G.seg_window(30.0) == 1 and G.seg_window(61) == 2


def test_gate_drops_by_start_and_keeps_order():
    segs = [{"start": 1.0, "end": 31.0, "text": " a"}, {"start": 31.0, "end": 40.0, "text": " b"},
            {"start": 41.0, "end": 45.0, "text": " c"}]
    t, d = G.gate_text("a b c", segs, [0.9, 0.001])
    assert d == [True, False, False] and t == "b c"
    t, d = G.gate_text("a b c", segs, [0.001, 0.9])
    assert d == [False, True, True] and t == "a"
    t, d = G.gate_text("a b c", segs, [0.001, 0.001])
    assert t == "a b c" and not any(d)


def test_gate_threshold_is_inclusive_and_short_clips():
    seg = [{"start": 0.0, "end": 3.0, "text": " x"}]
    assert G.gate_text("x", seg, [G.TAU])[0] == ""
    assert G.gate_text("x", seg, [G.TAU - 1e-6])[0] == "x"
    assert G.gate_text("hello", [], [0.9])[0] == ""
    assert G.gate_text("hello", [], [0.01])[0] == "hello"


def test_wilson():
    lo, hi = G.wilson(0, 360)
    assert lo == 0.0 and abs(hi - 0.0106) < 5e-4
    lo, hi = G.wilson(151, 360)
    assert abs(lo - 0.3696) < 5e-4 and abs(hi - 0.4710) < 5e-4


def test_summarize_counts_normalized_words():
    dec = [{"noise_type": "bubbles", "text": " Thank you.", "p_ns": [0.6],
            "segments": [{"start": 0.0, "end": 2.0, "text": " Thank you."}]},
           {"noise_type": "bubbles", "text": "", "p_ns": [0.9], "segments": []},
           {"noise_type": "thruster", "text": " Ah", "p_ns": [0.01],
            "segments": [{"start": 0.0, "end": 1.0, "text": " Ah"}]}]
    s = E.summarize(dec)
    assert s["overall"]["C8"]["with_output"] == 2 and s["overall"]["C8g"]["with_output"] == 1
    assert s["by_noise_type"]["bubbles"]["C8g"]["with_output"] == 0
    assert s["by_noise_type"]["thruster"]["C8g"]["words"] == 1
