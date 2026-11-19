"""Exploratory: does equalising per-band SNR close the noise0 female-male WER gap?

Two single-factor variants of the E7 noise0 cell (same 200 v0-test utterances, 1 / 3 / 6 m
scenes and per-item seed as `sim_baseline.py`), run through the stress-suite machinery:

- ``n0_ssn``: the -3 dB/oct noise replaced by noise with each item's own received-speech
  spectrum (1/3-octave smoothed), SNR 0 dB in 500-8000 Hz. Per-band SNR is then flat and
  the same for every talker.
- ``n0_fullband``: the noise0 noise, but the 0 dB SNR is set over 50-8000 Hz, so speech
  energy below 500 Hz counts towards the SNR instead of being a free bonus.

Systems: C0 (``noisy``) and the frozen C8 adapter (``c8``), guarded decode. The reading
(F - M gap per cell vs the E7 noise0 gap) is fixed in RESULTS_E7_SIM_DRYRUN.md before the run.

    python scripts/sexgap_eval.py --out runs/sexgap_v0
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stress_eval as st  # noqa: E402

CONDITIONS = {
    "n0_ssn": (0.0, {"ssn_on": True, "ssn_rel_db": 0.0}),
    "n0_fullband": (0.0, {"noise_band": [50.0, 8000.0]}),
}

if __name__ == "__main__":
    st.CONDITIONS = CONDITIONS
    argv = sys.argv[1:]
    if "--out" not in argv:
        argv += ["--out", str(ROOT / "runs" / "sexgap_v0")]
    raise SystemExit(st.main(argv))
