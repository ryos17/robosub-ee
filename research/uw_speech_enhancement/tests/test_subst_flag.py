"""Substitution-hallucination flag and the paired keep-mask (protocol addition 2026-11-08)."""
from __future__ import annotations

import numpy as np
import pytest

from uwse.evaluation.asr import paired_unflagged_mask, substitution_flags

REF = "in all the novel had a difficult gestation"


def test_substitution_flags():
    hyps = [
        REF,                                                      # correct
        "it all was not as bad as it was going to be at the station",  # fluent, 1-2 hits
        "the novel",                                               # dropout, not flagged
        "",                                                        # empty, not flagged
        "in all the novel had a hard time",                        # partly right
    ]
    f = substitution_flags([REF] * len(hyps), hyps)
    assert f.tolist() == [False, True, False, False, False]
    assert substitution_flags([""], ["hello there"]).tolist() == [False]


def test_paired_unflagged_mask():
    m = paired_unflagged_mask({"C0": [True, False, False], "C8": [False, False, True]})
    assert m.tolist() == [False, True, False]
    with pytest.raises(ValueError):
        paired_unflagged_mask({"C0": [True], "C8": [True, False]})
    assert paired_unflagged_mask({}).shape == (0,)
    assert isinstance(m, np.ndarray)
