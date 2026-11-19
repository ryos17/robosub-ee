"""Strict split checks (RESEARCH_PLAN.md section 5.4).

Speaker groups, which must be pairwise disjoint:

=========  ==========================================  =====================
group      roles                                       LibriSpeech subsets
=========  ==========================================  =====================
train      sim-train                                   train-clean-100/360
dev        sim-val, v0-dev, real-train (v1-train)      dev-clean
test       v0-test, v1-test (and their sim twins)      test-clean
=========  ==========================================  =====================

Roles inside the *test* group must also be utterance-disjoint (v1-test uses
"different" test-clean utterances from v0-test, 5.1). Roles inside the *dev*
group may share speakers and utterances (they are all decision/training data).

Noise: training uses T200 unit A; testing uses unit B (5.2, 5.4).
Positions: real fine-tuning uses positions disjoint from the test positions of
that analysis, and the measured-IR replay set never uses benchmark positions
(5.3, 5.4); :func:`assert_positions_disjoint` checks both.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping

ROLE_GROUP: dict[str, str] = {
    "sim-train": "train",
    "sim-val": "dev",
    "v0-dev": "dev",
    "real-train": "dev",
    "v1-train": "dev",
    "v0-test": "test",
    "v1-test": "test",
    "sim-twin-v0": "test",
    "sim-twin-v1": "test",
}

GROUP_SUBSETS: dict[str, frozenset[str]] = {
    "train": frozenset({"train-clean-100", "train-clean-360"}),
    "dev": frozenset({"dev-clean"}),
    "test": frozenset({"test-clean"}),
}

#: roles in the same group that must not share utterances
UTTERANCE_DISJOINT_GROUPS = frozenset({"test"})

NOISE_UNIT: dict[str, str] = {"train": "A", "test": "B"}


class SplitLeakError(AssertionError):
    """Raised when a split rule of section 5.4 is violated; ``.violations`` lists them."""

    def __init__(self, violations: list[str]):
        self.violations = list(violations)
        super().__init__("\n".join(self.violations))


def _fields(u: Any) -> tuple[str, str, str | None]:
    """(utt_id, speaker, subset) from an Utterance, a manifest dict or a metadata dict."""
    if isinstance(u, Mapping):
        uid = u.get("librispeech_id") or u.get("utt_id")
        spk = u.get("speaker") or u.get("speaker_id")
        subset = u.get("subset") or u.get("corpus_split")
    else:
        uid, spk, subset = u.utt_id, u.speaker, getattr(u, "subset", None)
    if uid is None or spk is None:
        raise ValueError(f"cannot read utt_id/speaker from {u!r}")
    return str(uid), str(spk), subset


def split_violations(manifests: Mapping[str, Iterable[Any]]) -> list[str]:
    """Return every section 5.4 violation among the named role manifests."""
    v: list[str] = []
    utts: dict[str, set[str]] = {}
    spks: dict[str, set[str]] = {}
    for role, items in manifests.items():
        if role not in ROLE_GROUP:
            v.append(f"unknown role {role!r}; known: {sorted(ROLE_GROUP)}")
            continue
        group = ROLE_GROUP[role]
        utts[role], spks[role] = set(), set()
        bad_subsets: set[str] = set()
        for it in items:
            uid, spk, subset = _fields(it)
            utts[role].add(uid)
            spks[role].add(spk)
            if subset is not None and subset not in GROUP_SUBSETS[group]:
                bad_subsets.add(subset)
        if bad_subsets:
            v.append(f"{role}: subsets {sorted(bad_subsets)} not allowed "
                     f"(expected {sorted(GROUP_SUBSETS[group])})")

    roles = list(utts)
    for a, b in combinations(roles, 2):
        ga, gb = ROLE_GROUP[a], ROLE_GROUP[b]
        if ga != gb:
            shared = spks[a] & spks[b]
            if shared:
                v.append(f"speaker leak {a} ({ga}) / {b} ({gb}): {sorted(shared)[:10]}"
                         f"{' ...' if len(shared) > 10 else ''} ({len(shared)} speakers)")
        elif ga in UTTERANCE_DISJOINT_GROUPS and not _twin_pair(a, b):
            shared = utts[a] & utts[b]
            if shared:
                v.append(f"utterance overlap {a} / {b}: {sorted(shared)[:10]}"
                         f"{' ...' if len(shared) > 10 else ''} ({len(shared)} utterances)")
    return v


def _twin_pair(a: str, b: str) -> bool:
    """A sim twin legitimately shares utterances with the real set it mirrors."""
    pair = {a, b}
    return pair in ({"v0-test", "sim-twin-v0"}, {"v1-test", "sim-twin-v1"})


def assert_disjoint(manifests: Mapping[str, Iterable[Any]]) -> None:
    """Raise :class:`SplitLeakError` if the role manifests break section 5.4.

    ``manifests`` maps a role name (keys of :data:`ROLE_GROUP`) to an iterable
    of :class:`uwse.bench.librispeech.Utterance`, manifest dicts or metadata
    rows. Iterables are consumed once.
    """
    v = split_violations({k: list(x) for k, x in manifests.items()})
    if v:
        raise SplitLeakError(v)


def check_noise_unit(group: str, unit: str) -> None:
    """Raise if a noise recording from ``unit`` is used in ``group`` ('train'/'test')."""
    if group not in NOISE_UNIT:
        raise ValueError(f"group must be one of {sorted(NOISE_UNIT)}")
    if unit != NOISE_UNIT[group]:
        raise SplitLeakError([f"noise unit {unit!r} used for {group}; "
                              f"{group} uses unit {NOISE_UNIT[group]!r} only"])


def assert_noise_split(train_units: Iterable[str], test_units: Iterable[str]) -> None:
    """Train noise from unit A only, test noise from unit B only."""
    v = []
    for u in sorted(set(train_units)):
        if u != NOISE_UNIT["train"]:
            v.append(f"train noise uses unit {u!r}")
    for u in sorted(set(test_units)):
        if u != NOISE_UNIT["test"]:
            v.append(f"test noise uses unit {u!r}")
    if v:
        raise SplitLeakError(v)


def assert_positions_disjoint(train_positions: Iterable[str],
                              test_positions: Iterable[str], what: str = "real-train") -> None:
    """Positions used for training/fitting (``what``) must not be test positions."""
    shared = set(train_positions) & set(test_positions)
    if shared:
        raise SplitLeakError([f"{what} positions overlap test positions: {sorted(shared)}"])
