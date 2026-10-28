"""LibriSpeech manifests and benchmark utterance selection (RESEARCH_PLAN.md 5.1, 5.4).

Manifest
--------
:func:`build_manifest` walks an OpenSLR-12 subset directory
(``<subset>/<speaker>/<chapter>/<speaker>-<chapter>-<NNNN>.flac`` plus one
``<speaker>-<chapter>.trans.txt`` per chapter) and returns one
:class:`Utterance` per FLAC with its transcript and duration
(``soundfile.info``, no decoding).

Selection (all deterministic given ``seed``)
--------------------------------------------
- **v0-test**: 200 test-clean utterances, 5 per speaker x 40 speakers
  (:func:`select_v0_test`).
- **v0-dev**: 40 dev-clean utterances, spread round-robin over speakers so
  each of the 40 dev-clean speakers contributes one (:func:`select_v0_dev`).
- **v1-test**: >= 1,000 test-clean utterances disjoint from v0-test, balanced
  round-robin over speakers (:func:`select_v1_test`).

Duration filter (documented choice, not from the plan): candidates must last
between ``MIN_DUR_S = 2.0`` s and ``MAX_DUR_S = 15.0`` s. Very short items
give unstable WER (one word error dominates) and too few frames for the
per-utterance lag search (5.4/6.5); very long items would overrun the ~60 s
playback block and dominate a block's time-of-day. The plan's playback budget
(Appendix A) uses the unfiltered test-clean mean of about 7.4 s; the filter
shifts the selected mean slightly, see :func:`summary`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
from tqdm import tqdm

MIN_DUR_S = 2.0
MAX_DUR_S = 15.0
DEFAULT_SEED = 20261027

SUBSETS = ("train-clean-100", "train-clean-360", "train-other-500",
           "dev-clean", "dev-other", "test-clean", "test-other")


@dataclass(frozen=True)
class Utterance:
    utt_id: str          # LibriSpeech id, <speaker>-<chapter>-<NNNN>
    speaker: str
    chapter: str
    transcript: str
    path: str
    duration_s: float
    subset: str


def _read_transcripts(trans_file: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uid, _, text = line.partition(" ")
            out[uid] = text
    return out


def find_subset_dir(root: str | Path, subset: str) -> Path:
    """Locate ``subset`` under ``root`` (accepts root, root/LibriSpeech, or the subset dir)."""
    root = Path(root)
    for cand in (root / subset, root / "LibriSpeech" / subset, root):
        if cand.is_dir() and cand.name == subset:
            return cand
    raise FileNotFoundError(f"LibriSpeech subset {subset!r} not found under {root}")


def build_manifest(subset_dir: str | Path, subset: str | None = None,
                   with_duration: bool = True, progress: bool = True) -> list[Utterance]:
    """Scan one subset directory; returns utterances sorted by id.

    ``subset`` defaults to the directory name. A FLAC without a transcript, or
    a transcript line without its FLAC, raises ``ValueError`` (a corrupt or
    partial extraction should not pass silently).
    """
    subset_dir = Path(subset_dir)
    subset = subset or subset_dir.name
    trans_files = sorted(subset_dir.glob("*/*/*.trans.txt"))
    if not trans_files:
        raise FileNotFoundError(f"no *.trans.txt under {subset_dir}")
    utts: list[Utterance] = []
    for tf in tqdm(trans_files, desc=f"manifest {subset}", disable=not progress):
        texts = _read_transcripts(tf)
        speaker, chapter = tf.parent.parent.name, tf.parent.name
        flacs = {p.stem: p for p in tf.parent.glob("*.flac")}
        missing = sorted(set(flacs) - set(texts))
        if missing:
            raise ValueError(f"{tf}: no transcript for {missing[:3]}")
        orphan = sorted(set(texts) - set(flacs))
        if orphan:  # transcript lines without audio: truncated download/extraction
            raise ValueError(f"{tf}: no FLAC for {orphan[:3]}")
        for uid in sorted(flacs):
            p = flacs[uid]
            dur = float(sf.info(str(p)).duration) if with_duration else float("nan")
            utts.append(Utterance(uid, speaker, chapter, texts[uid], str(p.resolve()), dur, subset))
    utts.sort(key=lambda u: u.utt_id)
    return utts


def write_manifest(path: str | Path, utts: Iterable[Utterance]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for u in utts:
            f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")


def read_manifest(path: str | Path) -> list[Utterance]:
    with open(path, "r", encoding="utf-8") as f:
        return [Utterance(**json.loads(line)) for line in f if line.strip()]


def filter_duration(utts: Iterable[Utterance], min_s: float = MIN_DUR_S,
                    max_s: float = MAX_DUR_S) -> list[Utterance]:
    return [u for u in utts if min_s <= u.duration_s <= max_s]


def by_speaker(utts: Iterable[Utterance]) -> dict[str, list[Utterance]]:
    out: dict[str, list[Utterance]] = {}
    for u in utts:
        out.setdefault(u.speaker, []).append(u)
    return {k: sorted(v, key=lambda u: u.utt_id) for k, v in sorted(out.items())}


def _require_subset(utts: Sequence[Utterance], subset: str, what: str) -> None:
    bad = {u.subset for u in utts if u.subset != subset}
    if bad:
        raise ValueError(f"{what} must be drawn from {subset}, manifest contains {sorted(bad)}")


def _round_robin(groups: dict[str, list[Utterance]], n: int,
                 rng: np.random.Generator) -> list[Utterance]:
    """Take ``n`` items cycling over speakers (shuffled order, shuffled pools)."""
    pools = {k: list(rng.permutation(len(v))) for k, v in groups.items()}
    order = [list(groups)[i] for i in rng.permutation(len(groups))]
    picked: list[Utterance] = []
    while len(picked) < n:
        progressed = False
        for spk in order:
            if pools[spk] and len(picked) < n:
                picked.append(groups[spk][pools[spk].pop(0)])
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(picked)} utterances available, {n} requested")
    return picked


def select_v0_test(manifest: Sequence[Utterance], n_speakers: int = 40, n_per_speaker: int = 5,
                   seed: int = DEFAULT_SEED, min_s: float = MIN_DUR_S,
                   max_s: float = MAX_DUR_S) -> list[Utterance]:
    """v0-test: ``n_per_speaker`` utterances from each of ``n_speakers`` test-clean speakers."""
    _require_subset(manifest, "test-clean", "v0-test")
    groups = by_speaker(filter_duration(manifest, min_s, max_s))
    eligible = [s for s, v in groups.items() if len(v) >= n_per_speaker]
    if len(eligible) < n_speakers:
        raise ValueError(f"only {len(eligible)} speakers have >= {n_per_speaker} eligible "
                         f"utterances, need {n_speakers}")
    rng = np.random.default_rng([seed, 0])
    speakers = sorted(rng.choice(eligible, size=n_speakers, replace=False).tolist())
    out: list[Utterance] = []
    for s in speakers:
        idx = sorted(rng.choice(len(groups[s]), size=n_per_speaker, replace=False).tolist())
        out.extend(groups[s][i] for i in idx)
    return sorted(out, key=lambda u: u.utt_id)


def select_v0_dev(manifest: Sequence[Utterance], n: int = 40, seed: int = DEFAULT_SEED,
                  min_s: float = MIN_DUR_S, max_s: float = MAX_DUR_S) -> list[Utterance]:
    """v0-dev: ``n`` dev-clean utterances, round-robin over speakers."""
    _require_subset(manifest, "dev-clean", "v0-dev")
    groups = by_speaker(filter_duration(manifest, min_s, max_s))
    rng = np.random.default_rng([seed, 1])
    return sorted(_round_robin(groups, n, rng), key=lambda u: u.utt_id)


def select_v1_test(manifest: Sequence[Utterance], exclude: Iterable[Utterance | str],
                   n: int = 1000, seed: int = DEFAULT_SEED, min_s: float = MIN_DUR_S,
                   max_s: float = MAX_DUR_S) -> list[Utterance]:
    """v1-test: ``n`` test-clean utterances not in ``exclude`` (normally v0-test)."""
    _require_subset(manifest, "test-clean", "v1-test")
    if n < 1000:
        raise ValueError("v1-test needs >= 1,000 utterances (RESEARCH_PLAN 5.1)")
    ex = {e if isinstance(e, str) else e.utt_id for e in exclude}
    pool = [u for u in filter_duration(manifest, min_s, max_s) if u.utt_id not in ex]
    rng = np.random.default_rng([seed, 2])
    return sorted(_round_robin(by_speaker(pool), n, rng), key=lambda u: u.utt_id)


def summary(utts: Sequence[Utterance]) -> dict[str, float]:
    d = np.array([u.duration_s for u in utts], dtype=float)
    return {
        "n": len(utts),
        "n_speakers": len({u.speaker for u in utts}),
        "total_h": float(d.sum() / 3600.0) if len(d) else 0.0,
        "mean_s": float(d.mean()) if len(d) else float("nan"),
        "min_s": float(d.min()) if len(d) else float("nan"),
        "max_s": float(d.max()) if len(d) else float("nan"),
    }
