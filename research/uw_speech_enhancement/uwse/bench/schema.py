"""Real-benchmark metadata schema (RESEARCH_PLAN.md section 6.8).

One row of ``bench/metadata.jsonl`` per utterance x channel. The dataclasses
below mirror the section 6.8 JSON example field for field and in the same
order, so ``asdict(row)`` serializes to the documented layout.

Validation is strict: unknown keys, missing keys, wrong types and values
outside the enumerations are all reported. ``drr_dB`` is the only acoustic
quantity that may be ``null`` (the direct path cannot always be gated,
section 8.2). For non-LibriSpeech material (``newprompt_tts``, ``live``)
``librispeech_id`` and ``corpus_split`` must be ``null``.

The ``utt_id`` format is ``<release>_s<session>_p<position>_b<block>_ch<channel>_<item>``,
e.g. ``v0_s2_p3_b07_ch0_1089-134686-0000``; its fields must agree with the
row's ``bench_split``, ``position_id``, ``block_id``, ``channel`` and
``librispeech_id``; when ``session_id`` ends in ``_s<N>`` (``2026-12-02_s2``)
``<N>`` must equal the utt_id session. Floats must be finite (NaN/inf are not
JSON), coordinates are ``[x, y, z]`` in metres and ``throttle_pct`` lies in
[0, 100].
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import types
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

# ---------------------------------------------------------------- vocabularies
MATERIALS = ("librispeech", "newprompt_tts", "live")
CORPUS_SPLITS = ("test-clean", "dev-clean")
BENCH_SPLITS = ("v0-dev", "v0-test", "v1-test", "v1-train", "site2")
BLOCK_ORDERS = ("ABBA",)
THRUSTER_STATES = ("off", "on")
THRUSTER_UNITS = ("A", "B")
THRUSTER_DRIVES = ("trapezoidal", "foc")
POOL_ACTIVITY = ("empty", "swimmers")
PUMPS = ("on", "off")
SL_ABS_CALIBRATED = ("le4kHz", "full", "relative")
FLAGS = ("clip", "dropout", "align_low_psr", "pinger_band_energy")
OCTAVE_BANDS = ("500", "1000", "2000", "4000")

#: bench_split -> utt_id release prefix
RELEASE_OF_SPLIT = {
    "v0-dev": "v0",
    "v0-test": "v0",
    "v1-test": "v1",
    "v1-train": "v1",
    "site2": "site2",
}

UTT_ID_RE = re.compile(
    r"^(?P<release>v\d+|site\d+)_s(?P<session>\d+)_(?P<position>p[0-9a-z]+)"
    r"_b(?P<block>\d{2,})_ch(?P<channel>\d+)_(?P<item>[A-Za-z0-9][A-Za-z0-9_\-]*)$"
)
SESSION_ID_RE = re.compile(r"_s(?P<session>\d+)$")
LIBRISPEECH_ID_RE = re.compile(r"^(?P<speaker>\d+)-(?P<chapter>\d+)-(?P<idx>\d{4})$")


class SchemaError(ValueError):
    """Raised by :func:`validate`; ``.errors`` holds every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# ------------------------------------------------------------------ dataclasses
@dataclass
class Pool:
    name: str
    L_m: float
    W_m: float
    H_profile_m: list[float]
    construction: str
    salinity_gkg: float


@dataclass
class Thruster:
    state: str
    unit: str
    throttle_pct: float
    xyz_m: list[float]
    jet_axis_deg: float
    drive: str
    flow_noise_flag: bool


@dataclass
class Acoustic:
    snr_speechband_dB: float
    snr_octave_dB: dict[str, float]
    drr_dB: Optional[float]
    c50_dB: float
    edt_s: dict[str, float]
    t20_s: dict[str, float]
    two_slope: bool


@dataclass
class MetadataRow:
    utt_id: str
    librispeech_id: Optional[str]
    speaker_id: str
    transcript: str
    material: str
    corpus_split: Optional[str]
    bench_split: str
    prereg_id: str
    session_id: str
    session_order_idx: int
    block_id: int
    block_order: str
    anchor_repeat: bool
    pool: Pool
    water_temp_C: list[float]
    c_marczak_mps: float
    c_slope_mps: float
    c_slope_se_mps: float
    tau0_ms: float
    position_id: str
    src_xyz_m: list[float]
    src_orient_deg: float
    rx_xyz_m: dict[str, list[float]]
    range_tape_m: float
    thruster: Thruster
    pool_activity: str
    pumps: str
    emi_check_id: str
    projector: str
    amp: str
    playback_gain_dB: float
    SL_speechactive_rms_dB_re1uPa_1m: float
    SL_peak_dB: float
    SL_abs_calibrated: str
    daisy_board: str
    daisy_rev: str
    preamp_gain_dB: float
    channel: int
    scored_channel: bool
    fs_raw: int
    bits: int
    raw_file: str
    start_sample_raw: int
    end_sample_raw: int
    drift_ppm: float
    fine_lag_samples_16k: int
    chirp_psr_dB: float
    ir_id: str
    acoustic: Acoustic
    flags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENUMS: dict[str, tuple[str, ...]] = {
    "material": MATERIALS,
    "corpus_split": CORPUS_SPLITS,
    "bench_split": BENCH_SPLITS,
    "block_order": BLOCK_ORDERS,
    "pool_activity": POOL_ACTIVITY,
    "pumps": PUMPS,
    "SL_abs_calibrated": SL_ABS_CALIBRATED,
    "thruster.state": THRUSTER_STATES,
    "thruster.unit": THRUSTER_UNITS,
    "thruster.drive": THRUSTER_DRIVES,
}


# ------------------------------------------------------------- type checking
def _is_optional(tp: Any) -> tuple[bool, Any]:
    origin = typing.get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(tp)) == 2:
            return True, args[0]
    return False, tp


def _check_type(value: Any, tp: Any, path: str, errors: list[str]) -> None:
    optional, tp = _is_optional(tp)
    if value is None:
        if not optional:
            errors.append(f"{path}: null not allowed")
        return
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {type(value).__name__}")
            return
        _check_dataclass(value, tp, path + ".", errors)
        return
    origin = typing.get_origin(tp)
    if origin is list:
        if not isinstance(value, list):
            errors.append(f"{path}: expected list, got {type(value).__name__}")
            return
        (sub,) = typing.get_args(tp)
        for i, v in enumerate(value):
            _check_type(v, sub, f"{path}[{i}]", errors)
        return
    if origin is dict:
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {type(value).__name__}")
            return
        _, sub = typing.get_args(tp)
        for k, v in value.items():
            if not isinstance(k, str):
                errors.append(f"{path}: non-string key {k!r}")
            _check_type(v, sub, f"{path}.{k}", errors)
        return
    # JSON booleans are Python bools, which are also ints: keep them apart.
    if tp is bool:
        ok = isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif tp is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok and not math.isfinite(value):
            # NaN/inf are not JSON; unknown quantities are null only where allowed.
            errors.append(f"{path}: non-finite value {value!r}")
            return
    elif tp is str:
        ok = isinstance(value, str)
    else:  # pragma: no cover - schema only uses the types above
        ok = isinstance(value, tp)
    if not ok:
        errors.append(f"{path}: expected {tp.__name__}, got {type(value).__name__}")


def _check_dataclass(d: dict, cls: type, prefix: str, errors: list[str]) -> None:
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    for k in d:
        if k not in hints:
            errors.append(f"{prefix}{k}: unknown key")
    for n in names:
        if n not in d:
            errors.append(f"{prefix}{n}: missing")
        else:
            _check_type(d[n], hints[n], prefix + n, errors)


def _get(d: dict, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


# ------------------------------------------------------------------ public API
def make_utt_id(release: str, session: int, position_id: str, block_id: int,
                channel: int, item_id: str) -> str:
    """Build ``v0_s2_p3_b07_ch0_1089-134686-0000`` style ids."""
    if not position_id.startswith("p"):
        position_id = "p" + position_id
    uid = f"{release}_s{int(session)}_{position_id}_b{int(block_id):02d}_ch{int(channel)}_{item_id}"
    if not UTT_ID_RE.match(uid):
        raise SchemaError([f"utt_id: cannot build a valid id from {uid!r}"])
    return uid


def parse_utt_id(utt_id: str) -> dict[str, Any]:
    """Split an utt_id into release, session, position, block, channel, item."""
    m = UTT_ID_RE.match(utt_id)
    if not m:
        raise SchemaError([f"utt_id: {utt_id!r} does not match {UTT_ID_RE.pattern}"])
    g = m.groupdict()
    return {
        "release": g["release"],
        "session": int(g["session"]),
        "position_id": g["position"],
        "block_id": int(g["block"]),
        "channel": int(g["channel"]),
        "item_id": g["item"],
    }


def validation_errors(row: dict | MetadataRow) -> list[str]:
    """Return every schema problem in ``row`` (empty list when valid)."""
    if isinstance(row, MetadataRow):
        row = row.to_dict()
    if not isinstance(row, dict):
        return [f"row: expected object, got {type(row).__name__}"]
    errors: list[str] = []
    _check_dataclass(row, MetadataRow, "", errors)

    for key, allowed in _ENUMS.items():
        v = _get(row, key)
        if v is not None and isinstance(v, str) and v not in allowed:
            errors.append(f"{key}: {v!r} not in {allowed}")
    for i, fl in enumerate(row.get("flags") or []):
        if isinstance(fl, str) and fl not in FLAGS:
            errors.append(f"flags[{i}]: {fl!r} not in {FLAGS}")
    octs = _get(row, "acoustic.snr_octave_dB")
    if isinstance(octs, dict):
        for k in octs:
            if k not in OCTAVE_BANDS:
                errors.append(f"acoustic.snr_octave_dB: band {k!r} not in {OCTAVE_BANDS}")

    # Material-dependent nullability.
    material = row.get("material")
    lid = row.get("librispeech_id")
    if material == "librispeech":
        if lid is None:
            errors.append("librispeech_id: required for material 'librispeech'")
        elif isinstance(lid, str):
            m = LIBRISPEECH_ID_RE.match(lid)
            if not m:
                errors.append(f"librispeech_id: {lid!r} is not <speaker>-<chapter>-<NNNN>")
            elif row.get("speaker_id") != m["speaker"]:
                errors.append(f"speaker_id: {row.get('speaker_id')!r} != speaker in {lid!r}")
        if row.get("corpus_split") is None:
            errors.append("corpus_split: required for material 'librispeech'")
    elif material in ("newprompt_tts", "live"):
        if lid is not None:
            errors.append(f"librispeech_id: must be null for material {material!r}")
        if row.get("corpus_split") is not None:
            errors.append(f"corpus_split: must be null for material {material!r}")

    # Split <-> corpus consistency (section 5.4).
    bs, cs = row.get("bench_split"), row.get("corpus_split")
    if material == "librispeech" and bs in BENCH_SPLITS and cs in CORPUS_SPLITS:
        want = "test-clean" if bs in ("v0-test", "v1-test") else "dev-clean"
        if bs != "site2" and cs != want:
            errors.append(f"corpus_split: {bs} must come from {want}, got {cs}")

    if isinstance(row.get("channel"), int) and row["channel"] < 0:
        errors.append("channel: must be >= 0")
    s0, s1 = row.get("start_sample_raw"), row.get("end_sample_raw")
    if isinstance(s0, int) and isinstance(s1, int) and (s0 < 0 or s1 < s0):
        errors.append(f"start/end_sample_raw: need 0 <= start <= end, got {s0}, {s1}")
    wt = row.get("water_temp_C")
    if isinstance(wt, list) and len(wt) != 2:
        errors.append("water_temp_C: expected [start, end]")
    # Coordinates are [x, y, z] in metres.
    src = row.get("src_xyz_m")
    if isinstance(src, list) and len(src) != 3:
        errors.append(f"src_xyz_m: expected [x, y, z], got {len(src)} values")
    rx = row.get("rx_xyz_m")
    if isinstance(rx, dict):
        for k, v in rx.items():
            if isinstance(v, list) and len(v) != 3:
                errors.append(f"rx_xyz_m.{k}: expected [x, y, z], got {len(v)} values")
    txyz = _get(row, "thruster.xyz_m")
    if isinstance(txyz, list) and len(txyz) not in (0, 3):
        errors.append(f"thruster.xyz_m: expected [] or [x, y, z], got {len(txyz)} values")
    thr = _get(row, "thruster.throttle_pct")
    if isinstance(thr, (int, float)) and not isinstance(thr, bool) and not 0 <= thr <= 100:
        errors.append(f"thruster.throttle_pct: {thr} outside [0, 100]")

    # utt_id must agree with the row.
    uid = row.get("utt_id")
    if isinstance(uid, str):
        m = UTT_ID_RE.match(uid)
        if not m:
            errors.append(f"utt_id: {uid!r} does not match <rel>_s<N>_p<pos>_b<NN>_ch<N>_<item>")
        else:
            g = m.groupdict()
            if bs in RELEASE_OF_SPLIT and g["release"] != RELEASE_OF_SPLIT[bs]:
                errors.append(f"utt_id: release {g['release']!r} inconsistent with bench_split {bs!r}")
            sid = row.get("session_id")
            ms = SESSION_ID_RE.search(sid) if isinstance(sid, str) else None
            if ms and int(ms["session"]) != int(g["session"]):
                errors.append(f"utt_id: session s{g['session']} != session_id {sid!r}")
            if row.get("position_id") is not None and g["position"] != row.get("position_id"):
                errors.append(f"utt_id: position {g['position']!r} != position_id {row.get('position_id')!r}")
            if isinstance(row.get("block_id"), int) and int(g["block"]) != row["block_id"]:
                errors.append(f"utt_id: block {g['block']} != block_id {row['block_id']}")
            if isinstance(row.get("channel"), int) and int(g["channel"]) != row["channel"]:
                errors.append(f"utt_id: channel {g['channel']} != channel {row['channel']}")
            if material == "librispeech" and isinstance(lid, str) and g["item"] != lid:
                errors.append(f"utt_id: item {g['item']!r} != librispeech_id {lid!r}")
            rx = row.get("rx_xyz_m")
            if isinstance(rx, dict) and f"ch{g['channel']}" not in rx:
                errors.append(f"rx_xyz_m: no entry for ch{g['channel']}")
    return errors


def validate(row: dict | MetadataRow) -> None:
    """Raise :class:`SchemaError` listing every problem; return None if valid."""
    errs = validation_errors(row)
    if errs:
        raise SchemaError(errs)


def from_dict(d: dict, check: bool = True) -> MetadataRow:
    """Build a :class:`MetadataRow` (nested dataclasses included) from a dict."""
    if check:
        validate(d)
    d = dict(d)
    d["pool"] = Pool(**d["pool"])
    d["thruster"] = Thruster(**d["thruster"])
    d["acoustic"] = Acoustic(**d["acoustic"])
    return MetadataRow(**d)


def write_jsonl(path: str | Path, rows: Iterable[dict | MetadataRow], check: bool = True) -> int:
    """Write one JSON object per line; returns the number of rows."""
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            d = r.to_dict() if isinstance(r, MetadataRow) else r
            if check:
                validate(d)
            # allow_nan=False: NaN/inf would make invalid JSON even with check=False.
            f.write(json.dumps(d, ensure_ascii=False, allow_nan=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path, check: bool = True) -> Iterator[MetadataRow]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            try:
                yield from_dict(d, check=check)
            except SchemaError as e:
                raise SchemaError([f"line {lineno}: {m}" for m in e.errors]) from None


def read_jsonl(path: str | Path, check: bool = True) -> list[MetadataRow]:
    return list(iter_jsonl(path, check=check))


def example_row() -> dict[str, Any]:
    """The section 6.8 example as a valid dict (used by tests and as a template)."""
    return {
        "utt_id": "v0_s2_p3_b07_ch0_1089-134686-0000",
        "librispeech_id": "1089-134686-0000", "speaker_id": "1089", "transcript": "...",
        "material": "librispeech", "corpus_split": "test-clean",
        "bench_split": "v0-test", "prereg_id": "osf:...",
        "session_id": "2026-12-02_s2", "session_order_idx": 0, "block_id": 7,
        "block_order": "ABBA", "anchor_repeat": False,
        "pool": {"name": "...", "L_m": 0.0, "W_m": 0.0, "H_profile_m": [],
                 "construction": "tile-on-concrete", "salinity_gkg": 0.0},
        "water_temp_C": [26.8, 26.9], "c_marczak_mps": 1501.0, "c_slope_mps": 1500.2,
        "c_slope_se_mps": 6.6, "tau0_ms": 0.0,
        "position_id": "p3", "src_xyz_m": [0, 0, 0], "src_orient_deg": 0,
        "rx_xyz_m": {"ch0": [0, 0, 0]}, "range_tape_m": 6.00,
        "thruster": {"state": "off", "unit": "A", "throttle_pct": 0, "xyz_m": [],
                     "jet_axis_deg": 90, "drive": "trapezoidal", "flow_noise_flag": False},
        "pool_activity": "empty", "pumps": "on", "emi_check_id": "emi_2026-12-02",
        "projector": "LL916C+AC205C", "amp": "...", "playback_gain_dB": -12.0,
        "SL_speechactive_rms_dB_re1uPa_1m": 155.0, "SL_peak_dB": 169.0,
        "SL_abs_calibrated": "le4kHz",
        "daisy_board": "376636603433", "daisy_rev": "rev7/PCM3060", "preamp_gain_dB": 40,
        "channel": 0, "scored_channel": True,
        "fs_raw": 96000, "bits": 24, "raw_file": "raw/ch0/12.wav",
        "start_sample_raw": 0, "end_sample_raw": 0,
        "drift_ppm": 0.0, "fine_lag_samples_16k": 0, "chirp_psr_dB": 0.0, "ir_id": "ess_p3_s2_a",
        "acoustic": {"snr_speechband_dB": 0.0,
                     "snr_octave_dB": {"500": 0.0, "1000": 0.0, "2000": 0.0, "4000": 0.0},
                     "drr_dB": None, "c50_dB": 0.0, "edt_s": {}, "t20_s": {}, "two_slope": False},
        "flags": ["clip", "dropout", "align_low_psr", "pinger_band_energy"], "notes": "",
    }
