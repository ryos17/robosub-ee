"""Import one pool take: chirp sync, clock-drift correction, per-utterance cuts.

A take is one recorder file that captured one played session file
(``session/<pos>_speech.wav``, or ``anchor_<pos>.wav`` with ``--anchor``). See
`uwse.bench.ingest` for the method.

    python scripts/import_recording.py --bench data/bench/v0_pilot --position p3 \
        --rec raw/2026-12-02/take07.wav --session 2 --channels 1 --sync-channel 0 \
        --out data/real/v0_s2

Writes under ``--out``:

- ``wav/<utt_id>.wav``  one cut per utterance x recorded channel, at the playback rate,
  drift-corrected, with ``--pad-s`` padding either side.
- ``align.jsonl``  one row per cut: the 6.8 schema's identity and alignment fields
  (``utt_id``, ``session_order_idx``, ``block_id``, ``channel``, ``fs_raw``, ``bits``,
  ``raw_file``, ``start_sample_raw``, ``end_sample_raw``, ``drift_ppm``, ``flags``) plus the
  transcript and the block's chirp QC. The remaining 6.8 fields (pool, geometry, IR,
  acoustics) are filled when the session log is merged.
- ``blocks_<take>.json``  per-block chirp onsets, PSRs and drift, and the take-level clock fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwse import audio  # noqa: E402
from uwse.bench import ingest  # noqa: E402
from uwse.bench.schema import RELEASE_OF_SPLIT, make_utt_id  # noqa: E402

CLIP_LEVEL = 0.999


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", type=Path, required=True, help="output dir of make_playlists/make_session_audio")
    ap.add_argument("--position", required=True, help="position id, e.g. p3")
    ap.add_argument("--rec", type=Path, required=True, help="recorded WAV of the whole session file")
    ap.add_argument("--session", type=int, required=True, help="session number in the utt_id (s<N>)")
    ap.add_argument("--channels", default="0", help="comma-separated recorded channels to cut (0-based)")
    ap.add_argument("--channel-ids", default=None,
                    help="comma-separated benchmark channel ids for --channels, used in utt_id and the "
                         "'channel' field (e.g. a second recorder's file channels 0,1 = hydrophones 2,3); "
                         "default: the file channel indices")
    ap.add_argument("--sync-channel", type=int, default=None,
                    help="channel for chirp detection (loopback if recorded); default: first of --channels")
    ap.add_argument("--anchor", action="store_true", help="the take is anchor_<pos>.wav")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad-s", type=float, default=0.25)
    ap.add_argument("--psr-db", type=float, default=6.0, help="chirp PSR threshold (6.5; frozen on v0-dev)")
    ap.add_argument("--max-drift-ppm", type=float, default=200.0)
    ap.add_argument("--search-s", type=float, default=0.5, help="chirp search half-window around its prediction")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    fs, blocks = ingest.load_session(a.bench, a.position, anchor=a.anchor)
    x, fs_raw, bits = ingest.load_take(a.rec, fs)
    chans = [int(c) for c in a.channels.split(",")]
    ids = chans if a.channel_ids is None else [int(c) for c in a.channel_ids.split(",")]
    if len(ids) != len(chans) or len(set(ids)) != len(ids):
        raise SystemExit("--channel-ids must give one distinct id per --channels entry")
    sync = chans[0] if a.sync_channel is None else a.sync_channel
    if max(chans + [sync]) >= x.shape[0]:
        raise SystemExit(f"{a.rec} has {x.shape[0]} channels")
    raw = x if fs_raw == fs else audio.load(str(a.rec), always_2d=True)[0]  # clip check on raw samples

    results, qc = ingest.align_take(x, fs, blocks, sync=sync, max_drift_ppm=a.max_drift_ppm,
                                    search_s=a.search_s, psr_threshold_db=a.psr_db, pad_s=a.pad_s)
    wav_dir = a.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    take = a.rec.stem
    rows = []
    to_raw = fs_raw / fs
    for res in tqdm(results, desc=f"cut {take}"):
        b = res.block
        cuts, bounds = ingest.cut_block(x[chans], fs, res, pad_s=a.pad_s)
        split = b.timing["bench_split"]
        for k, u in enumerate(b.timing["utterances"]):
            s_raw, e_raw = int(round(bounds[k, 0] * to_raw)), int(round(bounds[k, 1] * to_raw))
            for ci, ch in enumerate(chans):
                uid = make_utt_id(RELEASE_OF_SPLIT[split], a.session, a.position, b.block_id, ids[ci],
                                  u["item_id"])
                y = cuts[ci][k]
                audio.save(str(wav_dir / f"{uid}.wav"), y, fs, subtype="PCM_24")
                seg = raw[ch, max(0, s_raw): max(0, e_raw)]
                flags = list(res.flags)
                if seg.size and float(np.max(np.abs(seg))) >= CLIP_LEVEL:
                    flags.append("clip")
                if s_raw < 0 or e_raw > raw.shape[1]:
                    flags.append("dropout")
                rows.append({
                    "utt_id": uid, "librispeech_id": u["item_id"], "transcript": u.get("transcript"),
                    "bench_split": split, "session_order_idx": b.session_order_idx, "block_id": b.block_id,
                    "content_block": b.content_block, "position_id": a.position,
                    "thruster_state": b.thruster_state, "anchor_repeat": b.anchor_repeat,
                    "channel": ids[ci], "file_channel": ch, "sync_channel": sync, "fs_raw": fs_raw, "bits": bits,
                    "raw_file": str(a.rec), "start_sample_raw": s_raw, "end_sample_raw": e_raw,
                    "drift_ppm": res.drift_ppm, "drift_source": res.drift_source,
                    "chirp_psr_dB": min(c.psr_db for c in res.chirps), "pad_s": a.pad_s,
                    "wav": str(wav_dir / f"{uid}.wav"), "fs": fs, "flags": flags,
                })
    with open(a.out / "align.jsonl", "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    blocks_qc = [{"session_order_idx": r.block.session_order_idx, "content_block": r.block.content_block,
                  "chirp1": r.chirp1, "drift_ppm": r.drift_ppm, "drift_source": r.drift_source,
                  "flags": r.flags,
                  "chirps": [{"nominal": c.nominal, "found": c.found, "predicted": c.predicted,
                              "psr_dB": c.psr_db} for c in r.chirps]} for r in results]
    with open(a.out / f"blocks_{take}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"rec": str(a.rec), "position_id": a.position, "anchor": a.anchor, **qc,
                   "blocks": blocks_qc}, f, indent=1)
    n_flag = sum(bool(r.flags) for r in results)
    print(f"{take}: {len(results)} blocks, {len(rows)} cuts, take drift {qc['take_drift_ppm']:+.2f} ppm, "
          f"{qc['n_chirps_good']}/{qc['n_chirps']} chirps good, {n_flag} blocks flagged -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
