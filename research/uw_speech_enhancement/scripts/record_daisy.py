"""Record one Daisy Seed board's two hydrophone channels to a WAV for one take (Linux).

Uses the ``stream_audio`` firmware reader in ``../whisper_ivc/daisy_stream.py``.
Start it before the take's file starts playing and stop it (Ctrl-C) after the
file ends; ``--seconds`` stops it automatically.

    python scripts/record_daisy.py --board 376636603433 \
        --out sessions/2026-12-02_s2/raw/daisyA/s2_p1_speech_daisyA.wav

Writes the WAV (2 channels, the board's rate, PCM_24) and ``<wav>.json`` with the
board, rate, sample count and the number of USB frame drops. A drop removes
samples, which shifts every later sample of that block; redo a take whose
sidecar shows ``dropped > 0`` if its QC in ``run_session.py check`` fails.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def record(batches, out: Path, rate_expected: int | None, seconds: float | None, stats: dict) -> dict:
    """Write ``(pcm (N, 2), rate, bits)`` batches to ``out`` until exhausted, ``seconds`` or Ctrl-C."""
    import soundfile as sf

    out.parent.mkdir(parents=True, exist_ok=True)
    f, rate, n, t0 = None, None, 0, time.monotonic()
    try:
        for pcm, r, _bits in batches:
            if f is None:
                if rate_expected and r != rate_expected:
                    raise SystemExit(f"board streams {r} Hz, expected {rate_expected}")
                rate = r
                f = sf.SoundFile(str(out), "w", samplerate=rate, channels=2, subtype="PCM_24")
            elif r != rate:
                raise SystemExit(f"rate changed mid-take ({rate} -> {r} Hz)")
            f.write(pcm)
            n += len(pcm)
            if seconds is not None and n >= seconds * rate:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if f is not None:
            f.close()
    return {"fs": rate, "n_samples": n, "seconds": n / rate if rate else 0.0,
            "dropped": int(stats.get("dropped", 0)), "wall_s": round(time.monotonic() - t0, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--board", default=None, help="board USB serial (see daisy_stream.BOARDS); default: auto")
    ap.add_argument("--port", default=None, help="serial port, overrides --board")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=int, default=96000)
    ap.add_argument("--bits", type=int, default=24, choices=(16, 24))
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT.parent / "whisper_ivc"))
    import glob

    import daisy_stream as ds

    if a.port:
        port = a.port
    elif a.board:
        hits = glob.glob(f"/dev/serial/by-id/*{a.board}*")
        if not hits:
            raise SystemExit(f"board {a.board} is not connected")
        port = hits[0]
    else:
        port = ds.find_port()
    stream = ds.Stream(port)
    stream.configure(rate=a.rate, bits=a.bits)
    time.sleep(0.5)  # let the firmware switch format before the first kept frame
    stats: dict = {}
    started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"recording {port} -> {a.out} (Ctrl-C to stop)", flush=True)
    try:
        info = record(stream.batches(stats), a.out, a.rate, a.seconds, stats)
    finally:
        stream.close()
    info.update({"board": a.board, "port": port, "bits": a.bits, "started": started})
    a.out.with_suffix(a.out.suffix + ".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))
    if info["dropped"]:
        print(f"WARNING: {info['dropped']} frame drops; check the take's QC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
