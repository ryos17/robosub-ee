"""Shared reader for the Daisy stream_audio firmware's USB PCM stream.

Wire format (little-endian, 132-byte frames):
    [0xDA][0x7A][seq u8][reserved u8][32 x (ch0 s16, ch1 s16)]
"""
import glob
import sys

import numpy as np
import serial

MAGIC = b"\xda\x7a"
SAMPLES_PER_FRAME = 32          # per channel
PAYLOAD_BYTES = SAMPLES_PER_FRAME * 2 * 2
FRAME_BYTES = 4 + PAYLOAD_BYTES
SAMPLE_RATE = 16000

# board USB serial -> the two hydrophone channels wired to its codec
BOARDS = {
    "376C36533433": (0, 1),  # ex-master board: hydrophones 0/1
    "376C36573433": (2, 3),  # ex-slave board: hydrophones 2/3
}


def port_for_channel(channel):
    """Serial port of the board that carries this hydrophone channel."""
    for serial_no, chans in BOARDS.items():
        if channel in chans:
            hits = glob.glob(f"/dev/serial/by-id/*{serial_no}*")
            if not hits:
                sys.exit(f"board {serial_no} (hydrophone {channel}) "
                         "is not connected")
            return hits[0]
    sys.exit(f"no board carries channel {channel}")


def _streams_frames(port, probe_s=1.5):
    """True if this port is emitting stream_audio's binary frames."""
    try:
        with serial.Serial(port, 115200, timeout=probe_s) as ser:
            return MAGIC in ser.read(2 * FRAME_BYTES + 4096)
    except (OSError, serial.SerialException):
        return False


def find_port():
    ports = glob.glob("/dev/serial/by-id/usb-Electrosmith_Daisy_Seed*")
    if len(ports) == 1:
        return ports[0]
    if not ports:
        sys.exit("no Daisy serial port found")
    # Multiple boards: pick the one running stream_audio (binary frames);
    # other firmware (e.g. master_level) prints text lines instead
    streaming = [p for p in ports if _streams_frames(p)]
    if len(streaming) == 1:
        print(f"auto-selected streaming board: {streaming[0]}")
        return streaming[0]
    sys.exit(f"{len(streaming)} of {len(ports)} Daisy ports are streaming "
             f"audio frames ({ports}) — use --port to choose")


def frames(port, stats=None):
    """Yield (SAMPLES_PER_FRAME, 2) int16 arrays from the serial stream.

    stats, if given, is a dict updated in place with 'samples' (per channel)
    and 'dropped' (sequence-number gaps).
    """
    ser = serial.Serial(port, 115200, timeout=1)
    buf = bytearray()
    last_seq = None
    while True:
        data = ser.read(4096)
        if not data:
            continue
        buf += data
        while True:
            start = buf.find(MAGIC)
            if start < 0:
                del buf[:-1]
                break
            if start:
                del buf[:start]
            if len(buf) < FRAME_BYTES:
                break
            frame = bytes(buf[:FRAME_BYTES])
            del buf[:FRAME_BYTES]
            seq = frame[2]
            if stats is not None:
                if last_seq is not None and seq != (last_seq + 1) % 256:
                    stats["dropped"] = stats.get("dropped", 0) + 1
                stats["samples"] = stats.get("samples", 0) + SAMPLES_PER_FRAME
            last_seq = seq
            pcm = np.frombuffer(frame, dtype="<i2", offset=4)
            yield pcm.reshape(SAMPLES_PER_FRAME, 2)
