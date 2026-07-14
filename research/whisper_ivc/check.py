"""Print the RMS of all four hydrophone channels, one line per interval.

Output format: `ch0 ch1 ch2 ch3` with three decimal places, e.g.
    0.001 0.002 0.001 0.003

Requires both Daisy boards connected and running stream_audio.
Ctrl-C to quit.
"""
import threading
import time

import numpy as np

import daisy_stream

INTERVAL = 0.5  # seconds per printed line

# channel -> list of recent sample chunks, filled by reader threads
buffers = {ch: [] for ch in range(4)}
lock = threading.Lock()


def reader(serial_no, chans):
    port = daisy_stream.port_for_channel(chans[0])
    for frame in daisy_stream.frames(port):
        pcm = frame.astype(np.float32) / 32768.0
        with lock:
            buffers[chans[0]].append(pcm[:, 0])
            buffers[chans[1]].append(pcm[:, 1])


def main():
    for serial_no, chans in daisy_stream.BOARDS.items():
        threading.Thread(target=reader, args=(serial_no, chans),
                         daemon=True).start()

    while True:
        time.sleep(INTERVAL)
        with lock:
            rms = []
            for ch in range(4):
                if buffers[ch]:
                    x = np.concatenate(buffers[ch])
                    rms.append(float(np.sqrt(np.mean(x ** 2))))
                    buffers[ch].clear()
                else:
                    rms.append(0.0)
        print(" ".join(f"{v:.3f}" for v in rms), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
