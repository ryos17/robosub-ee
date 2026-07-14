"""Live Whisper transcription of hydrophone audio streamed from Daisy Seeds.

Each Daisy runs gen2/daisyseed_firmware/stream_audio.cpp, which sends 16 kHz
2-channel s16 PCM over USB CDC in 132-byte frames:
    [0xDA][0x7A][seq u8][reserved u8][32 x (ch0 s16, ch1 s16)]

The stream is cut into consecutive --window second chunks. Per chunk the
selected audio (--channel 0|1|2|3, or mix = all four channels averaged
across both boards) is normalized, optionally denoised (--denoise),
normalized again, saved under data/<year_date_time>/ as <segment>.wav, and
transcribed. The saved wav is byte-identical to what whisper receives.

Usage:
    python stream_transcribe.py                       # channel 0, raw
    python stream_transcribe.py --channel mix         # all 4 hydrophones
    python stream_transcribe.py --denoise metricgan   # neural enhancement
"""
import argparse
import os
import threading
import time
import wave

import numpy as np
import torch
import whisper
from tqdm import tqdm

import daisy_stream
from daisy_stream import SAMPLE_RATE

SB_CACHE = os.path.expanduser("~/.cache/speechbrain")


def normalize(x):
    return x / max(np.abs(x).max(), 1e-6) * 0.9


def make_denoiser(kind):
    """Return f(float32 mono) -> float32 mono speech-enhanced on the GPU."""
    if kind == "off":
        return lambda x: x
    if kind == "metricgan":
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        m = SpectralMaskEnhancement.from_hparams(
            "speechbrain/metricgan-plus-voicebank",
            savedir=os.path.join(SB_CACHE, "metricgan-plus-voicebank"),
            run_opts={"device": "cuda"})

        def run(x):
            wav = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).cuda()
            with torch.no_grad():
                out = m.enhance_batch(wav, lengths=torch.ones(1))
            return out.squeeze(0).cpu().numpy()
        return run
    if kind == "sepformer":
        from speechbrain.inference.separation import SepformerSeparation
        m = SepformerSeparation.from_hparams(
            "speechbrain/sepformer-wham16k-enhancement",
            savedir=os.path.join(SB_CACHE, "sepformer-wham16k-enhancement"),
            run_opts={"device": "cuda"})

        def run(x):
            wav = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).cuda()
            with torch.no_grad():
                out = m.separate_batch(wav)
            return out[:, :, 0].squeeze(0).cpu().numpy()
        return run
    raise ValueError(kind)


def write_segment(mono, seg_dir, n, pbar):
    """Save the final (whisper-ready) mono audio as <n>.wav."""
    path = os.path.join(seg_dir, f"{n}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((mono * 32767).astype("<i2").tobytes())
    pbar.write(f"[rec] {path} ({len(mono) / SAMPLE_RATE:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None,
                    help="Daisy serial port (single-board modes only)")
    ap.add_argument("--model", default="large-v3",
                    help="whisper model name")
    ap.add_argument("--window", type=float, default=10.0,
                    help="seconds per chunk (recorded and transcribed)")
    ap.add_argument("--channel", default="3",
                    choices=["0", "1", "2", "3", "mix"],
                    help="hydrophone channel to transcribe; mix averages "
                         "all four channels from both boards")
    ap.add_argument("--denoise", default="off",
                    choices=["off", "metricgan", "sepformer"],
                    help="neural speech enhancement before whisper "
                         "(default off = raw audio)")
    ap.add_argument("--record", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="save each chunk as a wav under data/")
    args = ap.parse_args()

    if args.port:
        ports = [args.port]
    elif args.channel == "mix":
        ports = [daisy_stream.port_for_channel(0),
                 daisy_stream.port_for_channel(2)]
    else:
        ports = [daisy_stream.port_for_channel(int(args.channel))]

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"loading whisper {args.model}...")
    model = whisper.load_model(args.model, device="cuda")
    print(f"loading denoiser {args.denoise}...")
    denoise = make_denoiser(args.denoise)
    print(f"listening on {', '.join(ports)} ({args.window:.0f}s chunks, "
          f"channel={args.channel}, denoise={args.denoise}, "
          f"record={'on' if args.record else 'off'})")

    seg_dir, seg_n = None, 0
    if args.record:
        seg_dir = os.path.join("data", time.strftime("%Y_%m%d_%H%M%S"))
        os.makedirs(seg_dir, exist_ok=True)
        print(f"recording to {seg_dir}/")

    # one reader thread per board; boards free-run on their own clocks, so
    # samples are aligned only approximately (fine for mixing/ASR)
    lock = threading.Lock()
    bufs = {i: [] for i in range(len(ports))}
    counts = {i: 0 for i in range(len(ports))}
    stats = {"samples": 0, "dropped": 0}

    def reader(idx, port):
        for frame in daisy_stream.frames(port, stats):
            pcm = frame.astype(np.float32) / 32768.0
            with lock:
                bufs[idx].append(pcm)
                counts[idx] += len(pcm)

    for i, p in enumerate(ports):
        threading.Thread(target=reader, args=(i, p), daemon=True).start()

    chunk_target = int(args.window * SAMPLE_RATE)
    pbar = tqdm(desc="audio received", unit="s", total=None)

    while True:
        time.sleep(0.2)
        with lock:
            avail = min(counts.values())
        pbar.n = round(stats["samples"] / SAMPLE_RATE / len(ports), 1)
        pbar.set_postfix(dropped=stats["dropped"])
        pbar.refresh()
        if avail < chunk_target:
            continue

        with lock:
            boards = []
            for i in range(len(ports)):
                cat = np.concatenate(bufs[i])
                boards.append(cat[:chunk_target])
                rest = cat[chunk_target:]
                bufs[i] = [rest] if len(rest) else []
                counts[i] = len(rest)

        if args.channel == "mix":
            # average all channels of all boards
            raw = np.concatenate(boards, axis=1).mean(axis=1)
        else:
            raw = boards[0][:, int(args.channel) % 2]

        t0 = time.time()
        mono = normalize(denoise(normalize(raw)))
        dt_dn = time.time() - t0

        if args.record:
            write_segment(mono, seg_dir, seg_n, pbar)
            seg_n += 1

        t0 = time.time()
        result = model.transcribe(mono, fp16=True, language="en",
                                  condition_on_previous_text=False)
        text = result["text"].strip()
        pbar.write(f"[{time.strftime('%H:%M:%S')} dn={dt_dn:.1f}s "
                   f"asr={time.time() - t0:.1f}s] {text}")


if __name__ == "__main__":
    main()
