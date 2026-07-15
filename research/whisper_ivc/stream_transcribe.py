"""Web UI for live Whisper transcription of the Daisy hydrophone streams.

Run:  python stream_transcribe.py   (then open http://<orin>:7860)

- Pick a whisper model from the dropdown and load it (nothing runs before).
- Two modes:
    manual     — Start/Stop button; the whole take is transcribed on Stop.
    continuous — audio is cut into window-length chunks, each transcribed.
- Every processed segment is saved to data/<year_date_time>/ as
  ch<C>_<N>.wav (C = the selected channel) or mix_<N>.wav for the channel
  mix, with its transcription in the matching .txt — byte-identical to what
  whisper heard (channel select -> resample to 16 kHz -> normalize ->
  optional denoise -> normalize).
- Each window's untouched audio is also written per channel, one mono file
  per hydrophone, to data/<session>/raw/ch<C>/<N>.wav at the NATIVE sample
  rate and bit depth (up to 96 kHz / 24-bit) — the highest-fidelity record,
  pinger band intact. N is the window index: manual mode records one
  fragment (0.wav); continuous mode records 0.wav, 1.wav, ... one per
  window (they concatenate back into the full take), matching the ch<C>_<N>
  whisper segments.
- Config (channel 0-3/mix, denoise, mode, window, rate, bits) is changeable
  in the UI; rate/bits are pushed live to the boards (not while recording).
- A log-scale spectrogram streams for every channel, one row each, up to
  the stream's Nyquist (48 kHz at the full rate: pings visible). The
  channel(s) whisper is listening to (the selected channel, or all on
  'mix') are outlined with a box.
"""
import glob
import json
import math
import os
import threading
import time
import wave

import numpy as np
import torch
import uvicorn
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from scipy.signal import resample_poly

import daisy_stream

WHISPER_RATE = 16000       # whisper's fixed input rate
SB_CACHE = os.path.expanduser("~/.cache/speechbrain")
SPECTRUM_N = 2048          # FFT window (Audacity default, Hann)
SPECTRUM_BINS = 256        # log-spaced display bins


def pcm_bytes(pcm, bits):
    """Float [-1,1] (N,2) -> little-endian PCM bytes at the stream's depth.

    Inverts daisy_stream's decode exactly, so the WAV holds the same ints
    the board sent."""
    if bits == 16:
        return np.round(pcm * 32768.0).clip(-32768, 32767) \
                 .astype("<i2").tobytes()
    v = np.round(pcm * 8388608.0).clip(-8388608, 8388607).astype("<i4")
    return np.frombuffer(v.tobytes(), np.uint8).reshape(-1, 4)[:, :3].tobytes()


def normalize(x):
    return x / max(np.abs(x).max(), 1e-6) * 0.9


# ----------------------------------------------------------------- audio in
class Boards:
    """Reader threads for every connected stream_audio board."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {ch: np.zeros(SPECTRUM_N, np.float32) for ch in range(4)}
        self.rec = {}          # board serial -> list of (N,2) arrays
        self.rec_counts = {}
        self.recording = False
        self.stats = {"samples": 0, "dropped": 0,
                      "rate": daisy_stream.DEFAULT_RATE,
                      "bits": daisy_stream.DEFAULT_BITS}
        self.connected = []
        self.streams = {}      # board serial -> daisy_stream.Stream
        self.raw_paths = []    # raw fragment WAVs written this session
        for serial_no, chans in daisy_stream.BOARDS.items():
            hits = glob.glob(f"/dev/serial/by-id/*{serial_no}*")
            if hits:
                self.connected.append((serial_no, chans))
                threading.Thread(target=self._reader,
                                 args=(serial_no, chans, hits[0]),
                                 daemon=True).start()

    def _reader(self, serial_no, chans, port):
        stream = daisy_stream.Stream(port)
        self.streams[serial_no] = stream
        try:
            # batches, not frames: per-frame Python work (3000 frames/s at
            # 96 kHz across both boards) starves the readers under GIL load
            # and overflows the kernel serial buffer -> host-side drops.
            for pcm, rate, bits in stream.batches(self.stats):
                with self.lock:
                    for i, ch in enumerate(chans):
                        self.latest[ch] = np.concatenate(
                            [self.latest[ch], pcm[:, i]])[-SPECTRUM_N:]
                    if self.recording:
                        # Just buffer; the raw WAVs are written per window in
                        # process() so the reader hot path does no disk I/O.
                        self.rec.setdefault(serial_no, []).append(pcm)
                        self.rec_counts[serial_no] = \
                            self.rec_counts.get(serial_no, 0) + len(pcm)
        finally:
            # a silently-dead reader looks like "0 drops": make it loud
            print(f"READER DIED: board {serial_no} ({port}) -- restart "
                  "stream_transcribe.py (did the board reboot/reflash?)",
                  flush=True)

    @property
    def rate(self):
        return self.stats["rate"]

    @property
    def bits(self):
        return self.stats["bits"]

    def configure(self, rate=None, bits=None):
        for stream in self.streams.values():
            stream.configure(rate=rate, bits=bits)

    def channels_available(self):
        return sorted(ch for _, chans in self.connected for ch in chans)

    def start_recording(self, session_dir):
        with self.lock:
            self.rec = {}
            self.rec_counts = {}
            self.raw_paths = []
            self.recording = True

    def stop_recording(self):
        """Stop recording; returns the raw fragment WAVs written so far."""
        with self.lock:
            self.recording = False
        return list(self.raw_paths)

    def save_raw(self, session_dir, chans, n):
        """Write one native-rate mono fragment per channel for window `n`:
        data/<session>/raw/ch<C>/<n>.wav. Manual mode has a single window
        (0.wav); continuous mode writes 0.wav, 1.wav, ... one per window,
        so the fragments concatenate back into the full recording."""
        rate, bits = self.rate, self.bits
        for ch in sorted(chans):
            ch_dir = os.path.join(session_dir, "raw", f"ch{ch}")
            os.makedirs(ch_dir, exist_ok=True)
            path = os.path.join(ch_dir, f"{n}.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(bits // 8)
                w.setframerate(rate)
                w.writeframes(pcm_bytes(chans[ch], bits))
            self.raw_paths.append(path)

    def take(self, n_samples=None):
        """Pop recorded audio, aligned across boards. Returns {ch: mono}."""
        with self.lock:
            if not self.rec_counts:
                return {}
            avail = min(self.rec_counts.values())
            take = avail if n_samples is None else min(n_samples, avail)
            if take <= 0:
                return {}
            out = {}
            for serial_no, chans in self.connected:
                if serial_no not in self.rec:
                    return {}
                cat = np.concatenate(self.rec[serial_no])
                for i, ch in enumerate(chans):
                    out[ch] = cat[:take, i]
                rest = cat[take:]
                self.rec[serial_no] = [rest] if len(rest) else []
                self.rec_counts[serial_no] = len(rest)
            return out

    def recorded_seconds(self):
        with self.lock:
            if not self.rec_counts:
                return 0.0
            return min(self.rec_counts.values()) / self.rate


# ----------------------------------------------------------------- pipeline
class Engine:
    def __init__(self):
        self.boards = Boards()
        self.model = None
        self.model_name = None
        self.busy = ""
        self.denoisers = {}
        self.cfg = {"channel": "0", "denoise": "off",
                    "mode": "manual", "window": 10.0}
        self.session_dir = None
        self.seg_n = 0
        self.events = []       # queued messages for the websocket
        self.ev_lock = threading.Lock()
        self._cont_thread = None

    # ---- messaging
    def emit(self, type_, **kw):
        print(f"[{time.strftime('%H:%M:%S')}] {type_}: {kw}", flush=True)
        with self.ev_lock:
            self.events.append({"type": type_, **kw})

    def drain(self):
        with self.ev_lock:
            ev, self.events = self.events, []
        return ev

    # ---- model / denoiser
    def load_model(self, name):
        self.busy = f"loading {name}..."
        try:
            if self.model is not None:
                del self.model
                self.model = None
                torch.cuda.empty_cache()
            self.model = whisper.load_model(name, device="cuda")
            self.model_name = name
            self.emit("log", text=f"model {name} loaded "
                      f"({torch.cuda.memory_allocated() / 1e9:.1f} GB)")
        except Exception as e:
            self.emit("log", text=f"model load failed: {e}")
        finally:
            self.busy = ""

    def unload_model(self):
        import gc
        if self.boards.recording:
            self.emit("log", text="stop recording before clearing the model")
            return
        self.model = None
        self.model_name = None
        self.denoisers = {}
        gc.collect()
        torch.cuda.empty_cache()
        self.emit("log", text=f"GPU cleared "
                  f"({torch.cuda.memory_allocated() / 1e9:.2f} GB still "
                  f"allocated)")

    def get_denoiser(self, kind):
        if kind == "off":
            return lambda x: x
        if kind in self.denoisers:
            return self.denoisers[kind]
        self.busy = f"loading denoiser {kind}..."
        try:
            if kind == "metricgan":
                from speechbrain.inference.enhancement import \
                    SpectralMaskEnhancement
                m = SpectralMaskEnhancement.from_hparams(
                    "speechbrain/metricgan-plus-voicebank",
                    savedir=os.path.join(SB_CACHE, "metricgan-plus-voicebank"),
                    run_opts={"device": "cuda"})

                def run(x):
                    wav = torch.from_numpy(
                        np.ascontiguousarray(x)).unsqueeze(0).cuda()
                    with torch.no_grad():
                        return m.enhance_batch(
                            wav, lengths=torch.ones(1)).squeeze(0).cpu().numpy()
            else:  # sepformer
                from speechbrain.inference.separation import \
                    SepformerSeparation
                m = SepformerSeparation.from_hparams(
                    "speechbrain/sepformer-wham16k-enhancement",
                    savedir=os.path.join(SB_CACHE,
                                         "sepformer-wham16k-enhancement"),
                    run_opts={"device": "cuda"})

                def run(x):
                    wav = torch.from_numpy(
                        np.ascontiguousarray(x)).unsqueeze(0).cuda()
                    with torch.no_grad():
                        return m.separate_batch(
                            wav)[:, :, 0].squeeze(0).cpu().numpy()
            self.denoisers[kind] = run
            return run
        finally:
            self.busy = ""

    # ---- record / process
    def select_mono(self, chans):
        ch = self.cfg["channel"]
        if ch == "mix":
            return np.mean(list(chans.values()), axis=0)
        return chans.get(int(ch))

    def process(self, chans):
        """one window: save raw fragments -> channel select -> 16 kHz ->
        normalize -> denoise -> save whisper segment + asr."""
        n = self.seg_n
        self.seg_n += 1

        # Native-rate raw fragment(s) for this window, all channels:
        # raw/ch<C>/<n>.wav (n == the whisper segment index below).
        self.boards.save_raw(self.session_dir, chans, n)

        raw = self.select_mono(chans)
        rate = self.boards.rate
        if raw is None or len(raw) < rate // 2:
            self.emit("log", text="segment too short / channel missing")
            return
        if rate != WHISPER_RATE:
            g = math.gcd(rate, WHISPER_RATE)
            raw = resample_poly(raw, WHISPER_RATE // g, rate // g)
        raw = raw.astype(np.float32)
        mono = normalize(self.get_denoiser(self.cfg["denoise"])(normalize(raw)))

        ch = self.cfg["channel"]
        stem = "mix" if ch == "mix" else f"ch{int(ch)}"
        base = f"{stem}_{n}"
        wav_path = os.path.join(self.session_dir, base + ".wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(WHISPER_RATE)
            w.writeframes((mono * 32767).astype("<i2").tobytes())

        self.busy = f"transcribing {base}.wav..."
        t0 = time.time()
        try:
            result = self.model.transcribe(mono, fp16=True, language="en",
                                           condition_on_previous_text=False)
            text = result["text"].strip()
        finally:
            self.busy = ""
        with open(os.path.join(self.session_dir, base + ".txt"), "w") as f:
            f.write(text + "\n")
        self.emit("transcript", n=n, name=base,
                  secs=round(len(mono) / WHISPER_RATE, 1),
                  asr=round(time.time() - t0, 1), text=text,
                  wav=wav_path)

    def start(self):
        if self.model is None:
            self.emit("log", text="load a model first")
            return
        self.session_dir = os.path.join("data",
                                        time.strftime("%Y_%m%d_%H%M%S"))
        os.makedirs(self.session_dir, exist_ok=True)
        self.seg_n = 0
        self.boards.start_recording(self.session_dir)
        self.emit("log", text=f"recording -> {self.session_dir}/ "
                  f"(mode={self.cfg['mode']}, "
                  f"{self.boards.rate} Hz / {self.boards.bits}-bit raw)")
        if self.cfg["mode"] == "continuous":
            self._cont_thread = threading.Thread(target=self._cont_loop,
                                                 daemon=True)
            self._cont_thread.start()

    def _cont_loop(self):
        rate = self.boards.rate
        target = int(float(self.cfg["window"]) * rate)
        while self.boards.recording:
            if self.boards.recorded_seconds() * rate >= target:
                chans = self.boards.take(target)
                if chans:
                    self.process(chans)
            else:
                time.sleep(0.2)

    def stop(self):
        was_continuous = self.cfg["mode"] == "continuous"
        raw_paths = self.boards.stop_recording()
        if self._cont_thread:
            self._cont_thread.join(timeout=30)
            self._cont_thread = None
        chans = self.boards.take()
        if chans and (not was_continuous
                      or len(next(iter(chans.values()))) >= self.boards.rate):
            self.process(chans)
        raw_names = ", ".join(os.path.relpath(p, self.session_dir)
                              for p in raw_paths) or "none"
        self.emit("log", text=f"stopped; saved to {self.session_dir}/ "
                  f"(raw: {raw_names})")

    # ---- spectrum
    def _bins_for(self, x, rate, denoise):
        # optionally visualize what whisper would hear: run the active
        # denoiser (only if already loaded; never trigger a load here)
        if denoise:
            kind = self.cfg["denoise"]
            if kind != "off" and kind in self.denoisers:
                try:
                    x = self.denoisers[kind](normalize(x))
                except Exception:
                    pass
        win = np.hanning(len(x))
        # scale to dBFS: full-scale sine -> 0 dB (hann coherent gain 0.5)
        mag = np.abs(np.fft.rfft(x * win)) / (len(x) / 4)
        freqs = np.fft.rfftfreq(len(x), 1 / rate)
        edges = np.geomspace(50, rate / 2, SPECTRUM_BINS + 1)
        bins = []
        for i in range(SPECTRUM_BINS):
            m = (freqs >= edges[i]) & (freqs < edges[i + 1])
            if m.any():
                v = mag[m].max()
            else:
                # display band narrower than one FFT bin (low end of the
                # log axis at high rates): interpolate, not a black stripe
                v = np.interp(np.sqrt(edges[i] * edges[i + 1]), freqs, mag)
            bins.append(round(20 * np.log10(v + 1e-9), 1))
        return bins

    def spectrum(self):
        """Per-channel log spectra + which channel(s) whisper is listening to
        (the selected channel, or all when 'mix'). The UI draws one row per
        channel and boxes the listened rows; those rows get the denoiser
        preview so they show what whisper actually hears."""
        with self.boards.lock:
            rate = self.boards.rate
            avail = self.boards.channels_available()
            latest = {c: self.boards.latest[c].copy() for c in avail}
        sel = self.cfg["channel"]
        if sel == "mix":
            listening = list(avail)
        else:
            listening = [int(sel)] if int(sel) in avail else []
        return {"rate": rate, "channels": avail, "listening": listening,
                "bins": {c: self._bins_for(latest[c], rate, c in listening)
                         for c in avail}}


ENGINE = Engine()
app = FastAPI()

HTML = """<!doctype html>
<html><head><title>Stanford Robosub Inter Vehicle Communication</title><style>
body{font-family:system-ui;background:#fff;color:#222;margin:0;padding:16px}
h2{margin:0 0 12px;color:#8c1515}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
select,input,button{background:#f6f6f6;color:#222;border:1px solid #ccc;
  border-radius:6px;padding:6px 10px;font-size:14px}
button{cursor:pointer}
#rec{background:#2c8c2c;color:#fff;min-width:110px}
#rec.on{background:#c33;color:#fff}
#status{color:#567;font-size:13px}
.layout{display:flex;gap:16px;align-items:flex-start}
.left{flex:1;min-width:0}
.right{width:420px;flex-shrink:0}
.right .row{margin-bottom:8px}
canvas{background:#fafafa;border:1px solid #ddd;border-radius:6px;display:block}
#transcript{background:#f4faf4;border:1px solid #bcd8bc;border-radius:6px;
  padding:14px;min-height:150px;max-height:340px;overflow-y:auto;
  font-size:16px;line-height:1.5;margin-top:10px;white-space:pre-wrap}
#transcript .seg{color:#7a9;font-size:12px;margin-right:8px}
#log{background:#f7f7f7;border:1px solid #ddd;border-radius:6px;padding:10px;
  height:200px;overflow-y:auto;font-family:monospace;font-size:13px;
  white-space:pre-wrap;margin-top:10px;color:#333}
.t{color:#282}.m{color:#679}
</style></head><body>
<h2>Stanford Robosub Inter Vehicle Communication</h2>
<div class=layout>
<div class=left>
<canvas id=spec width=1200 height=520></canvas>
</div>
<div class=right>
<div class=row>
 <label>model <select id=model></select></label>
 <button id=load>load model</button>
 <button id=unload>clear model</button>
 <span id=mstat class=m>no model loaded</span>
</div>
<div class=row>
 <label>mode <select id=mode>
   <option value=manual>manual (start/stop)</option>
   <option value=continuous>continuous (windowed)</option></select></label>
 <label>window(s) <input id=window type=number value=10 min=2 max=60
   style="width:60px"></label>
 <label>channel <select id=channel></select></label>
 <label>denoise <select id=denoise>
   <option>off</option><option>metricgan</option><option>sepformer</option>
 </select></label>
 <label>rate <select id=rate>
   <option>96000</option><option>48000</option><option>32000</option>
   <option>24000</option><option>16000</option></select></label>
 <label>bits <select id=bits>
   <option>16</option><option selected>24</option></select></label>
</div>
<div class=row>
 <button id=rec>● start</button>
 <button id=clear>clear transcription</button>
 <label>min dB <input id=mindb type=number value=-100 step=5
   style="width:64px"></label>
 <label>max dB <input id=maxdb type=number value=-60 step=5
   style="width:64px"></label>
 <span id=status></span>
</div>
<div id=transcript></div>
<div id=log></div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
let recording=false;
async function api(p,body){const r=await fetch(p,{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body||{})});return r.json();}
function cfg(){api('/api/config',{mode:$('mode').value,
  window:parseFloat($('window').value),channel:$('channel').value,
  denoise:$('denoise').value,rate:parseInt($('rate').value),
  bits:parseInt($('bits').value)});}
['mode','window','channel','denoise','rate','bits'].forEach(
  id=>$(id).onchange=cfg);
$('load').onclick=()=>api('/api/load_model',{name:$('model').value});
$('unload').onclick=()=>api('/api/unload_model');
$('rec').onclick=()=>{recording?api('/api/stop'):api('/api/start');};
$('clear').onclick=()=>{$('transcript').innerHTML='';};
function logLine(html){const d=$('log');d.innerHTML+=html+'\\n';
  d.scrollTop=d.scrollHeight;}
const ctx=$('spec').getContext('2d');
const F_LO=50,ML=52,MT=10,MB=24,MR=64,COLW=8,ROWGAP=16,NROWS=4,RIGHTW=420;
let F_HI=48000,DB_LO=-100,DB_HI=-60;
const specEl=$('spec');
// fill the left column; crisp (backing store == on-screen pixels), rows
// sized so the four spectrograms span the viewport height
specEl.width=Math.max(560,window.innerWidth-RIGHTW-48);
const availH=Math.max(360,
  window.innerHeight-specEl.getBoundingClientRect().top-16);
const ROWH=Math.max(70,
  Math.floor((availH-MT-MB-(NROWS-1)*ROWGAP)/NROWS));
specEl.height=MT+NROWS*ROWH+(NROWS-1)*ROWGAP+MB;
const PW=specEl.width-ML-MR,CW=specEl.width,CH=specEl.height;
const rowY=i=>MT+i*(ROWH+ROWGAP);
// re-fit on window resize (debounced; simplest correct path is a reload)
let _rt;window.addEventListener('resize',()=>{
  clearTimeout(_rt);_rt=setTimeout(()=>location.reload(),400);});
// one scrolling offscreen strip per channel row
const plots=[],pctxs=[],prev=[];
for(let i=0;i<NROWS;i++){const p=document.createElement('canvas');
  p.width=PW;p.height=ROWH;const c=p.getContext('2d');
  c.fillStyle='#14041f';c.fillRect(0,0,PW,ROWH);
  plots.push(p);pctxs.push(c);prev.push(null);}
let colCv=null,colCtx=null;
// Audacity-style colormap: dark purple -> magenta -> orange -> white
function cmapRGB(f){f=Math.max(0,Math.min(1,f));
  const s=[[0,20,4,45],[0.35,120,20,130],[0.62,225,80,45],
           [0.85,255,190,80],[1,255,255,255]];
  for(let i=1;i<s.length;i++)if(f<=s[i][0]){
    const t=(f-s[i-1][0])/(s[i][0]-s[i-1][0]);
    return [s[i-1][1]+t*(s[i][1]-s[i-1][1])|0,
            s[i-1][2]+t*(s[i][2]-s[i-1][2])|0,
            s[i-1][3]+t*(s[i][3]-s[i-1][3])|0];}
  return [255,255,255];}
function cmap(f){const c=cmapRGB(f);
  return `rgb(${c[0]},${c[1]},${c[2]})`;}
// push one new time-column into channel i's scrolling strip
function pushColumn(i,bins){
  // temporal smoothing then vertical interpolation for an Audacity-like look
  if(prev[i]&&prev[i].length===bins.length)
    bins=bins.map((v,k)=>0.6*v+0.4*prev[i][k]);
  prev[i]=bins;
  if(!colCv){colCv=document.createElement('canvas');colCv.width=1;
    colCv.height=bins.length;colCtx=colCv.getContext('2d');}
  const img=colCtx.createImageData(1,bins.length);
  bins.forEach((v,k)=>{const c=cmapRGB((v-DB_LO)/(DB_HI-DB_LO));
    const o=(bins.length-1-k)*4;
    img.data[o]=c[0];img.data[o+1]=c[1];img.data[o+2]=c[2];img.data[o+3]=255;});
  colCtx.putImageData(img,0,0);
  const pc=pctxs[i];pc.drawImage(plots[i],-COLW,0);
  pc.imageSmoothingEnabled=true;
  pc.drawImage(colCv,PW-COLW-2,0,COLW+2,ROWH);}
// full-frame redraw: four channel rows + a box around the listened one(s)
let lastMsg=null;
function render(msg){lastMsg=msg;
  const bins=(msg&&msg.bins)||{},listen=(msg&&msg.listening)||[];
  ctx.fillStyle='#fafafa';ctx.fillRect(0,0,CW,CH);
  const ticks=[100,1000,10000,20000,48000].filter(f=>f<=F_HI);
  for(let i=0;i<NROWS;i++){const y0=rowY(i);
    if(bins[i]!==undefined)ctx.drawImage(plots[i],ML,y0);
    else{ctx.fillStyle='#ededed';ctx.fillRect(ML,y0,PW,ROWH);
      ctx.fillStyle='#b0b0b0';ctx.font='12px system-ui';ctx.textAlign='center';
      ctx.fillText('no board',ML+PW/2,y0+ROWH/2+4);}
    ctx.font='10px system-ui';ctx.fillStyle='#888';ctx.textAlign='right';
    ticks.forEach(f=>{const y=y0+ROWH*(1-Math.log(f/F_LO)/Math.log(F_HI/F_LO));
      ctx.fillText(f>=1000?(f/1000)+'k':f,ML-5,y+3);});
    const boxed=listen.indexOf(i)>=0;
    ctx.textAlign='left';
    ctx.font=boxed?'bold 13px system-ui':'12px system-ui';
    ctx.fillStyle=boxed?'#8c1515':'#555';
    ctx.fillText('ch'+i,6,y0+15);
    if(boxed){ctx.strokeStyle='#8c1515';ctx.lineWidth=3;
      ctx.strokeRect(ML-2,y0-2,PW+4,ROWH+4);}}
  ctx.fillStyle='#666';ctx.font='11px system-ui';ctx.textAlign='center';
  ctx.fillText('time \\u2192',ML+PW/2,CH-7);
  // shared dB colorbar spanning all rows
  const bx=CW-MR+18,bw=14,cbH=CH-MT-MB;
  for(let y=0;y<cbH;y++){ctx.fillStyle=cmap(1-y/cbH);ctx.fillRect(bx,MT+y,bw,1);}
  ctx.fillStyle='#666';ctx.textAlign='left';ctx.font='11px system-ui';
  const step=(DB_HI-DB_LO)/4;
  for(let k=0;k<=4;k++){const db=DB_HI-k*step;
    const y=MT+cbH*(1-(db-DB_LO)/(DB_HI-DB_LO));
    ctx.fillText(Math.round(db),bx+bw+3,y+4);}
  ctx.fillText('dB',bx+bw+3,MT+cbH+14);}
function dbCfg(){const lo=parseFloat($('mindb').value),
  hi=parseFloat($('maxdb').value);
  if(isFinite(lo)&&isFinite(hi)&&hi>lo){DB_LO=lo;DB_HI=hi;render(lastMsg);}}
$('mindb').onchange=dbCfg;$('maxdb').onchange=dbCfg;
function drawSpec(msg){const bins=msg.bins||{};
  for(let i=0;i<NROWS;i++)if(bins[i]!==undefined)pushColumn(i,bins[i]);
  render(msg);}
render(null);
function connect(){const ws=new WebSocket(`ws://${location.host}/ws`);
 ws.onmessage=e=>{const m=JSON.parse(e.data);
  if(m.type==='spectrum')drawSpec(m);
  else if(m.type==='transcript'){
    const tr=$('transcript');
    tr.innerHTML+=`<div><span class=seg>#${m.n}</span>${m.text||'&mdash;'}</div>`;
    tr.scrollTop=tr.scrollHeight;
    logLine(`<span class=t>[${m.name}.wav ${m.secs}s asr=${m.asr}s]</span> ${m.text}`);}
  else if(m.type==='log')logLine(`<span class=m># ${m.text}</span>`);
  else if(m.type==='status'){recording=m.recording;
    if(m.rate/2!==F_HI){F_HI=m.rate/2;render(lastMsg);}
    $('rec').textContent=recording?'■ stop':'● start';
    $('rec').className=recording?'on':'';
    $('mstat').textContent=m.model?`model: ${m.model}`:'no model loaded';
    $('status').textContent=(m.busy?m.busy+' ':'')+
      (recording?`recording ${m.recorded}s | `:'')+
      `${m.rate/1000} kHz / ${m.bits}-bit`+
      (m.dropped?` | dropped ${m.dropped}`:'')+
      (m.channels.length?'':' | NO BOARDS');}};
 ws.onclose=()=>setTimeout(connect,1500);}
fetch('/api/init').then(r=>r.json()).then(d=>{
  d.models.forEach(m=>{const o=document.createElement('option');
    o.textContent=m;if(m===d.default)o.selected=true;
    $('model').appendChild(o);});
  d.channels.concat(['mix']).forEach(c=>{const o=document.createElement('option');
    o.textContent=c;$('channel').appendChild(o);});
  connect();});
</script></body></html>"""


@app.get("/")
def index():
    return HTMLResponse(HTML)


@app.post("/api/init")
@app.get("/api/init")
def init():
    return {"models": whisper.available_models(),
            "default": "large-v3",
            "channels": ENGINE.boards.channels_available()}


@app.post("/api/config")
async def config(body: dict):
    ENGINE.cfg.update({k: v for k, v in body.items()
                       if k in ("channel", "denoise", "mode", "window")})
    # rate/bits go straight to the boards (frozen while recording so the
    # raw WAV headers stay truthful)
    rate, bits = body.get("rate"), body.get("bits")
    if rate is not None or bits is not None:
        if ENGINE.boards.recording:
            ENGINE.emit("log", text="stop recording to change rate/bits")
        else:
            want_rate = None if rate == ENGINE.boards.rate else rate
            want_bits = None if bits == ENGINE.boards.bits else bits
            if want_rate or want_bits:
                try:
                    ENGINE.boards.configure(rate=want_rate, bits=want_bits)
                    ENGINE.emit("log", text=f"boards -> rate={rate} bits={bits}")
                except ValueError as e:
                    ENGINE.emit("log", text=f"bad rate/bits: {e}")
    ENGINE.emit("log", text=f"config: {ENGINE.cfg}")
    # preload the denoiser so the spectrogram can show enhanced audio
    kind = ENGINE.cfg["denoise"]
    if kind != "off" and kind not in ENGINE.denoisers:
        threading.Thread(target=ENGINE.get_denoiser, args=(kind,),
                         daemon=True).start()
    return {"ok": True}


@app.post("/api/load_model")
async def load_model(body: dict):
    threading.Thread(target=ENGINE.load_model, args=(body["name"],),
                     daemon=True).start()
    return {"ok": True}


@app.post("/api/unload_model")
async def unload_model():
    threading.Thread(target=ENGINE.unload_model, daemon=True).start()
    return {"ok": True}


@app.post("/api/start")
async def start():
    threading.Thread(target=ENGINE.start, daemon=True).start()
    return {"ok": True}


@app.post("/api/stop")
async def stop():
    threading.Thread(target=ENGINE.stop, daemon=True).start()
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    import asyncio
    try:
        while True:
            for ev in ENGINE.drain():
                await sock.send_text(json.dumps(ev))
            spec = await asyncio.to_thread(ENGINE.spectrum)
            await sock.send_text(json.dumps({"type": "spectrum", **spec}))
            await sock.send_text(json.dumps(
                {"type": "status",
                 "recording": ENGINE.boards.recording,
                 "model": ENGINE.model_name,
                 "busy": ENGINE.busy,
                 "recorded": round(ENGINE.boards.recorded_seconds(), 1),
                 "dropped": ENGINE.boards.stats["dropped"],
                 "rate": ENGINE.boards.rate,
                 "bits": ENGINE.boards.bits,
                 "channels": ENGINE.boards.channels_available()}))
            await asyncio.sleep(0.12)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"boards: {ENGINE.boards.connected}")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
