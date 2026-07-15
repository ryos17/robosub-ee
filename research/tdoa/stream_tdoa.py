"""Web UI for a live 4-hydrophone frequency-amplitude scope.

Run:  python stream_tdoa.py   (then open http://<orin>:7861)

The tdoa/pinger sibling of ../whisper_ivc/stream_transcribe.py. Instead of
transcribing, it streams — for every hydrophone — the amplitude at a single
frequency over time, so you can watch a pinger's band light up on all four
channels at once.

Faithful to gen2/daisyseed_firmware/master_level.cpp: each channel's
amplitude is exactly what the firmware logs,

    fftLibrary.getFrequencyMagnitude(buffer, kFftSize,
                                     targetFrequency, frequencyTolerance)

computed with the line-faithful fft_library.py port (recursive radix-2
Cooley-Tukey, Hanning window, float32, size_t bin truncation). The board
already bakes kGain = 100 into every sample, so — like the other tdoa
ports — the ×100 is NOT reapplied here.

Layout:
- LEFT: four scrolling line plots, one per hydrophone (ch0..ch3). X axis is
  time (last `window` seconds), Y axis is the amplitude at the selected
  frequency, updated ~10x/s from the live board stream.
- RIGHT: the FFT panel. The three master_level.cpp knobs are live:
    frequency  — targetFrequency (any Hz up to Nyquist)
    fft size   — kFftSize (the FFT/block length, power of two)
    tolerance  — frequencyTolerance; the panel shows the resulting bin
                 window [lower_bin, upper_bin] that getFrequencyMagnitude
                 sums (i.e. exactly which FFT bins the amplitude is over).
  Sample rate / bit depth are pushed live to the boards (as in
  stream_transcribe), plus client-side y-scale and time-window controls.
"""
import json
import os
import sys
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# daisy_stream lives with the whisper_ivc demo; reuse it verbatim (same
# path trick audio_source.py uses).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "whisper_ivc"))
import daisy_stream  # noqa: E402

from fft_library import FFTLibrary  # noqa: E402

FFT_SIZES = (32, 64, 128, 256, 512, 1024, 2048)
MAX_FFT = FFT_SIZES[-1]             # rolling buffer must hold the largest FFT


def frequency_magnitude(fftlib, buf, n_fft, target_freq, tolerance):
    """(amplitude, (lower_bin, upper_bin), peak_freq) for one channel.

    Inlines master_level.cpp's getFrequencyMagnitude so a single FFT feeds
    both the bin-window magnitude sum AND the peak-frequency readout — every
    step (Hanning, fft, size_t bin truncation, inclusive |z| sum) is the
    exact fft_library.py port, so the amplitude equals what the firmware
    logs. `buf` is the most recent n_fft samples of a channel.
    """
    sig = np.zeros(n_fft, np.complex64)
    n = min(len(buf), n_fft)
    sig[:n] = np.asarray(buf[:n], dtype=np.float32)
    sig = fftlib.applyHanningWindow(sig)
    sig = fftlib.fft(sig)

    # getFrequencyMagnitude bin bounds (float32 math, size_t truncation)
    lower_freq = np.float32(target_freq) * np.float32(1.0 - tolerance)
    upper_freq = np.float32(target_freq) * np.float32(1.0 + tolerance)
    lower_bin = int(np.float32(lower_freq * n_fft) / fftlib.m_sampleRate)
    upper_bin = int(np.float32(upper_freq * n_fft) / fftlib.m_sampleRate)
    half = n_fft // 2
    if lower_bin >= half:
        lower_bin = half - 1
    if upper_bin >= half:
        upper_bin = half - 1

    mags = np.abs(sig[lower_bin:upper_bin + 1]).astype(np.float32)
    amp = float(np.float32(mags.sum(dtype=np.float32)))
    peak = fftlib.findInterpolatedFrequency(sig, float(fftlib.m_sampleRate))
    return amp, (lower_bin, upper_bin), peak


# ----------------------------------------------------------------- audio in
class Boards:
    """Reader threads for every connected stream_audio board.

    Keeps only a rolling window of the most recent MAX_FFT samples per
    hydrophone — enough to slice the last kFftSize for any selected FFT
    size. No recording; this is a live scope.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {ch: np.zeros(MAX_FFT, np.float32) for ch in range(4)}
        self.stats = {"samples": 0, "dropped": 0,
                      "rate": daisy_stream.DEFAULT_RATE,
                      "bits": daisy_stream.DEFAULT_BITS}
        self.connected = []
        self.streams = {}          # board serial -> daisy_stream.Stream
        import glob
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
            # batches, not frames: per-frame Python work starves the readers
            # under GIL load and overflows the kernel serial buffer.
            for pcm, rate, bits in stream.batches(self.stats):
                with self.lock:
                    for i, ch in enumerate(chans):
                        self.latest[ch] = np.concatenate(
                            [self.latest[ch], pcm[:, i]])[-MAX_FFT:]
        finally:
            print(f"READER DIED: board {serial_no} ({port}) -- restart "
                  "stream_tdoa.py (did the board reboot/reflash?)", flush=True)

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


# ----------------------------------------------------------------- pipeline
class Engine:
    def __init__(self):
        self.boards = Boards()
        # master_level.cpp's active ("Testing") values: 25 kHz, 64-pt, 0.01
        self.cfg = {"freq": 25000.0, "fft": 64, "tol": 0.01}
        self._fftlib = None
        self._fftlib_rate = None
        self.events = []
        self.ev_lock = threading.Lock()

    def emit(self, type_, **kw):
        print(f"[{time.strftime('%H:%M:%S')}] {type_}: {kw}", flush=True)
        with self.ev_lock:
            self.events.append({"type": type_, **kw})

    def drain(self):
        with self.ev_lock:
            ev, self.events = self.events, []
        return ev

    def fftlib(self, rate):
        """FFTLibrary bound to the current sample rate (rebuilt on change)."""
        if self._fftlib is None or self._fftlib_rate != rate:
            self._fftlib = FFTLibrary(rate)
            self._fftlib_rate = rate
        return self._fftlib

    def scope(self):
        """Per-channel amplitude at the selected frequency + the bin window
        getFrequencyMagnitude sums and each channel's peak frequency."""
        with self.boards.lock:
            rate = self.boards.rate
            avail = self.boards.channels_available()
            latest = {c: self.boards.latest[c].copy() for c in avail}
        n_fft = int(self.cfg["fft"])
        freq = float(self.cfg["freq"])
        tol = float(self.cfg["tol"])
        fftlib = self.fftlib(rate)

        amp, peak, bins = {}, {}, None
        for c in avail:
            a, b, p = frequency_magnitude(fftlib, latest[c][-n_fft:], n_fft,
                                          freq, tol)
            amp[c] = round(a, 5)
            peak[c] = round(p, 1)
            bins = b            # identical for every channel at one rate
        return {"rate": rate, "channels": avail, "amp": amp, "peak": peak,
                "fft": n_fft, "freq": freq, "tol": tol,
                "bins": list(bins) if bins else None,
                "bin_hz": round(rate / n_fft, 2), "t": time.time()}


ENGINE = Engine()
app = FastAPI()

HTML = """<!doctype html>
<html><head><title>Stanford Robosub TDOA Frequency Scope</title><style>
body{font-family:system-ui;background:#fff;color:#222;margin:0;padding:16px}
h2{margin:0 0 12px;color:#8c1515}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
select,input,button{background:#f6f6f6;color:#222;border:1px solid #ccc;
  border-radius:6px;padding:6px 10px;font-size:14px}
button{cursor:pointer}
#status{color:#567;font-size:13px}
.layout{display:flex;gap:16px;align-items:flex-start}
.left{flex:1;min-width:0}
.right{width:400px;flex-shrink:0}
.right .row{margin-bottom:8px}
canvas{background:#fafafa;border:1px solid #ddd;border-radius:6px;display:block}
.panel{background:#f7f7f7;border:1px solid #ddd;border-radius:6px;padding:12px;
  margin-top:10px;font-size:13px}
.panel b{color:#8c1515}
#binfo{font-family:monospace;color:#333;margin-top:6px}
#log{background:#f7f7f7;border:1px solid #ddd;border-radius:6px;padding:10px;
  height:150px;overflow-y:auto;font-family:monospace;font-size:12px;
  white-space:pre-wrap;margin-top:10px;color:#333}
.m{color:#679}
label{font-size:13px}
</style></head><body>
<h2>Stanford Robosub TDOA Frequency Scope</h2>
<div class=layout>
<div class=left>
<canvas id=scope width=1200 height=560></canvas>
</div>
<div class=right>
<div class=row><b>FFT &mdash; master_level.cpp</b></div>
<div class=row>
 <label>frequency (Hz) <input id=freq type=number value=25000 step=500
   style="width:90px"></label>
 <label>fft size <select id=fft>
   <option>32</option><option selected>64</option><option>128</option>
   <option>256</option><option>512</option><option>1024</option>
   <option>2048</option></select></label>
</div>
<div class=row>
 <label>tolerance <input id=tol type=number value=0.01 step=0.005 min=0
   style="width:80px"></label>
 <span id=binfo></span>
</div>
<div class=row>
 <label>rate <select id=rate>
   <option>96000</option><option>48000</option><option>32000</option>
   <option>24000</option><option>16000</option></select></label>
 <label>bits <select id=bits>
   <option>16</option><option selected>24</option></select></label>
</div>
<div class=row>
 <label>window (s) <input id=window type=number value=15 min=2 max=120
   style="width:60px"></label>
 <label>y-max (0=auto) <input id=ymax type=number value=0 step=0.5 min=0
   style="width:70px"></label>
</div>
<div class=row><span id=status></span></div>
<div class=panel>
 <b>live levels</b> &mdash; amplitude at the selected frequency
 <div id=levels></div>
</div>
<div id=log></div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
async function api(p,body){const r=await fetch(p,{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body||{})});return r.json();}
function cfg(){api('/api/config',{freq:parseFloat($('freq').value),
  fft:parseInt($('fft').value),tol:parseFloat($('tol').value),
  rate:parseInt($('rate').value),bits:parseInt($('bits').value)});}
['freq','fft','tol','rate','bits'].forEach(id=>$(id).onchange=cfg);
function logLine(html){const d=$('log');d.innerHTML+=html+'\\n';
  d.scrollTop=d.scrollHeight;}

// ---- geometry: four stacked line plots filling the left column
const NROWS=4,ML=64,MT=12,MB=26,MR=16,ROWGAP=16,RIGHTW=400;
const COLORS=['#8c1515','#1c6ea4','#2c8c2c','#b8860b'];
const cv=$('scope'),ctx=cv.getContext('2d');
cv.width=Math.max(560,window.innerWidth-RIGHTW-56);
const availH=Math.max(380,window.innerHeight-cv.getBoundingClientRect().top-16);
const ROWH=Math.max(80,Math.floor((availH-MT-MB-(NROWS-1)*ROWGAP)/NROWS));
cv.height=MT+NROWS*ROWH+(NROWS-1)*ROWGAP+MB;
const PW=cv.width-ML-MR;
const rowY=i=>MT+i*(ROWH+ROWGAP);
let _rt;window.addEventListener('resize',()=>{
  clearTimeout(_rt);_rt=setTimeout(()=>location.reload(),400);});

// ---- rolling per-channel history: [{t, v, peak}]
const hist=[[],[],[],[]];
let WINDOW=15,YMAX=0,avail=[];
$('window').onchange=()=>{WINDOW=Math.max(2,parseFloat($('window').value)||15);};
$('ymax').onchange=()=>{YMAX=Math.max(0,parseFloat($('ymax').value)||0);};

function push(msg){
  avail=msg.channels||[];
  const t=msg.t;
  avail.forEach(ch=>{
    hist[ch].push({t:t,v:(msg.amp[ch]||0),peak:(msg.peak[ch]||0)});
  });
  // drop points older than the window (a little slack for a clean left edge)
  const cutoff=t-WINDOW*1.05;
  for(let i=0;i<NROWS;i++)
    while(hist[i].length&&hist[i][0].t<cutoff)hist[i].shift();
  render(msg);
}

function autoYmax(){
  if(YMAX>0)return YMAX;
  let m=0;for(let i=0;i<NROWS;i++)for(const p of hist[i])if(p.v>m)m=p.v;
  return m>1e-6?m*1.15:1;   // headroom; never a zero axis
}

function render(msg){
  const now=(msg&&msg.t)||(hist.flat().slice(-1)[0]||{}).t||0;
  const ymax=autoYmax();
  ctx.fillStyle='#fafafa';ctx.fillRect(0,0,cv.width,cv.height);
  for(let i=0;i<NROWS;i++){
    const y0=rowY(i),yb=y0+ROWH;
    const has=avail.indexOf(i)>=0;
    // plot frame
    ctx.strokeStyle='#e2e2e2';ctx.lineWidth=1;
    ctx.strokeRect(ML,y0,PW,ROWH);
    // y grid + labels (amplitude 0..ymax)
    ctx.fillStyle='#888';ctx.font='10px system-ui';ctx.textAlign='right';
    for(let k=0;k<=2;k++){const v=ymax*k/2;
      const y=yb-ROWH*(k/2);
      ctx.strokeStyle='#efefef';ctx.beginPath();
      ctx.moveTo(ML,y);ctx.lineTo(ML+PW,y);ctx.stroke();
      ctx.fillStyle='#999';ctx.fillText(v.toFixed(2),ML-6,y+3);}
    if(!has){
      ctx.fillStyle='#ededed';ctx.fillRect(ML+1,y0+1,PW-2,ROWH-2);
      ctx.fillStyle='#b0b0b0';ctx.font='12px system-ui';
      ctx.textAlign='center';ctx.fillText('no board',ML+PW/2,y0+ROWH/2+4);
    }else{
      // the amplitude line
      const pts=hist[i];
      ctx.strokeStyle=COLORS[i];ctx.lineWidth=1.6;ctx.beginPath();
      let started=false;
      for(const p of pts){
        const x=ML+PW*(1-(now-p.t)/WINDOW);
        if(x<ML-2)continue;
        const y=yb-ROWH*Math.min(1,p.v/ymax);
        if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);
      }
      ctx.stroke();
      // current value + peak-frequency readout
      const last=pts.length?pts[pts.length-1]:null;
      if(last){
        ctx.fillStyle=COLORS[i];ctx.font='bold 12px system-ui';
        ctx.textAlign='right';
        ctx.fillText(last.v.toFixed(3),ML+PW-6,y0+15);
        ctx.fillStyle='#999';ctx.font='11px system-ui';
        ctx.fillText('peak '+(last.peak>=1000?(last.peak/1000).toFixed(2)+'k'
          :Math.round(last.peak))+' Hz',ML+PW-6,y0+29);
      }
    }
    // channel label
    ctx.fillStyle=COLORS[i];ctx.font='bold 13px system-ui';
    ctx.textAlign='left';ctx.fillText('ch'+i,6,y0+16);
    ctx.fillStyle='#888';ctx.font='10px system-ui';
    ctx.fillText('amp',6,y0+30);
  }
  // shared x-axis time labels
  ctx.fillStyle='#666';ctx.font='11px system-ui';ctx.textAlign='center';
  for(let k=0;k<=3;k++){const frac=k/3;
    const x=ML+PW*frac;const secs=-(WINDOW*(1-frac));
    ctx.fillText(secs.toFixed(0)+'s',x,cv.height-8);}
  ctx.textAlign='right';
  ctx.fillText('time \\u2192  (now)',ML+PW,cv.height-8);
}
render(null);

function connect(){const ws=new WebSocket(`ws://${location.host}/ws`);
 ws.onmessage=e=>{const m=JSON.parse(e.data);
  if(m.type==='scope')push(m);
  else if(m.type==='log')logLine(`<span class=m># ${m.text}</span>`);
  else if(m.type==='status'){
    $('binfo').innerHTML=m.bins
      ?`bins <b>${m.bins[0]}\\u2013${m.bins[1]}</b> `
       +`(${m.bins[1]-m.bins[0]+1}) &nbsp;${m.bin_hz} Hz/bin`
      :'';
    $('status').textContent=`${m.rate/1000} kHz / ${m.bits}-bit`
      +` | fft ${m.fft} @ ${(m.freq/1000)} kHz`
      +(m.dropped?` | dropped ${m.dropped}`:'')
      +(m.channels.length?'':' | NO BOARDS');
    let h='';for(const ch of [0,1,2,3]){
      const on=m.channels.indexOf(ch)>=0;
      h+=`<div style="color:${on?COLORS[ch]:'#bbb'}">ch${ch}: `
        +`${on&&m.amp[ch]!==undefined?m.amp[ch].toFixed(3):'&mdash;'}`
        +`${on&&m.peak[ch]!==undefined?' &nbsp;<span style="color:#999">peak '
          +Math.round(m.peak[ch])+' Hz</span>':''}</div>`;}
    $('levels').innerHTML=h;}};
 ws.onclose=()=>setTimeout(connect,1500);}
connect();
</script></body></html>"""


@app.get("/")
def index():
    return HTMLResponse(HTML)


@app.post("/api/config")
async def config(body: dict):
    for k in ("freq", "fft", "tol"):
        if k in body and body[k] is not None:
            ENGINE.cfg[k] = body[k]
    if int(ENGINE.cfg["fft"]) not in FFT_SIZES:
        ENGINE.cfg["fft"] = 64
        ENGINE.emit("log", text=f"fft size must be one of {FFT_SIZES}")
    # rate/bits go straight to the boards
    rate, bits = body.get("rate"), body.get("bits")
    if rate is not None or bits is not None:
        want_rate = None if rate == ENGINE.boards.rate else rate
        want_bits = None if bits == ENGINE.boards.bits else bits
        if want_rate or want_bits:
            try:
                ENGINE.boards.configure(rate=want_rate, bits=want_bits)
                ENGINE.emit("log", text=f"boards -> rate={rate} bits={bits}")
            except ValueError as e:
                ENGINE.emit("log", text=f"bad rate/bits: {e}")
    ENGINE.emit("log", text=f"config: {ENGINE.cfg}")
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    import asyncio
    try:
        while True:
            for ev in ENGINE.drain():
                await sock.send_text(json.dumps(ev))
            scope = await asyncio.to_thread(ENGINE.scope)
            await sock.send_text(json.dumps({"type": "scope", **scope}))
            await sock.send_text(json.dumps(
                {"type": "status",
                 "rate": ENGINE.boards.rate,
                 "bits": ENGINE.boards.bits,
                 "dropped": ENGINE.boards.stats["dropped"],
                 "channels": scope["channels"],
                 "amp": scope["amp"], "peak": scope["peak"],
                 "fft": scope["fft"], "freq": scope["freq"],
                 "bins": scope["bins"], "bin_hz": scope["bin_hz"]}))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    print(f"boards: {ENGINE.boards.connected}")
    uvicorn.run(app, host="0.0.0.0", port=7861, log_level="warning")
