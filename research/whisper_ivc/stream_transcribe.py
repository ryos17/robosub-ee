"""Web UI for live Whisper transcription of the Daisy hydrophone streams.

Run:  python stream_transcribe.py   (then open http://<orin>:7860)

- Pick a whisper model from the dropdown and load it (nothing runs before).
- Two modes:
    manual     — Start/Stop button; the whole take is transcribed on Stop.
    continuous — audio is cut into window-length chunks, each transcribed.
- Every processed piece of audio is saved to data/<year_date_time>/<N>.wav
  with its transcription in <N>.txt — byte-identical to what whisper heard
  (channel select -> normalize -> optional denoise -> normalize).
- Config (channel 0-3/mix, denoise, mode, window) is changeable in the UI.
- A log-scale spectrum of the selected channel streams while recording.
"""
import glob
import json
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

import daisy_stream
from daisy_stream import SAMPLE_RATE

SB_CACHE = os.path.expanduser("~/.cache/speechbrain")
SPECTRUM_N = 4096          # samples per spectrum frame
SPECTRUM_BINS = 96         # log-spaced display bins


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
        self.stats = {"samples": 0, "dropped": 0}
        self.connected = []
        for serial_no, chans in daisy_stream.BOARDS.items():
            hits = glob.glob(f"/dev/serial/by-id/*{serial_no}*")
            if hits:
                self.connected.append((serial_no, chans))
                threading.Thread(target=self._reader,
                                 args=(serial_no, chans, hits[0]),
                                 daemon=True).start()

    def _reader(self, serial_no, chans, port):
        for frame in daisy_stream.frames(port, self.stats):
            pcm = frame.astype(np.float32) / 32768.0
            with self.lock:
                for i, ch in enumerate(chans):
                    self.latest[ch] = np.concatenate(
                        [self.latest[ch], pcm[:, i]])[-SPECTRUM_N:]
                if self.recording:
                    self.rec.setdefault(serial_no, []).append(pcm)
                    self.rec_counts[serial_no] = \
                        self.rec_counts.get(serial_no, 0) + len(pcm)

    def channels_available(self):
        return sorted(ch for _, chans in self.connected for ch in chans)

    def start_recording(self):
        with self.lock:
            self.rec = {}
            self.rec_counts = {}
            self.recording = True

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
            return min(self.rec_counts.values()) / SAMPLE_RATE


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
        """channel select -> normalize -> denoise -> normalize -> save+asr."""
        raw = self.select_mono(chans)
        if raw is None or len(raw) < SAMPLE_RATE // 2:
            self.emit("log", text="segment too short / channel missing")
            return
        mono = normalize(self.get_denoiser(self.cfg["denoise"])(normalize(raw)))

        n = self.seg_n
        wav_path = os.path.join(self.session_dir, f"{n}.wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((mono * 32767).astype("<i2").tobytes())

        self.busy = f"transcribing {n}.wav..."
        t0 = time.time()
        try:
            result = self.model.transcribe(mono, fp16=True, language="en",
                                           condition_on_previous_text=False)
            text = result["text"].strip()
        finally:
            self.busy = ""
        with open(os.path.join(self.session_dir, f"{n}.txt"), "w") as f:
            f.write(text + "\n")
        self.seg_n += 1
        self.emit("transcript", n=n, secs=round(len(mono) / SAMPLE_RATE, 1),
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
        self.boards.start_recording()
        self.emit("log", text=f"recording -> {self.session_dir}/ "
                  f"(mode={self.cfg['mode']})")
        if self.cfg["mode"] == "continuous":
            self._cont_thread = threading.Thread(target=self._cont_loop,
                                                 daemon=True)
            self._cont_thread.start()

    def _cont_loop(self):
        target = int(float(self.cfg["window"]) * SAMPLE_RATE)
        while self.boards.recording:
            if self.boards.recorded_seconds() * SAMPLE_RATE >= target:
                chans = self.boards.take(target)
                if chans:
                    self.process(chans)
            else:
                time.sleep(0.2)

    def stop(self):
        was_continuous = self.cfg["mode"] == "continuous"
        self.boards.recording = False
        if self._cont_thread:
            self._cont_thread.join(timeout=30)
            self._cont_thread = None
        chans = self.boards.take()
        if chans and (not was_continuous
                      or len(next(iter(chans.values()))) >= SAMPLE_RATE):
            self.process(chans)
        self.emit("log", text=f"stopped; saved to {self.session_dir}/")

    # ---- spectrum
    def spectrum(self):
        with self.boards.lock:
            ch = self.cfg["channel"]
            if ch == "mix":
                avail = self.boards.channels_available() or [0]
                x = np.mean([self.boards.latest[c] for c in avail], axis=0)
            else:
                x = self.boards.latest[int(ch)]
        win = np.hanning(len(x))
        # scale to dBFS: full-scale sine -> 0 dB (hann coherent gain 0.5)
        mag = np.abs(np.fft.rfft(x * win)) / (len(x) / 4)
        freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
        edges = np.geomspace(50, SAMPLE_RATE / 2, SPECTRUM_BINS + 1)
        bins = []
        for i in range(SPECTRUM_BINS):
            m = (freqs >= edges[i]) & (freqs < edges[i + 1])
            v = mag[m].max() if m.any() else 0.0
            bins.append(round(20 * np.log10(v + 1e-9), 1))
        return bins


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
canvas{background:#fafafa;border:1px solid #ddd;border-radius:6px;width:100%;
  height:220px;display:block}
#transcript{background:#f4faf4;border:1px solid #bcd8bc;border-radius:6px;
  padding:16px;min-height:180px;max-height:340px;overflow-y:auto;
  font-size:22px;line-height:1.5;margin-top:10px;white-space:pre-wrap}
#transcript .seg{color:#7a9;font-size:12px;margin-right:8px}
#log{background:#f7f7f7;border:1px solid #ddd;border-radius:6px;padding:10px;
  height:200px;overflow-y:auto;font-family:monospace;font-size:13px;
  white-space:pre-wrap;margin-top:10px;color:#333}
.t{color:#282}.m{color:#679}
</style></head><body>
<h2>Stanford Robosub Inter Vehicle Communication</h2>
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
 <button id=rec>● start</button>
 <button id=clear>clear</button>
 <span id=status></span>
</div>
<canvas id=spec width=1200 height=220></canvas>
<div id=transcript></div>
<div id=log></div>
<script>
const $=id=>document.getElementById(id);
let recording=false;
async function api(p,body){const r=await fetch(p,{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body||{})});return r.json();}
function cfg(){api('/api/config',{mode:$('mode').value,
  window:parseFloat($('window').value),channel:$('channel').value,
  denoise:$('denoise').value});}
['mode','window','channel','denoise'].forEach(id=>$(id).onchange=cfg);
$('load').onclick=()=>api('/api/load_model',{name:$('model').value});
$('unload').onclick=()=>api('/api/unload_model');
$('rec').onclick=()=>{recording?api('/api/stop'):api('/api/start');};
$('clear').onclick=()=>{$('transcript').innerHTML='';$('log').innerHTML='';};
function logLine(html){const d=$('log');d.innerHTML+=html+'\\n';
  d.scrollTop=d.scrollHeight;}
const ctx=$('spec').getContext('2d');
const F_LO=50,F_HI=8000,DB_LO=-100,DB_HI=0,ML=46,MB=24,MT=8,MR=8;
function drawSpec(bins){const W=$('spec').width,H=$('spec').height;
  const pw=W-ML-MR,ph=H-MT-MB;
  ctx.fillStyle='#fafafa';ctx.fillRect(0,0,W,H);
  ctx.font='11px system-ui';ctx.strokeStyle='#e2e2e2';ctx.fillStyle='#777';
  // y grid: dB
  for(let db=DB_LO;db<=DB_HI;db+=20){
    const y=MT+ph*(1-(db-DB_LO)/(DB_HI-DB_LO));
    ctx.beginPath();ctx.moveTo(ML,y);ctx.lineTo(W-MR,y);ctx.stroke();
    ctx.textAlign='right';ctx.fillText(db+' dB',ML-5,y+4);}
  // x grid: log frequency up to Nyquist
  [50,100,200,500,1000,2000,4000,8000].forEach(f=>{
    const x=ML+pw*Math.log(f/F_LO)/Math.log(F_HI/F_LO);
    ctx.beginPath();ctx.moveTo(x,MT);ctx.lineTo(x,H-MB);ctx.stroke();
    ctx.textAlign='center';
    ctx.fillText(f>=1000?(f/1000)+'k':f,x,H-MB+14);});
  ctx.fillText('Hz (log)',ML+pw/2,H-2);
  // EQ-style line with soft fill
  const pts=bins.map((v,i)=>{
    const x=ML+pw*(i+0.5)/bins.length;
    const f=Math.max(0,Math.min(1,(v-DB_LO)/(DB_HI-DB_LO)));
    return [x,MT+ph*(1-f)];});
  ctx.beginPath();ctx.moveTo(ML,H-MB);
  pts.forEach(p=>ctx.lineTo(p[0],p[1]));ctx.lineTo(W-MR,H-MB);
  ctx.closePath();ctx.fillStyle='rgba(40,140,70,0.12)';ctx.fill();
  ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1])
    :ctx.moveTo(p[0],p[1]));
  ctx.strokeStyle='#2c8c2c';ctx.lineWidth=1.6;ctx.stroke();ctx.lineWidth=1;}
function connect(){const ws=new WebSocket(`ws://${location.host}/ws`);
 ws.onmessage=e=>{const m=JSON.parse(e.data);
  if(m.type==='spectrum')drawSpec(m.bins);
  else if(m.type==='transcript'){
    const tr=$('transcript');
    tr.innerHTML+=`<div><span class=seg>#${m.n}</span>${m.text||'&mdash;'}</div>`;
    tr.scrollTop=tr.scrollHeight;
    logLine(`<span class=t>[${m.n}.wav ${m.secs}s asr=${m.asr}s]</span> ${m.text}`);}
  else if(m.type==='log')logLine(`<span class=m># ${m.text}</span>`);
  else if(m.type==='status'){recording=m.recording;
    $('rec').textContent=recording?'■ stop':'● start';
    $('rec').className=recording?'on':'';
    $('mstat').textContent=m.model?`model: ${m.model}`:'no model loaded';
    $('status').textContent=(m.busy?m.busy+' ':'')+
      (recording?`recording ${m.recorded}s | `:'')+
      '16 kHz sampling / 16-bit'+
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
    ENGINE.emit("log", text=f"config: {ENGINE.cfg}")
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
            await sock.send_text(json.dumps(
                {"type": "spectrum", "bins": ENGINE.spectrum()}))
            await sock.send_text(json.dumps(
                {"type": "status",
                 "recording": ENGINE.boards.recording,
                 "model": ENGINE.model_name,
                 "busy": ENGINE.busy,
                 "recorded": round(ENGINE.boards.recorded_seconds(), 1),
                 "dropped": ENGINE.boards.stats["dropped"],
                 "channels": ENGINE.boards.channels_available()}))
            await asyncio.sleep(0.12)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"boards: {ENGINE.boards.connected}")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
