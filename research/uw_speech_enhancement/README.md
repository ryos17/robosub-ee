# Underwater Speech Enhancement: Sim-to-Real for Any SE Model

Research project (lead: Ryota). Goal: make **any** off-the-shelf speech
enhancement (SE) model work on speech that has travelled through water, by
fine-tuning it on data from a physics-based underwater channel simulator and
proving the transfer on real projector-to-hydrophone recordings.

It joins three existing team efforts:

- **Pool acoustic simulator** (Kaito Kanazawa, Keiya Wada; EE project #12):
  the channel model that turns clean speech into underwater speech.
- **Loud underwater speaker** (Keiya Wada, Kaito Kanazawa; EE project #3):
  the projector that plays speech into the pool for the real test set.
- **[`whisper_ivc/`](../whisper_ivc/)**: hydrophone capture (Daisy Seed,
  4 ch, up to 96 kHz / 24-bit) and Whisper transcription on the Orin. Its
  `--denoise metricgan|sepformer` flag is the off-the-shelf baseline this
  project improves on.

Status: **proposal / design**. No code yet.

| Doc | What |
|---|---|
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | **The script of the research**: research questions and pre-registered hypotheses, simulator spec, pool benchmark protocol, models, evaluation, experiments E0-E24, week-by-week schedule with go/no-go gates, risks, paper skeleton and abstract template. Section 0 lists corrections that override this README. |
| [`PRIOR_ART.md`](PRIOR_ART.md) | First prior-art search (2026-10-26) and positioning |
| [`FINDINGS.md`](FINDINGS.md) | Second, deeper research pass (2026-10-27): pool physics, full texts of close papers, SE methodology, hardware, venue |
| [`RESEARCH_UPDATE_3.md`](RESEARCH_UPDATE_3.md) | Third research pass (2026-10-27): multilingual/thesis/patent prior art, Kiel lineage, verification of unverified claims, model and data licences, Stanford logistics, plan deltas |

## Has it been done?

No, not as a whole. A prior-art search (5 search angles plus a skeptic pass,
2026-10-26) found no paper that combines all three of:

1. a physics-based underwater channel simulator applied to clean speech
   corpora to make paired training data,
2. fine-tuning of several off-the-shelf SE architectures on that data, and
3. evaluation on real projector-to-hydrophone speech.

Closest work (the second pass, 2026-10-27, found the Kiel group, which
narrows the novelty; see `RESEARCH_PLAN.md` section 0):

- **Kiel DSS group (Röhrdanz, Wisch, Schmidt et al.), DAGA 2025 and
  DAGA 2026.** An underwater speech channel simulator, and a U-Net
  dereverberator trained on it. The DAGA 2026 paper presents no objective
  metrics or measured-data results. **The simulator is therefore not our
  lead contribution.** Kiel has worked on underwater-telephony SE since 2016
  (theses) and water-tested telephones since 2018.
- **Real through-water speech has been recorded or enhanced before,**
  classically and without release: IOA-CAS multichannel enhancement of
  sea-trial SSB voice (Cao et al. 2017), WTD 71/NATO sea-trial telephone
  speech with listening tests (Fricke & Nissen 2015; Nissen & Kuhnt-Matthé
  2017), Kiel in-pool speech (DAGA 2018).

- **Jiang et al., Acta Acustica 2024.** A learned denoiser inside a digital
  semantic-codec link over simulated underwater channels. It uses one
  bespoke model, and its speech is never sent acoustically. (Correction:
  its channels are *measured* CIRs, and the SE stage is MP-SENet.)
- **LoRA/PEFT for SE already exists** (Cheng & Liu, ICASSP 2026; Bereuter
  et al., IWAENC 2026). The open gap is only LoRA vs full fine-tuning across
  architecture families under a physical channel shift, measured on real data.
- **Rong et al., OCEANS 2025 Brest.** A diver-comms system that converts
  speech to text with fine-tuned Whisper and sends the text over an acoustic
  modem. It has a real water trial and reports WER, but speech never crosses
  the water as audio, so it has no SE. This is the competing design we must
  argue against.
- **In-air twins of the method:** Milo et al. (IWAENC 2026, DeepFilterNet
  plus accurate room simulation), SonicSim (ICLR 2025) and URGENT
  (Interspeech 2024/25). The recipe of simulating, training any model and
  testing sim-to-real is established in air. **The novelty is the
  underwater domain, the real benchmark and the cross-architecture study,
  not the recipe.**

Full table, related areas, risks and tools: [`PRIOR_ART.md`](PRIOR_ART.md).

## Contributions (in the order to pitch them)

1. **To our knowledge, the first openly released through-water speech
   benchmark, with a measured channel.** Speech played through a
   projector in a pool and captured on a 4-channel hydrophone array, with
   time-aligned clean references. No public corpus like this exists. This is
   the strongest claim, and it only holds if the data is released.
2. **Cross-architecture sim-to-real study.** N SE models from M families,
   each measured zero-shot, after sim fine-tuning, and after sim plus a small
   real fine-tune, on real pool data. Prior unusual-channel SE work
   (bone conduction, in-ear) fixes one architecture.
3. **Open underwater speech channel simulator, validated against measured
   pool impulse responses and noise.** Without the validation, reviewers
   will read it as "URGENT with water parameters".
4. **Downstream evidence for AUV and diver voice comms.** Whisper WER before
   and after enhancement, compared with the speech-to-text-over-modem design.
5. **Plug-and-play fine-tuning recipe.** Weak on its own, because toolkits
   like ESPnet-SE++ and SpeechBrain already exist. Say "validated across N
   architectures", never "works for any model". It becomes stronger with an
   architecture-agnostic adaptation method: LoRA/PEFT for SE (no prior work
   found) or RemixIT-style self-training on unlabeled pool audio.

Claims to avoid:

- "First underwater speech enhancement." Ju et al. (OCEANS 2007),
  Woodward & Sari (2007) and Gupta et al. (2022) exist.
- "First deep learning for underwater speech." Jiang et al. (2024) and
  Luo et al. (JASA 2026) exist.
- "Unified framework" as the headline. It reads as packaging.
- "First real through-water speech recordings" or "first enhancement of
  real through-water speech."
- "First to align with sync markers." Nissen & Kuhnt-Matthé (2017) did it.
- "Speech-to-text loses speaker identity." Titanic Calling restores it by
  voice cloning.

## Framework design

```
clean speech corpus ─┐
(LibriSpeech; VCTK   │   ┌──────────── uwsim ─────────────┐
 after licence check)├──▶│ projector response + nonlin.   │
                     │   │ pool/tank image-source or      │──▶ paired (noisy, clean)
noise bank ──────────┘   │ Bellhop/measured-CIR channel   │    multichannel WAVs
(our thrusters, ESCs,    │ absorption, Doppler            │         │
 ambient; DeepShip       │ hydrophone response, ADC       │         │
 internal-only)          │                                │         │
                         └────────────────────────────────┘         ▼
                                                     ┌──── adapters ────┐
 any SE model ──────────────────────────────────────▶│ load / fine-tune │──▶ enhanced
 (MetricGAN+, SepFormer, DeepFilterNet,              │ (full or LoRA)   │      speech
  CMGAN, MP-SENet, SGMSE...)                         └──────────────────┘        │
                                                                                 ▼
 real pool benchmark (projector → 4 hydrophones, aligned refs) ──────────▶   eval:
                                                                   Whisper WER, STOI,
                                                                   PESQ, DNSMOS (2nd)
```

Code layout (v0 implemented; see [`CODE.md`](CODE.md) for install, tests and
the list of uncalibrated placeholders):

```
research/uw_speech_enhancement/
  README.md          this file
  CODE.md            how to install, test, run; module -> plan section map
  environment.yml    conda env "uwse" (Python 3.10)
  uwse/
    uwsim/           channel simulator: signed ISM + R(theta), tail, low band,
                     transducer chain, noise, scene sampler, on-the-fly dataset
    irtools/         ESS sweeps, drift correction, EDC/T20/two-slope, DRR/C50,
                     arrivals, sound speed from TOF
    evaluation/      alignment, level normalization, Whisper WER, ESTOI/PESQ/
                     CI-SDR, cluster bootstrap, TOST, MAPSSWE
    adapters/        SEAdapter interface + identity, WPE, MetricGAN+,
                     SepFormer, DFN3, TF-GridNet, Kiel U-Net (C10), LoRA
    train/           model-agnostic loop, C7/C9 datasets, Whisper-LoRA (C8)
    bench/           metadata schema, LibriSpeech manifests, splits, playlists
  configs/           uwsim_v0.yaml and one YAML per training condition
  scripts/           train, score, smoke_e2e, make_rir_bank, make_playlists,
                     fetch_librispeech
  tests/             pytest suite
  data/              corpora, noise bank, RIR bank, recordings (git-ignored)
```

### Simulator (`uwsim`)

The physics should be as simple as possible, but calibrated against
measurements.

- **Pool:** a custom image-source model with *signed*, angle-dependent
  reflection coefficients. pyroomacoustics cannot represent the
  pressure-release surface (R = -1), because its reflection coefficients are
  non-negative, so it is kept only as the in-air-style baseline. Take c from
  a fresh-water sound-speed equation at the logged temperature (about
  1497-1502 m/s at 25-27 °C) and fit wall and floor reflection to measured
  pool IRs. The pool is a shallow waveguide (cutoff about 88-112 Hz in
  Stanford's 3.35-4.27 m Avery Competition pool, computed). The lower
  scoring edge is set by the projector: about 500-650 Hz for the LL916
  unless pre-emphasis works.
- **Open water:** Bellhop through arlpy, or replay of measured channels
  (Watermark, Underwater Acoustic Channel Library). Use it as a semi-real
  generalization test, not as the main claim.
- **Transducers:** apply the measured projector FIR and a Hammerstein-style
  nonlinearity fitted from a THD-vs-drive sweep. Apply each hydrophone's
  measured response; the baseline purchased hydrophones project (#2) can
  supply calibration data.
- **Noise:** our own thruster and ESC recordings first (the thruster
  denoising project, #15, measures them), then ambient noise and DeepShip.
- **Domain randomization:** pool dimensions, source and receiver positions,
  reflection coefficients, sound-speed jitter, SNR, transducer response.
  Ablate each component.

### Adapters

Every SE model is wrapped behind one interface, so `train.py` and `eval/`
never know which model they hold:

```python
class SEAdapter:
    sample_rate: int                       # model's native rate
    def load_pretrained(self) -> None: ...
    def enhance(self, wav: Tensor) -> Tensor: ...          # (B, T) -> (B, T)
    def loss(self, noisy: Tensor, clean: Tensor) -> Tensor: ...
    def trainable_params(self, mode: str) -> Iterable[Parameter]: ...  # "full" | "lora"
```

Reuse ESPnet-SE++, SpeechBrain and ClearerVoice-Studio checkpoints; don't
reimplement models. Start with three families: MetricGAN+ (masking),
SepFormer (time-domain; Demucs/denoiser is CC-BY-NC and archived), and
DeepFilterNet (deep filtering). Add
CMGAN, MP-SENet and SGMSE (diffusion) later.

### Real benchmark (`bench`)

- **Source:** LibriSpeech test-clean plus live talkers, played through the
  projector.
- **Positions:** a grid of distances (e.g. 1, 3, 10, 20 m), depths and
  orientations. Run each with thrusters off and on.
- **Capture:** 4 channels through `whisper_ivc/daisy_stream.py` at
  96 kHz / 24-bit, keeping the raw native-rate WAVs.
- **Alignment:** play a sync chirp at the start and end of every clip, then
  cross-correlate and correct drift. The two Daisys are on separate clocks,
  see the clock-sync project (#9).
- **Splits:** strict train/test split by position and by speaker. A small
  real set may be used only for the "sim + small real fine-tune" condition.
- **Ground truth for the simulator:** exponential sine sweeps at several
  source and receiver positions give the measured pool IRs.

### Evaluation

- **Primary:** Whisper large-v3 WER plus one non-Whisper ASR, and STOI.
- **Secondary:** PESQ; non-intrusive DNSMOS P.835, NISQA and Distill-MOS,
  used only if they pass sanity checks; SQUIM validated against intrusive
  metrics on our data. DNSMOS is trained in air, so don't trust it alone.
- **Statistics:** bootstrap confidence intervals and paired significance
  tests.
- **Results table:** for each model, compare no enhancement, the pretrained
  model zero-shot, sim fine-tuned, sim plus real fine-tuned, and real only.

## Main risks

| Risk | Mitigation |
|---|---|
| Sim-to-real gap | Calibrate to measured pool IRs; domain randomization; report the gap per model |
| Pool is not the ocean | Scope claims to the pool; semi-real test on measured ocean channels; second site if possible |
| Small real test set | Many positions and speakers, confidence intervals, release the data |
| Projector weak below ~1 kHz, distorts | Measure its response and THD; model it or pre-equalize; report usable band |
| "Why not text over a modem?" (Rong 2025) | Argue latency, identity and prosody without enrolment, no vocabulary limits, no fluent hallucinated text after ASR errors, graceful degradation; cite the lineage (ICNNSP 2003, US8682244, RU2658477, Teledyne US12136424, Rong 2025, Titanic Calling); compare WER vs range if possible |
| RoboSub acoustic rules | Check the handbook; never transmit in the 25-40 kHz pinger band |

## Venue: OCEANS 2027 Aberdeen

Venue details below are from the official sites, checked 2026-10-26.

- **Event:** IEEE/MTS OCEANS 2027, 21-24 June 2027, P&J Live, Aberdeen
  (<https://aberdeen27.oceansconference.org/>).
- **Deadlines:** abstracts are due **15 January 2027** through the EventsAir
  portal, notifications come out in March 2027, and the final paper is due
  **24 May 2027**.
- **Review:** abstract-only review. Recent editions asked for a 1-2 page
  abstract of 500-1000 words and a 4-6 page paper. The Aberdeen limits
  aren't published yet, so check the portal.
- **Selectivity:** historical OCEANS acceptance runs roughly 62-90%; Halifax
  2024 accepted about 66%.
- **Student Poster Competition:** about 20 students are chosen from more than
  100 submissions, and finalists get travel support. Submit to the regular
  program too.
- **Tracks:** Track 02 (Sonar Signal/Image Processing and Communication) is
  the main fit. Track 01 and the local "Digital Innovation in Oceans" track
  (AI, digital twins) are secondary fits.

**Will it be accepted?** It is likely but not guaranteed. The odds are good
for a competent abstract with real results, and OCEANS has published both
underwater SE (Ju et al., 2007) and diver speech ML (Rong et al., 2025). The
Student Poster Competition is much harder (under about 20%). The talk is in
person, so budget for travel. OCEANS counts less than ICASSP or Interspeech
in the speech community. Use it as a first venue, then extend the work to
IEEE JOE, Interspeech or ICASSP.

Things the abstract needs:

1. Real pool numbers, even preliminary: WER and STOI before and after
   fine-tuning, for 2-3 models.
2. One figure comparing the simulated and measured pool IR or noise
   spectrum.
3. Oceans framing: AUV and diver voice comms, and the channel physics. Avoid
   pitching it as an SE leaderboard.
4. Explicit positioning against Rong et al. 2025 and Jiang et al. 2024.
5. A commitment to release the simulator, the benchmark and the scripts.

## Timeline to the abstract (15 Jan 2027)

| Weeks (from 2026-10-26) | Milestone |
|---|---|
| 1-2 | Agree on scope and credit with Kaito and Keiya; read the full texts of Jiang 2024 and Jin 2024; check the RoboSub IVC rules; faculty sponsor named by 10-30; IRB submission 11-03 |
| 2-4 | Measure the projector response and THD; measure pool IRs with sine sweeps; record thruster noise |
| 3-6 | `uwsim` v0 (image source plus transducers plus noise), calibrated to the pool IRs; generate paired data |
| 4-7 | Adapters for 3 models; zero-shot and sim fine-tune runs |
| 6-9 | Record the real benchmark v0 (LibriSpeech playback, several positions, thrusters off and on) |
| 9-10 | Evaluate on real data; make the results table and the sim-vs-real IR figure |
| 10-11 | Write and submit the abstract (regular track plus the Student Poster Competition) |

After acceptance: expand to more models and LoRA, a second site, and the
data release, all by the 24 May final paper, then a journal extension.

## Open questions

- Authorship and credit with Kaito and Keiya (simulator, speaker). Agree on
  them before any submission.
- Which projector: Felicie's impedance-matched piezos, a Lubell, or a
  DIY exciter (EE project #3).
- Which pool (Avery Competition 3.35-4.27 m vs Maas dive well 18 ft) and
  access through stanfordaquatics@stanford.edu.
- Compute: the Orin for inference; training happens elsewhere.
- Whether the speech is sent in baseband or on an ultrasonic carrier (SSB or
  AM). Commercial diver comms use an ultrasonic carrier of about 32 kHz;
  baseband is simpler. Any carrier in 25-40 kHz is forbidden at RoboSub,
  because the handbook keeps the pinger band off limits for intervehicle
  comms.

_Last updated: 2026-10-27_
