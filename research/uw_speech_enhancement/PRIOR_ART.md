# Underwater Speech Enhancement Framework: Prior Art and Positioning

*Stanford RoboSub research proposal. Prior-art sweep done 2026-10-26.*

**Proposal in one line.** A physics-based underwater acoustic channel simulator turns clean speech corpora into paired underwater/clean training data. Any off-the-shelf speech enhancement (SE) model is fine-tuned on that data. The fine-tuned models are then evaluated on real speech played through an underwater projector and captured by a 4-channel hydrophone array in a pool, scored with Whisper WER, PESQ, STOI and DNSMOS.

> **Superseded in part (2026-10-27).** The deeper pass found an underwater speech simulator and U-Net from the Kiel DSS group (DAGA 2025/2026), prior LoRA work for SE, and several factual corrections (Jiang 2024 uses measured channels; pyroomacoustics cannot model R = -1). See `RESEARCH_PLAN.md` section 0 (K1-K22) and `FINDINGS.md`.

**Citation policy for this document.**
- Works listed without a label were confirmed by at least one search pass and were not flagged by the skeptic review.
- Works marked **(unverified)** could not be confirmed, or their details came only from search snippets. They need checking before they go into a paper.
- "Crossref only" means the paper's existence and title were confirmed, but its abstract was not read.

---

## 1. Verdict: has this been done?

**No.** Five independent search angles and a skeptic pass looked in English and Chinese sources across arXiv, IEEE, ISCA, JASA, MDPI, Crossref and Google Scholar. None found a paper that combines the three parts of this proposal:

- (a) a physics-based underwater channel simulator applied to clean speech corpora to produce paired training data;
- (b) fine-tuning of several off-the-shelf SE architectures on that data;
- (c) evaluation on real projector-to-hydrophone speech recordings.

A third pass (2026-10-27: Chinese, German, Japanese, Russian, Korean, theses, patents, DTIC) found classical enhancement of real through-water speech (Cao 2017; Zhang & Liu 2021), unreleased sea-trial telephone speech with intelligibility tests (WTD 71/CMRE), and Kiel's lineage since 2016. None combines the three parts or releases data.

arXiv searches for "underwater acoustic" AND "speech enhancement", and for hydrophone AND speech, returned zero relevant hits. The two closest works each cover only one part:

- **Jiang et al. (Acta Acustica, 2024)** pairs a learned denoiser with simulated underwater channels. It is a bespoke digital semantic-codec system, and the abstract reports simulation results only.
- **Rong et al. (OCEANS 2025)** runs a real underwater trial with fine-tuned Whisper and WER. It avoids the acoustic speech path entirely by converting speech to text and sending it over a modem.

The method itself (simulate a channel, train an unchanged SE model, test sim-to-real with ASR WER) is well established for in-air rooms: DeepFilterNet with accurate room simulation (IWAENC 2026), SonicSim (ICLR 2025) and URGENT (Interspeech 2024/2025). So the novelty is not the recipe. It is three things:
1. the underwater domain;
2. the first real through-water speech benchmark;
3. a cross-architecture study of the sim-to-real gap for underwater speech.

That novelty is real and defensible, but it has to be framed carefully.

---

## 2. Closest prior work

Rows are ordered by how closely each work overlaps with this proposal, closest first.

| Work | Year | Venue | What they did | Overlap | What's missing vs ours |
|---|---|---|---|---|---|
| Jiang, Liu, Zhang, Zhu, Yuan. *Semantic-guided underwater noisy speech communication* ([link](https://www.jac.ac.cn/en/article/doi/10.12395/0371-0025.2024187)) | 2024 | Acta Acustica (声学学报) 49(6) | A speech-denoising network at the transmitter, low-bit-rate semantic features sent over underwater multipath channels, a semantic-correction module, and an SE-fusion decoder. Evaluated under various simulated channel and noise conditions. | Medium-high. The only work that applies learned speech processing with an underwater channel to human speech. | Digital semantic-codec link, not analog speech propagated through water. One bespoke architecture, no plug-and-play SE fine-tuning, no cross-architecture comparison. The abstract reports simulation results only (no projector/hydrophone recordings, no Whisper WER). The channel model is not described in the abstract; the full text was not read. |
| Rong, Chen, Nan. *Underwater Diver Communication System Based on Speech/Text Conversion* ([PDF](https://ddfe.curtin.edu.au/yurong/Oceans25.pdf), [IEEE](https://ieeexplore.ieee.org/document/11104740/)) | 2025 | OCEANS 2025 Brest | Raspberry Pi 5 diver unit. In-mask mic, then Wav2vec2 or LoRA-fine-tuned Whisper, then ASCII text over Delphis modems (24-32 kHz, 640 bps), then PicoTTS at the receiver. Fremantle harbour trial: Whisper-base reached 2.4% WER on 41 sentences. | Medium. Same application, Whisper WER metric, fine-tuning of off-the-shelf speech models, real water trial, same conference series. | Speech never crosses the water as audio. No SE model, no channel simulator, no hydrophone-captured speech, no dataset released. This is a **competing system design** we must argue against. |
| Milo, Götz, Gudjónsson, Nielsen, Pedersen, Pind. *Training DeepFilterNet with Accurate Room Acoustic Simulations…* ([arXiv 2608.20971](https://arxiv.org/abs/2608.20971)) | 2026 | IWAENC 2026 | Trained an unchanged DeepFilterNet3 on image-source-method RIRs versus hybrid wave-based/geometrical RIRs, then tested on measured RIRs. Higher fidelity gave modest objective gains and a clearly lower ASR WER. | Medium-high on method. Physics simulator fidelity, off-the-shelf SE, sim-to-real, ASR WER. | In-air rooms only, one architecture, no underwater channel. |
| Li et al. *SonicSim* ([arXiv 2410.01481](https://arxiv.org/abs/2410.01481)) | 2025 | ICLR 2025 | Customizable acoustic simulator plus the SonicSet dataset. Trained multiple SE and separation models and evaluated them on real recordings, showing better sim-to-real than other synthetic sets. | Medium. Structurally the same contribution: simulator, multi-model training, real benchmark. | In-air moving sources; no underwater physics. |
| Zhang et al. *URGENT Challenge* ([arXiv 2406.04660](https://arxiv.org/abs/2406.04660)); Saijo et al. *Interspeech 2025 URGENT* ([arXiv 2505.23212](https://arxiv.org/abs/2505.23212)) | 2024-25 | Interspeech | One shared simulation pipeline from public corpora, any architecture allowed (discriminative, generative, hybrid), and a broad metric suite (12 metrics in 2024). | Medium. The template for "model-agnostic simulate-and-train". | In-air distortions only, no underwater channel, no hydrophone benchmark. |
| Luo, Zhou, Qiao et al. *Underwater short-utterance speaker identification via Active Identity Embedding…* ([doi](https://doi.org/10.1121/10.0044106)) | 2026 | JASA | LibriTTS speech passed through underwater channels; a HiFi-Codec with identity embedding, adapted with hybrid LoRA and full fine-tuning; PESQ-NB degradation of at most 0.07. | Medium-low. Clean corpus, underwater channel and fine-tuning of a pretrained speech model. | The task is speaker ID / watermarking, not enhancement. No off-the-shelf SE models and no pool benchmark. |
| Jin, Zhao, Zhang et al. *Underwater Embedded Offline Speech Recognition System with Ocean Noise Augmentation* ([doi](https://doi.org/10.1109/USYS62456.2024.11116558)) | 2024 | IEEE USYS 2024 | Embedded offline ASR trained with ocean-noise augmentation. | Medium-low. A partial "simulate, then make speech models robust underwater". | ASR, not SE. The characterization as additive noise only (no physical channel) is inferred from the title; no abstract was available. Read the full text before claiming novelty over it. |
| Li, Chen, Ji, Zhang. *A Helium Speech Correction Method Based on GANs* ([doi](https://doi.org/10.3390/bdcc8110158)) | 2024 | Big Data & Cognitive Computing 8(11) | Helium-speech dataset plus a GAN correction model (AMGAN) built on MetricGAN+. | Low-medium. The only diver-speech work found that adapts an off-the-shelf SE architecture. | The degradation is heliox vocal-tract distortion, not the waterborne acoustic channel. No channel simulation. |
| Wisch, Schmidt. *Mixed Analog-digital Speech Communication for Underwater Applications* ([IEEE](https://ieeexplore.ieee.org/document/9657513/)) | 2021 | 14th ITG Conf. on Speech Communication | Hybrid underwater telephone: power-normalized whitened speech on the analog path, with normalization factor and codebook index on the digital path. | Medium on problem. The same analog through-water speech problem, from a speech-DSP group (Kiel). | Transmitter-side coding. No receiver-side learned SE, no simulator-driven training. |
| Ju, Park, Bae et al. *Multi-Sensor Speech Enhancement using Modified LP Residual for Underwater Speech Communication* ([doi](https://doi.org/10.1109/OCEANS.2007.4449396)); Ju, Kim, Han et al. *Speech Enhancement Using Spatial Processing and Modified Excitation Source…* ([doi](https://doi.org/10.1109/EUC.2008.183)) | 2007-08 | OCEANS 2007; IEEE/IFIP EUC 2008 | Multi-sensor and array-based enhancement of underwater speech using classic DSP. | Medium on multichannel. A precedent for array SE of underwater speech, published at OCEANS. | Classic DSP, no learning, no simulator, no ASR metric. Crossref only; abstracts not read. |
| Gupta, Koul, Nathwani. *Underwater Speech Intelligibility Improvement Between Submarine to Surface Station in Compress Sensing Framework* ([link](https://idr.iitjammu.ac.in/jspui/handle/123456789/160)) | 2022 | ICONAT 2022 | Compressed-sensing SE that projects onto the null space of vessel-noise formants; reports STOI gains. | Medium-low. Underwater speech intelligibility enhancement scored with STOI. | Classical method, vessel noise only, no channel simulator, no DL fine-tuning. |
| Goldwater, Zitterbart, Wright, Bonnel. *ML-based simultaneous detection and ranging of impulsive baleen whale vocalizations…* ([link](https://pubs.aip.org/asa/jasa/article/153/2/1094/2866799/Machine-learning-based-simultaneous-detection-and)) | 2023 | JASA 153(2) | KRAKEN normal-mode simulation with a randomized environment (about 369k examples) plus real noise. A TCN trained only on synthetic data works on real Bering Sea calls. | Medium on method. The cleanest underwater "physics sim, then DL, then real test" precedent. | Bioacoustic detection and ranging, low-frequency open water; not speech, not a pool. |
| Liu, Zhao, Gerstoft, Ma. *Diffusion model-based underwater acoustic communication receiver* ([link](https://pubs.aip.org/asa/jasa/article/158/5/3834/3372528/Diffusion-model-based-underwater-acoustic)) | 2025 | JASA 158(5) | A DDPM denoises channel-distorted mel-spectrograms of FH-BFSK signals, trained on Bellhop CIRs; validated in sea trials. | Low-medium. Spectrogram denoising trained on simulated underwater channels, with sim-to-real. | Digital modem signals, not speech. |
| Heitkaemper et al. *Bone Conducted Signal Guided Speech Enhancement for Voice Assistant on Earbuds* ([link](https://research.google/pubs/bone-conducted-signal-guided-speech-enhancement-for-voice-assistant-on-earbuds/)) | 2025 | ICASSP 2025 | Pretrains on a simulated unusual channel (bone conduction), fine-tunes on under 7 h of real data, evaluates with ASR WER. | Medium on recipe. | One architecture; not underwater. |
| Wang, Chen, Lee, Chen, Wang. *Effective Noise-aware Data Simulation for Domain-adaptive SE…* ([arXiv 2409.01545](https://arxiv.org/abs/2409.01545)) | 2024 | IEEE SLT 2024 | Synthesizes target-domain noisy speech with a learned noise encoder and GAN, then fine-tunes an SE model on the simulated parallel data. | Medium on recipe. | Learned rather than physics-based simulation, one model, VoiceBank-DEMAND only. |

---

## 3. Related areas

### 3.1 Underwater speech and diver communications

**Classic through-water diver speech and noise removal**
- Woodward & Sari, JASA-EL 2007: breathing-noise gating with energy and zero-crossing thresholds, tested in a tank with 4 masks ([link](https://pubs.aip.org/asa/jasa/article/121/4/EL156/538528/Breathing-noise-elimination-in-through-water)).
- Sari & Woodward, LPC/PPM digital diver voice, Springer chapter 2002 ([doi](https://doi.org/10.1007/978-1-4757-3617-5_4)). Journal version: Woodward & Sari, IEEE JOE 21(2):181–192, 1996, doi 10.1109/48.486793. Thesis (Sari 1997) details remain **(unverified)**.
- Hollien & Coleman, JASA 46(1A Suppl.):115–116, 1969 (meeting abstract) ([doi](https://doi.org/10.1121/1.1972696)): Bugg Springs, 30 ft depth, divers 30 ft apart, mean 57–64% for four working units.

**Digital vocoder and speech-to-text links**
- Cheng & Huang, ICNNSP 2003: ASR+TTS for underwater speech transmission ([doi](https://doi.org/10.1109/ICNNSP.2003.1280739)).
- Jiang et al., ICCASM 2012: 300 bps MELPe vocoder with FH-4FSK, pool-tested ([link](https://www.atlantis-press.com/proceedings/iccasm-12/2644)).
- Rong & Chen, MAST extended abstract: earlier Curtin system; Wav2vec2 WER fell from 41.6% to 16% after fine-tuning; it mentions an in-mask pre-processing stage against bubble, breathing and background noise ([PDF](https://ddfe.curtin.edu.au/yurong/MAST.pdf)).
- Teledyne patent US12136424, 2024: speech-to-text over a modem, including AUV voice commands ([link](https://patents.google.com/patent/US12136424)).

**Blind source separation for underwater voice**
- Wang & Chen, OCEANS 2007 Europe ([doi](https://doi.org/10.1109/OCEANSE.2007.4302471)).
- Zhang & Liu, CISCE 2021 ([doi](https://doi.org/10.1109/CISCE52179.2021.9446036)).

**Helium (heliox) diver speech.** This is a different distortion (in the vocal tract, not the water channel), but it is the most active DL diver-speech area.
- Richards, IEEE TASSP 1982 ([doi](https://doi.org/10.1109/TASSP.1982.1163973)).
- Chen & Zhang, Information 2023 ([doi](https://doi.org/10.3390/info14030189)).
- Helium spectrogram ASR, BDCC 2025 ([doi](https://doi.org/10.3390/bdcc9050136)).
- PRL-DAS, BDCC 2026 ([doi](https://doi.org/10.3390/bdcc10050157)).
- LoRA-Whisper heliox ASR, Mao et al., Sci. Rep. 2026 ([link](https://pmc.ncbi.nlm.nih.gov/articles/PMC13136489/)).

**Robot-to-human audio and AUV language interfaces**
- SIREN, Fulton/Sattar/Absar, RA-L 2023: TTS played through an AUV hull in a pool, with a 12-person listener study ([link](https://irvlab.cs.umn.edu/projects/human-robot-collaboration/older-projects/siren-underwater-robot-human-communication-audio)).
- Robb et al., IEEE/OES AUV 2018: natural-language AUV command and control over acoustic modems ([arXiv 1811.03566](https://arxiv.org/abs/1811.03566)).

**Kiel DSS lineage (added by the third pass, 2026-10-27)**
- Theses since 2016: Lösch 2016 (adapted SE for wireless underwater telephony), Reimann 2016, Namenas 2016 (real-time channel simulation), Wisch 2017, Hochmuth 2022, Dyrßen 2024 (NN speech-quality improvement after underwater transmission). Titles only ([theses list](https://dss-kiel.de/index.php/teaching/theses)).
- Wisch et al., DAGA 2018: five voice commands spoken into water in pools, recorded by goggle mics; GMM recognizer at 81.7% ([PDF](https://pub.dega-akustik.de/DAGA_2018/data/articles/000435.pdf)).
- Own underwater telephone first water-tested in July 2018, a few m to about 200 m ([SONAR Sisters](https://dss-kiel.de/index.php/research/research-topics/underwater-signal-processing/sonar-sisters)); no published speech metrics.
- DAGA 2019 (transmit-side dynamics, STOI vs SNR, simulated), DAGA 2020 (telephone chain in a SONAR simulator, with ATLAS Elektronik), DAGA 2022 (MAD noise reduction and equalization, simulation), DAGA 2023 (dry-end AEC and postfilter, AECMOS/DNSMOS).
- Röhrdanz et al., DAS|DAGA 2025 (simulator) and DAGA 2026 (U-Net dereverberation; spectrograms only, no metrics, baselines or measured data).

**WTD 71 / NATO sea trials**
- Fricke & Nissen, DAGA 2015: REP14-Atlantic sea trial; pre-recorded speech over a submarine underwater telephone, about 300 NATO-alphabet messages scored by listeners ([PDF](https://pub.dega-akustik.de/DAGA_2015/data/articles/000096.pdf)).
- Alves & Fricke (NATO CMRE), IEEE UComms 2016: same REP14 data, voice intelligibility under JANUS interference ([doi](https://doi.org/10.1109/ucomms.2016.7583422)).
- Nissen & Kuhnt-Matthé, DAGA 2017: in-band OFDM pre- and postambles around STANAG 1475 speech for Doppler correction, segmentation, equalization and ASR; 2014 deep-water sea data ([PDF](https://pub.dega-akustik.de/DAGA_2017/data/articles/000223.pdf)). Direct precedent for chirp alignment.

**Chinese analog and digital voice (IOA-CAS, Xiamen, HEU)**
- Cao, Zhu, Wu (IOA-CAS), Technical Acoustics 36(5):473–478, 2017: multichannel adaptive enhancement of sea-trial SSB voice ([link](http://sxjs.cnjournals.cn/ch/reader/view_abstract.aspx?flag=1&file_no=20170513&journal_id=sxjs)). Closest precedent for multi-hydrophone enhancement of real through-water speech.
- Yan, Xu, Liu, ICICSP 2023: pre-enhanced DNN joint model for frogman speech, PESQ gains; whether the test data are field or simulated is not stated in the abstract ([doi](https://doi.org/10.1109/icicsp59554.2023.10390720)).
- Li, Wang, Tong (Xiamen), Ship Sci. Tech. 39(1), 2017: analog FM speech at 25 kHz in Wuyuan Bay, PESQ about 1.8–2.3 ([link](https://html.rhhz.net/jckxjsgw/html/51026.htm)).
- Digital vocoder and modem links (MELP over coherent modems, IOA 2008; HEU 2013; Xiamen patents and theses): a group citation for "why not a digital link" (see RESEARCH_UPDATE_3 §2.3).

**Japanese diver speech ML**
- Tsuchida et al. (Kobe), IPSJ DICOMO 2021: DL phoneme recognition from a regulator-mounted mic ([link](https://hdl.handle.net/20.500.14094/90008744)).
- Sasaki & Kosaka (Yamagata), IEICE ISS 2016: voice conversion from underwater to land speech **(unverified)**.
- Logosease (Casio with the Kosaka lab): ultrasonic transceiver with digital consonant enhancement.

**Patents**
- Speech-to-text: US8682244B2 (2005 priority) ([link](https://patents.google.com/patent/US8682244B2/en)); RU2658477C1 (2018, diver ASR → text → modem → TTS) ([link](https://patents.google.com/patent/RU2658477C1/ru)); Teledyne US12136424 (above).
- Mouthpiece and resynthesis: US9065561B2 family and US9219955B2 (Incube) ([link](https://patents.google.com/patent/US9219955)).
- Breath gating: US4154981A (Navy) ([link](https://patents.google.com/patent/US4154981)); US7139701B2 (Motorola, inhalation-noise attenuation) ([link](https://patents.google.com/patent/US7139701)).

**Historical intelligibility reports**
- Ranlet (Raytheon), JASA 89(4B) Suppl., 1991: baseband diver communicator in a pool, 82% words at 5 m ([link](https://pubs.aip.org/asa/jasa/article/89/4B_Supplement/2009/657081/Underwater-audiological-testing-of-a-diver)).
- Pelton, NEDU 2-88 (1988): Navy evaluation of three commercial SSB systems ([DTIC ADA199797](https://apps.dtic.mil/sti/html/tr/ADA199797/index.html)) **(unverified)**.
- USCG, extended-range underwater loudhailer, 2006: > 90% closed-set intelligibility at about 400 yd ([DTIC ADA454550](https://apps.dtic.mil/sti/tr/pdf/ADA454550.pdf)) **(unverified)**.

**Datasets, surveys and commercial context**
- DISPE divers' speech database, ICSLP 1990 ([link](https://www.isca-archive.org/icslp_1990/marchal90_icslp.html)); availability today is unclear.
- Review of underwater voice communication, SYMPOL 2021 ([doi](https://doi.org/10.1109/SYMPOL53555.2021.9689408)).
- Coda Octopus Voice HUB-4 claims "AI-based" noise removal, with no published method ([link](https://sea-technology.com/voice-hub-4)).
- Casio Logosease: ultrasonic (about 32 kHz) through-water voice ([link](https://www.ndiver.com/logosease-lgs-rg004-underwater-comms)).
- No public corpus of speech played through a projector and recorded on hydrophones was found.

### 3.2 Underwater channel simulation for deep learning (non-speech)

**Bellhop-trained models tested on real data**
- DOA estimation with a CRNN, tested on a real 16-element VLA, with a documented sim-to-real drop: Frontiers in Marine Science 2022 ([link](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.1027830/full)).
- Weiss, Singer, Wornell, ICASSP 2023: a CNN localizer with a chi-square-divergence bound on MSE under model mismatch, recommending domain randomization ([PDF](https://sia.mit.edu/wp-content/uploads/2023/06/2023-weiss-singer-wornell-icassp.pdf)).
- Weiss, Arikan, Wornell, MLSP 2022 ([arXiv 2207.10222](https://arxiv.org/abs/2207.10222)).
- Modulation recognition trained on Bellhop channels and tested on river, reservoir and Watermark channels: Sensors 2023 ([link](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346592)) and JMSE 2023 ([doi](https://doi.org/10.3390/jmse11081632)).

**Domain randomization plus a small real fine-tune**
- MTCL-Net, Remote Sensing 2026 ([doi](https://doi.org/10.3390/rs18091343)).

**Learned or hybrid channel models**
- Li & Chitre, data-aided ray propagation validated in a reverberant tank with image-source walls, IEEE JOE ([arXiv 2205.06066](https://arxiv.org/abs/2205.06066)).
- StableUASim, a diffusion channel surrogate ([arXiv 2511.18078](https://arxiv.org/abs/2511.18078)).
- Reservoir-computing channel model trained on water-tank and lake data ([arXiv 2205.14856](https://arxiv.org/abs/2205.14856)).

**Measured-channel replay**
- Watermark benchmark, IEEE JOE 2017 ([doi](https://doi.org/10.1109/joe.2017.2699078)).
- Underwater Acoustic Channel Library, 2026 ([arXiv 2609.03207](https://arxiv.org/abs/2609.03207)).

**Unverified items**
- Bellhop augmentation of ShipsEar/DeepShip in MSRANet, DSP 2025 ([link](https://www.sciencedirect.com/science/article/abs/pii/S1051200425002155)) **(unverified)** (reference list cites Porter's Acoustics Toolbox).
- Federated meta-learning receiver ([arXiv 2105.13296](https://arxiv.org/abs/2105.13296)). Training: simulated Rayleigh CIRs + BELLHOP; validation adds measured NOF/NCS (Watermark) and Wanlu Reservoir CIRs.

### 3.3 Domain adaptation and universal / model-agnostic SE

**Model-agnostic pipelines**
- URGENT 2024/2025 (see the table).
- *Less is More*, ASRU 2025: data curation matters more than scale for SE training ([arXiv 2506.23859](https://arxiv.org/abs/2506.23859)).

**Sim-to-real for SE**
- SonicSim and Milo et al. (see the table).
- RealMAN, NeurIPS 2024 D&B: reports that the sim-to-real gap depends on the architecture ([arXiv 2406.19959](https://arxiv.org/abs/2406.19959)).
- CHiME-7 UDASE: unsupervised adaptation of SE models to real unlabeled recordings ([arXiv 2307.03533](https://arxiv.org/abs/2307.03533)).

**Model-agnostic adaptation**
- RemixIT, IEEE JSTSP 2022: self-training that is claimed to combine with any separation model ([arXiv 2202.08862](https://arxiv.org/abs/2202.08862)).
- Test-time training for SE, Interspeech 2025: tied to its own architecture ([arXiv 2508.01847](https://arxiv.org/abs/2508.01847)).
- No architecture-agnostic PEFT/LoRA study for SE was found. That is a possible secondary contribution.

**Universal enhancers to plug in**
- AnyEnhance, TASLP 2025 ([arXiv 2501.15417](https://arxiv.org/abs/2501.15417)). The public weights are a 45.68M challenge baseline, not the paper's model.

**ASR-side analogue**
- NOPE-HYPE: a simulate-then-fine-tune workflow for Whisper and SeamlessM4T ([arXiv 2609.10058](https://arxiv.org/abs/2609.10058)).

### 3.4 SE for other unusual channels (same recipe, one architecture each)

**Bone conduction**
- Heitkaemper et al., ICASSP 2025 (see the table).
- VibOmni ([arXiv 2512.02515](https://arxiv.org/abs/2512.02515)).
- TRAMBA, Proc. ACM IMWUT 8(4) 2024, doi 10.1145/3699757 ([arXiv 2405.01242](https://arxiv.org/abs/2405.01242)).

**In-ear own voice**
- Ohlenbusch, Rollwage, Doclo ([arXiv 2405.11592](https://arxiv.org/abs/2405.11592)).

**mmWave radar speech**
- End-to-end mmWave eavesdropping, "We Can Hear You with mmWave Radar!" (arXiv only) ([arXiv 2511.06205](https://arxiv.org/abs/2511.06205)).

**Robot-to-robot acoustic communication through a learned differentiable channel**
- The Talking Robot ([arXiv 2603.07072](https://arxiv.org/abs/2603.07072)). Its operating environment is not stated in the abstract.

**Speech models reused for underwater audio**
- UATR-SLM: speech large models used for ship recognition ([arXiv 2601.18086](https://arxiv.org/abs/2601.18086)).
- Dual-path speech-separation networks applied to ship noise ([arXiv 2504.08371](https://arxiv.org/abs/2504.08371)).
- AquaSignal, a U-Net denoiser in an underwater audio pipeline ([arXiv 2505.14285](https://arxiv.org/abs/2505.14285)).

**Excluded:** ClariCodec (arXiv 2604.14654) was withdrawn by its authors (v2) because its bitrate accounting was incomplete. Do not cite it as a valid result. A revised version appeared as arXiv 2605.19541 ('Optimising Neural Speech Codecs for 300bps Communication using Reinforcement Learning', same authors; v2 2026-07-24; no withdrawal notice). Cite only that version, as a digital neural-codec alternative.

---

## 4. The gap and the contributions we can defend

**C1. First open, physics-based underwater speech channel simulator that produces paired data. Moderately strong.**
- The simulator covers pool/tank and open-water configurations. It models multipath with c ≈ 1480 m/s, boundary reflections (a pressure-release surface), frequency-dependent absorption, projector and hydrophone responses, thruster/ESC and ambient noise, and Doppler.
- Existing tools (Bellhop/KRAKEN, Watermark, the UWA Channel Library, pyroomacoustics) were never packaged or validated for speech.
- *Weakness:* reviewers will see this as "URGENT/FRA-RIR with water parameters" unless we validate the simulator against **measured** pool impulse responses and noise spectra.
- *Credit:* the simulator is based on Kaito's project, which has no public trace. Authorship and credit must be agreed with Kaito before any submission.

**C2. First real through-water speech benchmark. Strong, if released.**
- Speech is played through a projector and captured on a 4-channel hydrophone array (Daisy Seed, 96 kHz), with time-aligned clean references.
- No public projector-to-hydrophone speech corpus was found. The Curtin, heliox and DISPE data are private or of unclear availability.
- This is the most defensible claim. It depends entirely on collecting and releasing the data.

**C3. First cross-architecture sim-to-real study for underwater speech. Strong.**
- Compare N representative SE families (masking/discriminative, e.g. MetricGAN+ and DeepFilterNet; time-domain, e.g. SepFormer (Demucs/denoiser is CC-BY-NC and archived); metric-GAN/conformer, e.g. CMGAN and MP-SENet; generative/diffusion).
- Measure each one zero-shot and after fine-tuning, in sim and on real data.
- Prior unusual-channel work fixes one architecture each. RealMAN (NeurIPS 2024 D&B) suggests the gap depends on the architecture, which makes this a real research question rather than a demo.

**C4. Downstream task evidence for AUV and diver voice comms. Moderate.**
- Report Whisper WER before and after enhancement on real pool data.
- Compare directly against the speech-to-text-over-modem design (Rong et al., 2025), arguing on latency, prosody/speaker identity, graceful degradation and hardware cost.
- *Weakness:* without that comparison, reviewers will ask why we bother with analog speech at all.

**C5. A plug-and-play fine-tuning recipe that works for "any" SE model. Weak as stated.**
- Model-agnostic SE toolkits already exist (ESPnet-SE++, SpeechBrain, ClearerVoice-Studio, URGENT's pipeline). An adapter layer on top of them is engineering, not science.
- "Any" cannot be proven. Reword it as "validated across N architectures from M families".
- It becomes stronger if paired with a model-agnostic adaptation contribution, for example:
  - architecture-agnostic LoRA/PEFT for SE, for which no prior work was found;
  - RemixIT-style self-training on unlabeled pool recordings.

**Claims to avoid:**
- "First underwater speech enhancement." Ju et al. 2007/2008, Gupta et al. 2022 and Woodward & Sari 2007 exist.
- "First DL for underwater speech." Jiang et al. 2024 and Luo et al. 2026 exist.
- "Unified framework" as the headline. It reads as packaging; lead with the benchmark and the sim-to-real study.
- Before claiming novelty, read the full texts of Jiang et al. 2024 and Jin et al. 2024.

---

## 5. Risks reviewers will raise, and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Sim-to-real gap** | Every underwater sim-to-DL paper reports a drop on real data (e.g. Bellhop DOA, Frontiers 2022). Reviewers will want it measured, not assumed. | Measure pool IRs with exponential sine sweeps at several source/receiver positions and calibrate the simulator to them. Report a table of real-only, sim-only, sim+real fine-tune and zero-shot results. Use domain randomization over geometry, reflection coefficients, sound-speed jitter, noise SNR and transducer responses, citing Weiss et al. 2023 and MTCL-Net. Ablate each simulator component to show which ones matter. |
| **Pool vs ocean** | A small tiled pool has a dense, long reverberant field and low-frequency modal behaviour. At 1 kHz the wavelength is about 1.5 m, so image-source assumptions weaken at low frequencies. Ocean channels are sparse, time-varying and Doppler-dominated. Pool results do not transfer to open water. | Scope the claims explicitly to the pool, or add a second real site: a different pool or tank, a lake, or a harbour. Evaluate open-water generalization on measured-channel replay (Watermark, UWA Channel Library) as a semi-real test. Use a leave-one-environment-out protocol. |
| **Small real test set** | A few hundred utterances from one pool give wide confidence intervals and can overfit when the same recordings are used for fine-tuning. | Use many distances, depths and orientations, several speakers, and both played-back corpus speech (e.g. LibriSpeech test-clean) and live speech. Keep strict train/test splits by position and speaker. Report bootstrap confidence intervals and paired significance tests. Release the data. |
| **Projector nonlinearity and band limits** | Small projectors are often resonant well above the speech band and weak below about 1 kHz. They add harmonic distortion and a strongly coloured response. Commercial through-water voice links use ultrasonic carriers (around 32 kHz) for this reason. | Measure the projector's frequency response and THD versus drive level. Model it in the simulator (measured FIR plus a Hammerstein-style nonlinearity), or pre-equalize at the transmitter. Report the usable band. Consider an optional SSB/AM up-conversion path as an ablation, since it matches real diver equipment. |
| **Metric validity** | PESQ and STOI need a time-aligned clean reference; through-water delay and clock drift break naive alignment. DNSMOS is trained on in-air data and may be unreliable on underwater audio. Whisper can hallucinate on heavily degraded audio. | Align with cross-correlation and drift compensation; play known sync chirps. Treat DNSMOS as secondary. Make ASR WER (Whisper plus one non-Whisper ASR) and STOI the primary metrics. If feasible, add a small listening test (MUSHRA or DRT), following the SIREN listener-study precedent. |
| **"Why not speech-to-text over a modem?"** | Rong et al. (OCEANS 2025) reached 2.4% WER that way, and Teledyne patented the approach. | Argue on latency, preservation of speaker identity and prosody, working without vocabulary or language assumptions, graceful degradation at low SNR, and cost. If possible, show a WER-versus-range or WER-versus-SNR comparison against a text-over-modem baseline. |
| **Inter-vehicle comms frequency rules (RoboSub)** | Competition acoustic channels are shared with pingers and other teams' hydrophones. Speech-band transmission or an ultrasonic carrier could conflict with competition rules or interfere with localization. | Check the current RoboSub handbook for acoustic transmission rules and pinger bands before designing the IVC demo; this sweep did not verify those rules. Keep projector output out of the pinger band and characterize interference with the team's own TDOA pinger pipeline (described in the Stanford RS26 TDR). |
| **Incremental method** | The in-air twins (Milo et al. 2026, SonicSim, URGENT) make the method itself look like an obvious extension. | Lead with the benchmark (C2) and the cross-architecture finding (C3). Present the simulator as the enabling tool, and cite the in-air twins openly. |

---

## 6. Tools and datasets to reuse

**Channel simulation**
- **Acoustics Toolbox (Bellhop/KRAKEN)**: open-water ray and mode propagation. <http://oalib.hlsresearch.com/AcousticsToolbox/>
- **UnderwaterAcoustics.jl / arlpy** (org-arl): unified Julia/Python propagation interface, some models differentiable; useful for calibrating to measurements. <https://github.com/org-arl/UnderwaterAcoustics.jl>
- **VirtualAcousticOcean.jl**: real-time simulator with hydrophone-array nodes. <https://github.com/org-arl/VirtualAcousticOcean.jl>
- **pyroomacoustics**: image-source method for a pool/tank. Set c ≈ 1480 m/s, a surface reflection coefficient near −1, and fitted wall reflection coefficients. No prior underwater speech use was found. <https://github.com/LCAV/pyroomacoustics>
- **Watermark**: measured time-varying channel replay. <https://ffi.no/en/research/watermark>
- **Underwater Acoustic Channel Library**: measured CIRs plus site noise models, with Python/Julia/MATLAB replay. <https://arxiv.org/abs/2609.03207>
- **StableUASim**: diffusion surrogate for channel diversity. <https://arxiv.org/abs/2511.18078>
- **MATLAB Audio Toolbox, underwater acoustic channels**: reference implementation. <https://www.mathworks.com/help/audio/underwater-acoustic-channels.html>

**Noise sources**
- **DeepShip / ShipsEar**: real vessel and ambient hydrophone noise, no speech. DeepShip paper: <https://www.sciencedirect.com/science/article/abs/pii/S0957417421007016>. No licence stated; distributed by email; internal use only; release IDs and scripts, not audio.
- **Ocean Networks Canada** hydrophone archives: ambient noise.
- **Our own thruster/ESC recordings** from the team's vehicle. This is the most realistic self-noise source and should be part of the release.

**SE toolkits and pretrained models**
- **ESPnet-SE++** (ESPnet enh recipes): many architectures, with joint SE+ASR evaluation. <https://github.com/espnet/espnet>, paper (Interspeech 2022) <https://arxiv.org/abs/2207.09514>
- **SpeechBrain**: pretrained MetricGAN+ and SepFormer on Hugging Face. <https://github.com/speechbrain/speechbrain>, paper (SpeechBrain 1.0, JMLR MLOSS) <https://arxiv.org/abs/2407.00463>
- **ClearerVoice-Studio + SpeechScore**: FRCRN and MossFormer, plus a metrics toolkit (PESQ, STOI, DNSMOS). <https://github.com/modelscope/ClearerVoice-Studio>, paper (Interspeech 2025) <https://arxiv.org/abs/2506.19398>
- **URGENT simulation pipeline and metric suite**: reuse it with underwater CIRs in place of RIRs. <https://arxiv.org/abs/2406.04660>, <https://arxiv.org/abs/2505.23212>
- **Other candidate architectures**: DeepFilterNet, CMGAN, MP-SENet and diffusion SE (e.g. SGMSE). Fetch them from their official repositories; they were not covered by this sweep.

**Clean speech corpora**
- LibriSpeech: <https://www.openslr.org/12/>
- VCTK: <https://datashare.ed.ac.uk/handle/10283/3443>
- DNS Challenge clean speech: <https://github.com/microsoft/DNS-Challenge>
- Note *Less is More* (ASRU 2025) on curating clean data.

**In-air RIR augmentation precedent**
- OpenSLR 26 simulated RIRs: <https://www.openslr.org/26/>

**Adaptation and evaluation protocol**
- CHiME-7 UDASE: <https://arxiv.org/abs/2307.03533>

**Team reference**
- Stanford AUV RoboSub 2026 TDR: hydrophones and TDOA pinger pipeline; no speech work. <https://robonation.org/app/uploads/sites/4/2026/07/RS26_TDR_StanfordAUV.pdf>

---

## 7. Venue: OCEANS 2027 Aberdeen

### Facts (from the official site, checked 2026-10-26)

| Item | Detail |
|---|---|
| Event | IEEE/MTS OCEANS 2027 Aberdeen, hosted by the University of Aberdeen and RGU. <https://aberdeen27.oceansconference.org/> |
| Dates and place | 21-24 June 2027, P&J Live, Aberdeen, Scotland. One third-party listing says 20-24 June, which may include tutorial days. |
| Abstract deadline | **15 January 2027**. Submissions opened 1 October 2026 and are open now, with about 11.5 weeks left. Submit through the EventsAir portal ([call for abstracts](https://aberdeen27.oceansconference.org/2027-call-for-abstracts/), [important dates](https://aberdeen27.oceansconference.org/important-dates/)). |
| Notification | March 2027 (no exact day published). |
| Final paper | **24 May 2027**, for accepted abstracts only. |
| Review type | **Abstract-only review.** The full paper goes to IEEE Xplore without a second full-paper peer review. |
| Abstract and paper length | Not stated on the Aberdeen pages; check the portal. Recent editions: Monterey 2026 used a 500-1000-word abstract (at most 2 pages) and a 4-6-page paper plus references ([link](https://monterey26.oceansconference.org/instructions-for-abstracts-papers-posters-spc/)); Brest 2025 and Sanya 2026 allowed 4-10 pages. |
| Presentation | In person; oral (about 15 min) or poster, assigned by the Technical Committee. |
| Selectivity | No official 2027 rate. The IEEE OES VP-for-OCEANS report gives roughly 62-95% accepted for 2018-2024, mostly 70-90% ([OES Beacon, Dec 2023](https://beacon.ieeeoes.org/oes-beacon/december-2023/vp-oceans-report-december-2023/)). Examples: Kobe 2018 about 70%, Gulf Coast 2023 about 62%, Halifax 2024 about 66% ([report](https://beacon.ieeeoes.org/?p=18518)). |
| Best-fit tracks | **Track 02**, Sonar Signal/Image Processing and Communication (acoustic telemetry and communication; array processing; model-based signal processing). Secondary fits: Track 01 (propagation, ocean noise, transducers, calibration); the local track Digital Innovation in Oceans (AI in ocean science and technology; digital twins); Track 10 (AUVs); Track 05 (numerical modeling and simulation); Track 09 (diving). [Tracks page](https://aberdeen27.oceansconference.org/technical-tracks-2027/) |
| Student Poster Competition | Open to full-time undergraduate or graduate students who are lead author and presenter. About 20 are chosen from more than 100 submissions (under about 20%). Finalists get travel and registration support (amount not stated). There is no separate SPC deadline; past editions used the regular abstract deadline. Students not chosen can still be accepted into the regular programme. [SPC page](https://aberdeen27.oceansconference.org/student-poster-competition-2027/) |

### Can this be easily accepted?

**Likely, but not guaranteed, and "easily" is the wrong frame.**

Reasons the odds are good:
- OCEANS reviews only a 1-2-page abstract, and historically accepts roughly two-thirds or more of submissions.
- The topic fits the venue. OCEANS has published underwater speech enhancement before (Ju et al., 2007) and diver speech-ML systems recently (Rong et al., Brest 2025).

What still matters:
- The abstract has to be competent, on topic and backed by results.
- The **Student Poster Competition is much harder** (under about 20%). Submit to the regular programme as well; students not chosen for the SPC can still be accepted there.
- In-person attendance in Aberdeen is required, so budget for travel.
- OCEANS papers carry less weight in the speech community than ICASSP or Interspeech. OCEANS works best as a first venue that leaves room for a journal extension.

What makes the abstract strong:
1. **Real pool numbers in the abstract**, even preliminary: WER, STOI and PESQ before and after fine-tuning, for at least 2-3 SE architectures, on projector-to-hydrophone recordings.
2. **Simulator validation**: simulated versus measured pool impulse response or noise spectrum, shown in one figure.
3. **Framing for the oceans community**: AUV inter-vehicle and diver voice comms, and the channel physics. Present it as a sim-to-real "digital twin" for underwater voice, not as an SE leaderboard.
4. **Explicit positioning** against Rong et al. 2025 (speech-to-text over a modem) and Jiang et al. 2024 (semantic codec, simulation only).
5. **A promise to release** the simulator, the pool benchmark and the fine-tuning scripts.

Realistic timeline: with the abstract due 15 January 2027, there are about 11 weeks to build the projector/hydrophone rig, record a first real test set and fine-tune 2-3 models. That is tight but feasible for a pilot result. The full multi-architecture study can mature by the 24 May final-paper deadline and the journal extension.

### Alternative and backup venues

Deadlines below are typical patterns that were **not verified** in this sweep; confirm each on the official site.

- **IEEE Journal of Oceanic Engineering (JOE)**: the natural journal extension of an OCEANS paper (full simulator validation plus the cross-architecture study). Rolling submission, full peer review.
- **JASA Express Letters**: short, fast and peer reviewed. It published Woodward & Sari's diver-speech work, so the topic fits. A good home for a focused benchmark or result paper.
- **ICASSP**: high visibility in the speech community and a much higher bar; it would need the full multi-architecture sim-to-real study. The ICASSP 2027 deadline has probably already passed (it is usually in the autumn), which makes ICASSP 2028 the realistic target.
- **Interspeech**: same bar as ICASSP; the deadline is usually around early March. A good fit for "benchmark plus cross-architecture study", given that URGENT and ESPnet-SE++ live there.
- **IWAENC / WASPAA**: acoustic signal processing venues where the in-air twin (Milo et al.) appeared; a good fit for the simulator-fidelity angle.
- **UACE (Underwater Acoustics Conference and Exhibition)**: underwater-acoustics community, suited to the channel-modelling and validation contribution.
- **ACM WUWNet**: underwater networking and systems community, suited to the AUV inter-vehicle comms framing.
- **IEEE/OES AUV Symposium**: if the vehicle-integration demo becomes the main story; Robb et al. 2018 appeared there.

**Recommended path:**
1. Submit an OCEANS 2027 Aberdeen abstract by 15 January 2027 (regular track, plus the SPC if eligible).
2. Extend the work to IEEE JOE, or to Interspeech or ICASSP once the real benchmark and the cross-architecture study are complete.