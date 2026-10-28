# Research Update 3: Prior-Art Sweep, Verification, Licences, Logistics and Plan Deltas

*Underwater speech enhancement, sim-to-real (Stanford RoboSub). Written 2026-10-27, W1. It updates `RESEARCH_PLAN.md` (rev. 2), `README.md` and `PRIOR_ART.md`. It contains no measured results of our own.*

**Inputs and verification policy**

- Five targeted passes fed this update: a multilingual, thesis, patent and 2026 prior-art sweep; a Kiel deep-dive; a citation and fact check; a model and licence audit; and Stanford logistics. They had little or no web-search budget, so most checks opened known URLs directly.
- In this pass I re-checked a few high-impact items myself:
  - OpenAlex records for the frogman DNN paper and Zhang & Liu 2021;
  - arXiv version histories for ClariCodec;
  - the Crossref record for Woodward & Sari, JOE 1996;
  - the full text of Barański et al.;
  - an archived Stanford Athletics page for the Maas dive-well depth;
  - the Ifremer record of the 1990 diver-speech database paper;
  - one web search for 2026 Kiel output, which found nothing new.
- **(unverified)** marks anything taken from a search snippet or from memory, or whose page could not be opened. "Computed" marks my own arithmetic.

---

## 1. Executive summary

- **The core novelty still holds, but its wording must change.** No work found combines three things: (a) a simulator that turns clean speech corpora into paired training data; (b) fine-tuning of several pretrained, off-the-shelf SE architectures; and (c) evaluation with ASR and intrusive metrics on a **released**, aligned projector-to-hydrophone benchmark. Kiel (simulator plus U-Net, no metrics), the frogman DNN paper (ICICSP 2023, data source unclear) and Cao et al. 2017 (classical, sea-trial SSB speech) each cover only part of this.
- **"First enhancement of real through-water speech" is now false.** Several groups have recorded or enhanced real through-water speech:
  - IOA-CAS: multichannel adaptive enhancement of Jiaolong SSB sea-trial voice (Cao, Zhu, Wu 2017).
  - Harbin / Jilin: blind deconvolution of frogman SSB voice on "experiment data" (Zhang & Liu 2021).
  - WTD 71 / NATO CMRE: sea-trial underwater-telephone speech with listening tests (Fricke & Nissen 2015; Alves & Fricke 2016).
  - Kiel DSS: in-pool speech recordings (DAGA 2018).

  None of this data was released. The defensible claim stays "to our knowledge, the first **openly released** through-water speech benchmark with aligned references", plus the pretrained cross-architecture adaptation study.
- **Kiel has a deeper lineage than our docs show.** Underwater-telephony SE theses date back to 2016, there are DAGA papers in 2018, 2019, 2020, 2022 and 2023, and real-water telephone tests have run since July 2018. The DAGA 2026 U-Net paper reports spectrograms only: no metrics, baselines or measured data, although its abstract claims a measured-data evaluation. Their likely next venue is DAGA 2027 (March 2027), between our abstract (01-15) and our final paper (05-24). Risk F8 is real but moderate.
- **The sync-chirp alignment and measured-channel equalization have direct precedent.** Nissen & Kuhnt-Matthé (WTD 71, DAGA 2017) used in-band OFDM pre- and postambles for Doppler correction, segmentation, equalization and ASR. Cite it, and do not claim the alignment idea.
- **The text-over-modem lineage is older and broader than we thought.** It includes ICNNSP 2003, US8682244 (2005 priority), RU2658477 (2017), Teledyne US12136424, Rong 2025 and Titanic Calling (KIT/CMU, 2024). Titanic Calling restores each speaker's voice with YourTTS cloning, so "audio keeps speaker identity" no longer wins on its own. The argument has to rest on no enrolment, prosody, open vocabulary, graceful degradation and no hallucinated content after ASR errors.
- **The pool geometry was wrong.** Stanford's Avery Competition pool is 37 × 20 m and 3.35–4.27 m deep; it is not 25 × 12.5 × 2 m. An archived Stanford Athletics page gives the Maas dive well as 25 × 17 m and 18 ft (5.49 m) deep. The waveguide cutoff c/(4H) is therefore about 88–112 Hz (Avery) or about 68 Hz (Maas), not 187 Hz (computed). Every Appendix A number that depends on depth needs recomputing once the pool is chosen.
- **The projector, not the pool, sets the lower scoring edge.**
  - The Lubell LL916 measured SPL curve (NUWC) gives about 160 dB at 500 Hz against 179 dB at 1 kHz. The unequalized −10 dB point is therefore about 560–660 Hz (computed from plot readings), not the 300–500 Hz the plan expects.
  - There is a deep notch of about 148 dB near 6.2 kHz, inside the speech band.
  - The LL916's rated operating depth is 1–18 m, so the source must sit at ≥ 1 m.
- **Licences block parts of the model and data set for a CC BY 4.0 release.**
  - Demucs (facebookresearch/denoiser) is CC-BY-NC 4.0 and its repo is archived. Drop it from phase 2.
  - WHAM!/WHAMR! mixtures need licensed WSJ0. This also breaks the planned SepFormer forgetting test unless someone holds WSJ0.
  - DEMAND, and therefore VB-DEMAND, has conflicting licences, and DeepShip and ShipsEar state none. Use them for evaluation and internal ablation only.
  - The URGENT TF-GridNet checkpoint the plan could not find has been found: `kohei0209/tfgridnet_urgent25` (CC BY 4.0).
- **Act this week on Stanford logistics.**
  - IRB: new studies must be submitted by **2026-11-03** for December review. The average turnaround is 38 days.
  - An undergraduate Protocol Director needs a **faculty Academic Sponsor**. The same faculty member is needed to sponsor Sherlock and Marlowe access (Marlowe gives new PIs 5,000 free GPU-hours).
  - FarmShare (24 × L40S) is usable today without a sponsor.
  - Pools are not bookable through Rec Rentals. Email stanfordaquatics@stanford.edu.
- **Several "facts" in the plan were corrected:**
  - Salinity: 3 g/kg adds about +3.3 m/s at pool temperature, not +4.
  - Abshagen & Haak (DAGA 2024) is open water in Kiel Bay with a 20 dB reverberation fall in about 2 s. It is not a "harbour T60 > 2 s".
  - DFN3's VB-DEMAND result is not in-domain for DFN3 (it was trained on DNS4).
  - The source for the Navy 160 dB diver limit is dead (HTTP 410).
  - Two Inyo URLs now point to other products.
- **Methods tighten without changing the design:**
  - synchronized ESS with loopback deconvolution, with the THD sweep folded into the harmonic IRs;
  - multi-slope Bayesian/BIC decay analysis plus DecayFitNet;
  - leading-edge chirp detection with a session-level robust drift fit;
  - SQUIM validated against intrusive metrics on our own data;
  - a frozen underwater "bag of hallucinations" for Whisper;
  - a Poisson or negative-binomial GLMM with an offset, replacing the binomial one;
  - an OSF pre-registration with a deviations table.
- **New, cheap, high-value baseline:** the Kiel DAGA 2026 recipe (U-Net, IRM, 6 kHz, log-magnitude) trained on our uwsim. It is about a day of work and gives the closest prior work a direct row in Table II.

---

## 2. New prior work found

Rows are ordered by overlap: high, then medium, then low. "Already in PA" means `PRIOR_ART.md` lists the work but the new pass added facts. Items marked **(unverified)** were not opened.

### 2.1 High overlap

| Work | Year | Venue | URL | What | Overlap | Implication |
|---|---|---|---|---|---|---|
| Wisch, Kaak, Namenas, Schmidt (Kiel). *Spracherkennung in stark gestörten Unterwasserumgebungen* | 2018 | DAGA 2018 München, pp. 888ff | https://pub.dega-akustik.de/DAGA_2018/data/articles/000435.pdf | Five voice commands spoken directly into water, recorded by IP67 mics on swim goggles in pools and swimming baths. 100 Hz HPF plus 50 Hz comb, mel+LDA, GMM with mean-value tracking: 81.7%. | High | Real in-pool speech recordings existed in 2018. Never write "no real underwater speech recordings exist". Our case differs: projector-to-hydrophone, aligned references, release. |
| Kiel DSS theses: Lösch 2016 (adapted SE for wireless underwater telephony), Reimann 2016, Namenas 2016 (real-time channel simulation), Wisch 2017, Hochmuth 2022, Dyrßen 2024 (NN speech-quality improvement after underwater transmission) | 2016–2024 | Kiel DSS theses list | https://dss-kiel.de/index.php/teaching/theses | Titles only; texts not public. | High | Kiel has worked on underwater-telephony SE for about 10 years. Cite the lineage, not just DAGA 2025/26. |
| Cao Wenjing, Zhu Min, Wu Yanbo (IOA-CAS). *水下单边带语音多通道增强技术* (multichannel enhancement of underwater SSB voice) | 2017 | 声学技术 Technical Acoustics 36(5):473–478, doi 10.16300/j.cnki.1000-3630.2017.05.013 | http://sxjs.cnjournals.cn/ch/reader/view_abstract.aspx?flag=1&file_no=20170513&journal_id=sxjs | Low-compute multichannel adaptive enhancement of received analog SSB voice from several transducers. Mother-ship noise is the main degradation. Validated on sea-trial recordings; beats coherent stacking at low SNR. Classical, no DL. | High | The closest precedent for **multi-hydrophone enhancement of real through-water speech**. Cite it next to our WPE-4ch baseline. (Verified by the sweep; my re-open failed because of an expired TLS certificate.) |
| Yan, Xu, Liu (Shanghai Marine Electronic Equipment Research Institute). *A Frogman Speech Enhancement Algorithm with Pre-Enhanced DNN Joint Model* | 2023 | ICICSP 2023, pp. 624–628 | https://doi.org/10.1109/icicsp59554.2023.10390720 | Ten statistical SE front-ends pre-enhance the signal, then a multi-noise DNN; applied to frogman communication with PESQ gains. **Correction:** the OpenAlex abstract does not say whether the test data are field recordings or simulated mixtures (checked in this pass). | High | The closest DNN-SE-on-diver-speech paper. Read the full text before claiming that DL SE has never been evaluated on real diver comms. |
| Zhang & Liu. *Improved Underwater Acoustic Voice Communication Based on Blind Separation* | 2021 | CISCE 2021, pp. 42–48 | https://doi.org/10.1109/cisce52179.2021.9446036 | Blind deconvolution of multipath frogman SSB voice (complex-cepstrum peak filtering, minimum-phase decomposition). "Simulated case and experiment data"; similarity > 0.85. Already in PA. | High (raised from "related") | Classical dereverberation of through-water speech: a precedent for our WPE baseline. The experiment type (tank or sea) is not stated (checked in this pass). |

### 2.2 Medium overlap

| Work | Year | Venue | URL | What | Overlap | Implication |
|---|---|---|---|---|---|---|
| Wisch & Schmidt. *Nichtlineare Signalverarbeitung für Systeme zur Unterwassertelefonie* | 2019 | DAGA 2019, pp. 1241–1244 | http://pub.dega-akustik.de/DAGA_2019/data/articles/000125.pdf | Transmit-side limiter, compander and harmonic enhancer for analog SSB; STOI vs SNR with simulated 3-path multipath. | Medium | Kiel used STOI on simulated underwater speech in 2019. |
| Wisch, Wiegand, Schulz (ATLAS Elektronik), Schmidt. *Implementierung einer Unterwassertelefonieschnittstelle in einem SONAR-Simulator* | 2020 | DAGA 2020, pp. 592ff | https://pub.dega-akustik.de/DAGA_2020/data/articles/000341.pdf | VAD-gated Wiener filter, compressor, STANAG frequency compression and expansion, AGC, inside a sonar simulator. | Medium | Industry precedent for "simulator plus classical enhancement". |
| Wisch, Kaulen, Kühne, Schmidt. *Geräuschreduktion und Entzerrung für gemischt analog-digitale Sprachübertragungen…* | 2022 | DAGA 2022, pp. 735–738 | https://pub.dega-akustik.de/DAGA_2022/data/articles/000369.pdf | MAD system: LPC/LSF codebook plus energy index sent digitally, residual sent analog over SSB; the energy index drives a VAD, Wiener filter and ±20 dB equalizer. Simulation only. | Medium | Real-data evaluation is listed as their future work. |
| Hochmuth, Wisch, Eisert, Schmidt. *Realtime Improvement of Audio Signals in Underwater Communication* | 2023 | DAGA 2023, pp. 880ff | https://pub.dega-akustik.de/DAGA_2023/data/articles/000432.pdf | Dry-end NLMS AEC, postfilter, AGC; AECMOS/DNSMOS. | Low–medium | Completes the lineage. Not the wet channel. |
| Nissen & Kuhnt-Matthé (WTD 71). *Verbesserung der Sprachverständlichkeit… digitale Anfangs- und Endmarken* | 2017 | DAGA 2017, pp. 160ff | https://pub.dega-akustik.de/DAGA_2017/data/articles/000223.pdf | In-band OFDM-Hadamard BPSK pre- and postambles around STANAG 1475 speech, used for Doppler correction, segmentation, 256-coefficient channel estimation, equalization and ASR. 2014 deep-water sea data (CCUP14). | Medium | **Direct precedent for chirp alignment plus measured-channel equalization.** Cite it in §6.5. |
| Fricke & Nissen (WTD 71). *Experiment… JANUS… Verständlichkeit analoger Sprachkommunikation nach STANAG 1074* | 2015 | DAGA 2015, pp. 650ff | https://pub.dega-akustik.de/DAGA_2015/data/articles/000096.pdf | REP14-Atlantic sea trial: pre-recorded speech over a submarine underwater telephone, about 300 NATO-alphabet messages scored by listeners; word error vs SNR. | Medium | A real through-water intelligibility protocol (not released). Evaluation precedent for E22. |
| Alves & Fricke (NATO CMRE). *Analysis of JANUS and underwater telephone capabilities and co-existence* | 2016 | IEEE UComms 2016 | https://doi.org/10.1109/ucomms.2016.7583422 | Same REP14 sea data; voice intelligibility under JANUS interference. | Medium | English-language citation for the WTD 71 / CMRE sea data. |
| Li Jianwen, Wang Xiaoyang, Tong Feng (Xiamen). *浅海信道调频水声语音通信方法比较* | 2017 | 舰船科学技术 Ship Sci. Tech. 39(1):127–131 | https://html.rhhz.net/jckxjsgw/html/51026.htm | Analog FM speech, 25 kHz carrier, Wuyuan Bay at 100 m and 1 km; **PESQ about 1.8–2.3** at 16–28 dB SNR. | Medium | Precedent for objective metrics on real through-water speech, with baseline PESQ numbers. |
| Tsuchida, Arunothaikrit, Mao, Oonishi, Terada, Tsukamoto, Takiguchi (Kobe). *ダイバー同士の水中会話を支援するための深層学習を用いた音声認識手法* | 2021 | IPSJ DICOMO 2021, pp. 317–324 | https://hdl.handle.net/20.500.14094/90008744 | Regulator-mounted waterproof mic; DL phoneme recognition, with or without pressure sensors. | Medium | Japanese ML on real underwater speech. Source-side (regulator), not channel. |
| Sasaki & Kosaka (Yamagata). *音素の音響的特徴を利用した水中音声の声質変換* | 2016 | IEICE ISS poster ISS-SP-219 | https://www.ieice.org/iss/jpn/Publications/issposter_2016/data/pdf/ISS-SP-219.pdf | Voice conversion from underwater (mouthpiece) speech to land speech with a state-shared GMM. **(unverified: PDF empty)** | Medium | Paired-data "conversion" framing of underwater SE. |
| Logosease (Yamagata Casio with the Kosaka lab) | 2012–2024 | Product; DSCafé talk 2024 | https://www.edn.com/casio-develops-pocket-sized-transceiver-for-scuba-diving-use/ ; https://yuds.kj.yamagata-u.ac.jp/archives/5557 | Ultrasonic diver transceiver with digital consonant enhancement (n/m/b/p); the 2024 talk mentions ML. | Medium | Deployed commercial underwater SE (source side). Already partly in PA. |
| RU2658477C1 (Irkutsk NRTU) | 2018 | Russian patent | https://patents.google.com/patent/RU2658477C1/ru | Diver ASR → text → modem → TTS, up to 200 m. | Medium | Text-over-modem predates Rong 2025 by about 7 years. |
| Eyiokur et al. (KIT/CMU). *Titanic Calling* | 2024 | arXiv 2410.11434 | https://arxiv.org/abs/2410.11434 | In-sub ASR; only text crosses the sonar link; YourTTS voice cloning plus lip-synced video rebuild each speaker topside. Real Titanic dive (2022). | Medium | **Weakens the "speaker identity" argument.** Reframe as identity without enrolment, plus prosody, plus no ASR-error hallucination. |
| Teledyne US12136424B2 / US20240233730A1 (Scussel, Altshuler, Morozov, Coryer) | 2024 | US patent | https://patents.google.com/patent/US12136424 | Vosk ASR adapted to diver/heliox speech; text over Benthos modems (2400 bps MFSK / 15360 bps PSK); eSpeak TTS; AUV voice commands. Already in PA. | Medium | Industry digital alternative. New details only. |
| Ranlet (Raytheon SSD). *Underwater audiological testing of a diver communicator* | 1991 | JASA 89(4B) Suppl., p. 2009 | https://pubs.aip.org/asa/jasa/article/89/4B_Supplement/2009/657081/Underwater-audiological-testing-of-a-diver | Baseband diver communicator in a pool: 82% words at 5 m; about 100% for diving sentences. | Medium | Human-listening anchor for **baseband** through-water speech, which matches our transmission mode. |
| Hollien & Coleman. *Diver-to-Diver Speech Intelligibility of Underwater Communication Systems* | 1969 | JASA 46(1A Suppl.):115–116 | https://doi.org/10.1121/1.1972696 | Bugg Springs, 30 ft depth, divers 30 ft apart; mean 57–64% for four working units. Already in PA. **Corrected: two authors.** | Low–medium | Historical baseline. |
| Pelton (NEDU). *Through-Water Diver Communication System Test and Evaluation*, NEDU 2-88 | 1988 | DTIC ADA199797 | https://apps.dtic.mil/sti/html/tr/ADA199797/index.html | Navy evaluation of three commercial SSB systems; word intelligibility. **(unverified: DTIC 403)** | Medium | Navy SSB intelligibility baseline. |
| USCG. *Extended Range Underwater Loudhailer for Port Security* | 2006 | DTIC ADA454550 | https://apps.dtic.mil/sti/tr/pdf/ADA454550.pdf | Closed-set spoken warnings, > 90% intelligibility at about 400 yd. **(unverified: PDF 403)** | Medium | Rare long-range projector-speech intelligibility number. |
| Sari, PhD thesis (Loughborough) | 1997 | Loughborough PhD | https://repository.lboro.ac.uk/articles/thesis/Underwater_acoustic_voice_communications_using_digital_techniques/9514223 | LPC 2400 bps over digital PPM in a reverberant tank; ANC beat spectral subtraction for bubble noise; ZCR/energy breath gating. **(unverified: 403)** | Medium | Classical tank SE precedent. |
| Woodward & Sari. *Digital underwater acoustic voice communications* | 1996 | IEEE JOE 21(2):181–192 | https://doi.org/10.1109/48.486793 | Archival journal version of the Sari system. **Verified via Crossref in this pass.** | Medium | Cite this rather than the thesis. |
| Woodward & Sari. *Implementation of Digital Technology for Underwater Voice Communications* | 1995 | IOA Sonar Signal Processing conf. (doi 10.25144/20148) | https://www.ioa.org.uk/system/files/proceedings/b_woodward_h_sari_implementation_of_digital_technology_for_underwater_voice_communications.pdf | Taxonomy of through-water speech degradations. | Low–medium | Degradation-model citation for uwsim. |
| US7139701B2 (Motorola; Harton, Jasiuk, Kushner) | 2006 | US patent | https://patents.google.com/patent/US7139701 | Model-based inhalation-noise detection and attenuation; names SCUBA. | Medium | Breath-suppression baseline for in-mask noise (out of scope for v0). |
| Mao, Gu, He, Li, Wang (PLA Naval Medical Center). *LoRA-enhanced Whisper for heliox speech recognition* | 2026 | Sci. Rep. 16:14080 | https://pmc.ncbi.nlm.nih.gov/articles/PMC13136489/ | Whisper-large-v3 on real saturation-dive speech: zero-shot CER 75.8% / 46.9%; full FT 7.9% / 10.4%; **LoRA (q, v; 0.51%) 7.6% / 9.0%**. Already in PA; numbers new. | Medium | Supports the C8 design (LoRA on q/v) in a diving domain. |
| Sunil Kumar et al. *Doppler-Robust Zero-Crossing Deep Autoencoder Vocoder…* | 2025 | SYMPOL 2025 | https://doi.org/10.1109/sympol68153.2025.11395914 | 800 bps learned vocoder; simulated Doppler; STOI 0.65–0.75. | Medium | Learned digital alternative that reports STOI. |
| Wang & Chen. *Adaptive BSS in Underwater Wireless Speech Communication* | 2007 | OCEANS 2007 Europe | https://doi.org/10.1109/oceanse.2007.4302471 | Online Infomax BSS on simulated underwater speech. Already in PA. | Medium | No change. |
| Dmitriev & Aronov. *Improving the Range and Quality of Underwater Voice Communications Using the Modified Khurgin-Yakovlev Algorithm* | 2025 | SUMMA 2025 | https://doi.org/10.1109/summa68668.2025.11302344 | Simulation of channel plus speech reconstruction; claims range gain. | Medium | Recent Russian through-water voice paper. |
| Casanova, Avon, Marchal, Gavarry. *Intelligibilité de la parole en milieu subaquatique* (new in this pass) | 1990/1991 | ISM 90, Toulon | https://archimer.ifremer.fr/doc/00000/1148/ | Bilingual diver-speech database: 200-word phonetically balanced lists read by 17 divers underwater and in chambers, surface to −300 m; CD-ROM release planned but not confirmed. | Medium | Primary source for the DISPE line; availability still unknown. |
| Inria VALSE × 52 Hertz postdoc offer 2025-08624 | 2025 | GdR MACS job offer | https://gdr-macs.fr/sites/default/files/2025-03/2025-08624-offre3-en.pdf | Reconstruction of diver voice deformed by equipment. **(unverified beyond title, team and contact)** | Medium | Watch HAL for output; source-side problem. |
| Du Guiming, master's thesis (Hebei Univ. of Technology), cascaded SE + ASR for full-face-mask speech in ocean noise | 2018 | Thesis cited in CN111976924A | https://patents.google.com/patent/CN111976924A/zh | **(unverified: title and text not found)** | Medium | Possible missed Chinese diver-SE thesis. |
| *Improvement of the diver speech intelligibility in underwater communications using LPC* | 1996 | OCEANS 96 | https://doi.org/10.1109/oceans.1996.569133 | Title only. **(unverified)** | Medium | Early SE-type predecessor; read before citing. |

### 2.3 Low overlap

**Digital vocoder and modem links**
- Guo et al., IOA, MELP over a coherent modem, 6/11 km sea trials. Acta Acustica 33(5), 2008. https://www.jac.ac.cn/article/doi/10.15949/j.cnki.0371-0025.2008.05.016
- Cui et al., HEU, 1.2 kbps MELP with a vector hydrophone, tank test. AMM 2013. https://www.scientific.net/AMM.336-338.1939
- Yin et al., OFDM speech, Acta Armamentarii 2013 **(unverified)**.
- Zhou et al., TR + TF-differential OFDM, Applied Acoustics 2015 **(unverified)**.
- Wu, Zhou, Tong, Chen, TR-OFDM speech with sea trial and PESQ, OCEANS 2016 Shanghai. https://doi.org/10.1109/oceansap.2016.7485479 (may be the English twin of Zhou 2015).
- Xiamen theses (Huang 2013, Zeng 2015, Bu 2014) **(unverified)**.
- Du 2011 MELP thesis **(unverified)**.
- Sun 2010 HEU digitized-SSB thesis **(unverified)**.
- Cheng En et al. review, J. Xiamen Univ. 2011 **(unverified)**.
- Lei & Xu, JASA 131(4) Suppl. 2012. https://pubs.aip.org/asa/jasa/article/131/4_Supplement/3277/676021
- Jiang et al., 300 bps MELPe, ICCASM 2012 (already in PA). https://www.atlantis-press.com/article/2644.pdf
- CN103310793B, Xiamen MELP600 + OFDM, MOS about 2.2. https://patents.google.com/patent/CN103310793B/zh
- Yu, Wang et al., KAN-MELP, J. Appl. Acoust. 44(6) 2025: PESQ about 2.6 at 1.96 kbit/s. https://files.sciengine.com/pdfs/1000-310X/DAC633042A2A4026B5D1455D80C58FB7.pdf
- Bai et al., contact-conduction MELP, WUWNet 2013. https://doi.org/10.1145/2532378.2532395
- Ke et al., digital FM voice, WUWNet 2018. https://doi.org/10.1145/3291940.3291974
- Lin, UFMC + LDPC, Appl. Sci. 2021. https://doi.org/10.3390/app11041818
- OpenAlex cluster of 2003–2011 digital links **(unverified, titles only)**. https://doi.org/10.1109/milcom.2007.4455259
- MELP ARM+DSP prototype, 2013 **(unverified)**. https://files.core.ac.uk/download/41440392.pdf

**Implication:** cite these as a group for "why not a digital vocoder or modem". They do not threaten novelty.

**Deployed analog SSB voice**
- Zhu Weiqing et al., Jiaolong signal processing, Acta Acustica 2012. https://www.jac.ac.cn/article/doi/10.15949/j.cnki.0371-0025.2012.06.001
- CAS Bulletin 2019. http://old2022.bulletin.cas.cn/publish_article/2019/3/20190307.htm
- Cagdas et al., underwater telephone design, SIU 2020. https://doi.org/10.1109/siu49456.2020.9302188
- Song (KORDI), 1994 Korean review **(unverified)**.
- Korean SSB telephone with > 80% sentence intelligibility at 300 m, 2004 **(unverified)**. https://koreascience.kr/article/JAKO200411921990485.pdf

**Implication:** analog SSB voice is still fielded. This supports the analog-plus-SE framing.

**Quality metrics for through-water speech**
- Yuan, Chen, Li, Cheng, PESQ-mapped quality estimator, JDAP 31(2) 2016. https://sjcj.nuaa.edu.cn/sjcjyclen/article/abstract/S201602010
- Yu, Guo, Li, spectral subtraction on MELP + PESQ, AMMEE 2017. https://www.atlantis-press.com/article/25878464.pdf

**Implication:** precedent for PESQ on underwater speech.

**Classical noise-only SE in underwater noise**
- Caldeira & Coelho, EEMD-IF, IEEE SPL 30:294–298, 2023. https://doi.org/10.1109/LSP.2023.3258219
- Caldeira & Coelho, SBrT 2021. https://biblioteca.sbrt.org.br/articles/2820
- Zhang et al., NWPU cabin-noise Conv-TasNet, Acta Acustica 48(4) 2023. https://www.jac.ac.cn/cn/article/pdf/preview/10.15949/j.cnki.0371-0025.2023.04.012.pdf

**Implication:** additive noise only, with no channel. Cite as background.

**Pool or tank speech and audio measurement**
- Wang et al., TIMIT through an outdoor pool for TDOA, Shock and Vibration 2022. https://doi.org/10.1155/2022/5280154
- Mourali et al., music through a pool loudspeaker to a hydrophone, Forum Acusticum 2023. https://dael.euracoustics.org/confs/fa2023/data/articles/000668.pdf
- Murayama et al., low-cost channel sounder, Lake Biwa, J. Signal Process. 21(3) 2017. https://www.jstage.jst.go.jp/article/jsp/21/3/21_97/_article/-char/ja
- Kaulen & Schmidt, adaptive-ID transducer characterisation, IET RSN 17(7) 2023 **(unverified content)**.

**Implication:** method citations for our pool protocol and IR measurement.

**Standards and test protocols**
- NF S31-116 (1991), speech communication underwater and hyperbaric (listing only). https://www.dinmedia.de/en/standard/s31-116/12341260
- Navy SBIR N152-110 (MRT / ANSI S3.2). https://navysbir.com/n15_2/N152-110.htm
- IMCA D 008. https://www.imca-int.com/product/testing-of-through-water-communications
- NSMRL 658 (1971) **(unverified)**. https://apps.dtic.mil/sti/pdfs/AD0734127.pdf
- Navy/UF speech-material reports **(unverified)**. https://apps.dtic.mil/sti/citations/AD0751662
- NEDU 3-97 **(unverified)**. https://apps.dtic.mil/sti/pdfs/ADA329089.pdf
- Kuramoto et al., spectral EQ of speech in water, AST 27(1) 2006. https://www.jstage.jst.go.jp/article/ast/27/1/27_1_56/_article/-char/en
- Kasai et al., diver hearing in He-air, JASJ 55(8) 1999. https://www.jstage.jst.go.jp/article/jasj/55/8/55_KJ00001457227/_article/-char/ja/

**Implication:** an MRT / ANSI S3.2 closed-set option for E22.

**Diver mic, mask and breath patents**
- US9065561B2 / US9385817 / US9767821 (Incube; recognize and resynthesize). https://patents.google.com/patent/US9065561B2
- US9219955B2 (Incube; mouthpiece mic, breath filtering). https://patents.google.com/patent/US9219955
- US8682244B2 (WFS; speech-to-text over an EM link). https://patents.google.com/patent/US8682244B2/en
- US5572485A (Safare Crouzet; slow-playback SSB). https://patents.google.com/patent/US5572485A/en
- US4154981A (Navy breath gate). https://patents.google.com/patent/US4154981
- US4799263A **(unverified)**. https://patents.google.com/patent/US4799263
- US10342705B1 (Oceanit). https://patents.google.com/patent/US10342705
- US5579284A family (bone conduction). https://patents.google.com/patent/US5579284A/en
- CN1068456A / JPH0516879A (Divecomm differential mic). https://patents.google.com/patent/CN1068456A
- KR20030084457A. https://patents.google.com/patent/KR20030084457A/en
- CN104955526B (3M respirator SE). https://patents.google.com/patent/CN104955526B/enExample
- CN111976924A **(unverified)**. https://patents.google.com/patent/CN111976924A/zh
- CN113178207A (helium speech) **(unverified)**. https://patents.google.com/patent/CN113178207A/en

**Implication:** transmit-side and in-mask prior art. Our scope (the wet channel) is unaffected.

**Commercial or press-only claims of "AI" diver voice**
- GidroKom / HydroCom (MIPT, 2025). https://poisknews.ru/tehnologii/golosovaya-svyaz-pod-vodoj-rossijskie-inzhenery-sozdali-unikalnuyu-sistemu/ ; https://www1.ru/en/news/2025/05/04/vyzyvat-ixtiandra-stalo-proshhe-cifrovuiu-raciiu-gidrokom-dlia-sviazi-pod-vodoi-na-glubine-do-30-metro.html **(the second is unverified)**
- Talky-Divy (52 Hertz, French MoD funded). https://www.opex360.com/2025/08/12/le-ministere-des-armees-finance-une-innovation-permettant-de-communiquer-vocalement-sous-l'eau/

**Implication:** industry is moving to digital modem plus NN. Cite as motivation only; no published methods.

**Other alternatives and loosely related work**
- Rong & Chen MAST (already in PA). https://ddfe.curtin.edu.au/yurong/MAST.pdf
- AquaVLM. https://arxiv.org/abs/2510.21722
- Word2Wave **(unverified)**. https://arxiv.org/abs/2409.18405
- Optical voice:
  - CoDiver, SIU 2023. https://doi.org/10.1109/siu59756.2023.10223797
  - Sartori et al., TechRxiv 2022. https://doi.org/10.36227/techrxiv.19584082.v1
- Diver intent recognition, CASE 2023. https://doi.org/10.1109/case56687.2023.10260480
- Event2Audio (water box: mic PESQ 1.03, STOI 0.48), ICCP 2025. https://arxiv.org/abs/2507.03273
- WaLi **(unverified)**. https://arxiv.org/abs/2506.22311
- Sarkar et al., AFM 2024 **(unverified)**. https://doi.org/10.1002/adfm.202419049
- Deeb et al., HuBERT transfer to diver detection, Arch. Acoust. 51(3) 2026. https://wydawnictwo.pan.pl/index.php/aa/article/view/2753
- Kaneko & Kubota, ROBOMECH 2021. https://www.jstage.jst.go.jp/article/jsmermd/2021/0/2021_1P1-B09/_article/-char/ja
- Cho et al., Korean diver voice, KIICE 2002. https://koreascience.kr/article/CFKO200211921540408.page?lang=en
- Chithra et al. review, SYMPOL 2021 (already in PA).
- Chinese frogman reviews, Technical Acoustics 2019 **(unverified)**. http://www.sxjs.ac.cn/article/doi/10.16300/j.cnki.1000-3630.2019.02.010
- Use of speech production repair strategies, UHM 2003 **(unverified)**.
- Diver comms and localization, Global Oceans 2020 **(unverified)**. https://doi.org/10.1109/ieeeconf38699.2020.9389462
- Hydrophone pinger rules: no change.

**Implication:** framing only. HuBERT transfer and Event2Audio support the "speech models transfer" and "water strips the HF band" arguments.

**ClariCodec (corrects PA L166)**
- *Optimising Neural Speech Codecs for 300bps Communication using RL*, Wang et al., arXiv 2605.19541: v1 2026-05-19, v2 2026-07-24, no withdrawal notice. https://arxiv.org/abs/2605.19541
- It is a revised version of the withdrawn 2604.14654 (200 bps; same seven authors; withdrawn 2026-04-19 for incomplete bitrate accounting). https://arxiv.org/abs/2604.14654
- **Implication:** cite 2605.19541 if at all, as a digital neural-codec alternative. Never cite 2604.14654.

**Search caveats**
- CNKI and Wanfang were not searched directly.
- The Kiel ICASSP/EUSIPCO/OCEANS sweep was limited. My one search in this pass found only DAS|DAGA 2025 for Röhrdanz + Schmidt (https://pub.dega-akustik.de/DAS-DAGA_2025/konferenz-92.html?session=59).
- A general "underwater speech enhancement hydrophone 2026" search found nothing newer.

---

## 3. The Kiel DSS group (CAU Kiel, Prof. Gerhard Schmidt)

### 3.1 Full picture

| Year | Item | Real water? |
|---|---|---|
| 2016–17 | Theses: Lösch (adapted SE for wireless underwater telephony), Reimann (underwater telephony system), Namenas (real-time channel simulation), Wisch 2017 (thin-film hydrophones). Titles only, at https://dss-kiel.de/index.php/teaching/theses | ? |
| 2018 | DAGA 2018: in-water talkers, goggle mics, pools, 5-command GMM recognizer at 81.7% | Yes (pool) |
| 2018 | SONAR Sisters page: own underwater telephone **first water-tested July 2018, a few m to about 200 m, full duplex**, on research boat MISSy. https://dss-kiel.de/index.php/research/research-topics/underwater-signal-processing/sonar-sisters | Yes; no published speech metrics |
| 2019 | DAGA 2019: transmit-side dynamics processing, STOI vs SNR, simulated | No |
| 2020 | DAGA 2020 with ATLAS Elektronik: telephone chain inside a SONAR simulator | No |
| 2021 | ITG 2021 MAD paper (https://ieeexplore.ieee.org/document/9657513/; **not opened, unverified content**) | ? |
| 2022 | DAGA 2022: MAD noise reduction and equalization, simulation | No |
| 2023 | DAGA 2023: dry-end AEC/postfilter, AECMOS/DNSMOS | Partly |
| 2024 | Dyrßen thesis: NN speech-quality improvement after underwater transmission (title only) | ? |
| 2025 | Röhrdanz et al., DAS\|DAGA 2025 pp. 951–954: data generation (simulator). https://pub.dega-akustik.de/DAS-DAGA_2025/files/upload/paper/134.pdf | No |
| 2026 | Röhrdanz et al., DAGA 2026 pp. 814–817: U-Net dereverberation. https://doi.org/10.71568/daga2026.144. Presented 25 Mar 2026. | Abstract claims it; the paper shows none |

**Simulator (2025)**
- Random 2D/3D polygon environments, image sources with three validity checks, and one scalar reflection factor per surface.
- **"Each reflection imposes a phase shift of 180°"** on every boundary. This is physically wrong for hard walls and seabeds, where R > 0.
- Francois–Garrison absorption plus spreading.
- No Doppler, sound-speed profile, transducer responses, or SSB carrier offset.
- LibriSpeech speech; STANAG (300 Hz–3 kHz, SSB at 8087.5 Hz) and MAD modes.
- The example scene is the WTD 71 harbour, with spectrograms only.

**U-Net (2026)**
- Data and IRs:
  - Harbour shapes with 6–10 interfaces, 200×200×10 m to 800×800×30 m.
  - **500 IRs** in total; reflection factor now frequency-dependent.
  - 84,000 LibriSpeech examples.
- Features and network:
  - fs 6 kHz; FFT 512, hop 32; input a 128×256 log-magnitude block.
  - 4-level U-Net; IRM target; noisy phase reused.
  - Loss: MSE on the mask plus λ·L1 on magnitude; Adam 1e-4, 100 epochs, batch 32.
- Evaluation and gaps:
  - Spectrogram figures only; no PESQ, STOI, SI-SDR or WER, no baseline, no measured data.
  - Split and IR hold-out not stated; causality not discussed.
- Stated next steps: add MAD (they note it whitens away the envelope the U-Net uses), enlarge the databases, and combine with real recordings.

**Infrastructure**
- A 5 × 5 × 5 m chlorinated, rubber-lined tank: T20 about 77 ms unlined vs about 15 ms lined, but **measured at 40–80 kHz**. There is no speech-band characterisation (https://pub.dega-akustik.de/DAS-DAGA_2025/files/upload/paper/110.pdf).
- Boats MISSy and CASSy; the KiRAT real-time framework; ties to WTD 71 and the Bundeswehr.
- No funder is named for the speech line.
- No datasets or code have been released; the group's "Data Bases" page has no underwater corpus.

**People**
- Finn Röhrdanz (PhD student; first author on both NN papers).
- Tim Owe Wisch, author of the 2018–2023 work, appears to have left. This is inferred from staff pages and his 2025 affiliation.

**What they have and lack.** They have real-water hardware, a tank, boats and STANAG/MAD know-how. As published, they lack:
- objective metrics, ASR and baselines;
- a validated simulator;
- any data or code release;
- pretrained or SOTA models;
- wideband audio (they are fixed at 6 kHz);
- Doppler or SSB offset;
- vehicle self-noise.

**Next venue.** Most likely DAGA 2027 (March 2027, inferred from their cadence). ITG Speech Communication is possible in autumn 2027 **(unverified cadence)**. Expect one incremental step: MAD, more IRs, and maybe some real recordings.

### 3.2 Differentiation strategy

1. **Lead with the released, measured, paired benchmark:** projector → 4 hydrophones, chirp-aligned, many positions, thrusters on and off, CC BY 4.0. Kiel has released nothing in about 8 years.
2. **Validate the simulator against measured IRs at held-out positions,** and name the physics differences:
   - signed, angle-dependent R(θ) vs "180° at every reflection";
   - a measured shallow pool waveguide vs synthetic 200–800 m harbours;
   - measured transducer FIR and nonlinearity vs none;
   - reported EDC and multi-slope fit vs none.
3. **Run their recipe as a baseline (new condition C10):** U-Net + IRM at 6 kHz on log-magnitude with the published hyperparameters, trained on our uwsim. Table II then shows "Kiel-style U-Net on our sim" vs "pretrained SOTA + our sim" on real data.
4. **Pretrained cross-architecture study** (DFN3, MetricGAN+, zero-shot SepFormer and TF-GridNet, WPE) vs their single model trained from scratch.
5. **Metrics:** Whisper large-v3 plus Parakeet WER, ESTOI against R1, PESQ, cluster-bootstrap CIs, test–retest floor, and a pre-registered primary endpoint.
6. **Transmission-mode ablation (E17):** baseband wideband vs STANAG-style SSB on 8087.5 Hz, adding SSB carrier offset (pitch shift) and Doppler, which neither Kiel paper models. Keep everything below 25 kHz.
7. **AUV self-noise** (T200 thrusters and ESCs) vs their ship noise. This gives a different application framing.
8. **Wording:** "the DAGA 2026 paper presents no objective metrics or measured-data results, although its abstract states a measured-data evaluation." Never write "Kiel has no real data": they have had water-tested telephones since 2018.

### 3.3 Collaboration recommendation

- **Contact them, but only after the OCEANS abstract is submitted:** between 2027-01-18 and 01-31, with preliminary real-pool numbers. Sending our plan before 01-15 invites convergence on the same timeline.
- **The ask:**
  - (a) a few measured speech-band IRs or STANAG recordings as an external test set, under any licence;
  - (b) permission to cite their recipe as a baseline, plus an offer to run it on our released benchmark with credit.
- **The offer:** our benchmark in return. No co-authorship request up front.
- **Expected friction:** defence ties may limit data release (assumption, not verified).
- **If they decline:** cite them, keep the C10 baseline, and move on.
- **Contacts** (from official pages):
  - Prof. Dr.-Ing. Gerhard Schmidt, gus@tf.uni-kiel.de (https://dss-kiel.de/index.php/research/research-topics/underwater-signal-processing/underwater-telephone-systems).
  - M.Sc. Finn Röhrdanz, finr@tf.uni-kiel.de (https://dss-kiel.de/index.php/team/internal-phd-students/finn-roehrdanz).
  - Address Schmidt and copy Röhrdanz.
- **Owner and date:** Lead, 2027-01-25. Watch the DAGA 2027 programme when it is published (owner: Lead).

---

## 4. Verification results

### 4.1 Status of every previously unverified claim

RP = RESEARCH_PLAN.md, F = FINDINGS.md, PA = PRIOR_ART.md, with line numbers.

**Physics and simulator**

| Claim | Where | Status | Correct value | Source |
|---|---|---|---|---|
| Marczak 1997 citation | RP K2 L21 | Confirmed | JASA 102(5):2776–2779, doi 10.1121/1.420332; 0–95 °C, ITS-90 | https://api.crossref.org/works/10.1121/1.420332 |
| Marczak coefficients "from memory" | RP K2, App. A L920 | Confirmed | 1402.385 + 5.038813T − 5.799136e-2T² + 3.287156e-4T³ − 1.398845e-6T⁴ + 2.787860e-9T⁵ | https://resource.npl.co.uk/acoustics/techguides/soundpurewater/ |
| c at 20/25/27/28 °C; dc/dT | RP K2, App. A | Confirmed (recomputed) | 1482.38 / 1496.73 / 1501.92 / 1504.41 m/s; 2.60 m/s/°C at 26 °C | NPL |
| Salinity slope 1.3 m/s per g/kg; +4 m/s for 3 g/kg | RP K2 | **Corrected** | 1.09–1.11 m/s per g/kg at 25–27 °C (UNESCO), so **+3.3 m/s** | https://resource.npl.co.uk/acoustics/techguides/soundseawater/ |
| Pool 25 × 12.5 × 2 m | RP K8, §3; F | **False for Stanford** | Avery Competition 37 × 20 m, 3.35–4.27 m; Belardi 50 × 25 m, ≤ 3.35 m; Baker 50 m × 25 yd, 1.37–2.59 m | https://gostanford.com/facilities/avery-aquatic-center ; https://gostanford.com/news/2013/04/17/facilities-4 |
| Maas dive well depth | (new) | **Found (archived page)** | 25 × 17 m, 18 ft (5.49 m) deep; current facilities page gives no depth | https://gostanford.com/sports/2013/4/17/208433735.aspx |
| Pool salt-chlorinated? | RP K2, §6.3 | Unverified | Ask Aquatics | — |
| rir_generator accepts β < 0 | RP L248; F | Confirmed | Only shape and type checked; `pow(beta[i], abs(n))` keeps the sign; a scalar β is read as T60 | https://raw.githubusercontent.com/ehabets/RIR-Generator/master/rir_generator.cpp |
| Penn State ARL tank | F L79 | Partly | Paper exists: Barnard, Hambric, Maynard, JSV 331(17):3931–3944, 2012. Tank size and f_s unverified. | https://api.crossref.org/works?filter=alternative-id:S0022460X12002830 |
| Way et al. numbers | RP L228; F | Unverified | eprints 401 | — |
| Concrete/tile/steel constants; Brekhovskikh/Jensen page | RP L228 | Unverified | — | — |
| Abshagen & Haak "Baltic harbour T60 > 2 s" | RP L177 | **Corrected** | DAGA 2024 pp. 903–905. Bistatic reverberation off Schönhagen, Kiel Bay, 22–22.5 m deep, 452 m separation, 1–2 kHz HFM. Reports "a fall of the reverberation level by 20 dB within Δt ≈ 2 s". **No T60; not a harbour.** | https://pub.dega-akustik.de/DAGA_2024/files/upload/paper/467.pdf |
| T200 about 3500 RPM, BPF about 175 Hz | F L212, L245 | Corrected / partly | About 3075 RPM at 12 V, about 3600 RPM at 16 V, so BPF 154–180 Hz *if* 3 blades. Blade and pole count unverified. | https://bluerobotics.com/store/thrusters/t100-t200-thrusters/t200-thruster-r2-rp/ |
| DeepShip below 1 kHz | F L222 | Partly | Identity confirmed (ESWA 183, 2021); spectral claim unchecked | https://api.crossref.org/works/10.1016/j.eswa.2021.115270 |
| Autosub 124 dB | F (g) | Unverified | — | — |
| Watermark KAU band | RP K14 | Unverified | Bonus: the UWA Channel Library covers about 75 Hz–25 kHz; its Green channel has fc 6 kHz at 4.5 ksym/s | https://arxiv.org/pdf/2609.03207 |

**Hardware**

| Claim | Where | Status | Correct value | Source |
|---|---|---|---|---|
| LL916C specs | RP L366 | Confirmed | Also: operating depth **1–18 m**, 0–41 °C, 8.0 kg; box has a 50 W series resistor | https://www.lubell.com/products/lubell-ll916c-ll916h-portable-underwater-speakers-for-lap-synchro-swimming/ |
| LL916 SPL below 1 kHz | F | **Resolved** | NUWC 2012, 20 Vrms, 9.8 ft, 7.7 °C, dB re 1 µPa @ 1 m: ≈131 at 100 Hz, 144 at 200 Hz, 149 at 300 Hz, 160 at 500 Hz, 171 at 700 Hz, 179 at 1 kHz; 173–174 at 2–3 kHz; 170 at 4 kHz; 176 at 5 kHz; **≈148 notch near 6.2 kHz**; 179 at 10 kHz; 165 at 20 kHz | https://lubell.com/wp-content/uploads/2024/04/LL916C_SPL.pdf |
| LL916 diameter from LL964 sibling | RP L235, L944, L987 | **Confirmed directly** | 9.25" (234.95 mm) per Lubell; Inyo LL916C 9.0". a = 11.4–11.7 cm; ka = 1 at 2.03–2.09 kHz (computed). The cited Inyo …046138 page is the **LL916C**, not the LL964. | Lubell; https://www.inyopools.com/Products/19400029046138.htm |
| LL916 directivity | RP row 12 | Partly | Near-omni at 1–2 kHz, multi-lobed from 5 kHz (plots at 1, 2, 5, 10, 15 kHz) | https://lubell.com/wp-content/uploads/2024/04/LL916C_polar_plot.pdf |
| LL916 price $1,540–1,850 | RP L366 | **Corrected** | Inyo LL916C **$1,609.99**. The cited …046141 is now the AC202 box ($226.99, out of stock). | Inyo URLs above |
| LL9816 link | F L748 | **Wrong link** | …046144 is the AC211 splitter box ($304.99) | https://www.inyopools.com/Products/19400029046144.htm |
| Borrow an LL916 from Aquatics | RP L366 | Unverified | Context only: Avery hosts artistic swimming | — |
| UW30 specs | RP L367 | Confirmed | 100–10,000 Hz, 30 W, 8 Ω, ≤ 3.0 m depth (1.2 m recommended), 182.6 mm, 25 Ω/20 W parallel resistor + 1.5 A fuse | https://goknight.com/content/documentation/UW30%20EDS.pdf |
| UW30 discontinued; about $500 | RP L367 | Unverified | EV product URL redirects to the homepage (weak evidence) | — |
| DAEX25W-8 | F | Confirmed (spec); price partly | Fs 1,276 Hz, 10 W, IP67, MSRP $37.99 | https://daytonaudio.com/product/1181/daex25w-8-waterproof-25mm-exciter-10w-8-ohm |
| H2d about −165 dB ±4 dB, 20 Hz–4 kHz | RP L373 | Confirmed | XLR version on 48 V; the 3.5 mm version is −172 dB; $232–280 | https://www.aquarianaudio.com/products/h2d-hydrophone |
| Aquarian units uncalibrated individually | F L777 | Confirmed (inferred) | "Typical response" | same |
| Preamp $70 / IEPE $80 / S1 $99 | F | Confirmed | — | https://www.aquarianaudio.com/products.json?limit=250 |
| Reson/B&K at Hopkins or CCRMA | RP L373 | Unverified | No official page lists any | §6 |
| Small class-D boards are BTL | RP L369 | Unverified per model | — | — |
| EH&S sign-off needed | RP L382 | Unverified | No EH&S page retrievable | §6 |
| 145 dB sport divers; Navy 160 dB | RP L388; F L804 | Unverified; **160 dB source dead (HTTP 410)** | Find a primary Navy source | — |

**SE, ML and evaluation**

| Claim | Where | Status | Correct value | Source |
|---|---|---|---|---|
| DFN3 reports VB-DEMAND | RP L151, L577 | Confirmed with caveat | PESQ 3.17, STOI 0.944; trained on DNS4 (VCTK ×10), so the VB-DEMAND test is "unseen", **not in-domain** | https://arxiv.org/abs/2305.08227 |
| MetricGAN+ in-domain | RP L151 | Confirmed | Test PESQ 3.15, Apache-2.0 | https://huggingface.co/speechbrain/metricgan-plus-voicebank |
| VB-DEMAND test = 824 | RP L315 | Confirmed | 824 files | https://arxiv.org/pdf/1703.09452 |
| VB-DEMAND licence | RP L315 | Confirmed for the record, **but** | DataShare: CC BY 4.0. It contains DEMAND noise, which Zenodo lists as both CC BY-SA 3.0 and CC BY 4.0. Treat as evaluation only. The end-user licence file was not opened. | https://datashare.ed.ac.uk/handle/10283/2791 ; https://zenodo.org/records/1227121 |
| WHAM! size and licence | RP L315 | Confirmed (verification pass) | Noise CC BY-NC 4.0; tr/cv/tt 20,000 / 5,000 / 3,000; **mixtures need WSJ0 (LDC)**. The licence pass hit a TLS error on the same site. | http://wham.whisper.ai/ |
| sepformer-wham16k exists | RP L151 | Confirmed | Apache-2.0 | https://huggingface.co/speechbrain/sepformer-wham16k-enhancement |
| SepFormer about 26M | RP L580 | Confirmed | 26M | https://arxiv.org/abs/2010.13154 |
| SGMSE+ default N | RP L581 | Confirmed | N = 30; WSJ0-REVERB: `--N 50 --snr 0.33` | https://raw.githubusercontent.com/sp-uhh/sgmse/main/enhancement.py |
| SGMSE+ VB-DEMAND and WSJ0-REVERB checkpoints | RP L567 | Confirmed | Plus WSJ0-CHiME3, EARS-WHAM, EARS-Reverb | https://github.com/sp-uhh/sgmse |
| URGENT TF-GridNet checkpoint | RP L562 | **Found** | `kohei0209/tfgridnet_urgent25` (CC BY 4.0); also `wyz/tfgridnet_for_urgent24` | https://huggingface.co/kohei0209/tfgridnet_urgent25 |
| peft Conv1d | RP L582 | Confirmed | Linear, Embedding, Conv1d/2d/3d, MHA; **no nn.GRU** | https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py |
| nara_wpe | RP L563 | Confirmed | MIT; `pip install nara_wpe`; TF part legacy; CI on Python 3.7–3.10 | https://github.com/fgnt/nara_wpe |
| Whisper thresholds 2.4 / −1.0 / 0.6 | RP L661 | Confirmed | CLI beam = best_of = 5; Python API greedy | https://github.com/openai/whisper/blob/main/whisper/transcribe.py |
| Whisper floor about 2% on test-clean | RP L670 | Partly | Paper (greedy): large and large-v2 = **2.7%**; large-v3 unverified | https://arxiv.org/abs/2212.04356 |
| Barański 40.3% hallucination | RP L157 | **Confirmed (full text, this pass)** | 121,378 of 301,317 non-speech inferences (40.3%); 9.1% of those looped; large-v3, temperature 0, greedy | https://arxiv.org/abs/2501.11378 |
| Bisani & Ney DOI | RP L701 | Confirmed | doi 10.1109/ICASSP.2004.1326009 | Crossref |
| Prolific $200–400 | RP L680, L739 | Still an estimate | $8/h minimum, $12/h recommended, 33.3% fee | https://www.prolific.com/pricing |
| Stanford compute | RP D6 L905 | **Partly resolved** | FarmShare, Sherlock, Marlowe (see §6) | §6 |
| AnyEnhance weights | F L481 | Partly | Public repo ships a 45.68M challenge baseline, not the paper model | https://github.com/viewfinder-annn/anyenhance-v1-ccf-aatc |
| TRAMBA venue | F, PA L150 | Resolved | Proc. ACM IMWUT 8(4), 2024, doi 10.1145/3699757 | Crossref |
| MSRANet uses Bellhop | F, PA L120 | Unverified (weak support) | Cites the Acoustics Toolbox | — |
| Federated meta-learning channels | F, PA L121 | Resolved | Train: simulated Rayleigh + BELLHOP; validate: measured NOF/NCS (Watermark) + Wanlu Reservoir | https://arxiv.org/pdf/2105.13296 |
| Jin 2024 uses DeepShip / Zipformer / RNN-T | F L349 | References confirmed; usage inferred | — | https://api.crossref.org/works/10.1109/USYS62456.2024.11116558 |
| Heitkaemper test sets | F | Unverified | — | — |
| Chen OCEANS 2024 field data | F | Unverified | — | — |
| Wisch et al. DAGA 2018 | F L503 | **Confirmed; full text read** | pp. 888ff | §2 |
| Tu 2020 pool breathing set | F L538 | Unverified | — | — |
| Frogman SE (ICICSP 2023) uses real recordings | (new list) | **Corrected** | Abstract does not say whether data are field or simulated | OpenAlex (this pass) |
| Zhang & Liu 2021 real recordings | (new list) | Partly | "Simulated case and experiment data"; experiment type unstated | OpenAlex (this pass) |

**Prior-art IDs**

| Claim | Where | Status | Correct value | Source |
|---|---|---|---|---|
| ESPnet-SE++ | PA L241 | Confirmed | Interspeech 2022 | https://arxiv.org/abs/2207.09514 |
| SpeechBrain 1.0 | PA L242 | Confirmed | JMLR MLOSS | https://arxiv.org/abs/2407.00463 |
| ClearerVoice-Studio | PA L243 | Confirmed | Interspeech 2025 | https://arxiv.org/abs/2506.19398 |
| *Less is More* | PA L127, L251 | Confirmed | ASRU 2025 | https://arxiv.org/abs/2506.23859 |
| RealMAN | PA L131, L186 | Confirmed | NeurIPS 2024 D&B | https://arxiv.org/abs/2406.19959 |
| RemixIT | PA L135 | Confirmed | IEEE JSTSP | https://arxiv.org/abs/2202.08862 |
| mmWave eavesdropping | PA L156 | Confirmed (arXiv only) | Han et al., "We Can Hear You with mmWave Radar!" (mmSpeech) | https://arxiv.org/abs/2511.06205 |
| Hollien & Coleman 1969 | PA L69 | **Confirmed, corrected** | Two authors; JASA 46(1A Suppl.):115–116; Bugg Springs | https://doi.org/10.1121/1.1972696 |
| Sari & Woodward chapter 2002 | PA L68 | Confirmed; thesis still unverified | Springer pp. 127–165; JOE 1996 version confirmed (this pass) | https://doi.org/10.1109/48.486793 |
| ClariCodec | PA L166 | **Updated** | Revised as arXiv 2605.19541 (300 bps), not withdrawn | §2 |

**Venues and dates**

| Claim | Where | Status | Correct value | Source |
|---|---|---|---|---|
| OCEANS 2027 dates | README | Confirmed | Abstracts 2027-01-15; final paper 05-24; 21–24 June 2027 | https://aberdeen27.oceansconference.org/important-dates/ |
| Aberdeen length rules | RP L854, L872 | Unverified for Aberdeen | Monterey 2026: 500–1,000 words, ≤ 2 pages; papers 4–6 pages + refs | https://monterey26.oceansconference.org/instructions-for-abstracts-papers-posters-spc/ |
| Monterey ≤ 2250-character summary | F L904 | Unverified | — | — |
| 2025 SPC winners | F L942 | Resolved | Great Lakes 2025: 13 participants; 1st Braden Meyers (BYU), 2nd Leah Hopson (Rutgers), 3rd Nelson Durrant (BYU, acoustic cooperative localization) | https://ieeeoes.org/wp-content/uploads/2025/12/OES_Dec2025-web.pdf |
| JOE overlap policy | README; F | Confirmed | Prior preliminary conference version OK if disclosed; concurrent submission is unethical; OA fee US$2,645 | https://beacon.ieeeoes.org/publications/ieee-journal-of-oceanic-engineering/submitting-a-paper-to-joe/ |
| JOE special issue about 30% overlap | F | Confirmed (UT2025 only) | — | https://ieeeoes.org/wp-content/uploads/2025/05/UT2025_CfP_FinalV3.pdf |
| Interspeech 2027 | F | Confirmed | 29 Aug–2 Sep 2027, São Paulo; deadline **02-09**; 4+2 or 8+2 pages; desk rejection for nonexistent references | https://interspeech2027.org/calls/call_for_papers/ |
| Interspeech 2027 anonymity | F | Unverified | 2024 rule: no posting from one month before the deadline | https://interspeech2024.org/submission-policy/ |
| ICASSP 2027 / 2028 | F | Resolved | 2027 full paper was due 2026-09-23 (Toronto, 16–21 May 2027). ICASSP 2028: Tokyo, 28 May–3 June 2028; deadline TBA (expect about Sep 2027, inferred). | https://2027.ieeeicassp.org/call-for-papers/ |
| Stanford calendar | RP L796 | Partly | 2026-11-26 is Thanksgiving (computed). Finals and winter closure unverified. IRB office closed 12-21 to 01-01. | https://irb.stanford.edu/ |

### 4.2 Concrete text edits (old → new)

**RESEARCH_PLAN.md**

1. **§0 K2 (L21).**
   - Old: "(computed; coefficients from memory, citation JASA 102(5):2776, 1997 unverified)"
   - New: "(computed; coefficients confirmed against the NPL calculator; Marczak, JASA 102(5):2776–2779, 1997, doi:10.1121/1.420332)"
   - Old: "About 3 g/kg salinity adds roughly +4 m/s (unverified slope of about 1.3 m/s per g/kg)."
   - New: "About 3 g/kg salinity adds about +3.3 m/s (UNESCO slope 1.09–1.11 m/s per g/kg at 25–27 °C; NPL)."
2. **§0 K8 (L27).**
   - Old: "A 25 × 12.5 × 2 m pool (dimensions unverified) is a **shallow waveguide** with an aspect ratio of about 12:1."
   - New: "Stanford's Avery Competition pool (37 × 20 m, 3.35–4.27 m deep; gostanford.com) is a **shallow waveguide** with a length-to-depth ratio of about 10:1. The Maas dive well (25 × 17 m, 18 ft per an archived Stanford Athletics page) is the deepest option. The pool is chosen at G1."
3. **§0: add rows K16–K22.**
   - K16: pool geometry (as in edit 2).
   - K17: LL916 SPL curve, f_lo ≈ 560–660 Hz unequalized, 6.2 kHz notch, 1–18 m operating depth.
   - K18: Abshagen & Haak correction.
   - K19: DFN3 VB-DEMAND not in-domain.
   - K20: licence blockers (Demucs NC; WHAM! needs WSJ0; DEMAND conflict; DeepShip/ShipsEar none stated).
   - K21: real through-water speech has been recorded and enhanced before (Cao 2017; Zhang & Liu 2021; Kiel 2018; WTD 71 2014–2017), so the novelty claim is "openly released" plus "pretrained cross-architecture".
   - K22: alignment precedent (Nissen & Kuhnt-Matthé 2017).
4. **RQ6 (L151).**
   - Old: "(and for DFN3, whose paper reports VB-DEMAND results, unverified) and WHAM! for SepFormer-wham16k."
   - New: "VB-DEMAND for DFN3 is a *general benchmark*, not in-domain (DFN3 was trained on DNS4; it reports PESQ 3.17 on the unseen VB-DEMAND test). WHAM! for SepFormer-wham16k only if the team holds a WSJ0 licence; otherwise use LibriSpeech test-clean mixed with WHAM! noise (CC BY-NC, evaluation only) and label it as not in-domain."
5. **RQ7 (L157).** After "40.3% of non-speech inputs" add "(121,378 of 301,317 inferences, Whisper large-v3, greedy; 9.1% of those looped)".
6. **RQ9 (L177).**
   - Old: "One example is the Baltic harbour T60 above 2 s, cited by Kiel from Abshagen & Haak, DAGA 2024 (unverified)."
   - New: "One example is Abshagen & Haak (DAGA 2024, pp. 903–905): open water off Schönhagen, Kiel Bay, 22 m deep, bistatic 1–2 kHz sonar reverberation, a 20 dB fall within about 2 s. It reports no T60 and is not a harbour; Kiel DAGA 2026 cites it as 'T60 > 2 s'. Use it only as a qualitative long-decay example."
7. **RQ9 H9a (L171).**
   - Old: "A binomial GLMM on per-utterance word errors … `logit p_err ~ …`"
   - New: "A Poisson or negative-binomial GLMM on per-utterance error counts with offset log N_ref (insertions can make errors exceed N_ref): `log μ ~ offset(log N_ref) + SNR_speechband + C50 + T20 + system + system:SNR + system:C50 + (1|speaker) + (1|utterance) + (1|position:session)`. exp(β) is a WER ratio."
8. **§3 Band row (L188).**
   - Old: "c/(4H) ≈ 187 Hz in 2 m of water (computed, c = 1500). The LL916 is only ±10 dB from 500 Hz."
   - New: "c/(4H) ≈ 88–112 Hz in the Avery Competition pool and ≈ 68 Hz in an 18 ft dive well (computed, c = 1500). The projector sets f_lo. The NUWC LL916 SPL curve gives 160 dB at 500 Hz vs 179 dB at 1 kHz, so the unequalized −10 dB point is about 560–660 Hz (computed from plot readings)."
   - Also old: "expected about **300–500 Hz**". New: "expected about **500–650 Hz** unless E2 shows usable pre-emphasis".
9. **§4.2 row 12 (L235).**
   - Old: "if the LL916 transducer is about 9" in diameter like the sibling LL964 (retailer listing, … unverified for the LL916), then a ≈ 11.4 cm and ka = 1 at about 2.1 kHz (computed)."
   - New: "The LL916H is 9.25" (234.95 mm) in diameter per Lubell (Inyo's LL916C listing says 9.0"), so a ≈ 11.4–11.7 cm and ka = 1 at about 2.03–2.09 kHz (computed). Lubell's polar plots show near-omni behaviour at 1–2 kHz and multiple lobes from 5 kHz."
10. **§4.2 row 13.**
    - Old: "Source depth ≥ 0.5 m"
    - New: "Source depth ≥ 1 m for the LL916 (rated operating depth 1–18 m); ≥ 0.5 m for other projectors; UW30 ≤ 3 m (1.2 m recommended)"
11. **§4.2 Implementation (L248).**
    - Old: "(whether it accepts β < 0 is unverified)"
    - New: "(confirmed: only shape and type are checked and image amplitudes use `pow(beta, |n|)`, so the sign is preserved; pass a 6-element β, because a scalar is read as T60)"
12. **§5.1 (L315).**
    - Old: "824 utterances (unverified); WHAM! size unverified | VB-DEMAND CC BY 4.0 (unverified); WHAM! noise licence unverified, check before use"
    - New: "824 utterances; WHAM! tt 3,000 mixtures (9.0 h) | VB-DEMAND record CC BY 4.0, but it embeds DEMAND noise (licence listed both as CC BY-SA 3.0 and CC BY 4.0): evaluation only, never redistribute outputs. WHAM! noise CC BY-NC 4.0; the mixtures need licensed WSJ0 (LDC)."
13. **§6.1 Projector, preferred (L366).**
    - Old: "otherwise about $1,540–1,850 + $210 | … [Inyo Pools](https://www.inyopools.com/Products/19400029046141.htm). **Measure its dimensions** (directivity)."
    - New: "otherwise about $1,610 (Inyo LL916C, https://www.inyopools.com/Products/19400029046138.htm) + AC205C | Dimensions confirmed (9.25"); operating depth 1–18 m; SPL curve (https://lubell.com/wp-content/uploads/2024/04/LL916C_SPL.pdf) and polar plots (https://lubell.com/wp-content/uploads/2024/04/LL916C_polar_plot.pdf). The old …046141 URL is now the AC202 box."
14. **§6.1 UW30 (L367).** Append: "≤ 3.0 m depth, 1.2 m recommended; 25 Ω/20 W parallel resistor + 1.5 A fuse (datasheet confirmed); discontinuation still unverified (the EV product page redirects to the homepage)."
15. **§6.2 item 2.**
    - Old: "With A ≈ 23–46 m² that is **+0.4 to +3.4 dB above the 1 m level everywhere beyond R_c**"
    - New: "For the Avery Competition pool (about 1,174 m² of wall and floor, α = 0.05–0.1 guessed, so A ≈ 59–117 m²) that is **−3.7 to −0.7 dB re the 1 m level beyond R_c ≈ 1.1–1.5 m** (computed; Sabine estimate; K8 says the field is probably not diffuse)". The conclusion (swimmers are exposed to roughly the source level) is unchanged.
16. **§6.2 item 2 (L388).**
    - Old: "Navy guidance found: 160 dB for ≤ 100 s, trained divers only ([summary](https://sonicfield.org/…))."
    - New: "The secondary source for a Navy 160 dB limit now returns HTTP 410; cite no figure until a primary Navy source is found. Our ≤ 150 dB received-level rule and 'no swimmers while transmitting' stand as our own conservative limits."
17. **§7.1 (L562).**
    - Old: "**checkpoint not located (unverified)**; included only if found by W4"
    - New: "`kohei0209/tfgridnet_urgent25` (CC BY 4.0; ESPnet `SeparateSpeech.from_pretrained`); include in C1"
18. **§7.1 (L563).**
    - Old: "`nara_wpe` (fgnt; from memory, unverified)"
    - New: "`nara_wpe` (fgnt, MIT, `pip install nara_wpe`; use the NumPy implementation)"
19. **§7.1 (L567).**
    - Old: "(availability unverified)"
    - New: "(both confirmed on the README; WSJ0-REVERB recommends `--N 50 --snr 0.33`; prefer the VB-DMD checkpoint for anything released, because of WSJ0/EARS lineage)"
20. **§7.1 Demucs row.**
    - Old: "| 2 | time-domain U-Net | Demucs DNS48 | facebookresearch/denoiser | 16 kHz |"
    - New: "| 2 | time-domain | fine-tuned SepFormer replaces Demucs (facebookresearch/denoiser is CC-BY-NC 4.0 and archived; at most an NC-labelled zero-shot baseline) | SpeechBrain | 16 kHz |"
21. **§7.2.**
    - L577: replace "(DFN3 reports VB-DEMAND; unverified)" with "(reported in the DFN3 paper; not in-domain)".
    - L580: replace "~26M parameters per reviewer; unverified" with "26M (Subakan et al., arXiv 2010.13154)".
    - L581: replace "default N unverified" with "default N = 30".
    - L582: replace "(Conv1d support in `peft` unverified)" with "(peft supports Conv1d/2d/3d; GRU needs a custom wrapper)".
22. **§8.3 (L661).**
    - Old: "are recalled, unverified"
    - New: "are confirmed in `transcribe.py`; the CLI default is beam_size = best_of = 5 and the Python API default is greedy"
23. **§8.3 (L670).**
    - Old: "(expected about 2%, unverified; measured ourselves)"
    - New: "(Whisper paper: 2.7% for large/large-v2, greedy; no primary large-v3 figure found; measured ourselves)"
24. **§8.5 (L701).**
    - Old: "(Bisani & Ney 2004 bootstrap; DOI unverified)"
    - New: "(Bisani & Ney, ICASSP 2004, doi:10.1109/ICASSP.2004.1326009; speaker clustering justified by Liu & Peng, arXiv 1912.09508)"
25. **§12 Calendar (L796).** Add: "IRB office closed 12-21 → 01-01; IRB submission deadline for December review 11-03."
26. **§15 D6 (L905).**
    - Old: "Stanford cluster (availability unverified)"
    - New: "FarmShare (no sponsor, L40S) now; Sherlock and Marlowe (5,000 free GPU-h for a new PI) need a faculty sponsor"
27. **Appendix A (L920).** Old: "(coefficients recalled, unverified)". New: "(coefficients confirmed, NPL)".
28. **Appendix A (L944).**
    - Old: "(if a ≈ 11.4 cm, sibling-model dimensions, unverified) … ka = 1 at about 2.1 kHz"
    - New: "(a = 11.4–11.7 cm from Lubell / Inyo) … ka = 1 at about 2.03–2.09 kHz"
29. **Appendix A first-image table.** Add the note: "These delays are set by the surface image and hold for any depth ≥ 2 m with both ends at 1 m." Add the mid-depth rows from §7.1 of this update.
30. **Appendix A diffuse level.** Replace with the Avery values from edit 15.
31. **Reviewer note 9 (L987).**
    - Old: "The dimensions are taken from the LL964 sibling and flagged unverified."
    - New: "Dimensions confirmed from Lubell (9.25")."
32. **RQ2 strong-baselines list.**
    - Old: "URGENT TF-GridNet if a checkpoint is found"
    - New: "URGENT TF-GridNet (`kohei0209/tfgridnet_urgent25`)". Add "Kiel-recipe U-Net trained on uwsim (C10)".
33. **§13 F8.** Add the trigger "DAGA 2027 programme (March 2027) lists a Kiel underwater-speech paper with measured data".

**README.md**

1. **L143–144.**
   - Old: "The pool is a shallow waveguide, roughly 187 Hz cutoff at 2 m depth, so score speech from about 300-500 Hz up."
   - New: "The pool is a shallow waveguide (cutoff about 88–112 Hz in Stanford's 3.35–4.27 m Avery Competition pool, computed). The lower scoring edge is set by the projector: about 500–650 Hz for the LL916 unless pre-emphasis works."
2. **L101.** Old: "(LibriSpeech, VCTK, DNS clean)". New: "(LibriSpeech; VCTK after licence check)". DNS has mixed and share-alike terms.
3. **L106.** Old: "ambient, DeepShip)". New: "ambient; DeepShip internal-only)".
4. **L110 and L174.**
   - Old: "Demucs or SepFormer (time-domain)"
   - New: "SepFormer (time-domain; Demucs/denoiser is CC-BY-NC and archived)". Remove "Demucs" from the diagram list.
5. **"Closest work".** Add a bullet: "Real through-water speech has been recorded or enhanced before, classically and without release: IOA-CAS multichannel enhancement of sea-trial SSB voice (Cao et al. 2017), WTD 71/NATO sea-trial telephone speech with listening tests (Fricke & Nissen 2015; Nissen & Kuhnt-Matthé 2017), Kiel in-pool speech (DAGA 2018)." Extend the Kiel bullet: "Kiel has worked on underwater-telephony SE since 2016 (theses) and water-tested telephones since 2018."
6. **Claims to avoid.** Add: "First real through-water speech recordings / first enhancement of real through-water speech." "First to align with sync markers (Nissen 2017)." "Speech-to-text loses speaker identity (Titanic Calling restores it by voice cloning)."
7. **Risk table, "Why not text over a modem?"**
   - Old: "Argue latency, speaker identity, no vocabulary limits, graceful degradation"
   - New: "Argue latency, identity and prosody without enrolment, no vocabulary limits, no fluent hallucinated text after ASR errors, graceful degradation; cite the lineage (ICNNSP 2003, US8682244, RU2658477, Teledyne US12136424, Rong 2025, Titanic Calling)"
8. **Evaluation.**
   - Old: "**Secondary:** PESQ, and DNSMOS."
   - New: "**Secondary:** PESQ; non-intrusive DNSMOS P.835, NISQA and Distill-MOS, used only if they pass sanity checks; SQUIM validated against intrusive metrics on our data."
9. **Timeline.** Add "IRB submission 11-03; faculty sponsor named by 10-30".
10. **Open questions.**
    - Old: "Pool access schedule for the IR measurement and the benchmark recordings."
    - New: "Which pool (Avery Competition 3.35–4.27 m vs Maas dive well 18 ft) and access through stanfordaquatics@stanford.edu."

**PRIOR_ART.md**

1. **L68.**
   - Old: "The related Loughborough thesis and its tank and ANC details are **(unverified)**"
   - New: "Journal version: Woodward & Sari, IEEE JOE 21(2):181–192, 1996, doi 10.1109/48.486793. Thesis (Sari 1997) details remain (unverified)."
2. **L69.**
   - Old: "Hollien & Coleman et al., … **(unverified)**. Venue and author list are inconsistent across sources."
   - New: "Hollien & Coleman, JASA 46(1A Suppl.):115–116, 1969 (meeting abstract): Bugg Springs, 30 ft depth, divers 30 ft apart, mean 57–64% for four working units."
3. **L121.**
   - Old: "**(unverified)**. Sources disagree…"
   - New: "Training: simulated Rayleigh CIRs + BELLHOP; validation adds measured NOF/NCS (Watermark) and Wanlu Reservoir CIRs."
4. **L127, L131, L135, L140, L150, L156, L186, L241, L242, L243, L251.** Remove "(unverified)" and add venues:
   - *Less is More*: ASRU 2025.
   - RealMAN: NeurIPS 2024 D&B.
   - RemixIT: IEEE JSTSP.
   - AnyEnhance: TASLP 2025; public weights are a 45.68M challenge baseline.
   - TRAMBA: Proc. ACM IMWUT 8(4) 2024, doi 10.1145/3699757.
   - mmWave: arXiv only, "We Can Hear You with mmWave Radar!".
   - ESPnet-SE++: Interspeech 2022.
   - SpeechBrain 1.0: JMLR MLOSS.
   - ClearerVoice: Interspeech 2025.
5. **L120 MSRANet.** Keep "(unverified)" and add "(reference list cites Porter's Acoustics Toolbox)".
6. **L166.** Append: "A revised version appeared as arXiv 2605.19541 ('Optimising Neural Speech Codecs for 300bps Communication using Reinforcement Learning', same authors; v2 2026-07-24; no withdrawal notice). Cite only that version, as a digital neural-codec alternative."
7. **§1 Verdict.**
   - Old: "**No.** Five independent search angles…"
   - Add after it: "A third pass (2026-10-27: Chinese, German, Japanese, Russian, Korean, theses, patents, DTIC) found classical enhancement of real through-water speech (Cao 2017; Zhang & Liu 2021), unreleased sea-trial telephone speech with intelligibility tests (WTD 71/CMRE), and Kiel's lineage since 2016. None combines the three parts or releases data."
8. **§3.1.** Add subsections:
   - Kiel lineage (§3 of this update);
   - WTD 71 / NATO sea trials;
   - Chinese IOA / Xiamen / HEU analog and digital voice;
   - Japanese diver speech ML;
   - patents (speech-to-text, mouthpiece, breath gating);
   - historical intelligibility reports (Ranlet 1991, NEDU, USCG).
9. **§3.1 (L102).** Add to the Rong & Chen MAST bullet: "mentions an in-mask pre-processing stage against bubble, breathing and background noise".
10. **§6 DeepShip / ShipsEar.** Add: "no licence stated; distributed by email; internal use only; release IDs and scripts, not audio".

---

## 5. Models and licences

| Model | Checkpoint | fs | Licence (code / weights) | FT code | Params | Notes |
|---|---|---|---|---|---|---|
| DeepFilterNet3 | `models/DeepFilterNet3.zip`, `pip install deepfilternet` (https://github.com/Rikorose/DeepFilterNet) | 48 kHz only | MIT / Apache-2.0 (weights ship in the repo) | `df/train.py`; HDF5 data | ~2.1M (unverified) | CLI loads **DFN2 by default**, so pass DFN3. `[train]` extra is Linux only. |
| MetricGAN+ | https://huggingface.co/speechbrain/metricgan-plus-voicebank | 16 kHz | Apache-2.0 | SpeechBrain recipe | ~1–2M (unverified) | Recipe: batch 1, 750 epochs; per-step PESQ on CPU is the bottleneck |
| SepFormer wham16k-enhancement | https://huggingface.co/speechbrain/sepformer-wham16k-enhancement | 16 kHz | Apache-2.0 | Recipe in an unmerged PR | 26M | Weights descend from WSJ0; the card wrongly says WHAM! has reverb |
| SepFormer WHAMR! enhancement | https://huggingface.co/speechbrain/sepformer-whamr-enhancement | **8 kHz** | Apache-2.0 | as above | 26M | Output ≤ 4 kHz. `sepformer-whamr16k` is a separation model, not an enhancer. |
| Demucs (denoiser) | `dns48`/`dns64`/`master64` (https://github.com/facebookresearch/denoiser, archived 2023-10-31) | 16 kHz | **CC-BY-NC 4.0** | `train.py` | dns48 ~18.9M (unverified) | Fine-tuned weights inherit NC. **Drop from phase 2.** |
| CMGAN | https://github.com/ruizhecao96/CMGAN | 16 kHz | MIT | `src/train.py` | 1.83M | Memory driven by TF-grid length |
| MP-SENet | https://github.com/yxlu-0102/MP-SENet | 16 kHz | MIT | `train.py` | 2.26M | Use `batch_first=True` per the README; used by Jiang 2024 |
| SGMSE+ | https://github.com/sp-uhh/sgmse | 16 kHz (EARS 48 kHz) | MIT | `train.py` | ~65M (unverified) | N = 30; WSJ0 and EARS checkpoints carry NC/LDC data lineage |
| TF-GridNet (URGENT) | https://huggingface.co/kohei0209/tfgridnet_urgent25 ; https://huggingface.co/wyz/tfgridnet_for_urgent24 | 8–48 kHz (SFI) | ESPnet Apache-2.0 (from memory) / **weights CC BY 4.0** | ESPnet fork (https://github.com/kohei0209/espnet/tree/urgent2025/egs2/urgent25/enh1) | 8.5M | URGENT training mix may include NC corpora (unverified); fine for zero-shot |
| FRCRN / MossFormer2 / MossFormerGAN | https://huggingface.co/alibabasglab/FRCRN_SE_16K | 16 / 48 / 16 kHz | Apache-2.0 | ClearerVoice scripts | FRCRN 6.90M | Trained partly on private data |
| AnyEnhance | https://github.com/viewfinder-annn/anyenhance-v1-ccf-aatc | unstated | MIT | `trainer.py` | 45.68M + DAC + w2v-BERT 2.0 | Not the paper's model. **Defer.** |
| nara_wpe | https://github.com/fgnt/nara_wpe | any | MIT | n/a | 0 | Use the NumPy path; tune taps on v0-dev |
| Whisper large-v3 | https://huggingface.co/openai/whisper-large-v3 | 16 kHz, 128 mel | MIT (GitHub; the HF card says Apache-2.0) | `transformers` + `peft` | 1.55B | Hallucination and repetition; same normalizer for all systems |
| Whisper large-v3-turbo | https://huggingface.co/openai/whisper-large-v3-turbo | 16 kHz | MIT | same | 809M | Dev proxy only |
| Parakeet-TDT-0.6B-v2 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 | 16 kHz | CC BY 4.0 | NeMo (Linux) | ~600M | Outputs punctuation and case. **LibriLight in training: contamination risk on test-clean.** |
| OWSM v3.1 | https://huggingface.co/espnet/owsm_v3.1_ebf | 16 kHz | CC BY 4.0 | ESPnet | 1.02B | URGENT's ASR; comparability |

**Datasets**

| Dataset | Licence | Status |
|---|---|---|
| LibriSpeech | CC BY 4.0 | Release OK |
| VCTK 0.92 | CC BY 4.0 per the HF mirror; ODC-By per DNS; `license_text.txt` not opened | Likely OK; read the file first |
| DNS5 | Mixed, ShareAlike, ODbL | Skip |
| DEMAND | Zenodo conflict (CC BY-SA 3.0 vs CC BY 4.0) | Evaluation only |
| VB-DEMAND | CC BY 4.0 record, but embeds DEMAND | Evaluation only |
| WHAM!/WHAMR! | Noise CC BY-NC; mixtures need WSJ0 | Evaluation only, with WSJ0 |
| DeepShip | None stated | Internal only |
| ShipsEar | None stated | Internal only, with permission |

Sources: https://www.openslr.org/12/ , https://datashare.ed.ac.uk/handle/10283/3443 , https://github.com/microsoft/DNS-Challenge , https://zenodo.org/records/1227121 , https://arxiv.org/abs/1907.01160 , https://github.com/irfankamboh/DeepShip , https://underwaternoise.atlanttic.uvigo.es/

**Recommended redistributable phase-1 set**

- **Released data (CC BY 4.0):**
  - LibriSpeech (train-clean-100/360 for sim; dev-clean for v0-dev and real-train; test-clean for v0/v1-test);
  - our noise bank (T200 units A and B, ambient, bubbles, EMI);
  - our measured IRs and ESS sweeps;
  - VCTK only after `license_text.txt` is read;
  - TTS new-prompt audio only if the TTS licence allows releasing its outputs (D13).
- **Used, never redistributed:** VB-DEMAND (forgetting test); WHAM! 16k (only with WSJ0); DeepShip (ablation A5). Release IDs and scripts only.
- **Models whose fine-tuned weights we can release:**

  | Role | Model | Released-weight licence |
  |---|---|---|
  | FT primary | DFN3 | MIT/Apache |
  | FT | MetricGAN+ | Apache-2.0 |
  | Zero-shot | SepFormer-wham16k, SepFormer-WHAMR! | Apache-2.0 |
  | Zero-shot universal | TF-GridNet URGENT25 | CC BY 4.0 |
  | Classical | nara_wpe 1ch/4ch | MIT |
  | ASR primary + C8 LoRA | Whisper large-v3 | MIT |
  | ASR secondary | Parakeet-TDT-0.6B-v2 | CC BY 4.0 |
  | Baseline C10 (new) | Kiel-recipe U-Net | our own code (Apache-2.0) |

- **Phase 2:** CMGAN (MIT), MP-SENet (MIT), SGMSE+ (MIT; VB-DMD checkpoint), FRCRN (Apache-2.0).
- **Toolchain:** DFN3 training, NeMo and the ESPnet fork all need Linux. Train on FarmShare or Sherlock, or use WSL locally.

---

## 6. Stanford logistics

### 6.1 Facts

**Pools.** Sources: https://gostanford.com/facilities/avery-aquatic-center ; https://gostanford.com/news/2013/04/17/facilities-4 ; https://gostanford.com/sports/2013/4/17/208433735.aspx ; https://rec.stanford.edu/move/aquatics/stanford-campus-aquatics ; https://rec.stanford.edu/visit/rentals-and-reservations ; https://rec.stanford.edu/news/avery-rec-pool-reopening

- **Sizes and depths:**
  - Avery Competition: 37 × 20 m, 11–14 ft (3.35–4.27 m).
  - Belardi: 50 × 25 m, 11 ft at the centre.
  - Baker: 50 m × 25 yd, 4.5–8.5 ft.
  - Maas Diving Center: 25 × 17 m, 18 ft (archived Athletics page; the current page gives no depth).
  - Avery Recreation Pool (ARP) at AOERC: 50 m, depth not stated.
- **Water treatment** (salt or chlorine) is not published.
- **Booking:** Rec Rentals lists no pools. The Athletics (DAPER) pools need Aquatics staff, and probably a coach or facility manager (inferred). Contact stanfordaquatics@stanford.edu, (650) 498-0756.

**IRB**

- Exempt status is determined by the IRB, through eProtocol (https://eprotocol.stanford.edu/).
- An undergraduate Protocol Director needs a faculty Academic Sponsor (https://irb.stanford.edu/for-researchers/protocol-director-guidelines).
- **New studies must be submitted by 2026-11-03 for December review.** Average new-protocol turnaround is 38 days. The office is closed 2026-12-21 to 2027-01-01 (https://irb.stanford.edu/).
- LibriSpeech playback is likely Not Human Subjects Research. File an NHSR determination if unsure (https://irb.stanford.edu/for-researchers/do-i-need-irb-review).
- Live talkers for a public release: likely exempt Cat 2 or Cat 3, with the release named in the information sheet (my reading; confirm with the IRB). See https://irb.stanford.edu/for-researchers/irb-application-submission-guide/exempt-research
- Contacts (https://irb.stanford.edu/stanford-irb-office/contacts): irbnonmed@stanford.edu; Kateri Noble (Non-Medical IRB Manager), kateri.noble@stanford.edu.
- CITI requirement: unverified.

**Compute**

- **FarmShare:** any full-service SUNetID, no sponsor; 6 nodes × 4 L40S added Dec 2025; Slurm; not for sponsored research. https://docs.farmshare.stanford.edu/
- **Sherlock:** free, about 1,300 GPUs; the faculty PI emails srcc-support@stanford.edu to sponsor students; onboarding session 2026-11-04. https://www.sherlock.stanford.edu/docs/getting-started/
- **Marlowe:** DGX H100; the PI files a Basic Access form; **5,000 free GPU-h for one year** for new PIs, preemptible; from 2026-09-01 $0.25–0.30 per GPU-h. https://marlowe-research.stanford.edu/access/

**Hopkins Marine Station** (https://hopkinsmarinestation.stanford.edu/)

- Boating via oceans-marine-operations@stanford.edu. Level III boats need a WebBoater filing **at least 1 month ahead**.
- AAUS scientific diving.
- Science and Engineering team: Christopher Garsha (lead).
- Seawater and tank specs are not published.

**EH&S:** no electrical-near-water page could be retrieved. Contact form: https://ehs.stanford.edu/contact-us. GFCI and NEC 680 distances are general practice, not Stanford rules **(unverified)**.

**Calibrated hydrophones:** no Stanford holdings confirmed. Ask CCRMA (info@ccrma.stanford.edu) and Hopkins Science and Engineering.

### 6.2 Next actions

| # | Action | Owner | Due |
|---|---|---|---|
| L1 | Name a **faculty sponsor** for the IRB, Sherlock and Marlowe (new D16) | Lead (Ryota) | **2026-10-30** |
| L2 | Email stanfordaquatics@stanford.edu: pool choice (Avery Competition vs Maas well), 3 sessions (S1 W3, S2 W6, S3 W7), salt vs chlorine, LL916 or CAWSS availability, rules for powered gear and thrusters, drawings | SPK (Keiya) | **2026-10-28** |
| L3 | EH&S contact form: battery-only rig, thrusters, any mains gear near water; ask for written rules | SPK (Kaito) | 2026-10-30; answer by G1 (11-06) |
| L4 | IRB submission in eProtocol (live talkers plus crowd transcription), with NHSR determination request for LibriSpeech playback | Lead + faculty sponsor | **2026-11-03** (moved from W2) |
| L5 | FarmShare accounts; run the DFN3 Linux training smoke test | Lead / ML2 | 2026-10-30 |
| L6 | Sherlock sponsorship email (PI → srcc-support); attend onboarding | Faculty sponsor / Lead | email by 11-02; onboarding **11-04** |
| L7 | Marlowe Basic Access form (5k free GPU-h) | Faculty sponsor | 2026-11-06 (G1) |
| L8 | Calibrated reference request to Hopkins (Garsha via hmsinformation@lists.stanford.edu / oceans-marine-operations@stanford.edu) and CCRMA | SPK | send 10-30; answer by G1 11-06 |
| L9 | Second-site scouting: Hopkins pier or shallow cove (no boat licence) or tank time; Maas well as an in-campus geometry check | Lead | ask by 11-13; book by 03-15 (D14) |
| L10 | Fix the Inyo URL and price in RP §6.1 | Lead | 2026-10-30 |

---

## 7. Methods recommendations

**IR measurement (§6.4, E1–E3)**

1. **Use a synchronized ESS (Novák et al., JAES 2015, doi 10.17743/jaes.2015.0071; content unverified, 403).** Its harmonic IRs carry correct phase, so the Hammerstein fit (§4.2 #11) comes out of the same sweeps. The separate THD sweep in E2 can go.
2. **Deconvolve each hydrophone channel by the electrical loopback, not the digital sweep.** Apply drift correction before averaging (already in §6.4.5).
3. **Use pool depth to lengthen the time gate (computed, surface-image limited, c = 1500):**

   | Pool / placement | Range | First-image delay | Valid above (2–3/Δt) |
   |---|---|---|---|
   | Avery Competition, both ends at mid-depth 1.9 m | 1 m | 1.95 ms | about 1.0–1.5 kHz |
   | Avery Competition, both ends at mid-depth 1.9 m | 0.5 m | 2.22 ms | about 0.9–1.35 kHz |
   | Maas well, both ends at 2.75 m | 1 m | 3.05 ms | about 0.66–0.98 kHz |
   | Both ends at 1 m (current plan) | 1 m | 0.82 ms | about 2.4–3.6 kHz |

   This shrinks or closes the 2–3 kHz overlap problem between substitution and gated calibration.
4. **Below the gate limit,** add signal modelling of the truncated direct path (Robinson, Hayman, Harris, Beamiss, Meas. Sci. Technol. 2018, unverified). Cite IEC 60565-1:2020 and -2:2019 as the procedures followed, "adapted" (unverified).
5. **Low-frequency RS26 check:** run a multi-position spatial average of |H_rs26/H_ref|² over ≥ 10 positions of the co-located pair. Report the spread as an uncertainty-budget entry.
6. **Report per-octave IR SNR** from the late tail or a muted-amp sweep. Mark bands with inter-sweep coherence below 0.9 as time-variant and use single sweeps there.
7. **Projector-specific:**
   - Expect the 6.2 kHz notch in h_chain. Do not try to equalize a notch about 25–30 dB deep.
   - Test low-frequency pre-emphasis feasibility in E2 against THD; the target 160 dB source level leaves about 20 dB of drive headroom at 1 kHz (computed from 180 dB at 20 Vrms).
   - Kaulen & Schmidt's adaptive-ID transducer method is an alternative reference (unverified content).

**Decay analysis (H1a, §4.4)**

- Per octave band (500 Hz–8 kHz, plus 250 Hz where SNR allows):
  1. octave filter (zero-phase or time-reversed);
  2. Lundeby truncation (unverified citation);
  3. Schroeder EDC;
  4. DecayFitNet for 1/2/3 slopes plus noise (Götz et al., JASA 152(2), 2022, https://arxiv.org/abs/2205.09644, code https://github.com/georg-goetz/DecayFitNet);
  5. a confirmatory Bayesian fully-parameterized fit with BIC (Xiang et al., JASA 129(2):741–752, 2011, doi 10.1121/1.3518773).
- **Pre-register `two_slope`:** true iff ΔBIC (2 vs 1 slope) > 10 *and* each slope spans ≥ 10 dB above the noise term. These thresholds are my proposal, not from the papers.
- Report EDT, T1, T2, amplitude ratio and turning point. Do not report T20/T30 alone.
- Add the mixing time from echo density and kurtosis (Abel & Huang 2006, unverified) and the spatial variance of T as direct non-diffuseness evidence.
- Validate the simulator on multi-slope parameters, not on T60.

**Alignment (§6.5)**

- Matched-filter the LFM chirps at 96 kHz, with sub-sample peak refinement on the Hilbert envelope.
- Use the **first peak above −6 dB of the maximum** (leading edge), not argmax, because surface images can dominate.
- Fit t_rx against t_tx over all chirps in a session with a Huber regression. This gives a session-level ppm.
- Flag residuals > 20 µs as dropouts or clock glitches (pre-registered).
- Scale: 50 ppm over 60 s = 3 ms = 48 samples at 16 kHz (computed).
- Resample with soxr VHQ, or a fractional-delay time warp, then run the ±20 ms per-utterance lag search.
- Estimate drift per board against the **loopback (playback) clock**, never board vs board.
- Cross-check with paderwasn blind WACD SRO estimation (https://github.com/fgnt/paderwasn), and publish the residual lag vs time and the WACD-vs-chirp ppm difference.
- Cite Nissen & Kuhnt-Matthé 2017 as the underwater precedent for in-band markers.

**Metrics (§8.1, RQ8)**

- **Validate TorchAudio-SQUIM on v0-dev:** predicted vs computed ESTOI, PESQ and SI-SDR against R1 (https://arxiv.org/abs/2304.01448). This is cheap and gives a quantitative out-of-domain error on underwater audio.
- **Pre-registered sanity checks for every non-intrusive metric:**
  1. monotonic fall with distance and thruster-on on C0 (Kendall τ against distance and measured SNR);
  2. R0/R1 ≥ every enhanced output;
  3. generative outputs where WER rises while the MOS proxy rises are flagged.

  A metric that fails (1) or (2) is reported but never used to rank systems.
- **Report:** DNSMOS P.835 SIG/BAK/OVRL separately (https://arxiv.org/abs/2110.01763); NISQA's four dimensions, with Coloration as a projector and pool diagnostic (https://arxiv.org/abs/2104.09494); Distill-MOS (https://arxiv.org/abs/2502.05356).
- **Demote UTMOS** to "TTS-trained, out of domain" (https://arxiv.org/abs/2204.02152). MOS-Bench shows poor out-of-domain generalization (https://arxiv.org/abs/2411.03715).
- **Optional human anchor:** a closed-set MRT subset (ANSI S3.2, per Navy SBIR N152-110) next to free transcription. Historical underwater tests show closed-set scores run higher than open-set (DTIC AD0751662, unverified), so report both formats separately.

**Hallucination (§8.3, RQ7)**

- **Probe set:** thruster-only, pump-only and silence segments, passed through every system. Report the non-empty-output rate, words per minute of non-speech, and the BoH match rate.
- **Per-utterance flags, logged and never silently dropped:**
  - hyp/ref ratio > 2;
  - gzip compression ratio > 2.4;
  - avg_logprob < −1.0;
  - a 4-gram repeated ≥ 3 times;
  - a BoH match;
  - insertion runs ≥ 4 words outside the reference alignment (an operational stand-in for Koenecke's "phrase not in audio", https://arxiv.org/abs/2402.08021).
- **Build an underwater BoH** (Barański et al., https://arxiv.org/abs/2501.11378) on v0-dev only and freeze it before v0-test. Report WER three ways: raw, after BoH post-processing, and excluding flagged utterances.
- **Normalization:** `EnglishTextNormalizer` on both sides for every ASR, including Parakeet (https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py). Report S/D/I counts and the CER–WER gap.

**Pre-registration (§8.5, D15)**

- OSF "Preregistration" template; frozen analysis repo (commit hash plus Zenodo snapshot); embargo until the abstract is submitted.
- Frame v0-test as "data exist but not observed". State explicitly that v0-dev and S1 have been seen and v0-test has not.
- Include a **deviations table** (planned / actual / reason / impact) in the paper.
- Precedents: van Miltenburg et al., NAACL 2021 (https://arxiv.org/abs/2103.06944); Roettger 2021, doi 10.1515/lingvan-2019-0048 (content unverified).
- Claim "among the first pre-registered SE evaluations" only with "to our knowledge" (unverified).

**Statistics (§8.5)**

- **Primary CI:** paired, stratified cluster bootstrap. Clusters are speakers; strata are position × thruster. ΔWER is a ratio of sums on the same draw; 10,000 replicates; BCa intervals.
- Justify the speaker clustering with Liu & Peng (https://arxiv.org/abs/1912.09508; venue unverified).
- **For v1:** add a crossed speaker × position pigeonhole bootstrap (Owen 2007, unverified), because the same utterances repeat across positions.
- **GLMM (RQ9):** Poisson or negative binomial with offset log N_ref; random effects (1|speaker) + (1|utterance) + (1|position:session), plus a random system-by-position slope for RQ3. Fit with glmmTMB or lme4::glmer.nb, and check overdispersion.
- **Cross-checks:** MAPSSWE via NIST SCTK `sc_stats` (https://github.com/usnistgov/SCTK); Bisani & Ney probability of improvement; Holm across the exploratory grid; TOST with δ_WER from E21.

---

## 8. Plan deltas

### 8.1 Research questions

| RQ | Change | Reason |
|---|---|---|
| RQ1 / H1a | Replace "two-slope EDC RMS error lower by ≥ 1 dB and BIC lower" with "ΔBIC > 10 and each slope spans ≥ 10 dB above noise (DecayFitNet screen, Bayesian confirmation)". Add mixing time and spatial T variance as descriptive non-diffuseness evidence. Recompute the Sabine R_c and diffuse prediction for the chosen pool. | Precedent (Xiang 2011; Götz 2022); the old geometry was wrong. |
| RQ1 / H1b | Validate multi-slope parameters at held-out positions in addition to arrivals and DRR/C50. Name the Kiel "180° at every reflection" model as an extra competitor rung if cheap. | Stronger falsifiable test; differentiation from Kiel. |
| RQ2 | Add **C10 (Kiel-recipe U-Net on uwsim)** and **TF-GridNet URGENT25 zero-shot** to the strong baselines. H2-primary unchanged. | Closest prior work gets a direct row; the checkpoint now exists. |
| RQ2 / H2a | Unchanged. Cite Iwamoto 2022 plus Barański (confirmed). | — |
| RQ3 | Unchanged. The Demucs slot becomes fine-tuned SepFormer. | Licence. |
| RQ6 | Forgetting sets: MetricGAN+ on VB-DEMAND (in-domain); DFN3 on VB-DEMAND labelled "general, not in-domain"; SepFormer on WHAM! only with WSJ0, else LibriSpeech + WHAM! noise (evaluation only) labelled "not in-domain". Cite Mao 2026 as diving-domain evidence that LoRA ≈ full FT for Whisper. | DFN3 trained on DNS4; WHAM! needs WSJ0. |
| RQ7 | Two-sided, as before. Add the probe set, flags (d)/(e) and the frozen underwater BoH. | Methods §7. |
| RQ8 | Add SQUIM-vs-intrusive validation on our data and the sanity checks (1)–(3). Add Distill-MOS; demote UTMOS. Optional MRT closed-set anchor. | Cheap, novel and quantitative. |
| RQ9 | Binomial GLMM → Poisson/NegBin with offset. Transfer example: describe Abshagen & Haak correctly (not a harbour T60). | Insertions; factual correction. |

### 8.2 Experiments

| E | Change | Reason |
|---|---|---|
| E0 | Unchanged. Use the Lubell 1–18 m operating depth when planning the rig. | Spec. |
| E1 | Calibrate at mid-depth in the chosen pool (gate 1.95–3.05 ms); add signal-model extension; spatially averaged low-frequency RS26 ratio. | Deeper pool; methods. |
| E2 | Synchronized ESS at 3 levels gives h_chain plus Hammerstein kernels (separate THD sweep dropped). Check the 6.2 kHz notch and the f_lo of 500–650 Hz; test pre-emphasis. Source depth ≥ 1 m. | LL916 data; time saved. |
| E3 | Multi-slope pipeline, mixing time, per-band IR SNR, coherence-based time-variance flag. | H1a. |
| E5 | Validate multi-slope parameters; optional Kiel-style uniform −1 reflection competitor. | Differentiation. |
| E6 | Leading-edge chirp detection, Huber session drift fit, residual QC, WACD cross-check. | Alignment robustness. |
| E7 | Add TF-GridNet URGENT25 zero-shot; drop the Demucs zero-shot from the main table (NC footnote only if used). | Found; licence. |
| E8 | Add **C10** (DFN3-independent; 1 seed in the MVR if compute allows, else full paper). Implementation about 1 day; training a small U-Net on 84k examples is cheap (estimate). | Kiel baseline. |
| E13 | Fix the forgetting sets (§8.1 RQ6). | Licence; in-domain correctness. |
| E14 | Build the BoH on v0-dev; freeze before v0-test. | Methods. |
| E15 | Add SQUIM validation (can run in the MVR on v0-dev; zero GPU training). | RQ8. |
| E16 | Poisson/NegBin GLMM. | Statistics. |
| E17 | Add SSB carrier offset (pitch shift) and Doppler in uwsim; keep it offline and below 25 kHz. | Kiel lacks both. |
| E19 | Check whether the UWA Channel Library Green channel (fc 6 kHz, 4.5 ksym/s) spans enough of 0.5–8 kHz for baseband replay without frequency translation; if so, reinstate (ii). | The library covers about 75 Hz–25 kHz. |
| E22 | Optional MRT / ANSI S3.2 closed-set subset next to free transcription. | Navy and historical practice. |
| E23 | Candidates: Hopkins pier or cove (preferred, no boat licence), Maas dive well (different geometry, cheapest, still enclosed), Kiel recordings if shared (external). | Logistics; Kiel contact. |

### 8.3 Schedule

| When | Change |
|---|---|
| W1 (now → 11-01) | Add L1–L3, L5, L8, L10. Pick the pool question in the Aquatics email. Fold the §4.2 text edits in. |
| W2 | **IRB submission moves to 11-03** (December review deadline). Sherlock onboarding 11-04. G1 (11-06) adds: pool chosen; EH&S answer; faculty sponsor confirmed; FarmShare smoke test done. |
| W4 | Smoke tests on FarmShare L40S (Linux). C9 replay starts as planned. |
| W5 | Launch C10 alongside C2/C7/C8 after G3. |
| W6 | Build and freeze the BoH and the SQUIM validation on v0-dev. OSF pre-registration (by 12-06) adds the deviations-table template, the secondary-data framing and the multi-slope rule. |
| W7–W8 | Unchanged. Pool work must finish by 12-18. The IRB office is closed 12-21 → 01-01 (no IRB interaction expected). |
| 01-18 → 01-31 | **Contact Kiel** (Lead), after the 01-14 submission. |
| March 2027 | Check the DAGA 2027 programme for a Kiel real-data paper (F8 trigger); adjust full-paper positioning. |
| 07 → 09 2027 | ICASSP 2028 deadline expected around mid/late Sep 2027 (inferred from ICASSP 2027's 2026-09-23). |

### 8.4 Decisions D1–D15 (and new D16–D19)

| ID | Change | Reason |
|---|---|---|
| D1 Projector | Same rule (whatever is in hand at G1). Correct the LL916C price to $1,609.99 (Inyo …046138). Constraints: LL916 source depth ≥ 1 m; f_lo about 500–650 Hz; 6.2 kHz notch. UW30 ≤ 3 m. | Verified specs and plots. |
| D2 Reference | Unchanged (R1 primary). R1 includes the 6.2 kHz notch by construction. | — |
| D3 MVR models | DFN3 (×3) + MetricGAN+ fine-tuned; zero-shot SepFormer-wham16k, SepFormer-WHAMR!, TF-GridNet URGENT25, WPE; **add C10** if compute allows. Demucs removed. | Checkpoint found; licence; Kiel baseline. |
| D4 Interspeech 2027 | **Skip** (deadline 02-09 confirmed; concurrent-submission rule). Decide by 01-03 as planned. | Confirmed CFP. |
| D5 Dataset venue | Zenodo/HF at camera-ready under CC BY 4.0, with **only redistributable content** (§5). NeurIPS E&D only if v1 ≫ OCEANS. | Licence audit. |
| D6 Compute | FarmShare now (no sponsor); Sherlock plus Marlowe 5k free GPU-h through the faculty sponsor. ≥ 200 GPU-h by 11-06 unchanged. | Logistics facts. |
| D7 Authorship | Unchanged (W1 with Kaito and Keiya). Credit Kiel only by citation unless they share data. | — |
| D8 Code licence | Apache-2.0 unchanged. Released weights inherit upstream terms (Whisper MIT; Parakeet and TF-GridNet CC BY 4.0, which need attribution). | Licence audit. |
| D9 Live talkers | Yes if the IRB clears by 02-15, but **submit by 11-03**. Ask for an NHSR determination for LibriSpeech playback and crowd transcription. | 38-day average turnaround; December cutoff. |
| D10 Thrusters | Unchanged. Correct the BPF estimate to 154–180 Hz (if 3 blades). | Blue Robotics FAQ. |
| D11 Vehicle demo | Unchanged (benchmark only). | — |
| D12 Second ASR | Parakeet, with a **LibriLight contamination caveat** on test-clean; OWSM v3.1 as a fallback for URGENT comparability. | Licence and contamination. |
| D13 TTS voice | Add the criterion: its licence must allow redistributing generated audio. Decide by 01-18. | Release. |
| D14 Second site | Hopkins pier or cove via Marine Operations or Garsha (no Level III boat); Maas well as a geometry-change fallback; Kiel data as an external option. Ask by 11-13; book by 03-15. | Logistics; WebBoater lead time. |
| D15 Pre-registration | All confirmatory RQs by 12-06, plus the multi-slope rule, BoH freeze, metric sanity checks and a deviations table. | Methods. |
| **D16 (new) Faculty sponsor** | Name one by 10-30. They are needed for the IRB Academic Sponsor role, Sherlock and Marlowe. | Single point of failure for three resources. |
| **D17 (new) Pool** | Avery Competition (3.35–4.27 m) is the default; the Maas well (18 ft) gives the longest gates and the lowest cutoff. Decide at G1 (11-06). | Geometry drives gates, cutoff and Appendix A. |
| **D18 (new) Kiel contact** | After abstract submission, 01-18 → 01-31; small ask; no co-authorship request up front. | §3.3. |
| **D19 (new) Claim wording** | Lead claim: "to our knowledge, the first openly released through-water speech benchmark with aligned references, and the first study adapting pretrained SE architectures to it". Explicitly cite prior real-data classical work (Cao 2017, Zhang & Liu 2021, WTD 71, Kiel 2018) and the frogman DNN paper (ICICSP 2023). | New prior art. |

---

**Sources consulted in this pass** (in addition to those inline):
- https://api.openalex.org/works/doi:10.1109/icicsp59554.2023.10390720
- https://api.openalex.org/works/doi:10.1109/cisce52179.2021.9446036
- https://arxiv.org/abs/2605.19541
- https://arxiv.org/abs/2604.14654
- https://api.crossref.org/works/10.1109/48.486793
- https://arxiv.org/pdf/2501.11378
- https://archimer.ifremer.fr/doc/00000/1148/
- https://gostanford.com/sports/2013/4/17/208433735.aspx
- https://gostanford.com/news/2013/04/17/facilities-4
- https://pub.dega-akustik.de/DAS-DAGA_2025/konferenz-92.html?session=59
- https://www.swimmersguide.com/locations/5866