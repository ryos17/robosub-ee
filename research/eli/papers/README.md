# papers/ — full-text extracts (pdftotext) of the papers that matter

Numeric prefixes are IEEE Xplore arnumbers (`https://ieeexplore.ieee.org/document/<n>`).
Extracts are for our reading; do not add PDFs of paywalled papers.

## Closest competitors (read before writing)
- `10944684_kari2025_differentiable_forward_model.txt` — Kari, Zhuang, Singer 2025. Differentiable modular forward model, gradient-based localization with test-time adaptation. Simulation only. Our model differs by being fit to a real pool with a real vehicle array.
- `10094978_arikan2023_icassp_environment_structure.txt` — Arikan et al. ICASSP 2023. Boundary estimation from echoes with a CNN; known emitter/receivers.
- `10887904_arikan2025_icassp_boundaries_dl.txt` — Arikan et al. ICASSP 2025. Number and location of boundaries without user prior.
- `arikan2024_jasa_tank_boundary_estimation.txt` — Arikan et al. JASA 2024. Same line, real water-tank data.

## Method sources (what we copy)
- `wang2024_diffrir_hearing_anything_anywhere.txt` — DiffRIR, CVPR 2024 (Stanford). Differentiable image-source fit from ~12 RIRs; per-surface learned reflectivity, source directivity, axial boosting. Our in-air template.
- `9837411_lee2022_differentiable_artificial_reverb.txt` — Lee 2022 TASLP. SVF-parameterized filters, multi-res STFT loss, frequency-sampling. Wall-filter recipe.
- `10224658_chitre2023_data_aided_ray_model.txt` — Li & Chitre 2023 JOE. Data-aided ray model with learnable geometry error terms, fit in a 2.5 m tank; phase hypersensitive to cm errors.
- `11617090_zhou2026_tdoa_sensor_position_uncertainty.txt` — Zhou 2026. Sensor-position error as a dominant TDOA error; accept-if-better safeguard.
- `7485369_li2016_single_hydrophone_multipath_range.txt` — Li 2016. Range from direct/surface-echo delay with one hydrophone. Our single-channel baseline.
- `9708257_more2021_tank_multipath_model.txt` — More 2021. Concrete tank multipath model; |R| ≈ 0.47 concrete/water prior.

## Precedents we must cite honestly
- `6404991_hodgkinson2012_linear_array_small_uuv_tank.txt` — OCEANS 2012. Linear array for small UUVs in pools/tanks, 8 m steel tank, <5 % slant range; multipath survived, not exploited.
- `11166125_belmonte2025_rov_positioning_pool_harbour.txt` — 2025. ROV detection/positioning with hydrophone arrays in a pool and a harbour.
- `11104616_song2025_convlstm_pulse_doa.txt` — Song 2025. ConvLSTM pulse DOA on a 3-element array; reports collapse outside trained conditions (motivates the ablation).
- `8096292_pinger_localization_sparse_AUV.txt` (+ pdf, demo of the fetch tool) — Kasetkasem 2017. Sparse-representation first-arrival pinger DOA baseline.

## Dropped directions (kept for the related-work paragraph)
- `10623112_uwlora_plus_pool_css.txt` — UWLoRa+ 2024. Chirp-SS modem in a pool; later paths exceed first path.
- `9905674_steinmetz2022_lora_underwater.txt` — LoRa underwater, sea trials.
- `11571741_wu2026_dewater_learned_demod.txt` — DeWater, INFOCOM 2026. Learned demod on commodity hardware.
- `11104740_rong2025_stt_tts_diver_comm.txt` — Rong OCEANS 2025. Speech-to-text over the acoustic link; comparison point for the Whisper IVC.
- `10103424_n2n_auv_selfnoise.txt` — Noise2Noise AUV self-noise suppression (why D4 is a section, not a paper).
- `9389338_koh2020_waven2n_multichannel_denoise.txt` — WaveN2N multichannel denoising, Global OCEANS 2020.
- `10609976_hansen2024_ping360_mapping.txt` — Hansen & Birk ICRA 2024. Open-source Ping360 mapping; driver/codebase reference.
