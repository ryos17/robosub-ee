# src — differentiable pool channel model + matched-field localizer

Week-2 code from the plan in `../notes/01_critique.md` (P1), written before any
pool data exists. Everything here is validated on synthetic data only.

```
poolsim/render.py    PoolRenderer: image-source shoebox pool, frequency-domain,
                     differentiable in hydrophone positions, source position,
                     pool size/depth, sound speed, per-wall reflection filters,
                     per-channel gain/delay, source response. transfer_batch()
                     vectorizes over candidate source positions for the localizer.
poolsim/losses.py    multi-resolution STFT loss, bandlimit() for coarse-to-fine.
poolsim/localize.py  matched-field grid search, coarse-to-fine in space AND band.
probe.py             writes the WAV the speaker plays (sync sweep, 50 ms chirps,
                     2 s exp sweep, pink burst).
tests/test_poolsim.py  the three checks below.
```

## Environment

```sh
conda env create -f environment.yaml   # env name: oceans27
conda activate oceans27
cd research/eli/src && python -m tests.test_poolsim
```

## What the tests show (synthetic, updated 2026-09-08)

1. **Renderer vs pyroomacoustics** (25 x 12.5 x 2 m box, order 1, flat walls,
   c = 1480): arrival times of the first 7 paths agree to 0 µs at 96 kHz,
   relative amplitudes within 0.01.
2. **Self-calibration fit**: truth model has hydrophones perturbed by 1 cm RMS,
   c = 1495, concrete-like walls r = 0.55 with 6 kHz roll-off, floor r = 0.4 /
   3 kHz, vertical images to order 3, speaker clock +120 ppm. Start from
   nominal (6 cm square, c = 1480, r = 0.5, Gaussian priors on surveyed
   geometry). Fit on 5 known poses, 30 ms early window with a 1 ms soft gate,
   50 ms 2-18 kHz chirps, coarse-to-fine band schedule (<5, <10, <18 kHz).
   Result: hydrophone error 1.62 -> 0.41 cm, c -> 1495.9, floor and most walls
   recovered (one wall 0.72 vs 0.55: under-observed with 5 poses). 500 Adam
   iterations, ~100 s CPU. Earlier simpler variant (order 1, r 0.85) reached
   0.05 cm.
   Lessons: without the band schedule the fit diverges (phase wraps at
   18 kHz once position error exceeds ~2 cm). Per-channel delay must be frozen
   when fitting positions on a single fixed jig (degenerate). The speaker
   clock ratio is not observable from one 50 ms chirp (6 us stretch): estimate
   it from inter-chirp arrival drift across the 90 s block (10 ms at 120 ppm)
   in the alignment stage, then freeze `clock_ppm`.
3. **Localization** of a held-out pose with the fitted model: 0.0 cm error
   (noise-free, same model family: consistency check, not a result). Nominal
   unfitted model: 71 cm. Levels: 0.25 m grid with full-band CIR *envelope*
   score (phase-free, smooth), then 5 cm @ <12 kHz and 1 cm @ <17.5 kHz
   coherent, top-5 seeds kept. A single full-band coherent coarse grid fails
   (88 cm, wrong depth): grid step must stay under half a wavelength of the
   band used at that level, or use the envelope score.

## Known gaps before real data (see ../notes/03_method_notes.md for the ranked list)

- Wall filter is a first-order low-pass with learnable gain and corner; swap
  for K serial SVF sections with min-phase from magnitude (Lee 2022, DiffRIR).
- Source is an omni magnitude curve; add time-domain FIR + downward directivity.
- No learned late-field residual; early window only.
- Fit protocol: multiple random inits + held-out pose selection not yet in.
- Localizer: add continuous Adam refine after the grid with accept-if-better.
- No SNR weighting; no Orin timing yet. transfer_batch is plain torch, so CUDA should be a
  `.to("cuda")` away; measure ms/frame and W with jtop.
