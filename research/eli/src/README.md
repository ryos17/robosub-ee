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

## What the tests show (synthetic, 2026-09-07)

1. **Renderer vs pyroomacoustics** (25 x 12.5 x 2 m box, order 1, flat walls,
   c = 1480): arrival times of the first 7 paths agree to 0 µs at 96 kHz,
   relative amplitudes within 0.01.
2. **Self-calibration fit**: truth model has hydrophones perturbed by 1 cm RMS,
   c = 1495, walls r = 0.85 with 6 kHz roll-off, floor r = 0.6 / 3 kHz. Start
   from nominal (6 cm square, c = 1480, r = 0.9/0.7). Fit on 5 known poses,
   30 ms early window, 50 ms 2-18 kHz chirps, coarse-to-fine band schedule
   (<5 kHz, <10 kHz, full). Result: hydrophone error 1.62 -> 0.05 cm, c -> 1494.9,
   walls and floor recovered. 500 Adam iterations, ~36 s CPU.
   Without the band schedule the fit diverges (phase wraps at 18 kHz once
   position error exceeds ~2 cm). Per-channel delay must be frozen when fitting
   positions on a single fixed jig (degenerate).
3. **Localization** of a held-out pose with the fitted model: 0.0 cm error
   (noise-free, same model family, so this is a consistency check, not a
   result). Nominal unfitted model: 13 cm. Coarse-to-fine levels
   (0.25 m @ 2.5-4.5 kHz, 5 cm @ <12 kHz, 1 cm @ <17.5 kHz), top-5 seeds kept.
   A single full-band coarse grid fails (88 cm, wrong depth): grid step must
   stay under half a wavelength of the band used at that level.

## Known gaps before real data

- Wall filter is a first-order low-pass with learnable gain and corner; swap
  for SVF/biquad (Lee 2022, 9837411) if real walls need it.
- Order 1 only (direct + 6 first-order images). Pool corners and second-order
  vertical images are the first extension; keep the early window at 10-30 ms.
- No speaker clock-ratio parameter yet (speaker DAC vs H8 ADC). Add a learnable
  time-scale on the probe.
- Localizer uses coherent phase across channels; on real data add an
  envelope-only score for the coarse level and a noise floor / SNR weighting.
- No Orin timing yet. transfer_batch is plain torch, so CUDA should be a
  `.to("cuda")` away; measure ms/frame and W with jtop.
