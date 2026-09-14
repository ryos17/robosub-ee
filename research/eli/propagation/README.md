# propagation — how sound and EM waves travel through water

Two figures, each one Bode-style pair (magnitude on top, phase below), with
both media overlaid: **fresh (pool) water is the bold solid line, seawater
the dotted one**.

| Figure | Wave |
|---|---|
| [figs/acoustic.png](figs/acoustic.png) | sound in water |
| [figs/em.png](figs/em.png) | electromagnetic waves in water |

PNG (200 dpi) and PDF (vector, for the paper) of each.

Regenerate:

```
conda activate oceans27      # or: ~/miniconda3/envs/oceans27/bin/python
python propagation.py
```

Everything plotted is the response of the **medium only**. A plane wave over
distance `r` has transfer function

```
H(f) = exp(-alpha(f) * r) * exp(-j * beta(f) * r)
```

Geometric spreading is frequency-flat, so it is deliberately excluded — it
would shift the magnitude curves down without changing their shape.

## What is on each axis

Both panels carry their own frequency axis, so neither has to be read by
eye-jumping across to the other.

Shaded bands mark the frequency ranges we care about: **human speech**
(green, 85 Hz – 8 kHz — voiced fundamentals through fricative energy), the
**RoboSub pinger** (red, 25–40 kHz) and the **Ping360** sonar (orange,
~750 kHz). The speech band is drawn on the EM figures too, at the same
frequencies, so the two wave types compare directly where a human voice
lives.

Every log axis carries a full graticule — major decade rules plus the
minor 2…9 divisions within each decade — so values can be read off the plot
rather than estimated.

**Magnitude panel.** Attenuation in dB/m (left) and, on the right axis, the
loss over a reference range: 25 m (the pool) for acoustics, 1 m for EM. The
right axis is a pure rescaling of the left (and skin depth is just 1/α), so
one scale serves both overlaid curves. **The magnitude axis is inverted** — more loss is downward — so each
curve falls off with frequency the way a lowpass response does. The axis
stays logarithmic, because absorption covers eight decades and a linear dB
axis would flatten everything below the top decade into the baseline. Acoustic figures also break the total into its physical terms; EM figures
overlay the good-conductor asymptote `alpha = sqrt(pi*f*mu*sigma)` and put
skin depth `delta = 1/alpha` on the right axis.

**Phase panel.** Phase constant `beta` as deg/m (left). On the right axis:

- acoustic → phase-velocity dispersion `dc/c0` in **ppm**, because the
  absolute phase velocity is flat to within parts per million and a plot of
  it would look like a straight line. Both acoustic figures share the same
  ppm scale so they compare directly.
- EM → absolute phase velocity in m/s, because it genuinely swings over four
  decades.

On the acoustic magnitude panel the two salt relaxations are also drawn as
thin grey lines. Fresh water's total absorption *is* the pure-water viscous
term, so those two curves are exactly the gap between the bold and dotted
lines.

The boxed note gives the excess phase over the reference range with the
non-dispersive bulk delay removed — that residual is the part that actually
distorts a chirp.

## Models

**Acoustic absorption** — Francois & Garrison (1982), *JASA* 72(6) 1879 and
72(3) 896. Three terms: boric-acid relaxation (~1 kHz), MgSO₄ relaxation
(~100 kHz), viscous absorption of pure water (∝ f²). Setting S = 0 collapses
both salt relaxations to exactly zero, which *is* the fresh-water case — no
separate model needed.

**Acoustic dispersion** — each Francois-Garrison relaxation is the imaginary
part of a Debye-relaxing sound speed,

```
1/c(w)^2 = 1/c_inf^2 * (1 + sum_i eps_i / (1 + j*w/w_i))
```

so the same fitted `(A_i, f_i)` that give the absorption also fix the phase
velocity, with no extra free parameters. The relaxation strengths come out at
O(1e-5), so it is evaluated in the linearised limit. The pure-water f² term
is the low-frequency tail of water's structural relaxation, folded in at an
assumed f_r = 1 GHz; its in-band dispersion scales as (f/f_r)² and is
therefore negligible, which is the point of including it.

**Electromagnetic** — single-Debye water permittivity plus ionic conduction,

```
eps_r(w) = eps_inf + (eps_s - eps_inf)/(1 + j*w*tau) - j*sigma/(w*eps_0)
```

covering both the conduction-dominated low-frequency regime and the ~17 GHz
dipole relaxation. Parameters at 20 °C: fresh ε_s = 80.2, τ = 9.4 ps;
seawater ε_s = 73.0, τ = 9.0 ps (salt lowers the static permittivity).

Conductivity is the one number to argue about. Seawater at 35 psu / 20 °C is
≈ 4.8 S/m. A chlorinated pool is 0.02–0.1 S/m depending on TDS; 0.05 S/m is
plotted. Attenuation goes as √σ in the conduction regime, so the whole
low-frequency part of the pool curve moves ±1.5× across that range.

## Numbers worth remembering

Acoustic, at the 25–40 kHz RoboSub pinger band (30 kHz):

| | absorption | over its reference range |
|---|---|---|
| fresh (pool) | 2.0e-4 dB/m | 0.005 dB over 25 m |
| seawater | 5.2e-3 dB/m | 5.2 dB over 1 km |

The two curves never cross, but they converge hard: seawater is ~235x
lossier than fresh water at 1 kHz, 17x at 100 kHz and only 1.4x at 1 MHz.
Both salt relaxations saturate above ~200 kHz while the viscous f2 term keeps
climbing in both media, so at Ping360 frequencies salinity barely matters.

Absorption is a non-issue in a pool — four orders of magnitude below the
reverberation and spreading that actually limit us. Dispersion is likewise
negligible: < 0.01 ppm in fresh water (excess phase 2e-11° over 25 m), and
even in seawater only 3.8 ppm at 30 kHz, i.e. 27° over a full kilometre.
**Sound in a pool is, to the precision we can measure, a pure delay.** That
is what licenses the image-source renderer in [../src/](../src/) to treat
each echo path as a delay-and-scale with a frequency-flat wall reflectivity.

Electromagnetic, attenuation and skin depth:

| f | fresh (σ = 0.05) | seawater (σ = 4.8) |
|---|---|---|
| 10 kHz | 0.39 dB/m, δ = 22 m | 3.8 dB/m, δ = 2.3 m |
| 1 MHz | 3.7 dB/m, δ = 2.4 m | 38 dB/m, δ = 0.23 m |
| 100 MHz | 9.6 dB/m, δ = 0.91 m | 362 dB/m, δ = 0.024 m |
| 2.4 GHz | 263 dB/m, δ = 0.033 m | 1110 dB/m, δ = 0.0078 m |

At speech frequencies (85 Hz – 8 kHz):

| f | sound, pool / 25 m | sound, sea / 1 km | EM δ, pool | EM δ, sea | EM v_p, sea |
|---|---|---|---|---|---|
| 85 Hz | 4e-8 dB | 0.0005 dB | 244 m | 25 m | 13 km/s |
| 1 kHz | 6e-6 dB | 0.05 dB | 71 m | 7.3 m | 46 km/s |
| 8 kHz | 3e-4 dB | 0.53 dB | 25 m | 2.6 m | 129 km/s |

Sound across the whole speech band is lossless for any range we will ever
use — half a dB over a kilometre of ocean at the top of the band. EM at
those same frequencies is the *least* bad it ever gets, and it is still a
few metres of skin depth in seawater. Note the phase velocity: 13 km/s at
85 Hz in seawater is only about nine times the speed of sound there, against
200000× in vacuum. That is how thoroughly conduction spoils EM in salt
water.

Two regimes are visible in the magnitude panels. Below the point where
`sigma = w*eps_0*eps_r` the loss is pure conduction and rises as √f; above
it, conduction loss saturates and the Debye relaxation takes over near
17 GHz. The crossover sits near 10 MHz in the pool and near 1 GHz in
seawater — which is why the pool curve has a flat shelf that the seawater
curve does not.

The phase panels are the reason EM is unusable for anything but the shortest
links. In the conduction regime v_p = √(2w/(mu*sigma)) — 10⁵ m/s in seawater
at 10 kHz, four orders below c, and varying as √f. A pulse does not survive
that; only very narrowband signalling does, which is exactly what real VLF
submarine comms is.

Practical consequence for us: acoustics for anything through the water,
optical (blue-green, 470–550 nm, the one transparent EM window) for
short-range high-rate links, RF only once an antenna breaches the surface.
