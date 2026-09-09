# What we should study: the real gaps, in plain terms (2026-09-09)

Each entry: the intuition, why it is a genuine gap (evidence), why it is
reasonable for us, and the risk. Ranked. Evidence comes from the ~400-record
IEEE sweep (notes 00-01), full-text reads (notes 03), and targeted checks on
2026-09-09 (Semantic Scholar, Google Scholar, arXiv; IEEE session had expired).
Paper ids are IEEE arnumbers unless a DOI/arXiv id is given.

Closest competitor group, read before writing anything: Singer / Weiss /
Wornell / Vishnu / Kari / Arikan (MIT, UIUC, NUS, Scripps). Tank echo
boundary estimation with a CNN (Arikan et al., JASA 156:65, 2024), semi-blind
localization (IEEE TSP 70:3090, 2022), a differentiable modular forward model
for localization with test-time adaptation (Kari, Zhuang, Singer,
arXiv:2503.23260, 2025, simulation only), source-environment adaptation for
data-driven ranging (arXiv:2503.23258, 2025). They are moving toward exactly
our territory, from the fixed-emitter, big-tank, offline side. Our edge is the
vehicle-mounted compact array, the real pool, the fitted model reused as both
matched-field replica and training simulator, the embedded GPU, and the sonar
cross-check. Cite them generously and move fast.

---

## G1. Wall echoes as extra hydrophones (multipath as a virtual array)

**Intuition.** Four hydrophones 6 cm apart can tell you *which way* a sound
came from, not *how far*. But every pool wall is a mirror: the echo off the
far wall is the same sound heard by a virtual copy of your array placed
metres away. If you can model the pool, the walls hand you a free array the
size of the pool, and distance (and depth) drop out. Everyone today throws
those echoes away as "multipath error".

**Evidence it is a gap.**
- Ocean work uses only the surface and bottom echo (Li 2016, 7485369;
  Cao 2022, 9662075; Qu 2025/26, 11195126, 11370860). A pool has six
  reflectors and nobody has used them for range with a vehicle array.
- The only 4-hydrophone pool localization paper (Misiurewicz et al., Sensors
  22:1162, 2022) treats reflections as a problem to detect and reject.
- The confined-environment UUV review (Watson et al., Sensors 20:6203, 2020)
  treats multipath purely as a nuisance and says there is "virtually no
  literature that independently documents the use and accuracy of such
  systems when used in complex or confined environments".
- In air this works (Ribeiro 2010, ICME 5583886; DiffRIR, CVPR 2024).
  Underwater tank versions estimate the *walls* from echoes with fixed,
  synchronized emitter and receivers (Arikan 2024 JASA), not the *source*
  from a moving compact array.

**Why reasonable for us.** Our synthetic pipeline already does it: fitted
model plus matched-field search finds a held-out pose to ~cm where an
unfitted model is off by 70 cm (src/README.md). The pool is the ideal case:
known box, hard walls, short ranges, chirp sources we control.

**Risk.** Pool walls are not perfect planes (gutters, lane ropes, steps);
Chitre 2023 warns tank multipath is "surprisingly complicated" and phase is
hypersensitive to cm geometry errors. Mitigation: early-window only,
envelope-first fitting, learnable wall offsets, sonar-measured geometry (G4).

---

## G2. Let the pool calibrate the array (joint self-calibration)

**Intuition.** Nobody knows where their hydrophones really are on a hull to
the millimetre, or the pool dimensions to the centimetre, or the exact sound
speed today. All three errors look the same in the data: arrivals slightly
early or late. If you play a known sound from a few surveyed spots, one
gradient-descent fit can solve for all of them at once, and only then is the
localizer trustworthy.

**Evidence it is a gap.**
- Zhou 2026 (11617090) shows sensor-position error is a non-negligible
  error source for underwater TDOA and treats it statistically, not by
  calibrating the array from data.
- Array self-calibration literature is radar/antenna (IEEE TAES 2024,
  DSP 2018); the one underwater hit (JASA 2025) self-localizes distributed
  platforms, not elements on one hull.
- Chitre 2023 (10224658) fits geometry error terms for *fixed* tank
  hydrophones and finds recovered position errors of a few cm matter; no one
  has done it for a vehicle array together with wall filters and sound speed.

**Why reasonable for us.** It is the same fit as G1 with more free
parameters; synthetic test recovers 1 cm hydrophone perturbations to 0.4 cm
from five poses. Costs nothing extra at the pool.

**Risk.** Some parameters are degenerate (per-channel delay vs position).
Handle by freezing delays on a rigid jig and putting Gaussian priors on the
survey (already in the code).

---

## G3. How much of the sim-to-real gap is just geometry you could have measured?

**Intuition.** Learned direction finders trained on textbook simulations
(free field, or a generic box) fall apart in a real reverberant tank. Nobody
has asked the obvious question: if you train the same small network on a
simulator *fitted to the actual pool* (from G1/G2), how much of the failure
disappears? That single number tells the field whether the problem is
"neural nets can't generalize" or "you simulated the wrong room".

**Evidence it is a gap.**
- DL-DOA papers train on Bellhop or free-field sims (Nie 2023, 10124137;
  Song 2025, 11104616, which reports collapse outside trained conditions;
  Chen 2026, 11621547). None fit the simulator to the venue.
- Sim-to-real work underwater does test-time adaptation or fine-tuning
  (Yan 2026, 11616801; Kari 2025 arXiv:2503.23258/23260) rather than fixing
  the simulator. Lee 2022 (9837411) names sim-to-real as the main limitation
  of a 200k-shoebox-trained model.
- DiffRIR (CVPR 2024) shows 6 real measurements beat neural baselines
  trained on 100, but in air and with no downstream localizer ablation.

**Why reasonable for us.** Same recordings as G1, one small net, three
training sets (free-field / generic box / fitted). Cheapest headline number
we can produce, and the one reviewers remember.

**Risk.** Needs enough real poses to make the comparison statistically
meaningful (aim for 50+ across two sessions) and honest held-out ranges.

---

## G4. Sonar sees the walls, hydrophones hear the walls: do they agree?

**Intuition.** The Ping360 gives a direct picture of where the walls are.
G1's model *infers* the walls from echoes alone. If the two agree to a few
cm, the passive model is credible, the sonar seeds the fit, and the sonar
range to two walls doubles as cheap pose ground truth for the vehicle.

**Evidence it is a gap.** Ping360 research is mapping/SLAM (Hansen & Birk
2023/24, 10239384, 10609976), pool activity/drowning detection (AquaScan
MobiCom 2025, SwimSonar IPSN 2022), artifact segmentation (Hasan 2025).
Nothing fuses a scanning sonar with a hydrophone array or uses it to
validate an acoustic channel model (investigator report, notes/02).

**Why reasonable for us.** One 360° sweep per pose, ~15 s, open-source
driver, Hough on the polar scan. Zero extra pool sessions.

**Risk.** 25° vertical beam smears surface/floor into the scan at short
range; tiled corners saturate. Use gain 0, short transmit, side walls only.

---

## G5. What does acoustic localization cost on the vehicle's GPU?

**Intuition.** Every Jetson-underwater paper is a camera paper. Nobody has
reported latency, throughput and watts for hydrophone-array localization
running on the vehicle itself, even though that is what decides whether a
sub can use it in a mission.

**Evidence it is a gap.** 25 Jetson-underwater records, all vision (notes
01/02); GPU acoustic arrays only topside (Mohebbi-Kalkhoran 2024, 10704599,
160 elements). Edge-benchmark methodology exists for vision only (Mohammadi
2023, 10459973).

**Why reasonable for us.** The localizer is plain batched torch; CUDA is a
`.to("cuda")`. Report ms/frame and W at MAXN and 15 W with jtop.

**Risk.** Weak as a standalone paper; strong as a required table in G1.

---

## G6. A sub that localizes itself from its own echoes (beaconless, confined space)

**Intuition.** A bat in a swimming pool. Put the speaker on the vehicle next
to the hydrophones, chirp, and listen to the pool answer. The direct path is
useless (known offset), but the six wall echoes arrive at times that depend
only on where the vehicle is in the known box. No beacon, no modem, no
external infrastructure: the environment itself is the beacon. That is
exactly what nuclear storage ponds, ship-hull inspection and flooded tunnels
need (Watson 2020 list), and what the RoboSub pool offers as a testbed.

**Evidence it is a gap.** EchoSLAM exists in air (Krekovic & Dokmanić,
ICASSP 2016) with a single source and microphones on the robot; underwater,
echoes have been used to estimate boundaries from fixed emitters (Arikan
2024) and for mapping with sonar, but vehicle self-localization from its own
broadband echoes with a compact hydrophone array is not found in the IEEE
sweep, Semantic Scholar, or Google Scholar. Needs one more IEEE check with
fresh cookies before we claim it in print.

**Why reasonable for us.** Same hardware (jig with hydrophones plus the
speaker), same fitted model (source position is now a known offset from the
array, so fewer unknowns), same localizer with the vehicle pose as the search
variable. Ground truth is the same surveyed poses.

**Risk.** Near-field direct path dominates and saturates; speaker directivity
matters more when it is on the array. Start with the speaker 0.5-1 m from
the hydrophones on the jig and a short high-band chirp.

---

## Ranking and what to actually do

| Rank | Gap | Role |
|---|---|---|
| 1 | G1 + G2 | Headline paper (this is the narrowed P1 in notes/01) |
| 2 | G3 | The one ablation table inside the headline paper |
| 3 | G4 | Validation section, plus free ground truth |
| 4 | G5 | Required table (latency/power on the Orin) |
| 5 | G6 | Second paper or Student Poster; strongest "AUV navigation" story, revisit after session 1 |

One paper, one sentence: *a compact hull-mounted hydrophone array can
localize a sound in 3D inside a reverberant pool by treating the walls as
extra sensors, using a differentiable pool model that calibrates itself from
a handful of surveyed chirps, verified against a scanning sonar and run in
real time on the vehicle's GPU.*

Framing that makes reviewers care: confined environments (nuclear ponds,
ship hulls, harbours, tanks) where GPS, DVL and long baselines are
unavailable, and where a 2020 review found no documented accuracy for any
acoustic system. Aberdeen local tracks: Offshore Infrastructure, Underwater
Security and Protection.

## Verified NOT gaps (do not pitch these as new)
- "First pool hydrophone dataset" (Woodward 2020 PLOS ONE; Lu 2024 IEEE
  DataPort). Dataset release is still worth doing, framed as vehicle-hull
  array with thruster noise and sonar geometry.
- Underwater LoRa/chirp modem from cheap hardware, including in pools
  (UWLoRa+ 10623112; Steinmetz 9905674; AquaHelper SenSys 2023).
- Self-noise removal by Noise2Noise on AUVs (10103424).
- Room/tank boundary estimation from echoes (Dokmanić 2013; Arikan 2024).
- Differentiable/gradient-based underwater localization in simulation
  (Kari & Singer 2023-25; Chitre 2023 in a tank with fixed hydrophones).
