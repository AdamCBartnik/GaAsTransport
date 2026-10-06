# Validation summary

Comparisons with Chubenko et al. (2021) ("C21"). Published values were read off the figures by eye
unless they are stated in the text. All runs use the default `ModelAssumptions()` unless noted; the
scripts are in `validation/` and the figures go to `validation/out/`.

## Exact / text-stated numbers

| Quantity | This code | C21 |
|---|---|---|
| E_g at 5e17 / 1.7e18 / 1e19 cm⁻³ (Eq. 10) | 1.409 / 1.398 / 1.361 eV | 1.409 / 1.398 / 1.361 (Fig. 6 legend) |
| E_bb, W_bb at 1e19 (Eqs. 56–59) | 0.6939 eV, 9.947 nm | 0.694 eV, 9.947 nm (Fig. 16) |
| Initial ESP (Eq. 12), intrinsic | 50% at edge, 45.8% at 1.74 eV, 26% at 1.8 eV, 8.3% at 2.2 eV | 50% → ≈47% → steep drop (Fig. 6) |
| Initial ESP, Monte Carlo vs Eq. 12 | max deviation 2.8 standard errors over ~140 points (4 dopings, 2 spin rules) | — |

## Absorption (`validation/absorption.py`)

| Check | Result |
|---|---|
| Adachi (1989) parameters | all 14 GaAs values match Table I; n, k match the CC0 refractiveindex.info tabulation to 4 digits |
| Casey (1975) digitization | 6 p-type spectra (1.6e16–1.6e19 cm⁻³, 1.31–1.592 eV) from Figs. 6–8; cross-figure agreement for 1.2e18 and 2.2e17; the paper's stated trends are reproduced; bit-for-bit reproducible from the PDF |
| Composite seam α_Adachi/α_Casey (1.55 eV) | 1.17 (1.5e17), 1.18 (1e18), 1.37 (1e19); continuous, smooth blend |
| Absorption length at 1.45 eV | 1.59 / 1.51 / 2.11 μm (1.5e17 / 1e18 / 1e19) vs 1.18 μm for Adachi alone |

## Rates (Figs. 7, 8, 10–13): agree by eye at both dopings

* **Acoustic, POP abs/em, and impurity rates and τ_m.** Within plotting accuracy. Examples: impurity
  ≈2e14 → 6e13 s⁻¹ over 0–1 eV at 1e19; POP emission ≈3e12 s⁻¹ at 1e19 and ≈5e12 s⁻¹ at 1.5e17.
* **Intervalley rates.** Γ→L and Γ→X absorption/emission match **only with the DOS factor in the
  numerator** (at 1 eV: 7.4e13 Γ→L emission, 1.2e14 Γ→X emission). The as-printed Eq. 33 is up to 2.6× too low.
* **Spin relaxation.** EY per mechanism, DP per mechanism, the DP total (Matthiessen), BAP, and the
  total spin relaxation rate all agree, as does τ_s(E) for 1.5e17, 1.5e18, and 1e19 (Fig. 13).
* **POP τ_m.** C21 Eq. 31 agrees with the exact Eq. 22 to machine precision for a parabolic band, and to 2–10% with nonparabolicity.

## Transport

**Drift velocity vs field** (Fig. 9; Stage C, 1000 electrons, 8 ps, Maxwellian start; units 1e5 m/s):

| F (1e5 V/m) | 0.5 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 14 | C21 simulation (approx.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5e17 | 0.17 | 0.45 | 0.88 | 1.31 | **1.46** | 1.32 | 1.17 | 1.05 | 0.96 | peak ≈1.45 at 4, ≈1.0 at 14 |
| 1.5e18 | 0.07 | 0.12 | 0.30 | 0.43 | 0.56 | 0.93 | 1.05 | 1.08 | 0.98 | ≈1.0 plateau at 7–10 |
| 1e19 | 0.01 | 0.04 | 0.07 | 0.09 | 0.15 | 0.22 | 0.28 | 0.39 | 0.58 | slow rise to ≈0.6 at 14 |

**Spin relaxation time from the internal ESP decay** (Fig. 14 / Eq. 55; hν = 1.65 eV, reflecting
surface, 3000 electrons, 300 ps). τ_s is the C21 definition, i.e. the time for ESP to fall from ESP0
to ESP0/e. The statistical uncertainty is roughly ±10–15 ps.

| p (cm⁻³) | FD Pauli (default) | C21 step Pauli (Eq. 44) | C21 |
|---|---|---|---|
| 1.5e17 | 129 ps | 136 ps | 110 ps |
| 1.5e18 | 92 ps | 97 ps | 92 ps |
| 1e19 | 76 ps | 88 ps | 77 ps |

A weighted exponential fit of the late-time decay (30–300 ps) gives longer times: 141, 79, and
93 ps with FD Pauli blocking. C21's statement that the 1/e time is their definition matters for
this comparison.

## Internal consistency checks (unit tests)

* **Bands.** E(k) ↔ k(E), v = ħ⁻¹dE/dk, and the DOS from state counting all agree to ≤1e-6.
* **Rates.** Every closed-form rate equals numerical golden-rule integration with the same matrix
  element: acoustic, POP (screened), impurity, and intervalley rates and τ_m.
* **POP.** Absorption and emission satisfy detailed balance.
* **Stage A thermalization.** Energy steps are exactly ±ħω0. The comb populations match g(E)e^{−E/kT},
  and the early cooling rate matches −ħω0(W_em − W_abs) to 1%.
* **Spin flips.** With a constant τ_s, Eq. 54 gives an exponential ESP decay.
* **Electron–hole scattering.** The accepted rate equals the Born rate computed from velocities (1%).
  Momentum and energy are conserved in every collision. Thermalization to the lattice temperature
  (detailed balance) is tested; see `docs/MODEL_ASSUMPTIONS.md` §3.
* **Flight modes.** Direct W_total(E) flights and the self-scattering mode give the same cooling curves.
* **Bookkeeping.** In L/X the spin is frozen and the per-valley residence times sum to the elapsed time.

## Example: surface arrivals without band bending

`examples/surface_arrivals.py`, p = 1e19, hν = 1.60 eV (Adachi l = 616 nm, before the Casey default was adopted), 3000 electrons, 370 ps:
* **ESP:** 0.479 at excitation, 0.283 at arrival.
* **Arrival:** 55% of electrons reach z = 0; the median arrival time is 41 ps.
* **Upper valleys:** 7% of arrivals ever visited L (none visited X), spending 0.6% of their time there,
  and 0.2% arrive in L. At this photon energy the frozen-spin assumption for L/X affects few electrons.

## Stage D: band bending (`validation/stage_d_band_bending.py`)

| Check | Result |
|---|---|
| Fig. 16 (E_bb, W_bb vs p) | 0.6939 eV / 9.947 nm at 1e19; C21 quotes 0.694 / 9.947 |
| Fig. 17 (E_C(z), E_z(z) at 1e19) | reproduced (peak field 1.40e8 V/m) |
| Integrator, ballistic crossing of the band bending | max energy error 0.14 meV (1e19, default 0.25 fs step); C21's 1 fs step: 2.2 meV, always positive; error ∝ h² |
| Closed slab, reflecting walls | Boltzmann density exp(−E_C(z)/kT) stationary to within statistics, with and without band bending |
| Fig. 20 (depth distribution of electrons still inside at 0/100/300 ps) | qualitatively as in C21: slow drainage at 1e19 ("reduced diffusion"), fast at 5e17 |

**Surface-arrival summary** (Stage C bulk + C21 band bending, absorbing surface, 300 ps, N = 3000, default
assumptions). E_arr is the kinetic energy above the local valley minimum at z = 0.

| p (cm⁻³) | hν (eV) | arrived | ESP0 → ESP(arrival) | ⟨E_arr⟩ (meV) | median t (ps) | visited L/X | arrive in L/X | ⟨t in L/X⟩, visitors | share of all time in L/X |
|---|---|---|---|---|---|---|---|---|---|
| 1e19 | 1.45 | 23% | 0.52 → 0.25 | 671 | 64 | 25% | 21% | 0.18 ps | 0.05% |
| 1e19 | 1.60 | 53% | 0.49 → 0.26 | 672 | 35 | 28% | 22% | 0.23 ps | 0.09% |
| 1e19 | 1.75 | 57% | 0.27 → 0.16 | 678 | 31 | 48% | 25% | 0.60 ps | 0.43% |
| 1e19 | 1.90 | 69% | 0.13 → 0.06 | 688 | 18 | 66% | 27% | 0.75 ps | 0.93% |
| 5e17 | 1.45 | 59% | 0.48 → 0.34 | 471 | 34 | 57% | 52% | 0.11 ps | 0.09% |
| 5e17 | 1.60 | 80% | 0.50 → 0.37 | 483 | 12 | 56% | 52% | 0.15 ps | 0.20% |
| 5e17 | 1.75 | 82% | 0.34 → 0.19 | 483 | 5.8 | 72% | 57% | 0.70 ps | 1.4% |
| 5e17 | 1.90 | 87% | 0.13 → 0.07 | 482 | 3.9 | 82% | 62% | 1.08 ps | 3.2% |

Observations:
* **Arrival valley.** Electrons accelerated by the band bending (E_bb = 0.69 eV at 1e19, 0.63 eV at 5e17)
  exceed the Γ–L separation (0.28 eV), so a large fraction arrive at the surface in an L valley: about
  22–27% at 1e19, where the region is only 10 nm thick and largely crossed ballistically, and 52–62% at
  5e17, where it is 70 nm. The arrival valley will matter for the surface model (C21 allows emission only
  from Γ and some X valleys on (100)).
* **Γ arrivals at 1e19** come in nearly ballistically, with ⟨E⟩ ≈ E_bb − 20 meV.
* **Time spent in L/X** is small for near-gap excitation (≤ 0.2% of transport time), so the frozen-spin
  assumption for L/X is unimportant there. At 1.75–1.90 eV it reaches 0.4–3%. The ESP difference between
  arrivals that visited L/X and those that did not is mostly a selection effect (hot hh-band, spin +1
  electrons transfer more often), not relaxation.
* **Run time:** about 20–25 min per 1e19 run (3000 electrons, 300 ps), dominated by impurity and e–h events.

## Depletion-region scattering: local mobile holes (`validation/depletion.py`)

**Profiles versus z** (`depletion_profiles.png`; p = 1e19 and 5e17 cm⁻³, C21 band bending):
* p(z)/p_bulk falls to 3e-12 (1e19) and 3e-11 (5e17) at the surface.
* The accepted e–h rate follows p(z), dropping by about 8 orders of magnitude.
* β(z) is unchanged at 1e19 with the default cap, because the bulk screening length (4.7 nm) already
  exceeds the acceptor spacing (2.9 nm). At 5e17 the screening length grows from 6.1 to 7.8 nm. With
  the W_bb cap it can reach 9.9 nm and 42 nm.
* The ionized-impurity rate is unchanged at 1e19 with the default cap and up to 2× higher at 5e17.
  With the W_bb cap it is up to 6× (1e19) and 30× (5e17) higher at low energy.
* τ_BAP grows by many orders of magnitude toward the surface.

**Targeted transit test** (`depletion.py inject`): 20 000 thermal Γ electrons with spin +1 start at
z = W_bb and are followed for 10 ps. σ(ESP) ≈ 0.001, σ(⟨E⟩) ≈ 0.7 meV, σ(fraction) ≈ 0.3%.

| p (cm⁻³) | variant | Γ / L / X arrivals | ESP | ⟨E_Γ⟩ (meV) | ⟨cos θ⟩_Γ | ⟨t⟩, direct transits (fs) | e–h events / arrival | ii events / arrival |
|---|---|---|---|---|---|---|---|---|
| 1e19 | bulk (C21-like) | 81.0 / 10.8 / 8.3% | 0.9929 | 728.8 | 0.963 | 94 | 27.8 | 44.6 |
| 1e19 | local, cap = impurity spacing | 80.6 / 10.8 / 8.6% | 0.9919 | 734.7 | 0.970 | 95 | 26.7 | 44.2 |
| 1e19 | local, cap = W_bb | 80.2 / 11.0 / 8.8% | 0.9943 | 733.5 | 0.961 | 93 | 27.7 | 50.6 |
| 5e17 | bulk | 52.7 / 30.5 / 16.8% | 0.9732 | 659.3 | 0.955 | 222 | 8.8 | 10.4 |
| 5e17 | local, cap = impurity spacing | 52.7 / 31.0 / 16.4% | 0.9722 | 659.5 | 0.956 | 222 | 8.4 | 11.4 |
| 5e17 | local, cap = W_bb | 52.2 / 31.2 / 16.7% | 0.9842 | 656.5 | 0.948 | 222 | 8.3 | 48.9 |

**Full photoemission comparison** (`depletion.py compare`, 8 cases × 3 variants, N = 3000 each, common
random numbers; `depletion_compare.txt`, `depletion_compare_p*.png`). All differences are within
statistics (σ ≈ 1.5% for fractions, σ(ESP) ≈ 0.04 for ~700 arrivals at 1e19) except one: at 1e19 the
Γ arrival energy is 2–9 meV higher with the local model, consistently, and the bulk model's
600–700 meV tail of electrons that collided with holes inside the band bending disappears.

**Conclusions**
* For the C21 band bending the depletion region is crossed in about 0.1 ps (1e19) to 0.2 ps (5e17).
  With bulk rates an electron makes ≈ 1 (1e19) or ≈ 0.4 (5e17) e–h collision on the way. The local model
  removes these, so Γ arrivals are about 6 meV hotter at 1e19, with a narrower distribution and slightly
  more forward-directed.
* Valley fractions, response times and ESP are essentially unchanged with the default cap. Spin loss
  in the transit is below 1% (1e19) to 3% (5e17), and BAP is too slow (≥ 100 ps) to matter over 0.1 ps
  even in the bulk.
* The screening cap is the most consequential remaining choice at moderate doping. With the W_bb cap,
  the 5e17 transit has 5× more impurity events, a slightly broader angular distribution, and *less* spin
  loss (0.984 vs 0.972 ESP). More momentum scattering suppresses D'yakonov–Perel relaxation (motional
  narrowing).

## Stage E: C21 Fig. 18 end to end (`validation/stage_e_fig18.py`)

p = 1e19 cm⁻³, hν = 1.45–2.20 eV in 0.05 eV steps (C21's grid), χ = 0.64, 0.67, 0.70, 0.73 eV,
370 ps, 10⁵ electrons per photon energy (as in C21), Adachi absorption (as C21 Fig. 3), fast GPU
engine. Every χ and surface variant continues the same first-arrival ensemble. C21 curves:
`validation/reference/c21_fig18.csv`, read exactly from the vector graphics of the PDF
(`tools/extract_c21_fig18.py`). Full tables and plots: `validation/out/stageE/fig18_summary_*.txt`,
`fig18_qe_esp_*.png`, `fig18_details_*.png`. The numbers below are from runs after the surface-crossing
integrator fix (`docs/MODEL_ASSUMPTIONS.md` 6d); compared with runs before it, the QE ratios changed by
≤ 0.002 and the ESP offsets by ≤ 0.1 % abs. (emission per unit time is insensitive to the k_z drift
because T ∝ k_z and the return rate ∝ 1/k_z).

**Engines agree over the full 370 ps** (`validation/stage_e_engines.py`): reference-engine first
arrivals (6000–8000 electrons per hν) vs the GPU engine (10⁵): arrival fraction, arrival-time
median and 90th percentile, valley fractions, ESP and energy at arrival; 48 z-scores, rms 0.93,
max 2.7.

**QE and ESP vs C21** (mean over the 16 photon energies; σ(QE) ≈ 0.05–0.1 % abs., σ(ESP) ≈ 1 % abs.):

| depletion scattering | matching mass | QE ours/C21 (χ = 0.64 / 0.67 / 0.70 / 0.73) | ESP ours − C21, abs. % |
|---|---|---|---|
| bulk (C21) | band edge m* | 0.97 / 0.99 / 1.02 / 1.07 | −0.2 / −0.1 / −1.8 / −3.5 |
| bulk (C21) | velocity m*(1+2αE) | 1.25 / 1.32 / 1.40 / 1.50 | −0.4 / −1.0 / −1.6 / −4.4 |
| local | band edge m* | 1.36 / 1.39 / 1.42 / 1.46 | +0.5 / +0.1 / −0.6 / −2.7 |
| local | velocity | 1.65 / 1.74 / 1.84 / 1.93 | +0.3 / −0.2 / −1.3 / −3.4 |

* With bulk rates in the depletion region (as C21 state) and the band-edge mass in the transfer
  matrix (C21's sentence read literally), the QE curves are reproduced to a few % at all χ and hν
  without any adjustment (rms 0.2–0.5 % abs.). The largest relative deviation is the near-edge
  point 1.45 eV (+15 % at χ = 0.67), where the absorption data differ (C21: Adachi fit to Zollner;
  here Adachi 1989).
* ESP follows C21's shape (maximum ≈ 26 % at 1.55–1.65 eV, drop when the split-off band is excited at
  1.7–1.75 eV, ≈ 5–8 % at 2.2 eV) and agrees for χ ≤ 0.70. At χ = 0.73 ours is 3–8 % lower than C21's
  around 1.5–1.9 eV: C21's ESP rises with χ more strongly than ours. Not resolved.
* Surface valleys (bulk, band-edge mass, χ = 0.67): first arrivals Γ/L/X = 78/12/10 % (1.45 eV) to
  66/19/15 % (2.2 eV); emitted electrons come from Γ (77–91 %) and X[001] (9–23 %); L never emits.
  Encounters (median, 90th percentile): 2 (5) before emission, 4–5 (10–13) before trapping.
  MTE 12–30 meV, mean vacuum kinetic energy 94–162 meV.
* **local vs bulk** (same photoexcited ensembles, band-edge mass, χ = 0.67): first arrivals are
  unchanged, but QE rises by +1.6 % abs. (1.45 eV) to +5.3 % (2.2 eV), i.e. +34–46 % relative; ESP is
  unchanged within errors; X-valley emission roughly doubles (9 → 17 % at 1.45 eV, 23 → 33 % at
  2.2 eV); trapped electrons make twice as many encounters (median 8–10); MTE +4–8 meV. Reflected
  electrons re-enter the depletion region; with bulk hole densities there they lose energy to e–h
  scattering and fall below the vacuum level within a few encounters, with depleted holes they keep
  it and try again. The depletion treatment therefore matters for QE far more than the first-arrival
  diagnostics of Stage D′ suggested.
* **Response time / near-edge** (bulk, band-edge mass, χ = 0.67): only 21 % (1.45 eV) to 32 %
  (1.70 eV) of the QE is emitted within 10 ps; those early electrons have ESP 38–49 % against 21–26 %
  for all electrons emitted up to 370 ps.

### The χ = 0.73 ESP discrepancy (`validation/stage_e_chi_esp.py`, variant `bulk_step`)

C21's ESP rises steeply with χ (at 1.6 eV: 26 % for χ ≤ 0.70, 33 % for χ = 0.73); ours barely does
(25–26 %). Findings, bulk mode, band-edge mass:

* **Mechanism (energy/time selection by the surface).** In our model ESP depends strongly on the
  emission time: ≈ 43–50 % for electrons emitted in the first 3 ps at 1.6 eV, ≈ 5 % after 100 ps.
  ESP can therefore rise with χ only if a higher vacuum level removes the late, thermalized electrons.
  It does so only weakly: the share of QE emitted within 10 ps grows from 0.28 (χ = 0.64) to 0.32
  (χ = 0.73) at 1.6 eV and from 0.35 to 0.44 at 1.8 eV. A thermalized electron reaches the surface with
  E_tot ≈ E_bb + 3kT/2 ≈ 0.73 eV, i.e. right at χ = 0.73, so about half of them can still emit.
* **How cold the thermalized electrons are matters.** C21's Pauli step rule for e–h scattering (Eq. 44)
  does not preserve detailed balance and thermalizes electrons at ⟨E⟩/kT = 1.19 instead of 1.50
  (`eh_detailed_balance.txt`). With it (`bulk_step`) the selection is stronger (1.8 eV: within-10-ps
  share 0.34 → 0.49 from χ = 0.64 to 0.73) and the χ = 0.73 ESP rises by 4–6 % at 1.65–1.75 eV:
  mean deviation from C21 −3.6 → −2.5 %, rms 5.6 → 4.0 %. χ = 0.64–0.67 are unchanged within errors.
  But QE falls to 0.87 → 0.81 of C21's (χ = 0.64 → 0.73), so the step rule alone does not reproduce
  C21; it accounts for roughly half of the χ = 0.73 ESP gap.
* Not resolved. Candidates not checked: other differences in how much energy electrons lose in the
  band-bending region before the surface, C21's 1 fs Verlet step (≈ 2 meV energy error per crossing),
  and the exact absorption depth profile. Fermi–Dirac blocking stays the default (it is the
  physically consistent choice); `bulk_step` is kept as a documented sensitivity variant.

## Stage F: minimal finite GaAs layer (`tests/test_thin_film.py`, `validation/stage_f_thin_film.py`)

**Tests:** back reflection with R_back = 1 conserves energy (round-off for field-free flights;
Verlet tolerance < 0.05 meV inside the field), flips k_z and leaves k_par, valley and spin unchanged;
a closed slab with many back reflections shows no heating, cooling or density drift (both engines);
R_back = 0 terminates every electron reaching z = d exactly once; R_back = 0.7 is reproduced as the
reflected fraction of > 5000 encounters; the generated depths follow the truncated exponential (KS);
d = 20 µm reproduces the semi-infinite result; the d < 3 W_bb warning fires.

**Thickness study** (p = 1e19, χ = 0.67 eV, C21 baseline physics, Adachi absorption, 10⁵ electrons per
point, GPU; `validation/out/stageF/thin_film_summary.txt`, `thin_film.png`):

| hν | d | R_back | QE % | ESP % | median t_emit (ps) | lost to substrate |
|---|---|---|---|---|---|---|
| 1.55 | 100 nm | 0 / 0.5 / 1 | 0.86 / 0.95 / 1.65 | 47.6 / 47.1 / 44.0 | 0.3 / 0.4 / 1.8 | 0.47 / 0.41 / 0 |
| 1.55 | 200 nm | 0 / 0.5 / 1 | 1.64 / 1.74 / 3.10 | 46.4 / 45.1 / 39.9 | 1.7 / 1.9 / 8.2 | 0.46 / 0.44 / 0 |
| 1.55 | 500 nm | 0 / 0.5 / 1 | 3.56 / 3.59 / 6.25 | 36.3 / 38.1 / 26.4 | 11.9 / 12.1 / 42.5 | 0.44 / 0.43 / 0 |
| 1.55 | semi-infinite | – | 6.31 | 25.2 | 46.6 | – |
| 1.80 | 100 nm | 0 / 0.5 / 1 | 1.63 / 1.83 / 2.96 | 23.7 / 24.2 / 20.2 | 0.3 / 0.3 / 1.0 | 0.47 / 0.40 / 0 |
| 1.80 | 200 nm | 0 / 0.5 / 1 | 2.84 / 3.02 / 5.04 | 21.0 / 21.2 / 18.4 | 1.0 / 1.1 / 5.0 | 0.46 / 0.42 / 0 |
| 1.80 | 500 nm | 0 / 0.5 / 1 | 5.42 / 5.53 / 8.65 | 14.2 / 16.4 / 11.9 | 7.4 / 7.2 / 31.4 | 0.40 / 0.39 / 0 |
| 1.80 | semi-infinite | – | 8.32 | 11.3 | 26.2 | – |

σ(QE) ≤ 0.07 % abs., σ(ESP) ≈ 0.7–1 % abs.

* Thin layers trade QE for speed and polarization: at 100–200 nm almost all emission happens within
  ~10 ps and the ESP is that of the early electrons (≈ 45–48 % at 1.55 eV against 25 % for the
  semi-infinite sample). QE per absorbed photon is roughly unchanged at R_back = 0 (about 45 % of the
  electrons are lost to the substrate, while the semi-infinite sample loses a similar share to the
  370 ps cutoff and to diffusion away from the surface); the lower QE comes mainly from A_layer.
* R_back = 0.5 helps little: a reflected electron usually reaches the back again. R_back = 1
  roughly doubles QE at every thickness, at the price of slower emission and lower ESP (≈ 45 % of the
  emitted electrons were reflected at least once). At 500 nm with R_back = 1 QE and ESP approach the
  semi-infinite values (1.80 eV: QE slightly above: electrons that would diffuse away are returned).
* The front band bending is the semi-infinite C21 profile throughout (d ≥ 10 W_bb here); the model
  has no back-interface potential, no substrate transport and no optical interference.
