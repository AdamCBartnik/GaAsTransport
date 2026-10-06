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
