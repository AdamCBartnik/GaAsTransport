# Model assumptions

These are the choices the references leave open, or where this code deliberately departs from
them. Every one is a field of `gaas_mc.assumptions.ModelAssumptions` (defaults shown), so it can
be changed without editing physics code.

```python
from gaas_mc.assumptions import ModelAssumptions
a = ModelAssumptions(initial_spin_rule="chubenko_prose", pauli_blocking="step_c21")
```

Pass `a` to `build_mechanisms(sample, a, stage)`, `photoexcite(..., assumptions=a)`, and
`Simulation(..., assumptions=a)`.

References: **C21** = Chubenko et al., JAP 130, 063101 (2021) (accepted manuscript);
**K13** = Karkare et al., JAP 113, 104904 (2013); **K15** = erratum, JAP 117, 109901 (2015);
**KT** = Karkare PhD thesis.

---

## 1. Initial spin rule (`initial_spin_rule`)

| Value | Meaning |
|---|---|
| **`"per_band"`** (default) | C21 Eqs. 12–16 are authoritative. The valence band i ∈ {hh, lh, so} is chosen with probability K_i/ΣK, and the spin is +1 with probability (1+P_i)/2. The ensemble ESP equals Eq. 12. |
| `"chubenko_prose"` | Literal prose after C21 Eq. 16: hh → s = +1, lh/so → s = −1. The hh fraction is (1+ESP0)/2 so that the ESP still equals Eq. 12; lh : so = K_lh : K_so. |

*Rationale.* User decision 1: the prose is shorthand for the equations. Taken literally with the
K_i weights, the prose gives 30% instead of 50% at threshold. The compatibility mode is kept
because it reproduces the relative hh/lh/so peak heights of C21 Fig. 5 somewhat better.

*Validation.* `validation/excitation.py` → `fig6_esp0_mc.png`. The Monte Carlo ESP for 40 000
electrons per point, at four dopings and both rules, agrees with Eq. 12 everywhere (largest
deviation 2.8 standard errors over ~140 points). It reproduces 50% at threshold and the drop at
E_g + Δ_so.

## 2. Absorption model (`absorption_model`)

| Value | Meaning |
|---|---|
| **`"casey1975+adachi1989"`** (default) | `optics.CaseyAdachiAbsorption(p)`: the measured p-type near-edge absorption of Casey, Sell & Wecht, J. Appl. Phys. **46**, 250 (1975), interpolated in hν and in log10 p, blended into Adachi (1989) above the edge. |
| `"casey1975"` | Casey data only (1.31–1.59 eV). |
| `"adachi1989"` | Adachi model dielectric function alone (comparison). It refuses hν < 1.42 eV. |
| any object with `absorption_coefficient(hv)` | e.g. `optics.TabulatedAbsorption(hv, alpha)` for other measured α(hν), or `ConstantAbsorptionLength(l)` |

**Interface.** `absorption_coefficient(hv)` takes hν in J and returns α in 1/m. The depth
distribution is Eq. 7 with l = 1/α. A model raises `ValueError` where it is not valid, rather than
returning a number. `reflectivity(hv)` (from Adachi) is provided for later QE normalization.

**Default composite (user decision of 2026-10-05).**
* **1.31–1.55 eV:** Casey p-type data. The spectra are measured at p = 1.6e16, 2.2e17, 4.9e17, 1.2e18,
  2.4e18 and 1.6e19 cm⁻³ (297 K). α(hν; p) = exp[(1−w) ln α_lo + w ln α_hi], where
  w = (log10 p − log10 p_lo)/(log10 p_hi − log10 p_lo), between the two bracketing spectra. Each
  spectrum is interpolated log-linearly in hν.
* **Outside the measured doping range,** the nearest spectrum is used and a warning is issued
  (`out_of_range="raise"` is available). There is no extrapolation in p.
* **1.55 eV to the end of the Casey data (1.592 eV):** ln α is blended with the smoothstep weight
  s = 3x² − 2x³. The window ends at 1.592 eV rather than 1.65 eV because both curves must exist
  inside it. Above the window the model is Adachi (1989).
* **Seam mismatch α_Adachi/α_Casey at 1.55 / 1.592 eV:**

  | p (cm⁻³) | at 1.55 eV | at 1.592 eV |
  |---|---|---|
  | 1.5e17 | 1.17 | 1.17 |
  | 1e18 | 1.18 | 1.15 |
  | 1e19 | 1.37 | 1.25 |

  Heavy doping flattens the measured spectrum below the intrinsic value, so the blend slope is
  visible at 1e19. The paper states ±15% for α > 1e3 cm⁻¹ (Kramers–Kronig part).
* **Doping-induced edge.** The measured spectra contain the real doping-induced shift and broadening
  of the edge, so the window E_g(p) < hν < E_g(intrinsic) is absorbed (e.g. l ≈ 7.8 μm at 1.40 eV
  for 1e19). No 1.42 eV optical cutoff is imposed. C21's band-gap narrowing (Eq. 10) is applied
  only to the electronic structure, not to the optical data.
* **Below the data.** Under the lowest digitized point of a spectrum (α ≈ 13 cm⁻¹, the floor of the
  figures), the model raises. Such photons would have l > 0.7 mm anyway.
* **Excitation rule.** `photoexcite` requires (1) the band-transition energetics, hν > E_g(p), and
  (2) a finite α > 0 from the model.

**Casey data: digitization and provenance** (`tools/digitize_casey1975.py`,
`tools/casey1975_digitization.json`, `gaas_mc/data/casey1975_ptype.csv`)
* **Source.** The paper gives α(hν) only as figures (Figs. 6–8, log scale 10–1e5 cm⁻¹, 1.30–1.60 eV).
  No machine-readable table or trustworthy digitization was found, so the curves were digitized from
  the page scans in the publisher PDF (`refs/Casey1975_JAP46_250.pdf`, not committed).
* **Calibration.** Each figure's grid lines are located and fitted. A piecewise map, exact on every
  grid line, absorbs scan skew and drawing nonuniformity; an affine fit would leave residuals of
  ≤1.7 meV and ≤0.018 dex.
* **Extraction.** Sweeps along lines of constant α or constant E find dark-run crossings, and the
  configured curve order assigns them. A sweep is used only if exactly the expected number of
  crossings remains after removing grid lines and over-wide runs (labels, arrows).
* **Cleaning.** Two outlier filters run afterwards: a neighbour line in pixel space, and a robust
  Theil–Sen local fit (> 6% deviation). Curves that coincide within the line width are recorded as
  merged crossings.
* **Assembly** (`gaas_mc/optical_data.py`). Figures are averaged in ln α on a 1 meV grid. Merged
  crossings are used only where no figure resolves the curve, and gaps (curve crossings, labels) are
  bridged log-linearly and flagged.
* **Checks.**
  * The overlays `tools/out/casey1975_fig*_overlay.png` are regenerated by the script.
  * Cross-figure agreement: 1.2e18 (Figs. 7, 8) and 2.2e17 (Figs. 6–8) coincide.
  * The paper's own statements are reproduced: α below the gap increases with p; α ≈ 1e2 cm⁻¹ near
    1.35 eV for 1.6e19; all p-type spectra converge near 1.6 eV.
  * The digitization is bit-for-bit reproducible from the PDF (test).
* **Uncertainty.** About ±1.5 meV in hν and about ±2.5% in α from digitization, on top of the paper's ±15%.
* **Not used.** Solcore is not a dependency. Its Adachi implementation hard-codes modified
  parameters, so it is at most a loose cross-check.

**Adachi (1989).** All 14 GaAs parameters were verified against Table I
(`refs/Adachi1989_JAP66_6030.pdf`). The model reproduces the public-domain refractiveindex.info
tabulation to its 4-digit rounding. The E2 damped-oscillator term has a Lorentzian tail that gives
α > 0 at all energies (1.2e3 cm⁻¹ at 1.0 eV, 2.5e3 cm⁻¹ at 1.40 eV). This tail is **not** interband
absorption, so `Adachi1989GaAs` raises below E0 = 1.42 eV unless constructed with
`below_E0="model"`. Above E0 the tail still adds about 25% to α at 1.45 eV, which is why measured
data are used near the edge.

## 3. Holes and electron–hole scattering

| Field | Default | Alternatives |
|---|---|---|
| `hole_bands` | `("hh", "lh")` | `("hh",)` (C21: heavy holes only) |
| `hole_statistics` | **`"fermi_dirac"`** | `"maxwell_boltzmann"` |
| `pauli_blocking` | **`"fermi_dirac"`**: final hole accepted with probability 1 − f(E_h′) | `"step_c21"` (reject if E_h′ < E_F^h, C21 Eq. 44), `"none"` |
| `eh_electron_mass` | **`"band_edge"`** (m* in Eqs. 38–41, as published) | `"velocity_mass"` m*(1+2αE) |
| `eh_valleys` | `("Gamma", "L", "X")` (KT Fig. 2.4 shows hole scattering in all valleys) | any subset |

* **Hole bath (`holes.HoleGas`).** The hh and lh bands are parabolic and isotropic, with one shared
  chemical potential μ. It is solved from p = Σ_b N_b F_{1/2}(μ/kT) by root-finding (no
  Maxwell–Boltzmann shortcut at high doping); the split-off band is neglected. At 1e19, μ = +10.7 meV
  into the band, p_hh = 9.3e18 and p_lh = 6.9e17 cm⁻³. This μ is used only for e–h scattering. C21's own
  prescriptions remain unchanged where C21 uses them: Eq. 58 (single-band Fermi level) for band bending
  and degeneracy, and Eq. 44 for BAP.
* **Collision (`scattering/electron_hole.py`).** One mechanism per hole band. Eqs. 38–42 were
  re-derived independently: W_max, the acceptance 2gβ/(g²+β²), g = 2|k_rel|, and the relative-angle
  distribution. The final state is solved exactly. With K = k + k₀ and c = m_R K/m_h, the final state is
  k′ = c − s ĝ′/2 and k₀′ = K − k′, where s solves energy conservation for the nonparabolic electron.
  For α = 0 this reduces to C21 Eq. 43 (s = g). Momentum and energy are conserved to round-off in
  every collision. If no root exists (~1e-4 of attempts), the event is a self-scattering.
* **Erratum K15.** It only states that an e–h bug affected p ≥ 1e19 results; it does not describe
  the bug. The safeguards are in `tests/test_stage_bc.py`:
  1. The accepted rate equals p_b⟨|v_e − v_h| σ_Born⟩, computed from the velocities and the textbook
     cross-section (to 1%).
  2. Momentum and energy are conserved in each collision.
  3. Pauli blocking is negligible at 1.5e17 and significant at 1e19.
  4. **Detailed balance:** with e–h scattering alone, electrons must relax to the lattice-temperature
     Maxwellian.
* **Detailed-balance results** (`validation/eh_detailed_balance.py`; N = 8000; ⟨E⟩/kT after
  relaxation; the statistical σ is ≈ 0.014):

  | Band | Pauli rule | e–h mass | 1e19 | 1.5e17 | Exact |
  |---|---|---|---|---|---|
  | parabolic | Fermi–Dirac | band edge | 1.483 | – | 1.499 ✓ |
  | parabolic | C21 step (Eq. 44) | band edge | **1.194** | – | 1.499 ✗ (electrons cool to ≈0.8 T) |
  | parabolic | none | band edge | **1.776** | – | 1.499 ✗ |
  | nonparabolic | Fermi–Dirac | band edge (default) | 1.513 | 1.515 | 1.556 (−3%) |
  | nonparabolic | Fermi–Dirac | velocity mass | 1.471 | 1.461 | 1.556 (−6%) |

  * The C21 step rule combined with thermal holes violates detailed balance; Fermi–Dirac blocking is
    required for consistency.
  * With the nonparabolic band, the parabolic relative-motion cross-section of Eqs. 38–42 leaves
    the electrons ≈3% too cold. The velocity-mass variant is worse, so the band-edge mass (as published) is the default.
  * In the full model, phonon scattering also couples the electrons to the lattice, which reduces
    this bias. A fully consistent nonparabolic binary-collision model would be a separate development.

## 4. Valleys

| Field | Default | Note |
|---|---|---|
| `valley_multiplicity` | Γ→L 4, Γ→X 3, L→Γ 1, L→L 3, L→X 3, X→Γ 1, X→L 4, X→X 2 | The number of distinct equivalent destination valleys reachable from **one** initial valley; not the total degeneracy (4 L, 3 X). C21 Table I lists Z_L = 4 and Z_X = 3; KT uses 3 and 2 for same-type transfers. |
| `intervalley_dos_factor` | **`"numerator"`** | C21 Eq. 33 prints (1+2α_jE′) in the denominator. |

* **Applied once.** Eq. 33 contains Z_j once, and its density of states is that of a single
  destination valley. The test checks the rate against the golden rule (π D² Z/ρω) g_1(E′)·occupation.
* **The Eq. 33 typo is settled.** With the factor in the numerator, the Γ→L/X rates match C21 Fig. 7
  (e.g. Γ→L emission 7.4e13 and Γ→X emission 1.2e14 s⁻¹ at 1 eV). As printed, the rates come out up to 2.6× too low.

## 5. Spin in L and X (`side_valley_spin`)

Only `"frozen"` is implemented: the spin is preserved and there is no additional relaxation in L or X.
Γ-valley EY/DP/BAP rates are never applied to L/X by analogy (user decision 4). In the Eq. 54
bookkeeping, a flight in L/X contributes zero flip probability, because the flip test uses the
pre-event state.

Every particle and every surface-arrival record carries `time_in_valley` (Γ, L, X) and `visited`
flags. `SurfaceArrivals.upper_valley_summary()` reports:
- the fraction of arrivals that ever visited L or X;
- the fraction that arrive in L or X;
- the mean residence times.

These numbers indicate whether a literature-based L/X spin model is needed.

## 6a. Band bending and fields (Stage D)

| Item | Choice | Note |
|---|---|---|
| Profile | `fields.C21BandBending(sample)`: E_C(z) = E_Cb − E_bb(1 − z/W_bb)² for z < W_bb (Eq. 61), E_z = 2E_bb/(eW_bb)·(1 − z/W_bb) (Eq. 62) | E_bb = E_g/2 − E_F^b, with the surface Fermi level pinned mid-gap (Eqs. 56–58); W_bb from Eq. 59. Both can be overridden. Reproduces C21's 0.694 eV / 9.947 nm at 1e19. |
| User potential | `fields.PotentialField(E_C_fn, z_max=...)` or `CallableField(Ez_fn, band_edge_fn, z_max=...)` | For replacing C21's profile with your own electrostatics or image-charge model. |
| Valleys | All valleys shift rigidly with E_C(z) | The tracked E is the kinetic energy above the local valley minimum. |
| Scattering in the band-bending region | **Local mobile holes** (§6b); `depletion_scattering="bulk"` keeps bulk rates everywhere (C21-like) for comparison | — |
| Flights | `"auto"` = hybrid: direct W_total(E) for z ≥ z_max, null-collision with a constant bound inside the field region. Flights are stopped at z = z_max (exact by memorylessness). | `Result.flight_mode` reports `"hybrid"`. |
| Integrator | Velocity Verlet (kick–drift–kick). Default substep h = min(2 fs, 0.05/ω), with ω = √(e·max\|dE_z/dz\|/m_Γ) | 0.25 fs at 1e19, 0.65 fs at 1.5e18, 2 fs at 1.5e17. Error crossing the band bending ≤ 0.14 meV at 1e19; C21's 1 fs step gives ~2 meV, systematically positive. Second-order convergence is tested. |
| Surface arrivals | `SurfaceArrivals.band_edge_at_surface` = E_C(0) − E_C(bulk) (= −E_bb) | Puts arrival kinetic energies on an absolute scale for the surface model. |
| Back wall | `Simulation(z_back=L, back="absorb" \| "reflect")` | A reflecting wall stops the flight, flips k_z, and restarts (exact). |

Validation (`validation/stage_d_band_bending.py`, `tests/test_stage_d.py`):
* Figs. 16 and 17 are reproduced.
* Energy is conserved ballistically, and transverse k is unchanged.
* Flights stop and restart correctly at the region boundary.
* **Equilibrium:** in a closed slab with reflecting walls, the density follows n(z) ∝ exp(−E_C(z)/kT)
  within ±4–8% statistics, and ⟨E⟩ = 1.552 kT against the exact nonparabolic Maxwellian (1.556 kT).

## 6b. Depletion region: local mobile holes (`depletion_scattering`, `depletion_screening_cap`, `depletion_pop_screening`)

User decision of 2026-10-06. The band-bending potential and the fixed ionized-acceptor charge are
unchanged; only the mobile-hole population changes (`gaas_mc/depletion.py`).

| Item | Choice |
|---|---|
| Fermi level | Global, fixed once from the bulk p and T with the hh + lh Fermi–Dirac gas (the same gas the e–h scattering uses). It differs from C21's single-band Eq. 58 by ≈ 2 meV at 1e19; the potential, which uses Eq. 58, is left exactly as before. |
| Local holes | The valence bands shift with the same φ(z) = E_C(z) − E_C,bulk, so η(z) = η_bulk + φ/kT and p_b(z) = N_b F_{1/2}(η(z)). No local chemical potential is introduced. At the surface p/p_bulk ≈ 3e-12 (1e19) and 3e-11 (5e17). |
| Electron–hole scattering | Uses the local gas everywhere: density, Fermi–Dirac occupation of the sampled hole, and Pauli blocking of the final state. |
| BAP | Uses local p, local E_F^h (Eq. 44 with p(z)), and the C21 degeneracy criterion evaluated with the local E_F − E_V(z). The jump in τ_BAP where that criterion switches (Eqs. 50/51 ↔ 47) is C21's own discontinuity (ambiguity A11). |
| Ionized impurities | N_A⁻ stays at the dopant density. The screening is recomputed from the local mobile carriers: β(z) = β_bulk·√(χ(z)/χ_bulk), with χ = ∂p/∂μ = Σ_b N_b F_{−1/2}(η)/kT. This equals the C21 bulk β (Eq. 28 or 29) at φ = 0 and becomes Debye-like ∝ √p when nondegenerate. |
| Screening cap (user decision of 2026-10-06; frozen) | As p → 0 the screening length, and with it the Brooks–Herring rate, diverge. The local length is capped at L_cap = max(L_bulk, a). **This cap is not part of Brooks–Herring.** It is an approximate finite-impurity-spacing / third-body cutoff, used because the BH screened-Coulomb model (one ion screened by a mobile-carrier cloud) becomes questionable as the mobile-hole density tends to zero; beyond the ion spacing the potential is cut off by the neighbouring ions (the Conwell–Weisskopf idea), not by mobile holes. Default: a = acceptor Wigner–Seitz radius (3/4πN_A)^⅓, 2.9 nm at 1e19 (< L_bulk = 4.7 nm, **so at 1e19 the screening does not weaken at all**) and 7.8 nm at 5e17 (L_bulk 6.1 nm). `"band_bending_width"` (a = W_bb: 9.9 nm / 42 nm) is kept **only as a sensitivity test**, as is an explicit length. Not refined further. |
| POP | C21 Eq. 25 is screened by the same hole β, so the local β is used by default (`depletion_pop_screening=True`). The phonon coupling itself is unchanged. |
| Unchanged | Acoustic and intervalley parameters, band gap, and E_g(p) narrowing (bulk values), and the potential. Minority electrons (≈ n_i²/p) are neglected in screening. |
| Numerics | φ grid from φ(0) to 0 with spacing ≤ kT/4 (109 points at 1e19). Each density-dependent mechanism holds one variant per grid point. Rates are interpolated linearly in φ, and the variant for an event is drawn with probability w·W₁/[(1−w)W₀ + w·W₁], which is exact for the interpolated rate. The null-collision bound covers every φ. Spin-relaxation tables are (φ, E). |

Tests (`tests/test_depletion.py`):
* the p(z) formula holds, and p is monotonic and equal to the bulk value at φ = 0;
* the bulk grid row equals the bulk mechanisms, and the ions are never depleted;
* the cap options behave as specified, and β follows the FD susceptibility;
* BAP vanishes where holes are depleted;
* the variant choice is unbiased;
* a closed slab with local impurity screening keeps its Boltzmann density;
* the compatibility option uses no local mechanisms.

Profiles versus z: `validation/depletion.py profiles` → `depletion_profiles.png`.

## 6c. Valley identity at the surface

Surface arrivals keep `valley` (Γ/L/X), `eqv` (which of the 4 L or 3 X valleys), `k` (from that
valley's minimum), and `K` (full crystal wavevector = valley center + k). The lab axes are taken as the
cubic axes, with z = [001] the surface normal. The equivalent valley is chosen on each intervalley
transfer consistently with the multiplicities: uniformly among the destination type's valleys, or
among the *other* valleys for L→L and X→X (`gaas_mc/valleys.py`). L arrivals are **not** mapped onto
a scalar-mass Γ model; the surface stage will be valley-aware.

## 6d. Surface emission: the C21 model (Stage E, `gaas_mc/surface_c21.py`)

An optional module. Transport only calls `surface.interact(k, E, valley, K, rng)` when an electron
reaches z = 0 (`Simulation(..., surface=C21Surface(chi, material))`), so another surface model with the
same interface can replace it. The bulk Monte Carlo is unchanged.

| Item | Choice |
|---|---|
| Vacuum level | E_vac = E_C,Γ(z=0) + χ. χ is the fitting parameter; χ_eff = χ − E_bb (C21 Fig. 15(c)). |
| Barrier | Vacuum side: V(x) = E_vac + E_b(1 − x/L_b) for 0 < x < L_b, then E_vac. L_b = 0.15 nm, E_b = 4 eV (C21 Sec. III C 2). Overridable `barrier(x)`. |
| Transmission | Propagation (transfer) matrix: 200 constant slices, BenDaniel–Duke matching (ψ and ψ′/m continuous), m* → m0 at the interface. Inside, the normal wavevector is the electron's k_z with mass m_in; T = (k_out/m0)/(k_z/m_in)·\|t\|². **Default (user decision 2026-10-06): `matching_mass="band_edge"`, m_in = m*** (the literal reading of C21's "the electron mass changes from m*_e to m0"; it reproduces C21 Fig. 18). Option `"velocity"`: m_in = m*(1+2αE), so the incident flux equals the group velocity of the nonparabolic band; it gives a higher T. Tested against the analytic rectangular barrier and the mass step (flux conservation). C21 do not state the nonparabolic matching. |
| Energy | E_tot = valley offset + E_kin, relative to the Γ edge at z = 0. |
| (100) restriction | K∥ (valley centre + k, lab z = [001]) is folded into the (001) surface Brillouin zone (b1, b2 = (2π/a)(1, ±1)) and conserved. The vacuum normal energy is ε = E_tot − χ − ħ²K∥²/2m0, and emission needs ε > 0. Folded valley-centre K∥ costs 2.35 eV (all L) and 4.7 eV (X[100], X[010]), so only Γ and X[001] emit at the energies reached here: C21's "Γ and some X valleys". The restriction is energetic, not a hard rule. |
| Trapping | E_tot < E_vac: trapped and removed (C21 Sec. IV). An electron above E_vac with ε ≤ 0 (too much transverse energy) is reflected, not trapped. |
| Otherwise | Emitted with probability T, else specularly reflected (k_z → −k_z) back into the GaAs and transported again. |
| Bounce | A reflected electron at z = 0 is pushed back by the surface field. If it returns within one Verlet substep, the bounce is done analytically: t_ret = 2ħk_z/\|F(0)\|, k_z → −k_z (exact for a constant force by time reversal). |
| Records | `Result.emissions`: inside state at emission (t, E, k, K, valley, eqv, spin, spin0, z0, E0, band), surface-encounter count, vacuum momentum p_vac and E_vac. Statuses EMITTED and TRAPPED. `Result.arrivals` keeps the first arrival of each electron. |
| Spin | Unchanged at the surface. `spin_flip_at_arrival` (default False, as C21) applies Eq. 54 for the time since the last test at every encounter. |
| χ branching | The history up to the first surface encounter does not depend on χ, so `Ensemble.from_arrivals()` + `run(..., start_at_surface=True)` continues one first-arrival ensemble with each χ. |
| Surface crossing (integrator) | A field-region substep that crosses z = 0 is cut at the root of z_old + v t + a_z t²/2 (force at z_old, velocity mass), and k_z at the crossing follows from total-energy conservation with k∥ unchanged. Linear interpolation of z (used before 2026-10-06, and still for the back contact and the field-region boundary) underestimates the crossing time for short arcs near the surface, so with a reflecting surface model |k_z| drifted towards 0 over repeated returns (found when brute-force bouncing stalled; Stage E results recomputed). With an absorbing surface each electron crosses once; the Stage D energy-conservation tests pass unchanged. |
| Repeated returns | A reflected electron pushed back by the surface field returns after t_ret = 2ħk_z/\|F(0)\| with the same state, i.e. the same transmission trial. Both engines book the floor(dt/t_ret) returns of one flight at once (geometric number of failed trials before an emission): exact in distribution, and necessary because grazing electrons (k_z → 0) need ~1/(Γ0 t_ret) returns (~1e9 observed). `surface_bounce_aggregation=False` restores one return per iteration (tests). The encounter counter saturates at 2³¹−1; its mean is dominated by these electrons, so medians and percentiles are reported. |
| Matching mass: evidence | Fig. 18 benchmark (`docs/VALIDATION.md`): with `matching_mass="band_edge"` the computed QE agrees with C21 to a few % at every χ and hν; with the velocity mass it is 25–50 % higher. This indicates what C21 did; it is not a physical argument for either choice. The band-edge mass is therefore the default of the C21 surface model; the velocity mass stays as an option. |

## 6e. Finite GaAs layer (Stage F, `gaas_mc/back.py`, `photoexcite(..., thickness=d)`)

User request of 2026-10-06: the smallest extension toward thin active layers. The layer is
0 < z < d; z = 0 is the emitting surface (electrons leave toward −z, as before), z = d is the
GaAs/substrate boundary.

| Item | Choice |
|---|---|
| Material | GaAs everywhere in 0 < z < d with the unchanged physics: same scattering, the semi-infinite C21 band-bending profile at the front, no back-interface potential, no substrate transport. A warning is issued when d < 3 W_bb (the electrostatic profile is then not self-consistent; not corrected). |
| Back boundary | `Simulation(..., z_back=d, back=PartialReflector(R_back))`. The flight stops at z = d (exact crossing time and energy-conserving state, as at the front surface). One random decision per encounter: reflected with probability R_back (k_z → −k_z; energy, k_x, k_y, valley, equivalent valley and spin unchanged), otherwise lost into the substrate (status `BACK`). R_back = 0: absorbing substrate; R_back = 1: perfectly reflecting back surface. |
| Records | `Ensemble.n_back` (back encounters; reached the back ⇔ n_back > 0; lost ⇔ status BACK), also in `SurfaceArrivals.n_back` and `Emissions.n_back`. |
| Generation | Beer–Lambert truncated to the layer, P(z) = α e^(−αz)/(1 − e^(−αd)); fraction of incident photons absorbed in the layer `layer_absorption()` = (1 − R)(1 − e^(−αd)); QE = A_layer·N_emitted/N_generated. No optical interference, no reflection of light at z = d. |
| Modularity | The back boundary is an object with `interact(k, E, valley, rng, pid) → reflected?`, the counterpart of the surface model; a physical GaAs/AlGaAs interface model can replace `PartialReflector`. The fast engine compiles `PartialReflector`; other back models currently need the reference engine. |
| Not supported | A field that pushes electrons into the back boundary (repeated back returns would need the bounce aggregation of the front surface); raises. The C21 band bending always pushes toward z = 0. The legacy `back="reflect"` (mirror wall, closed-slab tests) and `back="absorb"` are unchanged. |

## 6f. Lateral position (bookkeeping)

The sample is laterally homogeneous, so the transport depends on z only. The lateral displacement
(x, y) from the excitation point is nevertheless tracked in both engines (`Ensemble.x/y`, and in the
arrival and emission records): k_x, k_y are constant during a flight (field along z) and
v_par = ħk_par/(m(1+2αE)); exact in the field-free region, velocity factor averaged over the start and
end energy of a flight in the field region (flights there are ~1 fs); electrons bouncing along the
surface (bounce trains) keep moving laterally. It does not affect any other result (reference engine
bitwise unchanged). The ParticleGroup output uses x, y = laser-spot point + lateral displacement.

## 6. Other

| Field | Default | Note |
|---|---|---|
| `initial_k_direction` | `"isotropic"` | C21 is silent. |
| `pop_angle` | `"chubenko_eq30"` | Eq. 30 is the *unscreened* POP angular law; `"screened"` samples the distribution consistent with the screened rate (Eq. 25). |
| `flight_mode` | `"auto"`: direct W_total(E) flights when the field is zero, null collisions otherwise | User decision 6. With direct flights, the only remaining null events are rejections inside mechanisms (e–h acceptance, kinematics, Pauli). Statistically identical to the self-scattering mode (test). |
| `spin_flip_at_arrival` | `False` | C21 applies Eq. 54 only at real scattering events. `True` applies it for the residual time at surface arrival. |

## 7. C21 statements used as written (not configurable)

* Eq. 10 band-gap narrowing is applied to the electronic structure: excitation energies, the EY/DP
  E/E_g factors, and band bending.
* Screening is Debye (Eq. 28) when nondegenerate, and Thomas–Fermi with light holes (Eq. 29) when
  degenerate. Degeneracy is read as E_F^b < 2kT (Eq. 58); under this reading 1.5e18 is degenerate.
* EY uses A = 32/27 and DP uses Q = 1/6 for every mechanism. The DP total uses the Matthiessen τ_m of
  Eq. 52 (ap + pop + ij + ii, excluding e–h).
* BAP follows Eqs. 47/50/51 with |ψ(0)|² = 1 and p = total hole density.
* Eq. 54 is applied with δt measured since the previous real (non-self) scattering event.
