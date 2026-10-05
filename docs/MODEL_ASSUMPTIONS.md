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
| **`"adachi1989"`** (default) | Adachi model dielectric function, JAP 66, 6030 (1989), for intrinsic GaAs (`optics.Adachi1989GaAs`). |
| any object with `absorption_coefficient(hv)` | e.g. `optics.TabulatedAbsorption(hv, alpha)` for measured α(hν), or `ConstantAbsorptionLength(l)` |

* **Interface.** `absorption_coefficient(hv)` takes hν in J and returns α in 1/m. The depth distribution
  is Eq. 7 with l = 1/α. `reflectivity(hv)` is provided for later QE normalization.
* **Parameters.** These are the Adachi (1989) GaAs values as transcribed in the public-domain (CC0)
  refractiveindex.info script by M. Polyanskiy. That transcription records three gaps in the paper,
  which are reproduced here: the E1+Δ1 term is omitted (no B2, B21 given), negative ε2 is clipped,
  and the phonon energy in the indirect term is set to zero.
  **TODO:** verify each value against Adachi (1989) Table I. The PDF is not yet in `refs/`.
* **Cross-check.** The model reproduces the refractiveindex.info tabulated n, k to their 4-digit
  rounding (`tests/test_optics.py`). Solcore is not a dependency; its Adachi implementation hard-codes
  modified parameters (b1 = 7, Γ = 0.06, E0 + 5 meV), so it is at best a loose cross-check.
* **Band-gap narrowing is not applied** to the optical data (user decision 2; C21 does not
  prescribe shifting the spectrum).
* **Sub-E0 artifact (open issue).** The Adachi E0 term has no broadening or Urbach tail, but the E2
  damped-oscillator term has a Lorentzian tail that gives absorption at all photon energies:
  1.2e3 cm⁻¹ at 1.0 eV, 2.5e3 cm⁻¹ at 1.40 eV, and about +2e3 cm⁻¹ added above E0 (≈25% of α at
  1.45 eV, ≈7% at 1.9 eV). The model is kept exactly as published. `photoexcite` refuses
  hν < E0 = 1.42 eV for this model unless `allow_optics_extrapolation=True`; this also covers the
  window E_g(p) < hν < 1.42 eV at high doping. **Decision needed:** accept the near-edge
  contribution, or substitute tabulated near-edge data via `TabulatedAbsorption`.
* **Differs from C21 Fig. 3.** C21 fitted Adachi's model to Zollner (2001) data with unpublished parameters.

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
