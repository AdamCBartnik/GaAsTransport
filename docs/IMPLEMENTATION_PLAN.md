# GaAs spin-polarized photoemission Monte Carlo — implementation plan

References (PDFs in `refs/`):

* **[C21]** O. Chubenko *et al.*, J. Appl. Phys. **130**, 063101 (2021), accepted manuscript
  (LA-UR-22-23422). Equation numbers below refer to this manuscript, which is the primary reference.
* **[K13]** S. Karkare *et al.*, J. Appl. Phys. **113**, 104904 (2013).
* **[K15]** Erratum, J. Appl. Phys. **117**, 109901 (2015).
* **[KT]** S. Karkare, PhD thesis (Cornell). It contains the same model as [K13] in more detail.

> Note on the accepted manuscript: the version of record may differ from it in typesetting. If you have
> the published version, please spot-check Table I and Eqs. 25, 31, 33, 45–51 against it.

---

## 1. Implementation plan

**Architecture.** Each electron is an independent history: event-driven, with self-scattering. The
loop is vectorized across particles, but every particle keeps its own clock. Particles never
interact, so no time synchronization is needed, and multiprocessing over seeds/chunks is
straightforward.

```
gaas_mc/
  constants.py     SI constants (scipy.constants) + eV / cm^-3 helpers
  material.py      Material dataclass (Table I, SI units) + Sample (material + p + T -> Eg(p),
                   E_F, degeneracy, screening beta, E_F^h, ...)
  bands.py         Kane dispersion gamma(E)=E(1+aE): E(k), k(E), v(k), DOS; direction helpers
  excitation.py    Eqs. 6-16: z0, band selection, E0, initial k direction, initial spin
  particle.py      Ensemble (structure of arrays: z, t, k[N,3], valley, spin, status, counters)
  scattering/
    base.py        Mechanism interface: rate(E), momentum_rate(E), final_energy, scatter_k(...)
    acoustic.py    Eq. 23-24          (Stage A)
    polar_optical.py Eq. 25-32        (Stage A)
    impurity.py    Eq. 35-37          (Stage B)
    electron_hole.py Eq. 38-44        (Stage B)
    intervalley.py Eq. 33-34          (Stage C)
  spin.py          Eqs. 45-54: EY, DP, BAP rates, total tau_s, flip probability
  fields.py        Field models: none / uniform / callable V(z) / C21 band bending Eqs. 56-62
  transport.py     Rate tables, free flight, self-scattering, mechanism choice, spin flips,
                   surface-arrival detection, time snapshots
  surface.py       SurfaceArrivals record (no emission physics). Optional C21 barrier later.
  diagnostics.py   Plots: rates, tau_m, spin rates, distributions
tests/             pytest unit tests (dispersion, DOS, rates vs. numerical golden rule, ...)
validation/        scripts that regenerate C21 figures for side-by-side comparison
```

**Units.** SI everywhere internally: J, m, s, kg, and m^-3. User-facing constructors take eV and
cm^-3 and convert them at the boundary. Nonparabolicity alpha is stored in 1/J.

**Transport loop (per iteration, vectorized over the alive particles).**

1. Draw a flight time `tau = -ln(U)/Gamma0[valley]`, where Gamma0 bounds the total rate over the tabulated energy range.
2. Propagate the particle. With no field: `z += v_z tau` (exact). With a field: velocity-Verlet substeps of at most `dt_max`, using `hbar dk/dt = -e E(z)` [Eq. 18] and `dz/dt = v_z` [Eq. 17].
3. If z crosses 0, record a **surface arrival** at the exact crossing time and stop that particle.
4. Choose a mechanism: `r*Gamma0` against the cumulative rates at the current E. The remainder is self-scattering.
5. Real event: apply the Eq. 54 spin-flip test with `dt` equal to the time since the last real event, then call the mechanism's `scatter`.
6. Particles that reach `t > t_max` are marked as timed out. A thin-film back boundary is optional.

**Stages.** A = Γ valley, acoustic + screened POP, no field. B = + ionized impurity + e–h.
C = + L/X valleys + intervalley. D = band bending. E = optional C21 surface barrier (benchmark only).

---

## 2. Physical parameters

T = 300 K throughout. In the Module column, `mat` is `material.py`.

| Symbol | Meaning | Value | Units | Source | Module |
|---|---|---|---|---|---|
| m*_Γ | Γ electron mass | 0.063 | m0 | C21 Tab. I | bands, all scattering, spin |
| m*_L | L electron mass | 0.22 | m0 | C21 Tab. I | (Stage C) |
| m*_X | X electron mass | 0.58 | m0 | C21 Tab. I | (Stage C) |
| α_Γ | Γ nonparabolicity | **0.61** | eV⁻¹ | C21 Tab. I | bands, excitation (Eq. 9) |
| α_L | L nonparabolicity | 0.461 | eV⁻¹ | C21 Tab. I | (Stage C) |
| α_X | X nonparabolicity | 0.204 | eV⁻¹ | C21 Tab. I | (Stage C) |
| m*_hh | heavy hole mass | 0.50 | m0 | C21 Tab. I | excitation, e–h, BAP, E_F, N_V |
| m*_lh | light hole mass | 0.088 | m0 | C21 Tab. I | excitation, ESP0 (ζ), TF screening |
| m*_so | split-off mass | 0.15 | m0 | C21 Tab. I | excitation |
| E_g0 | intrinsic gap | 1.423 | eV | C21 Tab. I | mat (Eq. 10) |
| Δ_so | spin-orbit splitting | 0.332 | eV | C21 Tab. I | excitation, ESP0, EY, DP |
| Δ_ΓL | Γ–L separation | 0.284 | eV | C21 Tab. I | (Stage C) |
| Δ_ΓX | Γ–X separation | 0.476 | eV | C21 Tab. I | (Stage C) |
| Ξ_dΓ, Ξ_dL, Ξ_dX | acoustic deformation potential | 7.01, 9.2, 9.0 | eV | C21 Tab. I | acoustic |
| ħω0 | LO phonon energy | 35.36 | meV | C21 Tab. I | polar_optical |
| D_ΓL, D_ΓX, D_LL, D_LX, D_XX | intervalley deformation potential | 10, 10, 10, 5, 7 | eV/Å | C21 Tab. I | (Stage C) |
| ħω_ΓL, ħω_ΓX, ħω_LL, ħω_LX, ħω_XX | intervalley phonon energy | 27.8, 29.9, 29, 29.3, 29.9 | meV | C21 Tab. I | (Stage C) |
| Z_Γ, Z_L, Z_X | equivalent final valleys | 1, 4, 3 | – | C21 Tab. I | (Stage C, see ambiguity A2) |
| A_i (ap, pop, ij, ii) | EY constant | 32/27 | – | C21 Tab. I / Eq. 45 | spin |
| Q_i (ap, pop, ij, ii) | DP constant | 1/6 | – | C21 Tab. I / Eq. 46 | spin |
| B | DP spin-splitting constant | 10 ħ²/(2m0) | J m² | C21 text after Eq. 46 | spin |
| Δ_exc | exciton exchange splitting | 47 | µeV | C21 Tab. I | spin (BAP, Eq. 48) |
| \|ψ(0)\|² | Sommerfeld factor | 1 (full screening) | – | C21 Tab. I, text after Eq. 49 | spin (BAP) |
| ε_∞ | high-frequency dielectric constant | 10.92 | ε0 | C21 Tab. I | polar_optical (ε_p) |
| ε_s | static dielectric constant | 12.90 | ε0 | C21 Tab. I | screening, impurity, e–h, BAP, Eg(p), band bending |
| ρ | density | 5360 | kg m⁻³ | C21 Tab. I | acoustic, intervalley |
| v_s | sound velocity | 5240 | m s⁻¹ | C21 Tab. I | acoustic (c_l = ρ v_s²) |
| Z | impurity charge | 1 | e | C21 Eq. 35, "full ionization" | impurity |
| N_a⁻ | ionized acceptors | = p | m⁻³ | C21 Sec. III B | impurity, e–h, BAP |
| R(ħω), l(ħω) | reflectivity, absorption length | Adachi fit to Zollner data (Fig. 3); **not tabulated** | –, m | C21 Fig. 3 | excitation (**user input for now**, A5) |
| t_sim | simulation time (recombination) | 370 | ps | C21 Sec. IV | transport (default t_max) |
| Lb, Eb | triangular surface barrier | 0.15 nm, 4 eV | – | C21 Sec. III C 2 | (Stage E, optional) |

**Derived, doping-dependent quantities** (computed in `material.Sample`):

| Quantity | Expression | Source | Check value |
|---|---|---|---|
| E_g(p) | E_g0 − (3e²/16π ε_s) √(e²p/ε_s k_B T) | Eq. 10 | 1.409 / 1.398 / 1.361 eV at 5e17 / 1.7e18 / 1e19 (Fig. 6 legend) |
| ε_p | (1/ε_∞ − 1/ε_s)⁻¹ | after Eq. 25 | |
| N_V | 2 [m_hh k_B T / 2πħ²]^{3/2} | after Eq. 58 | |
| E_F^b = E_F − E_V | Nilsson formula | Eq. 58 | E_bb(1e19) = E_g/2 − E_F^b = 0.694 eV (Fig. 16) |
| degenerate? | E_F^b < 2 k_B T (our reading, see A8) | text after Eq. 58 | |
| L_D (nondegenerate) | √(ε_s k_B T / e² p) | Eq. 28 | |
| L_TF (degenerate) | √(πħ²ε_s / e² m_lh) (π / 3N_lh)^{1/6}, with N_lh = m_lh^{3/2} p / (m_hh^{3/2} + m_lh^{3/2}) | Eq. 29 | |
| β | 1/L | after Eq. 25 | |
| E_F^h | (3π²ħ³p)^{2/3} / 2m_hh (hbar placement as printed; dimensionally it is ħ²(3π²p)^{2/3}/2m) | Eq. 44 | |
| m_R | m_e m_hh / (m_e + m_hh) | Eq. 39 | |
| a_B, v_B, E_B, 1/τ0 | 4πħ²ε_s / e²m_R; ħ/m_R a_B; ħ²/2m_R a_B²; (3/64) π Δ_exc² / ħE_B | Eqs. 47–48 | a_B ≈ 12 nm, E_B ≈ 4.6 meV |
| W_bb | √(2ε_s \|E_bb\| / e p) | Eq. 59 | 9.947 nm at 1e19 (Fig. 16) |

---

## 3. Mechanisms

### Momentum relaxation (Γ valley unless noted)

| Mechanism | Rate | Valleys | ΔE | Final direction | Parameters |
|---|---|---|---|---|---|
| Acoustic phonon (absorption + emission lumped, elastic) | Eq. 23: √2 m*^{3/2} Ξ_d² k_BT √γ (1+2αE) / (π c_l ħ⁴) | all (valley Ξ_d) | 0 | isotropic, cos θ = 1 − 2r (Eq. 24) | m*, α, Ξ_d, ρ, v_s, T |
| Polar optical, absorption | Eq. 25 (upper signs), × N0 | all | +ħω0 | Eq. 30 (ξ = 2√(γγ')/(√γ − √γ')²), azimuth uniform | ħω0, ε_∞, ε_s, β, m*, α |
| Polar optical, emission | Eq. 25 (lower signs), × (N0+1); needs E > ħω0 | all | −ħω0 | Eq. 30 | same |
| Intervalley i→j, absorption / emission | Eq. 33, × N or N+1; E' = E ± ħω_ij − Δ_ji | Γ↔L, Γ↔X, L↔L, L↔X, X↔X | ±ħω_ij − Δ_ji | isotropic | D_ij, ħω_ij, Z_j, m*_j, α_j, ρ |
| Ionized impurity (Brooks–Herring) | Eq. 35 | all | 0 | Eq. 36: cos θ = 1 − 2r / [1 + 4γ(1−r)/E_β] | p = N_a⁻, Z, ε_s, β |
| Electron–hole (binary, rejection) | W_max: Eq. 38; accept if r < 2gβ / (g² + β²) (Eq. 40) | Γ (C21) | electron E changes (energy exchanged with hole) | relative wavevector g rotated by Eq. 42 angle; k' = k − (g' − g)/2 (Eq. 43); Pauli rejection if E_hole' < E_F^h (Eq. 44) | p, m_hh, m_R, ε_s, β, hole distribution (A10) |

The momentum-relaxation rates (Eq. 22) are used only by the spin models. Acoustic and intervalley
are isotropic, so 1/τ_m = W. For POP, C21 gives Eq. 31; we also evaluate Eq. 22 in closed form and by
numerical quadrature (A7). For impurities it is Eq. 37. The total, Eq. 52, sums ap + pop + ij + ii
and leaves e–h out.

### Spin relaxation (Γ valley)

| Mechanism | Rate | Depends on | Parameters |
|---|---|---|---|
| Elliott–Yafet | Eq. 45: A_i (1 − m*/m0)² (η/(1+η))² ((1+η/2)/(1+2η/3))² (E/E_g)² / τ_m^i, with η = Δ_so/E_g; summed over i | each 1/τ_m^i | A_i, Δ_so, E_g(p), m* |
| D'yakonov–Perel | Eq. 46: (128/945) Q_i Δ_so² B² m*² / [(1+η)(1+2η/3) ħ⁶] (1 − m*/m0)(E/E_g)³ τ_m; total uses the Matthiessen τ_m of Eq. 52 | total τ_m | Q, B, Δ_so, E_g(p), m* |
| BAP, nondegenerate | Eq. 47: (2/τ0)(v_k/v_B) a_B³ p \|ψ(0)\|⁴ | v_k | Δ_exc, m_R, ε_s, p |
| BAP, degenerate, thermal (E ≤ m_h (k_BT)² / m_e E_F^h) | Eq. 50: (3/τ0)(v_k/v_B)(k_BT/E_F^h) a_B³ p \|ψ(0)\|⁴ | v_k | + E_F^h |
| BAP, degenerate, hot | Eq. 51: (2/τ0)(v_F/v_B)(E/E_F^h) a_B³ p \|ψ(0)\|⁴, with v_F = √(2E_F^h/m_h) | E | + E_F^h |
| Total | Eq. 53: 1/τ_s = 1/τ_EY + 1/τ_DP + 1/τ_BAP | | |
| Flip rule | Eq. 54: P = ½[1 − exp(−δt/τ_s)], where δt is the **time since the previous real (non-self) scattering**. This is verified from the text after Eq. 54. | | |

---

## 4. Ambiguities, inconsistencies, and missing details

> **Status (after the user decisions):** the current choice for every item below is recorded in
> `docs/MODEL_ASSUMPTIONS.md` and selectable via `ModelAssumptions`.
>
> | Item | Status |
> |---|---|
> | A1 | Resolved: the numerator form reproduces C21 Fig. 7, so the printed Eq. 33 is a typo. |
> | A2 | Resolved: per-initial-valley multiplicities. |
> | A3 | Isotropic. |
> | A4 | Resolved: per-band default; prose mode is optional. |
> | A5 | Adachi 1989 default, with a sub-E0 tail artifact (open). |
> | A10 | Resolved: FD hh+lh bath, exact kinematics, FD Pauli. The C21 step rule breaks detailed balance. |
> | A13 | Resolved: spin frozen in L/X, with bookkeeping. |
> | Self-scattering | Removed for field-free runs. |

**Affects Stage A**

* **A3 — Initial k direction.** C21 does not specify it. [K13] gives only the vertical-transition energy. We
  default to **isotropic**, which is consistent with the spherical-band angular averaging behind Eqs. 12–16.
  Real optical orientation also aligns momentum (hh transitions favour k ⟂ light), but that is not in the paper.
* **A4 — Band and spin assignment at excitation.** Eq. 12 (from D'yakonov–Perel 1971) gives
  ESP0 = Σ P_i K_i / Σ K_i with per-band polarizations P_hh = ½, P_lh → ½ (near the edge), and P_so ≈ −1.
  The C21 text instead says hh electrons go to spin-up and lh/so electrons to spin-down. Taken
  literally with weights K_i, that gives 30%, not 50%, at the edge. These two readings cannot both hold. Options:
  (a) `"chubenko_text"` (**default**): an hh fraction of (1+ESP0)/2, all spin +1; the rest are lh/so
      (split K_lh : K_so), all spin −1. This reproduces ESP0, and it also reproduces the relative peak heights of Fig. 5
      (at 1.90 eV: paper ≈ 6.8 : 3 : 2.2 for hh : lh : so; this model 6.8 : 3.5 : 2.0; per_band 5.9 : 4.3 : 2.4).
      So it is evidently what C21 did.
  (b) `"per_band"`: pick band i with probability K_i/ΣK_i, then spin +1 with probability (1+P_i)/2. This is the
      DP71-consistent reading; it differs in the hh/lh fractions and in the energy–spin correlation.
  Both are implemented. Physically (b) is better motivated, but (a) is what the paper used.
* **A5 — Absorption length l(ħω) and reflectivity R(ħω).** These come from an Adachi-model fit (Fig. 3) whose
  numbers are not given. For now l is a required user input (a scalar or a callable). We will add a
  tabulated optical-data module once you pick a data source.
* **A6 — POP angle.** Eq. 30 is the *unscreened* Fröhlich angular distribution, yet the rate (Eq. 25) is
  screened. Default: Eq. 30, as in C21. Option `pop_angle="screened"` samples the screened distribution exactly
  (by rejection from Eq. 30).
* **A7 — Eq. 31 (POP 1/τ_m).** The sign bookkeeping is hard to audit. We derived the closed form of Eq. 22 for
  the same matrix element and check it against direct numerical quadrature. Eq. 31 is transcribed only in a
  validation script and compared, not trusted.
* **A8 — Degeneracy criterion.** "Fermi level as close to the VBM as 2kT or lower" is read as E_F^b < 2k_BT.
  Under this reading 1.5e18 cm⁻³ is (just) degenerate (E_F^b ≈ 1.7 k_BT), so Thomas–Fermi screening applies there,
  and β is discontinuous at the threshold.
* **A12 — Q_i and DP Matthiessen sum.** The text says Q = 1 for isotropic and 1/6 for anisotropic processes, but
  then uses 1/6 for all. We follow the paper (configurable). The DP total uses Eq. 52, which excludes e–h.
* **A14 — Spin at surface arrival.** Eq. 54 applies only at real scattering events, so the residual time since
  the last event is not applied when an electron reaches z = 0. Default: as in C21. Option: apply Eq. 54 for the
  residual δt at arrival (unbiased for a two-state Markov process).
* **A15 — Eq. 11 broadening.** E0 = ΔE_e ± (3/2) k_BT ln(1−r), with a random sign, and E0 ≤ 0 is replaced by ΔE_e.
  This creates a δ-spike at ΔE_e for so electrons near threshold (visible in Fig. 5). Implemented as written.
* **A16 — ħω < E_g.** No prescription in C21. [K13] uses E0 = 5 meV. We raise an error unless the user opts in.
* **A17 — Elastic acoustic + discrete POP.** With only these two mechanisms an electron's energy stays on the
  comb E0 + nħω0 and can never reach a continuous Maxwellian. This is a property of the model, not an
  ambiguity: Stage A thermalization must be tested on the comb, and e–h scattering (Stage B) provides the
  continuous energy exchange.

**Affects later stages**

* **A1 — Eq. 33 intervalley rate.** The (1 + 2α_j E') factor appears in the *denominator*. The final-state DOS
  ∝ √γ'(1 + 2αE') puts it in the numerator (Vasileska; our derivation), so we believe this is a typo. Plan: numerator, flagged.
* **A2 — Z_j for L→L and X→X.** Table I gives Z_L = 4 and Z_X = 3 "to scatter into". [KT] uses 3 (L→L) and 2 (X→X),
  i.e. excluding the initial valley. Needs a decision in Stage C.
* **A9 — Screening for impurity and e–h scattering.** Presumably the same β as for POP (Debye or TF). TF counts light holes only.
* **A10 — Electron–hole scattering details.**
  (i) C21 says holes are "at rest", but also that a "randomly chosen hole" is used and that Pauli exclusion is
  tested on the final hole state. With k0 = 0 the Pauli test is ill-posed. [K13] samples k0 from thermalized hh and lh
  distributions. Our proposal: sample k0 from the equilibrium hh distribution (Fermi–Dirac with E_F from Eq. 58).
  (ii) p in Eq. 38 is presumably the total p, although only heavy holes participate.
  (iii) E_F^h (Eq. 44) is a T = 0, single-band estimate, inconsistent with Eq. 58.
  (iv) Eq. 43 assumes a parabolic electron band (|g'| = |g|); with α ≠ 0 energy is not exactly conserved, so we
  will measure the error.
  (v) **Erratum [K15]:** it states only that a bug in the e–h implementation affected p ≥ 1e19 results. It does
  **not** describe the bug. Our safeguards: Eqs. 38–43 re-derived independently (done: W_max, the acceptance ratio,
  g = 2 × relative wavevector, and Eq. 43 all check out), plus unit tests that (a) the accepted-event rate equals the
  hole-averaged Born rate, (b) momentum and energy are conserved per collision, and (c) Pauli rejection
  vanishes in the nondegenerate limit.
* **A11 — BAP.** p is the total p or the hh density (text ambiguous; we use the total). Eqs. 50 and 51 are not continuous
  at the regime boundary. |ψ(0)|² = 1.
* **A13 — Spin relaxation in L and X valleys.** Not specified; Eqs. 45–46 are Γ-specific. Options: none in L/X, or the Γ formulas with valley parameters. Needs a decision in Stage C.
* **A18 — Band bending + scattering.** C21 uses a 1 fs velocity-Verlet step in the band-bending region; how that is
  interleaved with scattering is not stated. We use event-driven flights with ≤ 1 fs Verlet substeps.
* **A19 — Doping-dependent E_g.** Assumed to be used everywhere (excitation, EY/DP E/E_g, band bending).

---

## 5. Stage A — smallest model with a quantitative published comparison

**Model.** Γ valley only, nonparabolic, with acoustic + screened POP absorption/emission, no field,
semi-infinite (z > 0), spin carried with EY/DP built from the enabled mechanisms plus BAP, and the full
photoexcitation module.

**Published comparisons possible with Stage A only:**

1. Eq. 10 band gaps vs. the Fig. 6 legend; E_bb and W_bb vs. the Fig. 16 caption (exact numbers).
2. ESP0(ħω) vs. Fig. 6 (analytic: 50% → ≈47% → steep drop at E_g + Δ_so).
3. **Fig. 7(b)/8(b) acoustic and POP scattering and momentum-relaxation rates** at p = 1.5e17 (Debye screening),
   and Fig. 7(a)/8(a) at p = 1e19 (TF screening). This is the primary Stage A benchmark.
4. Per-mechanism EY(ap), EY(pop), DP(ap), DP(pop) vs. Figs. 10–11, and the BAP curves of Fig. 12, which are independent of the scattering set.
5. Fig. 5 shapes (initial energy histograms) at 1.45 / 1.60 / 1.75 / 1.90 eV, given a doping (E_g).

**Exact internal checks:**

* round trips for k(E), E(k), and v = ħ⁻¹ dE/dk; DOS = state counting;
* closed-form rates = numerical golden-rule quadrature with the same matrix element;
* energy changes per event ∈ {0, ±ħω0};
* the stationary comb distribution ∝ DOS · exp(−E/k_BT), and the early-time cooling rate ⟨dE/dt⟩ = −ħω0 (W_em − W_abs).
