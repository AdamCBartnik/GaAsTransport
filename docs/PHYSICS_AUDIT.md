# Physics and code audit, 2026-10-06

Findings were recorded before changing the implementation. Measurements and final verification
are recorded below. This is an audit of the checked-out code, not a certification
of the underlying phenomenological model. The existing modified emission notebook is user work.

## Prioritized findings

1. **Definite: folded transverse momentum has the wrong direction.** `C21Surface.interact` and
   the compiled `emit` calculate the length of the shortest surface reciprocal-lattice representative,
   then scale the *unfolded* vector to that length. This is not a reciprocal-lattice translation.
   Two representations K and K+G produce different emitted px/py. A reproducible example gives a
   57% px difference, with identical energy and MTE. This affects directions and correlations when
   folding is nontrivial; normal Gamma/X[001] emission at the benchmark energies is usually unfolded.
   Fix both paths to retain the minimizing vector. Existing tests only check its length.

2. **Definite: snapshots inside an aggregated bounce train are missing.** `_bounce_trains` advances
   the clock before `_record_snapshots` sees it. Valid, alive electrons then have NaN energies and
   valley -1 at requested times; density and conditional spin diagnostics can be biased by 100%
   for a selected grazing population. Use explicit returns whenever snapshots are requested.
   This changes diagnostics and performance, not the intended stochastic transport law.

3. **Definite: FastSimulation ignores `surface_bounce_aggregation=False`.** The flag is absent
   from packed configuration. Supposed fast explicit/aggregated comparisons can silently compare
   aggregation with itself. Pass the flag through and honor it in the CPU/CUDA source.

4. **Definite: vertical excitation fails in the parabolic limit.** `excess_energy` divides by alpha
   and subtracts nearly equal numbers. alpha=0 produces NaN; very small alpha violates the vertical
   energy/momentum equation. Rationalize Eq. 9. No intended change to physical GaAs parameters.

5. **Definite validation weakness: dimensionally meaningless absolute tolerances.** Several tests
   compare joule energies, second times, or coulomb charges with default `atol=1e-8`. For energies
   of order 1e-20 J, even replacing the result by zero passes. In particular the e-h conservation
   and surface kinetic-energy assertions do not establish their claimed relative precision.
   Strengthen the relevant checks and use independently scaled output residuals.

## Important concerns requiring model judgment, not automatic replacement

* Spin flips are deferred to accepted collisions, using the final pre-collision rate for all
  time since the preceding accepted collision. This implements the paper's Eq. 54 convention,
  including no flip at null events and frozen L/X. With `spin_flip_at_arrival=False`, the last
  interval is never applied to an emitted electron. Aggregation also accumulates this time without
  an emission flip, so it shares the explicit algorithm's limitation. This is not skipped scattering
  caused by aggregation. For a collisionless spin lifetime tau, recorded ESP stays at its initial
  value instead of exp(-t/tau); the effect depends on residual time/rate. Changing this silently
  would change a documented compatibility choice. A continuous integrated spin hazard is separate work.
* `2*hbar*kz/|F(0)|` is exact for constant force and any isotropic dispersion; the C21 force varies
  with z. The short-return criterion limits the excursion, making the constant-force treatment an
  approximation, not an exact identity for the C21 potential. The brute-force comparison also uses
  that same short-return approximation. Its agreement cannot independently validate the approximation.
* Crossing-time acceleration uses the velocity mass m(1+2 alpha E), not the full longitudinal
  differential mass. At finite kz the latter includes the derivative of E in the denominator.
  Crossing momentum is repaired by energy conservation, but this alone does not establish accurate
  crossing time. Step refinement is needed for timing tails; do not infer it from energy conservation.
* Electron-hole kinematics conserve energy after solving the scalar root, but the nonparabolic
  collision probability remains a parabolic cross section. Exact conservation is not detailed balance.
  Rejecting F(0)>=0 is a restriction to rays bracketing a positive root, not proof that no positive
  root exists: for a convex energy function a ray can intersect twice even with F(0)>0. It is tied
  to the chosen nonparabolic collision extension; changing the root prescription alone would not
  restore the missing scattering Jacobian.
* Default POP uses the screened total rate and unscreened Eq. 30 angle. The computed momentum rate
  uses the screened kernel. This is a documented C21 compatibility approximation, but transport
  angular relaxation and the rate entering DP/EY are consequently not the same stochastic kernel.
* The band-edge interface mass intentionally gives the parabolic BDD current used for transmission,
  rather than the Kane group velocity. Retained as requested; matching QE does not uniquely validate
  the physical surface boundary condition.
* A 370 ps deterministic cutoff is not an exponentially sampled recombination lifetime. Timing and
  QE are conditional on that termination model. The finite layer retains the semi-infinite front
  potential and intentionally omits optical interference and substrate transport.

The e-h root-domain claim is demonstrably false, even though its effect in the default thermal bath
was not resolved statistically. In a constructed Gamma/hh state with electron E=0.4700064 eV and
nearly comoving momenta, s=1.776198934e7 /m is a positive energy-conserving root (residual 6.1e-16 eV),
yet `_solve_s` returns `ok=False`. `validation/audit_edges.py` reproduces it. Among 100,000 proposals
at each of 0.01, 0.3 and 1 eV, none failed this domain condition. No numerical bound on the general
distribution bias follows from that small rare-event sample. The restriction is preserved pending
a physically justified nonparabolic collision measure/root selection, rather than pretending that
a different root choice alone fixes detailed balance.

Additional edge cases not repaired in this audit: exactly E=0 is masked out by the transport's
strict threshold test even for finite e-h candidate rates; a field flight can exit the energy table
before the next loop-start range check; and `simulate_emission` raises when no electrons emit,
preventing a valid zero-QE run from returning its summary. These do not explain the benchmark
curves but should be covered before treating the public API as robust for arbitrary inputs.

## Energy and normalization trace

| Stage | Energy reference / conversion |
|---|---|
| Excitation | E = kinetic energy above the local Gamma minimum; vertical transition uses Eg(p) and valence offset |
| Dispersion | E(1+alpha E)=hbar²k²/(2m); k is measured from the current valley center |
| Intervalley | E' = E +/- phonon energy - (Delta_destination-Delta_source) |
| Field flight | Conserved global energy is E + Delta_valley + U(z), U=-E_bb(1-z/W)² |
| Surface | E_tot = E + Delta_valley, referred to local Gamma edge; chi is local affinity |
| Vacuum | K_vac=E+Delta_valley-chi; global vacuum level is -E_bb+chi |
| Transverse | E_perp=hbar²|K_parallel folded|²/(2m0); pz<0 from remaining vacuum energy |

No missing/double valley or band-bending offset was found in this trace. Band bending is already
in the surface kinetic energy and must not be added again. Photoexcitation samples conditional
depths. Absolute QE uses `(1-R)*(1-exp(-alpha*d))*N_emit/N_generated`, with the finite-layer
factor omitted for a semi-infinite sample. Surface transmission is a Bernoulli decision and is
not multiplied into QE a second time. Exported ParticleGroup charges are a separate user-selected
bunch normalization. The optional exporter changes the sign of pz as documented; transport does not.

## Source comparison and retained choices

Read the local Chubenko accepted manuscript, Karkare 2013 and 2015 erratum. Visually inspected
Chubenko Fig. 15: the triangular-barrier height is measured above the asymptotic vacuum level.
The sign and offset in `chi + E_b*(1-x/L_b)` agree with that diagram. Eq. 54 explicitly excludes
self-scattering when defining the elapsed time. The erratum identifies an e-h implementation bug
but does not specify its algorithmic cause; it also corrects the emission-angle statement and
piezoelectric constants. This project does not implement Karkare's surface randomization as its
C21 surface model.

The intervalley factor belongs in the numerator independently of the printed Eq. 33:
state counting gives k² dk/dE proportional to sqrt(E(1+alpha E))*(1+2 alpha E).
The destination multiplicity is applied once in the rate prefactor, with same-family exclusion
handled in equivalent-valley sampling. FD initial holes and final vacancy blocking, optional step
blocking, bulk/local depletion choices, band-edge matching mass, and frozen L/X spin are retained.

Sources: [Chubenko accepted manuscript](https://www.osti.gov/servlets/purl/1869631),
[published article](https://doi.org/10.1063/5.0060151),
[Karkare 2013](https://doi.org/10.1063/1.4794822),
[Karkare 2015 erratum](https://doi.org/10.1063/1.4914297).

## Test quality and remaining scope

* Rectangular-barrier and pure mass-step analytic tests are useful independent checks. Added a
  continuum triangular-barrier Schrödinger ODE comparison: four energies from chi+1 micro-eV to
  1.5 eV agree within 1e-5 relative with 400 slices. This independently checks current normalization
  and the triangular profile; CPU/reference formula agreement alone would not do so.
* Nonparabolic band E/k inverses, velocity derivative and DOS state counting are covered. The
  intervalley golden-rule test shares `bands.dos`, but the separate DOS state-counting test and
  analytic derivative justify the numerator. Rate plots are visual comparisons, not numerical
  acceptance tests against digitized Figs. 7-13 at every point.
* Existing e-h conservation tests now actually constrain relative energy errors; thermalization
  tests already checked a binned parabolic distribution. The new audit compares a continuous
  nonparabolic CDF against independent dimensionless DOS integration, not merely 3kT/2.
* Existing bounce A/B tests checked counts, emission fraction and mean time, not spin relaxation
  or all joint correlations. Added a compiled aggregation-on/off execution check with time and
  energy conservation. The geometric trial count is cut off at the sampled null-collision time
  and at t_max; the remaining flight ends at the original candidate collision. No weighting,
  skipped candidate collision, or duplicate post-termination event was found in that path.
* Back-wall tests check specular momentum components, absorption once, the R_back fraction,
  truncated Beer-Lambert CDF and layer QE normalization. The existing equilibrium-density test
  starts from parabolic energy sampling despite Kane dispersion; its broad density tolerance
  should not be read as an exact joint equilibrium test. The thin-film equilibrium-mean test
  correctly integrates the nonparabolic DOS, but is still a mean test.
* Six independent Hamiltonian ODE transits (0.01/0.1/0.5 eV, normal cosine 0.05/1) gave maximum
  absolute crossing-time error 0.00379 fs at 0.25 fs steps and 0.000921 fs at 0.125 fs steps.
  These support accuracy for the tested transits, not a universal bound. The short-bounce C21
  constant-force approximation remains distinct from the exact uniform-force solution.
* Full drift-velocity Fig. 9, long spin-decay Fig. 14, local-depletion production curves and the
  100,000-electron thin-film thickness study were not regenerated. Those kernels were not changed;
  the full existing tests and relevant rate/boundary/emission benchmarks were rerun. No claim is
  made to have reproduced every historical validation output. Far-tail/high-energy, arbitrary
  surface models, and full multi-dimensional distribution equality remain unproven.

## Before/after emitted distributions

Same seed, 5,000 generated electrons per row, GPU, 370 ps, p=1e19 /cm³, chi=0.67 eV,
bulk depletion and Adachi absorption. These are measurements at modest statistics, not fitted values.

| hv (eV) | QE (%) | ESP (%) | Mean vacuum energy (meV) | MTE (meV) | Median emission time (ps) |
|---|---:|---:|---:|---:|---:|
| 1.45 | 4.5204 | 27.7108 | 96.7734 | 12.8387 | 58.7364 |
| 1.60 | 7.2119 | 23.1638 | 102.2623 | 15.2355 | 43.8756 |
| 1.80 | 9.0662 | 13.6499 | 120.8463 | 16.9156 | 27.6053 |
| 2.20 | 11.5266 | 2.2676 | 163.5341 | 28.9018 | 3.8087 |

All four rows had identical emitted IDs and spins before and after. Differences per particle were
at most 4.3e-15 eV in vacuum energy and 5.2e-11 fs in time; normalized momentum differences were
below 7.8e-14 in units sqrt(2m0 eV). Mean MTE differences were below 1e-15 eV. Thus the corrected
direction does not change these sampled benchmark ensembles beyond round-off, and their recorded
joint correlation matrices are unchanged to numerical accuracy. It still corrects the general
folding invariant, including possible upper-valley/high-energy or alternate-affinity emission.

Fresh 20,000-electron e-h-only tests at 12 ps gave FD mean E/kT=1.50947 ±0.00868 versus 1.55732
from Kane DOS weighting (3.07% low). The Kolmogorov distance from that equilibrium CDF was 0.01427.
Step blocking gave 1.19562 ±0.00712 and distance 0.12669. The quoted uncertainties are sample mean
standard errors; the nonparabolic model does not pass exact thermal equilibrium. FD remains the
intended and better default.

## Final verification record

Python: C:/ProgramData/miniforge3/python.exe. NumPy/Numba/CUDA available.
Existing suite launched before changes. Baseline scripts completed: stage_a_rates,
stage_bc_rates, excitation, absorption, stage_a_cooling, stage_d_band_bending fast.
Baseline GPU output: four 5,000-particle runs, hv=1.45/1.60/1.80/2.20 eV, bulk depletion,
chi=0.67 eV, band-edge interface mass, Adachi optics, 370 ps.
Independent vacuum energy residual: at most 1.6e-16 eV.
Detailed outputs and per-particle joint distributions: validation/out/audit/.

* Initial complete suite: 148 passed, 1 skipped, 1 failed and 2 errors in 361 s. All three
  non-passing cases encountered Windows sandbox permissions (temporary directories or named pipes).
  A subsequent intermediate parallel repeat overlapped source edits and is not a valid baseline.
* Final full collected suite with workspace-local temporary files and multiprocessing access:
  **164 passed, 1 skipped in 414 s**. The skip is an inaccessible optical band test case, not CUDA.
  Five more independent checks were added after collection; running the expanded audit file gave
  **18 passed**. Across both runs, all 169 applicable final tests passed, with one skipped case.
* Fresh post-fix rate/excitation/band-bending scripts completed. CUDA exercised the same scalar
  transport source as CPU and passed the existing GPU comparison test.
* Joint engine check: 3,000 generated electrons per engine, hv=1.9 eV, bulk depletion, chi=.67 eV,
  depths capped at 60 nm, 5 ps. Emitted counts: NumPy 374, CPU 363, CUDA 391. Across 44 first/second
  moments and mixed products of (t,px,py,pz,E,E_perp,spin,X-valley indicator), maximum absolute
  z-score versus NumPy was 1.90 (CPU) and 2.85 (CUDA); RMS 0.87 and 1.18. No discrepancy detected
  at this sample size. These tests constrain mixed moments, not full equality of distributions.
* Fig. 18 was rerun at all 16 photon energies, all four affinities and both matching masses,
  with 5,000 generated electrons per energy (not the historical 100,000). For bulk/band-edge mass,
  mean QE ratios to the digitized C21 curves were **0.989 / 1.003 / 1.008 / 1.076** for
  chi=0.64/0.67/0.70/0.73 eV. Mean ESP offsets were -0.21/-0.38/+1.08/-4.61 percentage points.
  The high-affinity ESP discrepancy remains; the modest sample size makes individual ESP points
  noisy. Output: `validation/out/audit/fig18/fig18_summary_bulk_fast-bmass.txt` and adjacent plots.

Reproduction commands (set PYTHONPATH to the repository root):

```text
C:/ProgramData/miniforge3/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=validation/out/audit/pytest-new
C:/ProgramData/miniforge3/python.exe validation/audit_joint.py fixed 5000
C:/ProgramData/miniforge3/python.exe validation/audit_joint.py equilibrium
C:/ProgramData/miniforge3/python.exe validation/audit_joint.py engines
C:/ProgramData/miniforge3/python.exe validation/audit_edges.py
```

The audit has improved numerical and implementation confidence in the sampled emission output.
It does not establish physical exactness: nonparabolic e-h balance, residual-time spin treatment,
surface matching assumptions and extreme tails remain relevant to the intended downstream source.

## Response (2026-10-07)

All five definite findings were confirmed and the fixes kept as made in the audit:
the folded transverse-momentum vector (K_par + G for the minimizing surface reciprocal-lattice
vector, not the unfolded direction rescaled), snapshots with bounce aggregation, the fast engine's
`surface_bounce_aggregation` switch, the rationalized Eq. 9 (algebraically identical to the printed
form, finite at alpha = 0) and the dimensionless test tolerances. Full suite: 169 passed, 1 skipped.

Two of the edge cases listed as not repaired are now fixed (tests in `test_audit_regressions.py`):
`simulate_emission` returns an empty ParticleGroup and QE = 0 when nothing is emitted, and both
engines stop with "electron energy left the rate table" when a field flight ends above E_table_max
(previously the following scattering used clamped rates).

Two of the concerns, quantified (p = 1e19, chi = 0.67 eV, 1e5 electrons, GPU):

* **Unapplied residual spin interval at emission** (`spin_flip_at_arrival=False`, the C21
  convention): the time since the last real collision at emission has median ≈ 5 fs (90 %: 15 fs).
  Emitted electrons are hot (~0.7 eV), where D'yakonov-Perel relaxation is fast, so the mean omitted
  flip probability is 0.9 %, i.e. the reported ESP is high by ≈ 0.4 % abs. at 1.55 eV (ESP 24.8 %) and
  ≈ 0.2 % at 1.9 eV (ESP 10.5 %), below the ≈ 1 % statistical errors of the benchmark. Default kept
  (C21 compatibility); `spin_flip_at_arrival=True` applies Eq. 54 at every encounter.
* **Constant-force short returns:** a return is treated analytically only when
  t_ret = 2 hbar k_z/|F(0)| <= dt_max = 0.25 fs, i.e. k_z <= 2.7e7 /m, a normal energy <= 0.43 meV and
  an excursion <= 0.003 nm. Over that height the C21 field changes by <= 3e-4 (relative), so the
  approximation error of the bounce time is of that order.

The remaining concerns (nonparabolic e-h detailed balance and root domain, POP rate/angle
consistency, the deterministic 370 ps cutoff, the band-edge matching mass) are model choices
documented in `docs/MODEL_ASSUMPTIONS.md`; they are not changed without a decision.
