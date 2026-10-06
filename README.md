# gaas_mc — Monte Carlo of spin-polarized photoexcitation and transport in p-GaAs

This package implements the bulk model of O. Chubenko *et al.*, J. Appl. Phys. **130**, 063101 (2021), ("C21").
S. Karkare *et al.*, J. Appl. Phys. **113**, 104904 (2013) and its 2015 erratum serve as benchmarks.

The main output is the **surface-arrival ensemble**: the time, energy, k, valley, spin, initial depth,
scattering history, and valley residence times of each electron when it first reaches z = 0. No emission
physics is applied, so these records can be handed to a separate surface/interface model. The C21
surface model (Stage E) is available as an optional, replaceable module for benchmarking.

* `docs/IMPLEMENTATION_PLAN.md`: plan, parameter table, mechanism table, original list of ambiguities
* `docs/MODEL_ASSUMPTIONS.md`: **every modelling choice, its default, alternatives, and validation evidence**
* `docs/VALIDATION.md`: comparison with C21 figures (rates, Fig. 6, Fig. 9 drift velocity, Fig. 14 spin relaxation time, Fig. 18 QE/ESP)
* `docs/PERFORMANCE.md`: the two engines (NumPy reference, compiled fast engine for CPU threads / CUDA), measured speeds, parallel and cluster runs
* `refs/README.md`: references (PDFs are kept locally, not committed)
* Units: SI internally. Use `constants.ev()` and `constants.per_cm3()` at the boundary.
* Every formula cites its C21 equation number in the docstrings.

## Status

| Stage | Content | State |
|---|---|---|
| A | Γ valley; acoustic + screened POP; EY/DP/BAP spin; photoexcitation (Eqs. 6–16); surface arrivals | done, validated |
| B | ionized impurity (Brooks–Herring); electron–hole with an hh+lh Fermi–Dirac bath, exact kinematics, FD Pauli blocking | done, validated |
| C | L, X valleys; intervalley (Eq. 33, DOS factor in numerator); spin frozen in L/X with residence-time bookkeeping | done, validated |
| D | C21 band bending (Eqs. 56–62); user potentials; hybrid direct / null-collision flights; adaptive Verlet step | done, validated |
| D′ | Depletion region: local mobile holes (global E_F), local screening of impurity/POP/e–h, local BAP; valley identity + full crystal momentum at arrival | done, validated |
| E | optional C21 surface model (`surface_c21.py`): triangular barrier, transfer matrix, m*→m0, (001) K∥ conservation, trapping, reflection; Fig. 18 benchmark | done, benchmark in `docs/VALIDATION.md` |

## Running

```
python -m pytest -q                                   # unit tests (~4 min)
PYTHONPATH=. python validation/stage_a_rates.py        # C21 Figs. 7, 8, 10-12 (Stage A subset)
PYTHONPATH=. python validation/stage_bc_rates.py       # C21 Figs. 7, 8, 10-13 (full Gamma set)
PYTHONPATH=. python validation/excitation.py           # C21 Figs. 5, 6 (MC-sampled ESP0)
PYTHONPATH=. python validation/absorption.py           # Casey 1975 + Adachi 1989 absorption model
python tools/digitize_casey1975.py                     # regenerate the Casey dataset from the PDF
PYTHONPATH=. python validation/stage_a_cooling.py      # phonon steps, detailed balance (Stage A)
PYTHONPATH=. python validation/eh_detailed_balance.py  # e-h thermalization variants
PYTHONPATH=. python validation/fig9_drift_velocity.py <N> <p_cm3>     # C21 Fig. 9 (one doping), then `plot`
PYTHONPATH=. python validation/fig14_spin_relaxation_time.py <N> <p_cm3> <pauli>   # C21 Fig. 14
PYTHONPATH=. python validation/stage_d_band_bending.py fast   # C21 Figs. 16, 17 + integrator accuracy
PYTHONPATH=. python validation/stage_d_band_bending.py run <p> <hv> <N>; ... plot   # C21 Fig. 20
PYTHONPATH=. python validation/depletion.py profiles | inject | compare   # depletion-region diagnostics
python tools/extract_c21_fig18.py                      # C21 Fig. 18 curves from the PDF's vector graphics
PYTHONPATH=. python validation/stage_e_fig18.py run bulk|local [N] [workers]; ... report bulk local   # C21 Fig. 18
PYTHONPATH=. python examples/surface_arrivals.py [N] [p_cm3] [hv_eV]
```

Outputs go to `validation/out/` and `examples/out/`; both are gitignored.

Fast engine and parallel runs (`docs/PERFORMANCE.md`):

```python
from gaas_mc.fast import FastSimulation
res = FastSimulation(sim, "cuda").run(ensemble, rng)    # or "cpu" (Numba threads); same Result as sim.run
from gaas_mc.parallel import run_parallel                # chunked, max_workers, worker-count independent
```
Optional dependencies: `numba` (fast CPU engine), `numba-cuda` + CUDA 12 runtime (GPU), `cupy` (CuPy array backend).

## Minimal use

```python
import numpy as np
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import ev, per_cm3, PS
from gaas_mc.material import gaas_chubenko2021, Sample
from gaas_mc.scattering import build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.excitation import photoexcite
from gaas_mc.transport import Simulation
from gaas_mc.fields import C21BandBending

a = ModelAssumptions()                         # defaults; see docs/MODEL_ASSUMPTIONS.md
s = Sample(gaas_chubenko2021(), per_cm3(1e19))
field = C21BandBending(s)
mech = build_mechanisms(s, a, stage="C", field=field)   # field -> local depletion-region rates
spin = SpinModel(s, [m for m in mech if m.valley_from == 0])
rng = np.random.default_rng(1)
ens = photoexcite(s, ev(1.6), 10_000, rng, assumptions=a)          # Casey 1975 + Adachi 1989 absorption
res = Simulation(s, mech, spin, field=field, t_max=370 * PS, assumptions=a).run(ens, rng)
res.arrivals.save_npz("arrivals.npz")          # t, E, k, valley, spin, z0, n_events, time_in_valley, ...
print(res.arrivals.upper_valley_summary())
```
