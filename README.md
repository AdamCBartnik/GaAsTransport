# gaas_mc — Monte Carlo of spin-polarized photoexcitation and transport in p-GaAs

This package implements the bulk model of O. Chubenko *et al.*, J. Appl. Phys. **130**, 063101 (2021), ("C21").
S. Karkare *et al.*, J. Appl. Phys. **113**, 104904 (2013) and its 2015 erratum serve as benchmarks. The
reference PDFs are in `refs/`.

The main output is the **surface-arrival ensemble**: the time, energy, k, valley, spin, initial depth, and
scattering history of each electron when it first reaches z = 0. No emission physics is applied, so these
records can be handed to a separate surface/interface model.

* Plan, parameter table, mechanism table, and the list of ambiguities: `docs/IMPLEMENTATION_PLAN.md`
* Units: SI internally. Use `constants.ev()` and `constants.per_cm3()` at the boundary.
* Every formula cites its C21 equation number in the docstrings.

## Status

| Stage | Content | State |
|---|---|---|
| A | Γ valley; acoustic + screened POP; EY/DP/BAP spin; photoexcitation; field-free transport; surface arrivals | **implemented, validated** |
| B | ionized impurity, electron–hole (+ Pauli) | todo |
| C | L, X valleys, intervalley | todo |
| D | C21 band bending (Eqs. 56–62) | todo (generic `fields.py` exists) |
| E | optional C21 surface barrier, for benchmarking only | todo |

## Running

```
python -m pytest -q                          # unit tests
python validation/stage_a_rates.py           # C21 Figs. 7, 8, 10-12 analogues -> validation/out/
python validation/stage_a_excitation.py      # C21 Figs. 5, 6
python validation/stage_a_cooling.py         # cooling / phonon steps / detailed balance
python examples/stage_a_surface_arrivals.py  # data-flow demo (Stage A physics only)
```

The scripts import `gaas_mc` from the repository root. Run them from there with `PYTHONPATH=.`.

## Minimal use

```python
import numpy as np
from gaas_mc.constants import ev, per_cm3, NM, PS
from gaas_mc.material import gaas_chubenko2021, Sample
from gaas_mc.scattering import stage_a_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.excitation import photoexcite
from gaas_mc.transport import Simulation

s = Sample(gaas_chubenko2021(), per_cm3(1e19))
mech = stage_a_mechanisms(s)                 # drop any mechanism from the list to disable it
rng = np.random.default_rng(1)
ens = photoexcite(s, ev(1.6), 10_000, rng, absorption_len=500 * NM)
res = Simulation(s, mech, SpinModel(s, mech), t_max=370 * PS).run(ens, rng)
res.arrivals.save_npz("arrivals.npz")       # t, E, k, valley, spin, z0, n_events, ...
```
