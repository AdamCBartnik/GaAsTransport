# gaas_mc — Spin-polarized photoemission from p-GaAs

Monte Carlo simulation of photoexcitation, electron transport, spin relaxation, and emission
from p-doped GaAs. The model follows O. Chubenko *et al.*, J. Appl. Phys. **130**, 063101 (2021)
(“C21”), with documented extensions and comparisons to Karkare *et al.* (2013) and its 2015 erratum.

The main workflow produces an ensemble of **electrons emitted into vacuum**: emission times,
positions, momenta, energies, spins, and valley identities. Beam coordinates are returned as an
openPMD-beamphysics `ParticleGroup`, ready for analysis or use as a source in a separate vacuum-side
simulation.

## Start with the emission notebook

[**examples/example.ipynb**](examples/example.ipynb) is the recommended
entry point. Open it in Jupyter, select your Python environment, and run the cells in order:

1. Set the photon energy, doping, surface affinity, layer geometry, and simulation time in `params`.
2. Call `simulate_emission(**params)` to run photoexcitation, transport, and surface emission.
3. Inspect QE, emitted spin polarization, MTE, and the position, momentum, energy, and time distributions.
4. Save the beam, emission records, and settings to `examples/out/particlegroup_tutorial/`.
5. Compare fresh QE/ESP simulations with the digitized simulation curves and experimental data in C21 Fig. 18.

The tutorial explains every `simulate_emission` argument and uses ParticleGroup's native plots.
Its Fig. 18 section starts with a modest five-energy sweep, with an option for the full 16-energy
grid and larger ensembles. Results are cached for repeat runs.

[examples/example_gpt_tools.ipynb](examples/example_gpt_tools.ipynb) preserves the original example
using `GPT_tools.ParticleGroupExtension` and `GPT_tools.gpt_plot`.

### Environment

Run from a checkout of this repository. The notebook locates the repository root automatically
when opened from `examples/`.

| Use | Dependencies |
|---|---|
| Simulation and particle output | NumPy, SciPy, openPMD-beamphysics |
| Compiled CPU engine | Numba |
| CUDA engine | Numba CUDA support (`numba-cuda`), CuPy, a compatible CUDA runtime and NVIDIA GPU |
| Main tutorial | Jupyter and Matplotlib; ordinary inline plots |
| GPT_tools example | Also `ipympl` for `%matplotlib widget` and an importable `GPT_tools` installation |
| Tests | pytest |

`device="auto"` selects CUDA when available, otherwise the compiled CPU engine. Use `"cpu"` or
`"cuda"` to choose explicitly, or `"reference"` for the readable NumPy implementation. The first
compiled run includes compilation time. See [performance and engine details](docs/PERFORMANCE.md)
for measured throughput and parallel execution.

### Run a simulation

This is the same workflow as the notebook, with example settings near the GaAs band edge:

```python
from gaas_mc.emission import simulate_emission

params = dict(
    hv_eV=1.55,
    p_cm3=1e19,
    chi_eV=0.67,
    n=100_000,
    t_max_ps=370.0,
    thickness_nm=None,          # semi-infinite; use e.g. 200 for a finite layer
    R_back=0.0,                 # back reflection probability for a finite layer
    depletion_scattering="bulk",
    matching_mass="band_edge",
    absorption_model=None,     # default Casey+Adachi absorption; "adachi1989" for C21
    sigma_xy=0.0,              # added Gaussian laser spot: rms per axis, in metres
    total_charge=None,         # or a total emitted bunch charge in coulombs
    device="auto",
    seed=1,
)

run = simulate_emission(**params)
pg = run.particle_group

print(f"Generated: {run.n_generated}; emitted: {run.n_emitted}")
print(f"QE: {100 * run.qe:.2f}%; ESP: {100 * run.esp:.1f}%")
```

The tutorial starts with 20,000 generated electrons; increase this for better statistics.
In particular, `t_max_ps` is a hard transport cutoff: a short run measures emission within that
time window and excludes the later tail. `n` counts generated photoelectrons, not incident photons
or emitted particles. `run.qe` includes optical reflection and, for finite layers, absorption in
the layer. Setting `total_charge` changes the exported particle weights, not the calculated QE.

| Parameter | Meaning |
|---|---|
| `hv_eV`, `p_cm3` | Photon energy in eV and acceptor/hole density in cm⁻³ |
| `chi_eV` | Surface affinity relative to the local Gamma conduction edge; effective bulk affinity is `chi − E_bb` |
| `thickness_nm`, `R_back` | Layer thickness and probability of specular reflection at the back; otherwise the electron is lost to the substrate |
| `depletion_scattering` | `"bulk"` for C21 comparisons (driver default); `"local"` for depleted hole density and local scattering/screening |
| `matching_mass` | `"band_edge"` by default; `"velocity"` is a surface-model sensitivity option |
| `absorption_model` | Default: measured Casey data blended with Adachi; use `"adachi1989"` for the C21 benchmark setup |

At p = 1e19 cm⁻³, the C21 band-bending depth is about 0.694 eV, so `chi_eV=0.67` corresponds to
an effective affinity of about −0.024 eV. Full definitions and approximation choices are in
[Model assumptions](docs/MODEL_ASSUMPTIONS.md).

### Analyze and save the emitted particles

Use the returned ParticleGroup directly for beam properties and plots:

```python
import numpy as np
from gaas_mc.constants import M0, C_LIGHT, Q_E

pg = run.particle_group
if len(pg):
    rest_energy_eV = M0 * C_LIGHT**2 / Q_E
    mte_eV = np.average((pg.px**2 + pg.py**2) / (2 * rest_energy_eV), weights=pg.weight)
    print(f"Mean kinetic energy: {1e3 * pg['mean_kinetic_energy']:.1f} meV")
    print(f"MTE: {1e3 * mte_eV:.1f} meV")
    pg.plot("px", "py", bins=50)
    pg.plot("t", bins=70)
```

`run.emissions` retains spin, emission valley, equivalent-valley identity, and the semiconductor
state at emission. Its rows correspond to the original `run.particle_group` rows; preserve that
association if you filter or reorder particles. `run.result` contains transport diagnostics and
final particle statuses. A run with no emission returns an empty `ParticleGroup`, QE = 0, and
an undefined (`NaN`) emitted ESP.

Save the beam from the repository root:

```python
from pathlib import Path
import numpy as np

out = Path("examples/out/particlegroup_tutorial")
out.mkdir(parents=True, exist_ok=True)
pg.write(str(out / "emitted_electrons.h5"))

# Spin and valley are separate from the ParticleGroup's standard beam fields.
em = run.emissions
np.savez(out / "emitted_metadata.npz",
         id=em.pid + 1, spin=em.spin, valley=em.valley, eqv=em.eqv)
```

### Coordinates and units

The simulation places the surface at z = 0, GaAs at positive z, and vacuum at negative z.
The exported `ParticleGroup` reverses z and pz so that the outgoing beam travels along **+z**.

| Output | Convention |
|---|---|
| `pg.x`, `pg.y`, `pg.z` | Metres; z = 0 at emission. x/y include lateral transport in GaAs plus the optional Gaussian laser spot |
| `pg.t` | Seconds after an instantaneous excitation pulse at t = 0; each particle retains its own emission time |
| `pg.px`, `pg.py`, `pg.pz` | Vacuum momentum in eV/c; pz > 0 in the output frame |
| `pg.weight` | Charge in coulombs per macroparticle; sums to `total_charge` when supplied and particles emit |
| `em.E_vac`, `em.E_perp` | Vacuum kinetic and transverse energies in joules; `em.E_perp.mean()` is MTE |
| `em.p_vac` | Vacuum momentum in kg·m/s in the simulation frame, with pz < 0 |
| `em.spin`, `em.valley` | Discrete ±1 spin label along the simulation's +z axis; valley 0/1/2 = Gamma/L/X at emission |

With `sigma_xy=0`, x/y show lateral transport from a point excitation spot. These are surface
emission events, rather than particle positions at a common later time. Vacuum image-charge,
space-charge, and DC gun fields belong in the downstream simulation.

## Model and validation

The implementation includes hh/lh/split-off excitation; nonparabolic Gamma/L/X bands; acoustic,
polar-optical, intervalley, impurity, and electron–hole scattering; Gamma-valley EY/DP/BAP spin
relaxation; C21 band bending and quantum surface transmission; and optional finite-layer back
reflection. Spin is frozen during residence in L/X. NumPy and compiled CPU/CUDA engines share
the physics setup and rate tables.

Validation covers conservation laws, analytic limits, thermalization, engine comparisons, and
published C21 benchmarks. Known approximations and unresolved differences are documented:

- [Model assumptions](docs/MODEL_ASSUMPTIONS.md): defaults, alternatives, and their physical basis.
- [Validation results](docs/VALIDATION.md): scattering, transport, spin, band bending, QE/ESP, and finite layers.
- [Physics audit and response](docs/PHYSICS_AUDIT.md): independent checks, fixes, and remaining limitations.
- [Performance](docs/PERFORMANCE.md): CPU/CUDA engines, parallel runs, and timings.
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md): parameter tables, mechanisms, and development history.
- [References](refs/README.md): source papers; local PDFs are not committed.

Run the tests from the repository root:

```text
python -m pytest tests -q
```

Validation scripts live in `validation/`. Set `PYTHONPATH` to the repository root before running
them (`$env:PYTHONPATH="."` in PowerShell, or `export PYTHONPATH=.` in a POSIX shell). For example:

```text
python validation/stage_bc_rates.py
python validation/excitation.py
python validation/eh_detailed_balance.py
python validation/stage_d_band_bending.py fast
python validation/stage_e_fig18.py fast bulk 100000 cuda
python validation/stage_e_fig18.py report bulk_fast/bmass
```

Generated results go to `validation/out/` and `examples/out/`; both are gitignored.

## Lower-level transport and custom surface models

Use `Simulation` directly when you need custom fields, scattering choices, surface models, or
first-arrival records. Its default absorbing boundary records the first arrival at z = 0;
applying `C21Surface` instead continues through reflection, trapping, and emission, as the
high-level `simulate_emission` driver does.

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

a = ModelAssumptions(depletion_scattering="bulk")
s = Sample(gaas_chubenko2021(), per_cm3(1e19))
field = C21BandBending(s)
mech = build_mechanisms(s, a, stage="C", field=field)
spin = SpinModel(s, [m for m in mech if m.valley_from == 0])
rng = np.random.default_rng(1)
ens = photoexcite(s, ev(1.6), 10_000, rng, assumptions=a)
res = Simulation(s, mech, spin, field=field, t_max=370 * PS, assumptions=a).run(ens, rng)
res.arrivals.save_npz("arrivals.npz")
```

This lower-level API uses SI units. Use `ev()` and `per_cm3()` to convert input energies and densities.
