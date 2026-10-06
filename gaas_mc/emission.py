"""Emitted-electron output as an openPMD-beamphysics ParticleGroup, and a one-call driver.

    from gaas_mc.emission import simulate_emission
    run = simulate_emission(hv_eV=1.6, p_cm3=1e19, chi_eV=0.67, n=100_000)
    run.particle_group          # ParticleGroup of the emitted electrons
    run.qe, run.esp             # quantum efficiency, spin polarization of the emitted electrons

Coordinates of the ParticleGroup (output convention only; the simulation itself is unchanged):
  * The simulation has the GaAs at z > 0 and emits toward -z. For the output, z -> -z and
    p_z -> -p_z, so the emitted beam travels along +z.
  * z = 0 for every particle (the emitting surface); t = emission time [s] after the (delta-function)
    excitation pulse at t = 0.
  * px, py, pz: vacuum momentum just outside the surface model [eV/c] (C21 surface: transverse
    crystal momentum folded into the surface Brillouin zone and conserved, energy from the vacuum
    level). Image-charge acceleration, space charge and any applied field are not included.
  * x, y = (excitation point in the laser spot) + (lateral displacement inside the GaAs between
    excitation and emission). The excitation point is drawn from a round Gaussian of rms sigma_xy [m]
    per axis (sigma_xy = 0: a point source, so x, y show the lateral diffusion alone). The lateral
    displacement is tracked during transport (bookkeeping: the sample is laterally homogeneous).
  * weight [C]: total_charge / N_emitted per macroparticle if total_charge is given, otherwise the
    electron charge (each macroparticle is one electron).
  * status = 1, species = "electron", id = the simulation's particle id.
Spin, emission valley and the other per-electron records are not part of a ParticleGroup; they are
in the returned EmissionRun (run.emissions).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .constants import C_LIGHT, PS, Q_E, ev, per_cm3

try:                                            # newer package name
    from beamphysics import ParticleGroup
except ImportError:                             # pragma: no cover
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from pmd_beamphysics import ParticleGroup


def to_particle_group(emissions, sigma_xy=0.0, rng=None, total_charge=None):
    """ParticleGroup of emitted electrons (gaas_mc.surface.Emissions); see the module docstring for
    the coordinate convention (z -> -z, p_z -> -p_z: the beam travels along +z)."""
    n = len(emissions)
    if n == 0:
        raise ValueError("no emitted electrons")
    p = np.asarray(emissions.p_vac, float) * C_LIGHT / Q_E            # kg m/s -> eV/c
    x = np.asarray(emissions.x, float).copy() if emissions.x is not None else np.zeros(n)
    y = np.asarray(emissions.y, float).copy() if emissions.y is not None else np.zeros(n)
    if sigma_xy > 0:
        rng = rng if rng is not None else np.random.default_rng()
        x += rng.normal(0.0, sigma_xy, n)
        y += rng.normal(0.0, sigma_xy, n)
    weight = (total_charge / n) if total_charge is not None else Q_E
    data = dict(x=x, px=p[:, 0], y=y, py=p[:, 1],
                z=np.zeros(n), pz=-p[:, 2],                             # output convention: beam along +z
                t=np.asarray(emissions.t, float), status=np.ones(n, int),
                weight=np.full(n, float(weight)), species="electron",
                id=np.asarray(emissions.pid, int) + 1)
    if np.any(data["pz"] <= 0):
        raise RuntimeError("emitted electron with p_z <= 0 in the output frame")
    return ParticleGroup(data=data)


@dataclass
class EmissionRun:
    particle_group: object        # ParticleGroup of the emitted electrons
    emissions: object             # gaas_mc.surface.Emissions (inside state, spin, valley, ...)
    result: object                # transport.Result of the run
    n_generated: int              # photoexcited electrons simulated
    absorbed_fraction: float      # fraction of incident photons absorbed in the GaAs (A_layer or 1 - R)
    qe: float                     # emitted electrons per incident photon
    esp: float                    # spin polarization of the emitted electrons
    settings: dict

    @property
    def n_emitted(self):
        return len(self.emissions)


def simulate_emission(hv_eV, p_cm3=1e19, chi_eV=0.67, n=100_000, t_max_ps=370.0, thickness_nm=None,
                      R_back=0.0, depletion_scattering="bulk", matching_mass="band_edge",
                      absorption_model=None, device="auto", seed=0, sigma_xy=0.0, total_charge=None):
    """Photoexcite n electrons at photon energy hv_eV in p-GaAs (doping p_cm3), transport them with
    the C21 band bending and the C21 surface model (electron affinity chi_eV), and return the emitted
    electrons as a ParticleGroup (EmissionRun.particle_group).

    thickness_nm: None = semi-infinite cathode; otherwise a GaAs layer 0 < z < d with a back boundary
                  of reflection coefficient R_back (gaas_mc/back.py).
    depletion_scattering: "bulk" (C21 baseline) or "local". matching_mass: "band_edge" (default) or
                  "velocity". absorption_model: None = the ModelAssumptions default (Casey 1975 +
                  Adachi 1989); "adachi1989" as in C21.
    device: "auto" (GPU if available, else compiled CPU), "cuda", "cpu" (fast engine) or "reference"
            (NumPy reference engine).
    sigma_xy: rms laser spot size per axis [m] for the output x, y; total_charge [C]: bunch charge
              for the ParticleGroup weights (default: one electron per macroparticle).
    """
    from .assumptions import ModelAssumptions
    from .back import PartialReflector
    from .excitation import layer_absorption, photoexcite
    from .fields import C21BandBending
    from .material import Sample, gaas_chubenko2021
    from .scattering import build_mechanisms
    from .spin import SpinModel
    from .surface_c21 import C21Surface
    from .transport import Simulation

    kw = dict(depletion_scattering=depletion_scattering)
    if absorption_model is not None:
        kw["absorption_model"] = absorption_model
    a = ModelAssumptions(**kw)
    mat = gaas_chubenko2021()
    s = Sample(mat, per_cm3(p_cm3))
    field = C21BandBending(s)
    mech = build_mechanisms(s, a, "C", field=field)
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    d = None if thickness_nm is None else float(thickness_nm) * 1e-9
    surface = C21Surface(chi=ev(chi_eV), material=mat, matching_mass=matching_mass)
    sim = Simulation(s, mech, sm, field=field, surface=surface, z_back=d,
                     back=PartialReflector(R_back) if d is not None else "absorb",
                     t_max=float(t_max_ps) * PS, assumptions=a)
    rng = np.random.default_rng(seed)
    ens = photoexcite(s, ev(hv_eV), int(n), rng, assumptions=a, thickness=d)
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if device == "reference":
        res = sim.run(ens, rng)
    elif device in ("cuda", "cpu"):
        from .fast import FastSimulation
        res = FastSimulation(sim, device).run(ens, rng)
    else:
        raise ValueError(f"device {device!r}")
    em = res.emissions
    A = layer_absorption(s, ev(hv_eV), d, assumptions=a)
    pg = to_particle_group(em, sigma_xy=sigma_xy, rng=rng, total_charge=total_charge)
    settings = dict(hv_eV=hv_eV, p_cm3=p_cm3, chi_eV=chi_eV, n=int(n), t_max_ps=t_max_ps,
                    thickness_nm=thickness_nm, R_back=R_back if d is not None else None,
                    depletion_scattering=depletion_scattering, matching_mass=matching_mass,
                    absorption_model=a.absorption_model, device=device, seed=seed, sigma_xy=sigma_xy)
    return EmissionRun(particle_group=pg, emissions=em, result=res, n_generated=int(n),
                       absorbed_fraction=A, qe=A * len(em) / int(n),
                       esp=float(em.spin.mean()) if len(em) else float("nan"), settings=settings)


def _cuda_available():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False
