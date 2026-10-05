"""Detailed-balance check of electron-hole scattering alone (no phonons, no impurities).

Electrons start at 0.15 eV in Gamma and interact only with the equilibrium hh+lh hole bath. The
stationary state must be the lattice-temperature Maxwellian of the electron band:
    parabolic:     <E>/kT = 1.5
    nonparabolic:  <E>/kT = Int E g(E) e^{-E/kT} / Int g(E) e^{-E/kT} = 1.556 (alpha = 0.61/eV)
The run compares Pauli rules and e-h electron-mass models.

Usage: PYTHONPATH=. python validation/eh_detailed_balance.py <variant-index>   (or no arg: all)
"""
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from scipy import integrate

from gaas_mc import bands
from gaas_mc.constants import PS, ev, per_cm3
from gaas_mc.holes import HoleGas
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import ElectronHole
from gaas_mc.transport import Simulation

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()
MAT_PAR = dataclasses.replace(MAT, valleys=(dataclasses.replace(MAT.gamma, alpha=0.0),) + MAT.valleys[1:])
VARIANTS = [  # (label, material, p, pauli, mass_model, t_end)
    ("parabolic   FD     band_edge", MAT_PAR, 1e19, "fermi_dirac", "band_edge", 12),
    ("parabolic   step   band_edge", MAT_PAR, 1e19, "step_c21", "band_edge", 12),
    ("parabolic   none   band_edge", MAT_PAR, 1e19, "none", "band_edge", 12),
    ("nonparab.   FD     band_edge", MAT, 1e19, "fermi_dirac", "band_edge", 12),
    ("nonparab.   FD     vel_mass ", MAT, 1e19, "fermi_dirac", "velocity_mass", 12),
    ("nonparab.   FD     band_edge", MAT, 1.5e17, "fermi_dirac", "band_edge", 40),
    ("nonparab.   FD     vel_mass ", MAT, 1.5e17, "fermi_dirac", "velocity_mass", 40),
]


def mb_mean(alpha, kT):
    g = lambda E: np.sqrt(E * (1 + alpha * E)) * (1 + 2 * alpha * E) * np.exp(-E / kT)
    return integrate.quad(lambda E: E * g(E), 0, 60 * kT)[0] / integrate.quad(g, 0, 60 * kT)[0] / kT


def run(i, n=8000):
    lab, mat, p, pauli, mass, t_end = VARIANTS[i]
    s = Sample(mat, per_cm3(p))
    hg = HoleGas(s)
    mech = [ElectronHole(s, hg, b, 0, pauli=pauli, mass_model=mass) for b in ("hh", "lh")]
    rng = np.random.default_rng(100 + i)
    g = mat.gamma
    E = np.full(n, ev(0.15))
    ens = Ensemble.create(z=np.ones(n), k=bands.random_unit_vectors(n, rng) * bands.k_of_E(E, g.m_eff, g.alpha)[:, None],
                          E=E, spin=1)
    t0 = time.time()
    ts = np.array([0.5, 1.0]) * t_end * PS
    r = Simulation(s, mech, surface="none", t_max=t_end * PS, snapshot_times=ts).run(ens, rng)
    x = [np.nanmean(r.snapshots.E[j]) / s.kT for j in range(2)]
    line = (f"{lab}  p={p:.1e}: <E>/kT = {x[1]:.3f} (at t/2: {x[0]:.3f}); expected {mb_mean(g.alpha, s.kT):.3f}"
            f"  [N={n}, {time.time() - t0:.0f}s]")
    print(line, flush=True)
    with open(OUT / "eh_detailed_balance.txt", "a") as fh:
        fh.write(line + chr(10))


if __name__ == "__main__":
    for i in ([int(sys.argv[1])] if len(sys.argv) > 1 else range(len(VARIANTS))):
        run(i)
