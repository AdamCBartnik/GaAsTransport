"""Spin relaxation time from the decay of the internal ESP (cf. [C21] Eq. 55 and Fig. 14).

As in C21: photoexcitation at hw = 1.65 eV, emission prohibited (here: specular reflection at
z = 0), full Stage C model, ESP_int(t) of electrons inside the material, tau_s from an
exponential fit. C21 reports 110 ps (1.5e17), 92 ps (1.5e18), 77 ps (1e19) with their model
(step-function Pauli rule, Eq. 44), about 1.4-1.8x longer than experiment.

Run: PYTHONPATH=. python validation/fig14_spin_relaxation_time.py [n_particles]
"""
import sys
import time
from pathlib import Path

import numpy as np

from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import PS, ev, per_cm3
from gaas_mc.excitation import photoexcite
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.transport import Simulation

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()
C21 = {1.5e17: 110, 1.5e18: 92, 1e19: 77}
T_MAX = 300 * PS


def run(p, pauli, n, rng):
    a = ModelAssumptions(pauli_blocking=pauli)
    s = Sample(MAT, per_cm3(p))
    mech = build_mechanisms(s, a, "C")
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    times = np.arange(0, 301, 10) * PS
    ens = photoexcite(s, ev(1.65), n, rng, assumptions=a)
    sim = Simulation(s, mech, sm, surface="reflect", t_max=T_MAX, snapshot_times=times, assumptions=a)
    r = sim.run(ens, rng)
    esp = np.array([r.snapshots.esp(i) for i in range(times.size)])
    fit = times >= 30 * PS
    slope, icpt = np.polyfit(times[fit], np.log(esp[fit]), 1)
    return -1 / slope, esp, times, r


def main(n, p, pauli):
    rng = np.random.default_rng(int(p / 1e15) + (0 if pauli == "fermi_dirac" else 7))
    t0 = time.time()
    tau, esp, times, r = run(p, pauli, n, rng)
    line = (f"pauli={pauli:12s} p={p:.1e}: tau_s = {tau / PS:6.1f} ps (C21: {C21[p]} ps); "
            f"ESP(0)={esp[0]:.3f} ESP(100ps)={esp[10]:.3f} ESP(300ps)={esp[-1]:.3f}  "
            f"[N={n}, {time.time() - t0:.0f}s, iters {r.n_iterations}]")
    print(line, flush=True)
    np.savez(OUT / f"fig14_{pauli}_p{p:.1e}.npz", times=times, esp=esp, tau=tau, n=n)
    with open(OUT / "fig14_spin_relaxation_time.txt", "a") as fh:
        fh.write(line + chr(10))


if __name__ == "__main__":
    # usage: fig14_spin_relaxation_time.py <n> <p_cm3> <pauli: fermi_dirac|step_c21>
    main(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
