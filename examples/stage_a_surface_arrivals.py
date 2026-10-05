"""Stage A demonstration: photoexcite, transport, and collect the surface-arrival ensemble.

STAGE A PHYSICS ONLY (Gamma valley, acoustic + POP, no impurity / e-h / intervalley, no band
bending). The numbers are NOT yet physical predictions. This shows the data flow and the
output format that a separate surface/interface model will consume.

The absorption length is a placeholder (ambiguity A5): replace it with real optical data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import diagnostics as dg
from gaas_mc.constants import NM, PS, ev, per_cm3, to_ev
from gaas_mc.excitation import photoexcite
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import stage_a_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.transport import Simulation

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

sample = Sample(gaas_chubenko2021(), per_cm3(1e19))
mech = stage_a_mechanisms(sample)
spin = SpinModel(sample, mech)
rng = np.random.default_rng(12345)

hw = ev(1.60)
ABSORPTION_LENGTH = 500 * NM      # placeholder, not a measured value
ens = photoexcite(sample, hw, 20_000, rng, ABSORPTION_LENGTH)

sim = Simulation(sample, mech, spin, t_max=370 * PS)     # C21 Sec. IV simulation time
res = sim.run(ens, rng)
arr = res.arrivals
arr.save_npz(OUT / "stageA_arrivals_1e19_1.60eV.npz")

print(f"excited {len(ens)}, ESP0 = {ens.esp():.3f}")
print(f"arrived {len(arr)} ({len(arr) / len(ens):.1%}), timed out {(res.ensemble.status == 2).sum()}")
print(f"arrival ESP = {arr.esp():.3f};  median arrival time = {np.median(arr.t) / PS:.2f} ps")
print(f"mean arrival kinetic energy = {to_ev(arr.E).mean() * 1e3:.1f} meV")
print("real events per mechanism:", dict(zip(res.mechanism_names, res.n_real.tolist())),
      "self:", res.n_self)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
axs[0].hist(to_ev(arr.E), bins=np.linspace(0, 0.35, 71), color=dg.SLOTS[0])
axs[0].set_xlabel("kinetic energy at z = 0, eV"); axs[0].set_ylabel("count")
axs[1].hist(arr.t / PS, bins=np.geomspace(1e-3, 370, 60), color=dg.SLOTS[1])
axs[1].set_xscale("log"); axs[1].set_xlabel("arrival time, ps")
k_perp = np.hypot(arr.k[:, 0], arr.k[:, 1])
axs[2].hist2d(-arr.k[:, 2] * 1e-9, k_perp * 1e-9, bins=60, cmap="Blues")
axs[2].set_xlabel("-k_z, 1/nm"); axs[2].set_ylabel("|k_perp|, 1/nm")
fig.suptitle("Stage A surface arrivals (NOT yet physical: Stage A mechanisms only)", fontsize=10)
dg.save(fig, OUT / "stageA_arrivals.png")
