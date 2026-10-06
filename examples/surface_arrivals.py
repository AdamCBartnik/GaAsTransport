"""Photoexcite, transport, and collect the surface-arrival ensemble (full bulk model + band bending).

Physics: 3 valleys; acoustic, POP, ionized impurity, e-h (hh + lh, FD), intervalley; EY/DP/BAP
spin relaxation in Gamma, spin frozen in L/X; C21 surface band bending (Eqs. 56-62); absorption from
Casey (1975) + Adachi (1989). Model choices are listed in docs/MODEL_ASSUMPTIONS.md.
No emission physics: the output is the state of each electron when it reaches z = 0.

Output: examples/out/arrivals_*.npz (SurfaceArrivals) + summary figure.

Usage: PYTHONPATH=. python examples/surface_arrivals.py [n_electrons] [p_cm3] [hv_eV] [flat]
       ("flat" switches the band bending off)
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import diagnostics as dg
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import PS, ev, per_cm3, to_ev
from gaas_mc.excitation import photoexcite
from gaas_mc.fields import C21BandBending, NoField
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.transport import Simulation

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
p = float(sys.argv[2]) if len(sys.argv) > 2 else 1e19
hv = float(sys.argv[3]) if len(sys.argv) > 3 else 1.60
flat = len(sys.argv) > 4 and sys.argv[4] == "flat"

assumptions = ModelAssumptions()                       # all defaults, see docs/MODEL_ASSUMPTIONS.md
sample = Sample(gaas_chubenko2021(), per_cm3(p))
mech = build_mechanisms(sample, assumptions, "C")
spin = SpinModel(sample, [m for m in mech if m.valley_from == 0])
rng = np.random.default_rng(12345)

ens = photoexcite(sample, ev(hv), n, rng, assumptions=assumptions)
field = NoField() if flat else C21BandBending(sample)
sim = Simulation(sample, mech, spin, field=field, t_max=370 * PS, assumptions=assumptions)   # C21 Sec. IV
t0 = time.time()
res = sim.run(ens, rng)
arr = res.arrivals
tag = f"p{p:.0e}_{hv:.2f}eV" + ("_flat" if flat else "")
arr.save_npz(OUT / f"arrivals_{tag}.npz")

print(f"p = {p:.1e} cm^-3, hv = {hv:.2f} eV, N = {n}, flight mode {res.flight_mode}, {time.time() - t0:.0f} s")
print(f"ESP0 (excited)          = {ens.esp():.3f}")
print(f"arrived at z = 0        = {len(arr)} ({len(arr) / n:.1%}); timed out {(res.ensemble.status == 2).sum()}")
print(f"ESP at arrival          = {arr.esp():.3f}")
print(f"arrival time median     = {np.median(arr.t) / PS:.1f} ps; mean {arr.t.mean() / PS:.1f} ps")
print(f"mean kinetic energy     = {to_ev(arr.E).mean() * 1e3:.1f} meV above the local CBM "
      f"(surface CBM is {to_ev(arr.band_edge_at_surface) * 1e3:.0f} meV relative to bulk)")
print("upper-valley participation:", {k: (f"{v:.3g}") for k, v in arr.upper_valley_summary().items()})
names = res.mechanism_names
print("real events:", {nm: int(c) for nm, c in zip(names, res.n_real) if c}, " null:", res.n_self)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
axs[0].hist(to_ev(arr.E) * 1e3, bins=np.linspace(0, 1000 if not flat else 200, 101), color=dg.SLOTS[0])
axs[0].set_xlabel("kinetic energy at z = 0, meV"); axs[0].set_ylabel("count")
axs[1].hist(arr.t / PS, bins=np.geomspace(0.01, 370, 60), color=dg.SLOTS[1])
axs[1].set_xscale("log"); axs[1].set_xlabel("arrival time, ps")
for j, (lab, m) in enumerate((("spin +1", arr.spin > 0), ("spin -1", arr.spin < 0))):
    axs[2].hist(arr.t[m] / PS, bins=np.geomspace(0.01, 370, 40), histtype="step", lw=2,
                color=dg.SLOTS[j], label=lab)
axs[2].set_xscale("log"); axs[2].set_xlabel("arrival time, ps"); axs[2].legend(frameon=False)
fig.suptitle(f"Surface arrivals ({'flat band' if flat else 'C21 band bending'}): {tag}", fontsize=10)
dg.save(fig, OUT / f"arrivals_{tag}.png")
