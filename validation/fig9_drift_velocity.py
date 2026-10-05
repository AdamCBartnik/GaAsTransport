"""Steady-state drift velocity vs electric field (cf. [C21] Fig. 9).

Full Stage C model (3 valleys, all mechanisms), uniform field, no boundaries, null-collision
(self-scattering) flights because the field changes E during a flight. Following C21, the
electrons start from a thermal (Maxwellian) distribution in Gamma. The drift velocity is
-d<z>/dt over the window [t_max/2, t_max].

C21 Fig. 9 simulated curves (read off the figure, approximate, peak values):
    p = 1.5e17: peak ~1.45e5 m/s near 4e5 V/m
    p = 1.5e18: ~1.0e5 m/s plateau, 7-10e5 V/m
    p = 1e19:   rising slowly, ~0.4e5 m/s at 10e5 V/m

Run: PYTHONPATH=. python validation/fig9_drift_velocity.py [n_particles]
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import bands
from gaas_mc import diagnostics as dg
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import PS, per_cm3
from gaas_mc.fields import UniformField
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import build_mechanisms
from gaas_mc.transport import Simulation

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()
FIELDS = np.array([0.5, 1, 2, 3, 4, 6, 8, 10, 14]) * 1e5         # V/m
DOPINGS = (1.5e17, 1.5e18, 1e19)
T_MAX = 8 * PS


def thermal_gamma(n, s, rng):
    """Nonparabolic Maxwellian in Gamma: sample E from g(E) exp(-E/kT) by rejection."""
    g = MAT.gamma
    out = np.empty(0)
    while out.size < n:
        E = rng.gamma(1.5, s.kT, 4 * n)
        # proposal ~ sqrt(E) exp(-E/kT); weight ~ sqrt(1 + aE)(1 + 2aE), bounded on E < 20 kT
        w = np.sqrt(1 + g.alpha * E) * (1 + 2 * g.alpha * E)
        wmax = np.sqrt(1 + g.alpha * 20 * s.kT) * (1 + 40 * g.alpha * s.kT)
        keep = (rng.random(E.size) * wmax < w) & (E < 20 * s.kT)
        out = np.concatenate([out, E[keep]])
    E = out[:n]
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, g.m_eff, g.alpha)[:, None]
    return Ensemble.create(z=np.zeros(n), k=k, E=E, spin=1)


def run_doping(p, n):
    rng = np.random.default_rng(int(p / 1e15) + 99)
    times = np.linspace(T_MAX / 2, T_MAX, 9)
    s = Sample(MAT, per_cm3(p))
    mech = build_mechanisms(s, ModelAssumptions(), "C")
    v, up = [], []
    for F in FIELDS:
        t0 = time.time()
        sim = Simulation(s, mech, field=UniformField(F), surface="none", t_max=T_MAX,
                         snapshot_times=times, dt_max_field=5e-15)
        r = sim.run(thermal_gamma(n, s, rng), rng)
        zbar = np.nanmean(r.snapshots.z, axis=1)
        v.append(-np.polyfit(times, zbar, 1)[0])
        up.append(np.mean(r.snapshots.valley[-1] > 0))
        print(f"p={p:.1e} F={F:.1e} V/m  v_d={v[-1]:.3e} m/s  upper-valley fraction={up[-1]:.2f}  "
              f"iters={r.n_iterations}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez(OUT / f"fig9_p{p:.1e}.npz", fields=FIELDS, v=np.array(v), upper=np.array(up), n=n)


def plot():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for j, p in enumerate(DOPINGS):
        f = OUT / f"fig9_p{p:.1e}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        ax.plot(d["fields"] / 1e5, d["v"] / 1e5, "o-", color=dg.SLOTS[j], lw=2, ms=5,
                label=f"p = {p:.1e} cm$^{{-3}}$ (N = {int(d['n'])})")
    ax.set_xlabel("Electric field, 1e5 V/m"); ax.set_ylabel("Drift velocity, 1e5 m/s")
    ax.set_title("Drift velocity, Stage C model (cf. C21 Fig. 9)", fontsize=10)
    ax.grid(True, color="#e4e3dd", lw=0.6); ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, 15); ax.set_ylim(0, 2.2)
    dg.save(fig, OUT / "fig9_drift_velocity.png")


if __name__ == "__main__":
    # usage: fig9_drift_velocity.py <n> <p_cm3>   (one doping; run several in parallel)
    #        fig9_drift_velocity.py plot
    if sys.argv[1:2] == ["plot"]:
        plot()
    else:
        run_doping(float(sys.argv[2]), int(sys.argv[1]))
