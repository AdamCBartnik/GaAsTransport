"""Stage A validation 4-5: hot-electron cooling in infinite homogeneous GaAs (Gamma valley,
acoustic + screened POP only, no field, no boundary).

Checks:
  * energy changes come in exact quanta of hw0 (or 0 for acoustic);
  * early-time cooling rate d<E>/dt matches -hw0 [W_em(E) - W_abs(E)] at the injection energy;
  * long-time populations of the comb E0 + n hw0 match g(E_n) exp(-E_n/kT) (detailed balance);
    a continuous Maxwellian is unreachable with these two mechanisms alone (ambiguity A17).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import bands
from gaas_mc import diagnostics as dg
from gaas_mc.constants import PS, ev, per_cm3, to_ev
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import stage_a_mechanisms
from gaas_mc.transport import Simulation

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()
G = MAT.gamma


def run(p, E_inj, n, rng):
    s = Sample(MAT, per_cm3(p))
    mech = stage_a_mechanisms(s)
    times = np.concatenate([[0], np.geomspace(0.01, 100, 60)]) * PS
    sim = Simulation(s, mech, surface="none", t_max=100 * PS, snapshot_times=times, log_events=True)
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(np.full(n, E_inj), G.m_eff, G.alpha)[:, None]
    ens = Ensemble.create(z=np.full(n, 1.0), k=k, E=np.full(n, E_inj), spin=1)
    return s, mech, sim.run(ens, rng)


def main():
    rng = np.random.default_rng(7)
    E_inj = ev(0.30)
    n = 20_000
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for j, p in enumerate((1.5e17, 1e19)):
        s, mech, r = run(p, E_inj, n, rng)
        _, pa, pe = mech
        sn = r.snapshots
        meanE = np.nanmean(sn.E, axis=1)
        axs[0].semilogx(sn.times[1:] / PS, to_ev(meanE[1:]), color=dg.SLOTS[j], lw=2,
                        label=f"MC, p = {p:.1e}")
        slope = -MAT.hw0 * (pe.rate(E_inj)[0] - pa.rate(E_inj)[0])          # J/s
        tt = np.array([0.01, 0.3]) * PS
        axs[0].semilogx(tt / PS, to_ev(E_inj + slope * tt), "--", color=dg.SLOTS[j], lw=1.2,
                        label=f"initial slope -hw0(W_em-W_abs), p = {p:.1e}")
        # measured early slope from the first snapshots
        early = sn.times < 0.2 * PS
        fit = np.polyfit(sn.times[early], meanE[early], 1)[0]
        print(f"p={p:.1e}: predicted dE/dt = {to_ev(slope) * 1e-12:.4f} eV/ps, "
              f"MC = {to_ev(fit) * 1e-12:.4f} eV/ps, ratio {fit / slope:.3f}")
        # comb populations at the final time vs detailed balance
        Ef = sn.E[-1]
        E0 = E_inj - np.floor(E_inj / MAT.hw0) * MAT.hw0
        nidx = np.round((Ef - E0) / MAT.hw0).astype(int)
        comb = E0 + np.arange(nidx.max() + 1) * MAT.hw0
        w = bands.dos(comb, G.m_eff, G.alpha) * np.exp(-comb / s.kT)
        cnt = np.bincount(nidx)
        off = 0.004 * (1 if j else -1)
        axs[1].bar(to_ev(comb[:10]) + off, cnt[:10] / cnt.sum(), width=0.008, color=dg.SLOTS[j],
                   label=f"MC t=100 ps, p = {p:.1e}")
        axs[1].plot(to_ev(comb[:10]) + off, (w / w.sum())[:10], "k_", ms=14, mew=2)
        print(f"  <E>(100 ps) = {to_ev(meanE[-1]) * 1e3:.2f} meV;  comb prediction "
              f"{to_ev(np.sum(w * comb) / w.sum()) * 1e3:.2f} meV;  (3/2)kT = {to_ev(1.5 * s.kT) * 1e3:.2f} meV")
        dE = to_ev(r.event_log["E_after"] - r.event_log["E_before"]) * 1e3
        axs[2].hist(dE, bins=np.linspace(-45, 45, 181), color=dg.SLOTS[j], alpha=0.7,
                    label=f"p = {p:.1e}")
    axs[0].set_xlabel("time, ps"); axs[0].set_ylabel("<E>, eV")
    axs[0].set_title("Cooling from 0.30 eV (Stage A)", fontsize=10)
    axs[0].legend(fontsize=7, frameon=False)
    axs[1].set_xlabel("E, eV"); axs[1].set_ylabel("fraction")
    axs[1].set_title("Comb populations vs g(E)exp(-E/kT) (black ticks)", fontsize=10)
    axs[1].legend(fontsize=7, frameon=False)
    axs[2].set_yscale("log"); axs[2].set_xlabel("dE per real event, meV")
    axs[2].set_title("Energy change per event (expect 0, +-35.36 meV)", fontsize=10)
    axs[2].legend(fontsize=7, frameon=False)
    for ax in axs:
        ax.grid(True, color="#e4e3dd", lw=0.6)
    dg.save(fig, OUT / "stageA_cooling.png")
    print("figure written to", OUT)


if __name__ == "__main__":
    main()
