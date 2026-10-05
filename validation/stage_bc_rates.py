"""Stage B/C validation: full Gamma-valley rate set vs [C21] Figs. 7, 8, 10, 11, 12, 13.

Mechanisms: acoustic, POP abs/em, ionized impurity, Gamma->L and Gamma->X abs/em (Eq. 33 with
the DOS factor in the numerator), plus the electron-hole *accepted* rate (MC estimate; e-h is not
plotted in C21 Fig. 7, and it is excluded from tau_m in Eq. 52).

Run: PYTHONPATH=. python validation/stage_bc_rates.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import diagnostics as dg
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import PS, ev, per_cm3, to_ev
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import ElectronHole, build_mechanisms
from gaas_mc.spin import SpinModel

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()
E = ev(np.linspace(1e-4, 1.0, 800))
DOPINGS = [("a", 1e19), ("b", 1.5e17)]


def gamma_mechs(s):
    mech = build_mechanisms(s, ModelAssumptions(), "C")
    return [m for m in mech if m.valley_from == 0]


def main():
    rng = np.random.default_rng(3)
    fig_r, axs_r = plt.subplots(2, 1, figsize=(6.5, 9))
    fig_m, axs_m = plt.subplots(2, 1, figsize=(6.5, 9))
    fig_s, axs_s = plt.subplots(2, 3, figsize=(16, 9))
    for i, (lab, p) in enumerate(DOPINGS):
        s = Sample(MAT, per_cm3(p))
        mech = gamma_mechs(s)
        sm = SpinModel(s, mech)
        tag = f"p = {p:.1e} cm$^{{-3}}$"
        ax = axs_r[i]
        for m in mech:
            if isinstance(m, ElectronHole):
                Es = ev(np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]))
                w = [m.effective_rate(e, rng, 20000) for e in Es]
                c, mk = dg.style_for(m.name)
                ax.plot(to_ev(Es), w, ":", color=c, marker=mk, lw=1.5, label=f"{m.name} accepted (not in C21 Fig. 7)")
            else:
                dg.plot_curve(ax, E, m.rate(E), m.name)
        dg.setup_axes(ax, "Scattering rate, s$^{-1}$")
        ax.set_title(f"({lab}) Gamma-valley rates (cf. C21 Fig. 7{lab}), {tag}", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, ncol=2)

        ax = axs_m[i]
        for m in mech:
            if m.spin_class is not None:
                dg.plot_curve(ax, E, m.momentum_rate(E), m.name)
        dg.setup_axes(ax, "Momentum relaxation rate, s$^{-1}$")
        ax.set_title(f"({lab}) 1/tau_m (cf. C21 Fig. 8{lab}), {tag}", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, ncol=2)

        ax = axs_s[i, 0]
        ey = sm.ey_by_mechanism(E)
        # group POP/intervalley like C21 Fig. 10 (intervalley summed)
        iv = sum(v for k, v in ey.items() if k.startswith("iv_"))
        for k, v in ey.items():
            if not k.startswith("iv_"):
                dg.plot_curve(ax, E, v, k, label=f"EY ({k})")
        dg.plot_curve(ax, E, iv, "iv_em[Gamma->L]", label="EY (intervalley, summed)")
        dg.plot_curve(ax, E, sm.ey(E), "total", label="EY (total)")
        dg.setup_axes(ax, "Spin relaxation rate, s$^{-1}$", ylim=(1e5, 1e13))
        ax.set_title(f"({lab}) EY (cf. Fig. 10{lab})", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False)

        ax = axs_s[i, 1]
        dp = sm.dp_by_mechanism(E)
        rm_iv = sum(m.momentum_rate(E) for m in mech if m.name.startswith("iv_"))
        for k, v in dp.items():
            if not k.startswith("iv_"):
                dg.plot_curve(ax, E, v, k, label=f"DP ({k})")
        with np.errstate(divide="ignore"):
            dp_iv = np.where(rm_iv > 0, MAT.Q_DP * sm.dp_factor(E) / np.where(rm_iv > 0, rm_iv, 1), 0)
        dg.plot_curve(ax, E, dp_iv, "iv_em[Gamma->L]", label="DP (intervalley, summed tau_m)")
        dg.plot_curve(ax, E, sm.dp(E), "total", label="DP (total, Eq. 52)")
        dg.setup_axes(ax, "Spin relaxation rate, s$^{-1}$", ylim=(1e5, 1e15))
        ax.set_title(f"({lab}) DP (cf. Fig. 11{lab})", fontsize=9)
        ax.legend(fontsize=6.5, frameon=False)

        ax = axs_s[i, 2]
        dg.plot_curve(ax, E, sm.ey(E), "EY", label="EY")
        dg.plot_curve(ax, E, sm.dp(E), "DP", label="DP")
        dg.plot_curve(ax, E, sm.bap(E), "BAP", label="BAP")
        dg.plot_curve(ax, E, sm.total(E), "total", label="Total")
        dg.setup_axes(ax, "Spin relaxation rate, s$^{-1}$", ylim=(1e5, 1e13))
        ax.set_title(f"({lab}) totals (cf. Fig. 12{lab})", fontsize=9)
        ax.legend(fontsize=7, frameon=False)

        print(f"\np = {p:.1e}: E[eV]  ii  tm_ii  iv_em(G->L)  iv_em(G->X)  EY  DP  BAP  total")
        for e in (0.05, 0.2, 0.5, 1.0):
            x = ev(np.array([e]))
            d = {m.name: m for m in mech}
            print(f"  {e:4.2f} {d['impurity[Gamma]'].rate(x)[0]:9.2e} {d['impurity[Gamma]'].momentum_rate(x)[0]:9.2e} "
                  f"{d['iv_em[Gamma->L]'].rate(x)[0]:9.2e} {d['iv_em[Gamma->X]'].rate(x)[0]:9.2e} "
                  f"{sm.ey(x)[0]:9.2e} {sm.dp(x)[0]:9.2e} {sm.bap(x)[0]:9.2e} {sm.total(x)[0]:9.2e}")

    dg.save(fig_r, OUT / "stageC_fig7_rates.png")
    dg.save(fig_m, OUT / "stageC_fig8_momentum_rates.png")
    dg.save(fig_s, OUT / "stageC_fig10-12_spin_rates.png")

    # Fig. 13: spin relaxation time vs energy for three dopings
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j, p in enumerate((1e19, 1.5e18, 1.5e17)):
        s = Sample(MAT, per_cm3(p))
        sm = SpinModel(s, gamma_mechs(s))
        ax.semilogy(to_ev(E), 1 / sm.total(E) / PS, color=dg.SLOTS[j], lw=2, label=f"p = {p:.1e} cm$^{{-3}}$")
        print(f"tau_s(p={p:.1e}) at 0.01/0.1/0.5 eV [ps]:",
              [round(float(1 / sm.total(ev(np.array([e])))[0] / PS), 2) for e in (0.01, 0.1, 0.5)])
    ax.set_xlim(0, 1); ax.set_ylim(0.1, 1e5)
    ax.set_xlabel("Electron energy, eV"); ax.set_ylabel("Spin relaxation time, ps")
    ax.set_title("tau_s(E) (cf. C21 Fig. 13)", fontsize=10)
    ax.grid(True, color="#e4e3dd", lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    dg.save(fig, OUT / "stageC_fig13_tau_s.png")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
