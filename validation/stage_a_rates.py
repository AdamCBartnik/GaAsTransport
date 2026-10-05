"""Stage A validation 1-3: scattering rates, momentum relaxation rates, spin relaxation rates.

Compare with [C21] Fig. 7 (rates), Fig. 8 (tau_m), Figs. 10/11 (EY/DP per mechanism), and
Fig. 12 (BAP). Only the acoustic and POP curves (and BAP) are comparable at Stage A;
impurity, e-h, and intervalley arrive in Stages B and C, so the Stage A "total" spin curves
are NOT comparable to the paper's totals.

Run:  python validation/stage_a_rates.py      -> validation/out/*.png + printed table
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import diagnostics as dg
from gaas_mc.constants import ev, per_cm3, to_ev
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import stage_a_mechanisms
from gaas_mc.spin import SpinModel

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()
E = ev(np.linspace(1e-4, 1.0, 600))
DOPINGS = [("a", 1e19), ("b", 1.5e17)]          # panel labels as in [C21] Figs. 7, 8, 10-12


def main():
    fig_r, axs_r = plt.subplots(2, 1, figsize=(6, 8.5))
    fig_m, axs_m = plt.subplots(2, 1, figsize=(6, 8.5))
    fig_s, axs_s = plt.subplots(2, 3, figsize=(15, 8.5))
    print(f"{'p':>8} {'E[eV]':>6} | {'W_ap':>9} {'W_abs':>9} {'W_em':>9} | {'tm_abs':>9} {'tm_abs31':>9} "
          f"{'tm_em':>9} {'tm_em31':>9} | {'BAP':>9}")
    for i, (lab, p) in enumerate(DOPINGS):
        s = Sample(MAT, per_cm3(p))
        mech = stage_a_mechanisms(s)
        ac, pa, pe = mech
        sm = SpinModel(s, mech)
        tag = f"p = {p:.1e} cm$^{{-3}}$ ({s.screening_model}, L = {1e9 / s.beta:.1f} nm)"

        ax = axs_r[i]
        for m in mech:
            dg.plot_curve(ax, E, m.rate(E), m.name)
        dg.setup_axes(ax, "Scattering rate, s$^{-1}$")
        ax.set_title(f"({lab}) Stage A rates (cf. C21 Fig. 7{lab}) — {tag}", fontsize=9)
        ax.legend(fontsize=8, frameon=False)

        ax = axs_m[i]
        for m in mech:
            dg.plot_curve(ax, E, m.momentum_rate(E), m.name)
        for m in (pa, pe):
            ax.plot(to_ev(E), np.where(m.tau_m_eq31(E) > 0, m.tau_m_eq31(E), np.nan), ":",
                    color="#2b2b29", lw=1.2, label=f"{m.name} via Eq. 31 (literal)")
        dg.setup_axes(ax, "Momentum relaxation rate, s$^{-1}$")
        ax.set_title(f"({lab}) Stage A 1/tau_m (cf. C21 Fig. 8{lab}) — {tag}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)

        ey, dp = sm.ey_by_mechanism(E), sm.dp_by_mechanism(E)
        ax = axs_s[i, 0]
        for name, y in ey.items():
            dg.plot_curve(ax, E, y, name, label=f"EY ({name})")
        dg.setup_axes(ax, "Spin relaxation rate, s$^{-1}$", ylim=(1e5, 1e13))
        ax.set_title(f"({lab}) EY per mechanism (cf. Fig. 10{lab})", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax = axs_s[i, 1]
        for name, y in dp.items():
            dg.plot_curve(ax, E, y, name, label=f"DP ({name})")
        dg.setup_axes(ax, "Spin relaxation rate, s$^{-1}$", ylim=(1e5, 1e15))
        ax.set_title(f"({lab}) DP per mechanism (cf. Fig. 11{lab})", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax = axs_s[i, 2]
        dg.plot_curve(ax, E, sm.bap(E), "BAP", label="BAP (Eqs. 47/50/51)")
        dg.plot_curve(ax, E, sm.ey(E), "EY", label="EY, Stage A mechanisms only")
        dg.plot_curve(ax, E, sm.dp(E), "DP", label="DP, Stage A mechanisms only")
        if s.degenerate:
            ax.axvline(to_ev(sm.bap_regime_boundary()), color=dg.MUTED, lw=1, ls="--")
        dg.setup_axes(ax, "Spin relaxation rate, s$^{-1}$", ylim=(1e5, 1e13))
        ax.set_title(f"({lab}) BAP comparable to Fig. 12{lab}; EY/DP totals are NOT (no ii, ij)",
                     fontsize=8)
        ax.legend(fontsize=7, frameon=False)

        for e in (0.05, 0.1, 0.3, 0.5, 1.0):
            x = ev(np.array([e]))
            print(f"{p:8.1e} {e:6.2f} | {ac.rate(x)[0]:9.2e} {pa.rate(x)[0]:9.2e} {pe.rate(x)[0]:9.2e} | "
                  f"{pa.momentum_rate(x)[0]:9.2e} {pa.tau_m_eq31(x)[0]:9.2e} {pe.momentum_rate(x)[0]:9.2e} "
                  f"{pe.tau_m_eq31(x)[0]:9.2e} | {sm.bap(x)[0]:9.2e}")
        print("   sample:", {k: (round(float(v), 5) if not isinstance(v, (bool, str, np.bool_)) else v)
                             for k, v in s.summary().items()})

    dg.save(fig_r, OUT / "stageA_fig7_rates.png")
    dg.save(fig_m, OUT / "stageA_fig8_momentum_rates.png")
    dg.save(fig_s, OUT / "stageA_fig10-12_spin_rates.png")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
