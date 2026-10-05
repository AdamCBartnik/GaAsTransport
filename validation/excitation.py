"""Photoexcitation validation against [C21] Figs. 3, 5, and 6.

1. Fig. 6 (required, user decision 1): ESP of Monte Carlo-sampled electrons vs photon energy
   for the default per-band rule (and the prose compatibility rule), overlaid on Eq. 12. Checks
   ~50% at threshold and the drop at Eg + Delta_so.
2. Fig. 5: initial energy histograms (10 meV bins) for both spin rules.
3. Fig. 3: absorption length and reflectivity from the Adachi (1989) model (intrinsic GaAs).

Run: PYTHONPATH=. python validation/excitation.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import diagnostics as dg
from gaas_mc.constants import ev, per_cm3, to_ev
from gaas_mc.excitation import BAND_NAMES, esp0, photoexcite
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.optics import Adachi1989GaAs

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()
CASES = [("Intrinsic", 1e10), ("p = 5e17", 5e17), ("p = 1.7e18", 1.7e18), ("p = 1e19", 1e19)]
N_MC = 40_000


def fig6(rng):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    worst = 0.0
    for j, (lab, p) in enumerate(CASES):
        s = Sample(MAT, per_cm3(p))
        hw = np.linspace(to_ev(s.Eg) + 1e-3, 2.2, 400)
        ax.plot(hw, [100 * esp0(s, ev(h)) for h in hw], color=dg.SLOTS[j], lw=1.6,
                label=f"Eq. 12, {lab} (Eg = {to_ev(s.Eg):.3f} eV)")
        # MC points; photon energies below the Adachi E0 = 1.42 eV are not absorbed (assumption)
        pts = np.array([h for h in np.arange(1.43, 2.21, 0.05)] + [to_ev(s.Eg + MAT.Delta_so) - 0.01,
                                                                  to_ev(s.Eg + MAT.Delta_so) + 0.01])
        for rule, mk in (("per_band", "o"), ("chubenko_prose", "x")):
            y, e = [], []
            for h in pts:
                ens = photoexcite(s, ev(h), N_MC, rng, spin_rule=rule)
                y.append(100 * ens.esp())
                e.append(100 * np.sqrt(max(1 - ens.esp() ** 2, 1e-12) / N_MC))
                worst = max(worst, abs(ens.esp() - esp0(s, ev(h))) / max(e[-1] / 100, 1e-9))
            ax.errorbar(pts, y, yerr=e, fmt=mk, color=dg.SLOTS[j], ms=4, lw=0, elinewidth=1,
                        label=f"MC ({rule})" if j == 0 else None)
    ax.set_xlim(1.4, 2.2); ax.set_ylim(0, 52)
    ax.set_xlabel("hw, eV"); ax.set_ylabel("ESP$_0$, %")
    ax.set_title(f"Initial ESP: Monte Carlo ({N_MC} electrons/point) vs Eq. 12 (cf. C21 Fig. 6)", fontsize=9)
    ax.grid(True, color="#e4e3dd", lw=0.6)
    ax.legend(fontsize=7, frameon=False)
    dg.save(fig, OUT / "fig6_esp0_mc.png")
    print(f"Fig. 6: largest |MC - Eq. 12| = {worst:.2f} standard errors")


def fig5(p, rule, rng):
    s = Sample(MAT, per_cm3(p))
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, hw in zip(axs.flat, (1.45, 1.60, 1.75, 1.90)):
        ens = photoexcite(s, ev(hw), 100_000, rng, spin_rule=rule)
        bins = np.arange(0, 1.0 + 1e-9, 0.010)
        data = [to_ev(ens.E[ens.band == b]) for b in range(3)]
        ax.hist(data, bins=bins, stacked=True, color=dg.SLOTS[:3], label=list(BAND_NAMES),
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"hw = {hw:.2f} eV, ESP0 = {100 * esp0(s, ev(hw)):.1f}%, sampled {100 * ens.esp():.1f}%",
                     fontsize=9)
        ax.set_xlabel("E$_0$, eV"); ax.set_ylabel("counts / 10 meV")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Initial energies (cf. C21 Fig. 5), p = {p:.1e} cm$^{{-3}}$, rule '{rule}'", fontsize=10)
    dg.save(fig, OUT / f"fig5_E0_p{p:.0e}_{rule}.png")


def fig3():
    m = Adachi1989GaAs()
    hv = np.linspace(1.425, 6.0, 2000)
    a = m.absorption_coefficient(ev(hv))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].semilogy(hv, 1e9 / a, color=dg.SLOTS[0], lw=2)
    axs[0].set_xlabel("hw, eV"); axs[0].set_ylabel("absorption length l, nm")
    axs[0].set_title("Adachi (1989), intrinsic GaAs (cf. C21 Fig. 3, blue)", fontsize=9)
    axs[1].plot(hv, m.reflectivity(ev(hv)), color=dg.SLOTS[1], lw=2)
    axs[1].set_xlabel("hw, eV"); axs[1].set_ylabel("R")
    axs[1].set_title("Normal-incidence reflectivity (cf. C21 Fig. 3, red)", fontsize=9)
    for ax in axs:
        ax.grid(True, color="#e4e3dd", lw=0.6)
    dg.save(fig, OUT / "fig3_adachi_absorption.png")
    for h in (1.45, 1.60, 1.75, 1.90, 2.2):
        print(f"Adachi 1989: hw = {h:.2f} eV  l = {1e9 / m.absorption_coefficient(ev(h)):7.1f} nm  "
              f"R = {m.reflectivity(ev(h)):.3f}")


if __name__ == "__main__":
    rng = np.random.default_rng(2021)
    fig3()
    fig6(rng)
    for p in (1e10, 1e19):
        for rule in ("per_band", "chubenko_prose"):
            fig5(p, rule, rng)
    print("figures written to", OUT)
