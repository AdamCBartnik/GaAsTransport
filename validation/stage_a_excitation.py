"""Stage A validation 6-7: initial ESP (cf. [C21] Fig. 6) and initial energy histograms (Fig. 5).

Fig. 5 does not state its doping, so it is drawn for p = 1e19 cm^-3 and for intrinsic Eg,
with both spin-assignment models (ambiguity A4).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import diagnostics as dg
from gaas_mc.constants import NM, ev, per_cm3, to_ev
from gaas_mc.excitation import BAND_NAMES, esp0, photoexcite
from gaas_mc.material import Sample, gaas_chubenko2021

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()


def fig6():
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cases = [("Intrinsic", 1e10), ("p = 5e17", 5e17), ("p = 1.7e18", 1.7e18), ("p = 1e19", 1e19)]
    for j, (lab, p) in enumerate(cases):
        s = Sample(MAT, per_cm3(p))
        hw = np.linspace(to_ev(s.Eg) + 1e-3, 2.2, 400)
        y = [100 * esp0(s, ev(h)) for h in hw]
        ax.plot(hw, y, color=dg.SLOTS[j], lw=2, label=f"{lab} (Eg = {to_ev(s.Eg):.3f} eV)")
    ax.set_xlim(1.4, 2.2)
    ax.set_ylim(0, 52)
    ax.set_xlabel("hw, eV")
    ax.set_ylabel("ESP$_0$, %")
    ax.set_title("Initial ESP, Eqs. 12-16 (cf. C21 Fig. 6)", fontsize=10)
    ax.grid(True, color="#e4e3dd", lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    dg.save(fig, OUT / "stageA_fig6_esp0.png")


def fig5(p, model, rng):
    s = Sample(MAT, per_cm3(p))
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, hw in zip(axs.flat, (1.45, 1.60, 1.75, 1.90)):
        ens = photoexcite(s, ev(hw), 100_000, rng, 1000 * NM, spin_model=model)
        bins = np.arange(0, 1.0 + 1e-9, 0.010)                       # 10 meV, as in Fig. 5
        data = [to_ev(ens.E[ens.band == b]) for b in range(3)]
        ax.hist(data, bins=bins, stacked=True, color=dg.SLOTS[:3], label=list(BAND_NAMES),
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"hw = {hw:.2f} eV,  ESP0 = {100 * esp0(s, ev(hw)):.1f}%,  sampled "
                     f"{100 * ens.esp():.1f}%", fontsize=9)
        ax.set_xlabel("E$_0$, eV")
        ax.set_ylabel("counts / 10 meV")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Initial energies (cf. C21 Fig. 5), p = {p:.1e} cm$^{{-3}}$ (Eg = {to_ev(s.Eg):.3f} eV), "
                 f"spin model '{model}'", fontsize=10)
    dg.save(fig, OUT / f"stageA_fig5_E0_p{p:.0e}_{model}.png")


if __name__ == "__main__":
    rng = np.random.default_rng(2021)
    fig6()
    for p in (1e10, 1e19):
        for model in ("per_band", "chubenko_text"):
            fig5(p, model, rng)
    print("figures written to", OUT)
