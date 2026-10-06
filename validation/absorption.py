"""Absorption model validation: digitized Casey (1975) data, the doping-interpolated composite
model, and pure Adachi (1989) for comparison.

Panels:
  (a) Casey spectra as assembled per hole concentration (lines), raw digitized points by source figure
  (b) default composite alpha(hv) for p = 1.5e17, 1e18, 1e19 cm^-3 vs Adachi (dashed); blend window shaded
  (c) absorption length l = 1/alpha (photoexcitation depth scale) vs hv, same models
Printed: seam ratios Adachi/Casey at the blend edges and l at the C21 photon energies.

Run: PYTHONPATH=. python validation/absorption.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import diagnostics as dg
from gaas_mc.constants import EV, ev, per_cm3
from gaas_mc.optical_data import _read_raw, casey1975_ptype
from gaas_mc.optics import Adachi1989GaAs, CaseyAdachiAbsorption

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def main():
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    ax = axs[0]
    ds = sorted(casey1975_ptype(), key=lambda d: d["p"])
    raw = _read_raw()
    marks = {"fig6": "s", "fig7": "^", "fig8": "o"}
    for j, d in enumerate(ds):
        c = dg.SLOTS[j]
        ax.semilogy(d["hv"] / EV, d["alpha"] / 100, "-", color=c, lw=1.5, label=f"p = {d['p'] * 1e-6:.1e}")
        name = [k for k, v in {r[1]: r[3] for r in raw}.items() if abs(v - d["p"] * 1e-6) < 1e-3 * v][0]
        for f, mk in marks.items():
            pts = np.array([(r[4], r[5]) for r in raw if r[1] == name and r[0] == f])
            if pts.size:
                ax.semilogy(pts[:, 0], pts[:, 1], mk, ms=2.5, color=c, mfc="none", alpha=0.6)
    ax.set_xlim(1.30, 1.60); ax.set_ylim(10, 3e4)
    ax.set_xlabel("hv, eV"); ax.set_ylabel("alpha, cm$^{-1}$")
    ax.set_title("Casey, Sell & Wecht (1975), digitized (markers: Fig. 6 s, 7 ^, 8 o)", fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.grid(True, which="both", color="#e4e3dd", lw=0.5)

    ad = Adachi1989GaAs()
    hv_ad = np.linspace(1.4201, 2.0, 600)
    for k, (ax, quantity) in enumerate(((axs[1], "alpha"), (axs[2], "length"))):
        for j, p in enumerate((1.5e17, 1e18, 1e19)):
            m = CaseyAdachiAbsorption(per_cm3(p))
            hv = np.linspace(m.hv_min / EV + 1e-4, 2.0, 1500)
            a = m.absorption_coefficient(ev(hv)) / 100
            y = a if quantity == "alpha" else 1e7 / a
            ax.semilogy(hv, y, color=dg.SLOTS[j], lw=2, label=f"Casey+Adachi, p = {p:.1e}")
            if k == 0:
                print(f"p = {p:.1e}: seam alpha_Adachi/alpha_Casey = {m.seam_ratio[0]:.3f} (1.550 eV), "
                      f"{m.seam_ratio[1]:.3f} ({m.b1 / EV:.3f} eV); interpolation between "
                      f"{[d['label'] for d in m.near.used]}, w = {m.near.w:.2f}")
                print("   l [nm]: " + "  ".join(f"{e:.2f}:{1e9 / m.absorption_coefficient(ev(e)):.0f}"
                                               for e in (1.40, 1.45, 1.60, 1.65, 1.75, 1.90) if ev(e) >= m.hv_min))
        a_ad = ad.absorption_coefficient(ev(hv_ad)) / 100
        ax.semilogy(hv_ad, a_ad if quantity == "alpha" else 1e7 / a_ad, "--", color="#2b2b29", lw=1.5,
                    label="Adachi (1989) alone")
        ax.axvspan(1.55, 1.592, color="#cfcfc9", alpha=0.5, lw=0)
        ax.set_xlabel("hv, eV")
        ax.set_ylabel("alpha, cm$^{-1}$" if quantity == "alpha" else "absorption length l, nm")
        ax.set_xlim(1.33, 2.0)
        ax.set_title("Default composite (shaded: blend window)" if quantity == "alpha"
                     else "Absorption length (cf. C21 Fig. 3)", fontsize=9)
        ax.legend(fontsize=7, frameon=False); ax.grid(True, which="both", color="#e4e3dd", lw=0.5)
    print("Adachi alone: l [nm] " + "  ".join(f"{e:.2f}:{1e9 / ad.absorption_coefficient(ev(e)):.0f}"
                                            for e in (1.45, 1.60, 1.75, 1.90)))
    dg.save(fig, OUT / "absorption_casey_adachi.png")
    print("figure written to", OUT)


if __name__ == "__main__":
    main()
