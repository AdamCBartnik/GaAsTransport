"""Depletion-region diagnostics versus depth z, and the local-vs-bulk comparison of surface arrivals.

  profiles   python validation/depletion.py profiles
      For p = 1e19 and 5e17 cm^-3 (C21 band bending), versus z:
        band edges (E_C, E_V) and the global Fermi level; p(z)/p_bulk (hh, lh); screening
        wavevector beta(z) for both screening caps (and uncapped); the accepted electron-hole rate at
        fixed electron energies; the ionized-impurity rate; the BAP spin-relaxation time.
  compare    python validation/depletion.py compare
      Compares surface-arrival ensembles from validation/stage_d_band_bending.py runs for
      depletion = bulk / local(impurity_spacing cap) / local(band_bending_width cap):
      Gamma/L/X fractions, arrival energy distributions, polar-angle distributions, arrival-time
      (response) distributions, ESP.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import diagnostics as dg
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import NM, PS, ev, per_cm3, to_ev
from gaas_mc.depletion import LocalMechanism
from gaas_mc.fields import C21BandBending
from gaas_mc.holes import HoleGas
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import build_mechanisms
from gaas_mc.spin import SpinModel

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()
E_FIXED = (0.05, 0.2, 0.5)                   # eV


def profiles():
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 6, figsize=(26, 8.5))
    for row, p in enumerate((1e19, 5e17)):
        s = Sample(MAT, per_cm3(p))
        f = C21BandBending(s)
        hg = HoleGas(s)
        mech = build_mechanisms(s, ModelAssumptions(), "C", field=f)
        mech_w = build_mechanisms(s, ModelAssumptions(depletion_screening_cap="band_bending_width"), "C", field=f)
        d = [m for m in mech if isinstance(m, LocalMechanism)][0].depletion
        d_w = [m for m in mech_w if isinstance(m, LocalMechanism)][0].depletion
        z = np.linspace(0, 1.4 * f.W, 300)
        fi = d.frac_index(z)
        idx = np.arange(d.n)
        zn = z / NM
        tag = f"p = {p:.0e} cm$^{{-3}}$ (W = {f.W / NM:.1f} nm)"

        ax = axs[row, 0]
        EC = to_ev(f.band_edge(z))
        ax.plot(zn, EC, color=dg.SLOTS[0], lw=2, label="E_C(z) - E_C,bulk")
        ax.plot(zn, EC - to_ev(s.Eg), color=dg.SLOTS[1], lw=2, label="E_V(z)")
        ax.axhline(-to_ev(s.Eg) - to_ev(hg.mu), color="#2b2b29", ls="--", lw=1.2,
                   label="global E_F (hh+lh FD; mu>0 means inside the VB)")
        ax.set_title(f"band edges, {tag}", fontsize=9); ax.set_ylabel("E, eV")

        ax = axs[row, 1]
        for key, c, lab in (("p", "#2b2b29", "total"), ("p_hh", dg.SLOTS[0], "hh"), ("p_lh", dg.SLOTS[1], "lh")):
            ax.semilogy(zn, np.interp(fi, idx, getattr(d, key)) / hg.p, color=c, lw=2, label=lab)
        ax.set_ylim(1e-13, 2); ax.set_title("p(z) / p_bulk", fontsize=9)

        ax = axs[row, 2]
        ax.plot(zn, np.interp(fi, idx, d.beta_uncapped) * 1e-9, color=dg.MUTED, lw=1.5, ls=":", label="uncapped (FD chi)")
        ax.plot(zn, np.interp(fi, idx, d.beta) * 1e-9, color=dg.SLOTS[0], lw=2,
                label=f"cap: impurity spacing (L_cap {d.L_cap / NM:.1f} nm)")
        ax.plot(zn, np.interp(fi, idx, d_w.beta) * 1e-9, color=dg.SLOTS[1], lw=2, ls="--",
                label=f"cap: W_bb (L_cap {d_w.L_cap / NM:.1f} nm)")
        ax.set_title("screening wavevector beta(z)", fontsize=9); ax.set_ylabel("beta, 1/nm")
        ax.set_ylim(0, 1.15 * s.beta * 1e-9)

        ax = axs[row, 3]
        eh = [m for m in mech if isinstance(m, LocalMechanism) and m.name.startswith("eh_") and m.valley_from == 0]
        jj = np.unique(np.round(np.linspace(0, d.n - 1, 28)).astype(int))
        zin = np.linspace(0, f.W, 2000)                                 # phi(z) is increasing on [0, W]
        zj = np.interp(d.phi[jj], f.band_edge(zin), zin) / NM
        for c, e in zip(dg.SLOTS, E_FIXED):
            w = [sum(m.variants[j].effective_rate(ev(e), rng, 3000) for m in eh) for j in jj]
            ax.semilogy(zj, np.maximum(w, 1e3), "o-", color=c, ms=3, lw=1.5, label=f"E = {e} eV")
        ax.set_ylim(1e6, 1e15); ax.set_title("e-h accepted rate (hh + lh), Gamma", fontsize=9)
        ax.set_ylabel("rate, 1/s")

        ax = axs[row, 4]
        ii = [m for m in mech if m.name == "impurity[Gamma]"][0]
        ii_w = [m for m in mech_w if m.name == "impurity[Gamma]"][0]
        for c, e in zip(dg.SLOTS, (0.01,) + E_FIXED[:2]):
            r = np.array([v.rate(ev(np.array([e])))[0] for v in ii.variants])
            r_w = np.array([v.rate(ev(np.array([e])))[0] for v in ii_w.variants])
            ax.semilogy(zn, np.interp(fi, idx, r), color=c, lw=2, label=f"E = {e} eV")
            ax.semilogy(zn, np.interp(fi, idx, r_w), color=c, lw=1.2, ls="--")
        ax.set_title("ionized-impurity rate, Gamma (dashed: W_bb cap)", fontsize=9)

        ax = axs[row, 5]
        gam = lambda j: [m.variants[j] if isinstance(m, LocalMechanism) else m for m in mech if m.valley_from == 0]
        for c, e in zip(dg.SLOTS, (0.01,) + E_FIXED[:2]):
            inv = np.array([SpinModel(d.samples[j], gam(j)).bap(ev(np.array([e])))[0] for j in idx])
            ax.semilogy(zn, 1 / np.maximum(np.interp(fi, idx, inv), 1e-30) / PS, color=c, lw=2, label=f"E = {e} eV")
        ax.set_ylim(1, 1e9); ax.set_title("BAP spin-relaxation time", fontsize=9); ax.set_ylabel("tau_BAP, ps")
        for a in axs[row]:
            a.set_xlabel("z, nm"); a.grid(True, color="#e4e3dd", lw=0.5); a.legend(fontsize=7, frameon=False)
            a.axvline(f.W / NM, color=dg.MUTED, lw=0.8)
        print(f"{tag}: bulk L_screen {1 / s.beta / NM:.2f} nm; caps: impurity spacing -> {d.L_cap / NM:.2f} nm, "
              f"W_bb -> {d_w.L_cap / NM:.2f} nm; p(0)/p_bulk = {d.p[0] / hg.p:.2e}; eta_bulk = {hg.eta:.3f}")
    dg.save(fig, OUT / "depletion_profiles.png")
    print("figure written to", OUT)


VARIANTS = (("bulk", "bulk rates (C21-like)"), ("local", "local, cap = impurity spacing"),
            ("local_w", "local, cap = W_bb"))


def load(p, hv, mode):
    f = OUT / f"stageD_p{p:.0e}_hv{hv:.2f}_{mode}.npz"
    return np.load(f) if f.exists() else None


def compare():
    hvs = (1.45, 1.60, 1.75, 1.90)
    lines = ["p      hv    variant                         arrived  Gamma   L      X     ESP_all ESP_G   "
             "<E_G> <E_L> meV  <cos>_G <cos>_L  t_med  t_90 ps"]
    for p in (1e19, 5e17):
        for hv in hvs:
            for mode, lab in VARIANTS:
                d = load(p, hv, mode)
                if d is None:
                    continue
                n = int(d["n"]); v = d["arr_valley"]
                k = d["arr_k"]; cos = -k[:, 2] / np.linalg.norm(k, axis=1)
                E = to_ev(d["arr_E"]) * 1e3
                g, l_ = v == 0, v == 1
                lines.append(f"{p:.0e} {hv:.2f}  {lab:31s} {v.size / n:6.1%} {g.mean():6.1%} {l_.mean():6.1%} "
                             f"{(v == 2).mean():5.1%}  {d['arr_spin'].mean():6.3f} {d['arr_spin'][g].mean():6.3f}  "
                             f"{E[g].mean():5.0f} {E[l_].mean() if l_.any() else np.nan:5.0f}      "
                             f"{cos[g].mean():.3f}   {cos[l_].mean() if l_.any() else np.nan:.3f}  "
                             f"{np.median(d['arr_t']) / PS:6.1f} {np.percentile(d['arr_t'], 90) / PS:6.1f}")
    print("\n".join(lines))
    (OUT / "depletion_compare.txt").write_text("\n".join(lines) + "\n")

    for p in (1e19, 5e17):
        fig, axs = plt.subplots(4, 4, figsize=(20, 15))
        for col, hv in enumerate(hvs):
            for (mode, lab), c in zip(VARIANTS, dg.SLOTS):
                d = load(p, hv, mode)
                if d is None:
                    continue
                n = int(d["n"]); v = d["arr_valley"]; k = d["arr_k"]
                E = to_ev(d["arr_E"]) * 1e3; cos = -k[:, 2] / np.linalg.norm(k, axis=1)
                for row, (val, bins, xl) in enumerate(((E[v == 0], np.linspace(0, 900, 91), "E (Gamma arrivals), meV"),
                                                        (E[v == 1], np.linspace(0, 900, 91), "E (L arrivals, above L min), meV"),
                                                        (cos[v == 0], np.linspace(0, 1, 41), "cos(theta) from -z, Gamma arrivals"),
                                                        (d["arr_t"] / PS, np.geomspace(0.01, 300, 50), "arrival time, ps"))):
                    ax = axs[row, col]
                    ax.hist(val, bins=bins, histtype="step", lw=1.8, color=c,
                            weights=np.full(val.size, 1 / n), label=lab)
                    ax.set_xlabel(xl, fontsize=8)
                    if row == 3:
                        ax.set_xscale("log")
            axs[0, col].set_title(f"p = {p:.0e}, hv = {hv:.2f} eV", fontsize=10)
            for r in range(4):
                axs[r, col].legend(fontsize=6.5, frameon=False); axs[r, col].grid(True, color="#e4e3dd", lw=0.5)
        dg.save(fig, OUT / f"depletion_compare_p{p:.0e}.png")
    print("figures written to", OUT)


def inject(n=20000):
    """Isolate the depletion-region effect with high statistics: thermal Gamma electrons (spin +1)
    start at the edge of the band-bending region, z = W_bb, and are followed for 10 ps. The
    arrivals then sample only the transit of the depletion region, and spin loss there is ESP < 1."""
    from gaas_mc import bands
    from gaas_mc.particle import Ensemble
    from gaas_mc.transport import Simulation
    G = MAT.gamma
    lines = ["p      variant                       arrived  Gamma   L      X     ESP     <E_G> meV  "
             "<cos>_G  <t> fs   e-h events/arrival  ii events/arrival"]
    for p in (1e19, 5e17):
        s = Sample(MAT, per_cm3(p))
        f = C21BandBending(s)
        for mode, lab in VARIANTS:
            kw = {"bulk": dict(depletion_scattering="bulk"), "local": {},
                  "local_w": dict(depletion_screening_cap="band_bending_width")}[mode]
            a = ModelAssumptions(**kw)
            mech = build_mechanisms(s, a, "C", field=f)
            sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
            rng = np.random.default_rng(7)
            E = rng.gamma(1.5, s.kT, n)
            k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, G.m_eff, G.alpha)[:, None]
            ens = Ensemble.create(z=np.full(n, f.W), k=k, E=E, spin=1)
            r = Simulation(s, mech, sm, field=f, t_max=10 * PS, assumptions=a).run(ens, rng)
            A = r.arrivals
            names = r.mechanism_names
            eh = sum(A.n_events[:, i] for i, nm in enumerate(names) if nm.startswith("eh_")).mean()
            ii = sum(A.n_events[:, i] for i, nm in enumerate(names) if nm.startswith("impurity")).mean()
            v = A.valley
            cos = -A.k[:, 2] / np.linalg.norm(A.k, axis=1)
            fast = A.t < 1 * PS                 # direct transits (exclude bulk random walks)
            lines.append(f"{p:.0e}  {lab:29s} {len(A) / n:6.1%} {np.mean(v == 0):6.1%} {np.mean(v == 1):6.1%} "
                         f"{np.mean(v == 2):5.1%}  {A.esp():.4f}  {to_ev(A.E[v == 0]).mean() * 1e3:7.1f}  "
                         f"   {cos[v == 0].mean():.4f}  {A.t[fast].mean() / 1e-15:6.0f}   "
                         f"{eh:8.2f}            {ii:8.2f}")
            print(lines[-1], flush=True)
    (OUT / "depletion_inject.txt").write_text(chr(10).join(lines) + chr(10))


if __name__ == "__main__":
    {"profiles": profiles, "compare": compare, "inject": inject}[sys.argv[1]]()
