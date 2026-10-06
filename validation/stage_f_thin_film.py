"""Stage F: minimal finite-slab model, 0 < z < d, probabilistic back boundary (gaas_mc/back.py).

QE, ESP, response time, surface-arrival fraction and substrate-loss fraction versus the layer
thickness d and the back reflection coefficient R_back. The whole layer is GaAs with the C21
baseline physics (p = 1e19, bulk depletion scattering, C21 band bending, C21 surface model with the
band-edge matching mass); no back-interface potential, no substrate transport, no optical
interference. Not meant to match a particular experiment.

    python validation/stage_f_thin_film.py [N=100000] [cuda|cpu]

QE = A_layer * N_emitted / N_generated, A_layer = (1 - R_opt)(1 - exp(-alpha d)) (semi-infinite:
1 - R_opt). Response time: emission-time median and 90th percentile, and the share emitted within
10 ps.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaas_mc.assumptions import ModelAssumptions                      # noqa: E402
from gaas_mc.back import PartialReflector                             # noqa: E402
from gaas_mc.constants import NM, PS, ev, per_cm3                     # noqa: E402
from gaas_mc.excitation import layer_absorption, photoexcite          # noqa: E402
from gaas_mc.fast import FastSimulation                               # noqa: E402
from gaas_mc.fields import C21BandBending                             # noqa: E402
from gaas_mc.material import Sample, gaas_chubenko2021                # noqa: E402
from gaas_mc.particle import BACK, EMITTED, TIMEOUT, TRAPPED          # noqa: E402
from gaas_mc.scattering import build_mechanisms                       # noqa: E402
from gaas_mc.spin import SpinModel                                    # noqa: E402
from gaas_mc.surface_c21 import C21Surface                            # noqa: E402
from gaas_mc.transport import Simulation                              # noqa: E402

OUT = ROOT / "validation" / "out" / "stageF"
MAT = gaas_chubenko2021()
A = ModelAssumptions(depletion_scattering="bulk", absorption_model="adachi1989")
CHI = 0.67
HVS = (1.55, 1.80)
THICKNESSES = (100 * NM, 200 * NM, 500 * NM, None)          # None: semi-infinite ("effectively bulk")
R_BACKS = (0.0, 0.5, 1.0)
T_MAX = 370 * PS


def run(n=100_000, device="cuda"):
    s = Sample(MAT, per_cm3(1e19))
    field = C21BandBending(s)
    mech = build_mechanisms(s, A, "C", field=field)
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    rows = []
    for hv in HVS:
        for d in THICKNESSES:
            for R in (R_BACKS if d is not None else (None,)):
                t0 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sim = Simulation(s, mech, sm, field=field, surface=C21Surface(chi=ev(CHI), material=MAT),
                                     z_back=d, back=PartialReflector(R) if d is not None else "absorb",
                                     t_max=T_MAX, assumptions=A)
                rng = np.random.default_rng([int(hv * 1000), 0 if d is None else int(d / NM), 0 if R is None else int(R * 10)])
                ens = photoexcite(s, ev(hv), n, rng, assumptions=A, thickness=d)
                r = FastSimulation(sim, device).run(ens, rng)
                e, em = r.ensemble, r.emissions
                st = e.status
                ne = len(em)
                A_layer = layer_absorption(s, ev(hv), d, assumptions=A)
                t = em.t / PS
                esp = em.spin.mean() if ne else np.nan
                rows.append(dict(
                    hv=hv, d=d, R=R, n=n, A_layer=A_layer, QE=A_layer * ne / n, dQE=A_layer * np.sqrt(ne) / n,
                    ESP=esp, dESP=np.sqrt(max(1 - esp**2, 0) / ne) if ne else np.nan,
                    t50=np.median(t) if ne else np.nan, t90=np.percentile(t, 90) if ne else np.nan,
                    early=np.mean(t <= 10) if ne else np.nan, arrived=np.mean(e.n_surface > 0),
                    lost=np.mean(st == BACK), reached_back=np.mean(e.n_back > 0),
                    trapped=np.mean(st == TRAPPED), timeout=np.mean(st == TIMEOUT),
                    emitted_after_back=np.mean(em.n_back > 0) if ne else np.nan,
                    back_enc=e.n_back.mean(), seconds=time.time() - t0))
                rr = rows[-1]
                print(f"hv={hv:.2f} d={'bulk' if d is None else f'{d / NM:.0f} nm':>7s} R_back={'-' if R is None else R}: "
                      f"QE {100 * rr['QE']:.2f}%  ESP {100 * rr['ESP']:.1f}%  t50 {rr['t50']:.1f} ps  "
                      f"lost {rr['lost']:.3f}  ({rr['seconds']:.0f} s)", flush=True)
    return rows


def report(rows):
    lines = ["Stage F: finite GaAs layer, p = 1e19, chi = 0.67 eV, C21 baseline physics, 370 ps",
             "QE = A_layer N_emit / N; response: median and 90th percentile of the emission time, share "
             "emitted within 10 ps; arrived = ever reached the surface; lost = absorbed by the substrate",
             ""]
    hdr = ("  hv   d(nm)  R_back | A_layer  QE %(+-)     ESP %(+-)  | t50 ps  t90 ps  <=10ps | arrived  "
           "reached_back  lost   trapped  timeout | emitted after a back reflection  <back encounters>")
    for hv in HVS:
        lines.append(hdr)
        for r in (x for x in rows if x["hv"] == hv):
            dn = "bulk" if r["d"] is None else f"{r['d'] / NM:.0f}"
            Rn = "  -" if r["R"] is None else f"{r['R']:.1f}"
            lines.append(
                f"{hv:.2f} {dn:>6s}  {Rn:>5s}  | {r['A_layer']:.4f}  {100 * r['QE']:5.2f}({100 * r['dQE']:.2f})  "
                f"{100 * r['ESP']:5.1f}({100 * r['dESP']:.1f}) | {r['t50']:6.1f}  {r['t90']:6.1f}  {r['early']:.3f}  | "
                f"{r['arrived']:.3f}    {r['reached_back']:.3f}       {r['lost']:.3f}  {r['trapped']:.3f}   "
                f"{r['timeout']:.3f}  | {r['emitted_after_back']:.3f}                            {r['back_enc']:.2f}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thin_film_summary.txt").write_text(txt + "\n")

    import matplotlib.pyplot as plt
    from gaas_mc import diagnostics as dg
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for row, hv in enumerate(HVS):
        for col, (key, lab, scale) in enumerate((("QE", "QE, %", 100), ("ESP", "ESP, %", 100),
                                                 ("t50", "median emission time, ps", 1))):
            ax = axs[row, col]
            bulk = [x for x in rows if x["hv"] == hv and x["d"] is None][0]
            for j, R in enumerate(R_BACKS):
                rr = [x for x in rows if x["hv"] == hv and x["R"] == R]
                ax.plot([x["d"] / NM for x in rr], [scale * x[key] for x in rr], "-o", color=dg.SLOTS[j], lw=2,
                        ms=4, label=f"R_back = {R:.1f}")
            ax.axhline(scale * bulk[key], color="#555555", ls="--", lw=1.2, label="semi-infinite")
            ax.set_xscale("log")
            ax.set_xlabel("layer thickness d, nm"); ax.set_ylabel(lab)
            ax.set_title(f"hv = {hv:.2f} eV, chi = {CHI:.2f} eV", fontsize=10)
            ax.grid(True, color="#e4e3dd", lw=0.6)
            ax.legend(fontsize=8, frameon=False)
    dg.save(fig, OUT / "thin_film.png")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    dev = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    report(run(n, dev))
