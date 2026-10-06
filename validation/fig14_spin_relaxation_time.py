"""Spin relaxation time from the decay of the internal ESP (cf. [C21] Eq. 55 and Fig. 14).

As in C21: photoexcitation at hw = 1.65 eV, emission prohibited (here: specular reflection at
z = 0), full Stage C model, ESP_int(t) of electrons inside the material. tau_s follows the C21
definition (text after Eq. 54): the time for ESP to fall from ESP0 to ESP0/e. A weighted
late-time exponential fit is reported as well. C21 reports 110 ps (1.5e17), 92 ps (1.5e18), 77 ps (1e19) with their model
(step-function Pauli rule, Eq. 44), about 1.4-1.8x longer than experiment.

Run: PYTHONPATH=. python validation/fig14_spin_relaxation_time.py [n_particles]
"""
import sys
import time
from pathlib import Path

import numpy as np

from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import PS, ev, per_cm3
from gaas_mc.excitation import photoexcite
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.transport import Simulation

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()
C21 = {1.5e17: 110, 1.5e18: 92, 1e19: 77}
T_MAX = 300 * PS


def run(p, pauli, n, rng):
    a = ModelAssumptions(pauli_blocking=pauli)
    s = Sample(MAT, per_cm3(p))
    mech = build_mechanisms(s, a, "C")
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    times = np.arange(0, 301, 10) * PS
    ens = photoexcite(s, ev(1.65), n, rng, assumptions=a)
    sim = Simulation(s, mech, sm, surface="reflect", t_max=T_MAX, snapshot_times=times, assumptions=a)
    r = sim.run(ens, rng)
    esp = np.array([r.snapshots.esp(i) for i in range(times.size)])
    fit = times >= 30 * PS
    slope, icpt = np.polyfit(times[fit], np.log(esp[fit]), 1)
    return -1 / slope, esp, times, r


def main(n, p, pauli):
    rng = np.random.default_rng(int(p / 1e15) + (0 if pauli == "fermi_dirac" else 7))
    t0 = time.time()
    tau, esp, times, r = run(p, pauli, n, rng)
    line = (f"pauli={pauli:12s} p={p:.1e}: tau_s = {tau / PS:6.1f} ps (C21: {C21[p]} ps); "
            f"ESP(0)={esp[0]:.3f} ESP(100ps)={esp[10]:.3f} ESP(300ps)={esp[-1]:.3f}  "
            f"[N={n}, {time.time() - t0:.0f}s, iters {r.n_iterations}]")
    print(line, flush=True)
    np.savez(OUT / f"fig14_{pauli}_p{p:.1e}.npz", times=times, esp=esp, tau=tau, n=n)
    with open(OUT / "fig14_spin_relaxation_time.txt", "a") as fh:
        fh.write(line + chr(10))


def t1e(t, esp):
    """First time at which ESP(t) <= ESP(0)/e, by log-linear interpolation between snapshots."""
    target = esp[0] / np.e
    i = int(np.argmax(esp <= target))
    if esp[i] > target:
        return np.nan
    l0, l1 = np.log(esp[i - 1]), np.log(esp[i])
    return t[i - 1] + (np.log(target) - l0) / (l1 - l0) * (t[i] - t[i - 1])


def summarize():
    """Weighted fit ESP(t) = A exp(-t/tau) with binomial errors sigma = sqrt((1 - ESP^2)/N).
    (An unweighted fit of ln ESP over-weights the noisy late-time points.)"""
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    from gaas_mc import diagnostics as dg
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    lines = []
    for j, p in enumerate((1.5e17, 1.5e18, 1e19)):
        for pauli, ls in (("fermi_dirac", "-"), ("step_c21", "--")):
            f = OUT / f"fig14_{pauli}_p{p:.1e}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            t, esp, n = d["times"], d["esp"], int(d["n"])
            sig = np.sqrt(np.clip(1 - esp**2, 1e-12, None) / n)
            out = []
            for t_lo, t_hi in ((30, 300), (30, 150)):
                m = (t >= t_lo * PS) & (t <= t_hi * PS)
                (A, tau), cov = curve_fit(lambda x, A, tau: A * np.exp(-x / tau), t[m], esp[m],
                                          p0=(esp[0], 100 * PS), sigma=sig[m], absolute_sigma=True)
                out.append((tau / PS, np.sqrt(cov[1, 1]) / PS))
            # C21 definition (text after Eq. 54): time for ESP to fall from ESP0 to ESP0/e
            t_e = t1e(t, esp)
            line = (f"p={p:.1e} pauli={pauli:12s} 1/e time (C21 definition) = {t_e / PS:6.1f} ps;  "
                    f"late-time fit {out[0][0]:6.1f} +- {out[0][1]:4.1f} ps (30-300 ps), "
                    f"{out[1][0]:6.1f} +- {out[1][1]:4.1f} ps (30-150 ps);  C21: {C21[p]} ps")
            print(line)
            lines.append(line)
            ax.errorbar(t / PS, esp, yerr=sig, fmt="o", ms=3, color=dg.SLOTS[j], ls=ls, lw=1,
                        label=f"p={p:.1e}, {pauli}, tau={out[0][0]:.0f} ps")
    ax.set_yscale("log"); ax.set_xlabel("t, ps"); ax.set_ylabel("internal ESP")
    ax.set_title("Internal ESP decay, hv = 1.65 eV, no emission (cf. C21 Fig. 14 / Eq. 55)", fontsize=9)
    ax.grid(True, color="#e4e3dd", lw=0.6); ax.legend(fontsize=6.5, frameon=False)
    dg.save(fig, OUT / "fig14_internal_esp.png")
    (OUT / "fig14_summary.txt").write_text(chr(10).join(lines) + chr(10))


if __name__ == "__main__":
    # usage: fig14_spin_relaxation_time.py <n> <p_cm3> <pauli: fermi_dirac|step_c21>
    #        fig14_spin_relaxation_time.py summarize
    if sys.argv[1:2] == ["summarize"]:
        summarize()
    else:
        main(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
