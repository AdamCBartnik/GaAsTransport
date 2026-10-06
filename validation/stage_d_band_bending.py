"""Stage D validation: C21 surface band bending (Eqs. 56-62).

  fast   python validation/stage_d_band_bending.py fast
         - Fig. 16: E_bb(p) and W_bb(p); C21 quotes E_bb = 0.694 eV, W_bb = 9.947 nm at 1e19 cm^-3
         - Fig. 17: E_C(z) and E_z(z) at 1e19 cm^-3
         - energy conservation of the velocity-Verlet integrator across the band bending
  run    python validation/stage_d_band_bending.py run <p_cm3> <hv_eV> <N>
         full Stage C bulk model + band bending, absorbing surface, 300 ps; snapshots at 0/100/300 ps
         (C21 Fig. 20: depth distribution of electrons still inside) and the surface-arrival
         ensemble -> validation/out/stageD_<p>_<hv>.npz
  plot   python validation/stage_d_band_bending.py plot
         Fig. 20-style histograms + arrival energy / time / ESP summary from the saved runs

C21 removes electrons that reach the surface and are trapped or emitted; only those reflected by the
surface barrier return. Our absorbing surface removes all of them, so the z-distributions are
comparable to C21 Fig. 20 up to that small difference.
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaas_mc import bands
from gaas_mc import diagnostics as dg
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import FS, NM, PS, ev, per_cm3, to_ev
from gaas_mc.excitation import photoexcite
from gaas_mc.fields import C21BandBending
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import build_mechanisms, stage_a_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.transport import EV_SURFACE, Simulation

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MAT = gaas_chubenko2021()


def fast():
    ps = np.geomspace(1e15, 1e20, 200)
    Ebb = [to_ev(Sample(MAT, per_cm3(p)).E_bb) for p in ps]
    Wbb = [Sample(MAT, per_cm3(p)).W_bb / NM for p in ps]
    s19 = Sample(MAT, per_cm3(1e19))
    print(f"1e19: E_bb = {to_ev(s19.E_bb):.4f} eV (C21 0.694), W_bb = {s19.W_bb / NM:.3f} nm (C21 9.947)")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axs[0]
    ax.semilogx(ps, Ebb, color=dg.SLOTS[7], lw=2, label="E_bb (left axis)")
    ax.set_xlabel("p, cm$^{-3}$"); ax.set_ylabel("E_bb, eV"); ax.set_ylim(0.45, 0.78)
    ax2 = ax.twinx()   # C21 Fig. 16 itself uses two axes; kept for direct comparison
    ax2.semilogx(ps, Wbb, "--", color=dg.SLOTS[0], lw=2, label="W_bb (right axis)")
    ax2.set_ylabel("W_bb, nm"); ax2.set_ylim(0, 850)
    ax.set_title("Band-bending parameters (cf. C21 Fig. 16)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False); ax2.legend(loc="upper right", fontsize=8, frameon=False)
    f = C21BandBending(s19)
    z = np.linspace(0, 12, 500) * NM
    axs[1].plot(z / NM, to_ev(f.band_edge(z)), color=dg.SLOTS[0], lw=2, label="E_C(z) - E_C(bulk)")
    axs[1].plot(z / NM, to_ev(f.band_edge(z) - s19.Eg), color=dg.SLOTS[1], lw=2, label="E_V(z) - E_C(bulk)")
    axs[1].invert_xaxis(); axs[1].set_xlabel("z, nm"); axs[1].set_ylabel("E, eV")
    axs[1].set_title("Band edges, p = 1e19 (cf. C21 Fig. 17a)", fontsize=10); axs[1].legend(fontsize=8, frameon=False)
    axs[2].plot(z / NM, f.Ez(z) / 1e8, color=dg.SLOTS[2], lw=2)
    axs[2].invert_xaxis(); axs[2].set_xlabel("z, nm"); axs[2].set_ylabel("E_z, 1e8 V/m")
    axs[2].set_title("Field, p = 1e19 (cf. C21 Fig. 17b)", fontsize=10)
    for a in axs:
        a.grid(True, color="#e4e3dd", lw=0.6)
    dg.save(fig, OUT / "stageD_fig16_17.png")

    print("energy conservation, electrons injected at z = W_bb with 50 meV, field only (no scattering):")
    g = MAT.gamma
    for p in (1e19, 1.5e18, 1.5e17):
        s = Sample(MAT, per_cm3(p))
        f = C21BandBending(s)
        for dt in (1.0, 0.5, 0.25, 0.1):
            sim = Simulation(s, stage_a_mechanisms(s), field=f, dt_max_field=dt * FS)
            rng = np.random.default_rng(0)
            n = 500
            E0 = np.full(n, ev(0.05))
            u = bands.random_unit_vectors(n, rng); u[:, 2] = -np.abs(u[:, 2])
            k = u * bands.k_of_E(E0, g.m_eff, g.alpha)[:, None]
            zi = np.full(n, f.W)
            z1, k1, E1, du, evt = sim.propagate(zi, k, E0, np.zeros(n, np.int8), np.full(n, 20 * PS))
            hit = evt == EV_SURFACE
            err = to_ev(E1[hit] + f.band_edge(z1[hit]) - E0[hit] - f.band_edge(zi[hit])) * 1e3
            print(f"  p={p:.1e} dt={dt:4.2f} fs: max |dE| = {np.abs(err).max():.3f} meV (mean {err.mean():+.3f})")


def run(p, hv, n):
    a = ModelAssumptions()
    s = Sample(MAT, per_cm3(p))
    mech = build_mechanisms(s, a, "C")
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    rng = np.random.default_rng(int(p / 1e15) + int(hv * 1000))
    ens = photoexcite(s, ev(hv), n, rng, assumptions=a)
    times = np.array([0, 100, 300]) * PS
    sim = Simulation(s, mech, sm, field=C21BandBending(s), t_max=300 * PS, snapshot_times=times, assumptions=a)
    t0 = time.time()
    r = sim.run(ens, rng)
    arr = r.arrivals
    tag = f"stageD_p{p:.0e}_hv{hv:.2f}"
    np.savez(OUT / f"{tag}.npz", times=times, z=r.snapshots.z, valley=r.snapshots.valley, z0=ens.z,
             arr_t=arr.t, arr_E=arr.E, arr_k=arr.k, arr_spin=arr.spin, arr_valley=arr.valley,
             arr_visited=arr.visited, arr_tv=arr.time_in_valley, arr_spin0=arr.spin0, arr_z0=arr.z0,
             esp0=ens.esp(), n=n, E_bb=s.E_bb, W_bb=s.W_bb, Eg=s.Eg)
    print(f"{tag}: N={n}, arrived {len(arr) / n:.1%}, ESP0 {ens.esp():.3f} -> ESP(arrival) {arr.esp():.3f}, "
          f"<E_arrival> {to_ev(arr.E).mean() * 1e3:.0f} meV (E_bb {to_ev(s.E_bb) * 1e3:.0f} meV), "
          f"median t {np.median(arr.t) / PS:.1f} ps, visited L/X {np.mean(arr.visited[:, 1:].any(1)):.3f}, "
          f"mode {r.flight_mode}, {time.time() - t0:.0f} s", flush=True)


def plot():
    files = sorted(OUT.glob("stageD_p*_hv*.npz"))
    if not files:
        print("no runs found"); return
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    hvs = (1.45, 1.60, 1.75, 1.90)
    for row, p in enumerate((1e19, 5e17)):
        for col, hv in enumerate(hvs):
            f = OUT / f"stageD_p{p:.0e}_hv{hv:.2f}.npz"
            ax = axs[row, col]
            if not f.exists():
                ax.set_visible(False); continue
            d = np.load(f)
            bins = np.arange(0, 6.01, 0.1)
            for i, (t, c) in enumerate(zip(d["times"], dg.SLOTS)):
                z = d["z"][i][np.isfinite(d["z"][i])] * 1e6
                ax.hist(z, bins=bins, histtype="step", lw=2, color=c, label=f"t = {t / PS:.0f} ps ({z.size})",
                        weights=np.full(z.size, 1 / (int(d["n"]) * 0.1)))
            ax.set_title(f"p = {p:.0e}, hv = {hv:.2f} eV (cf. C21 Fig. 20)", fontsize=9)
            ax.set_xlabel("z, um"); ax.set_ylabel("electrons / (N * um)")
            ax.legend(fontsize=7, frameon=False)
    dg.save(fig, OUT / "stageD_fig20.png")
    print("run                 arrived  ESP0   ESP_arr  <E_arr>  median_t  visited_L/X  arrive_in_L/X  "
          "<t_L/X | visited>  <t_L/X>/<t_total>")
    lines = []
    for f in files:
        d = np.load(f)
        n = int(d["n"])
        tv = d["arr_tv"] if "arr_tv" in d else None
        vis = d["arr_visited"][:, 1:].any(1)
        line = (f"{f.stem[7:]:18s} {d['arr_t'].size / n:7.1%} {float(d['esp0']):6.3f} {d['arr_spin'].mean():7.3f} "
                f"{to_ev(d['arr_E']).mean() * 1e3:6.0f} meV {np.median(d['arr_t']) / PS:6.1f} ps "
                f"{vis.mean():10.1%} {np.mean(d['arr_valley'] > 0):12.1%}")
        if tv is not None:
            tlx = tv[:, 1:].sum(1)
            line += (f" {tlx[vis].mean() / PS * 1e3 if vis.any() else 0:12.0f} fs "
                     f"{tlx.sum() / tv.sum():14.2%}")
            # spin of arrivals that visited L/X vs not
            if vis.any() and (~vis).any():
                line += f"   ESP(visited) {d['arr_spin'][vis].mean():.3f} / ESP(not) {d['arr_spin'][~vis].mean():.3f}"
        print(line)
        lines.append(line)
    (OUT / "stageD_summary.txt").write_text(chr(10).join(lines) + chr(10))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fast":
        fast()
    elif cmd == "run":
        run(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "plot":
        plot()
