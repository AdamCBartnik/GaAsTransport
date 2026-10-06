"""Stage E benchmark: Chubenko et al. (2021) Fig. 18 -- QE and ESP vs photon energy at p = 1e19 cm^-3
for chi = 0.64, 0.67, 0.70, 0.73 eV, end to end (photoexcitation -> transport -> C21 surface model).

  run      python validation/stage_e_fig18.py run <bulk|local> [N_per_hv=16000] [workers=14]
           transport to the first surface arrival (absorbing surface), saved per chunk, then the
           surface step for every surface variant
  surface  python validation/stage_e_fig18.py surface <bulk|local> <vmass|bmass|all> [workers=14]
           only the surface step, from the saved first arrivals
  report   python validation/stage_e_fig18.py report <bulk|local>/<vmass|bmass> ...

Model (C21-compatible baseline, "bulk"):
  * Stage C bulk mechanisms + C21 band bending (Eqs. 56-62), depletion_scattering = "bulk" (the
    depletion region uses bulk rates, as C21 state for Fig. 18) or "local" (the improved model);
  * absorption: Adachi's model (C21 Fig. 3 uses Adachi's dielectric function). The default
    Casey(1975)+Adachi absorption differs only below ~1.6 eV; its effect is reported by exact
    importance reweighting of the photoexcitation depth (the bulk history does not depend on alpha);
  * C21 surface model (gaas_mc/surface_c21.py), 370 ps (C21 Sec. IV), infinitely thick sample;
  * QE = (1 - R) N_emitted / N_absorbed (C21 Eqs. 5, 6); statistical errors as in C21 (Poisson on
    N_emitted, binomial for ESP).

Surface variants (C21Surface.matching_mass): "vmass" = velocity mass m*(1+2 alpha E) on the GaAs side
of the transfer matrix (default, flux-consistent); "bmass" = band-edge m*, the literal reading of
C21's "the electron mass changes from m*_e to m0". C21 do not say how nonparabolicity enters.

Every chi branch and surface variant continues the SAME first-arrival ensemble (the bulk history up
to the first surface encounter does not depend on the surface model): Ensemble.from_arrivals(...)
with Simulation.run(..., start_at_surface=True). The "bulk" and "local" variants use the same seeds,
so they start from identical photoexcited ensembles.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gaas_mc.assumptions import ModelAssumptions                     # noqa: E402
from gaas_mc.constants import PS, ev, per_cm3, to_ev                 # noqa: E402
from gaas_mc.excitation import photoexcite                           # noqa: E402
from gaas_mc.fields import C21BandBending                            # noqa: E402
from gaas_mc.material import Sample, gaas_chubenko2021               # noqa: E402
from gaas_mc.optics import Adachi1989GaAs, CaseyAdachiAbsorption     # noqa: E402
from gaas_mc.particle import EMITTED, TIMEOUT, TRAPPED, Ensemble     # noqa: E402
from gaas_mc.scattering import build_mechanisms                      # noqa: E402
from gaas_mc.spin import SpinModel                                   # noqa: E402
from gaas_mc.surface import SurfaceArrivals                          # noqa: E402
from gaas_mc.surface_c21 import C21Surface                           # noqa: E402
from gaas_mc.transport import Simulation                             # noqa: E402

OUT = ROOT / "validation" / "out" / "stageE"
REF = ROOT / "validation" / "reference" / "c21_fig18.csv"
MAT = gaas_chubenko2021()
P = 1e19
HVS = np.round(np.arange(1.45, 2.2001, 0.05), 2)
CHIS = (0.64, 0.67, 0.70, 0.73)
T_MAX = 370 * PS
T_CUTS = (10 * PS, 30 * PS, 100 * PS, 370 * PS)     # QE/ESP with emission-time cutoffs (post-processing)
CHUNK = 2000
SURFACE_VARIANTS = {"vmass": dict(matching_mass="velocity"), "bmass": dict(matching_mass="band_edge")}
PID_STRIDE = 10_000_000                     # branch b of a batched surface run has pid + b * PID_STRIDE


class BranchedSurface:
    """Runs several surface models in ONE transport run: branch b = pid // PID_STRIDE uses models[b].
    The branches are independent copies of the same first arrivals; batching them only reduces the
    per-iteration overhead of the long tail of reflected electrons."""
    is_surface_model = True

    def __init__(self, models):
        self.models = models

    def interact(self, k, E, valley, K, rng, pid=None):
        b = np.asarray(pid) // PID_STRIDE
        n = E.size
        outcome = np.zeros(n, np.int8)
        info = dict(p_vac=np.zeros((n, 3)), E_vac_kin=np.zeros(n))
        for j, m in enumerate(self.models):
            i = np.flatnonzero(b == j)
            if i.size:
                o, inf = m.interact(k[i], E[i], valley[i], K[i], rng)
                outcome[i] = o
                info["p_vac"][i] = inf["p_vac"]
                info["E_vac_kin"][i] = inf["E_vac_kin"]
        return outcome, info


def _setup(variant):
    a = ModelAssumptions(depletion_scattering=variant, absorption_model="adachi1989")
    s = Sample(MAT, per_cm3(P))
    field = C21BandBending(s)
    mech = build_mechanisms(s, a, "C", field=field)
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    return a, s, Simulation(s, mech, sm, field=field, t_max=T_MAX, assumptions=a)


def _stem(variant, hv, chunk):
    return OUT / variant / "arrivals" / f"hv{hv:.2f}_c{chunk:02d}"


def transport_job(args):
    """Photoexcitation + transport to the first surface arrival (absorbing surface)."""
    variant, hv, chunk, n = args
    stem = _stem(variant, hv, chunk)
    if Path(f"{stem}_arr.npz").exists():
        return stem.name, 0.0
    t0 = time.time()
    a, s, sim = _setup(variant)
    rng = np.random.default_rng([int(round(hv * 1000)), chunk])
    ens = photoexcite(s, ev(hv), n, rng, assumptions=a)
    arr = sim.run(ens, rng).arrivals
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{stem}_gen.npz", n=n, hv=hv, z0_all=ens.z0, spin0_all=ens.spin0)
    arr.save_npz(f"{stem}_arr.npz")
    return stem.name, time.time() - t0


def surface_job(args):
    """Continue the saved first arrivals with the C21 surface model for every chi, and for the
    surface variant svar ("all": every variant), in one batched run (BranchedSurface)."""
    variant, svar, hv, chunk = args
    svars = list(SURFACE_VARIANTS) if svar == "all" else [svar]
    stem = _stem(variant, hv, chunk)
    outs = {sv: OUT / variant / f"surf_{sv}" / f"hv{hv:.2f}_c{chunk:02d}.npz" for sv in svars}
    svars = [sv for sv in svars if not outs[sv].exists()]
    if not svars:
        return f"hv{hv:.2f}_c{chunk:02d}", 0.0
    t0 = time.time()
    a, s, sim = _setup(variant)
    gen = dict(np.load(f"{stem}_gen.npz"))
    arr = SurfaceArrivals.load_npz(f"{stem}_arr.npz")
    assert arr.pid.size == 0 or arr.pid.max() < PID_STRIDE
    branches = [(sv, chi) for sv in svars for chi in CHIS]
    parts = []
    for b in range(len(branches)):
        e = Ensemble.from_arrivals(arr)
        e.pid = e.pid + b * PID_STRIDE
        parts.append(e)
    sim.surface_model = BranchedSurface([C21Surface(chi=ev(chi), material=MAT, **SURFACE_VARIANTS[sv])
                                         for sv, chi in branches])
    r = sim.run(Ensemble.concatenate(parts), np.random.default_rng([int(round(hv * 1000)), chunk, 7]),
                start_at_surface=True)
    em, fin = r.emissions, r.ensemble
    eb, fb = em.pid // PID_STRIDE, fin.pid // PID_STRIDE
    for sv in svars:
        rec = dict(n=int(gen["n"]), hv=hv, z0_all=gen["z0_all"], spin0_all=gen["spin0_all"],
                   arr_t=arr.t, arr_E=arr.E, arr_valley=arr.valley, arr_eqv=arr.eqv, arr_spin=arr.spin,
                   arr_pid=arr.pid, arr_z0=arr.z0)
        for chi in CHIS:
            b = branches.index((sv, chi))
            i, j = eb == b, fb == b
            tag = f"chi{chi:.2f}_"
            rec.update({tag + "t": em.t[i], tag + "E": em.E[i], tag + "valley": em.valley[i],
                        tag + "eqv": em.eqv[i], tag + "spin": em.spin[i], tag + "spin0": em.spin0[i],
                        tag + "z0": em.z0[i], tag + "band": em.band[i], tag + "n_surface": em.n_surface[i],
                        tag + "E_vac": em.E_vac[i], tag + "E_perp": em.E_perp[i], tag + "pz": em.p_vac[i, 2],
                        tag + "status": fin.status[j], tag + "fin_valley": fin.valley[j],
                        tag + "fin_n_surface": fin.n_surface[j], tag + "fin_t": fin.t[j]})
        outs[sv].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outs[sv], **rec)
    return f"hv{hv:.2f}_c{chunk:02d} ({'+'.join(svars)})", time.time() - t0


def _pool(fn, jobs, workers, label):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers) as pool:
        for i, (name, dt) in enumerate(pool.imap_unordered(fn, jobs)):
            print(f"{label} [{i + 1}/{len(jobs)}] {name} {dt:.0f} s (elapsed {time.time() - t0:.0f} s)", flush=True)


def surface(variant, svar, workers=14):
    files = sorted((OUT / variant / "arrivals").glob("hv*_c*_arr.npz"))
    jobs = [(variant, svar, float(f.name[2:6]), int(f.name.split("_c")[1][:2])) for f in files]
    _pool(surface_job, jobs, workers, f"surface {svar}")


def run(variant, n_per_hv=16000, workers=14):
    n_chunks = int(np.ceil(n_per_hv / CHUNK))
    jobs = [(variant, float(hv), c, CHUNK) for c in range(n_chunks) for hv in HVS]
    _pool(transport_job, jobs, workers, "transport")
    surface(variant, "all", workers)


# ------------------------------------------------------------------------------------------ report
def load(tag):
    variant, svar = tag.split("/")
    by_hv = {}
    for f in sorted((OUT / variant / f"surf_{svar}").glob("hv*_c*.npz")):
        d = dict(np.load(f))
        by_hv.setdefault(round(float(d["hv"]), 2), []).append(d)
    return by_hv


def cat(parts, key):
    return np.concatenate([np.atleast_1d(p[key]) for p in parts])


def reference():
    import csv
    ref = {}
    with open(REF) as fh:
        rows = [r for r in csv.reader(line for line in fh if not line.startswith("#"))][1:]
    for src, q, chi, hv, v in rows:
        ref.setdefault((src, q, float(chi) if chi else None), []).append((float(hv), float(v)))
    return {k: np.array(sorted(v)) for k, v in ref.items()}


def summarize(tag):
    """All quantities per (hv, chi)."""
    by_hv = load(tag)
    s = Sample(MAT, per_cm3(P))
    A, C = Adachi1989GaAs(), CaseyAdachiAbsorption(s.p)
    res = []
    for hv in sorted(by_hv):
        parts = by_hv[hv]
        N = int(sum(int(p["n"]) for p in parts))
        R = float(A.reflectivity(ev(hv)))
        aA, aC = float(A.absorption_coefficient(ev(hv))), float(C.absorption_coefficient(ev(hv)))
        RC = float(C.reflectivity(ev(hv)))
        # exact reweighting of the photoexcitation depth Adachi -> Casey+Adachi
        z_all = cat(parts, "z0_all")
        w_all = (aC / aA) * np.exp(-(aC - aA) * z_all)
        arr_v = cat(parts, "arr_valley")
        row0 = dict(hv=hv, N=N, R=R, esp0=float(np.mean(cat(parts, "spin0_all"))),
                    n_arr=arr_v.size, arr_frac=arr_v.size / N,
                    arr_valley=np.bincount(arr_v, minlength=3), esp_arr=float(np.mean(cat(parts, "arr_spin"))),
                    ess_casey=w_all.sum() ** 2 / np.sum(w_all ** 2) / N)
        for chi in CHIS:
            tg = f"chi{chi:.2f}_"
            spin = cat(parts, tg + "spin").astype(float)
            ne = spin.size
            st = cat(parts, tg + "status")
            nfin = cat(parts, tg + "fin_n_surface")
            fv = cat(parts, tg + "fin_valley")
            esp = spin.mean() if ne else np.nan
            we = (aC / aA) * np.exp(-(aC - aA) * cat(parts, tg + "z0"))
            E_perp = to_ev(cat(parts, tg + "E_perp"))
            t_em = cat(parts, tg + "t")
            cut = {tc: ((1 - R) * np.sum(t_em <= tc) / N,
                        float(spin[t_em <= tc].mean()) if np.any(t_em <= tc) else np.nan) for tc in T_CUTS}
            res.append(dict(row0, chi=chi, n_emit=ne, QE=(1 - R) * ne / N, dQE=(1 - R) * np.sqrt(ne) / N, ESP=esp,
                            dESP=np.sqrt(max(1 - esp ** 2, 0) / ne) if ne else np.nan,
                            QE_casey=(1 - RC) * we.sum() / w_all.sum(),
                            ESP_casey=float(np.sum(we * spin) / we.sum()) if ne else np.nan,
                            emit_valley=np.bincount(cat(parts, tg + "valley"), minlength=3),
                            n_trap=int(np.sum(st == TRAPPED)), n_timeout=int(np.sum(st == TIMEOUT)),
                            trap_valley=np.bincount(fv[st == TRAPPED], minlength=3),
                            enc_emit=float(nfin[st == EMITTED].mean()) if ne else np.nan,
                            enc_trap=float(nfin[st == TRAPPED].mean()) if np.any(st == TRAPPED) else np.nan,
                            enc_all=float(nfin[(st == EMITTED) | (st == TRAPPED)].mean()),
                            E_vac=to_ev(cat(parts, tg + "E_vac")), E_perp=E_perp,
                            MTE=float(E_perp.mean()) if ne else np.nan,
                            t_emit=t_em, cut=cut))
    return res


def report(tags=("bulk/vmass",)):
    import matplotlib.pyplot as plt
    from gaas_mc import diagnostics as dg
    ref = reference()
    tabs = {v: summarize(v) for v in tags}
    lines = []
    for v, res in tabs.items():
        lines.append(f"===== {v} (depletion scattering / surface matching mass)  p = 1e19 cm^-3, 370 ps, "
                     f"Adachi absorption; C21 = Fig. 18 curves")
        lines.append("  hv  chi    N   QE%(+-)      C21   QE%cas  ESP%(+-)      C21  ESP%cas | arrive  G/L/X %   "
                     "| emitted G/L/X %   | trapped | enc emit/trap | MTE meV <E_vac> meV")
        for r in res:
            qe_c = np.interp(r["hv"], *ref[("C21_simulation", "QE", r["chi"])].T)
            esp_c = np.interp(r["hv"], *ref[("C21_simulation", "ESP", r["chi"])].T)
            av = 100 * r["arr_valley"] / max(r["n_arr"], 1)
            ev_ = 100 * r["emit_valley"] / max(r["n_emit"], 1)
            lines.append(
                f"{r['hv']:.2f} {r['chi']:.2f} {r['N']:6d} {100 * r['QE']:5.2f}({100 * r['dQE']:.2f}) {qe_c:6.2f}"
                f"  {100 * r['QE_casey']:6.2f}  {100 * r['ESP']:5.1f}({100 * r['dESP']:.1f}) {esp_c:6.1f}"
                f"  {100 * r['ESP_casey']:6.1f} | {r['n_arr']:6d} {av[0]:4.1f}/{av[1]:4.1f}/{av[2]:4.1f} "
                f"| {r['n_emit']:5d} {ev_[0]:5.1f}/{ev_[1]:3.1f}/{ev_[2]:4.1f} | {r['n_trap']:6d} "
                f"| {r['enc_emit']:5.2f}/{r['enc_trap']:5.2f} | {1e3 * r['MTE']:6.1f} {1e3 * np.mean(r['E_vac']):7.1f}")
        lines.append("")
        lines.append("  emission-time cutoff (same run): QE% / ESP% for t_emit <= " +
                     ", ".join(f"{tc / PS:.0f}" for tc in T_CUTS) + " ps")
        for r in res:
            lines.append(f"{r['hv']:.2f} {r['chi']:.2f}  " + "   ".join(
                f"{100 * r['cut'][tc][0]:5.2f} / {100 * r['cut'][tc][1]:5.1f}" for tc in T_CUTS))
        lines.append("")
        for q, key, dkey in (("QE", "QE", "dQE"), ("ESP", "ESP", "dESP")):
            for chi in CHIS:
                rr = [r for r in res if r["chi"] == chi]
                ours = np.array([100 * r[key] for r in rr])
                err = np.array([100 * r[dkey] for r in rr])
                theirs = np.interp([r["hv"] for r in rr], *ref[("C21_simulation", q, chi)].T)
                d = ours - theirs
                lines.append(f"  {q:3s} chi={chi:.2f}: ours - C21 mean {d.mean():+.2f} %, rms {np.sqrt(np.mean(d**2)):.2f} %, "
                             f"mean |d|/sigma {np.mean(np.abs(d) / err):.2f}, ratio ours/C21 mean {np.mean(ours / theirs):.3f}")
        lines.append("")
    pairs = [(b, b.replace("bulk/", "local/")) for b in tabs if b.startswith("bulk/")]
    pairs += [(b, b.replace("/vmass", "/bmass")) for b in tabs if b.endswith("/vmass")]
    for b0, b1 in pairs:
        if b1 not in tabs:
            continue
        lines.append(f"===== {b1} minus {b0} (same photoexcited ensembles)")
        for rb, rl in zip(tabs[b0], tabs[b1]):
            assert rb["hv"] == rl["hv"] and rb["chi"] == rl["chi"]
            lines.append(f"{rb['hv']:.2f} {rb['chi']:.2f}  dQE {100 * (rl['QE'] - rb['QE']):+.2f} %  "
                         f"dESP {100 * (rl['ESP'] - rb['ESP']):+.1f} %  dMTE {1e3 * (rl['MTE'] - rb['MTE']):+.1f} meV  "
                         f"d<E_vac> {1e3 * (np.mean(rl['E_vac']) - np.mean(rb['E_vac'])):+.1f} meV  "
                         f"arrive {rb['n_arr']}->{rl['n_arr']}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fig18_summary_{'_'.join(v.replace('/', '-') for v in tags)}.txt").write_text(txt + "\n")

    # ---------------------------------------------------------------- figures
    cols = dict(zip(CHIS, dg.SLOTS[:4]))
    for v, res in tabs.items():
        vt = v.replace("/", "-")
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
        for ax, q, key, dkey in ((axs[0], "QE", "QE", "dQE"), (axs[1], "ESP", "ESP", "dESP")):
            for chi in CHIS:
                rr = [r for r in res if r["chi"] == chi]
                ax.errorbar([r["hv"] for r in rr], [100 * r[key] for r in rr], yerr=[100 * r[dkey] for r in rr],
                            color=cols[chi], lw=2, marker="o", ms=4, capsize=2, label=f"this MC, chi = {chi:.2f} eV")
                c = ref[("C21_simulation", q, chi)]
                ax.plot(c[:, 0], c[:, 1], "--", color=cols[chi], lw=1.2, label=f"C21 Fig. 18, chi = {chi:.2f}")
            for src, mk in (("Chubenko2014_experiment", "s"), ("Liu2017_experiment", "^")):
                c = ref.get((src, q, None))
                if c is not None:
                    ax.plot(c[:, 0], c[:, 1], mk, color="#555555", ms=4, mfc="none", label=src.replace("_", " "))
            ax.set_xlabel("photon energy, eV"); ax.set_ylabel(f"{q}, %")
            ax.grid(True, color="#e4e3dd", lw=0.6)
        axs[0].set_title(f"QE, p = 1e19 ({v})", fontsize=10)
        axs[1].set_title(f"ESP of emitted electrons ({v})", fontsize=10)
        axs[1].legend(fontsize=7, frameon=False, ncol=2)
        dg.save(fig, OUT / f"fig18_qe_esp_{vt}.png")

        fig, axs = plt.subplots(2, 3, figsize=(18, 9))
        ax = axs[0, 0]
        for chi in CHIS:
            rr = [r for r in res if r["chi"] == chi]
            ax.plot([r["hv"] for r in rr], [1e3 * r["MTE"] for r in rr], "-o", ms=3, color=cols[chi], lw=2,
                    label=f"chi = {chi:.2f}")
        ax.set_xlabel("photon energy, eV"); ax.set_ylabel("MTE = <E_perp>, meV"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("Mean transverse energy of emitted electrons", fontsize=10)
        ax = axs[0, 1]
        rr = [r for r in res if r["chi"] == 0.67]
        hv = [r["hv"] for r in rr]
        for j, name in enumerate(("Gamma", "L", "X")):
            ax.plot(hv, [100 * r["arr_valley"][j] / r["n_arr"] for r in rr], "-o", ms=3, lw=2, color=dg.SLOTS[j],
                    label=f"first arrival in {name}")
        ax.plot(hv, [100 * r["emit_valley"][2] / max(r["n_emit"], 1) for r in rr], "--s", ms=3, lw=1.5,
                color=dg.SLOTS[2], label="emitted from X (chi = 0.67)")
        ax.set_xlabel("photon energy, eV"); ax.set_ylabel("%"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("Valley at the surface", fontsize=10)
        ax = axs[0, 2]
        for chi in CHIS:
            rr = [r for r in res if r["chi"] == chi]
            ax.plot([r["hv"] for r in rr], [r["enc_emit"] for r in rr], "-o", ms=3, lw=2, color=cols[chi],
                    label=f"emitted, chi = {chi:.2f}")
            ax.plot([r["hv"] for r in rr], [r["enc_trap"] for r in rr], ":", lw=1.5, color=cols[chi])
        ax.set_xlabel("photon energy, eV"); ax.set_ylabel("mean surface encounters")
        ax.set_title("Encounters before emission (solid) / trapping (dotted)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        for ax, key, lab in ((axs[1, 0], "E_vac", "kinetic energy in vacuum above E_vac, eV"),
                             (axs[1, 1], "E_perp", "transverse energy in vacuum, eV")):
            for hvs, ls in ((1.50, "-"), (1.75, "--"), (2.10, ":")):
                r = [x for x in res if x["chi"] == 0.67 and abs(x["hv"] - hvs) < 1e-6]
                if not r or r[0][key].size == 0:
                    continue
                bins = np.linspace(0, 0.35 if key == "E_vac" else 0.12, 71)
                h, b = np.histogram(r[0][key], bins)
                ax.plot(0.5 * (b[1:] + b[:-1]), h / max(h.sum(), 1) / np.diff(b), ls, lw=2, color=dg.SLOTS[0],
                        label=f"hv = {hvs:.2f} eV (n = {r[0][key].size})")
            ax.set_xlabel(lab); ax.set_ylabel("pdf, 1/eV"); ax.legend(fontsize=8, frameon=False)
            ax.set_title("Emitted electrons, chi = 0.67 eV", fontsize=10)
        ax = axs[1, 2]
        for chi in CHIS:
            r = [x for x in res if x["chi"] == chi and abs(x["hv"] - 1.75) < 1e-6]
            if r and r[0]["E_perp"].size:
                h, b = np.histogram(r[0]["E_perp"], np.linspace(0, 0.12, 61))
                ax.plot(0.5 * (b[1:] + b[:-1]), h / max(h.sum(), 1) / np.diff(b), lw=2, color=cols[chi],
                        label=f"chi = {chi:.2f}, MTE {1e3 * r[0]['MTE']:.1f} meV")
        ax.set_xlabel("transverse energy, eV"); ax.set_ylabel("pdf, 1/eV"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("Transverse energy, hv = 1.75 eV", fontsize=10)
        for a in axs.ravel():
            a.grid(True, color="#e4e3dd", lw=0.6)
        dg.save(fig, OUT / f"fig18_details_{vt}.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16000,
            int(sys.argv[4]) if len(sys.argv) > 4 else 14)
    elif cmd == "surface":
        surface(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 14)
    elif cmd == "report":
        report(tuple(sys.argv[2:]) or ("bulk/vmass",))
    else:
        raise SystemExit(__doc__)
