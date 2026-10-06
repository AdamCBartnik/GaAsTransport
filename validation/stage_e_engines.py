"""Full-length (370 ps) cross-check of the two transport engines on the Fig. 18 problem.

Compares the first-arrival ensembles of the reference engine (validation/out/stageE/bulk/arrivals,
chunks of 2000 electrons from `stage_e_fig18.py run bulk`) with the fast GPU engine
(validation/out/stageE/bulk_fast/arrivals, `stage_e_fig18.py fast bulk`) at every photon energy for
which both exist: arrival fraction, arrival-time quantiles, valley fractions, ESP and mean energy at
arrival, each with its statistical error.

    python validation/stage_e_engines.py [bulk]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gaas_mc.constants import PS, to_ev                       # noqa: E402
from gaas_mc.surface import SurfaceArrivals                   # noqa: E402

OUT = ROOT / "validation" / "out" / "stageE"


def load(dirname, hv):
    files = sorted((OUT / dirname / "arrivals").glob(f"hv{hv:.2f}_c*_arr.npz"))
    if not files:
        return None
    arrs = [SurfaceArrivals.load_npz(f) for f in files]
    n = sum(int(np.load(str(f).replace("_arr", "_gen"))["n"]) for f in files)
    cat = lambda k: np.concatenate([getattr(a, k) for a in arrs])
    return n, cat("t"), cat("E"), cat("valley"), cat("spin")


def main(variant="bulk"):
    lines = [f"Reference ({variant}) vs fast GPU ({variant}_fast): first arrivals at 370 ps, p = 1e19",
             "  hv   N_ref  N_fast | arrived ref/fast (z)  | median t ps ref/fast | t90 ps ref/fast | "
             "Gamma/L/X ref | Gamma/L/X fast | ESP ref/fast (z) | <E> meV ref/fast (z)"]
    zs = []
    for hv in np.round(np.arange(1.45, 2.2001, 0.05), 2):
        r, f = load(variant, hv), load(f"{variant}_fast", hv)
        if r is None or f is None:
            continue
        (nr, tr, Er, vr, sr), (nf, tf, Ef, vf, sf) = r, f
        pr, pf = tr.size / nr, tf.size / nf
        z_arr = (pr - pf) / np.sqrt(pr * (1 - pr) / nr + pf * (1 - pf) / nf)
        er, ef = sr.mean(), sf.mean()
        z_esp = (er - ef) / np.sqrt((1 - er**2) / sr.size + (1 - ef**2) / sf.size)
        Emr, Emf = 1e3 * to_ev(Er), 1e3 * to_ev(Ef)
        z_E = (Emr.mean() - Emf.mean()) / np.sqrt(Emr.var() / Emr.size + Emf.var() / Emf.size)
        fr = np.bincount(vr, minlength=3) / vr.size
        ff = np.bincount(vf, minlength=3) / vf.size
        zs += [z_arr, z_esp, z_E]
        lines.append(f"{hv:.2f} {nr:6d} {nf:7d} | {pr:.4f}/{pf:.4f} ({z_arr:+.1f}) | "
                     f"{np.median(tr) / PS:6.1f}/{np.median(tf) / PS:6.1f} | "
                     f"{np.percentile(tr, 90) / PS:6.1f}/{np.percentile(tf, 90) / PS:6.1f} | "
                     f"{fr[0]:.3f}/{fr[1]:.3f}/{fr[2]:.3f} | {ff[0]:.3f}/{ff[1]:.3f}/{ff[2]:.3f} | "
                     f"{er:.4f}/{ef:.4f} ({z_esp:+.1f}) | {Emr.mean():.1f}/{Emf.mean():.1f} ({z_E:+.1f})")
    if zs:
        zs = np.array(zs)
        lines.append(f"z-scores: n = {zs.size}, rms {np.sqrt(np.mean(zs**2)):.2f} (1 expected), "
                     f"max |z| {np.abs(zs).max():.2f}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / f"engines_{variant}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bulk")
