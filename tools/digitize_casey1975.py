"""Digitize the p-type absorption spectra of Casey, Sell & Wecht, J. Appl. Phys. 46, 250 (1975).

The paper gives alpha(E) only as figures (Figs. 6-8, 297 K, log scale 10 to 1e5 cm^-1, 1.30-1.60 eV),
so the curves are digitized from the page scans embedded in refs/Casey1975_JAP46_250.pdf.

Reproduce:   python tools/digitize_casey1975.py
Inputs:      tools/casey1975_digitization.json (calibration guesses and sweep segments)
Outputs:     gaas_mc/data/casey1975_ptype.csv          digitized points with provenance header
             tools/out/casey1975_<fig>_overlay.png       digitized points drawn on the scan (check these)
             tools/out/casey1975_report.txt             calibration residuals, skipped sweeps, cross-checks

Method
------
1. Calibration (tools/digitize_lib.py): each grid line of the figure (E = 1.30, 1.40, 1.50, 1.60 eV;
   alpha = 10 ... 1e5 cm^-1) is located at 12 positions along its length and fitted with a straight
   line. A piecewise-linear map between neighbouring grid lines is exact on every grid line and
   absorbs scan skew and the slight nonuniformity of the hand-drawn grid.
2. Sweeps: lines of constant alpha ("row") or constant E ("col") are sampled in calibrated
   coordinates. Each dark run along a sweep line is one crossing; its center is a data point.
   For each segment the config lists the expected number of curves in a data window, in their
   order (left to right for rows, bottom to top for columns). A sweep line is used only if,
   after removing runs that coincide with a grid line and runs too wide to be a single curve
   (labels, arrows, merged curves), exactly the expected number of crossings remain. Otherwise it is
   skipped and logged. This makes curve identity a property of the ordering, never a guess at
   a crossing.
3. Curves that merge within the drawn line width (e.g. 1.2e18 and high purity above ~1.50 eV) are
   assigned the same crossing and flagged "merged" in the CSV.
Uncertainty: line width ~3-4 px ~ +-1.5 meV in E and +-0.01 in log10(alpha) (+-2.5%) on top of the
paper's own stated +-15% (Kramers-Kronig part, alpha > 1e3 cm^-1).
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from digitize_lib import calibrate, load_scan, overlay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = Path(__file__).parent / "casey1975_digitization.json"
OUT = Path(__file__).parent / "out"


def sweep(img, cal, kind, level, window, n=6000, thresh=110):
    """Dark-run crossings along one sweep line. Returns list of (coordinate, width_px)."""
    lo, hi = window
    if kind == "row":                       # constant alpha, variable E
        t = np.linspace(lo, hi, n)
        uv = np.column_stack([t, np.full(n, np.log10(level))])
    else:                                   # constant E, variable log10 alpha
        t = np.linspace(np.log10(lo), np.log10(hi), n)
        uv = np.column_stack([np.full(n, level), t])
    P = cal.to_pixel(uv)
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, img.shape[0] - 1)
    dark = img[yi, xi] < thresh
    step = np.hypot(*np.diff(P, axis=0).T).mean()
    out, i = [], 0
    while i < n:
        if dark[i]:
            j = i
            while j + 1 < n and dark[j + 1]:
                j += 1
            out.append((0.5 * (t[i] + t[j]), (j - i + 1) * step))
            i = j + 1
        else:
            i += 1
    return out


def is_grid(kind, coord, grid_E, grid_la, tol_E=0.0015, tol_la=0.011):
    if kind == "row":
        return any(abs(coord - g) < tol_E for g in grid_E)
    return any(abs(coord - g) < tol_la for g in grid_la)


def run_segment(img, cal, seg, grid_E, grid_la, log):
    kind = seg["kind"]
    a, b, m = seg["levels"]
    levels = np.geomspace(a, b, int(m)) if kind == "row" else np.linspace(a, b, int(m))
    exclude = [tuple(x) for x in seg.get("exclude_levels", [])]
    names = seg["assign"]                        # "a+b": one crossing shared by merged curves
    got = {n: [] for nm in names if nm for n in nm.split("+")}
    used = skipped = 0
    for lev in levels:
        if any(lo <= lev <= hi for lo, hi in exclude):
            continue
        runs = sweep(img, cal, kind, lev, seg["window"])
        runs = [(c, w) for c, w in runs if w >= seg.get("min_width", 1.5)]
        runs = [(c, w) for c, w in runs if w <= seg.get("max_width", 8.0)]
        if len(runs) > len(names) or seg.get("always_drop_grid", False):
            runs = [(c, w) for c, w in runs if not (is_grid(kind, c, grid_E, grid_la) and w <= 5)]
        if len(runs) != len(names):
            skipped += 1
            log.append(f"  skip {kind} {lev:.5g}: {len(runs)} crossings "
                       f"{[round(c if kind == 'row' else 10**c, 4) for c, _ in runs]}")
            continue
        used += 1
        for nm, (c, _) in zip(names, runs):
            if not nm:
                continue
            E, la = (c, np.log10(lev)) if kind == "row" else (lev, c)
            parts = nm.split("+")
            for n in parts:
                got[n].append((E, la, int(len(parts) > 1)))
    log.append(f"segment {seg['id']}: used {used}, skipped {skipped}")
    return got


def reject_outliers(cal, pts, max_px, k=3):
    """Drop points farther than max_px (scan pixels) from the straight line fitted to their k
    neighbours on each side (the point itself excluded). Points are ordered by E; this is
    adequate because every digitized curve is single-valued and monotonic over the swept ranges."""
    if len(pts) < 2 * k + 1:
        return pts, 0
    P = cal.to_pixel(pts[:, :2])
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        nb = [j for j in range(max(0, i - k), min(len(pts), i + k + 1)) if j != i]
        Q = P[nb]
        c = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - c)
        normal = vt[1]
        if abs(np.dot(P[i] - c, normal)) > max_px:
            keep[i] = False
    return pts[keep], int((~keep).sum())


def reject_inconsistent(pts, half_window_eV=0.0025, max_dex=0.025, min_neighbours=4):
    """Robust consistency filter in data space: for each point, fit a Theil-Sen line (median of
    pairwise slopes, median intercept; tolerant of ~29% contamination) to the other points within
    +-half_window_eV, and drop the point if its log10(alpha) deviates by more than max_dex (6%).
    This removes clusters of mis-assigned crossings (labels/arrowheads) that a neighbour-line
    fit cannot, because the neighbours themselves are contaminated."""
    E, la = pts[:, 0], pts[:, 1]
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        nb = np.flatnonzero((np.abs(E - E[i]) <= half_window_eV) & (np.arange(len(pts)) != i))
        if nb.size < min_neighbours:
            continue
        x, y = E[nb], la[nb]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        m = np.abs(dx) > 1e-6
        if not m.any():
            continue
        slope = np.median(dy[m] / dx[m])
        icpt = np.median(y - slope * x)
        if abs(la[i] - (slope * E[i] + icpt)) > max_dex:
            keep[i] = False
    return pts[keep], int((~keep).sum())


def main():
    cfg = json.loads(CFG.read_text())
    OUT.mkdir(exist_ok=True)
    log = []
    rows = []
    for fkey, fig in cfg["figures"].items():
        if not fig.get("segments"):
            continue
        img = load_scan(ROOT / fig["pdf"], fig["page"], tuple(fig["crop"]))
        cal, affine, vfits, hfits = calibrate(img, fig["vlines"], fig["hlines"])
        log.append(f"== {fkey}: affine-fit residual (for reference) max |dE| = "
                   f"{np.abs(affine.residual[:, 0]).max() * 1e3:.2f} meV, max |dlog10a| = "
                   f"{np.abs(affine.residual[:, 1]).max():.4f}; grid-warp map is exact on grid lines")
        grid_E = [v for v, _ in fig["vlines"]]
        grid_la = [v for v, _ in fig["hlines"]]
        curves = {}
        for seg in fig["segments"]:
            for nm, pts in run_segment(img, cal, seg, grid_E, grid_la, log).items():
                curves.setdefault(nm, []).extend(pts)
        ov = []
        for nm, pts in curves.items():
            pts = np.array(sorted(set(pts)))
            pts, n_out = reject_outliers(cal, pts, fig.get("outlier_px", 2.5))
            pts, n_bad = reject_inconsistent(pts)
            if n_bad:
                log.append(f"  {fkey} {nm}: removed {n_bad} inconsistent point(s) (robust local fit, > 6%)")
            mflag = pts[:, 2].astype(int)
            pts = pts[:, :2]
            if n_out:
                log.append(f"  {fkey} {nm}: removed {n_out} outlier(s) (> {fig.get('outlier_px', 2.5)} px from neighbours)")
            label = fig["curves"][nm]
            for (E, la), mf in zip(pts, mflag):
                rows.append((fkey, nm, label["carrier"], label["conc_cm3"], E, 10 ** la, int(mf)))
            ov.append((nm, pts[:, 0], 10 ** pts[:, 1], np.ones(len(pts), bool)))
            log.append(f"  {fkey} {nm}: {len(pts)} points, E {pts[0, 0]:.4f}-{pts[-1, 0]:.4f} eV, "
                       f"alpha {10 ** pts[:, 1].min():.3g}-{10 ** pts[:, 1].max():.3g} cm^-1")
        overlay(img, cal, ov, OUT / f"casey1975_{fkey}_overlay.png",
                grid=(grid_E, grid_la), scale=2)
    write_csv(cfg, rows)
    (OUT / "casey1975_report.txt").write_text("\n".join(log) + "\n")
    print("\n".join(l for l in log if not l.startswith("  skip")))


def write_csv(cfg, rows):
    path = ROOT / "gaas_mc" / "data" / "casey1975_ptype.csv"
    path.parent.mkdir(exist_ok=True)
    head = [
        "# Digitized absorption coefficient of GaAs at 297 K",
        f"# Source: {cfg['source']}",
        "# Figures 6-8 (p-type and high-purity); digitized from the page scans in the publisher PDF",
        f"# Produced by tools/digitize_casey1975.py with tools/casey1975_digitization.json on {datetime.date.today()}",
        "# Uncertainty: digitization ~ +-1.5 meV in E, ~ +-2.5% in alpha; paper: +-15% for alpha > 1e3 cm^-1",
        "# merged = 1: this crossing is shared by curves that coincide within the drawn line width",
        "# columns: figure,curve,carrier,conc_cm3,hv_eV,alpha_cm1,merged",
    ]
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(head) + "\n")
        for r in sorted(rows, key=lambda r: (r[1], r[0], r[4])):
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.3g},{r[4]:.5f},{r[5]:.5g},{r[6]}\n")
    print("wrote", path, len(rows), "points")


if __name__ == "__main__":
    main()
