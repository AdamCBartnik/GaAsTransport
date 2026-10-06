"""Measured optical data shipped with the package.

casey1975_ptype()
    Near-edge absorption of p-type GaAs at 297 K from H. C. Casey, Jr., D. D. Sell, K. W. Wecht,
    J. Appl. Phys. 46, 250 (1975), Figs. 6-8 (hole concentrations 1.6e16 to 1.6e19 cm^-3).
    Digitized by tools/digitize_casey1975.py; raw points with provenance are in
    gaas_mc/data/casey1975_ptype.csv. See that script's docstring for the method and uncertainties.

Assembly of one spectrum per hole concentration from the raw points:
  * Several figures can show the same sample (e.g. 2.2e17 cm^-3 in Figs. 6, 7, and 8). Each figure's
    points are interpolated in ln(alpha) on a 1 meV grid, but only across gaps of at most
    MAX_GAP_EV between neighbouring points.
  * Where more than one figure covers a grid energy, the values are averaged in ln(alpha). Points
    from a crossing shared by curves that coincide within the drawn line width ("merged" in the CSV)
    are used only where no figure resolves the curve separately.
  * Grid energies covered by no figure (where curves cross or labels hide a curve) are filled by
    linear interpolation in ln(alpha) and flagged in the returned "bridged" mask.
  * The high-purity curve is not a p-type dataset and is returned separately by casey1975_high_purity().
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .constants import EV

DATA = Path(__file__).parent / "data" / "casey1975_ptype.csv"
MAX_GAP_EV = 0.006
GRID_STEP_EV = 0.001


def _read_raw(path=DATA):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fig, curve, carrier, conc, hv, alpha, merged = line.split(",")
        rows.append((fig, curve, carrier, float(conc), float(hv), float(alpha), int(merged)))
    return rows


def _coverage(grid, pts):
    """ln(alpha) of one point set on the grid, and the mask of grid energies it covers."""
    pts = pts[np.argsort(pts[:, 0])]
    E, la = pts[:, 0], np.log(pts[:, 1])
    Eu, inv = np.unique(np.round(E, 6), return_inverse=True)
    lau = np.bincount(inv, la) / np.bincount(inv)
    if Eu.size < 2:
        return np.zeros(grid.size), np.zeros(grid.size, bool)
    j = np.searchsorted(Eu, grid)
    inside = (j > 0) & (j < Eu.size)
    jj = np.clip(j, 1, Eu.size - 1)
    ok = inside & (Eu[jj] - Eu[jj - 1] <= MAX_GAP_EV + 1e-9)
    ok |= np.isin(grid, np.round(Eu, 6))
    return np.interp(grid, Eu, lau), ok


def _assemble(points_by_fig):
    """points_by_fig: {(fig, merged): array (n, 2) of (hv_eV, alpha_cm1)} -> (grid, alpha_cm1, bridged)."""
    lo = min(p[:, 0].min() for p in points_by_fig.values())
    hi = max(p[:, 0].max() for p in points_by_fig.values())
    grid = np.round(np.arange(np.ceil(lo / GRID_STEP_EV) * GRID_STEP_EV, hi + 1e-9, GRID_STEP_EV), 6)
    cov = {key: _coverage(grid, pts) for key, pts in points_by_fig.items()}
    resolved = np.zeros(grid.size, bool)
    for (fig, merged), (_, ok) in cov.items():
        if not merged:
            resolved |= ok
    acc = np.zeros(grid.size)
    cnt = np.zeros(grid.size)
    for (fig, merged), (vals, ok) in cov.items():
        use = ok & ~resolved if merged else ok
        acc[use] += vals[use]
        cnt[use] += 1
    covered = cnt > 0
    la = np.full(grid.size, np.nan)
    la[covered] = acc[covered] / cnt[covered]
    bridged = ~covered
    la[bridged] = np.interp(grid[bridged], grid[covered], la[covered])
    return grid, np.exp(la), bridged


@lru_cache(maxsize=None)
def _casey_curves():
    raw = _read_raw()
    curves = {}
    for fig, curve, carrier, conc, hv, alpha, merged in raw:
        curves.setdefault((curve, carrier, conc), {}).setdefault((fig, merged), []).append((hv, alpha))
    out = {}
    for (curve, carrier, conc), figs in curves.items():
        grid, alpha, bridged = _assemble({k: np.array(v) for k, v in figs.items()})
        out[curve] = dict(label=f"Casey 1975 {curve}", carrier=carrier, p=conc * 1e6,
                          hv=grid * EV, alpha=alpha * 100.0, bridged=bridged,
                          figures=tuple(sorted({f for f, _ in figs})))
    return out


def casey1975_ptype():
    """List of p-type spectra: dicts with p [m^-3], hv [J], alpha [1/m], bridged mask, label, figures."""
    return [dict(d) for d in _casey_curves().values() if d["carrier"] == "p"]


def casey1975_high_purity():
    return dict(_casey_curves()["hp"])
