"""List dark-run crossings along lines of constant alpha (or constant E) in calibrated coordinates.

usage: python tools/digitize_scan.py <figure-key> alpha <a1,a2,...> <E_min> <E_max>
       python tools/digitize_scan.py <figure-key> energy <E1,E2,...> <a_min> <a_max>
Prints the data coordinate of each dark run's center and its width (px) for anchor selection.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from digitize_lib import calibrate, load_scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def runs_along(img, cal, fixed, vals, lo, hi, kind, n=4000, thresh=110):
    out = {}
    for v in vals:
        if kind == "alpha":
            t = np.linspace(lo, hi, n)
            uv = np.column_stack([t, np.full(n, np.log10(v))])
        else:
            t = np.linspace(np.log10(lo), np.log10(hi), n)
            uv = np.column_stack([np.full(n, v), t])
        P = cal.to_pixel(uv)
        xi = np.clip(np.round(P[:, 0]).astype(int), 0, img.shape[1] - 1)
        yi = np.clip(np.round(P[:, 1]).astype(int), 0, img.shape[0] - 1)
        dark = img[yi, xi] < thresh
        step_px = np.hypot(*np.diff(P, axis=0).T).mean()
        res, i = [], 0
        while i < n:
            if dark[i]:
                j = i
                while j + 1 < n and dark[j + 1]:
                    j += 1
                c = 0.5 * (t[i] + t[j])
                res.append((c if kind == "alpha" else 10 ** c, (j - i + 1) * step_px))
                i = j + 1
            else:
                i += 1
        out[v] = res
    return out


if __name__ == "__main__":
    key, kind, vals, lo, hi = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")], \
        float(sys.argv[4]), float(sys.argv[5])
    cfg = json.loads((Path(__file__).parent / "casey1975_digitization.json").read_text())["figures"][key]
    img = load_scan(ROOT / cfg["pdf"], cfg["page"], tuple(cfg["crop"]))
    cal, *_ = calibrate(img, cfg["vlines"], cfg["hlines"])
    for v, res in runs_along(img, cal, None, vals, lo, hi, kind).items():
        print(f"{kind}={v:g}: " + "  ".join(f"{c:.4f}({w:.0f})" if kind == "alpha" else f"{c:.4g}({w:.0f})"
                                            for c, w in res))
