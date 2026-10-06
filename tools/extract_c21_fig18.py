"""Extract the curves of Chubenko et al. (2021) Fig. 18 from the vector graphics of the accepted
manuscript (refs/Chubenko2021_JAP130_063101_accepted_manuscript.pdf, page 16).

Fig. 18 is embedded as vector paths, so this is an exact read-out of the plotted values, not image
digitization:
  * simulation curves: one polyline per chi (stroke color), 16 vertices at hv = 1.45 ... 2.20 eV;
  * axis calibration: the axis frame and tick-mark strokes (QE panel y: 0 -> 20 %, ESP panel y:
    0 -> 40 %; x: 1.5 -> 2.2 eV ticks);
  * experimental points of Chubenko 2014 (red): the filled-circle marker glyphs (text characters);
    their glyph-box centers are corrected by the mean offset between the open-circle glyphs and the
    vertices of the chi = 0.64 eV polyline they sit on;
  * Liu 2017 (blue): its polyline.
Legend -> color: chi = 0.64 green, 0.67 orange, 0.70 purple, 0.73 black (legend glyphs o, square,
triangle, diamond, matched to the markers on each polyline).
Error bars of the simulated ESP (black strokes) are not extracted.

Output: validation/reference/c21_fig18.csv
    python tools/extract_c21_fig18.py
"""
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "refs" / "Chubenko2021_JAP130_063101_accepted_manuscript.pdf"
OUT = ROOT / "validation" / "reference" / "c21_fig18.csv"
PAGE = 15                                   # 0-based (printed page 15, PDF page 16)

COLORS = {(0.0, 0.67, 0.0): 0.64, (1.0, 0.5, 0.0): 0.67, (0.5, 0.0, 0.5): 0.70, (0.0, 0.0, 0.0): 0.73}
RED, BLUE = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
# panels: (y range of the frame in pt, data value at the bottom and top frame lines)
PANELS = {"QE": dict(y0=381.72, y1=256.51, v1=20.0), "ESP": dict(y0=576.58, y1=451.38, v1=40.0)}
X_TICK = {1.5: 193.72, 2.2: 369.0}


def _xcal(x):
    (a, xa), (b, xb) = X_TICK.items()
    return a + (x - xa) * (b - a) / (xb - xa)


def _ycal(y, panel):
    p = PANELS[panel]
    return (p["y0"] - y) * p["v1"] / (p["y0"] - p["y1"])


def _panel(y):
    for name, p in PANELS.items():
        if p["y1"] - 2 <= y <= p["y0"] + 2:
            return name
    return None


def _key(c):
    return tuple(round(v, 2) for v in c)


def _check_frame(page):
    """The axis frames and ticks must be where the calibration assumes."""
    ys = set()
    for d in page.get_drawings():
        if d["color"] and max(d["color"]) < 0.01 and len(d["items"]) == 1:
            p0, p1 = d["items"][0][1], d["items"][0][2]
            if abs(p0.y - p1.y) < 1e-3 and abs(p0.x - p1.x) > 150:
                ys.add(round(p0.y, 2))
    for p in PANELS.values():
        assert {p["y0"], p["y1"]} <= ys, (p, sorted(ys))


def extract():
    page = pymupdf.open(PDF)[PAGE]
    _check_frame(page)
    rows = []
    sim = {}
    for d in page.get_drawings():
        if not d["color"] or len(d["items"]) < 11:
            continue
        pts = [d["items"][0][1]] + [it[-1] for it in d["items"]]
        x = np.array([q.x for q in pts]); y = np.array([q.y for q in pts])
        panel = _panel(np.median(y))
        if panel is None:
            continue
        c = _key(d["color"])
        if c in COLORS:
            hv = _xcal(x)
            assert np.allclose(np.diff(hv), 0.05, atol=2e-3), hv
            sim[(panel, COLORS[c])] = (x, y)
            for h, v in zip(hv, _ycal(y, panel)):
                rows.append(("C21_simulation", panel, COLORS[c], round(h, 3), v))
        elif c == BLUE:
            for h, v in zip(_xcal(x), _ycal(y, panel)):
                rows.append(("Liu2017_experiment", panel, np.nan, h, v))
    assert len(sim) == 8, sorted(sim)
    # experimental Chubenko 2014: filled-circle glyphs; offset from open circles on the 0.64 curve
    glyphs = {"open": [], "filled": []}
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                t = s["text"].strip()
                if t in ("○", "●"):
                    x0, y0, x1, y1 = s["bbox"]
                    glyphs["open" if t == "○" else "filled"].append(((x0 + x1) / 2, (y0 + y1) / 2))
    offs = []
    for panel in PANELS:
        xs, ys = sim[(panel, 0.64)]
        for gx, gy in glyphs["open"]:
            j = np.argmin(np.abs(xs - gx))
            if abs(xs[j] - gx) < 0.5 and abs(ys[j] - gy) < 3:
                offs.append((gx - xs[j], gy - ys[j]))
    offs = np.array(offs)
    assert len(offs) == 32, len(offs)
    dx, dy = offs.mean(axis=0)
    for gx, gy in glyphs["filled"]:
        gx, gy = gx - dx, gy - dy
        panel = _panel(gy)
        if panel is None or gx < 190:            # legend glyphs sit at x ~ 187
            continue
        if panel == "ESP" and gx > 280 and gy < 465:  # ESP legend glyph
            continue
        rows.append(("Chubenko2014_experiment", panel, np.nan, _xcal(gx), _ycal(gy, panel)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="\n") as f:
        f.write("# Chubenko et al., J. Appl. Phys. 130, 063101 (2021), Fig. 18; p = 1e19 cm^-3.\n")
        f.write("# Extracted from the vector graphics of the accepted manuscript by tools/extract_c21_fig18.py\n")
        f.write(f"# glyph-center offset correction for experimental markers: dx={dx:.3f} pt, dy={dy:.3f} pt "
                f"(std {offs[:, 1].std():.3f} pt; 1 pt = {20 / (381.72 - 256.51):.4f} % QE, "
                f"{40 / (576.58 - 451.38):.4f} % ESP)\n")
        f.write("source,quantity,chi_eV,hv_eV,value_percent\n")
        for src, q, chi, h, v in rows:
            f.write(f"{src},{q},{'' if np.isnan(chi) else f'{chi:.2f}'},{h:.4f},{v:.3f}\n")
    print(f"wrote {OUT} ({len(rows)} rows), glyph offset dx={dx:.3f} dy={dy:.3f} pt")


if __name__ == "__main__":
    extract()
