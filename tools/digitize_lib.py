"""Small, dependency-light helpers for digitizing curves from scanned figures (numpy + Pillow).

Workflow (see tools/digitize_casey1975.py):
  1. load_scan(): the embedded page scan from the PDF, cropped to one figure (grayscale, 0 = black).
  2. calibrate(): locate the plot's grid lines (known data values) along several scan lines, and fit an
     affine map pixel -> (E, log10 alpha) to their intersections. This absorbs small rotation and skew
     of the scan. The residuals are reported.
  3. trace(): follow one curve through user-supplied anchor points given in DATA coordinates. Between
     anchors the path is sampled densely, and each sample is snapped to the nearest dark-pixel ridge
     along the local normal, within a small window. The anchors only fix the curve's identity (which
     branch at crossings); the snapped ridge positions are the digitized data.
  4. overlay(): draw the digitized points on the scan for visual verification.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def load_scan(pdf_path, page, box):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    xref = doc[page].get_images(full=True)[0][0]
    pix = pymupdf.Pixmap(doc, xref)
    img = Image.frombytes("L" if pix.n == 1 else "RGB", (pix.width, pix.height), pix.samples).convert("L")
    return np.asarray(img.crop(box), dtype=float)


def _line_center(profile, guess, half_window, thresh):
    """Center (intensity-weighted) of the dark run nearest `guess` in a 1D profile."""
    lo, hi = max(0, int(guess - half_window)), min(profile.size, int(guess + half_window) + 1)
    seg = 255.0 - profile[lo:hi]
    seg = np.where(seg > thresh, seg, 0.0)
    if seg.sum() == 0:
        return np.nan
    return lo + np.sum(np.arange(seg.size) * seg) / seg.sum()


def find_line_positions(img, axis, guess, samples, half_window=6, thresh=100, band=3):
    """For a near-vertical (axis='x') or near-horizontal (axis='y') grid line, measure its position
    at several sample coordinates along the line. Returns an array of (x, y) points on the line."""
    pts = []
    for s in samples:
        s = int(s)
        if axis == "x":                       # vertical line: profile along x at row s
            prof = img[s - band:s + band + 1, :].mean(axis=0)
            c = _line_center(prof, guess, half_window, thresh)
            pts.append((c, s))
        else:
            prof = img[:, s - band:s + band + 1].mean(axis=1)
            c = _line_center(prof, guess, half_window, thresh)
            pts.append((s, c))
    pts = np.array(pts, float)
    return pts[np.isfinite(pts).all(axis=1)]


def fit_line(pts, axis):
    """Least-squares x = a*y + b (vertical) or y = a*x + b (horizontal)."""
    if axis == "x":
        a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    else:
        a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
    return a, b


def intersect(vline, hline):
    """vertical x = a1 y + b1, horizontal y = a2 x + b2."""
    a1, b1 = vline
    a2, b2 = hline
    y = (a2 * b1 + b2) / (1 - a1 * a2)
    return a1 * y + b1, y


class Calibration:
    """Affine map pixel (x, y) <-> data (u, v) = (E [eV], log10 alpha)."""

    def __init__(self, pix, dat):
        pix, dat = np.asarray(pix, float), np.asarray(dat, float)
        A = np.column_stack([pix, np.ones(len(pix))])
        self.M, *_ = np.linalg.lstsq(A, dat, rcond=None)            # (3, 2)
        self.residual = dat - A @ self.M
        P = np.column_stack([dat, np.ones(len(dat))])
        self.Minv, *_ = np.linalg.lstsq(P, pix, rcond=None)

    def to_data(self, xy):
        xy = np.atleast_2d(xy)
        return np.column_stack([xy, np.ones(len(xy))]) @ self.M

    def to_pixel(self, uv):
        uv = np.atleast_2d(uv)
        return np.column_stack([uv, np.ones(len(uv))]) @ self.Minv


class GridCalibration:
    """Piecewise-linear map that is exact on every measured grid line.

    Each vertical grid line is x = a y + b (value E_i); each horizontal line is y = a x + b
    (value v_j = log10 alpha). At pixel (x, y), E is interpolated linearly between the two vertical
    lines bracketing x at height y, and log10 alpha between the two horizontal lines bracketing y at
    position x (linear extrapolation outside). This absorbs scan skew and the slight nonuniformity
    of the hand-drawn grid, which an affine map cannot.
    """

    def __init__(self, vfits, hfits):
        self.v = sorted(vfits, key=lambda t: t[0])
        self.h = sorted(hfits, key=lambda t: -t[0])          # top (large alpha) first, increasing y

    @staticmethod
    def _interp(pos, line_pos, vals):
        """Vectorized: pos (n,), line_pos (n, L) increasing along axis 1, vals (L,)."""
        L = line_pos.shape[1]
        j = np.clip((line_pos < pos[:, None]).sum(axis=1), 1, L - 1)
        r = np.arange(pos.size)
        p0, p1 = line_pos[r, j - 1], line_pos[r, j]
        f = (pos - p0) / (p1 - p0)
        return vals[j - 1] + f * (vals[j] - vals[j - 1])

    def _lines(self):
        av = np.array([a for _, (a, b) in self.v]); bv = np.array([b for _, (a, b) in self.v])
        ah = np.array([a for _, (a, b) in self.h]); bh = np.array([b for _, (a, b) in self.h])
        return av, bv, ah, bh, np.array([v for v, _ in self.v]), np.array([v for v, _ in self.h])

    def to_data(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        av, bv, ah, bh, Ev, Lh = self._lines()
        x, y = xy[:, 0], xy[:, 1]
        xl = av[None, :] * y[:, None] + bv[None, :]               # x of each vertical line at y
        yl = ah[None, :] * x[:, None] + bh[None, :]               # y of each horizontal line at x
        return np.column_stack([self._interp(x, xl, Ev), self._interp(y, yl, Lh)])

    def to_pixel(self, uv, iters=25):
        uv = np.atleast_2d(np.asarray(uv, float))
        av, bv, ah, bh, Ev, Lh = self._lines()
        E, la = uv[:, 0], uv[:, 1]
        y = np.full(E.size, bh.mean())
        for _ in range(iters):
            xl = av[None, :] * y[:, None] + bv[None, :]
            x = _interp_vals(E, Ev, xl)
            yl = ah[None, :] * x[:, None] + bh[None, :]
            y = _interp_vals(la, Lh[::-1], yl[:, ::-1])
        return np.column_stack([x, y])


def _interp_vals(val, grid_vals, positions):
    """Inverse lookup: grid_vals (L,) increasing; positions (n, L) pixel position of each grid line
    for each point. Returns the pixel position interpolated at data value val (n,)."""
    L = grid_vals.size
    j = np.clip(np.searchsorted(grid_vals, val), 1, L - 1)
    r = np.arange(val.size)
    f = (val - grid_vals[j - 1]) / (grid_vals[j] - grid_vals[j - 1])
    return positions[r, j - 1] + f * (positions[r, j] - positions[r, j - 1])


def calibrate(img, vlines, hlines, n_samples=12, margin=0.08):
    """vlines: list of (E_value, x_guess); hlines: list of (log10_alpha, y_guess). Guesses in pixels.
    Measures each grid line along n_samples positions spread across the plot interior."""
    xs = [g for _, g in vlines]
    ys = [g for _, g in hlines]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    ysamp = np.linspace(y0 + margin * (y1 - y0), y1 - margin * (y1 - y0), n_samples)
    xsamp = np.linspace(x0 + margin * (x1 - x0), x1 - margin * (x1 - x0), n_samples)
    vfits = [(val, fit_line(find_line_positions(img, "x", g, ysamp), "x")) for val, g in vlines]
    hfits = [(val, fit_line(find_line_positions(img, "y", g, xsamp), "y")) for val, g in hlines]
    pix, dat = [], []
    for ev, vl in vfits:
        for la, hl in hfits:
            pix.append(intersect(vl, hl))
            dat.append((ev, la))
    affine = Calibration(pix, dat)            # reported for comparison only
    return GridCalibration(vfits, hfits), affine, vfits, hfits


def snap_to_ridge(img, p, normal, half_window=7, thresh=80):
    """Move pixel point p along `normal` to the intensity-weighted center of the nearest dark run."""
    t = np.arange(-half_window, half_window + 1)
    q = p[None, :] + t[:, None] * normal[None, :]
    xi = np.clip(np.round(q[:, 0]).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.round(q[:, 1]).astype(int), 0, img.shape[0] - 1)
    dark = 255.0 - img[yi, xi]
    dark = np.where(dark > thresh, dark, 0.0)
    if dark.sum() == 0:
        return None
    # keep only the dark run that contains / is nearest to the center
    runs, cur = [], []
    for i, d in enumerate(dark):
        if d > 0:
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    center = half_window
    run = min(runs, key=lambda r: min(abs(i - center) for i in r))
    w = dark[run]
    tc = np.sum(t[run] * w) / w.sum()
    return p + tc * normal, len(run)


def trace(img, cal, anchors_data, step_px=3.0, half_window=6, thresh=80, max_run=9):
    """Trace a curve through anchor points given in data coordinates (E [eV], alpha [cm^-1]).
    Returns (E, alpha, ok) for densely sampled, ridge-snapped points. Points whose dark run is
    wider than max_run px (another curve, a label, or a grid line merging) are flagged ok = False."""
    A = np.array([(e, np.log10(a)) for e, a in anchors_data], float)
    P = cal.to_pixel(A)
    out, ok = [], []
    for i in range(len(P) - 1):
        p0, p1 = P[i], P[i + 1]
        seg = p1 - p0
        L = np.hypot(*seg)
        tangent = seg / L
        normal = np.array([-tangent[1], tangent[0]])
        n = max(2, int(L / step_px))
        for s in np.linspace(0, 1, n, endpoint=(i == len(P) - 2)):
            p = p0 + s * seg
            r = snap_to_ridge(img, p, normal, half_window, thresh)
            if r is None:
                continue
            q, width = r
            out.append(q)
            ok.append(width <= max_run)
    out = np.array(out)
    uv = cal.to_data(out)
    return uv[:, 0], 10 ** uv[:, 1], np.array(ok)


def overlay(img, cal, curves, path, grid=None, scale=2):
    """Save an RGB overlay: scan in gray, digitized curves as colored dots, calibration grid in cyan."""
    rgb = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    rgb = rgb.resize((rgb.width * scale, rgb.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(rgb)
    if grid is not None:
        E_ticks, la_ticks = grid
        for e in E_ticks:
            p = cal.to_pixel([[e, la_ticks[0]], [e, la_ticks[-1]]]) * scale
            d.line([tuple(p[0]), tuple(p[1])], fill=(0, 200, 220), width=1)
        for la in la_ticks:
            p = cal.to_pixel([[E_ticks[0], la], [E_ticks[-1], la]]) * scale
            d.line([tuple(p[0]), tuple(p[1])], fill=(0, 200, 220), width=1)
    colors = [(230, 30, 30), (30, 120, 230), (20, 170, 60), (220, 120, 0), (160, 40, 200), (0, 160, 160),
              (200, 0, 120)]
    for j, (label, E, a, ok) in enumerate(curves):
        P = cal.to_pixel(np.column_stack([E, np.log10(a)])) * scale
        c = colors[j % len(colors)]
        for (x, y), good in zip(P, ok):
            r = 2 if good else 1
            d.ellipse([x - r, y - r, x + r, y + r], fill=c if good else (120, 120, 120))
    rgb.save(path)
