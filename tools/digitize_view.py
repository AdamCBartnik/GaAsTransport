"""Render a zoomed tile of a calibrated scan with a fine data-coordinate grid, for choosing anchor
points (E [eV], alpha [cm^-1]) by eye. Used while preparing tools/casey1975_digitization.json.

usage: python tools/digitize_view.py <figure-key> <E_min> <E_max> <alpha_min> <alpha_max> <out.png> [scale]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from digitize_lib import calibrate, load_scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(key, e0, e1, a0, a1, out, scale=3):
    cfg = json.loads((Path(__file__).parent / "casey1975_digitization.json").read_text())["figures"][key]
    img = load_scan(ROOT / cfg["pdf"], cfg["page"], tuple(cfg["crop"]))
    cal, *_ = calibrate(img, cfg["vlines"], cfg["hlines"])
    p = cal.to_pixel([[e0, np.log10(a1)], [e1, np.log10(a0)], [e0, np.log10(a0)], [e1, np.log10(a1)]])
    x0, x1 = int(p[:, 0].min()) - 10, int(p[:, 0].max()) + 10
    y0, y1 = int(p[:, 1].min()) - 10, int(p[:, 1].max()) + 10
    tile = Image.fromarray(img[y0:y1, x0:x1].astype(np.uint8)).convert("RGB")
    tile = tile.resize((tile.width * scale, tile.height * scale), Image.LANCZOS)
    d = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    Es = np.round(np.arange(np.ceil(e0 * 200) / 200, e1 + 1e-9, 0.005), 3)
    decades = range(int(np.floor(np.log10(a0))), int(np.ceil(np.log10(a1))) + 1)
    alphas = [m * 10**k for k in decades for m in (1, 2, 3, 5, 7)]
    alphas = [a for a in alphas if a0 <= a <= a1]
    lo, hi = np.log10(a0), np.log10(a1)
    for e in Es:
        q = (cal.to_pixel([[e, lo], [e, hi]]) - [x0, y0]) * scale
        major = abs(e * 100 - round(e * 100)) < 1e-6
        d.line([tuple(q[0]), tuple(q[1])], fill=(255, 0, 0) if major else (255, 170, 170), width=1)
        if major:
            d.text((q[1][0] + 2, q[1][1] + 2), f"{e:.2f}", fill=(255, 0, 0), font=font)
    for a in alphas:
        q = (cal.to_pixel([[e0, np.log10(a)], [e1, np.log10(a)]]) - [x0, y0]) * scale
        d.line([tuple(q[0]), tuple(q[1])], fill=(0, 120, 255), width=1)
        d.text((q[0][0] + 2, q[0][1] - 11), f"{a:g}", fill=(0, 90, 255), font=font)
    tile.save(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]), float(a[2]), float(a[3]), float(a[4]), a[5], int(a[6]) if len(a) > 6 else 3)
