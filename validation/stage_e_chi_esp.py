"""How does the emitted population change with chi? (ESP-vs-chi diagnostic for C21 Fig. 18.)

For each photon energy and chi: ESP of all emitted electrons next to C21's; the share and ESP of
early emission (t <= 10 ps); ESP binned by emission time and by the vacuum kinetic energy E_vac
(the energy above the vacuum level). If the surface becomes an energy selector at high chi, the
emitted population shifts to early, hot electrons and ESP rises; this shows how strongly that
happens in our model.

    python validation/stage_e_chi_esp.py [bulk_fast/bmass] [hv ...]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.stage_e_fig18 import CHIS, load, reference   # noqa: E402
from gaas_mc.constants import PS, to_ev                       # noqa: E402

TBINS = np.array([0, 3, 10, 30, 100, 370]) * PS
EBINS = np.array([0, 20, 50, 100, 200, 400, 1000]) * 1e-3


def esp_err(s):
    return (s.mean(), np.sqrt(max(1 - s.mean() ** 2, 0) / s.size)) if s.size else (np.nan, np.nan)


def main(tag="bulk_fast/bmass", hvs=(1.55, 1.65, 1.80, 1.90)):
    ref = reference()
    by_hv = load(tag)
    out = [f"{tag}: ESP vs chi (ours / C21, %), early emission, ESP by emission time and by E_vac"]
    for hv in hvs:
        parts = by_hv[round(hv, 2)]
        out.append(f"\nhv = {hv:.2f} eV")
        out.append("  chi   ESP ours(+-)  C21  | QE share t<=10ps  ESP(t<=10ps) | ESP by t bin "
                   + " ".join(f"[{a / PS:.0f},{b / PS:.0f})" for a, b in zip(TBINS[:-1], TBINS[1:]))
                   + " | share by t bin")
        for chi in CHIS:
            tg = f"chi{chi:.2f}_"
            t = np.concatenate([p[tg + "t"] for p in parts])
            s = np.concatenate([p[tg + "spin"] for p in parts]).astype(float)
            c21 = np.interp(hv, *ref[("C21_simulation", "ESP", chi)].T)
            e, de = esp_err(s)
            early = t <= 10 * PS
            ee, _ = esp_err(s[early])
            bins = np.digitize(t, TBINS) - 1
            eb = [esp_err(s[bins == b])[0] for b in range(len(TBINS) - 1)]
            sh = [np.mean(bins == b) for b in range(len(TBINS) - 1)]
            out.append(f"  {chi:.2f}  {100 * e:5.1f}({100 * de:.1f})  {c21:5.1f} | {early.mean():6.3f}        "
                       f"{100 * ee:5.1f}     | " + " ".join(f"{100 * x:7.1f}" for x in eb) + " | "
                       + " ".join(f"{x:.2f}" for x in sh))
        out.append("  ESP (%) by E_vac bin (meV) " + " ".join(f"[{1e3 * a:.0f},{1e3 * b:.0f})"
                                                           for a, b in zip(EBINS[:-1], EBINS[1:])))
        for chi in CHIS:
            tg = f"chi{chi:.2f}_"
            Ev = to_ev(np.concatenate([p[tg + "E_vac"] for p in parts]))
            s = np.concatenate([p[tg + "spin"] for p in parts]).astype(float)
            bins = np.digitize(Ev, EBINS) - 1
            out.append(f"  {chi:.2f}  " + " ".join(f"{100 * esp_err(s[bins == b])[0]:6.1f}({np.mean(bins == b):.2f})"
                                                    for b in range(len(EBINS) - 1)))
    txt = "\n".join(out)
    print(txt)
    return txt


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "bulk_fast/bmass"
    hvs = tuple(float(x) for x in sys.argv[2:]) or (1.55, 1.65, 1.80, 1.90)
    main(tag, hvs)
