"""Optical absorption of GaAs: replaceable ``absorption_coefficient(hv)`` interface.

Every absorption model provides
    absorption_coefficient(hv) -> alpha [1/m]   (hv in J, scalar or array)
and photoexcitation uses l = 1/alpha in [C21] Eq. 7. A model raises ValueError outside its range of
validity instead of returning an unphysical value.

Models
------
CaseyAdachiAbsorption(p)        default (docs/MODEL_ASSUMPTIONS.md)
    near edge   measured p-type absorption, Casey, Sell & Wecht, J. Appl. Phys. 46, 250 (1975)
                (gaas_mc/optical_data.py), interpolated in hv and in log10(p)
    blend       ln(alpha) blended smoothly over [1.55 eV, end of the Casey data, 1.592 eV]
    above       Adachi (1989) model dielectric function
    The measured p-type spectra already include the real doping-induced shift and broadening of
    the edge, so photons with Eg(p) < hv < Eg(intrinsic) are absorbed. No 1.42 eV cutoff applies.
DopingTabulatedAbsorption       generic alpha(hv; p) from several measured spectra
Adachi1989GaAs                  Adachi MDF alone (kept for comparison)
TabulatedAbsorption, ConstantAbsorptionLength   user data / idealized studies

Adachi (1989) MDF
-----------------
S. Adachi, "Optical dispersion relations for GaP, GaAs, GaSb, InP, InAs, InSb, AlxGa1-xAs, and
In1-xGaxAsyP1-y", J. Appl. Phys. 66, 6030 (1989). The implementation follows the public-domain
(CC0) transcription by M. Polyanskiy (refractiveindex.info-scripts, "Adachi 1989 - GaAs.py"), which
notes three gaps in the paper, reproduced here:
  (i)   the E1 + Delta1 term is omitted (no B2, B21 given);
  (ii)  negative eps2 from the E1 term is clipped to zero;
  (iii) the phonon energy in the indirect-gap term is set to zero.
All 14 GaAs parameters were verified against Adachi (1989) Table I (refs/Adachi1989_JAP66_6030.pdf).
Eg_ind = 1.73 eV is the Gamma8v -> L6c indirect gap (Table I, footnote c).
Limitations:
  * The E0 term has no broadening, exciton, or Urbach tail; it vanishes below E0 = 1.42 eV.
  * The E2 damped-oscillator term has a Lorentzian tail with eps2 > 0 at ALL energies (1.2e3 /cm at
    1.0 eV, 2.5e3 /cm at 1.40 eV). This tail is NOT fundamental interband absorption and must not
    drive photoexcitation: absorption_coefficient() raises below E0 unless the model is
    constructed with below_E0="model". Above E0 it adds about 2e3 /cm (~25% of alpha at 1.45 eV,
    ~7% at 1.9 eV), which is why the default uses measured data near the edge.
  * Intrinsic GaAs only. C21 Fig. 3 used Adachi's model *fitted* to Zollner (2001) data, with
    unpublished parameters.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .constants import EV, HBAR

C_LIGHT = 299_792_458.0


def _out(arr, like):
    """Return a float for scalar input and an array otherwise."""
    return float(np.squeeze(arr)) if np.ndim(like) == 0 else arr


# ---------------------------------------------------------------------------------- Adachi --

@dataclass(frozen=True)
class Adachi1989Params:
    E0: float = 1.42          # eV
    E0_D0: float = 1.77       # E0 + Delta0, eV
    E1: float = 2.90          # eV
    E1_D1: float = 3.13       # E1 + Delta1, eV (unused, see note (i))
    E2: float = 4.7           # eV
    Eg_ind: float = 1.73      # indirect gap (Gamma -> L), eV
    A: float = 3.45           # eV^1.5
    B1: float = 6.37
    B11: float = 13.08        # eV^-0.5
    Gamma: float = 0.10       # eV, broadening of the E1 term
    C: float = 2.39
    gamma: float = 0.146
    D: float = 24.2
    eps_inf: float = 1.6


class Adachi1989GaAs:
    """Adachi (1989) MDF for intrinsic GaAs at room temperature."""

    def __init__(self, params: Adachi1989Params = Adachi1989Params(), below_E0="raise"):
        if below_E0 not in ("raise", "model"):
            raise ValueError(below_E0)
        self.p = params
        self.below_E0 = below_E0

    def dielectric(self, hv_eV):
        """Complex dielectric function eps1 + i eps2 at photon energy hv [eV]."""
        p = self.p
        E = np.atleast_1d(np.asarray(hv_eV, float))
        x0, xs = E / p.E0, E / p.E0_D0           # E0, E0 + Delta0 (3D M0 critical points)

        def f(x):
            return x**-2 * (2 - np.sqrt(1 + x) - np.sqrt(np.clip(1 - x, 0, None)))
        eps2_A = p.A / E**2 * (np.sqrt(np.clip(E - p.E0, 0, None))
                               + 0.5 * np.sqrt(np.clip(E - p.E0_D0, 0, None)))
        eps1_A = p.A * p.E0**-1.5 * (f(x0) + 0.5 * (p.E0 / p.E0_D0) ** 1.5 * f(xs))
        x1 = E / p.E1                            # E1 (2D M0); E1 + Delta1 omitted (note (i))
        eps2_B = np.clip(np.pi * x1**-2 * (p.B1 - p.B11 * np.sqrt(np.clip(p.E1 - E, 0, None))), 0, None)
        x1c = (E + 1j * p.Gamma) / p.E1
        eps1_B = (-p.B1 * x1c**-2 * np.log(1 - x1c**2)).real
        x2 = E / p.E2                            # E2 (damped harmonic oscillator)
        den = (1 - x2**2) ** 2 + (x2 * p.gamma) ** 2
        eps2_C = p.C * x2 * p.gamma / den
        eps1_C = p.C * (1 - x2**2) / den
        # indirect gap, below E1 only; phonon energy neglected (note (iii))
        eps2_D = p.D / E**2 * np.clip(E - p.Eg_ind, 0, None) ** 2 * (E <= p.E1)
        return (p.eps_inf + eps1_A + eps1_B + eps1_C) + 1j * (eps2_A + eps2_B + eps2_C + eps2_D)

    def nk(self, hv_eV):
        n_c = np.sqrt(self.dielectric(hv_eV))
        return n_c.real, n_c.imag

    def absorption_coefficient(self, hv):
        """alpha [1/m] = 2 k omega / c at photon energy hv [J]."""
        hv_a = np.atleast_1d(np.asarray(hv, float))
        if self.below_E0 == "raise" and np.any(hv_a / EV < self.p.E0):
            raise ValueError(f"Adachi (1989): hv < E0 = {self.p.E0} eV, where alpha is only the E2-oscillator "
                             "tail (not interband absorption). Use CaseyAdachiAbsorption, or "
                             "Adachi1989GaAs(below_E0='model') to evaluate the tail deliberately.")
        _, k = self.nk(hv_a / EV)
        return _out(2 * k * (hv_a / HBAR) / C_LIGHT, hv)

    def reflectivity(self, hv):
        """Normal-incidence reflectivity R = ((n-1)^2 + k^2)/((n+1)^2 + k^2) (for QE normalization)."""
        n, k = self.nk(np.atleast_1d(np.asarray(hv, float)) / EV)
        return _out(((n - 1) ** 2 + k**2) / ((n + 1) ** 2 + k**2), hv)


# -------------------------------------------------------------------------- tabulated data --

class TabulatedAbsorption:
    """alpha(hv) from one table (e.g. measured data). Log-linear interpolation; raises outside
    the tabulated range."""

    def __init__(self, hv, alpha):
        hv, alpha = np.asarray(hv, float), np.asarray(alpha, float)
        order = np.argsort(hv)
        self.hv, self.alpha = hv[order], alpha[order]
        if np.any(self.alpha <= 0):
            raise ValueError("tabulated alpha must be positive")

    def absorption_coefficient(self, hv):
        hv_a = np.atleast_1d(np.asarray(hv, float))
        if np.any(hv_a < self.hv[0]) or np.any(hv_a > self.hv[-1]):
            raise ValueError("photon energy outside tabulated range")
        return _out(np.exp(np.interp(hv_a, self.hv, np.log(self.alpha))), hv)


class DopingTabulatedAbsorption:
    """alpha(hv; p) from measured spectra at several hole concentrations.

    datasets : list of dicts with keys "p" [m^-3], "hv" [J] (increasing), "alpha" [1/m], "label"
    p        : hole concentration of the sample [m^-3]
    out_of_range : "clamp_warn" (default) uses the nearest measured spectrum and warns;
                   "raise" refuses p outside [p_min, p_max].

        alpha(hv; p) = exp[(1 - w) ln a_lo(hv) + w ln a_hi(hv)],
        w = (log10 p - log10 p_lo) / (log10 p_hi - log10 p_lo),
    where a_lo and a_hi are the bracketing spectra, each interpolated log-linearly in hv. The
    valid hv range is the overlap of the spectra used; outside it, absorption_coefficient raises.
    """

    def __init__(self, datasets, p, out_of_range="clamp_warn", name="tabulated"):
        if out_of_range not in ("clamp_warn", "raise"):
            raise ValueError(out_of_range)
        ds = sorted(datasets, key=lambda d: d["p"])
        for d in ds:
            if np.any(np.diff(d["hv"]) <= 0) or np.any(np.asarray(d["alpha"]) <= 0):
                raise ValueError(f"dataset {d.get('label')}: hv must increase and alpha be positive")
        self.name = name
        self.p = float(p)
        ps = np.array([d["p"] for d in ds])
        self.p_range = (ps[0], ps[-1])
        if self.p < ps[0] or self.p > ps[-1]:
            msg = (f"{name}: p = {self.p * 1e-6:.3g} cm^-3 is outside the measured range "
                   f"[{ps[0] * 1e-6:.3g}, {ps[-1] * 1e-6:.3g}] cm^-3")
            if out_of_range == "raise":
                raise ValueError(msg)
            warnings.warn(msg + "; using the nearest measured spectrum", stacklevel=2)
            i = 0 if self.p < ps[0] else len(ds) - 1
            self.used, self.w = (ds[i], ds[i]), 0.0
        else:
            j = int(np.clip(np.searchsorted(ps, self.p), 1, len(ds) - 1))
            lo, hi = ds[j - 1], ds[j]
            self.used = (lo, hi)
            self.w = float((np.log10(self.p) - np.log10(lo["p"])) / (np.log10(hi["p"]) - np.log10(lo["p"])))
        self.hv_min = max(d["hv"][0] for d in self.used)
        self.hv_max = min(d["hv"][-1] for d in self.used)

    def absorption_coefficient(self, hv):
        hv_a = np.atleast_1d(np.asarray(hv, float))
        if np.any(hv_a < self.hv_min * (1 - 1e-12)) or np.any(hv_a > self.hv_max * (1 + 1e-12)):
            raise ValueError(f"{self.name}: hv outside the measured range "
                             f"[{self.hv_min / EV:.4f}, {self.hv_max / EV:.4f}] eV for p = {self.p * 1e-6:.3g} cm^-3")
        lo, hi = self.used
        la = [np.interp(hv_a, d["hv"], np.log(d["alpha"])) for d in (lo, hi)]
        return _out(np.exp((1 - self.w) * la[0] + self.w * la[1]), hv)


class CaseyAdachiAbsorption:
    """Default absorption: measured p-type near-edge data (Casey, Sell & Wecht 1975) joined to
    Adachi (1989) above the edge.

        hv <= b0          near-edge model (Casey, interpolated in hv and log10 p)
        b0 < hv < b1      ln alpha = (1 - s) ln alpha_Casey + s ln alpha_Adachi,
                          s = 3x^2 - 2x^3,  x = (hv - b0)/(b1 - b0)
        hv >= b1          Adachi (1989)
    Default window: b0 = 1.55 eV, b1 = min(1.65 eV, end of the Casey data = 1.592 eV), because the
    blend needs both curves. ``seam_ratio`` = alpha_Adachi/alpha_Casey at b0 and b1 measures how
    well the two sources agree there. The reflectivity comes from Adachi.
    """

    def __init__(self, p, near_edge=None, high=None, blend_eV=(1.55, 1.65), out_of_range="clamp_warn"):
        if near_edge is None:
            from .optical_data import casey1975_ptype
            near_edge = DopingTabulatedAbsorption(casey1975_ptype(), p, out_of_range=out_of_range,
                                                  name="Casey 1975 p-type")
        self.near = near_edge
        self.high = high if high is not None else Adachi1989GaAs(below_E0="raise")
        b0 = blend_eV[0] * EV
        b1 = min(blend_eV[1] * EV, self.near.hv_max)
        if not (self.near.hv_min < b0 < b1):
            raise ValueError("blend window must lie inside the near-edge data range")
        self.b0, self.b1 = b0, b1
        self.hv_min = self.near.hv_min
        self.seam_ratio = (self.high.absorption_coefficient(b0) / self.near.absorption_coefficient(b0),
                           self.high.absorption_coefficient(b1) / self.near.absorption_coefficient(b1))

    def absorption_coefficient(self, hv):
        hv_a = np.atleast_1d(np.asarray(hv, float))
        if np.any(hv_a < self.hv_min * (1 - 1e-12)):
            raise ValueError(f"no measured near-edge data below {self.hv_min / EV:.3f} eV for this doping "
                             "(there alpha is below the 10-13 cm^-1 floor of the Casey figures)")
        out = np.empty_like(hv_a)
        lo, hi = hv_a <= self.b0, hv_a >= self.b1
        mid = ~lo & ~hi
        if lo.any():
            out[lo] = self.near.absorption_coefficient(hv_a[lo])
        if hi.any():
            out[hi] = self.high.absorption_coefficient(hv_a[hi])
        if mid.any():
            x = (hv_a[mid] - self.b0) / (self.b1 - self.b0)
            s = x * x * (3 - 2 * x)
            out[mid] = np.exp((1 - s) * np.log(self.near.absorption_coefficient(hv_a[mid]))
                              + s * np.log(self.high.absorption_coefficient(hv_a[mid])))
        return _out(out, hv)

    def reflectivity(self, hv):
        return self.high.reflectivity(hv)


class ConstantAbsorptionLength:
    """Fixed absorption length l [m] (for tests and idealized studies)."""

    def __init__(self, length):
        self.length = float(length)

    def absorption_coefficient(self, hv):
        return _out(np.full(np.shape(np.atleast_1d(hv)), 1.0 / self.length), hv)


def as_absorption_model(obj, sample=None):
    """Accept a model object, a name, a scalar absorption length [m], or a callable hv -> l [m].

    Names: "casey1975+adachi1989" (default; needs the sample for p), "casey1975" (near-edge data
    only; needs the sample), "adachi1989".
    """
    if isinstance(obj, str):
        if obj == "adachi1989":
            return Adachi1989GaAs()
        if obj in ("casey1975+adachi1989", "casey1975"):
            if sample is None:
                raise ValueError(f"absorption model {obj!r} needs the sample (hole concentration)")
            if obj == "casey1975":
                from .optical_data import casey1975_ptype
                return DopingTabulatedAbsorption(casey1975_ptype(), sample.p, name="Casey 1975 p-type")
            return CaseyAdachiAbsorption(sample.p)
        raise ValueError(f"unknown absorption model {obj!r}")
    if hasattr(obj, "absorption_coefficient"):
        return obj
    if callable(obj):
        class _FromLength:
            def absorption_coefficient(self, hv):
                return 1.0 / obj(hv)
        return _FromLength()
    if np.isscalar(obj):
        return ConstantAbsorptionLength(obj)
    raise TypeError(f"cannot interpret {obj!r} as an absorption model")
