"""Optical absorption of GaAs: replaceable ``absorption_coefficient(hv)`` interface.

Every absorption model provides
    absorption_coefficient(hv) -> alpha [1/m]   (hv in J, scalar or array)
and photoexcitation uses l = 1/alpha in [C21] Eq. 7.

Default: Adachi's model dielectric function (MDF),
    S. Adachi, "Optical dispersion relations for GaP, GaAs, GaSb, InP, InAs, InSb, AlxGa1-xAs,
    and In1-xGaxAsyP1-y", J. Appl. Phys. 66, 6030 (1989).

The parameter values and implementation choices below follow the public-domain (CC0)
transcription by M. Polyanskiy (refractiveindex.info-scripts, "Adachi 1989 - GaAs.py").
That script notes three gaps in the paper, which are reproduced here:
  (i)   the E1 + Delta1 term is omitted (no B2, B21 given for it);
  (ii)  negative eps2 from the E1 term is clipped to zero;
  (iii) the phonon energy in the indirect-gap term is set to zero.
TODO(verify): check each parameter against Adachi (1989) Table I once the PDF is in refs/.

Limitations relevant to photoemission:
  * The E0 term has no broadening, exciton, or Urbach tail, so alpha = 0 for hv < E0 = 1.42 eV.
  * The model is for intrinsic GaAs. Doping-induced gap narrowing (C21 Eq. 10) is deliberately
    NOT applied to the optical data (docs/MODEL_ASSUMPTIONS.md). Photons with
    Eg(p) < hv < E0 therefore give alpha = 0, and photoexcite() refuses them.
  * C21 Fig. 3 uses Adachi's model *fitted* to Zollner (2001) data. Those fit parameters are not
    published, so this curve need not coincide with C21 Fig. 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import EV, HBAR

C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class Adachi1989Params:
    E0: float = 1.42          # eV
    E0_D0: float = 1.77       # E0 + Delta0, eV
    E1: float = 2.90          # eV
    E1_D1: float = 3.13       # E1 + Delta1, eV (unused, see note (i))
    E2: float = 4.7           # eV
    Eg_ind: float = 1.73      # indirect gap, eV
    A: float = 3.45           # eV^1.5
    B1: float = 6.37
    B11: float = 13.08        # eV^-0.5
    Gamma: float = 0.10       # eV, broadening of the E1 term
    C: float = 2.39
    gamma: float = 0.146
    D: float = 24.2
    eps_inf: float = 1.6


def _H(x):
    return (np.asarray(x) >= 0).astype(float)


class Adachi1989GaAs:
    """Adachi (1989) MDF for intrinsic GaAs at room temperature."""

    def __init__(self, params: Adachi1989Params = Adachi1989Params()):
        self.p = params

    def dielectric(self, hv_eV):
        """Complex dielectric function eps1 + i eps2 at photon energy hv [eV]."""
        p = self.p
        E = np.atleast_1d(np.asarray(hv_eV, float))
        # E0, E0 + Delta0 (3D M0 critical points)
        x0, xs = E / p.E0, E / p.E0_D0

        def f(x):
            return x**-2 * (2 - np.sqrt(1 + x) - np.sqrt(np.clip(1 - x, 0, None)))
        eps2_A = p.A / E**2 * (np.sqrt(np.clip(E - p.E0, 0, None))
                               + 0.5 * np.sqrt(np.clip(E - p.E0_D0, 0, None)))
        eps1_A = p.A * p.E0**-1.5 * (f(x0) + 0.5 * (p.E0 / p.E0_D0) ** 1.5 * f(xs))
        # E1 (2D M0); E1 + Delta1 omitted (note (i))
        x1 = E / p.E1
        eps2_B = np.pi * x1**-2 * (p.B1 - p.B11 * np.sqrt(np.clip(p.E1 - E, 0, None)))
        eps2_B = np.clip(eps2_B, 0, None)                                 # note (ii)
        x1c = (E + 1j * p.Gamma) / p.E1
        eps1_B = (-p.B1 * x1c**-2 * np.log(1 - x1c**2)).real
        # E2 (damped harmonic oscillator)
        x2 = E / p.E2
        den = (1 - x2**2) ** 2 + (x2 * p.gamma) ** 2
        eps2_C = p.C * x2 * p.gamma / den
        eps1_C = p.C * (1 - x2**2) / den
        # indirect gap, below E1 only; phonon energy neglected (note (iii))
        eps2_D = p.D / E**2 * np.clip(E - p.Eg_ind, 0, None) ** 2 * _H(p.E1 - E)
        eps1 = p.eps_inf + eps1_A + eps1_B + eps1_C
        eps2 = eps2_A + eps2_B + eps2_C + eps2_D
        return eps1 + 1j * eps2

    def nk(self, hv_eV):
        n_c = np.sqrt(self.dielectric(hv_eV))
        return n_c.real, n_c.imag

    def absorption_coefficient(self, hv):
        """alpha [1/m] at photon energy hv [J]: alpha = 2 k omega / c."""
        hv = np.asarray(hv, float)
        _, k = self.nk(hv / EV)
        out = 2 * k * (hv / HBAR) / C_LIGHT
        return out if out.size > 1 or hv.ndim else float(out[0])

    def reflectivity(self, hv):
        """Normal-incidence reflectivity R = ((n-1)^2 + k^2)/((n+1)^2 + k^2) (for QE normalization)."""
        n, k = self.nk(np.asarray(hv, float) / EV)
        R = ((n - 1) ** 2 + k**2) / ((n + 1) ** 2 + k**2)
        return R if R.size > 1 else float(R[0])


class TabulatedAbsorption:
    """alpha(hv) from a table (e.g. measured data). Log-linear interpolation; raises outside
    the tabulated range."""

    def __init__(self, hv, alpha):
        hv, alpha = np.asarray(hv, float), np.asarray(alpha, float)
        order = np.argsort(hv)
        self.hv, self.alpha = hv[order], alpha[order]
        if np.any(self.alpha <= 0):
            raise ValueError("tabulated alpha must be positive")

    def absorption_coefficient(self, hv):
        hv = np.asarray(hv, float)
        if np.any(hv < self.hv[0]) or np.any(hv > self.hv[-1]):
            raise ValueError("photon energy outside tabulated range")
        out = np.exp(np.interp(hv, self.hv, np.log(self.alpha)))
        return out if hv.ndim else float(out)


class ConstantAbsorptionLength:
    """Fixed absorption length l [m] (for tests and idealized studies)."""

    def __init__(self, length):
        self.length = float(length)

    def absorption_coefficient(self, hv):
        return np.full_like(np.asarray(hv, float), 1.0 / self.length) if np.ndim(hv) else 1.0 / self.length


def as_absorption_model(obj):
    """Accept a model object, a name, a scalar absorption length [m], or a callable hv -> l [m]."""
    if obj is None or obj == "adachi1989":
        return Adachi1989GaAs()
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
