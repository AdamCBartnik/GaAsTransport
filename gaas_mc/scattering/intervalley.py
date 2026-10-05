"""Intervalley (non-polar optical phonon) scattering i -> j, absorption or emission.

[C21] Eq. 33 (as printed):
    W_ij(k) = (m_j)^(3/2) D_ij^2 Z_j sqrt(gamma_j(E')) / (sqrt(2) pi rho hbar^2 (hbar w_ij) (1 + 2 alpha_j E'))
              * (N + 1/2 -+ 1/2)
    E' = E +- hbar w_ij - Delta_ji,   Delta_ji = (minimum of j) - (minimum of i)
    N = Bose(hbar w_ij), Eq. 34. The process is isotropic, so 1/tau_m = W.

Golden rule with |M|^2 = hbar D^2 / (2 rho Omega w) (N + 1/2 -+ 1/2) per destination valley gives
    W = pi D^2 Z / (rho w) * g_1spin(E') = m_j^(3/2) D^2 Z sqrt(gamma') (1 + 2 alpha_j E') / (sqrt(2) pi rho hbar^2 hbar w),
with the DOS factor (1 + 2 alpha_j E') in the NUMERATOR. C21 prints it in the denominator,
presumably a typo. ModelAssumptions.intervalley_dos_factor selects "numerator" (default) or
"c21_as_printed".

Multiplicity Z (user decision 5): number of distinct equivalent destination valleys reachable
from one initial valley (Gamma->L 4, Gamma->X 3, L->L 3, L->X 3, L->Gamma 1, X->X 2, X->L 4,
X->Gamma 1). Eq. 33 contains Z_j once, and the DOS factor is that of a single valley, so Z is
applied exactly once.
"""
from __future__ import annotations

import numpy as np

from .. import bands
from ..constants import HBAR
from .base import Mechanism, bose, new_k_isotropic


def _sym_lookup(d, a, b):
    if (a, b) in d:
        return d[(a, b)]
    if (b, a) in d:
        return d[(b, a)]
    raise KeyError((a, b))


class Intervalley(Mechanism):
    spin_class = "ij"

    def __init__(self, sample, valley_from, valley_to, emission, Z, dos_factor="numerator"):
        super().__init__(sample, valley_from)
        mat = self.material
        self.valley_to = valley_to
        self.vj = mat.valleys[valley_to]
        self.emission = emission
        self.Z = Z
        if dos_factor not in ("numerator", "c21_as_printed"):
            raise ValueError(dos_factor)
        self.dos_factor = dos_factor
        a, b = self.valley.name, self.vj.name
        self.D = _sym_lookup(mat.D_iv, a, b)
        self.hw = _sym_lookup(mat.hw_iv, a, b)
        self.Delta = self.vj.offset - self.valley.offset                       # Delta_ji
        self.occupation = bose(self.hw, sample.kT) + (1.0 if emission else 0.0)
        self.threshold = max(0.0, self.Delta + (self.hw if emission else -self.hw))
        self.name = f"iv_{'em' if emission else 'abs'}[{a}->{b}]"
        self._C = (self.vj.m_eff**1.5 * self.D**2 * Z * self.occupation
                   / (np.sqrt(2) * np.pi * mat.rho * HBAR**2 * self.hw))

    def final_energy(self, E):
        return E + (-self.hw if self.emission else self.hw) - self.Delta

    def rate(self, E):
        E = np.atleast_1d(np.asarray(E, float))
        Ep = self.final_energy(E)
        W = np.zeros_like(E)
        ok = Ep > 0
        aj = self.vj.alpha
        gp = bands.gamma_of_E(Ep[ok], aj)
        dosf = (1 + 2 * aj * Ep[ok])
        W[ok] = self._C * np.sqrt(gp) * (dosf if self.dos_factor == "numerator" else 1.0 / dosf)
        return W

    def momentum_rate(self, E):
        return self.rate(E)

    def scatter(self, k, E, rng):
        Ep = self.final_energy(np.asarray(E, float))
        if np.any(Ep <= 0):
            raise RuntimeError(f"{self.name}: forbidden transition selected")
        k_new = new_k_isotropic(Ep, rng, self.vj.m_eff, self.vj.alpha)
        return k_new, np.ones(Ep.size, bool), np.full(Ep.size, self.valley_to)
