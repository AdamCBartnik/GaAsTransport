"""Intravalley acoustic-phonon scattering (deformation potential, elastic, equipartition).

[C21] Eq. 23:
    W_ap(k) = sqrt(2) m*^(3/2) Xi_d^2 kB T sqrt(gamma_k) (1 + 2 alpha E_k) / (c_l pi hbar^4),
    c_l = rho v_s^2.
The rate lumps absorption and emission; the phonon energy is neglected (elastic).
[C21] Eq. 24: isotropic final direction, cos(theta) = 1 - 2r.
Isotropic, so 1/tau_m = W (text after Eq. 24).
"""
from __future__ import annotations

import numpy as np

from ..constants import HBAR
from .base import Mechanism, new_k_isotropic


class AcousticPhonon(Mechanism):
    spin_class = "ap"

    def __init__(self, sample, valley_from=0):
        super().__init__(sample, valley_from)
        self.name = f"acoustic[{self.valley.name}]"
        mat = self.material
        # constant prefactor of Eq. 23 (everything except sqrt(gamma)(1+2 alpha E))
        self._C = (np.sqrt(2) * self.m**1.5 * self.valley.Xi_d**2 * sample.kT
                   / (mat.c_l * np.pi * HBAR**4))

    def rate(self, E):
        E = np.asarray(E, dtype=float)
        return self._C * np.sqrt(self.gamma(E)) * (1 + 2 * self.alpha * E)

    def momentum_rate(self, E):
        return self.rate(E)

    def scatter(self, k, E, rng):
        k_new = new_k_isotropic(E, rng, self.m, self.alpha)
        return k_new, np.ones(len(E), bool), np.full(len(E), self.valley_from)
