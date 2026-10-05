"""Ionized-impurity scattering, Brooks-Herring (screened Coulomb, Born approximation).

[C21] Eq. 35:
    W_ii(k) = N_a e^4 Z^2 (2m*)^(3/2) / (2 pi hbar^4 eps_s^2 beta^4)
              * sqrt(gamma_k) (1 + 2 alpha E_k) / (1 + 4 gamma_k / E_beta),    E_beta = hbar^2 beta^2/(2m*)
[C21] Eq. 36 (final angle, elastic):
    cos(theta) = 1 - 2r / (1 + 4 gamma_k (1 - r) / E_beta)
[C21] Eq. 37 (momentum relaxation):
    1/tau_m = N_a e^4 Z^2 / (16 pi eps_s^2 sqrt(2 m*)) (1 + 2 alpha E)/gamma^(3/2)
              * [ ln(1 + 4 gamma/E_beta) - (4 gamma/E_beta)/(1 + 4 gamma/E_beta) ]
N_a^- = p (full ionization, C21 Sec. III B); Z = 1; beta from Sample (Eqs. 28-29).
All three follow from |M_q|^2 = (Z e^2 / (eps_s Omega))^2 / (q^2 + beta^2)^2 (verified in tests).
"""
from __future__ import annotations

import numpy as np

from ..constants import HBAR, Q_E
from .base import Mechanism, new_k_from_angle


class IonizedImpurity(Mechanism):
    spin_class = "ii"

    def __init__(self, sample, valley_from=0, Z=1.0, N_imp=None):
        super().__init__(sample, valley_from)
        self.name = f"impurity[{self.valley.name}]"
        self.N = sample.p if N_imp is None else N_imp
        self.Z = Z
        self.beta = sample.beta
        self.E_beta = HBAR**2 * self.beta**2 / (2 * self.m)                  # Eq. 32
        es = sample.eps_s
        self._Cw = self.N * Q_E**4 * Z**2 * (2 * self.m) ** 1.5 / (2 * np.pi * HBAR**4 * es**2 * self.beta**4)
        self._Cm = self.N * Q_E**4 * Z**2 / (16 * np.pi * es**2 * np.sqrt(2 * self.m))

    def rate(self, E):
        E = np.asarray(E, dtype=float)
        g = self.gamma(E)
        return self._Cw * np.sqrt(g) * (1 + 2 * self.alpha * E) / (1 + 4 * g / self.E_beta)   # Eq. 35

    def momentum_rate(self, E):
        E = np.atleast_1d(np.asarray(E, dtype=float))
        g = self.gamma(E)
        x = 4 * g / self.E_beta
        out = np.zeros_like(E)
        ok = g > 0
        # ln(1+x) - x/(1+x): use log1p; for tiny x the bracket ~ x^2/2
        br = np.where(x[ok] > 1e-4, np.log1p(x[ok]) - x[ok] / (1 + x[ok]), 0.5 * x[ok] ** 2 - 2 / 3 * x[ok] ** 3)
        out[ok] = self._Cm * (1 + 2 * self.alpha * E[ok]) / g[ok] ** 1.5 * br              # Eq. 37
        return out

    def scatter(self, k, E, rng):
        E = np.asarray(E, dtype=float)
        r = rng.random(E.size)
        cos_t = 1 - 2 * r / (1 + 4 * self.gamma(E) * (1 - r) / self.E_beta)              # Eq. 36
        k_new = new_k_from_angle(k, E, cos_t, rng, self.m, self.alpha)
        return k_new, np.ones(E.size, bool), np.full(E.size, self.valley_from)
