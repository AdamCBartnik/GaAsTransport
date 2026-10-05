"""Common interface for momentum-scattering mechanisms.

Each mechanism is a small object bound to a ``Sample`` (material + doping + T) and to an
initial valley. It exposes:

    rate(E)            total scattering rate W(E) [1/s] for kinetic energy E [J] (vectorized)
    momentum_rate(E)   momentum relaxation rate 1/tau_m(E) [1/s], Eq. 22 of [C21]
    scatter(k, E, rng) -> (k_new, accepted, valley_new)
                       k: (n, 3) wavevectors, E: (n,) energies. `accepted` is False where a
                       rejection technique (e.g. e-h, Eq. 40) turns the event into a
                       self-scattering; those rows of k_new are unchanged.

Attributes:
    name         unique label used in tables / histories
    threshold    minimum energy for the event. The transport inserts it into the rate grid and
                 forces the interpolated rate to 0 at E <= threshold (interpolating across a
                 threshold would otherwise allow forbidden events).
    valley_from  initial valley index (material.GAMMA, L, X)
    spin_class   "ap", "pop", "ij", "ii" -> contributes to EY/DP via its tau_m (Eqs. 45, 46, 52);
                 None -> excluded from the Matthiessen sum of Eq. 52 (e.g. electron-hole)

To disable a mechanism, leave it out of the list passed to ``transport.Simulation``.
"""
from __future__ import annotations

import numpy as np

from .. import bands


class Mechanism:
    name: str = "base"
    spin_class: str | None = None
    threshold: float = 0.0     # the event is possible only for E > threshold [J] (e.g. hw0 for emission)

    def __init__(self, sample, valley_from=0):
        self.sample = sample
        self.material = sample.material
        self.valley_from = valley_from
        self.valley = self.material.valleys[valley_from]

    # ---- to be provided by subclasses --------------------------------------------------
    def rate(self, E):
        raise NotImplementedError

    def momentum_rate(self, E):
        raise NotImplementedError

    def scatter(self, k, E, rng):
        raise NotImplementedError

    # ---- helpers -----------------------------------------------------------------------
    @property
    def m(self):
        return self.valley.m_eff

    @property
    def alpha(self):
        return self.valley.alpha

    def gamma(self, E):
        return bands.gamma_of_E(E, self.alpha)

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"


def bose(hw, kT):
    """Bose-Einstein occupation, Eqs. 26 and 34."""
    return 1.0 / np.expm1(hw / kT)


def new_k_from_angle(k, E_new, cos_t, rng, m, alpha):
    """Final wavevector with magnitude k(E_new) at polar angle theta (cos_t) from the initial
    direction, azimuth uniform in [0, 2 pi)."""
    u = bands.unit(k, rng)
    phi = 2 * np.pi * rng.random(len(cos_t))
    return bands.rotate_about(u, cos_t, phi) * bands.k_of_E(E_new, m, alpha)[:, None]


def new_k_isotropic(E_new, rng, m, alpha):
    """Final wavevector with magnitude k(E_new) in a uniformly random direction (Eq. 24)."""
    return bands.random_unit_vectors(len(E_new), rng) * bands.k_of_E(E_new, m, alpha)[:, None]
