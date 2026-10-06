"""Nonparabolic (Kane) conduction-band dispersion and kinematics helpers.

[C21] Eq. 1:   gamma(E) = E (1 + alpha E) = hbar^2 k^2 / (2 m*)
[C21] Eq. 2:   E(k) = [sqrt(1 + 4 alpha hbar^2 k^2 / (2 m*)) - 1] / (2 alpha)
[C21] Eq. 19:  v = hbar k / (m* (1 + 2 alpha E))

Energies are kinetic energies measured from the bottom of the valley, in J.
All functions are vectorized over numpy or cupy arrays (gaas_mc/backend.py). alpha = 0 gives the
parabolic limit.
"""
from __future__ import annotations

import numpy as np

from .backend import asarray, xp_of
from .constants import HBAR


def gamma_of_E(E, alpha):
    """Eq. 1: gamma = E (1 + alpha E)."""
    return E * (1.0 + alpha * E)


def E_of_gamma(g, alpha):
    """Inverse of Eq. 1 (Eq. 2 written in terms of gamma). Stable form for small alpha*gamma:
    E = 2 gamma / (1 + sqrt(1 + 4 alpha gamma))."""
    return 2.0 * g / (1.0 + np.sqrt(1.0 + 4.0 * alpha * g))


def E_of_k(k, m, alpha):
    """Eq. 2, k = |k| [1/m]."""
    return E_of_gamma(HBAR**2 * asarray(k) ** 2 / (2.0 * m), alpha)


def k_of_E(E, m, alpha):
    """|k| from Eq. 1: k = sqrt(2 m gamma(E)) / hbar."""
    return np.sqrt(2.0 * m * gamma_of_E(asarray(E), alpha)) / HBAR


def speed_of_E(E, m, alpha):
    """Eq. 19 magnitude: v = hbar k / (m (1 + 2 alpha E))."""
    E = asarray(E)
    return HBAR * k_of_E(E, m, alpha) / (m * (1.0 + 2.0 * alpha * E))


def velocity(kvec, m, alpha):
    """Eq. 19 vector form, kvec shape (..., 3) -> v shape (..., 3)."""
    kvec = asarray(kvec)
    k = np.linalg.norm(kvec, axis=-1)
    E = E_of_k(k, m, alpha)
    return HBAR * kvec / (m * (1.0 + 2.0 * alpha * E))[..., None]


def dos(E, m, alpha, spin_degeneracy=2):
    """Density of states per unit volume per unit energy [1/(J m^3)] for one valley:

    g(E) = s/(4 pi^2) (2m/hbar^2)^(3/2) sqrt(gamma) (1 + 2 alpha E),   s = spin degeneracy.

    Follows from g = s k^2 / (2 pi^2) dk/dE with Eq. 1. The scattering-rate formulas of [C21]
    (Eqs. 23, 25, 33, 35) all contain the factor sqrt(gamma)(1 + 2 alpha E) of the final state.
    """
    E = asarray(E)
    return spin_degeneracy / (4 * np.pi**2) * (2 * m / HBAR**2) ** 1.5 \
        * np.sqrt(gamma_of_E(E, alpha)) * (1 + 2 * alpha * E)


# --------------------------------------------------------------------------------------
# Direction helpers
# --------------------------------------------------------------------------------------

def random_unit_vectors(n, rng):
    """n isotropic unit vectors, shape (n, 3)."""
    cos_t = 1.0 - 2.0 * rng.random(n)
    phi = 2 * np.pi * rng.random(n)
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, None))
    return xp_of(cos_t).column_stack((sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t))


def rotate_about(u, cos_t, phi):
    """Return unit vectors at polar angle theta (cos_t) and azimuth phi relative to the unit
    vectors u (shape (n, 3)). The azimuth is measured in an arbitrary orthonormal frame
    perpendicular to u, which is fine because phi is always uniform in [0, 2 pi) here."""
    u = asarray(u, dtype=float)
    # pick a helper axis not parallel to u
    helper = np.zeros_like(u)
    use_x = np.abs(u[:, 0]) < 0.9
    helper[use_x, 0] = 1.0
    helper[~use_x, 1] = 1.0
    e1 = np.cross(u, helper)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(u, e1)
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, None))
    return (cos_t[:, None] * u
            + (sin_t * np.cos(phi))[:, None] * e1
            + (sin_t * np.sin(phi))[:, None] * e2)


def unit(kvec, rng=None):
    """Normalize rows of kvec. Rows with |k| = 0 get a random direction (needs rng)."""
    kvec = asarray(kvec, dtype=float)
    k = np.linalg.norm(kvec, axis=1)
    out = xp_of(kvec).empty_like(kvec)
    ok = k > 0
    out[ok] = kvec[ok] / k[ok, None]
    if np.any(~ok):
        if rng is None:
            raise ValueError("zero wavevector and no rng given")
        out[~ok] = random_unit_vectors(int((~ok).sum()), rng)
    return out
