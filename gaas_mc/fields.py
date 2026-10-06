"""Electric-field models E_z(z) [V/m] along the surface normal.

Equation of motion, [C21] Eq. 18:  hbar dk/dt = -e E.  The material occupies z > 0, so a
positive E_z pushes electrons toward the surface (-z).

The conduction-band edge follows E_C(z) = -e V(z) + const, with E_z = -dV/dz, i.e.
dE_C/dz = e E_z (Eq. 60). The electron's tracked energy E is always kinetic energy above the
*local* band edge, so total energy E + E_C(z) is conserved during a flight. All valleys shift
rigidly with E_C(z).

Every field exposes
    Ez(z)          field [V/m]
    band_edge(z)   E_C(z) - E_C(bulk) [J]
    z_max          the field vanishes for z >= z_max (0 for no field, inf if it extends everywhere).
                   Transport uses direct W_total(E) flights at z >= z_max and null-collision flights
                   with velocity-Verlet substeps at z < z_max.
"""
from __future__ import annotations

import numpy as np

from .constants import Q_E


class NoField:
    is_zero = True
    z_max = 0.0

    def Ez(self, z):
        return np.zeros_like(np.asarray(z, float))

    def band_edge(self, z):
        return np.zeros_like(np.asarray(z, float))


class UniformField:
    is_zero = False
    z_max = np.inf

    def __init__(self, Ez):
        self.E0 = float(Ez)

    def Ez(self, z):
        return np.full_like(np.asarray(z, float), self.E0)

    def band_edge(self, z):
        return Q_E * self.E0 * np.asarray(z, float)


class C21BandBending:
    """Surface band bending of p-type GaAs, [C21] Sec. III C 1.

        E_bb = E_F^s - E_F^b,  E_F^s = Eg/2 (Fermi level pinned mid-gap)          Eqs. 56-57
        E_F^b from Eq. 58 (Nilsson), W_bb = sqrt(2 eps_s |E_bb| / (e p))            Eq. 59
        E_C(z) = E_Cb - E_bb (1 - z/W_bb)^2   for 0 < z < W_bb, E_Cb beyond          Eq. 61
        E_z(z) = (2 E_bb / (e W_bb)) (1 - z/W_bb)  for 0 < z < W_bb, 0 beyond          Eq. 62
    The bands bend DOWN toward the surface, so electrons are accelerated toward z = 0.
    E_bb and W_bb can be overridden (e.g. for a different surface Fermi-level pinning).
    """
    is_zero = False

    def __init__(self, sample, E_bb=None, W_bb=None):
        self.E_bb = float(sample.E_bb if E_bb is None else E_bb)
        self.W = float(sample.W_bb if W_bb is None else W_bb)
        self.z_max = self.W

    def Ez(self, z):
        z = np.asarray(z, float)
        x = np.clip(1.0 - z / self.W, 0.0, None)
        return 2 * self.E_bb / (Q_E * self.W) * x                         # Eq. 62

    def band_edge(self, z):
        z = np.asarray(z, float)
        x = np.clip(1.0 - z / self.W, 0.0, None)
        return -self.E_bb * x**2                                          # Eq. 61


class CallableField:
    """User-supplied E_z(z) [V/m]; band_edge only if band_edge_fn is given."""
    is_zero = False

    def __init__(self, Ez_fn, band_edge_fn=None, z_max=np.inf):
        self._f = Ez_fn
        self._u = band_edge_fn
        self.z_max = float(z_max)

    def Ez(self, z):
        return np.asarray(self._f(z), float)

    def band_edge(self, z):
        if self._u is None:
            raise NotImplementedError("band_edge_fn not supplied")
        return np.asarray(self._u(z), float)


class PotentialField:
    """From a user conduction-band-edge profile E_C(z) - E_C(bulk) [J] (callable):
    E_z = (1/e) dE_C/dz (Eq. 60), by central differences with step h [m]. Pass z_max if the
    profile is flat beyond some depth (enables direct flights there)."""
    is_zero = False

    def __init__(self, band_edge_fn, h=1e-11, z_max=np.inf):
        self._u = band_edge_fn
        self.h = h
        self.z_max = float(z_max)

    def Ez(self, z):
        z = np.asarray(z, float)
        return (self._u(z + self.h) - self._u(z - self.h)) / (2 * self.h) / Q_E

    def band_edge(self, z):
        return np.asarray(self._u(z), float)
