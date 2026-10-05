"""Electric-field models E_z(z) [V/m] along the surface normal.

Equation of motion, [C21] Eq. 18:  hbar dk/dt = -e E.  The material occupies z > 0, so a
positive E_z pushes electrons toward the surface (-z).

The conduction-band edge follows E_C(z) = -e V(z) + const, with E_z = -dV/dz, i.e.
dE_C/dz = e E_z (Eq. 60). The electron's tracked energy E is always kinetic energy above
the *local* band edge, so total energy E + E_C(z) is conserved during a flight.

Stage A uses NoField. The [C21] band-bending profile (Eqs. 56-62) will be added in Stage D;
any user potential can already be passed as CallableField / PotentialField.
"""
from __future__ import annotations

import numpy as np

from .constants import Q_E


class NoField:
    is_zero = True

    def Ez(self, z):
        return np.zeros_like(z)

    def band_edge(self, z):
        """E_C(z) - E_C(bulk) [J]"""
        return np.zeros_like(z)


class UniformField:
    is_zero = False

    def __init__(self, Ez):
        self.E0 = float(Ez)

    def Ez(self, z):
        return np.full_like(z, self.E0)

    def band_edge(self, z):
        return Q_E * self.E0 * z


class CallableField:
    """User-supplied E_z(z) [V/m]; band_edge only if band_edge_fn is given."""
    is_zero = False

    def __init__(self, Ez_fn, band_edge_fn=None):
        self._f = Ez_fn
        self._u = band_edge_fn

    def Ez(self, z):
        return np.asarray(self._f(z), float)

    def band_edge(self, z):
        if self._u is None:
            raise NotImplementedError("band_edge_fn not supplied")
        return np.asarray(self._u(z), float)


class PotentialField:
    """From a user conduction-band-edge profile E_C(z) [J] (callable): E_z = (1/e) dE_C/dz (Eq. 60),
    evaluated by central differences with step h [m]."""
    is_zero = False

    def __init__(self, band_edge_fn, h=1e-11):
        self._u = band_edge_fn
        self.h = h

    def Ez(self, z):
        z = np.asarray(z, float)
        return (self._u(z + self.h) - self._u(z - self.h)) / (2 * self.h) / Q_E

    def band_edge(self, z):
        return np.asarray(self._u(z), float)
