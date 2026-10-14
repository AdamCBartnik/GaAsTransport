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
                   For a fully depleted finite layer z_max is its mandatory back boundary;
                   the GaAs-side field need not vanish there. No transport beyond it is modeled.
"""
from __future__ import annotations

import numpy as np

from .backend import asarray, xp_of
from .constants import Q_E


class NoField:
    is_zero = True
    z_max = 0.0

    def Ez(self, z):
        z = asarray(z, float)
        return xp_of(z).zeros_like(z)

    def band_edge(self, z):
        z = asarray(z, float)
        return xp_of(z).zeros_like(z)


class UniformField:
    is_zero = False
    z_max = np.inf

    def __init__(self, Ez):
        self.E0 = float(Ez)

    def Ez(self, z):
        z = asarray(z, float)
        return xp_of(z).full_like(z, self.E0)

    def band_edge(self, z):
        return Q_E * self.E0 * asarray(z, float)


class C21BandBending:
    """Surface band bending of p-type GaAs, [C21] Sec. III C 1.

        E_bb = E_F^s - E_F^b,  E_F^s = Eg/2 (Fermi level pinned mid-gap)          Eqs. 56-57
        E_F^b from Eq. 58 (Nilsson), W_bb = sqrt(2 eps_s |E_bb| / (e p))            Eq. 59
        E_C(z) = E_Cb - E_bb (1 - z/W_bb)^2   for 0 < z < W_bb, E_Cb beyond          Eq. 61
        E_z(z) = (2 E_bb / (e W_bb)) (1 - z/W_bb)  for 0 < z < W_bb, 0 beyond          Eq. 62
    The bands bend DOWN toward the surface, so electrons are accelerated toward z = 0.
    E_bb and W_bb can be overridden (e.g. for a different surface Fermi-level pinning).

    thickness [m] fixes the rear band edge to neutral bulk (an ideal reservoir).
    For d < W the fully depleted slab has U(z) = -E_bb*(1-z/d)
    + E_bb*z*(d-z)/W**2. For d >= W the bulk profile is unchanged.
    This is a depletion approximation, not a self-consistent mobile-carrier solve.
    The thin-slab expressions extend smoothly outside the slab for flight integration;
    transport must supply a back boundary at d. Ez(d) is the GaAs-side field.
    """
    is_zero = False

    def __init__(self, sample, E_bb=None, W_bb=None, *, thickness=None):
        self.E_bb = float(sample.E_bb if E_bb is None else E_bb)
        self.W = float(sample.W_bb if W_bb is None else W_bb)
        self.thickness = None if thickness is None else float(thickness)
        if self.thickness is not None and (not np.isfinite(self.thickness) or self.thickness <= 0):
            raise ValueError("thickness must be finite and positive")
        self.fully_depleted = self.thickness is not None and self.thickness < self.W
        self.z_max = self.thickness if self.fully_depleted else self.W

    def Ez(self, z):
        z = asarray(z, float)
        if self.fully_depleted:
            d = self.thickness
            return self.E_bb / Q_E * (1 / d + (d - 2 * z) / self.W**2)
        x = np.clip(1.0 - z / self.W, 0.0, None)
        return 2 * self.E_bb / (Q_E * self.W) * x                         # Eq. 62

    def band_edge(self, z):
        z = asarray(z, float)
        if self.fully_depleted:
            d = self.thickness
            return -self.E_bb * (1 - z / d) + self.E_bb * z * (d - z) / self.W**2
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
