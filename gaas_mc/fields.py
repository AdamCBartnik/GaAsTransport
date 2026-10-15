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

    thickness [m] makes a finite layer 0 < z < d; rear sets the electrostatic condition at z = d.
    With U = E_C(z) - E_C(bulk), B = E_bb and W = W_bb:
      rear="bulk": the rear band edge is fixed to neutral bulk (an ideal reservoir).
        d < W:  U(z) = -B(1-z/d) + B z(d-z)/W**2 (fully depleted, nonzero rear field);
        d >= W: the bulk C21 profile, unchanged.
      rear="vacuum": the rear surface has the same mid-gap pinning as the front, U(0) = U(d) = -B.
        d < 2W:  U(z) = -B + B z(d-z)/W**2 (fully depleted, symmetric about d/2);
        d >= 2W: U(z) = -B[(1-z/W)_+**2 + (1-(d-z)/W)_+**2] (a C21 profile at each surface).
        The field pushes electrons in z > d/2 toward the rear; z_max = d.
    This is a depletion approximation, not a self-consistent mobile-carrier solve.
    The finite-layer expressions extend smoothly outside the slab for flight integration;
    transport must supply a back boundary at d. Ez(d) is the GaAs-side field.
    """
    REARS = ("bulk", "vacuum")
    is_zero = False

    def __init__(self, sample, E_bb=None, W_bb=None, *, thickness=None, rear="bulk"):
        self.E_bb = float(sample.E_bb if E_bb is None else E_bb)
        self.W = float(sample.W_bb if W_bb is None else W_bb)
        self.thickness = None if thickness is None else float(thickness)
        if self.thickness is not None and (not np.isfinite(self.thickness) or self.thickness <= 0):
            raise ValueError("thickness must be finite and positive")
        if rear not in self.REARS:
            raise ValueError(f"rear {rear!r}: use one of {self.REARS}")
        if rear == "vacuum" and self.thickness is None:
            raise ValueError('rear="vacuum" needs a finite thickness')
        self.rear = rear
        if rear == "vacuum":
            self.fully_depleted = self.thickness < 2 * self.W
            self.z_max = self.thickness
        else:
            self.fully_depleted = self.thickness is not None and self.thickness < self.W
            self.z_max = self.thickness if self.fully_depleted else self.W

    def Ez(self, z):
        z = asarray(z, float)
        B, W, d = self.E_bb, self.W, self.thickness
        if self.rear == "vacuum":
            if self.fully_depleted:
                return B / Q_E * (d - 2 * z) / W**2
            return 2 * B / (Q_E * W) * (np.clip(1.0 - z / W, 0.0, None) - np.clip(1.0 - (d - z) / W, 0.0, None))
        if self.fully_depleted:
            return B / Q_E * (1 / d + (d - 2 * z) / W**2)
        x = np.clip(1.0 - z / W, 0.0, None)
        return 2 * B / (Q_E * W) * x                                      # Eq. 62

    def band_edge(self, z):
        z = asarray(z, float)
        B, W, d = self.E_bb, self.W, self.thickness
        if self.rear == "vacuum":
            if self.fully_depleted:
                return -B + B * z * (d - z) / W**2
            return -B * (np.clip(1.0 - z / W, 0.0, None)**2 + np.clip(1.0 - (d - z) / W, 0.0, None)**2)
        if self.fully_depleted:
            return -B * (1 - z / d) + B * z * (d - z) / W**2
        x = np.clip(1.0 - z / W, 0.0, None)
        return -B * x**2                                                  # Eq. 61

    @property
    def E_bb_eff(self):
        """Band-bending height actually present [J]: highest band edge in the layer minus the surface
        band edge, max_z U(z) - U(0). Equals E_bb except for rear="vacuum" with d < 2W, where the
        band edge peaks below bulk at z = d/2 and E_bb_eff = E_bb d**2/(4 W**2). The effective
        affinity relative to that peak is chi - E_bb_eff."""
        if self.rear == "vacuum" and self.fully_depleted:
            return self.E_bb * self.thickness**2 / (4 * self.W**2)
        return self.E_bb


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
