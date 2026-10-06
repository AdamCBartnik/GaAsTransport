"""Electron ensemble stored as a structure of arrays (ready for vectorization or numba later).

Geometry: 1D real space z (the surface is z = 0; the material is z > 0), 3D k-space.
E is the kinetic energy measured from the bottom of the electron's current valley [J] and is
kept consistent with |k| through Eq. 2 by every routine that changes k.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

ALIVE, SURFACE, TIMEOUT, BACK, EMITTED, TRAPPED = 0, 1, 2, 3, 4, 5
STATUS_NAMES = {ALIVE: "alive", SURFACE: "surface", TIMEOUT: "timeout", BACK: "back",
                EMITTED: "emitted", TRAPPED: "surface_trapped"}


@dataclass
class Ensemble:
    z: np.ndarray          # position [m]
    t: np.ndarray          # time [s]
    k: np.ndarray          # wavevector (n, 3) [1/m]
    E: np.ndarray          # kinetic energy above the valley minimum [J]
    valley: np.ndarray     # int8: 0 Gamma, 1 L, 2 X
    spin: np.ndarray       # int8: +1 / -1 along +z (light propagation)
    status: np.ndarray     # int8: ALIVE / SURFACE / TIMEOUT / BACK
    z0: np.ndarray         # initial depth [m]
    E0: np.ndarray         # initial energy [J]
    spin0: np.ndarray      # initial spin
    band: np.ndarray       # int8: excitation band (0 hh, 1 lh, 2 so, -1 unknown)
    dt_spin: np.ndarray    # time since last real (non-self) scattering [s] (Eq. 54)
    n_flips: np.ndarray    # number of spin flips
    pid: np.ndarray        # particle id
    time_in_valley: np.ndarray   # (n, 3) time spent in Gamma, L, X [s]
    visited: np.ndarray          # (n, 3) bool, valley ever occupied
    eqv: np.ndarray              # int8: which equivalent valley (L: 0-3, X: 0-2, Gamma: 0), see valleys.py
    n_surface: np.ndarray        # number of surface encounters

    @classmethod
    def create(cls, z, k, E, spin, valley=0, t=0.0):
        z = np.asarray(z, float).copy()
        n = z.size
        k = np.asarray(k, float).reshape(n, 3).copy()
        E = np.asarray(E, float).copy()
        spin = np.broadcast_to(np.asarray(spin, np.int8), (n,)).copy()
        return cls(
            z=z, t=np.full(n, float(t)), k=k, E=E,
            valley=np.full(n, valley, np.int8), spin=spin,
            status=np.full(n, ALIVE, np.int8), z0=z.copy(), E0=E.copy(), spin0=spin.copy(),
            band=np.full(n, -1, np.int8), dt_spin=np.zeros(n), n_flips=np.zeros(n, np.int32),
            pid=np.arange(n), time_in_valley=np.zeros((n, 3)),
            visited=np.eye(3, dtype=bool)[np.full(n, valley)], eqv=np.zeros(n, np.int8),
            n_surface=np.zeros(n, np.int32),
        )

    @classmethod
    def from_arrivals(cls, arr):
        """Ensemble positioned at z = 0 in the state of each surface arrival (to continue those
        electrons, e.g. with different surface models from a common bulk history)."""
        n = len(arr)
        return cls(z=np.zeros(n), t=arr.t.copy(), k=arr.k.copy(), E=arr.E.copy(), valley=arr.valley.copy(),
                   spin=arr.spin.copy(), status=np.full(n, ALIVE, np.int8), z0=arr.z0.copy(),
                   E0=arr.E0.copy(), spin0=arr.spin0.copy(), band=arr.band.copy(), dt_spin=arr.dt_spin.copy(),
                   n_flips=arr.n_flips.copy(), pid=arr.pid.copy(), time_in_valley=arr.time_in_valley.copy(),
                   visited=arr.visited.copy(), eqv=arr.eqv.copy(), n_surface=np.zeros(n, np.int32))

    def __len__(self):
        return self.z.size

    def copy(self):
        return Ensemble(**{f.name: getattr(self, f.name).copy() for f in fields(self)})

    @classmethod
    def concatenate(cls, parts):
        return cls(**{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(cls)})

    def subset(self, idx):
        return Ensemble(**{f.name: getattr(self, f.name)[idx].copy() for f in fields(self)})

    def esp(self, mask=None):
        s = self.spin if mask is None else self.spin[mask]
        return float(s.mean()) if s.size else np.nan
