"""Surface-arrival records.

When an electron coming from z > 0 reaches z = 0, its bulk trajectory stops and its full
state is stored here. No emission physics is applied: electron affinity, mass discontinuity,
image charge, and transmission belong to a separate surface model that consumes these records.

Valley identity is preserved: ``valley`` (0 Gamma, 1 L, 2 X), ``eqv`` (which of the 4 L or 3 X
valleys), ``k`` (wavevector measured from that valley's minimum), and ``K`` (full crystal wavevector
= valley center + k; lab z = [001], the surface normal). No valley is mapped onto a scalar-mass Gamma
model here; that belongs to the valley-aware surface stage.

Energies are kinetic energies above the local conduction-band minimum of the electron's
valley at z = 0. ``band_edge_at_surface`` = E_C(0) - E_C(bulk) (negative with downward band bending)
places them on an absolute scale: E_total - E_C(bulk) = E + band_edge_at_surface + valley offset. Spin s = +1/-1 along +z, the light propagation direction (into the material),
so s = +1 points *away* from the vacuum.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .constants import HBAR


@dataclass
class SurfaceArrivals:
    t: np.ndarray          # arrival time [s]
    E: np.ndarray          # kinetic energy above the local valley minimum [J]
    k: np.ndarray          # (n, 3) wavevector [1/m]; k_z < 0 (moving toward the surface)
    valley: np.ndarray     # int8
    spin: np.ndarray       # int8 at arrival
    spin0: np.ndarray      # int8 at excitation
    z0: np.ndarray         # initial depth [m]
    E0: np.ndarray         # initial energy [J]
    band: np.ndarray       # excitation band (0 hh, 1 lh, 2 so, -1 n/a)
    n_flips: np.ndarray    # number of spin flips along the path
    n_events: np.ndarray   # (n, M) number of real events of each mechanism
    pid: np.ndarray
    time_in_valley: np.ndarray = None   # (n, 3) time spent in Gamma, L, X before arrival [s]
    visited: np.ndarray = None          # (n, 3) bool: valley ever occupied
    dt_spin: np.ndarray = None          # time since the last real scattering at arrival [s] (Eq. 54)
    eqv: np.ndarray = None              # int8: which equivalent valley (see gaas_mc/valleys.py)
    K: np.ndarray = None                # (n, 3) full crystal wavevector = valley center + k [1/m]
    band_edge_at_surface: float = 0.0   # E_C(z=0) - E_C(bulk) [J] (e.g. -E_bb for C21 band bending)
    mechanism_names: tuple = ()

    @property
    def p(self):
        """Crystal momentum hbar k [kg m/s]."""
        return HBAR * self.k

    def __len__(self):
        return self.t.size

    def upper_valley_summary(self):
        """Upper-valley participation. Spin is frozen in L/X (user decision 4), so this measures
        how many arrivals that untested assumption touches."""
        n = len(self)
        if n == 0:
            return {}
        tv = self.time_in_valley
        return {
            "fraction_visited_L": float(self.visited[:, 1].mean()),
            "fraction_visited_X": float(self.visited[:, 2].mean()),
            "fraction_visited_L_or_X": float((self.visited[:, 1] | self.visited[:, 2]).mean()),
            "fraction_arriving_in_L_or_X": float((self.valley > 0).mean()),
            "mean_time_in_L [s]": float(tv[:, 1].mean()),
            "mean_time_in_X [s]": float(tv[:, 2].mean()),
            "mean_fraction_of_time_in_L_or_X":
                float(((tv[:, 1] + tv[:, 2]) / np.maximum(tv.sum(1), 1e-300)).mean()),
        }

    def esp(self, mask=None):
        s = self.spin if mask is None else self.spin[mask]
        return float(s.mean()) if s.size else np.nan

    @classmethod
    def empty(cls, n_mech, names=()):
        z = np.zeros(0)
        return cls(t=z, E=z, k=np.zeros((0, 3)), valley=np.zeros(0, np.int8),
                   spin=np.zeros(0, np.int8), spin0=np.zeros(0, np.int8), z0=z, E0=z,
                   band=np.zeros(0, np.int8), n_flips=np.zeros(0, np.int32),
                   n_events=np.zeros((0, n_mech), np.int32), pid=np.zeros(0, int),
                   time_in_valley=np.zeros((0, 3)), visited=np.zeros((0, 3), bool),
                   eqv=np.zeros(0, np.int8), K=np.zeros((0, 3)), dt_spin=np.zeros(0),
                   mechanism_names=tuple(names))

    @classmethod
    def concatenate(cls, parts, n_mech, names=()):
        if not parts:
            return cls.empty(n_mech, names)
        out = {}
        for f in fields(cls):
            if f.name in ("mechanism_names", "band_edge_at_surface"):
                continue
            out[f.name] = np.concatenate([getattr(p, f.name) for p in parts])
        return cls(**out, band_edge_at_surface=parts[0].band_edge_at_surface, mechanism_names=tuple(names))

    def save_npz(self, path):
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "mechanism_names"}
        d["band_edge_at_surface"] = np.array(self.band_edge_at_surface)
        np.savez_compressed(path, mechanism_names=np.array(self.mechanism_names), **d)

    @classmethod
    def load_npz(cls, path):
        d = dict(np.load(path, allow_pickle=False))
        names = tuple(str(x) for x in d.pop("mechanism_names"))
        d["band_edge_at_surface"] = float(d.get("band_edge_at_surface", 0.0))
        return cls(**d, mechanism_names=names)


@dataclass
class Emissions:
    """Electrons emitted by a surface model (e.g. surface_c21.C21Surface). Inside-state at the moment
    of emission plus the vacuum state returned by the surface model."""
    t: np.ndarray            # emission time [s]
    E: np.ndarray            # kinetic energy inside, above the valley minimum [J]
    k: np.ndarray            # (n, 3) inside wavevector from the valley minimum
    K: np.ndarray            # (n, 3) full crystal wavevector inside
    valley: np.ndarray
    eqv: np.ndarray
    spin: np.ndarray
    spin0: np.ndarray
    z0: np.ndarray
    E0: np.ndarray
    band: np.ndarray
    n_surface: np.ndarray    # surface encounters including the emitting one
    pid: np.ndarray
    p_vac: np.ndarray        # (n, 3) momentum in vacuum [kg m/s] (p_z < 0: leaving toward -z)
    E_vac: np.ndarray        # kinetic energy in vacuum above the vacuum level [J]

    def __len__(self):
        return self.t.size

    @property
    def E_perp(self):
        """Transverse (in-plane) kinetic energy in vacuum [J]."""
        from .constants import M0
        return (self.p_vac[:, 0] ** 2 + self.p_vac[:, 1] ** 2) / (2 * M0)

    def esp(self):
        return float(self.spin.mean()) if self.t.size else np.nan

    @classmethod
    def concatenate(cls, parts):
        if not parts:
            z = np.zeros(0)
            return cls(t=z, E=z, k=np.zeros((0, 3)), K=np.zeros((0, 3)), valley=np.zeros(0, np.int8),
                       eqv=np.zeros(0, np.int8), spin=np.zeros(0, np.int8), spin0=np.zeros(0, np.int8),
                       z0=z, E0=z, band=np.zeros(0, np.int8), n_surface=np.zeros(0, np.int32),
                       pid=np.zeros(0, int), p_vac=np.zeros((0, 3)), E_vac=z)
        return cls(**{f.name: np.concatenate([getattr(q, f.name) for q in parts]) for f in fields(cls)})
