"""Surface-arrival records.

When an electron coming from z > 0 reaches z = 0, its bulk trajectory stops and its full
state is stored here. No emission physics is applied: electron affinity, mass discontinuity,
image charge, and transmission belong to a separate surface model that consumes these records.

Energies are kinetic energies above the local conduction-band minimum of the electron's
valley at z = 0. Spin s = +1/-1 along +z, the light propagation direction (into the material),
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
                   mechanism_names=tuple(names))

    @classmethod
    def concatenate(cls, parts, n_mech, names=()):
        if not parts:
            return cls.empty(n_mech, names)
        out = {}
        for f in fields(cls):
            if f.name == "mechanism_names":
                continue
            out[f.name] = np.concatenate([getattr(p, f.name) for p in parts])
        return cls(**out, mechanism_names=tuple(names))

    def save_npz(self, path):
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "mechanism_names"}
        np.savez_compressed(path, mechanism_names=np.array(self.mechanism_names), **d)

    @classmethod
    def load_npz(cls, path):
        d = dict(np.load(path, allow_pickle=False))
        names = tuple(str(x) for x in d.pop("mechanism_names"))
        return cls(**d, mechanism_names=names)
