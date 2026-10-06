"""Equivalent conduction-band valleys and crystal momentum.

The transport treats L and X as spherical valleys with k measured from the valley minimum. To keep
the full crystal momentum (needed by a valley-aware surface model) we also track WHICH equivalent
valley the electron occupies.

Orientation assumption: lab axes = cubic axes, with z = [001] the surface normal ((001) surface).
Valley centers (a = lattice constant):
    Gamma:  (0, 0, 0)
    L:      (pi/a) (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)   [4 valleys; L and -L are the
            same valley, differing by a reciprocal-lattice vector; one representative is used]
    X:      (2pi/a) (1, 0, 0), (0, 1, 0), (0, 0, 1)                    [3 valleys, likewise]
Equivalent-valley choice on intervalley transfer, consistent with the multiplicities of the
intervalley rates (user decision 5): to a different valley type, any of its equivalent valleys
uniformly (L: 4, X: 3, Gamma: 1); to the same type (L->L, X->X), uniformly among the others (3 / 2).
Spherical-valley rates and isotropic intervalley scattering do not depend on which valley is chosen,
so the bookkeeping does not change the transport.
"""
from __future__ import annotations

import numpy as np

L_DIRS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
X_DIRS = np.eye(3)
N_EQUIV = (1, 4, 3)


def valley_center(valley, eqv, a):
    """Crystal wavevector of each valley minimum [1/m], shape (n, 3)."""
    valley = np.asarray(valley)
    eqv = np.asarray(eqv)
    out = np.zeros((valley.size, 3))
    L = valley == 1
    X = valley == 2
    out[L] = (np.pi / a) * L_DIRS[eqv[L]]
    out[X] = (2 * np.pi / a) * X_DIRS[eqv[X]]
    return out


def choose_equivalent_valley(v_old, eqv_old, v_new, rng):
    """New equivalent-valley index after a transfer v_old -> v_new (arrays)."""
    v_old, v_new = np.asarray(v_old), np.asarray(v_new)
    n_eq = np.array(N_EQUIV)[v_new]
    same = v_old == v_new
    out = np.floor(rng.random(v_new.size) * n_eq).astype(np.int8)       # different type: uniform
    if np.any(same & (n_eq > 1)):                                        # same type: one of the others
        sel = same & (n_eq > 1)
        shift = 1 + np.floor(rng.random(int(sel.sum())) * (n_eq[sel] - 1)).astype(np.int8)
        out[sel] = ((np.asarray(eqv_old)[sel] + shift) % n_eq[sel]).astype(np.int8)
    out[v_new == 0] = 0
    return out
