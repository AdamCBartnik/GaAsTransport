"""Local mobile-hole population in the band-bending (depletion) region.

The electrostatic potential (fields.C21BandBending or a user potential) is unchanged, and so is
the fixed ionized-acceptor charge N_A^- = p_bulk. What changes with depth is the mobile-hole
population, and with it every hole-density-dependent process (user decision of 2026-10-06):

  1. The Fermi level is global, fixed once by the bulk doping and T (hole gas: hh + lh,
     Fermi-Dirac; holes.HoleGas). No local chemical potential is solved for.
  2. At depth z the valence bands shift by the same phi(z) = E_C(z) - E_C(bulk) as the conduction
     bands, so eta(z) = eta_bulk + phi(z)/kT and
         p_b(z) = N_b F_{1/2}(eta(z)),   b = hh, lh;   p(z) = p_hh + p_lh.
  3. Local p(z) replaces bulk p in electron-hole scattering (density, FD occupation and Pauli
     blocking of the local hole gas), in BAP spin relaxation (p, E_F^h of Eq. 44, and the
     degenerate/nondegenerate choice), and in any other hole-density-dependent process.
  4. Ionized impurities: N_A^- stays at the dopant density, and the screening is recomputed from
     the local mobile carriers with the Fermi-Dirac susceptibility chi = dp/dmu = sum_b N_b F_{-1/2}(eta)/kT:
         beta(z) = beta_bulk * sqrt(chi(z) / chi_bulk)
     This equals the bulk C21 value (Eq. 28 or 29) at phi = 0, and goes over to Debye
     (beta ~ sqrt(p)) in the nondegenerate limit. The same local beta enters the e-h rate
     (Eqs. 38-42) and, by default, the screened POP rate (Eq. 25), which C21 writes with the same
     hole beta.
  5. Screening cap (ambiguity, see docs/MODEL_ASSUMPTIONS.md): as p(z) -> 0 the screening length
     diverges, and so does the Brooks-Herring rate (it goes as 1/beta^2 at fixed energy).
     Brooks-Herring is meaningless once the screening length exceeds the distance between
     ions (the Conwell-Weisskopf argument), so the local length is capped:
         L(z) = min(1/beta(z), L_cap),   L_cap = max(L_bulk, a)
     a = "impurity_spacing" (default): the acceptor Wigner-Seitz radius (3/(4 pi N_A))^(1/3);
         "band_bending_width": W_bb;  or an explicit length [m].
     Taking the max with L_bulk keeps the bulk exactly unchanged.
  6. Phonon (acoustic, POP coupling) and intervalley parameters are unchanged.
  Minority electrons (n ~ n_i^2/p) and photoexcited carriers contribute nothing to screening.

Implementation: the quantities are tabulated on a uniform grid of phi (spacing <= kT/4) from
phi(0) to 0. Each density-dependent mechanism is replaced by a LocalMechanism holding one variant
per grid point. Rates are interpolated linearly in phi; the variant used for a scattering event is
chosen so that this interpolation is exact in expectation.
"""
from __future__ import annotations

import numpy as np

from .constants import HBAR


class LocalSample:
    """Read-only view of a Sample with some attributes overridden (local p, beta, E_F^h, ...)."""

    def __init__(self, bulk, **overrides):
        self._bulk = bulk
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._bulk, name)


class DepletionModel:
    def __init__(self, sample, field, holes, screening_cap="impurity_spacing", pop_screening=True,
                 dphi_max_kT=0.25):
        self.sample = sample
        self.field = field
        self.holes = holes
        self.pop_screening = pop_screening
        kT = sample.kT
        phi0 = float(field.band_edge(np.array([0.0]))[0])
        if phi0 >= 0:
            raise ValueError("depletion model needs downward band bending (phi(0) < 0)")
        n = int(np.ceil(-phi0 / (dphi_max_kT * kT))) + 1
        self.phi = np.linspace(phi0, 0.0, n)
        # screening cap
        N_A = sample.p
        if screening_cap == "impurity_spacing":
            a = (3.0 / (4 * np.pi * N_A)) ** (1 / 3)
        elif screening_cap == "band_bending_width":
            a = float(field.z_max)
        else:
            a = float(screening_cap)
        self.screening_cap = screening_cap
        self.L_cap = max(1.0 / sample.beta, a)
        chi_b = holes.susceptibility
        self.gas, self.samples = [], []
        self.p = np.empty(n); self.p_hh = np.empty(n); self.p_lh = np.empty(n)
        self.beta = np.empty(n); self.beta_uncapped = np.empty(n); self.EF_h = np.empty(n)
        self.degenerate = np.empty(n, bool)
        for j, ph in enumerate(self.phi):
            g = holes.local(ph)
            beta_raw = sample.beta * np.sqrt(g.susceptibility / chi_b)
            beta = max(beta_raw, 1.0 / self.L_cap)
            EF_h = HBAR**2 * (3 * np.pi**2 * g.p) ** (2 / 3) / (2 * sample.material.m_hh)   # Eq. 44, local p
            degen = (sample.EF_bulk - ph) < 2 * kT        # E_F - E_V(z), C21 criterion (Eq. 58)
            self.gas.append(g)
            self.p[j], self.beta[j], self.beta_uncapped[j], self.EF_h[j] = g.p, beta, beta_raw, EF_h
            self.p_hh[j] = g.density.get("hh", 0.0); self.p_lh[j] = g.density.get("lh", 0.0)
            self.degenerate[j] = degen
            self.samples.append(LocalSample(sample, p=g.p, beta=beta, EF_h=EF_h, degenerate=bool(degen),
                                            p_ionized=N_A))
        self.n = n
        self.bulk_index = n - 1
        if not np.isclose(self.beta[-1], sample.beta, rtol=1e-9):
            raise RuntimeError("local screening at phi = 0 must equal the bulk value")

    def frac_index(self, z):
        """Fractional grid index of phi(z) (bulk index where the field vanishes)."""
        z = np.asarray(z, float)
        ph = self.field.band_edge(np.clip(z, 0.0, None))
        return np.interp(ph, self.phi, np.arange(self.n, dtype=float))

    def profile(self, z):
        """Local quantities at depths z (for diagnostics)."""
        fi = self.frac_index(z)
        return {key: np.interp(fi, np.arange(self.n), getattr(self, key))
                for key in ("phi", "p", "p_hh", "p_lh", "beta", "beta_uncapped", "EF_h")}


class LocalMechanism:
    """A density-dependent mechanism with one variant per phi grid point."""

    def __init__(self, variants, depletion):
        self.variants = list(variants)
        self.depletion = depletion
        b = self.variants[-1]
        self.bulk = b
        self.name = b.name
        self.valley_from = b.valley_from
        self.spin_class = b.spin_class
        self.threshold = min(v.threshold for v in self.variants)
        self.valley = b.valley

    def rate(self, E):
        return self.bulk.rate(E)

    def momentum_rate(self, E):
        return self.bulk.momentum_rate(E)

    def __getattr__(self, name):          # e.g. hw, Z, valley_to for diagnostics
        return getattr(self.__dict__["bulk"], name)

    def tabulate(self, E_grid):
        """Rate table (n_phi, n_E) on the transport energy grid (called by Simulation)."""
        self.E_grid = np.asarray(E_grid, float)
        self.table = np.array([v.rate(self.E_grid) for v in self.variants]).reshape(self.depletion.n, -1)
        return self.table

    def rates_at(self, E, fi):
        """Rate at energies E and fractional phi indices fi (bilinear in the table)."""
        g = self.E_grid
        j = np.clip(np.searchsorted(g, E, side="right"), 1, g.size - 1)
        u = np.clip((E - g[j - 1]) / (g[j] - g[j - 1]), 0.0, 1.0)
        i0 = np.clip(np.floor(fi).astype(int), 0, self.depletion.n - 1)
        i1 = np.minimum(i0 + 1, self.depletion.n - 1)
        w = np.clip(fi - i0, 0.0, 1.0)
        T = self.table
        r0 = T[i0, j - 1] * (1 - u) + T[i0, j] * u
        r1 = T[i1, j - 1] * (1 - u) + T[i1, j] * u
        return r0, r1, w, i0, i1

    def choose_variant(self, E, fi, rng):
        """Pick a variant for each event so that the linear-in-phi rate interpolation is exact:
        variant i1 with probability w W1 / ((1 - w) W0 + w W1)."""
        W0, W1, w, j0, j1 = self.rates_at(E, fi)
        tot = (1 - w) * W0 + w * W1
        with np.errstate(invalid="ignore", divide="ignore"):
            p1 = np.where(tot > 0, w * W1 / tot, w)
        return np.where(rng.random(E.size) < p1, j1, j0)

    def scatter(self, k, E, rng, phi_index=None):
        E = np.asarray(E, float)
        if phi_index is None:
            return self.bulk.scatter(k, E, rng)
        j = self.choose_variant(E, np.asarray(phi_index, float), rng)
        k_new = np.array(k, float, copy=True)
        acc = np.zeros(E.size, bool)
        vnew = np.full(E.size, self.valley_from)
        for jj in np.unique(j):
            sel = j == jj
            kn, a, vn = self.variants[jj].scatter(k[sel], E[sel], rng)
            k_new[sel], acc[sel], vnew[sel] = kn, a, vn
        return k_new, acc, vnew
