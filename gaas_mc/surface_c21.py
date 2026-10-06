"""Optional Chubenko et al. (2021) surface-emission model (Stage E benchmark module).

Kept separate from the bulk Monte Carlo: transport only calls ``interact()`` when an electron
reaches z = 0, so any other surface model with the same interface can replace this one.

[C21] Sec. III C 2 and Fig. 15(c):
  * Vacuum level at the surface: E_vac = E_C,Gamma(z=0) + chi; chi is the electron affinity
    (the free fitting parameter; chi_eff = chi - E_bb).
  * Triangular barrier on the vacuum side, x = distance from the interface:
        V(x) = E_vac + E_b (1 - x/L_b)  for 0 < x < L_b,   V = E_vac beyond;   L_b = 0.15 nm, E_b = 4 eV.
  * Quantum transmission by the propagation (transfer) matrix method; the mass changes from m* to m0
    at the interface.
  * Escape only from Gamma and "some X valleys" (ideal (100) surface, transverse crystal momentum
    conserved).
  * Electrons with total energy below the vacuum level are trapped at the surface and removed (Sec. IV).
  * Otherwise an electron that is not transmitted is reflected back into the GaAs.

Implementation choices (docs/MODEL_ASSUMPTIONS.md, Stage E):
  * Transverse crystal momentum: K_par (valley center + k, lab z = [001]) is folded into the (001)
    surface Brillouin zone (2D reciprocal lattice (2 pi/a)(1, +-1)) and conserved. The vacuum normal
    energy is eps = E_tot - E_vac - hbar^2 K_par^2 / (2 m0); emission requires eps > 0. This removes
    L and the in-plane X valleys (folded valley-center K_par: hbar^2 K_par^2 / 2m0 = 2.35 eV and 4.7 eV)
    at every energy reachable in these simulations, leaving Gamma and X[001], i.e. C21's "Gamma and
    some X valleys". The restriction is energetic, not a hard valley rule: the electron's own k shifts
    K_par, so an L electron ~1 eV above its minimum with a small chi could escape.
  * Energies inside: E_tot = (valley offset) + E_kin, relative to the Gamma band edge at z = 0.
  * Transfer matrix: BenDaniel-Duke matching (psi and psi'/m continuous). Inside the GaAs the normal
    wavevector is the electron's k_z with mass m_in: matching_mass="band_edge" (default since
    2026-10-06, user decision) uses the band-edge m*, the literal reading of C21's "the electron mass
    changes from m*_e to m0", which reproduces C21 Fig. 18 QE (docs/VALIDATION.md);
    matching_mass="velocity" uses m*(1 + 2 alpha E), so that the incident flux hbar k_z / m_in equals
    the group velocity of the nonparabolic band (option). The barrier is split into n_slices
    constant slices of mass m0; the vacuum has mass m0.
        T = (k_out / m0) / (k_in / m_in) * |t|^2
  * Reflection is specular (k_z -> -k_z, k_par unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import asarray, dev, xp_of
from .constants import EV, HBAR, M0, NM

EMIT, TRAP, REFLECT = 1, 2, 3


@dataclass
class C21Surface:
    chi: float                      # electron affinity [J]
    material: object
    E_b: float = 4.0 * EV
    L_b: float = 0.15 * NM
    n_slices: int = 200
    matching_mass: str = "band_edge"
    """GaAs-side mass in the transfer matrix: "band_edge" (default): the band-edge m* of C21's
    sentence "the electron mass changes from m*_e to m0" read literally, which reproduces C21 Fig. 18;
    "velocity": m*(1 + 2 alpha E), so the incident flux hbar k_z / m equals the group velocity (k_z
    is the electron's own wavevector in both cases). C21 do not state how nonparabolicity enters."""
    is_surface_model: bool = True   # marker used by transport

    def __post_init__(self):
        if self.matching_mass not in ("velocity", "band_edge"):
            raise ValueError(f"matching_mass={self.matching_mass!r}")
        vs = self.material.valleys
        self._offset = np.array([v.offset for v in vs])
        self._m = np.array([v.m_eff for v in vs])
        self._alpha = np.array([v.alpha for v in vs])

    # ------------------------------------------------------------------ geometry / momentum ---
    def fold_kpar(self, K):
        """Fold (Kx, Ky) into the (001) surface BZ; returns |K_par| of the folded vector [1/m]."""
        a = self.material.a_lat
        g = 2 * np.pi / a
        kx, ky = K[:, 0], K[:, 1]
        # coordinates in the reciprocal basis b1 = g(1, 1), b2 = g(1, -1)
        u = (kx + ky) / (2 * g)
        v = (kx - ky) / (2 * g)
        best = xp_of(kx).full(kx.size, np.inf)
        for du in (0, 1):
            for dv in (0, 1):
                m = np.floor(u) + du
                n = np.floor(v) + dv
                rx = kx - g * (m + n)
                ry = ky - g * (m - n)
                best = np.minimum(best, rx * rx + ry * ry)
        return np.sqrt(best)

    def total_energy(self, E_kin, valley):
        """E_tot relative to the Gamma band edge at z = 0 [J]."""
        return dev(self._offset, xp_of(valley))[valley] + E_kin

    def barrier(self, x):
        """Potential on the vacuum side, relative to the Gamma band edge at z = 0 (Fig. 15(c))."""
        return self.chi + self.E_b * (1 - x / self.L_b)

    # ------------------------------------------------------------------ transfer matrix -------
    def transmission(self, E_tot, k_in, m_in, kpar):
        """Transmission probability for incident normal wavevector k_in (> 0, toward vacuum) with
        incident mass m_in, total energy E_tot (relative to the Gamma band edge at the surface), and
        conserved transverse wavevector kpar. Arrays of equal length."""
        xp = xp_of(E_tot, k_in, m_in, kpar)
        E_tot, k_in, m_in, kpar = (xp.atleast_1d(asarray(x, float)) for x in (E_tot, k_in, m_in, kpar))
        n = E_tot.size
        eps_vac = E_tot - self.chi - HBAR**2 * kpar**2 / (2 * M0)       # normal energy in vacuum
        T = xp.zeros(n)
        ok = (eps_vac > 0) & (k_in > 0)
        if not ok.any():
            return T
        Et, kin, min_, kp = E_tot[ok], k_in[ok], m_in[ok], kpar[ok]
        N = self.n_slices
        dx = self.L_b / N
        xs = (np.arange(N) + 0.5) * dx                                    # slice centres
        V = self.barrier(xs)
        k_par_term = HBAR**2 * kp**2 / (2 * M0)
        # wavevectors: region 0 = GaAs, 1..N = slices, N+1 = vacuum
        ks = [kin.astype(complex)]
        ms = [min_]
        for Vj in V:
            ks.append(np.sqrt((2 * M0 * (Et - Vj - k_par_term) / HBAR**2).astype(complex)))
            ms.append(xp.full(Et.size, M0))
        k_out = np.sqrt(2 * M0 * (Et - self.chi - k_par_term)) / HBAR
        ks.append(k_out.astype(complex))
        ms.append(xp.full(Et.size, M0))
        x_if = np.arange(N + 1) * dx                                      # interfaces 0..N
        # back-propagate from the vacuum: A = 1 (transmitted), B = 0
        A = xp.ones(Et.size, complex)
        B = xp.zeros(Et.size, complex)
        for i in range(N, -1, -1):                                        # interface between i and i+1
            ka, ma = ks[i], ms[i]
            kb, mb = ks[i + 1], ms[i + 1]
            x = x_if[i]
            ka = np.where(np.abs(ka) < 1e-6, 1e-6, ka)
            rho = (kb / mb) / (ka / ma)
            eb_p, eb_m = np.exp(1j * kb * x), np.exp(-1j * kb * x)
            ea_p, ea_m = np.exp(1j * ka * x), np.exp(-1j * ka * x)
            Aa = 0.5 * ((1 + rho) * A * eb_p + (1 - rho) * B * eb_m) / ea_p
            Ba = 0.5 * ((1 - rho) * A * eb_p + (1 + rho) * B * eb_m) / ea_m
            A, B = Aa, Ba
        T[ok] = (k_out / M0) / (kin / min_) / np.abs(A) ** 2
        return T

    # ------------------------------------------------------------------ transport interface ---
    def interact(self, k, E, valley, K, rng, pid=None):
        """Decide the fate of electrons arriving at z = 0 (k_z < 0 toward the surface).
        pid (particle ids) is part of the surface-model interface; this model does not use it.

        Returns (outcome, info): outcome in {EMIT, TRAP, REFLECT}; info has arrays for all
        particles: E_tot, eps_vac, kpar, T, and the vacuum momentum p_vac (n, 3) and vacuum kinetic
        energy E_vac_kin (relative to the vacuum level), meaningful for the emitted ones.
        """
        k = asarray(k, float)
        valley = asarray(valley)
        xp = xp_of(k)
        m = dev(self._m, xp)[valley]
        a = dev(self._alpha, xp)[valley]
        E_tot = self.total_energy(E, valley)
        kpar = self.fold_kpar(K)
        outcome = xp.full(E.size, REFLECT, np.int8)
        # 1. absolute total energy below the asymptotic vacuum level E_vac = E_C,Gamma(0) + chi:
        #    surface-trapped and terminated (C21 Sec. IV); no barrier dynamics
        trapped = E_tot < self.chi
        outcome[trapped] = TRAP
        # 2. E_tot >= E_vac: barrier transmission; transmitted -> emitted, otherwise reflected
        T = xp.zeros(E.size)
        up = ~trapped
        m_v = m * (1 + 2 * a * E) if self.matching_mass == "velocity" else m
        T[up] = self.transmission(E_tot[up], -k[up, 2], m_v[up], kpar[up])
        u = rng.random(E.size)                     # drawn for all rows (stream independent of the split)
        emit = up & (u < T)
        outcome[emit] = EMIT
        eps = E_tot - self.chi - HBAR**2 * kpar**2 / (2 * M0)
        # vacuum momentum: folded K_par direction kept, p_z from the normal energy (toward vacuum = -z)
        Kp = K[:, :2]
        nrm = np.linalg.norm(Kp, axis=1)
        scale = np.where(nrm > 0, kpar / np.where(nrm > 0, nrm, 1), 0.0)
        p_vac = xp.zeros((E.size, 3))
        p_vac[:, :2] = HBAR * Kp * scale[:, None]
        p_vac[:, 2] = -np.sqrt(2 * M0 * np.clip(eps, 0, None))
        info = dict(E_tot=E_tot, eps_vac=eps, kpar=kpar, T=T, p_vac=p_vac, E_vac_kin=E_tot - self.chi)
        return outcome, info
