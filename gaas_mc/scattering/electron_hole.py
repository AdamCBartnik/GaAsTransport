"""Electron-hole binary scattering (screened Coulomb, Born), rejection technique.

[C21] Eqs. 38-44; holes sampled from an equilibrium hh/lh gas (holes.py, Karkare 2013 approach).
One Mechanism instance per hole band ('hh', 'lh'), so each can be disabled separately.

Derivation (checked independently because a bug in this mechanism is the subject of the
Karkare 2015 erratum, which does not describe the bug):
  For one hole of wavevector k0 (band mass m_h) and an electron k (mass m_e), the relative
  wavevector is k_rel = m_R (k0/m_h - k/m_e), with m_R = m_e m_h/(m_e + m_h) (Eq. 39), and
  v_rel = hbar |k_rel| / m_R. The Born cross-section for V(r) = e^2 exp(-beta r)/(4 pi eps_s r) is
      sigma(k_rel) = m_R^2 e^4 / (pi eps_s^2 hbar^4 beta^2 (beta^2 + 4 k_rel^2)).
  With g = 2 |k_rel| (Eq. 41), the rate against a hole density p_b is
      W(g) = p_b v_rel sigma = p_b m_R e^4 / (2 pi eps_s^2 hbar^3 beta^2) * g/(g^2 + beta^2),
  and g/(g^2+beta^2) <= 1/(2 beta) gives the constant majorant
      W_max = p_b m_R e^4 / (4 pi eps_s^2 hbar^3 beta^3)                          (Eq. 38)
  with acceptance  r < W(g)/W_max = 2 g beta/(g^2 + beta^2)                       (Eq. 40).
  The relative-motion scattering angle follows the screened-Coulomb distribution
      cos(theta) = 1 - 2r / (1 + g^2 (1 - r)/beta^2)                              (Eq. 42),
  and in the parabolic case k' = k - (g' - g)/2 with |g'| = |g|                     (Eq. 43).

Final state (user decision 3, "proper two-body kinematics"): with K = k + k0 and
c = m_R K/m_h, we have k = c - g/2 exactly for any m_e. The final state is
k' = c - s g_hat'/2 and k0' = K - k', where g_hat' is the sampled direction and s >= 0 solves
    E_e(|k'|) + hbar^2 |k0'|^2/(2 m_h) = E_e(|k|) + hbar^2 |k0|^2/(2 m_h)
with the nonparabolic E_e (Eq. 2). For alpha = 0 this gives s = g, identical to Eq. 43.
Momentum and energy are conserved to round-off for every accepted collision. If no
root exists (only possible for the nonparabolic band), the event is rejected and counted.

Pauli blocking of the final hole state (ModelAssumptions.pauli_blocking):
  "fermi_dirac": accept with probability 1 - f(E_h'), the same f used to sample the initial hole
                 (keeps detailed balance with the hole bath);
  "step_c21":    reject if E_h' < E_F^h of C21 Eq. 44;
  "none".
Rejected events count as self-scattering (C21 text after Eq. 44): no spin-flip test.
"""
from __future__ import annotations

import numpy as np

from .. import bands
from ..backend import asarray, xp_of
from ..constants import HBAR, Q_E
from .base import Mechanism


class ElectronHole(Mechanism):
    spin_class = None          # excluded from the Matthiessen sum, Eq. 52

    def __init__(self, sample, holes, band="hh", valley_from=0, pauli="fermi_dirac",
                 mass_model="band_edge"):
        super().__init__(sample, valley_from)
        if band not in holes.bands:
            raise ValueError(f"hole band {band} not in hole gas {holes.bands}")
        if pauli not in ("fermi_dirac", "step_c21", "none"):
            raise ValueError(pauli)
        if mass_model not in ("band_edge", "velocity_mass"):
            raise ValueError(mass_model)
        self.holes = holes
        self.band = band
        self.pauli = pauli
        self.mass_model = mass_model
        self.name = f"eh_{band}[{self.valley.name}]"
        self.mh = holes.mass[band]
        self.p_b = holes.density[band]
        self.beta = sample.beta
        self._C = self.p_b * Q_E**4 / (4 * np.pi * sample.eps_s**2 * HBAR**3 * self.beta**3)
        self.stats = dict(attempted=0, rejected_eq40=0, rejected_kinematics=0, rejected_pauli=0, accepted=0)
        self.record = False
        self.last = None

    def m_e(self, E):
        E = asarray(E, float)
        if self.mass_model == "band_edge":
            return xp_of(E).full_like(E, self.m)
        return self.m * (1 + 2 * self.alpha * E)

    def m_R(self, E):
        me = self.m_e(E)
        return me * self.mh / (me + self.mh)                                           # Eq. 39

    def rate(self, E):
        """Majorant W_max(E), Eq. 38 (the energy dependence enters only via the velocity mass)."""
        return self._C * self.m_R(E)

    def momentum_rate(self, E):
        raise NotImplementedError("e-h is excluded from Eq. 52; use effective_rate() for diagnostics")

    # ------------------------------------------------------------------------------------------
    def _solve_s(self, c, K, ghat, E_tot, s_guess):
        """Root s >= 0 of F(s) = E_e(|c - s ghat/2|) + hbar^2 |K - c + s ghat/2|^2/(2 m_h) - E_tot.
        Newton iteration from s = g (the exact root for a parabolic band) with analytic dF/ds;
        any row that does not converge falls back to vectorized bisection."""
        m, a, mh = self.m, self.alpha, self.mh

        def F(s):
            ke = c - 0.5 * s[:, None] * ghat
            kh = K - ke
            return (bands.E_of_k(np.linalg.norm(ke, axis=1), m, a)
                    + HBAR**2 * np.sum(kh * kh, axis=1) / (2 * mh) - E_tot)

        n = c.shape[0]
        xp = xp_of(c)
        lo = xp.zeros(n)
        f_lo = F(lo)
        ok = f_lo < 0
        # --- Newton
        s = s_guess.astype(float).copy()
        done = xp.zeros(n, bool)
        tol = 1e-13 * np.abs(E_tot)
        for _ in range(12):
            ke = c - 0.5 * s[:, None] * ghat
            kh = K - ke
            kn = np.linalg.norm(ke, axis=1)
            Ee = bands.E_of_k(kn, m, a)
            f = Ee + HBAR**2 * np.sum(kh * kh, axis=1) / (2 * mh) - E_tot
            dEdk = HBAR**2 * kn / (m * (1 + 2 * a * Ee))
            with np.errstate(invalid="ignore", divide="ignore"):
                dk = -np.sum(ke * ghat, axis=1) / (2 * np.where(kn > 0, kn, 1.0))
            df = dEdk * dk + HBAR**2 * np.sum(kh * ghat, axis=1) / (2 * mh)
            done = np.abs(f) <= tol
            if np.all(done | ~ok):
                break
            with np.errstate(invalid="ignore", divide="ignore"):
                step = np.where(done, 0.0, f / df)
            s = s - np.where(np.isfinite(step), step, 0.0)
        good = ok & done & (s >= 0)
        if np.all(good | ~ok):
            return s, ok
        # --- bisection fallback for the remaining rows
        rest = xp.flatnonzero(ok & ~good)
        s_b, _ = self._bisect(c[rest], K[rest], ghat[rest], E_tot[rest], s_guess[rest])
        s[rest] = s_b
        return s, ok

    def _bisect(self, c, K, ghat, E_tot, s_guess):
        m, a, mh = self.m, self.alpha, self.mh

        def F(s):
            ke = c - 0.5 * s[:, None] * ghat
            kh = K - ke
            return (bands.E_of_k(np.linalg.norm(ke, axis=1), m, a)
                    + HBAR**2 * np.sum(kh * kh, axis=1) / (2 * mh) - E_tot)

        n = c.shape[0]
        lo = xp_of(c).zeros(n)
        f_lo = F(lo)
        ok = f_lo < 0
        hi = np.maximum(2.0 * s_guess, 1.0)
        for _ in range(200):
            bad = ok & (F(hi) <= 0)
            if not bad.any():
                break
            hi[bad] *= 2.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fm = F(mid)
            neg = fm < 0
            lo = np.where(neg, mid, lo)
            hi = np.where(neg, hi, mid)
        return 0.5 * (lo + hi), ok

    def scatter(self, k, E, rng):
        k = asarray(k, float)
        E = asarray(E, float)
        xp = xp_of(E)
        n = E.size
        accepted = xp.zeros(n, bool)
        k_new = k.copy()
        self.stats["attempted"] += n
        k0 = self.holes.sample_k(self.band, n, rng)
        me = self.m_e(E)
        mR = me * self.mh / (me + self.mh)
        gvec = 2 * mR[:, None] * (k0 / self.mh - k / me[:, None])                      # Eq. 41
        g = np.linalg.norm(gvec, axis=1)
        b = self.beta
        a1 = rng.random(n) < 2 * g * b / (g**2 + b**2)                                  # Eq. 40
        self.stats["rejected_eq40"] += int((~a1).sum())
        i1 = xp.flatnonzero(a1)
        if i1.size == 0:
            return k_new, accepted, xp.full(n, self.valley_from)
        g1 = g[i1]
        r = rng.random(i1.size)
        cos_t = 1 - 2 * r / (1 + g1**2 * (1 - r) / b**2)                                # Eq. 42
        ghat = bands.unit(gvec[i1], rng)
        ghat_new = bands.rotate_about(ghat, np.clip(cos_t, -1, 1), 2 * np.pi * rng.random(i1.size))
        K = k[i1] + k0[i1]
        c = (mR[i1] / self.mh)[:, None] * K
        E_tot = E[i1] + HBAR**2 * np.sum(k0[i1] ** 2, axis=1) / (2 * self.mh)
        s, ok = self._solve_s(c, K, ghat_new, E_tot, g1)
        self.stats["rejected_kinematics"] += int((~ok).sum())
        kp = c - 0.5 * s[:, None] * ghat_new                                            # final electron
        k0p = K - kp                                                                    # final hole
        Eh_new = HBAR**2 * np.sum(k0p**2, axis=1) / (2 * self.mh)
        if self.pauli == "fermi_dirac":
            free = rng.random(i1.size) < 1 - self.holes.occupation(Eh_new)
        elif self.pauli == "step_c21":
            free = Eh_new >= self.sample.EF_h                                           # Eq. 44
        else:
            free = xp.ones(i1.size, bool)
        self.stats["rejected_pauli"] += int((ok & ~free).sum())
        acc = ok & free
        accepted[i1[acc]] = True
        k_new[i1[acc]] = kp[acc]
        self.stats["accepted"] += int(acc.sum())
        if self.record:
            self.last = dict(idx=i1[acc], k=k[i1[acc]], k0=k0[i1[acc]], kp=kp[acc], k0p=k0p[acc],
                             E=E[i1[acc]], g=g1[acc], s=s[acc])
        return k_new, accepted, xp.full(n, self.valley_from)

    def effective_rate(self, E, rng, n=20_000):
        """Monte Carlo estimate of the actual (accepted) rate at energy E [J] for an isotropic
        electron: W_max * acceptance fraction (Eq. 40 x kinematics x Pauli)."""
        Ea = np.full(n, float(E))
        kk = bands.random_unit_vectors(n, rng) * bands.k_of_E(Ea, self.m, self.alpha)[:, None]
        _, acc, _ = self.scatter(kk, Ea, rng)
        return float(self.rate(np.array([E]))[0] * acc.mean())
