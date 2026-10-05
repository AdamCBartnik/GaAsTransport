"""Spin relaxation in the Gamma valley: Elliott-Yafet, D'yakonov-Perel, Bir-Aronov-Pikus.

[C21] Sec. III B 2, Eqs. 45-54. Discrete spin s = +1/-1 along +z, the light propagation
direction (into the material). s = +1 is "parallel to the direction of light propagation"
(Fig. 2 caption), which is the majority state in Eq. 4.

EY, Eq. 45 (for each momentum-scattering mechanism i):
    1/tau_EY^i = A_i (1 - m*/m0)^2 (eta/(1+eta))^2 ((1+eta/2)/(1+2eta/3))^2 (E/Eg)^2 / tau_m^i,
    eta = Delta_so / Eg;    total EY = sum over i.
DP, Eq. 46:
    1/tau_DP = (128/945) Q Delta_so^2 B^2 / ((1+eta)(1+2eta/3)) m*^2/hbar^6 (1 - m*/m0) (E/Eg)^3 tau_m
    The total uses the Matthiessen total tau_m of Eq. 52 (ap + pop + ij + ii; e-h is not included).
    Per-mechanism DP curves (Fig. 11) use tau_m^i alone.
BAP:
    nondegenerate, Eq. 47:        (2/tau0) (v_k/v_B) a_B^3 p |psi(0)|^4
    degenerate, thermal, Eq. 50:  (3/tau0) (v_k/v_B) (kT/E_F^h) a_B^3 p |psi(0)|^4,  E <= m_h (kT)^2/(m_e E_F^h)
    degenerate, hot, Eq. 51:      (2/tau0) (v_F/v_B) (E/E_F^h) a_B^3 p |psi(0)|^4,   v_F = sqrt(2 E_F^h / m_h)
    1/tau0 from Eq. 48; |psi(0)|^2 = 1 (total screening, text after Eq. 49); p = total hole density (A11).
Total, Eq. 53:  1/tau_s = 1/tau_EY + 1/tau_DP + 1/tau_BAP.
Flip rule, Eq. 54:  P = 0.5 (1 - exp(-dt/tau_s)), with dt the time since the previous real
(non-self) scattering event. tau_s is evaluated at the electron state at the event.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bands
from .constants import HBAR, M0

MATTHIESSEN_CLASSES = ("ap", "pop", "ij", "ii")      # Eq. 52


@dataclass
class SpinModel:
    sample: object
    mechanisms: list            # momentum mechanisms of the Gamma valley; spin_class decides use
    enable_EY: bool = True
    enable_DP: bool = True
    enable_BAP: bool = True

    def __post_init__(self):
        mat = self.sample.material
        self.m = mat.gamma.m_eff
        self.alpha = mat.gamma.alpha
        self.eta = mat.Delta_so / self.sample.Eg
        self._mech = [mm for mm in self.mechanisms
                      if mm.valley_from == 0 and mm.spin_class in MATTHIESSEN_CLASSES]

    # ---- per-mechanism ingredients --------------------------------------------------------
    def ey_factor(self, E):
        """1/tau_EY^i = A_i * ey_factor(E) * (1/tau_m^i)   (Eq. 45 without A_i and 1/tau_m^i)."""
        eta = self.eta
        return ((1 - self.m / M0) ** 2 * (eta / (1 + eta)) ** 2
                * ((1 + eta / 2) / (1 + 2 * eta / 3)) ** 2 * (E / self.sample.Eg) ** 2)

    def dp_factor(self, E):
        """1/tau_DP = Q * dp_factor(E) * tau_m   (Eq. 46 without Q and tau_m)."""
        mat = self.sample.material
        eta = self.eta
        return (128 / 945 * mat.Delta_so**2 * mat.B_DP**2 / ((1 + eta) * (1 + 2 * eta / 3))
                * self.m**2 / HBAR**6 * (1 - self.m / M0) * (E / self.sample.Eg) ** 3)

    # ---- rates ----------------------------------------------------------------------------
    def ey_by_mechanism(self, E):
        mat = self.sample.material
        f = self.ey_factor(E)
        return {mm.name: mat.A_EY * f * mm.momentum_rate(E) for mm in self._mech}

    def dp_by_mechanism(self, E):
        """DP with each tau_m^i alone (what Fig. 11 plots per mechanism)."""
        mat = self.sample.material
        f = self.dp_factor(E)
        out = {}
        for mm in self._mech:
            rm = mm.momentum_rate(E)
            with np.errstate(divide="ignore"):
                out[mm.name] = np.where(rm > 0, mat.Q_DP * f / np.where(rm > 0, rm, 1.0), 0.0)
        return out

    def total_momentum_rate(self, E):
        """Eq. 52 (Matthiessen)."""
        return sum((mm.momentum_rate(E) for mm in self._mech), np.zeros_like(np.asarray(E, float)))

    def ey(self, E):
        return sum(self.ey_by_mechanism(E).values(), np.zeros_like(np.asarray(E, float)))

    def dp(self, E):
        rm = self.total_momentum_rate(E)
        f = self.dp_factor(E)
        with np.errstate(divide="ignore"):
            return np.where(rm > 0, self.sample.material.Q_DP * f / np.where(rm > 0, rm, 1.0), 0.0)

    def bap(self, E):
        s = self.sample
        mat = s.material
        E = np.asarray(E, dtype=float)
        v = bands.speed_of_E(E, self.m, self.alpha)                        # Eq. 19
        common = s.inv_tau0 * s.a_B**3 * s.p * mat.psi0_sq**2             # |psi(0)|^4
        if not s.degenerate:
            return 2 * common * v / s.v_B                                  # Eq. 47
        EF = s.EF_h
        v_F = np.sqrt(2 * EF / mat.m_hh)
        E_star = mat.m_hh * s.kT**2 / (self.m * EF)
        thermal = 3 * common * (v / s.v_B) * (s.kT / EF)                   # Eq. 50
        hot = 2 * common * (v_F / s.v_B) * (E / EF)                        # Eq. 51
        return np.where(E <= E_star, thermal, hot)

    def bap_regime_boundary(self):
        """E* = m_h (kT)^2 / (m_e E_F^h) separating Eqs. 50 and 51 (degenerate case)."""
        s = self.sample
        return s.material.m_hh * s.kT**2 / (self.m * s.EF_h)

    def total(self, E):
        """Eq. 53, honoring the enable switches."""
        E = np.asarray(E, dtype=float)
        r = np.zeros_like(E)
        if self.enable_EY:
            r = r + self.ey(E)
        if self.enable_DP:
            r = r + self.dp(E)
        if self.enable_BAP:
            r = r + self.bap(E)
        return r


def flip_probability(dt, inv_tau_s):
    """Eq. 54: P = 0.5 (1 - exp(-dt / tau_s))."""
    return 0.5 * (-np.expm1(-dt * inv_tau_s))
