"""Material parameters and doping-dependent derived quantities.

Reference: [C21] O. Chubenko et al., J. Appl. Phys. 130, 063101 (2021), Table I
(T = 300 K). Equation numbers refer to the accepted manuscript (refs/).

Everything stored here is SI. Only the factory ``gaas_chubenko2021()`` touches
the published (eV, meV, m0, eV/Angstrom, ...) units.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

from .constants import (ANGSTROM, EPS0, EV, HBAR, K_B, M0, MEV, Q_E)

GAMMA, L, X = 0, 1, 2
VALLEY_NAMES = ("Gamma", "L", "X")


@dataclass(frozen=True)
class Valley:
    """One conduction-band valley, Kane dispersion E(1 + alpha E) = hbar^2 k^2 / 2m [Eq. 1]."""
    name: str
    m_eff: float        # effective mass [kg]
    alpha: float        # nonparabolicity [1/J]
    offset: float       # energy of valley minimum above the Gamma minimum [J]
    Xi_d: float         # acoustic deformation potential [J]
    Z: int              # number of equivalent valleys of this type (Table I "to scatter into")


@dataclass(frozen=True)
class Material:
    valleys: tuple[Valley, ...]
    # valence bands (parabolic, Eq. 3)
    m_hh: float
    m_lh: float
    m_so: float
    Eg0: float              # intrinsic gap [J]
    Delta_so: float         # split-off energy [J]
    # lattice / dielectric
    eps_inf: float          # relative
    eps_s: float            # relative
    rho: float              # [kg/m^3]
    v_s: float              # [m/s]
    hw0: float              # polar optical phonon energy [J]
    # intervalley (Stage C); keys like ("Gamma", "L")
    D_iv: dict = field(default_factory=dict)     # [J/m]
    hw_iv: dict = field(default_factory=dict)    # [J]
    # spin relaxation constants
    A_EY: float = 32 / 27
    Q_DP: float = 1 / 6
    B_DP: float = 10 * HBAR**2 / (2 * M0)        # [J m^2]
    Delta_exc: float = 47e-6 * EV                 # [J]
    psi0_sq: float = 1.0                          # |psi(0)|^2
    a_lat: float = 5.65325e-10                    # lattice constant [m], 300 K (Blakemore 1982, C21 ref. 29)

    @property
    def gamma(self) -> Valley:
        return self.valleys[GAMMA]

    @property
    def c_l(self) -> float:
        """Longitudinal elastic constant c_l = rho v_s^2 (text after Eq. 23)."""
        return self.rho * self.v_s**2

    @property
    def eps_p(self) -> float:
        """Effective polar dielectric constant [F/m]: (1/eps_inf - 1/eps_s)^-1 eps0 (after Eq. 25)."""
        return EPS0 / (1.0 / self.eps_inf - 1.0 / self.eps_s)


def gaas_chubenko2021() -> Material:
    """GaAs at 300 K, [C21] Table I. Every number below is from Table I."""
    ev_a = EV / ANGSTROM
    valleys = (
        Valley("Gamma", 0.063 * M0, 0.61 / EV, 0.0, 7.01 * EV, 1),
        Valley("L", 0.22 * M0, 0.461 / EV, 0.284 * EV, 9.2 * EV, 4),
        Valley("X", 0.58 * M0, 0.204 / EV, 0.476 * EV, 9.0 * EV, 3),
    )
    D_iv = {("Gamma", "L"): 10 * ev_a, ("Gamma", "X"): 10 * ev_a, ("L", "L"): 10 * ev_a,
            ("L", "X"): 5 * ev_a, ("X", "X"): 7 * ev_a}
    hw_iv = {("Gamma", "L"): 27.8 * MEV, ("Gamma", "X"): 29.9 * MEV, ("L", "L"): 29.0 * MEV,
             ("L", "X"): 29.3 * MEV, ("X", "X"): 29.9 * MEV}
    return Material(
        valleys=valleys,
        m_hh=0.50 * M0, m_lh=0.088 * M0, m_so=0.15 * M0,
        Eg0=1.423 * EV, Delta_so=0.332 * EV,
        eps_inf=10.92, eps_s=12.90, rho=5360.0, v_s=5240.0,
        hw0=35.36 * MEV,
        D_iv=D_iv, hw_iv=hw_iv,
    )


@dataclass(frozen=True)
class Sample:
    """Material + p-doping + temperature, with the doping-dependent quantities of [C21].

    Parameters
    ----------
    material : Material
    p : hole concentration = ionized acceptor density N_a^- [m^-3] (C21 Sec. III B: p = N_a^- = N_a)
    T : lattice temperature [K]
    screening : "auto" (Debye if nondegenerate, Thomas-Fermi if degenerate, C21 Eqs. 28-29),
                "debye", or "thomas_fermi"
    """
    material: Material
    p: float
    T: float = 300.0
    screening: str = "auto"

    def with_(self, **kw) -> "Sample":
        return replace(self, **kw)

    @property
    def kT(self) -> float:
        return K_B * self.T

    @property
    def eps_s(self) -> float:
        return EPS0 * self.material.eps_s

    # --- band gap -------------------------------------------------------------------
    @cached_property
    def Eg(self) -> float:
        """Doping-dependent gap, Eq. 10:  Eg = Eg0 - 3 e^2/(16 pi eps_s) sqrt(e^2 p / (eps_s kB T))."""
        es = self.eps_s
        return self.material.Eg0 - 3 * Q_E**2 / (16 * np.pi * es) * np.sqrt(Q_E**2 * self.p / (es * self.kT))

    # --- hole statistics ------------------------------------------------------------
    @cached_property
    def N_V(self) -> float:
        """Effective VB density of states, text after Eq. 58: 2 [m_h kB T/(2 pi hbar^2)]^(3/2), m_h = m_hh."""
        return 2.0 * (self.material.m_hh * self.kT / (2 * np.pi * HBAR**2)) ** 1.5

    @cached_property
    def EF_bulk(self) -> float:
        """E_F^b = E_F - E_V^b, Eq. 58 (Nilsson). Positive = Fermi level in the gap.

        E_F^b = -kT { ln(p/N_V) + (p/N_V) / [64 + 0.05524 (64 + sqrt(p/N_V)) p/N_V]^(1/4) }
        """
        u = self.p / self.N_V
        return -self.kT * (np.log(u) + u / (64 + 0.05524 * (64 + np.sqrt(u)) * u) ** 0.25)

    @cached_property
    def degenerate(self) -> bool:
        """Degenerate if E_F is within 2 kT of the VBM or inside the VB (text after Eq. 58).
        Interpretation flagged as ambiguity A8 in docs/IMPLEMENTATION_PLAN.md."""
        return self.EF_bulk < 2 * self.kT

    @cached_property
    def EF_h(self) -> float:
        """Hole Fermi energy for Pauli blocking, Eq. 44: hbar^2 (3 pi^2 p)^(2/3) / (2 m_hh).

        (The manuscript prints (3 pi^2 hbar^3 p)^(2/3)/(2 m_h), i.e. the same expression.)"""
        return HBAR**2 * (3 * np.pi**2 * self.p) ** (2 / 3) / (2 * self.material.m_hh)

    @cached_property
    def N_lh(self) -> float:
        """Light-hole density, text after Eq. 29: m_lh^1.5 / (m_hh^1.5 + m_lh^1.5) p."""
        m = self.material
        return m.m_lh**1.5 / (m.m_hh**1.5 + m.m_lh**1.5) * self.p

    # --- screening ------------------------------------------------------------------
    @cached_property
    def L_debye(self) -> float:
        """Eq. 28: sqrt(eps_s kB T / (e^2 p))."""
        return np.sqrt(self.eps_s * self.kT / (Q_E**2 * self.p))

    @cached_property
    def L_TF(self) -> float:
        """Eq. 29: sqrt(pi hbar^2 eps_s / (e^2 m_lh)) (pi / (3 N_lh))^(1/6)."""
        m_lh = self.material.m_lh
        return np.sqrt(np.pi * HBAR**2 * self.eps_s / (Q_E**2 * m_lh)) * (np.pi / (3 * self.N_lh)) ** (1 / 6)

    @cached_property
    def screening_model(self) -> str:
        if self.screening == "auto":
            return "thomas_fermi" if self.degenerate else "debye"
        return self.screening

    @cached_property
    def beta(self) -> float:
        """Inverse screening length beta = 1/L [1/m] (after Eq. 25)."""
        return 1.0 / (self.L_TF if self.screening_model == "thomas_fermi" else self.L_debye)

    # --- band bending (used in Stage D; cheap, so defined now) ----------------------
    @cached_property
    def E_bb(self) -> float:
        """Band-bending depth, Eqs. 56-57: E_bb = Eg/2 - E_F^b  [J]."""
        return 0.5 * self.Eg - self.EF_bulk

    @cached_property
    def W_bb(self) -> float:
        """Band-bending width, Eq. 59: sqrt(2 eps_s |E_bb| / (e p))  [m] (E_bb/e in volts)."""
        return np.sqrt(2 * self.eps_s * abs(self.E_bb) / Q_E / (Q_E * self.p))

    # --- electron-hole / exciton quantities (BAP, e-h scattering) ---------------------
    @cached_property
    def m_R(self) -> float:
        """Reduced mass of a Gamma electron and a heavy hole, Eq. 39."""
        me, mh = self.material.gamma.m_eff, self.material.m_hh
        return me * mh / (me + mh)

    @cached_property
    def a_B(self) -> float:
        """Exciton Bohr radius, text after Eq. 47: 4 pi hbar^2 eps_s / (e^2 m_R)."""
        return 4 * np.pi * HBAR**2 * self.eps_s / (Q_E**2 * self.m_R)

    @cached_property
    def v_B(self) -> float:
        """Exciton Bohr velocity hbar / (m_R a_B)."""
        return HBAR / (self.m_R * self.a_B)

    @cached_property
    def E_B(self) -> float:
        """Exciton Bohr energy hbar^2 / (2 m_R a_B^2)."""
        return HBAR**2 / (2 * self.m_R * self.a_B**2)

    @cached_property
    def inv_tau0(self) -> float:
        """Eq. 48: 1/tau0 = (3/64) pi Delta_exc^2 / (hbar E_B)."""
        return 3 / 64 * np.pi * self.material.Delta_exc**2 / (HBAR * self.E_B)

    def summary(self) -> dict:
        from .constants import to_ev, to_per_cm3
        return {
            "p [cm^-3]": to_per_cm3(self.p),
            "T [K]": self.T,
            "Eg [eV]": to_ev(self.Eg),
            "E_F^b [eV]": to_ev(self.EF_bulk),
            "degenerate": self.degenerate,
            "screening": self.screening_model,
            "L_screen [nm]": 1e9 / self.beta,
            "E_F^h [eV]": to_ev(self.EF_h),
            "E_bb [eV]": to_ev(self.E_bb),
            "W_bb [nm]": self.W_bb * 1e9,
            "a_B [nm]": self.a_B * 1e9,
            "E_B [meV]": to_ev(self.E_B) * 1e3,
            "1/tau0 [1/s]": self.inv_tau0,
        }
