"""Model assumptions: every modelling choice that the references leave open, in one place.

Each field is documented in docs/MODEL_ASSUMPTIONS.md (default, alternatives, rationale,
source). Code that implements a choice reads it from a ``ModelAssumptions`` instance; nothing
is hard-wired silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

# Equivalent final-valley multiplicity Z_ij = number of distinct equivalent destination
# valleys reachable from ONE initial valley (GaAs: 4 L valleys, 3 X valleys in total).
# This differs from the total valley degeneracy for same-type transfers (L->L: 3, X->X: 2).
DEFAULT_VALLEY_MULTIPLICITY = {
    ("Gamma", "L"): 4, ("Gamma", "X"): 3,
    ("L", "Gamma"): 1, ("L", "L"): 3, ("L", "X"): 3,
    ("X", "Gamma"): 1, ("X", "L"): 4, ("X", "X"): 2,
}


@dataclass(frozen=True)
class ModelAssumptions:
    # --- photoexcitation ------------------------------------------------------------------
    initial_spin_rule: str = "per_band"
    """"per_band": C21 Eqs. 12-16 authoritative. Band i is chosen with probability K_i/sum K and
    spin +1 with probability (1+P_i)/2.  "chubenko_prose": the literal prose after Eq. 16
    (hh -> +1, lh/so -> -1, with the hh fraction (1+ESP0)/2)."""
    initial_k_direction: str = "isotropic"
    absorption_model: str = "adachi1989"
    """"adachi1989": Adachi, JAP 66, 6030 (1989) model dielectric function for intrinsic GaAs,
    unshifted by doping (band-gap narrowing is NOT applied to the optical data). Any object
    with ``absorption_coefficient(hv)`` may be passed to photoexcite() instead."""

    # --- holes and electron-hole scattering -------------------------------------------------
    hole_bands: tuple = ("hh", "lh")
    hole_statistics: str = "fermi_dirac"           # "fermi_dirac" | "maxwell_boltzmann"
    pauli_blocking: str = "fermi_dirac"
    """"fermi_dirac": the final hole state is accepted with probability 1 - f(E_h'), consistent
    with the sampled FD hole distribution.  "step_c21": reject if E_h' < E_F^h (C21 Eq. 44).
    "none": no blocking."""
    eh_electron_mass: str = "band_edge"
    """Electron mass in the reduced mass of C21 Eqs. 38-41: "band_edge" (m*, as published) or
    "velocity_mass" (m*(1 + 2 alpha E)). Final-state kinematics are exact either way."""
    eh_valleys: tuple = ("Gamma", "L", "X")         # Karkare thesis Fig. 2.4 includes all valleys
    impurity_valleys: tuple = ("Gamma", "L", "X")

    # --- valleys -------------------------------------------------------------------------
    valley_multiplicity: dict = field(default_factory=lambda: dict(DEFAULT_VALLEY_MULTIPLICITY))
    intervalley_dos_factor: str = "numerator"
    """C21 Eq. 33 prints (1 + 2 alpha_j E') in the denominator; the final-state DOS puts it in the
    numerator. "numerator" (default, DOS-consistent) or "c21_as_printed"."""
    side_valley_spin: str = "frozen"
    """Spin in L/X: "frozen" = spin preserved, zero additional relaxation. No other option is
    implemented by design: Gamma-valley EY/DP/BAP must not be applied to L/X by analogy."""

    # --- momentum scattering details ----------------------------------------------------------
    pop_angle: str = "chubenko_eq30"                 # or "screened"

    # --- transport ---------------------------------------------------------------------------
    flight_mode: str = "auto"
    """"auto": sample free flights directly from W_total(E) when the field is zero (no
    self-scattering apart from rejection steps); null-collision with a constant bound otherwise.
    "direct" requires a zero field; "self_scattering" forces the null-collision method."""
    spin_flip_at_arrival: bool = False

    def with_(self, **kw) -> "ModelAssumptions":
        return replace(self, **kw)

    def validate(self):
        allowed = {
            "initial_spin_rule": ("per_band", "chubenko_prose"),
            "initial_k_direction": ("isotropic",),
            "hole_statistics": ("fermi_dirac", "maxwell_boltzmann"),
            "pauli_blocking": ("fermi_dirac", "step_c21", "none"),
            "eh_electron_mass": ("band_edge", "velocity_mass"),
            "intervalley_dos_factor": ("numerator", "c21_as_printed"),
            "side_valley_spin": ("frozen",),
            "pop_angle": ("chubenko_eq30", "screened"),
            "flight_mode": ("auto", "direct", "self_scattering"),
        }
        for k, opts in allowed.items():
            if getattr(self, k) not in opts:
                raise ValueError(f"{k}={getattr(self, k)!r}; allowed: {opts}")
        if not set(self.hole_bands) <= {"hh", "lh"} or not self.hole_bands:
            raise ValueError("hole_bands must be a non-empty subset of ('hh', 'lh')")
        return self


DEFAULT = ModelAssumptions()
