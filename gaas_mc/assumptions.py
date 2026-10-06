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
    absorption_model: str = "casey1975+adachi1989"
    """"casey1975+adachi1989" (default): measured p-type near-edge absorption of Casey, Sell & Wecht,
    JAP 46, 250 (1975), interpolated in hv and log10(p), blended into Adachi (1989) over 1.55-1.592 eV.
    "casey1975": near-edge data only (1.31-1.59 eV). "adachi1989": Adachi MDF alone (comparison;
    refuses hv < 1.42 eV). Any object with ``absorption_coefficient(hv)`` may be passed to
    photoexcite() instead."""

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

    # --- depletion region (band-bending field) ---------------------------------------------------
    depletion_scattering: str = "local"
    """"local" (default): hole-density-dependent processes use the local mobile-hole population
    p(z) from the global Fermi level and the local valence-band shift (gaas_mc/depletion.py):
    e-h scattering, BAP, and the screening of impurity (and POP) scattering. N_A^- stays at the
    dopant density. "bulk": bulk rates everywhere, as C21 assume for Fig. 18; the C21-compatible
    baseline used to reproduce C21's published results. Both modes are frozen (no further
    transport-model changes until the C21 end-to-end benchmark is complete)."""
    depletion_screening_cap: object = "impurity_spacing"
    """Upper bound on the local screening length, which otherwise diverges as p(z) -> 0. An
    approximate finite-impurity-spacing / third-body cutoff, not part of Brooks-Herring.
    "impurity_spacing" (default; acceptor Wigner-Seitz radius); "band_bending_width" (W_bb) or a
    length [m] only as sensitivity tests. Always max'ed with the bulk length (bulk unchanged)."""
    depletion_pop_screening: bool = True
    """Use the local hole screening in the screened POP rate (C21 Eq. 25 contains the hole beta)."""

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
            "depletion_scattering": ("local", "bulk"),
        }
        for k, opts in allowed.items():
            if getattr(self, k) not in opts:
                raise ValueError(f"{k}={getattr(self, k)!r}; allowed: {opts}")
        cap = self.depletion_screening_cap
        if not (cap in ("impurity_spacing", "band_bending_width") or
                (isinstance(cap, (int, float)) and cap > 0)):
            raise ValueError(f"depletion_screening_cap={cap!r}")
        if not set(self.hole_bands) <= {"hh", "lh"} or not self.hole_bands:
            raise ValueError("hole_bands must be a non-empty subset of ('hh', 'lh')")
        return self


DEFAULT = ModelAssumptions()
