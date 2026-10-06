"""Back boundary of a finite GaAs layer, 0 < z < d (z = d: GaAs/substrate interface).

Minimal finite-slab model (Stage F). The whole layer is GaAs with the existing physics; the
substrate is not transported. When an electron reaches z = d, the transport asks the back-boundary
model once per encounter whether it is reflected back into the GaAs (specularly: k_z -> -k_z, with
energy, k_x, k_y, valley, equivalent valley and spin unchanged) or lost into the substrate (status
BACK, terminated).

Interface, so that a physical GaAs/AlGaAs interface model can replace PartialReflector later:

    is_back_model = True
    interact(k, E, valley, rng, pid=None) -> bool array, True = reflected

Use with transport.Simulation(..., z_back=d, back=PartialReflector(R_back)). The fast engine
evaluates PartialReflector inside its kernel (gaas_mc/fast); other back models currently need the
reference engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from .backend import xp_of


@dataclass
class PartialReflector:
    """Energy- and angle-independent back boundary: reflected with probability R_back, lost with
    probability 1 - R_back (one random decision per encounter). R_back = 0: fully absorbing
    substrate; R_back = 1: perfectly reflecting back surface."""
    R_back: float
    is_back_model: bool = True

    def __post_init__(self):
        if not 0.0 <= float(self.R_back) <= 1.0:
            raise ValueError(f"R_back = {self.R_back} must lie in [0, 1]")
        self.R_back = float(self.R_back)

    def interact(self, k, E, valley, rng, pid=None):
        if self.R_back >= 1.0:
            return xp_of(E).ones(E.size, bool)
        if self.R_back <= 0.0:
            return xp_of(E).zeros(E.size, bool)
        return rng.random(E.size) < self.R_back
