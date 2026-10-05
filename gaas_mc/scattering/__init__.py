"""Momentum-scattering mechanisms. See base.py for the interface."""
from .base import Mechanism
from .acoustic import AcousticPhonon
from .polar_optical import PolarOptical


def stage_a_mechanisms(sample, pop_angle="chubenko_eq30"):
    """Gamma valley only: acoustic + screened POP absorption/emission ([C21] Eqs. 23-32)."""
    return [
        AcousticPhonon(sample, 0),
        PolarOptical(sample, emission=False, valley_from=0, angle=pop_angle),
        PolarOptical(sample, emission=True, valley_from=0, angle=pop_angle),
    ]


__all__ = ["Mechanism", "AcousticPhonon", "PolarOptical", "stage_a_mechanisms"]
