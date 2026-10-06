"""Momentum-scattering mechanisms. See base.py for the interface.

``build_mechanisms(sample, assumptions, stage)`` assembles the standard sets:
    "A": Gamma valley; acoustic + POP absorption/emission
    "B": A + ionized impurity + electron-hole (hh, lh) in Gamma
    "C": all valleys; acoustic, POP, impurity, e-h in each valley listed in the assumptions,
         plus intervalley absorption/emission between every valley pair with D_ij in Table I
Drop entries from the returned list to disable individual mechanisms.

With ``field`` (a band-bending field with finite extent z_max) and
assumptions.depletion_scattering == "local", the hole-density-dependent mechanisms (e-h, ionized
impurity screening, and POP screening if enabled) are returned as depletion.LocalMechanism
objects holding one variant per local potential (gaas_mc/depletion.py).
"""
import numpy as np

from ..assumptions import DEFAULT
from ..depletion import DepletionModel, LocalMechanism
from ..holes import HoleGas
from ..material import VALLEY_NAMES
from .acoustic import AcousticPhonon
from .base import Mechanism
from .electron_hole import ElectronHole
from .impurity import IonizedImpurity
from .intervalley import Intervalley
from .polar_optical import PolarOptical


def stage_a_mechanisms(sample, pop_angle="chubenko_eq30"):
    """Gamma valley only: acoustic + screened POP absorption/emission ([C21] Eqs. 23-32)."""
    return [
        AcousticPhonon(sample, 0),
        PolarOptical(sample, emission=False, valley_from=0, angle=pop_angle),
        PolarOptical(sample, emission=True, valley_from=0, angle=pop_angle),
    ]


def build_mechanisms(sample, assumptions=DEFAULT, stage="C", holes=None, field=None):
    a = assumptions.validate()
    if stage not in ("A", "B", "C"):
        raise ValueError(stage)
    valleys = [0] if stage in ("A", "B") else [0, 1, 2]
    if holes is None and stage != "A":
        holes = HoleGas(sample, a.hole_bands, a.hole_statistics)
    depl = None
    if (field is not None and a.depletion_scattering == "local" and not field.is_zero
            and 0 < getattr(field, "z_max", np.inf) < np.inf):
        if holes is None:
            holes = HoleGas(sample, a.hole_bands, a.hole_statistics)
        depl = DepletionModel(sample, field, holes, a.depletion_screening_cap, a.depletion_pop_screening)

    def local(factory):
        """factory(local_sample, local_holes) -> mechanism; wrapped if the depletion model is on."""
        if depl is None:
            return factory(sample, holes)
        return LocalMechanism([factory(ls, g) for ls, g in zip(depl.samples, depl.gas)], depl)

    mech = []
    for v in valleys:
        name = VALLEY_NAMES[v]
        mech.append(AcousticPhonon(sample, v))
        for em in (False, True):
            if depl is not None and a.depletion_pop_screening:
                mech.append(local(lambda ls, g, em=em, v=v: PolarOptical(ls, emission=em, valley_from=v,
                                                                         angle=a.pop_angle)))
            else:
                mech.append(PolarOptical(sample, emission=em, valley_from=v, angle=a.pop_angle))
        if stage == "A":
            continue
        if name in a.impurity_valleys:      # ionized acceptors stay at the dopant density
            mech.append(local(lambda ls, g, v=v: IonizedImpurity(ls, v, N_imp=sample.p)))
        if name in a.eh_valleys:
            mech += [local(lambda ls, g, b=b, v=v: ElectronHole(ls, g, b, v, pauli=a.pauli_blocking,
                                                               mass_model=a.eh_electron_mass))
                     for b in a.hole_bands]
    if stage == "C":
        mat = sample.material
        for i in valleys:
            for j in valleys:
                ni, nj = VALLEY_NAMES[i], VALLEY_NAMES[j]
                if i == j == 0:
                    continue
                if (ni, nj) not in mat.D_iv and (nj, ni) not in mat.D_iv:
                    continue
                Z = a.valley_multiplicity[(ni, nj)]
                for em in (False, True):
                    mech.append(Intervalley(sample, i, j, em, Z, a.intervalley_dos_factor))
    return mech


__all__ = ["Mechanism", "AcousticPhonon", "PolarOptical", "IonizedImpurity", "ElectronHole",
           "Intervalley", "stage_a_mechanisms", "build_mechanisms"]
