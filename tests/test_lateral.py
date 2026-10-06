"""Lateral (x, y) displacement bookkeeping: isotropic diffusion in a homogeneous bulk, and agreement
between the reference and fast engines at emission."""
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore", DeprecationWarning)

from gaas_mc import bands
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import NM, PS, ev, per_cm3
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.transport import Simulation

MAT = gaas_chubenko2021()
G = MAT.gamma


@pytest.mark.parametrize("engine", ["ref", "cpu"])
def test_isotropic_diffusion_in_homogeneous_bulk(engine):
    """No field, no boundaries: the lateral and normal mean-square displacements must agree."""
    s = Sample(MAT, per_cm3(1e19))
    a = ModelAssumptions()
    mech = build_mechanisms(s, a, "C")
    sim = Simulation(s, mech, SpinModel(s, [m for m in mech if m.valley_from == 0]), surface="none",
                     t_max=2 * PS, assumptions=a)
    rng = np.random.default_rng(1)
    n = 8000
    E = np.full(n, ev(0.3))
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, G.m_eff, G.alpha)[:, None]
    ens = Ensemble.create(z=np.full(n, 1e-6), k=k, E=E, spin=1)
    if engine == "ref":
        r = sim.run(ens, rng)
    else:
        from gaas_mc.fast import FastSimulation
        r = FastSimulation(sim, "cpu").run(ens, rng)
    e = r.ensemble
    dz = e.z - e.z0
    m2 = [np.mean(e.x**2), np.mean(e.y**2), np.mean(dz**2)]
    assert m2[0] > (5 * NM) ** 2
    for q in m2[:2]:
        assert abs(q / m2[2] - 1) < 4.5 * np.sqrt(2 / n) * 1.5     # variance of a mean of squares
    assert abs(np.mean(e.x)) < 4.5 * np.sqrt(m2[0] / n)


def test_engines_agree_on_lateral_spread_at_emission():
    from gaas_mc.emission import simulate_emission
    kw = dict(hv_eV=1.6, n=3000, t_max_ps=10.0, thickness_nm=150, R_back=1.0, seed=4)
    rr = simulate_emission(device="reference", **kw)
    rf = simulate_emission(device="cpu", **{**kw, "seed": 5})
    for c in ("x", "y"):
        a, b = getattr(rr.emissions, c), getattr(rf.emissions, c)
        va, vb = np.mean(a**2), np.mean(b**2)
        err = np.sqrt(np.var(a**2) / a.size + np.var(b**2) / b.size)
        assert abs(va - vb) < 4.5 * err
