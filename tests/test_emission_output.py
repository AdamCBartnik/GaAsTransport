"""ParticleGroup output of emitted electrons (gaas_mc/emission.py)."""
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore", DeprecationWarning)

from gaas_mc.constants import C_LIGHT, M0, Q_E, ev
from gaas_mc.emission import simulate_emission, to_particle_group


@pytest.fixture(scope="module")
def run():
    # thin layer + short time: a fast run on the reference engine
    return simulate_emission(hv_eV=1.6, n=1500, t_max_ps=5.0, thickness_nm=100, R_back=1.0,
                             device="reference", seed=3)


def test_particle_group_convention_and_units(run):
    pg, em = run.particle_group, run.emissions
    assert len(pg) == len(em) > 20
    p_si = em.p_vac
    # momenta in eV/c; z -> -z, p_z -> -p_z: the beam leaves along +z
    assert np.allclose(pg.px, p_si[:, 0] * C_LIGHT / Q_E) and np.allclose(pg.py, p_si[:, 1] * C_LIGHT / Q_E)
    assert np.allclose(pg.pz, -p_si[:, 2] * C_LIGHT / Q_E) and np.all(pg.pz > 0)
    assert np.all(pg.z == 0) and np.all(pg.x == 0) and np.all(pg.y == 0)
    assert np.allclose(pg.t, em.t)
    # kinetic energy in vacuum (nonrelativistic, < 1 eV) equals the record's E_vac
    KE = (pg.px**2 + pg.py**2 + pg.pz**2) / (2 * M0 * C_LIGHT**2 / Q_E)        # eV
    assert np.allclose(KE, em.E_vac / Q_E, rtol=1e-6)
    assert pg.species == "electron" and np.all(pg.status == 1)
    assert np.isclose(pg.charge, Q_E * len(pg))


def test_spot_and_charge(run):
    rng = np.random.default_rng(0)
    pg = to_particle_group(run.emissions, sigma_xy=1e-3, rng=rng, total_charge=1e-12)
    assert np.isclose(pg.charge, 1e-12)
    assert abs(np.std(pg.x) / 1e-3 - 1) < 0.25 and abs(np.std(pg.y) / 1e-3 - 1) < 0.25


def test_summary_numbers(run):
    assert np.isclose(run.qe, run.absorbed_fraction * run.n_emitted / run.n_generated)
    assert np.isclose(run.esp, run.emissions.spin.mean())
    assert 0 < run.absorbed_fraction < 1
