"""Pinned-front / neutral-reservoir rear depletion electrostatics and flights."""
import warnings

import numpy as np
import pytest

from gaas_mc import bands
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.back import PartialReflector
from gaas_mc.constants import Q_E, ev, per_cm3
from gaas_mc.fields import C21BandBending
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import build_mechanisms
from gaas_mc.transport import Simulation, EV_BACK, EV_SURFACE


@pytest.fixture
def sample():
    return Sample(gaas_chubenko2021(), per_cm3(1e19))


@pytest.mark.parametrize("ratio", [0.01, 0.5, 0.999999, 1.0, 2.0])
def test_poisson_boundary_values_and_force(sample, ratio):
    d = ratio * sample.W_bb
    f = C21BandBending(sample, thickness=d)
    np.testing.assert_allclose(f.band_edge(np.array([0, d])) / sample.E_bb,
                               [-1, 0], atol=1e-14, rtol=0)
    z = np.linspace(0.1, 0.9, 101) * min(d, sample.W_bb)
    # Gauss's law: dE/dz = rho/epsilon = -e*N_A/epsilon.
    slope = np.gradient(f.Ez(z), z)
    np.testing.assert_allclose(slope / (-Q_E * sample.p / sample.eps_s), 1, rtol=1e-9)
    h = min(d, sample.W_bb) * 1e-5
    derivative = (f.band_edge(z+h) - f.band_edge(z-h)) / (2*h)
    np.testing.assert_allclose(derivative / (Q_E * f.Ez(z)), 1, rtol=1e-8)
    assert np.all(f.Ez(np.linspace(0, d, 100)) >= 0)


def test_thick_limit_and_transition(sample):
    old = C21BandBending(sample)
    z = np.linspace(0, 3 * old.W, 200)
    for d in (old.W, 3 * old.W):
        new = C21BandBending(sample, thickness=d)
        np.testing.assert_array_equal(new.Ez(z), old.Ez(z))
        np.testing.assert_array_equal(new.band_edge(z), old.band_edge(z))
    near = C21BandBending(sample, thickness=old.W * (1 - 1e-7))
    z = np.linspace(0, near.thickness, 100)
    np.testing.assert_allclose(near.band_edge(z) / old.E_bb, old.band_edge(z) / old.E_bb,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(near.Ez(z) / old.Ez(0), old.Ez(z) / old.Ez(0),
                               atol=1e-12, rtol=0)


@pytest.mark.parametrize("d", [0, -1e-9, np.nan, np.inf])
def test_invalid_thickness(sample, d):
    with pytest.raises(ValueError, match="thickness"):
        C21BandBending(sample, thickness=d)


def make_sim(sample, ratio=0.5):
    d = ratio * sample.W_bb
    f = C21BandBending(sample, thickness=d)
    a = ModelAssumptions(depletion_scattering="bulk")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sim = Simulation(sample, build_mechanisms(sample, a, "A"), field=f,
                         z_back=d, back=PartialReflector(1), assumptions=a)
    return sim


def test_field_requires_matching_back_boundary(sample):
    sim = make_sim(sample)
    with pytest.raises(ValueError, match="thickness"):
        Simulation(sample, sim.mechanisms, field=sim.field, z_back=2 * sim.z_back)


@pytest.mark.parametrize("device", ["reference", "cpu", "cuda"])
def test_thin_driver_runs_with_local_scattering(device):
    from gaas_mc.emission import simulate_emission, _cuda_available
    if device == "cuda" and not _cuda_available():
        pytest.skip("CUDA unavailable")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        run = simulate_emission(1.6, n=80, thickness_nm=5, R_back=0.5,
                                depletion_scattering="local", device=device,
                                t_max_ps=0.1, seed=42)
    assert run.n_emitted > 0
    assert np.isfinite(run.qe) and 0 < run.qe < run.absorbed_fraction
    np.testing.assert_allclose(run.result.arrivals.band_edge_at_surface / Q_E,
                               -Sample(gaas_chubenko2021(), per_cm3(1e19)).E_bb / Q_E)


@pytest.mark.parametrize("ratio", [0.1, 0.5, 0.99])
def test_reference_and_compiled_flights_conserve_energy(sample, ratio):
    from gaas_mc.fast.testing import propagate
    sim = make_sim(sample, ratio)
    # Outward flights to each boundary; rear-directed electrons have enough energy to arrive.
    z = sim.z_back * np.array([0.3, 0.9])
    E = ev(np.array([0.15, 0.3]))
    v = np.zeros(2, dtype=np.int8)
    g = sample.material.gamma
    k = np.zeros((2, 3))
    k[:, 2] = bands.k_of_E(E, g.m_eff, g.alpha) * [-1, 1]
    dt = np.full(2, 1e-12)
    zr, kr, er, tr, events = sim.propagate(z, k, E, v, dt)
    zc, kzc, ec, tc, eventc = propagate(sim, z, k, E, v, dt)
    np.testing.assert_array_equal(events, [EV_SURFACE, EV_BACK])
    np.testing.assert_array_equal(eventc, events)
    np.testing.assert_allclose(zc / sim.z_back, zr / sim.z_back, atol=1e-12)
    np.testing.assert_allclose(ec / Q_E, er / Q_E, atol=1e-8)
    total = (E + sim.field.band_edge(z)) / Q_E
    for energy, pos in ((er, zr), (ec, zc)):
        np.testing.assert_allclose((energy + sim.field.band_edge(pos)) / Q_E,
                                   total, atol=5e-5, rtol=0)


# ------------------------------------------------------------- rear = "vacuum" (pinned both sides)
@pytest.mark.parametrize("ratio", [0.01, 0.5, 1.0, 1.999999, 2.0, 3.0])
def test_vacuum_rear_poisson(sample, ratio):
    d = ratio * sample.W_bb
    f = C21BandBending(sample, thickness=d, rear="vacuum")
    assert f.z_max == d and f.fully_depleted == (d < 2 * sample.W_bb)
    np.testing.assert_allclose(f.band_edge(np.array([0, d])) / sample.E_bb, [-1, -1], atol=1e-14, rtol=0)
    z = np.linspace(0, d, 401)
    np.testing.assert_allclose(f.band_edge(z) / sample.E_bb, f.band_edge(d - z) / sample.E_bb,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(f.Ez(z) / f.Ez(0), -f.Ez(d - z) / f.Ez(0), atol=1e-12, rtol=0)
    # Depleted regions (within W of a surface) carry the acceptor charge; Gauss's law there.
    zz = np.linspace(0.05, 0.95, 101) * min(d / 2, sample.W_bb)
    for zs in (zz, d - zz):
        slope = np.gradient(f.Ez(zs), zs)
        np.testing.assert_allclose(slope / (-Q_E * sample.p / sample.eps_s), 1, rtol=1e-6)
        h = min(d, sample.W_bb) * 1e-5
        derivative = (f.band_edge(zs + h) - f.band_edge(zs - h)) / (2 * h)
        np.testing.assert_allclose(derivative / (Q_E * f.Ez(zs)), 1, rtol=1e-6)
    assert np.all(f.Ez(z[z < d / 2]) >= 0) and np.all(f.Ez(z[z > d / 2]) <= 0) and f.Ez(0) > 0


def test_vacuum_rear_transition_and_thick_limit(sample):
    W = sample.W_bb
    z = np.linspace(0, 2 * W, 300)
    below = C21BandBending(sample, thickness=2 * W * (1 - 1e-9), rear="vacuum")
    at = C21BandBending(sample, thickness=2 * W, rear="vacuum")
    np.testing.assert_allclose(below.band_edge(z) / sample.E_bb, at.band_edge(z) / sample.E_bb,
                               atol=1e-7, rtol=0)
    # Thick layer: the front half is the bulk C21 profile.
    thick = C21BandBending(sample, thickness=5 * W, rear="vacuum")
    bulk = C21BandBending(sample)
    z = np.linspace(0, 2.5 * W, 300)
    np.testing.assert_array_equal(thick.band_edge(z), bulk.band_edge(z))
    np.testing.assert_array_equal(thick.Ez(z), bulk.Ez(z))


def test_rear_option_validation(sample):
    with pytest.raises(ValueError, match="rear"):
        C21BandBending(sample, thickness=1e-8, rear="metal")
    with pytest.raises(ValueError, match="thickness"):
        C21BandBending(sample, rear="vacuum")
    f = C21BandBending(sample, thickness=0.5 * sample.W_bb, rear="vacuum")
    # Field pushes into the back: only a non-reflecting back boundary is supported.
    with pytest.raises(NotImplementedError):
        Simulation(sample, build_mechanisms(sample, ModelAssumptions(), "A"), field=f,
                   z_back=f.thickness, back=PartialReflector(0.5))


def make_vacuum_sim(sample, ratio):
    d = ratio * sample.W_bb
    f = C21BandBending(sample, thickness=d, rear="vacuum")
    a = ModelAssumptions(depletion_scattering="bulk")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return Simulation(sample, build_mechanisms(sample, a, "A"), field=f,
                          z_back=d, back=PartialReflector(0), assumptions=a)


@pytest.mark.parametrize("ratio", [0.3, 1.5, 3.0])
def test_vacuum_rear_flights_conserve_energy(sample, ratio):
    from gaas_mc.fast.testing import propagate
    sim = make_vacuum_sim(sample, ratio)
    d = sim.z_back
    # Front-half electron heading out the front; rear-half electron accelerated into the back.
    z = d * np.array([0.3, 0.7])
    E = ev(np.array([0.05, 0.05]))
    v = np.zeros(2, dtype=np.int8)
    g = sample.material.gamma
    k = np.zeros((2, 3))
    k[:, 2] = bands.k_of_E(E, g.m_eff, g.alpha) * [-1, 1]
    dt = np.full(2, 1e-11)
    zr, kr, er, tr, events = sim.propagate(z, k, E, v, dt)
    zc, kzc, ec, tc, eventc = propagate(sim, z, k, E, v, dt)
    np.testing.assert_array_equal(events, [EV_SURFACE, EV_BACK])
    np.testing.assert_array_equal(eventc, events)
    np.testing.assert_allclose(zc / d, zr / d, atol=1e-12)
    np.testing.assert_allclose(ec / Q_E, er / Q_E, atol=1e-8)
    # Slow electrons spend longest in the field: second-order Verlet error ~0.1 meV at the
    # default step (the documented crossing accuracy), shrinking 4x per halved step.
    total = (E + sim.field.band_edge(z)) / Q_E
    for energy, pos in ((er, zr), (ec, zc)):
        np.testing.assert_allclose((energy + sim.field.band_edge(pos)) / Q_E, total, atol=1.5e-4, rtol=0)


@pytest.mark.parametrize("device", ["reference", "cpu", "cuda"])
@pytest.mark.parametrize("thickness_nm", [5, 30])
def test_vacuum_rear_driver(device, thickness_nm):
    from gaas_mc.emission import simulate_emission, _cuda_available
    if device == "cuda" and not _cuda_available():
        pytest.skip("CUDA unavailable")
    with pytest.warns(UserWarning, match="R_back"):
        run = simulate_emission(1.6, chi_eV=0.05, n=200, thickness_nm=thickness_nm, R_back=0.5,
                                rear="vacuum", depletion_scattering="local", device=device,
                                t_max_ps=1.0, seed=42)
    assert run.settings["rear"] == "vacuum" and run.settings["R_back"] == 0.0
    assert run.result.ensemble.n_back.max() <= 1          # no reflections at the rear
    assert run.n_emitted > 0 and 0 < run.qe < run.absorbed_fraction


def test_rear_ignored_for_semi_infinite():
    from gaas_mc.emission import simulate_emission
    run = simulate_emission(1.6, n=50, rear="vacuum", device="reference", t_max_ps=0.05, seed=1)
    assert run.settings["rear"] is None


@pytest.mark.parametrize("rear, ratio", [("bulk", 0.3), ("bulk", 3.0), ("vacuum", 0.3),
                                         ("vacuum", 1.9), ("vacuum", 2.0), ("vacuum", 4.0)])
def test_effective_band_bending_height(sample, rear, ratio):
    d = ratio * sample.W_bb
    f = C21BandBending(sample, thickness=d, rear=rear)
    U = f.band_edge(np.linspace(0, d, 20001))
    np.testing.assert_allclose(f.E_bb_eff / sample.E_bb, (U.max() - U[0]) / sample.E_bb, atol=1e-8, rtol=0)
    assert C21BandBending(sample).E_bb_eff == sample.E_bb


def test_run_exposes_field():
    from gaas_mc.emission import simulate_emission
    run = simulate_emission(1.6, n=20, thickness_nm=15, rear="vacuum", device="reference",
                            t_max_ps=0.02, seed=1)
    s = Sample(gaas_chubenko2021(), per_cm3(1e19))
    assert run.field.rear == "vacuum" and run.field.thickness == pytest.approx(15e-9)
    np.testing.assert_allclose(run.field.E_bb_eff, s.E_bb * (15e-9)**2 / (4 * s.W_bb**2), rtol=1e-12)
