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
