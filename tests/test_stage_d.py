"""Stage D: C21 band bending (Eqs. 56-62) and field transport."""
import numpy as np
import pytest

from gaas_mc import bands
from gaas_mc.constants import FS, NM, PS, Q_E, ev, per_cm3
from gaas_mc.fields import C21BandBending, PotentialField
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import stage_a_mechanisms
from gaas_mc.transport import EV_REGION, EV_SURFACE, Simulation

MAT = gaas_chubenko2021()
G = MAT.gamma


def test_band_bending_profile_eqs_61_62():
    s = Sample(MAT, per_cm3(1e19))
    f = C21BandBending(s)
    assert np.isclose(f.E_bb, s.E_bb) and np.isclose(f.W, s.W_bb)
    z = np.linspace(0, 1.5 * f.W, 301)
    assert np.isclose(f.band_edge(np.array([0.0]))[0], -s.E_bb)
    assert np.all(f.band_edge(z[z >= f.W]) == 0) and np.all(f.Ez(z[z >= f.W]) == 0)
    h = 1e-13
    dEC = (f.band_edge(z + h) - f.band_edge(z - h)) / (2 * h)
    inner = (z > 2 * h) & (z < f.W - 2 * h)
    assert np.allclose(f.Ez(z[inner]), dEC[inner] / Q_E, rtol=1e-6)               # Eq. 60
    assert np.isclose(f.Ez(np.array([0.0]))[0], 2 * s.E_bb / (Q_E * s.W_bb))      # Eq. 62


def ballistic(p, dt_fs):
    s = Sample(MAT, per_cm3(p))
    f = C21BandBending(s)
    sim = Simulation(s, stage_a_mechanisms(s), field=f, dt_max_field=dt_fs * FS)
    rng = np.random.default_rng(0)
    n = 300
    E0 = np.full(n, ev(0.05))
    u = bands.random_unit_vectors(n, rng); u[:, 2] = -np.abs(u[:, 2])
    k = u * bands.k_of_E(E0, G.m_eff, G.alpha)[:, None]
    z0 = np.full(n, f.W)
    z1, k1, E1, du, evt = sim.propagate(z0, k, E0, np.zeros(n, np.int8), np.full(n, 20 * PS))
    assert np.all(evt == EV_SURFACE) and np.allclose(z1, 0)
    assert np.allclose(k1[:, :2], k[:, :2])                  # transverse momentum untouched
    return np.abs(E1 + f.band_edge(z1) - E0 - f.band_edge(z0)).max()


def test_energy_conservation_and_second_order_convergence():
    e1 = ballistic(1e19, 1.0)
    e025 = ballistic(1e19, 0.25)
    assert e025 < ev(0.2e-3)                                 # < 0.2 meV at the default step
    assert 10 < e1 / e025 < 22                               # ~16 for second order


def test_region_boundary_stops_and_restarts():
    s = Sample(MAT, per_cm3(1e18))
    f = C21BandBending(s)
    sim = Simulation(s, stage_a_mechanisms(s), field=f)
    assert sim.flight_mode == "hybrid"
    n = 50
    E = np.full(n, ev(0.1))
    k = np.tile([0, 0, -bands.k_of_E(ev(0.1), G.m_eff, G.alpha)], (n, 1))
    z = np.full(n, 3 * f.W)
    z1, k1, E1, du, evt = sim.propagate(z, k, E, np.zeros(n, np.int8), np.full(n, 10 * PS))
    assert np.all(evt == EV_REGION) and np.allclose(z1, f.W)
    v = bands.speed_of_E(ev(0.1), G.m_eff, G.alpha)
    assert np.allclose(du, 2 * f.W / v)
    assert np.all(sim.in_field(z1, k1, E1, np.zeros(n, np.int8)))      # moving inward: inside now
    z2, *_ , evt2 = sim.propagate(z1, k1, E1, np.zeros(n, np.int8), np.full(n, 10 * PS))
    assert np.all(evt2 == EV_SURFACE)


@pytest.mark.parametrize("bending", [False, True])
def test_boltzmann_density_is_stationary_in_closed_slab(bending):
    """Field + scattering + walls: in a closed slab (reflecting surface and back wall) the Boltzmann
    density n(z) ~ exp(-E_C(z)/kT) must be stationary. The ensemble starts from that density; any bias
    in the field integrator, the region boundary, or wall reflections makes it drift. (Regression:
    image-field reflection inside the field region; stop-and-restart reflection outside it.)"""
    from gaas_mc.fields import NoField
    s = Sample(MAT, per_cm3(1.5e17))
    f = C21BandBending(s, E_bb=ev(0.05), W_bb=100 * NM) if bending else NoField()
    L = 200 * NM
    zz = np.linspace(0, L, 4001)
    w = np.exp(-f.band_edge(zz) / s.kT)
    cdf = np.cumsum(w); cdf /= cdf[-1]
    rng = np.random.default_rng(1)
    n = 3000
    E = rng.gamma(1.5, s.kT, n)
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, G.m_eff, G.alpha)[:, None]
    ens = Ensemble.create(z=np.interp(rng.random(n), cdf, zz), k=k, E=E, spin=1)
    ts = np.linspace(2, 20, 10) * PS
    r = Simulation(s, stage_a_mechanisms(s), field=f, surface="reflect", z_back=L, back="reflect",
                   t_max=20 * PS, snapshot_times=ts).run(ens, rng)
    z = r.snapshots.z.ravel()
    z = z[np.isfinite(z)]
    assert z.size == n * ts.size and z.min() >= 0 and z.max() <= L
    edges = np.linspace(0, L, 9)
    h, _ = np.histogram(z, edges)
    P = np.array([np.trapezoid(w[(zz >= a) & (zz <= b)], zz[(zz >= a) & (zz <= b)])
                  for a, b in zip(edges[:-1], edges[1:])])
    dev = h / (P / P.sum() * z.size) - 1
    assert np.max(np.abs(dev)) < 0.07
    assert abs(dev[0]) < 0.04                           # the surface bin, where the earlier bias showed


def test_user_potential_equivalent_to_c21_profile():
    s = Sample(MAT, per_cm3(1e18))
    f = C21BandBending(s)
    pf = PotentialField(f.band_edge, z_max=f.W)
    z = np.linspace(0.01, 0.99, 50) * f.W
    assert np.allclose(pf.Ez(z), f.Ez(z), rtol=1e-6)


def test_arrivals_record_surface_band_offset():
    s = Sample(MAT, per_cm3(1e19))
    f = C21BandBending(s)
    rng = np.random.default_rng(4)
    n = 200
    E = np.full(n, ev(0.04))
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, G.m_eff, G.alpha)[:, None]
    ens = Ensemble.create(z=np.full(n, 0.5 * f.W), k=k, E=E, spin=1)
    a = Simulation(s, stage_a_mechanisms(s), field=f, t_max=5 * PS).run(ens, rng).arrivals
    assert len(a) > 0.9 * n
    assert np.isclose(a.band_edge_at_surface, -s.E_bb)
    assert a.E.mean() > 0.5 * s.E_bb                        # accelerated by the band bending
