"""Fast engine (gaas_mc/fast) against the NumPy reference (gaas_mc/transport.py).

Deterministic pieces must agree exactly or to round-off: table searches, rates, flights, spin
tables, surface transmission. Stochastic pieces must agree in distribution: every mechanism's final
states, hot-electron relaxation, the surface model end to end, and the exact aggregation of repeated
surface bounces (brute force in the reference). CPU results must not depend on the thread count;
the GPU build (skipped without CUDA) must match the CPU build.
"""
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")

from gaas_mc import bands
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import EV, M0, NM, PS, ev, per_cm3, to_ev
from gaas_mc.depletion import LocalMechanism
from gaas_mc.excitation import photoexcite
from gaas_mc.fast import FastSimulation, testing as T
from gaas_mc.fast.engine import SURF_C21, SURF_HOST, pack_tables
from gaas_mc.fields import C21BandBending, UniformField
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import EMITTED, TIMEOUT, TRAPPED, Ensemble
from gaas_mc.scattering import build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.surface_c21 import C21Surface
from gaas_mc.transport import Simulation

MAT = gaas_chubenko2021()
G = MAT.gamma


def cuda_ok():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


def c21_sim(variant="bulk", surface=None, t_max=5 * PS, p=1e19):
    s = Sample(MAT, per_cm3(p))
    f = C21BandBending(s)
    a = ModelAssumptions(depletion_scattering=variant)
    mech = build_mechanisms(s, a, "C", field=f)
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    kw = dict(surface=surface) if surface is not None else {}
    return s, a, Simulation(s, mech, sm, field=f, t_max=t_max, assumptions=a, **kw)


@pytest.fixture(scope="module")
def local_sim():
    return c21_sim("local")


@pytest.fixture(scope="module")
def bulk_sim():
    return c21_sim("bulk")


# ------------------------------------------------------------------------------ deterministic
def test_lookup_searches_exact(local_sim):
    s, a, sim = local_sim
    rng = np.random.default_rng(0)
    g = sim.E_grid
    E = np.concatenate([ev(rng.uniform(0, 2, 50000)), g, np.nextafter(g, 0), np.nextafter(g, 1)])
    E = E[(E >= 0) & (E <= g[-1])]
    b = T.brackets(sim, E)
    assert np.array_equal(b[:, 0], b[:, 1]) and np.array_equal(b[:, 2], b[:, 3])
    tab = pack_tables(sim)
    for gas in (0, tab["hole_cdf"].shape[0] - 1):
        u = np.concatenate([rng.random(50000), tab["hole_cdf"][gas]])
        ref = np.interp(u, tab["hole_cdf"][gas], tab["hole_x"][gas])
        assert np.allclose(T.hole_energy_quantiles(sim, gas, u), ref, rtol=1e-14, atol=0)


@pytest.mark.parametrize("variant", ["bulk", "local"])
def test_rates_and_spin_tables_exact(variant, local_sim, bulk_sim):
    s, a, sim = local_sim if variant == "local" else bulk_sim
    rng = np.random.default_rng(1)
    E = ev(rng.uniform(0, 1.5, 3000))
    for zpos in (2 * sim.field.W, 0.3 * sim.field.W, 0.0):
        z = np.full(E.size, zpos)
        for v in range(3):
            R = sim.rates_at(E, v, sim.phi_index(z))
            for row, mi in enumerate(sim.mech_by_valley[v]):
                assert np.allclose(T.rates(sim, mi, E, z), R[row], rtol=1e-12, atol=0)
            vv = np.full(E.size, v)
            assert np.allclose(T.inv_tau(sim, E, vv, z), sim._inv_tau_s_at(E, vv, sim.phi_index(z)),
                               rtol=1e-12, atol=0)


@pytest.mark.parametrize("config", ["c21", "uniform_walls"])
def test_flights_exact(config, bulk_sim):
    if config == "c21":
        s, a, sim = bulk_sim
        zmax = 3 * sim.field.W
    else:
        s = Sample(MAT, per_cm3(5e17))
        a = ModelAssumptions()
        sim = Simulation(s, build_mechanisms(s, a, "C"), field=UniformField(3e6), surface="reflect",
                         z_back=200 * NM, back="reflect", assumptions=a)
        zmax = 200 * NM
    rng = np.random.default_rng(2)
    n = 20000
    valley = rng.integers(0, 3, n).astype(np.int8)
    m = np.array([x.m_eff for x in MAT.valleys])[valley]
    al = np.array([x.alpha for x in MAT.valleys])[valley]
    E = ev(rng.uniform(0.001, 0.6, n))
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, m, al)[:, None]
    z = rng.uniform(0, zmax, n)
    dt = rng.exponential(5e-15, n)
    zr, kr, Er, dr, evr = sim.propagate(z, k, E, valley, dt)
    zk, kzk, Ek, dk, evk = T.propagate(sim, z, k, E, valley, dt)
    assert np.array_equal(evr, evk)
    assert np.allclose(zk, zr, rtol=1e-11, atol=1e-21)
    assert np.allclose(kzk, kr[:, 2], rtol=1e-11, atol=1e-6 * np.abs(kr[:, 2]).max())
    assert np.allclose(Ek, Er, rtol=1e-11, atol=0) and np.allclose(dk, dr, rtol=1e-11, atol=1e-30)


@pytest.mark.parametrize("matching", ["velocity", "band_edge"])
def test_transmission_and_folding_exact(matching):
    surf = C21Surface(chi=ev(0.67), material=MAT, matching_mass=matching)
    rng = np.random.default_rng(3)
    n = 20000
    E = ev(rng.uniform(0.5, 1.5, n))
    kin = rng.uniform(0, 3e9, n)
    kin[:50] = 0
    m_in = G.m_eff * rng.uniform(1, 2, n)
    kp = rng.uniform(0, 2e9, n)
    Tr = surf.transmission(E, kin, m_in, kp)
    Tk = T.transmission(surf, E, kin, m_in, kp)
    assert np.allclose(Tk, Tr, rtol=1e-11, atol=1e-13) and np.array_equal(Tr == 0, Tk == 0)
    K = rng.normal(size=(n, 3)) * 1.5e10
    assert np.allclose(T.fold_kpar(K, MAT.a_lat), surf.fold_kpar(K), rtol=1e-14, atol=0)


# ------------------------------------------------------------------------------ distributions
def _zscore(a, b):
    """Two-sample z of the means; quantities without spread (e.g. fixed final energies) must agree
    to round-off instead."""
    sa, sb = a.std(), b.std()
    scale = max(abs(a.mean()), abs(b.mean()), 1e-300)
    if sa < 1e-9 * scale and sb < 1e-9 * scale:
        return 0.0 if abs(a.mean() - b.mean()) <= 1e-9 * scale else np.inf
    return (a.mean() - b.mean()) / np.sqrt(sa**2 / a.size + sb**2 / b.size)


@pytest.mark.parametrize("variant", ["bulk", "local"])
def test_final_states_match_reference(variant, local_sim, bulk_sim):
    s, a, sim = local_sim if variant == "local" else bulk_sim
    n = 30000
    worst = []
    for mi, m in enumerate(sim.mechanisms):
        j = 0
        vm = m
        if isinstance(m, LocalMechanism):
            j = m.depletion.n // 3                          # a depleted-region variant
            vm = m.variants[j]
        V = MAT.valleys[m.valley_from]
        for E_eV in (0.04, 0.4):
            E0 = ev(E_eV)
            if not E0 > sim.thresholds[mi]:
                continue
            rng = np.random.default_rng(10 + mi)
            E = np.full(n, E0)
            k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, V.m_eff, V.alpha)[:, None]
            kr, ar, vr = vm.scatter(k, E, rng)
            kk, ak, vk = T.final_states(sim, mi, j, k, E, seed=20 + mi)
            ar, ak = np.asarray(ar, bool), np.asarray(ak, bool)
            vr, vk = np.asarray(vr), np.asarray(vk)
            z_acc = (ar.mean() - ak.mean()) / np.sqrt(max(ar.mean() * (1 - ar.mean()), 1e-12) * 2 / n)
            assert np.array_equal(np.unique(vr[ar]), np.unique(vk[ak]))
            q = []
            for kn, acc, vn in ((kr, ar, vr), (kk, ak, vk)):
                mm = np.array([x.m_eff for x in MAT.valleys])[vn[acc]]
                aa = np.array([x.alpha for x in MAT.valleys])[vn[acc]]
                En = bands.E_of_k(np.linalg.norm(kn[acc], axis=1), mm, aa)
                c = np.sum(kn[acc] * k[acc], 1) / np.linalg.norm(kn[acc], axis=1) / np.linalg.norm(k[acc], axis=1)
                q.append((En, c))
            z = max(abs(z_acc), abs(_zscore(q[0][0], q[1][0])), abs(_zscore(q[0][1], q[1][1])))
            worst.append((z, m.name, E_eV))
    worst.sort()
    # ~140 comparisons of 3 quantities: |z| > 4.5 has probability ~ 3e-3 in total under the null
    assert worst[-1][0] < 4.5, worst[-3:]


def test_hot_electron_relaxation_matches_reference():
    s = Sample(MAT, per_cm3(1e19))
    a = ModelAssumptions()
    mech = build_mechanisms(s, a, "C")
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    sim = Simulation(s, mech, sm, surface="none", t_max=0.3 * PS, assumptions=a)
    out = {}
    for dev, n in (("ref", 20000), ("cpu", 100000)):
        rng = np.random.default_rng(31)
        E = np.full(n, ev(0.6))
        k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, G.m_eff, G.alpha)[:, None]
        ens = Ensemble.create(z=np.full(n, 1e-6), k=k, E=E, spin=1)
        r = sim.run(ens, rng) if dev == "ref" else FastSimulation(sim, "cpu").run(ens, rng)
        e = r.ensemble
        Etot = e.E + np.array([v.offset for v in MAT.valleys])[e.valley]
        out[dev] = (np.eye(3)[e.valley], Etot, e.spin.astype(float), r.n_real.sum() / n)
    for i in range(3):
        assert abs(_zscore(out["ref"][i], out["cpu"][i]) if i else
                   max(abs(_zscore(out["ref"][0][:, v], out["cpu"][0][:, v])) for v in range(3))) < 4.5
    assert abs(out["ref"][3] / out["cpu"][3] - 1) < 0.01


def test_thread_count_does_not_change_results(bulk_sim):
    import numba
    s, a, sim = bulk_sim
    res = []
    nthreads = numba.get_num_threads()
    try:
        for t in (1, min(4, nthreads)):
            numba.set_num_threads(t)
            rng = np.random.default_rng(5)
            ens = photoexcite(s, ev(1.6), 300, rng, assumptions=a)
            r = FastSimulation(sim, "cpu").run(ens, rng)
            res.append(r)
    finally:
        numba.set_num_threads(nthreads)
    for name in ("z", "t", "E", "k", "spin", "status"):
        assert np.array_equal(getattr(res[0].ensemble, name), getattr(res[1].ensemble, name))


def _surface_run(sim, dev, n, seed=5):
    s = sim.sample
    rng = np.random.default_rng(seed)
    ens = photoexcite(s, ev(1.9), n, rng, assumptions=sim.assumptions)
    ens.z[:] = np.minimum(ens.z, 60 * NM)
    return sim.run(ens, rng) if dev == "ref" else FastSimulation(sim, dev).run(ens, rng)


def test_surface_model_end_to_end_matches_reference():
    s, a, sim = c21_sim("bulk", surface=C21Surface(chi=ev(0.67), material=MAT), t_max=10 * PS)
    assert FastSimulation(sim, "cpu").mode == SURF_C21
    n = 6000
    rr, rf = _surface_run(sim, "ref", n), _surface_run(sim, "cpu", n, seed=6)
    for status in (EMITTED, TRAPPED, TIMEOUT):
        assert abs(_zscore((rr.ensemble.status == status).astype(float),
                           (rf.ensemble.status == status).astype(float))) < 4.5
    assert abs(_zscore(rr.emissions.spin.astype(float), rf.emissions.spin.astype(float))) < 4.5
    assert abs(_zscore(rr.emissions.E_vac, rf.emissions.E_vac)) < 4.5
    assert abs(_zscore(rr.emissions.E_perp, rf.emissions.E_perp)) < 4.5
    assert len(rf.arrivals) == np.sum(rf.ensemble.n_surface > 0)
    assert np.allclose(np.sum(rf.emissions.p_vac**2, 1) / (2 * M0), rf.emissions.E_vac, rtol=1e-9, atol=0)
    em = rf.emissions
    assert np.all(rf.ensemble.status[np.isin(rf.ensemble.pid, em.pid)] == EMITTED)


class HostOnlyC21(C21Surface):
    """Same physics, but a subclassed interact(): the fast engine must use the host path."""
    def interact(self, k, E, valley, K, rng, pid=None):
        return super().interact(k, E, valley, K, rng, pid)


def test_host_surface_path_matches_kernel_path():
    s, a, sim_k = c21_sim("bulk", surface=C21Surface(chi=ev(0.64), material=MAT), t_max=4 * PS)
    sim_h = Simulation(s, sim_k.mechanisms, sim_k.spin_model, field=sim_k.field,
                       surface=HostOnlyC21(chi=ev(0.64), material=MAT), t_max=4 * PS, assumptions=a)
    fh = FastSimulation(sim_h, "cpu")
    assert fh.mode == SURF_HOST
    n = 6000
    rk, rh = _surface_run(sim_k, "cpu", n), _surface_run(sim_h, "cpu", n, seed=7)
    for status in (EMITTED, TRAPPED):
        assert abs(_zscore((rk.ensemble.status == status).astype(float),
                           (rh.ensemble.status == status).astype(float))) < 4.5
    assert abs(_zscore(rk.emissions.E_vac, rh.emissions.E_vac)) < 4.5


def test_bounce_train_matches_brute_force():
    """Electrons at z = 0 moving inward with small k_z under the strong 1e19 surface field: many
    returns per flight. Reference: one analytic bounce at a time; fast engine: exact aggregation."""
    surf = C21Surface(chi=ev(0.55), material=MAT)
    s, a, sim = c21_sim("bulk", surface=surf, t_max=0.02 * PS)
    n = 4000
    rng = np.random.default_rng(8)
    E = np.full(n, ev(0.70))
    kk = bands.k_of_E(E, G.m_eff, G.alpha)
    kz = rng.uniform(2e6, 2e7, n)                          # t_ret ~ 0.02 - 0.2 fs
    phi = 2 * np.pi * rng.random(n)
    kp = np.sqrt(kk**2 - kz**2)
    k = np.column_stack([kp * np.cos(phi), kp * np.sin(phi), kz])
    ens = Ensemble.create(z=np.zeros(n), k=k, E=E, spin=1)
    brute = Simulation(s, sim.mechanisms, sim.spin_model, field=sim.field, surface=surf, t_max=sim.t_max,
                       assumptions=a, surface_bounce_aggregation=False)
    rb = brute.run(ens, np.random.default_rng(9))
    ra = sim.run(ens, np.random.default_rng(11))                 # reference, aggregated (default)
    rf = FastSimulation(sim, "cpu").run(ens, np.random.default_rng(10))
    assert ra.n_iterations < rb.n_iterations
    fb = np.mean(rb.ensemble.status == EMITTED)
    assert fb > 0.1
    for r in (ra, rf):
        f = np.mean(r.ensemble.status == EMITTED)
        assert abs(fb - f) < 4.5 * np.sqrt(2 * fb * (1 - fb) / n)
        # encounter counts are exact bookkeeping in all three: same distribution
        nb, nx = rb.ensemble.n_surface.astype(float), r.ensemble.n_surface.astype(float)
        assert abs(np.median(nb) - np.median(nx)) <= max(2.0, 0.1 * np.median(nb))
        assert abs(_zscore(rb.emissions.t, r.emissions.t)) < 4.5
    # the exact surface crossing keeps |k_z| (no drift towards grazing incidence): bounded counts
    assert rb.ensemble.n_surface.max() < 10_000


@pytest.mark.skipif(not cuda_ok(), reason="no CUDA device")
def test_gpu_matches_cpu():
    s, a, sim = c21_sim("bulk", surface=C21Surface(chi=ev(0.67), material=MAT), t_max=5 * PS)
    rc = _surface_run(sim, "cpu", 4000)
    rg = _surface_run(sim, "cuda", 4000)
    # same per-electron streams; GPU arithmetic (FMA contraction) may make histories diverge late
    for status in (EMITTED, TRAPPED, TIMEOUT):
        assert abs(np.sum(rc.ensemble.status == status) - np.sum(rg.ensemble.status == status)) <= 0.02 * 4000
    assert abs(_zscore(rc.emissions.E_vac, rg.emissions.E_vac)) < 4.5
