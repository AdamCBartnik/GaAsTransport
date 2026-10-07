"""Stage E: C21 surface-emission model (gaas_mc/surface_c21.py) and its transport hook."""
import numpy as np
import pytest

from gaas_mc import bands
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import EV, HBAR, M0, NM, PS, Q_E, ev, per_cm3
from gaas_mc.excitation import photoexcite
from gaas_mc.fields import C21BandBending, UniformField
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import ALIVE, EMITTED, SURFACE, TIMEOUT, TRAPPED, Ensemble
from gaas_mc.scattering import build_mechanisms, stage_a_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.surface_c21 import EMIT, REFLECT, TRAP, C21Surface
from gaas_mc.transport import EV_SURFACE, Simulation
from gaas_mc.valleys import valley_center

MAT = gaas_chubenko2021()
G = MAT.gamma


class Rect(C21Surface):
    """Rectangular barrier of height V0 above the vacuum level, width L_b."""
    V0: float = 0.0

    def barrier(self, x):
        return self.chi + self.V0 + 0 * x


def rect_T(E, V0, L):
    """Textbook transmission through a rectangular barrier, equal masses m0."""
    if E < V0:
        kap = np.sqrt(2 * M0 * (V0 - E)) / HBAR
        return 1 / (1 + V0**2 * np.sinh(kap * L) ** 2 / (4 * E * (V0 - E)))
    q = np.sqrt(2 * M0 * (E - V0)) / HBAR
    return 1 / (1 + V0**2 * np.sin(q * L) ** 2 / (4 * E * (E - V0)))


@pytest.mark.parametrize("E_eV", [0.1, 0.3, 0.55, 0.9])
def test_rectangular_barrier_matches_analytic(E_eV):
    s = Rect(chi=0.0, material=MAT, L_b=0.5 * NM, n_slices=7)
    s.V0 = 0.5 * EV
    E = E_eV * EV
    k_in = np.sqrt(2 * M0 * E) / HBAR
    T = s.transmission(np.array([E]), np.array([k_in]), np.array([M0]), np.array([0.0]))[0]
    assert np.isclose(T, rect_T(E, s.V0, s.L_b), rtol=1e-9)


def test_mass_step_flux_conservation():
    """Flat barrier at the vacuum level: a pure m* -> m0 step. T = 4 r/(1+r)^2, r = (k2/m0)/(k1/m*),
    and therefore R = 1 - T = ((1-r)/(1+r))^2 (flux conservation)."""
    s = C21Surface(chi=0.3 * EV, material=MAT, E_b=0.0, n_slices=3)
    E = np.array([0.35, 0.5, 1.0]) * EV
    m1 = np.full(3, 0.067 * M0)
    k1 = np.sqrt(2 * m1 * E) / HBAR
    k2 = np.sqrt(2 * M0 * (E - s.chi)) / HBAR
    r = (k2 / M0) / (k1 / m1)
    T = s.transmission(E, k1, m1, np.zeros(3))
    assert np.allclose(T, 4 * r / (1 + r) ** 2, rtol=1e-10)
    assert np.all((T > 0) & (T < 1))


def test_c21_barrier_converged_and_monotone():
    s50 = C21Surface(chi=ev(0.67), material=MAT, n_slices=50)
    s400 = C21Surface(chi=ev(0.67), material=MAT, n_slices=400)
    E = ev(np.linspace(0.68, 1.2, 30))
    k = bands.k_of_E(E, G.m_eff, G.alpha)
    mv = G.m_eff * (1 + 2 * G.alpha * E)
    T50, T400 = s50.transmission(E, k, mv, 0 * E), s400.transmission(E, k, mv, 0 * E)
    assert np.allclose(T50, T400, atol=2e-4)
    assert np.all(np.diff(T400) > 0) and np.all((T400 > 0) & (T400 < 1))


def test_kpar_folding_and_valley_restriction():
    s = C21Surface(chi=ev(0.3), material=MAT)
    a = MAT.a_lat
    rng = np.random.default_rng(0)
    # the (001) surface BZ folding: K + b1 is the same point
    K = rng.normal(size=(200, 3)) * 2e9
    b1 = 2 * np.pi / a * np.array([1.0, 1.0, 0.0])
    assert np.allclose(s.fold_kpar(K), s.fold_kpar(K + b1)) and np.allclose(s.fold_kpar(K), s.fold_kpar(-K))
    assert np.all(s.fold_kpar(K) <= np.hypot(*K[:, :2].T) + 1e-6)
    # L (all four) and the in-plane X valleys keep |K_par| near pi/a*sqrt(2), 2pi/a (2.35, 4.7 eV in
    # vacuum): not emitted at any energy reachable here (the restriction is energetic, from the
    # conserved K_par; an L electron with E_kin ~ 1 eV and a low chi could in principle escape)
    n = 4000
    E = np.full(n, ev(0.6))
    for valley, eqvs in ((1, [0, 1, 2, 3]), (2, [0, 1])):
        vm = MAT.valleys[valley]
        k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, vm.m_eff, vm.alpha)[:, None]
        k[:, 2] = -np.abs(k[:, 2])
        for e in eqvs:
            eqv = np.full(n, e)
            vv = np.full(n, valley)
            Kf = k + valley_center(vv, eqv, a)
            out, info = s.interact(k, E, vv, Kf, rng)
            assert np.all(out == REFLECT) and np.all(info["T"] == 0)
    # Gamma and X[001] can be emitted
    for valley, eqv in ((0, 0), (2, 2)):
        vm = MAT.valleys[valley]
        k = np.zeros((n, 3)); k[:, 2] = -bands.k_of_E(E, vm.m_eff, vm.alpha)
        vv = np.full(n, valley)
        out, info = s.interact(k, E, vv, k + valley_center(vv, np.full(n, eqv), a), rng)
        assert np.all(info["T"] > 0) and np.any(out == EMIT)
        em = out == EMIT
        assert np.allclose(np.hypot(*info["p_vac"][em, :2].T), 0, atol=1e-30)
        assert np.allclose(info["p_vac"][em, 2] ** 2 / (2 * M0), info["E_vac_kin"][em], rtol=1e-9, atol=0)


def test_trapping_below_vacuum_level():
    s = C21Surface(chi=ev(0.67), material=MAT)
    rng = np.random.default_rng(1)
    E = ev(np.array([0.1, 0.5, 0.669, 0.671, 0.9]))
    k = np.zeros((5, 3)); k[:, 2] = -bands.k_of_E(E, G.m_eff, G.alpha)
    out, info = s.interact(k, E, np.zeros(5, int), k.copy(), rng)
    assert list(out[:3]) == [TRAP] * 3 and not np.any(out[3:] == TRAP)
    # transverse energy: above the vacuum level but eps_vac < 0 -> reflected, not trapped
    k2 = np.zeros((1, 3)); k2[0, 0] = bands.k_of_E(ev(0.70), G.m_eff, G.alpha)
    out2, info2 = s.interact(k2 * 0.99 + np.array([[0, 0, -1e7]]), ev(np.array([0.70])), np.zeros(1, int), k2, rng)
    assert info2["eps_vac"][0] < 0 and out2[0] == REFLECT


def test_analytic_bounce_in_field():
    s = Sample(MAT, per_cm3(1e19))
    Ez = 1e8
    surf = C21Surface(chi=ev(10.0), material=MAT)
    sim = Simulation(s, stage_a_mechanisms(s), field=UniformField(Ez), surface=surf)
    E = ev(np.array([0.001, 0.05]))
    k = np.zeros((2, 3)); k[:, 2] = bands.k_of_E(E, G.m_eff, G.alpha)          # moving inward
    z, k1, E1, dt, ev_ = sim.propagate(np.zeros(2), k, E, np.zeros(2, np.int8), np.full(2, 1 * PS))
    t_ret = 2 * HBAR * k[:, 2] / (Q_E * Ez)
    bounced = t_ret <= sim.dt_max
    assert bounced.any() and not bounced.all()
    assert np.all(ev_[bounced] == EV_SURFACE) and np.allclose(dt[bounced], t_ret[bounced])
    assert np.allclose(k1[bounced, 2], -k[bounced, 2]) and np.allclose(E1, E)
    # the slower-returning one is integrated and comes back with the same energy
    i = np.flatnonzero(~bounced)[0]
    assert ev_[i] == EV_SURFACE and np.isclose(dt[i], t_ret[i], rtol=1e-3)
    assert np.isclose(E1[i], E[i], rtol=1e-3)


def _bench(n=300, t_max=20 * PS, chi=0.67):
    s = Sample(MAT, per_cm3(1e19))
    f = C21BandBending(s)
    a = ModelAssumptions(depletion_scattering="bulk")
    mech = build_mechanisms(s, a, "C", field=f)
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    rng = np.random.default_rng(3)
    ens = photoexcite(s, ev(1.7), n, rng, assumptions=a)
    ens.z[:] = np.minimum(ens.z, 30 * NM)          # short test: start near the surface
    return s, f, a, mech, sm, ens, rng


def test_transport_bookkeeping_and_branching():
    s, f, a, mech, sm, ens, rng = _bench()
    surf = C21Surface(chi=ev(0.67), material=MAT)
    r = Simulation(s, mech, sm, field=f, surface=surf, t_max=20 * PS, assumptions=a).run(ens, rng)
    st = r.ensemble.status
    n = len(ens)
    assert not np.any(st == ALIVE) and not np.any(st == SURFACE)
    assert np.sum(st == EMITTED) + np.sum(st == TRAPPED) + np.sum(st == TIMEOUT) == n
    em = r.emissions
    assert len(em) == np.sum(st == EMITTED) > 0
    assert set(em.pid) == set(r.ensemble.pid[st == EMITTED])
    assert np.all(em.n_surface >= 1) and np.all(em.valley != 1)
    assert len(r.arrivals) == np.sum(r.ensemble.n_surface > 0)               # first arrivals only
    assert np.all(r.ensemble.n_surface[st == TRAPPED] >= 1)
    # E_vac = E_tot - chi > 0 and the vacuum momentum carries exactly that energy
    assert np.all(em.E_vac > 0)
    assert np.allclose(np.sum(em.p_vac**2, axis=1) / (2 * M0), em.E_vac, rtol=1e-9, atol=0)

    # branching: absorb at the first arrival, then continue the arrivals with the surface model
    s, f, a, mech, sm, ens, rng = _bench()
    r0 = Simulation(s, mech, sm, field=f, t_max=20 * PS, assumptions=a).run(ens, rng)
    e2 = Ensemble.from_arrivals(r0.arrivals)
    r1 = Simulation(s, mech, sm, field=f, surface=surf, t_max=20 * PS, assumptions=a).run(
        e2, rng, start_at_surface=True)
    st1 = r1.ensemble.status
    assert np.all(r1.ensemble.n_surface >= 1)
    assert np.sum(st1 == EMITTED) + np.sum(st1 == TRAPPED) + np.sum(st1 == TIMEOUT) == len(e2)
    assert len(r1.arrivals) == len(e2)
    assert np.allclose(r1.arrivals.t, r0.arrivals.t) and np.array_equal(r1.arrivals.pid, r0.arrivals.pid)


def test_matching_mass_option():
    """band_edge uses m* instead of the velocity mass: lower T for a nonparabolic electron, and
    identical when alpha E -> 0."""
    sb = C21Surface(chi=ev(0.67), material=MAT)                         # default: band edge
    assert sb.matching_mass == "band_edge"
    sv = C21Surface(chi=ev(0.67), material=MAT, matching_mass="velocity")
    rng = np.random.default_rng(0)
    E = ev(np.array([0.7, 0.9]))
    k = np.zeros((2, 3)); k[:, 2] = -bands.k_of_E(E, G.m_eff, G.alpha)
    Tv = sv.interact(k, E, np.zeros(2, int), k.copy(), rng)[1]["T"]
    Tb = sb.interact(k, E, np.zeros(2, int), k.copy(), rng)[1]["T"]
    assert np.all(Tb < Tv) and np.all(Tb > 0)
    with pytest.raises(ValueError):
        C21Surface(chi=ev(0.67), material=MAT, matching_mass="other")
