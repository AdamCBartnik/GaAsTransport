"""Depletion-region scattering: local mobile holes, local screening, valley identity."""
import numpy as np
import pytest

from gaas_mc import bands
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import NM, PS, ev, per_cm3
from gaas_mc.depletion import LocalMechanism
from gaas_mc.fields import C21BandBending
from gaas_mc.holes import HoleGas, fermi_integral_half
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import AcousticPhonon, IonizedImpurity, PolarOptical, build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.transport import Simulation
from gaas_mc.valleys import choose_equivalent_valley, valley_center

MAT = gaas_chubenko2021()
G = MAT.gamma


def depl_for(p, **kw):
    s = Sample(MAT, per_cm3(p))
    f = C21BandBending(s)
    mech = build_mechanisms(s, ModelAssumptions(**kw), "C", field=f)
    loc = [m for m in mech if isinstance(m, LocalMechanism)]
    return s, f, mech, loc[0].depletion


@pytest.mark.parametrize("p", [5e17, 1e19])
def test_local_hole_density_global_fermi_level(p):
    s, f, mech, d = depl_for(p)
    hg = HoleGas(s)
    for j in (0, d.n // 3, d.n - 1):
        expect = fermi_integral_half(hg.eta + d.phi[j] / s.kT) / fermi_integral_half(hg.eta)
        assert np.isclose(d.p[j] / hg.p, expect, rtol=1e-9)
    assert d.p[0] / hg.p < 1e-9                         # fully depleted at the surface
    assert np.isclose(d.p[-1], hg.p) and np.isclose(d.beta[-1], s.beta)
    assert np.all(np.diff(d.p) > 0)                     # monotonic toward the bulk
    z = np.array([0.0, 0.5 * f.W, f.W, 3 * f.W])
    fi = d.frac_index(z)
    assert fi[-1] == d.bulk_index and fi[-2] == d.bulk_index and fi[0] == 0


def test_bulk_row_equals_bulk_mechanisms_and_ions_fixed():
    s, f, mech, d = depl_for(5e17)
    E = ev(np.linspace(0.001, 1.0, 50))
    plain = build_mechanisms(s, ModelAssumptions(), "C")
    by_name = {m.name: m for m in plain}
    for m in mech:
        if isinstance(m, LocalMechanism) and "eh_" not in m.name:
            assert np.allclose(m.variants[-1].rate(E), by_name[m.name].rate(E), rtol=1e-12)
    ii = [m for m in mech if m.name == "impurity[Gamma]"][0]
    assert all(np.isclose(v.N, s.p) for v in ii.variants)          # N_A^- never depleted


def test_screening_cap_options():
    s, f, mech, d = depl_for(5e17)
    a_ws = (3 / (4 * np.pi * s.p)) ** (1 / 3)
    assert np.isclose(d.L_cap, max(1 / s.beta, a_ws))
    assert np.all(d.beta >= 1 / d.L_cap * (1 - 1e-12))
    s2, f2, mech2, d2 = depl_for(5e17, depletion_screening_cap="band_bending_width")
    assert np.isclose(d2.L_cap, max(1 / s.beta, f2.W)) and d2.beta[0] < d.beta[0]
    # uncapped screening follows the FD susceptibility: Debye-like sqrt(p) when nondegenerate
    j = d.n // 2
    assert np.isclose(d.beta_uncapped[j] / s.beta, np.sqrt(d.gas[j].susceptibility / d.gas[-1].susceptibility))


def test_bap_vanishes_where_holes_are_depleted():
    s, f, mech, d = depl_for(1e19)
    E = ev(np.array([0.05, 0.3]))
    gam = lambda j: [m.variants[j] if isinstance(m, LocalMechanism) else m for m in mech if m.valley_from == 0]
    bulk = SpinModel(d.samples[-1], gam(-1)).bap(E)
    surf = SpinModel(d.samples[0], gam(0)).bap(E)
    assert np.all(surf < 1e-9 * bulk)
    assert np.allclose(SpinModel(s, gam(-1)).bap(E), bulk)


def test_variant_choice_reproduces_interpolated_rate():
    s, f, mech, d = depl_for(1e19)
    m = [x for x in mech if x.name == "eh_hh[Gamma]"][0]
    sim = Simulation(s, mech, field=f)                 # tabulates
    E = np.full(40000, ev(0.1))
    fi = np.full(E.size, 10.3)
    j = m.choose_variant(E, fi, np.random.default_rng(0))
    W0, W1, w, j0, j1 = m.rates_at(E[:1], fi[:1])
    p1 = w * W1 / ((1 - w) * W0 + w * W1)
    assert set(np.unique(j)) <= {10, 11}
    assert abs((j == 11).mean() - p1[0]) < 0.01


def test_equivalent_valleys_and_crystal_momentum():
    rng = np.random.default_rng(1)
    n = 60000
    new = choose_equivalent_valley(np.zeros(n, int), np.zeros(n, int), np.ones(n, int), rng)   # Gamma -> L
    assert np.allclose(np.bincount(new, minlength=4) / n, 0.25, atol=0.01)
    old = np.full(n, 2)
    new = choose_equivalent_valley(np.ones(n, int), old, np.ones(n, int), rng)                # L -> L
    assert not np.any(new == 2) and np.allclose(np.bincount(new, minlength=4)[[0, 1, 3]] / n, 1 / 3, atol=0.01)
    new = choose_equivalent_valley(np.full(n, 2), np.ones(n, int), np.full(n, 2), rng)        # X -> X
    assert not np.any(new == 1)
    a = MAT.a_lat
    K = valley_center(np.array([0, 1, 2]), np.array([0, 3, 2]), a)
    assert np.allclose(K[0], 0) and np.allclose(K[1], np.pi / a * np.array([-1, -1, 1]))
    assert np.allclose(K[2], 2 * np.pi / a * np.array([0, 0, 1]))


def test_arrivals_keep_valley_identity_and_K():
    s = Sample(MAT, per_cm3(5e17))
    f = C21BandBending(s)
    mech = build_mechanisms(s, ModelAssumptions(), "C", field=f)
    rng = np.random.default_rng(2)
    n = 400
    E = np.full(n, ev(0.05))
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, G.m_eff, G.alpha)[:, None]
    ens = Ensemble.create(z=np.full(n, 1.5 * f.W), k=k, E=E, spin=1)
    a = Simulation(s, mech, field=f, t_max=3 * PS).run(ens, rng).arrivals
    assert len(a) > 50 and (a.valley == 1).any()
    Kc = valley_center(a.valley, a.eqv, MAT.a_lat)
    assert np.allclose(a.K - a.k, Kc)
    assert np.all(a.eqv[a.valley == 0] == 0)


def test_stationary_density_with_local_impurity_screening():
    """Closed slab with weak artificial bending and LOCAL ionized-impurity screening: position-
    dependent elastic scattering must not disturb the Boltzmann density exp(-E_C(z)/kT)."""
    s = Sample(MAT, per_cm3(1.5e17))
    f = C21BandBending(s, E_bb=ev(0.05), W_bb=100 * NM)
    mech = [m for m in build_mechanisms(s, ModelAssumptions(), "B", field=f)
            if "eh_" not in m.name]                     # phonons + local impurity screening
    assert any(isinstance(m, LocalMechanism) for m in mech)
    L = 200 * NM
    zz = np.linspace(0, L, 4001)
    w = np.exp(-f.band_edge(zz) / s.kT)
    cdf = np.cumsum(w); cdf /= cdf[-1]
    rng = np.random.default_rng(5)
    n = 2500
    E = rng.gamma(1.5, s.kT, n)
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, G.m_eff, G.alpha)[:, None]
    ens = Ensemble.create(z=np.interp(rng.random(n), cdf, zz), k=k, E=E, spin=1)
    ts = np.linspace(2, 12, 6) * PS
    r = Simulation(s, mech, field=f, surface="reflect", z_back=L, back="reflect", t_max=12 * PS,
                   snapshot_times=ts).run(ens, rng)
    z = r.snapshots.z.ravel(); z = z[np.isfinite(z)]
    edges = np.linspace(0, L, 9)
    h, _ = np.histogram(z, edges)
    P = np.array([np.trapezoid(w[(zz >= a) & (zz <= b)], zz[(zz >= a) & (zz <= b)])
                  for a, b in zip(edges[:-1], edges[1:])])
    assert np.max(np.abs(h / (P / P.sum() * z.size) - 1)) < 0.08


def test_bulk_compatibility_option_has_no_local_mechanisms():
    s = Sample(MAT, per_cm3(1e19))
    f = C21BandBending(s)
    mech = build_mechanisms(s, ModelAssumptions(depletion_scattering="bulk"), "C", field=f)
    assert not any(isinstance(m, LocalMechanism) for m in mech)
    assert Simulation(s, mech, field=f).depl is None
