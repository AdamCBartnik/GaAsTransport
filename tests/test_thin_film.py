"""Stage F: finite GaAs layer 0 < z < d with a probabilistic back boundary (gaas_mc/back.py)."""
import warnings

import numpy as np
import pytest
from scipy import stats

from gaas_mc import bands
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.back import PartialReflector
from gaas_mc.constants import NM, PS, ev, per_cm3, to_ev
from gaas_mc.excitation import layer_absorption, photoexcite
from gaas_mc.fields import C21BandBending
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import ALIVE, BACK, EMITTED, TIMEOUT, Ensemble
from gaas_mc.scattering import build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.surface_c21 import C21Surface
from gaas_mc.transport import EV_BACK, Simulation

MAT = gaas_chubenko2021()
G = MAT.gamma
A = ModelAssumptions(depletion_scattering="bulk")


def bulk_sim(d, R, field=True, surface="absorb", t_max=5 * PS, p=1e19, mech_stage="C"):
    s = Sample(MAT, per_cm3(p))
    f = C21BandBending(s) if field else None
    mech = build_mechanisms(s, A, mech_stage, field=f)
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(s, mech, sm, field=f, surface=surface, z_back=d,
                         back=PartialReflector(R) if R is not None else "absorb", t_max=t_max, assumptions=A)
    return s, sim


def thermal(n, rng, z, kT):
    E = rng.gamma(1.5, kT, n)
    valley = rng.integers(0, 3, n).astype(np.int8)
    m = np.array([v.m_eff for v in MAT.valleys])[valley]
    a = np.array([v.alpha for v in MAT.valleys])[valley]
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, m, a)[:, None]
    return E, valley, k


# ------------------------------------------------------------------------------ R_back = 1
@pytest.mark.parametrize("field", [False, True])
def test_back_reflection_conserves_energy_exactly(field):
    """Flights to z = d and the reflection: total energy E + E_C(z) at every back encounter equals
    its value at the start (exact crossing + energy-conserving boundary state); k_z flips; k_par,
    valley and spin are untouched."""
    d = 8 * NM if field else 200 * NM                        # 8 nm < W_bb: the back is in the field
    s, sim = bulk_sim(d, 1.0, field=field)
    rng = np.random.default_rng(1)
    n = 4000
    # energetic electrons heading for the back (within ~30 deg of +z): in the field case the force
    # points toward z = 0, and slow electrons would turn around before reaching z = d
    valley = rng.integers(0, 3, n).astype(np.int8)
    m = np.array([v.m_eff for v in MAT.valleys])[valley]
    a = np.array([v.alpha for v in MAT.valleys])[valley]
    E = ev(rng.uniform(0.1, 0.3, n))
    u = bands.random_unit_vectors(n, rng)
    u[:, 2] = np.abs(u[:, 2]) + 1.7
    u /= np.linalg.norm(u, axis=1)[:, None]
    k = u * bands.k_of_E(E, m, a)[:, None]
    z = np.full(n, 0.9 * d)
    Etot0 = E + sim.field.band_edge(z)
    z1, k1, E1, du, ev_ = sim.propagate(z, k, E, valley, np.full(n, 1e-12))
    hit = ev_ == EV_BACK
    assert hit.mean() > 0.9
    assert np.allclose(z1[hit], d, rtol=0, atol=1e-18)
    Etot1 = E1 + sim.field.band_edge(z1)
    # field-free flights and the boundary state are exact (round-off); inside the field the substeps
    # before the crossing carry the velocity-Verlet error (Stage D: < 0.15 meV across W_bb)
    tol = ev(0.05e-3) if field else 1e-12 * np.abs(Etot0).max()
    assert np.max(np.abs(Etot1[hit] - Etot0[hit])) < tol
    # the reflection itself
    ens = Ensemble.create(z=z1, k=k1, E=E1, spin=np.where(rng.random(n) < 0.5, 1, -1), valley=0)
    ens.valley[:] = valley
    before = ens.copy()
    ib = np.flatnonzero(hit)
    sim._back_interaction(ens, ib, rng)
    assert np.all(ens.status[ib] == ALIVE) and np.all(ens.n_back[ib] == 1)
    assert np.array_equal(ens.k[ib, 2], -before.k[ib, 2]) and np.all(ens.k[ib, 2] < 0)
    for name in ("E", "valley", "spin", "eqv"):
        assert np.array_equal(getattr(ens, name)[ib], getattr(before, name)[ib])
    assert np.array_equal(ens.k[ib, :2], before.k[ib, :2])


@pytest.mark.parametrize("engine", ["ref", "cpu"])
def test_reflecting_slab_no_heating_or_drift(engine):
    """Closed field-free slab: reflecting front, PartialReflector(1) back, phonons only (detailed
    balance with the lattice). Thermal electrons undergo many back reflections; the mean energy stays
    at the nonparabolic equilibrium value and the density stays uniform."""
    s, sim = bulk_sim(60 * NM, 1.0, field=False, surface="reflect", t_max=3 * PS, p=1e15, mech_stage="A")
    rng = np.random.default_rng(2)
    n = 6000
    E = rng.gamma(1.5, s.kT, n)
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, G.m_eff, G.alpha)[:, None]
    ens = Ensemble.create(z=rng.uniform(0, 60 * NM, n), k=k, E=E, spin=1)
    if engine == "ref":
        r = sim.run(ens, rng)
    else:
        from gaas_mc.fast import FastSimulation
        r = FastSimulation(sim, "cpu").run(ens, rng)
    e = r.ensemble
    assert np.all(e.status == TIMEOUT) and e.n_back.mean() > 3
    x = np.linspace(0, 1, 20001) * 40 * s.kT                 # exact nonparabolic Maxwellian mean
    w = np.sqrt(bands.gamma_of_E(x, G.alpha)) * (1 + 2 * G.alpha * x) * np.exp(-x / s.kT)
    E_eq = np.trapezoid(x * w, x) / np.trapezoid(w, x)
    assert abs(e.E.mean() / E_eq - 1) < 4 * e.E.std() / np.sqrt(n) / E_eq + 0.005
    h, _ = np.histogram(e.z, np.linspace(0, 60 * NM, 7))
    assert np.max(np.abs(h / (n / 6) - 1)) < 0.08


# ------------------------------------------------------------------------------ R_back = 0
@pytest.mark.parametrize("engine", ["ref", "cpu"])
def test_absorbing_back_terminates_exactly_once(engine):
    s, sim = bulk_sim(100 * NM, 0.0, surface=C21Surface(chi=ev(0.67), material=MAT), t_max=20 * PS)
    rng = np.random.default_rng(3)
    ens = photoexcite(s, ev(1.6), 3000, rng, assumptions=A, thickness=100 * NM)
    if engine == "ref":
        r = sim.run(ens, rng)
    else:
        from gaas_mc.fast import FastSimulation
        r = FastSimulation(sim, "cpu").run(ens, rng)
    e = r.ensemble
    lost = e.status == BACK
    assert lost.sum() > 100
    assert np.all(e.n_back[lost] == 1) and np.all(e.n_back[~lost] == 0)
    assert np.all(e.z[lost] == 100 * NM)
    assert np.all(r.emissions.n_back == 0)


# ------------------------------------------------------------------------------ 0 < R_back < 1
@pytest.mark.parametrize("engine", ["ref", "cpu"])
def test_partial_reflection_fraction(engine):
    """Thin field-free slab with a reflecting front: electrons meet the back many times; reflected /
    encounters must match R_back."""
    R = 0.7
    s, sim = bulk_sim(40 * NM, R, field=False, surface="reflect", t_max=3 * PS, p=1e15, mech_stage="A")
    rng = np.random.default_rng(4)
    n = 4000
    E = rng.gamma(1.5, s.kT, n)
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E, G.m_eff, G.alpha)[:, None]
    ens = Ensemble.create(z=rng.uniform(0, 40 * NM, n), k=k, E=E, spin=1)
    if engine == "ref":
        r = sim.run(ens, rng)
    else:
        from gaas_mc.fast import FastSimulation
        r = FastSimulation(sim, "cpu").run(ens, rng)
    e = r.ensemble
    enc = int(e.n_back.sum())
    lost = int(np.sum(e.status == BACK))
    assert enc > 5000
    frac = (enc - lost) / enc
    assert abs(frac - R) < 4.5 * np.sqrt(R * (1 - R) / enc)
    assert np.all(e.n_back[e.status == BACK] >= 1)


# ------------------------------------------------------------------------------ generation
@pytest.mark.parametrize("d_nm", [50, 300, 2000])
def test_truncated_generation_profile(d_nm):
    s = Sample(MAT, per_cm3(1e19))
    d = d_nm * NM
    rng = np.random.default_rng(5)
    ens = photoexcite(s, ev(1.6), 40000, rng, assumptions=A, thickness=d)
    from gaas_mc.optics import as_absorption_model
    alpha = float(as_absorption_model(A.absorption_model, s).absorption_coefficient(ev(1.6)))
    assert np.all((ens.z0 > 0) & (ens.z0 <= d))
    cdf = lambda z: -np.expm1(-alpha * z) / -np.expm1(-alpha * d)
    assert stats.kstest(ens.z0, cdf).pvalue > 1e-3
    A_layer = layer_absorption(s, ev(1.6), d, assumptions=A)
    assert np.isclose(A_layer, layer_absorption(s, ev(1.6), None, assumptions=A) * -np.expm1(-alpha * d))


def test_thin_layer_warning():
    s = Sample(MAT, per_cm3(1e19))
    f = C21BandBending(s)
    with pytest.warns(UserWarning, match="band-bending width"):
        Simulation(s, build_mechanisms(s, A, "A"), field=f, z_back=2 * f.W, back=PartialReflector(0.5),
                   assumptions=A)


def test_back_model_needs_z_back():
    s = Sample(MAT, per_cm3(1e19))
    with pytest.raises(ValueError):
        Simulation(s, build_mechanisms(s, A, "A"), back=PartialReflector(0.5), assumptions=A)
    with pytest.raises(ValueError):
        PartialReflector(1.5)


# ------------------------------------------------------------------------------ large-d limit
def test_large_thickness_reproduces_semi_infinite():
    from gaas_mc.fast import FastSimulation
    s, sim_d = bulk_sim(20e-6, 0.0, surface=C21Surface(chi=ev(0.67), material=MAT), t_max=100 * PS)
    sim_inf = Simulation(s, sim_d.mechanisms, sim_d.spin_model, field=sim_d.field,
                         surface=C21Surface(chi=ev(0.67), material=MAT), t_max=100 * PS, assumptions=A)
    n = 20000
    out = []
    for sim, d in ((sim_inf, None), (sim_d, 20e-6)):
        rng = np.random.default_rng(6)
        ens = photoexcite(s, ev(1.6), n, rng, assumptions=A, thickness=d)
        r = FastSimulation(sim, "cpu").run(ens, rng)
        out.append(r)
    (ri, rd) = out
    assert np.sum(rd.ensemble.status == BACK) == 0
    for status in (EMITTED, TIMEOUT):
        pi, pd = np.mean(ri.ensemble.status == status), np.mean(rd.ensemble.status == status)
        assert abs(pi - pd) < 4.5 * np.sqrt(2 * pi * (1 - pi) / n)
    ei, ed = ri.emissions.spin.mean(), rd.emissions.spin.mean()
    assert abs(ei - ed) < 4.5 * np.sqrt((1 - ei**2) / len(ri.emissions) * 2)
