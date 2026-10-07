"""Stage B (impurity, electron-hole) and Stage C (intervalley, upper-valley bookkeeping) tests.

The electron-hole tests are deliberately redundant: a bug in this mechanism is the subject of
the Karkare 2015 erratum, which does not say what the bug was."""
import dataclasses

import numpy as np
import pytest
from scipy import integrate

from gaas_mc import bands
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import EPS0, HBAR, PS, Q_E, ev, per_cm3
from gaas_mc.holes import HoleGas, fermi_integral_half
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import (AcousticPhonon, ElectronHole, IonizedImpurity, Intervalley,
                                build_mechanisms)
from gaas_mc.transport import Simulation

MAT = gaas_chubenko2021()
MAT_PARABOLIC = dataclasses.replace(
    MAT, valleys=(dataclasses.replace(MAT.gamma, alpha=0.0),) + MAT.valleys[1:])
ENERGIES = ev(np.array([0.003, 0.03, 0.1, 0.4, 1.0]))


def iso_k(E, m, a, rng):
    E = np.asarray(E, float)
    return bands.random_unit_vectors(E.size, rng) * bands.k_of_E(E, m, a)[:, None]


# ------------------------------------------------------------------ ionized impurities ----

@pytest.mark.parametrize("p", [1.5e17, 1e19])
def test_impurity_rates_vs_golden_rule(p):
    s = Sample(MAT, per_cm3(p))
    ii = IonizedImpurity(s)
    M2 = (Q_E**2 / s.eps_s) ** 2                               # |M|^2 Omega^2 (1/(q^2+beta^2)^2 apart)
    for E in ENERGIES:
        k = bands.k_of_E(E, ii.m, ii.alpha)
        dEdk = HBAR**2 * k / (ii.m * (1 + 2 * ii.alpha * E))
        a, b = 2 * k * k + s.beta**2, 2 * k * k
        # integrate over t = ln(a - b cos(theta)) (smooth even when strongly forward-peaked)

        def f(t, w):
            u = np.exp(t)
            c = (a - u) / b
            return w(c) * u / b / u**2
        lims = (np.log(a - b), np.log(a + b))
        pref = 2 * np.pi / HBAR * s.p * M2 / (2 * np.pi) ** 3 * k**2 / dEdk * 2 * np.pi
        W = pref * integrate.quad(f, *lims, args=(lambda c: 1.0,), epsabs=0, epsrel=1e-12, limit=200)[0]
        Wm = pref * integrate.quad(f, *lims, args=(lambda c: 1 - c,), epsabs=0, epsrel=1e-12, limit=200)[0]
        assert np.isclose(ii.rate(E), W, rtol=1e-8)
        assert np.isclose(ii.momentum_rate(E)[0], Wm, rtol=1e-7)


def test_impurity_angle_distribution():
    s = Sample(MAT, per_cm3(1e18))
    ii = IonizedImpurity(s)
    rng = np.random.default_rng(1)
    n = 200_000
    E = np.full(n, ev(0.1))
    k0 = np.tile([0, 0, bands.k_of_E(ev(0.1), ii.m, ii.alpha)], (n, 1))
    kn, _, _ = ii.scatter(k0, E, rng)
    assert np.allclose(np.linalg.norm(kn, axis=1), k0[0, 2])
    c = kn[:, 2] / k0[0, 2]
    a, b = 2 * k0[0, 2] ** 2 + s.beta**2, 2 * k0[0, 2] ** 2           # density ~ 1/(a - b c)^2
    edges = np.linspace(-1, 1, 21)
    cdf = lambda x: 1 / (b * (a - b * x))
    expected = np.diff(cdf(edges)) / (cdf(1) - cdf(-1)) * n
    hist, _ = np.histogram(c, edges)
    ok = expected > 20
    assert np.sum((hist[ok] - expected[ok]) ** 2 / expected[ok]) < 3 * ok.sum()


# ------------------------------------------------------------------ hole gas ---------------

def test_fermi_integral_limits():
    assert np.isclose(fermi_integral_half(0.0), 0.765147024625, rtol=1e-9)
    assert np.isclose(fermi_integral_half(-25) / np.exp(-25), 1, rtol=1e-8)


@pytest.mark.parametrize("p", [1.5e17, 1e19])
@pytest.mark.parametrize("stat", ["fermi_dirac", "maxwell_boltzmann"])
def test_hole_gas_density_and_sampling(p, stat):
    s = Sample(MAT, per_cm3(p))
    hg = HoleGas(s, ("hh", "lh"), stat)
    assert np.isclose(sum(hg.density.values()), s.p, rtol=1e-9)
    # lh/hh ratio = (m_lh/m_hh)^(3/2) for a common chemical potential
    assert np.isclose(hg.density["lh"] / hg.density["hh"], (MAT.m_lh / MAT.m_hh) ** 1.5)
    rng = np.random.default_rng(2)
    Eh = hg.sample_energy(400_000, rng)
    assert abs(Eh.mean() / hg.mean_energy - 1) < 5e-3
    if stat == "maxwell_boltzmann":
        assert abs(hg.mean_energy / (1.5 * s.kT) - 1) < 1e-3


# ------------------------------------------------------------------ electron-hole ----------

def _eh(p, band="hh", mat=MAT, pauli="none", stat="fermi_dirac", valley=0):
    s = Sample(mat, per_cm3(p))
    hg = HoleGas(s, ("hh", "lh"), stat)
    m = ElectronHole(s, hg, band, valley, pauli=pauli)
    return s, hg, m


@pytest.mark.parametrize("mat", [MAT, MAT_PARABOLIC])
@pytest.mark.parametrize("band", ["hh", "lh"])
def test_eh_conserves_momentum_and_energy(mat, band):
    s, hg, m = _eh(1e19, band, mat, pauli="fermi_dirac")
    m.record = True
    rng = np.random.default_rng(3)
    E = np.full(50_000, ev(0.3))
    k = iso_k(E, m.m, m.alpha, rng)
    m.scatter(k, E, rng)
    L = m.last
    assert L["kp"].shape[0] > 1000
    assert np.allclose(L["k"] + L["k0"], L["kp"] + L["k0p"], rtol=0, atol=1e-6 * np.abs(L["k"]).max())
    Ee = lambda kk: bands.E_of_k(np.linalg.norm(kk, axis=1), m.m, m.alpha)
    Eh = lambda kk: HBAR**2 * np.sum(kk * kk, axis=1) / (2 * m.mh)
    before, after = Ee(L["k"]) + Eh(L["k0"]), Ee(L["kp"]) + Eh(L["k0p"])
    assert np.allclose(after, before, rtol=1e-9, atol=0)
    if mat is MAT_PARABOLIC:          # exact kinematics reduce to C21 Eq. 43, |g'| = |g|
        assert np.allclose(L["s"], L["g"], rtol=1e-9)


@pytest.mark.parametrize("p,band", [(1.5e17, "hh"), (1e19, "hh"), (1e19, "lh")])
def test_eh_accepted_rate_equals_born_rate(p, band):
    """Independent check of Eqs. 38-41: with Pauli blocking off, the accepted rate must equal
    p_b < |v_e - v_h| sigma_Born(k_rel) > over the hole distribution, computed here from the
    velocities and the textbook screened-Coulomb Born cross-section (not from g)."""
    s, hg, m = _eh(p, band, MAT_PARABOLIC, pauli="none")
    rng = np.random.default_rng(4)
    n = 400_000
    for Eev in (0.02, 0.3):
        E = np.full(n, ev(Eev))
        k = iso_k(E, m.m, m.alpha, rng)
        _, acc, _ = m.scatter(k, E, rng)
        W_mc = m.rate(E[:1])[0] * acc.mean()
        k0 = hg.sample_k(band, n, rng)
        v_e, v_h = HBAR * k / m.m, HBAR * k0 / m.mh
        mR = m.m * m.mh / (m.m + m.mh)
        k_rel = mR * np.linalg.norm(v_e - v_h, axis=1) / HBAR
        sigma = 16 * np.pi * mR**2 * (Q_E**2 / (4 * np.pi * s.eps_s)) ** 2 / HBAR**4 \
            / (s.beta**2 * (s.beta**2 + 4 * k_rel**2))
        W_ref = hg.density[band] * np.mean(np.linalg.norm(v_e - v_h, axis=1) * sigma)
        assert abs(W_mc / W_ref - 1) < 0.01, (Eev, W_mc, W_ref)


def test_eh_pauli_blocking_scaling():
    """Negligible blocking for nondegenerate holes, significant at 1e19 cm^-3."""
    rng = np.random.default_rng(5)
    out = {}
    for p in (1.5e17, 1e19):
        s, hg, m = _eh(p, pauli="fermi_dirac")
        E = np.full(100_000, ev(0.03))
        m.scatter(iso_k(E, m.m, m.alpha, rng), E, rng)
        st = m.stats
        out[p] = st["rejected_pauli"] / (st["attempted"] - st["rejected_eq40"])
    assert out[1.5e17] < 0.01 and out[1e19] > 0.1


def _thermalize(mat, p, pauli, n=12_000, t_end=12 * PS, seed=6):
    s = Sample(mat, per_cm3(p))
    hg = HoleGas(s)
    mech = [ElectronHole(s, hg, b, 0, pauli=pauli) for b in ("hh", "lh")]
    rng = np.random.default_rng(seed)
    g = mat.gamma
    E = np.full(n, ev(0.15))
    ens = Ensemble.create(z=np.ones(n), k=iso_k(E, g.m_eff, g.alpha, rng), E=E, spin=1)
    r = Simulation(s, mech, surface="none", t_max=t_end, snapshot_times=[t_end]).run(ens, rng)
    return s, r.snapshots.E[0]


@pytest.mark.parametrize("p,t_end", [(1.5e17, 40 * PS), (1e19, 12 * PS)])
def test_eh_alone_thermalizes_to_lattice_temperature(p, t_end):
    """Detailed balance with the hole bath. With e-h scattering alone (parabolic band, FD holes,
    FD Pauli blocking) electrons must relax to a Maxwellian at T: <E> = 3/2 kT and
    P(E) ~ sqrt(E) exp(-E/kT). A wrong g, W_max, acceptance, angle, or Pauli rule breaks this."""
    s, E = _thermalize(MAT_PARABOLIC, p, "fermi_dirac", n=6000, t_end=t_end)
    x = E / s.kT
    assert abs(x.mean() / 1.5 - 1) < 0.03
    edges = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 40])
    from scipy.special import gammainc
    cdf = gammainc(1.5, edges)
    expected = np.diff(cdf) * x.size
    hist, _ = np.histogram(x, edges)
    assert np.sum((hist - expected) ** 2 / expected) < 30


def test_c21_step_pauli_breaks_detailed_balance():
    """Documents why 'fermi_dirac' is the default: the C21 Eq. 44 step rule combined with FD-sampled
    holes does not leave the Maxwellian stationary at 1e19 cm^-3."""
    s, E = _thermalize(MAT_PARABOLIC, 1e19, "step_c21", n=3000)
    assert abs((E / s.kT).mean() / 1.5 - 1) > 0.05


# ------------------------------------------------------------------ intervalley -------------

def test_intervalley_golden_rule_and_multiplicity():
    s = Sample(MAT, per_cm3(1e18))
    gl = Intervalley(s, 0, 1, emission=True, Z=4)
    gl1 = Intervalley(s, 0, 1, emission=True, Z=1)
    vL = MAT.valleys[1]
    E = ev(np.array([0.5, 0.8, 1.2]))
    Ep = E - gl.hw - vL.offset
    g1 = bands.dos(Ep, vL.m_eff, vL.alpha, spin_degeneracy=1)
    occ = 1 / np.expm1(gl.hw / s.kT) + 1
    W = np.pi * gl.D**2 * 4 / (MAT.rho * gl.hw / HBAR) * g1 * occ
    assert np.allclose(gl.rate(E), W, rtol=1e-12)
    assert np.allclose(gl.rate(E), 4 * gl1.rate(E))
    assert gl.rate(ev(0.2))[0] == 0 and np.isclose(gl.threshold, vL.offset + gl.hw)


def test_default_multiplicities():
    s = Sample(MAT, per_cm3(1e18))
    mech = build_mechanisms(s, ModelAssumptions(), "C")
    Z = {m.name: m.Z for m in mech if isinstance(m, Intervalley)}
    expect = {"Gamma->L": 4, "Gamma->X": 3, "L->Gamma": 1, "L->L": 3, "L->X": 3,
              "X->Gamma": 1, "X->L": 4, "X->X": 2}
    for key, z in expect.items():
        assert Z[f"iv_em[{key}]"] == z and Z[f"iv_abs[{key}]"] == z


# ------------------------------------------------------------------ upper-valley spin -------

class HugeSpin:
    def total(self, E):
        return np.full_like(E, 1e15)


def test_spin_frozen_in_upper_valleys_and_time_bookkeeping():
    s = Sample(MAT, per_cm3(1e18))
    mech = [m for m in build_mechanisms(s, ModelAssumptions(), "C") if m.valley_from == 2
            and not isinstance(m, Intervalley)]
    rng = np.random.default_rng(7)
    vX = MAT.valleys[2]
    n = 3000
    E = np.full(n, ev(0.05))
    ens = Ensemble.create(z=np.ones(n), k=iso_k(E, vX.m_eff, vX.alpha, rng), E=E, spin=1, valley=2)
    r = Simulation(s, mech, HugeSpin(), surface="none", t_max=2 * PS).run(ens, rng)
    assert np.all(r.ensemble.spin == 1)                       # no relaxation in X
    tv = r.ensemble.time_in_valley
    assert np.allclose(tv.sum(axis=1), r.ensemble.t, rtol=1e-12, atol=0)
    assert np.all(tv[:, :2] == 0) and np.all(r.ensemble.visited[:, 2])


def test_valley_transfer_bookkeeping_stage_c():
    s = Sample(MAT, per_cm3(1e18))
    mech = build_mechanisms(s, ModelAssumptions(), "C")
    rng = np.random.default_rng(8)
    g = MAT.gamma
    n = 2000
    E = np.full(n, ev(0.8))
    ens = Ensemble.create(z=np.ones(n), k=iso_k(E, g.m_eff, g.alpha, rng), E=E, spin=1)
    r = Simulation(s, mech, surface="none", t_max=3 * PS).run(ens, rng)
    e = r.ensemble
    assert np.allclose(e.time_in_valley.sum(1), e.t, rtol=1e-12, atol=0)
    assert e.visited[:, 1:].any(axis=1).mean() > 0.5          # 0.8 eV electrons transfer
    assert np.all(e.visited[e.time_in_valley[:, 1] > 0, 1])


# ------------------------------------------------------------------ flight modes ------------

def test_direct_and_self_scattering_modes_agree():
    s = Sample(MAT, per_cm3(1e19))
    mech = build_mechanisms(s, ModelAssumptions(), "B")
    g = MAT.gamma
    times = np.array([0.1, 0.5, 2.0]) * PS
    means = {}
    for mode in ("direct", "self_scattering"):
        rng = np.random.default_rng(9)
        n = 6000
        E = np.full(n, ev(0.4))
        ens = Ensemble.create(z=np.ones(n), k=iso_k(E, g.m_eff, g.alpha, rng), E=E, spin=1)
        sim = Simulation(s, mech, surface="none", t_max=2 * PS, snapshot_times=times,
                         assumptions=ModelAssumptions(flight_mode=mode))
        r = sim.run(ens, rng)
        assert r.flight_mode == mode
        means[mode] = np.nanmean(r.snapshots.E, axis=1)
        if mode == "direct":
            n_eh_rej = sum(r.n_rejected)
            assert r.n_self == n_eh_rej                       # only mechanism rejections remain
    assert np.allclose(means["direct"], means["self_scattering"], rtol=0.03)
