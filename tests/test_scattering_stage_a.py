"""Stage A scattering: closed forms vs. independent numerical golden-rule integration."""
import numpy as np
import pytest
from scipy import integrate

from gaas_mc import bands
from gaas_mc.constants import EV, HBAR, Q_E, ev, per_cm3
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import AcousticPhonon, PolarOptical

MAT = gaas_chubenko2021()
SAMPLES = [Sample(MAT, per_cm3(1.5e17)), Sample(MAT, per_cm3(1e19))]
ENERGIES = ev(np.array([0.005, 0.04, 0.1, 0.3, 0.8, 1.5]))


def golden_rule_pop(mech, E, weight_tau_m=False):
    """W = (2 pi/hbar) Omega/(2 pi)^3 Int d^3k' |M|^2 delta(E' - E -+ hw), with
    |M|^2 Omega = e^2 hbar w0 /(2 eps_p) * q^2/(q^2+beta^2)^2 * occupation  (Froehlich, screened).
    Integrated numerically over cos(theta)."""
    m, a = mech.m, mech.alpha
    Ep = mech.final_energy(E)
    k = bands.k_of_E(E, m, a)
    kp = bands.k_of_E(Ep, m, a)
    dEdk_p = HBAR**2 * kp / (m * (1 + 2 * a * Ep))
    M2 = Q_E**2 * mech.hw / (2 * mech.material.eps_p) * mech.occupation

    def f(c):
        q2 = k**2 + kp**2 - 2 * k * kp * c
        w = q2 / (q2 + mech.a) ** 2
        if weight_tau_m:
            w *= 1 - kp * c / k
        return w

    ang, _ = integrate.quad(f, -1, 1, epsabs=0, epsrel=1e-11, limit=200)
    return (2 * np.pi / HBAR) / (2 * np.pi) ** 3 * M2 * (kp**2 / dEdk_p) * 2 * np.pi * ang


@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("emission", [False, True])
def test_pop_rate_and_tau_m_vs_quadrature(sample, emission):
    mech = PolarOptical(sample, emission=emission)
    for E in ENERGIES:
        if emission and E <= mech.hw:
            assert mech.rate(E)[0] == 0
            continue
        assert np.isclose(mech.rate(E)[0], golden_rule_pop(mech, E), rtol=1e-8)
        assert np.isclose(mech.momentum_rate(E)[0], golden_rule_pop(mech, E, True), rtol=1e-7)


def test_pop_unscreened_limit():
    s = SAMPLES[0]
    m = PolarOptical(s, emission=True, screened=False)
    E = ev(0.3)
    k, kp, _ = m._k_kp(np.array([E]))
    expected = m._C0(E, E - m.hw) * 2 * np.log((k + kp) / (k - kp))
    assert np.isclose(m.rate(E)[0], expected[0], rtol=1e-12)


@pytest.mark.parametrize("sample", SAMPLES)
def test_pop_detailed_balance(sample):
    """g(E) e^{-E/kT} W_abs(E) = g(E') e^{-E'/kT} W_em(E'),  E' = E + hw0."""
    ab = PolarOptical(sample, emission=False)
    em = PolarOptical(sample, emission=True)
    E = ev(np.linspace(0.002, 1.0, 50))
    Ep = E + ab.hw
    g = lambda x: bands.dos(x, ab.m, ab.alpha)
    lhs = g(E) * np.exp(-E / sample.kT) * ab.rate(E)
    rhs = g(Ep) * np.exp(-Ep / sample.kT) * em.rate(Ep)
    assert np.allclose(lhs, rhs, rtol=1e-10)


def test_acoustic_eq23_vs_golden_rule():
    """Equipartition: W = (2 pi/hbar) Xi^2 kT / c_l * g_1spin(E)."""
    s = SAMPLES[0]
    ac = AcousticPhonon(s)
    g1 = bands.dos(ENERGIES, ac.m, ac.alpha, spin_degeneracy=1)
    expected = 2 * np.pi / HBAR * ac.valley.Xi_d**2 * s.kT / MAT.c_l * g1
    assert np.allclose(ac.rate(ENERGIES), expected, rtol=1e-12)
    assert np.allclose(ac.momentum_rate(ENERGIES), ac.rate(ENERGIES))


def test_eq31_parabolic_limit_matches_closed_form():
    """[C21] Eq. 31 (transcribed) should equal Eq. 22 in the parabolic limit; in the
    nonparabolic band it differs by a few % (reported in validation, ambiguity A7)."""
    import dataclasses
    mat0 = dataclasses.replace(MAT, valleys=(dataclasses.replace(MAT.gamma, alpha=0.0),) + MAT.valleys[1:])
    for p in (1.5e17, 1e19):
        s = Sample(mat0, per_cm3(p))
        for em in (False, True):
            m = PolarOptical(s, emission=em)
            E = ENERGIES[ENERGIES > m.hw] if em else ENERGIES
            assert np.allclose(m.tau_m_eq31(E), m.momentum_rate(E), rtol=1e-8)


@pytest.mark.parametrize("angle", ["chubenko_eq30", "screened"])
def test_pop_angle_sampling_distribution(angle):
    """Sampled u = q^2 must follow 1/u (Eq. 30) or u/(u+beta^2)^2 (screened)."""
    s = SAMPLES[1]
    mech = PolarOptical(s, emission=True, angle=angle)
    rng = np.random.default_rng(3)
    n = 200_000
    E = np.full(n, ev(0.2))
    k0 = np.tile([0, 0, bands.k_of_E(ev(0.2), mech.m, mech.alpha)], (n, 1))
    knew, acc, _ = mech.scatter(k0, E, rng)
    assert acc.all()
    kk, kp, Ep = mech._k_kp(E[:1])
    assert np.allclose(np.linalg.norm(knew, axis=1), kp[0], rtol=1e-12)
    u = np.sum((knew - k0) ** 2, axis=1)
    umin, umax = (kp[0] - kk[0]) ** 2, (kp[0] + kk[0]) ** 2
    edges = np.geomspace(umin, umax, 21)
    hist, _ = np.histogram(u, edges)
    a = mech.a if angle == "screened" else 0.0
    cdf = (lambda x: np.log(x + a) + a / (x + a)) if a else np.log
    expected = np.diff(cdf(edges)) / (cdf(umax) - cdf(umin)) * n
    chi2 = np.sum((hist - expected) ** 2 / expected)
    assert chi2 < 60          # 20 bins; generous but catches a wrong distribution


def test_acoustic_scatter_isotropic_elastic():
    s = SAMPLES[0]
    ac = AcousticPhonon(s)
    rng = np.random.default_rng(4)
    n = 100_000
    E = np.full(n, ev(0.1))
    k0 = np.tile([0, 0, bands.k_of_E(ev(0.1), ac.m, ac.alpha)], (n, 1))
    kn, _, _ = ac.scatter(k0, E, rng)
    assert np.allclose(np.linalg.norm(kn, axis=1), k0[0, 2])
    c = kn[:, 2] / k0[0, 2]
    assert abs(c.mean()) < 0.01 and abs((c**2).mean() - 1 / 3) < 0.01
