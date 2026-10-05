import numpy as np
import pytest

from gaas_mc import bands
from gaas_mc.constants import EV, HBAR, NM, ev, per_cm3, to_ev
from gaas_mc.material import Sample, gaas_chubenko2021

MAT = gaas_chubenko2021()
G = MAT.gamma


@pytest.mark.parametrize("alpha", [0.0, G.alpha])
def test_E_k_roundtrip(alpha):
    E = ev(np.linspace(1e-6, 2.0, 500))
    k = bands.k_of_E(E, G.m_eff, alpha)
    assert np.allclose(bands.E_of_k(k, G.m_eff, alpha), E, rtol=1e-12)
    # Eq. 1 directly
    assert np.allclose(E * (1 + alpha * E), HBAR**2 * k**2 / (2 * G.m_eff), rtol=1e-12)


def test_parabolic_limit():
    E = ev(0.1)
    assert np.isclose(bands.k_of_E(E, G.m_eff, 0.0), np.sqrt(2 * G.m_eff * E) / HBAR)


def test_velocity_is_dEdk_over_hbar():
    """Eq. 19 must equal (1/hbar) dE/dk of Eq. 2."""
    k = np.linspace(1e7, 2e9, 300)
    h = 1e-6 * k
    dEdk = (bands.E_of_k(k + h, G.m_eff, G.alpha) - bands.E_of_k(k - h, G.m_eff, G.alpha)) / (2 * h)
    v = bands.speed_of_E(bands.E_of_k(k, G.m_eff, G.alpha), G.m_eff, G.alpha)
    assert np.allclose(v, dEdk / HBAR, rtol=1e-7)


def test_velocity_vector_parallel_to_k():
    rng = np.random.default_rng(0)
    kv = rng.normal(size=(50, 3)) * 1e9
    v = bands.velocity(kv, G.m_eff, G.alpha)
    assert np.allclose(np.cross(v, kv), 0, atol=1e-6 * np.abs(v).max() * 1e9)


def test_dos_matches_state_counting():
    """g(E) = 2 * 4 pi k^2 (dk/dE) / (2 pi)^3, evaluated numerically."""
    E = ev(np.linspace(0.01, 1.5, 100))
    dE = 1e-7 * EV
    dkdE = (bands.k_of_E(E + dE, G.m_eff, G.alpha) - bands.k_of_E(E - dE, G.m_eff, G.alpha)) / (2 * dE)
    k = bands.k_of_E(E, G.m_eff, G.alpha)
    g_count = 2 * 4 * np.pi * k**2 * dkdE / (2 * np.pi) ** 3
    assert np.allclose(bands.dos(E, G.m_eff, G.alpha), g_count, rtol=1e-6)


def test_rotate_about_gives_requested_angle():
    rng = np.random.default_rng(1)
    u = bands.random_unit_vectors(1000, rng)
    c = 1 - 2 * rng.random(1000)
    w = bands.rotate_about(u, c, 2 * np.pi * rng.random(1000))
    assert np.allclose(np.linalg.norm(w, axis=1), 1)
    assert np.allclose(np.sum(u * w, axis=1), c, atol=1e-12)


def test_isotropic_directions():
    rng = np.random.default_rng(2)
    u = bands.random_unit_vectors(200_000, rng)
    assert np.allclose(u.mean(axis=0), 0, atol=5e-3)
    assert np.allclose((u**2).mean(axis=0), 1 / 3, atol=5e-3)


# --- doping-dependent quantities: numbers printed in [C21] ---------------------------------

@pytest.mark.parametrize("p,Eg", [(5e17, 1.409), (1.7e18, 1.398), (1e19, 1.361)])
def test_bandgap_eq10_vs_fig6_legend(p, Eg):
    assert abs(to_ev(Sample(MAT, per_cm3(p)).Eg) - Eg) < 6e-4


def test_band_bending_vs_fig16_caption():
    s = Sample(MAT, per_cm3(1e19))
    assert abs(to_ev(s.E_bb) - 0.694) < 6e-4          # Eqs. 56-58
    assert abs(s.W_bb / NM - 9.947) < 6e-3            # Eq. 59


def test_degeneracy_and_screening_selection():
    assert not Sample(MAT, per_cm3(1.5e17)).degenerate
    assert Sample(MAT, per_cm3(1e19)).degenerate
    s = Sample(MAT, per_cm3(1.5e17))
    assert s.screening_model == "debye" and np.isclose(s.beta, 1 / s.L_debye)


def test_thomas_fermi_matches_textbook_form():
    """Eq. 29 equals beta^2 = e^2 g(E_F)/eps for a parabolic light-hole gas (spin 2)."""
    from gaas_mc.constants import Q_E
    s = Sample(MAT, per_cm3(1e19))
    kF = (3 * np.pi**2 * s.N_lh) ** (1 / 3)
    gEF = MAT.m_lh * kF / (np.pi**2 * HBAR**2)
    assert np.isclose(1 / s.L_TF**2, Q_E**2 * gEF / s.eps_s, rtol=1e-12)
