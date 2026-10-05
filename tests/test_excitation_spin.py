import numpy as np
import pytest

from gaas_mc import bands, excitation
from gaas_mc.constants import M0, NM, PS, ev, per_cm3
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import stage_a_mechanisms
from gaas_mc.spin import SpinModel, flip_probability

MAT = gaas_chubenko2021()


@pytest.mark.parametrize("band", [0, 1, 2])
@pytest.mark.parametrize("hw", [1.5, 1.8, 2.2])
def test_eq9_conserves_energy_and_momentum(band, hw):
    s = Sample(MAT, per_cm3(1e18))
    hw = ev(hw)
    dEe = excitation.excess_energy(s, hw, band)
    if dEe <= 0:
        pytest.skip("band inaccessible")
    g = MAT.gamma
    mh = (MAT.m_hh, MAT.m_lh, MAT.m_so)[band]
    Delta = MAT.Delta_so if band == 2 else 0.0
    dEh = hw - s.Eg - Delta - dEe                                      # Eq. 8
    # equal |k|: hbar^2 k^2/2 = gamma(E_e) m_e = E_h m_h
    assert np.isclose(bands.gamma_of_E(dEe, g.alpha) * g.m_eff, dEh * mh, rtol=1e-10)


def test_esp0_intrinsic_vs_fig6():
    """[C21] Fig. 6 / text: 50% near the edge, ~47% before so onset, then a fast drop."""
    s = Sample(MAT, per_cm3(1e10))          # essentially intrinsic (Eg ~ 1.423 eV)
    assert abs(excitation.esp0(s, ev(1.43)) - 0.50) < 0.002
    assert 0.45 < excitation.esp0(s, ev(1.74)) < 0.48
    assert 0.22 < excitation.esp0(s, ev(1.80)) < 0.30
    assert 0.06 < excitation.esp0(s, ev(2.20)) < 0.10


@pytest.mark.parametrize("model", ["per_band", "chubenko_text"])
def test_sampled_esp_matches_eq12(model):
    s = Sample(MAT, per_cm3(1e19))
    rng = np.random.default_rng(5)
    for hw in (1.45, 1.75, 1.9):
        ens = excitation.photoexcite(s, ev(hw), 200_000, rng, 500 * NM, spin_model=model)
        assert abs(ens.esp() - excitation.esp0(s, ev(hw))) < 0.01


def test_depth_distribution_and_reproducibility():
    s = Sample(MAT, per_cm3(1e19))
    e1 = excitation.photoexcite(s, ev(1.6), 100_000, np.random.default_rng(7), 400 * NM)
    e2 = excitation.photoexcite(s, ev(1.6), 100_000, np.random.default_rng(7), 400 * NM)
    assert np.array_equal(e1.z, e2.z) and np.array_equal(e1.k, e2.k) and np.array_equal(e1.spin, e2.spin)
    assert abs(e1.z.mean() / (400 * NM) - 1) < 0.01


def test_below_gap_requires_opt_in():
    s = Sample(MAT, per_cm3(1e19))
    with pytest.raises(ValueError):
        excitation.photoexcite(s, ev(1.3), 10, np.random.default_rng(0), 1e-6)


# --- spin relaxation ------------------------------------------------------------------------

def test_flip_probability_limits():
    assert flip_probability(0.0, 1e10) == 0
    assert np.isclose(flip_probability(1.0, 1e10), 0.5)
    assert np.isclose(flip_probability(1e-12, 1e9), 0.5 * (1 - np.exp(-1e-3)))


def test_ey_dp_formulas_spot_value():
    """Hand evaluation of Eqs. 45/46 at one energy with a fixed tau_m."""
    s = Sample(MAT, per_cm3(1.5e17))
    sm = SpinModel(s, stage_a_mechanisms(s))
    E = ev(0.2)
    eta = MAT.Delta_so / s.Eg
    me = MAT.gamma.m_eff
    ey = 32 / 27 * (1 - me / M0) ** 2 * (eta / (1 + eta)) ** 2 * ((1 + eta / 2) / (1 + 2 * eta / 3)) ** 2 \
        * (E / s.Eg) ** 2
    assert np.isclose(MAT.A_EY * sm.ey_factor(E), ey)
    from gaas_mc.constants import HBAR
    B = 10 * HBAR**2 / (2 * M0)
    dp = 128 / 945 / 6 * MAT.Delta_so**2 * B**2 / ((1 + eta) * (1 + 2 * eta / 3)) * me**2 / HBAR**6 \
        * (1 - me / M0) * (E / s.Eg) ** 3
    assert np.isclose(MAT.Q_DP * sm.dp_factor(E), dp)


def test_bap_regimes():
    nd = Sample(MAT, per_cm3(1.5e17))
    dg = Sample(MAT, per_cm3(1e19))
    E = ev(np.array([0.01, 0.5]))
    smn, smd = SpinModel(nd, []), SpinModel(dg, [])
    v = bands.speed_of_E(E, MAT.gamma.m_eff, MAT.gamma.alpha)
    assert np.allclose(smn.bap(E), 2 * nd.inv_tau0 * v / nd.v_B * nd.a_B**3 * nd.p)        # Eq. 47
    Es = smd.bap_regime_boundary()
    assert E[0] < Es < E[1]
    assert np.isclose(smd.bap(E)[1], 2 * dg.inv_tau0 * np.sqrt(2 * dg.EF_h / MAT.m_hh) / dg.v_B
                      * E[1] / dg.EF_h * dg.a_B**3 * dg.p)                                 # Eq. 51
