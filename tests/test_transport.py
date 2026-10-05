import numpy as np
import pytest

from gaas_mc import bands
from gaas_mc.constants import NM, PS, ev, per_cm3
from gaas_mc.excitation import photoexcite
from gaas_mc.fields import UniformField
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import AcousticPhonon, stage_a_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.transport import Simulation

MAT = gaas_chubenko2021()
S = Sample(MAT, per_cm3(1.5e17))
G = MAT.gamma


def bulk_ensemble(n, E0, rng, z=1.0):
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(np.full(n, E0), G.m_eff, G.alpha)[:, None]
    return Ensemble.create(z=np.full(n, z), k=k, E=np.full(n, E0), spin=1)


def test_energy_changes_are_phonon_quanta():
    rng = np.random.default_rng(10)
    sim = Simulation(S, stage_a_mechanisms(S), surface="none", t_max=5 * PS, log_events=True)
    r = sim.run(bulk_ensemble(500, ev(0.4), rng), rng)
    dE = r.event_log["E_after"] - r.event_log["E_before"]
    q = dE / MAT.hw0
    assert np.allclose(q, np.round(q), atol=1e-8)
    assert set(np.unique(np.round(q)).astype(int)) <= {-1, 0, 1}
    mech = r.event_log["mech"]
    names = r.mechanism_names
    assert np.allclose(dE[mech == names.index("acoustic[Gamma]")], 0, atol=1e-30)


def test_no_forbidden_emission_near_threshold():
    """Electrons sitting just below hw0 must never emit (regression: interpolation across the
    threshold once produced NaN wavevectors)."""
    rng = np.random.default_rng(15)
    sim = Simulation(S, stage_a_mechanisms(S), surface="none", t_max=2 * PS, log_events=True)
    for E0 in MAT.hw0 * np.array([0.9999, 0.99999999, 1.00000001]):
        r = sim.run(bulk_ensemble(3000, E0, rng), rng)
        assert np.all(np.isfinite(r.ensemble.E)) and np.all(np.isfinite(r.ensemble.k))
        em = r.event_log["mech"] == r.mechanism_names.index("pop_em[Gamma]")
        assert np.all(r.event_log["E_before"][em] > MAT.hw0)


def test_stationary_comb_distribution():
    """Elastic acoustic + POP keep E on the comb E0 + n hw0 (ambiguity A17). Detailed balance
    then gives stationary populations P_n ~ g(E_n) exp(-E_n / kT)."""
    rng = np.random.default_rng(11)
    E0 = 0.3 * MAT.hw0
    n = 20_000
    sim = Simulation(S, stage_a_mechanisms(S), surface="none", t_max=60 * PS, snapshot_times=[60 * PS])
    r = sim.run(bulk_ensemble(n, E0, rng), rng)
    E = r.snapshots.E[0]
    nidx = np.round((E - E0) / MAT.hw0).astype(int)
    assert np.allclose(E, E0 + nidx * MAT.hw0, rtol=1e-9)
    En = E0 + np.arange(12) * MAT.hw0
    w = bands.dos(En, G.m_eff, G.alpha) * np.exp(-En / S.kT)
    expected = w / w.sum() * n
    counts = np.bincount(nidx, minlength=12)[:12]
    ok = expected > 50
    chi2 = np.sum((counts[ok] - expected[ok]) ** 2 / expected[ok])
    assert chi2 < 3 * ok.sum()


def test_surface_arrival_geometry():
    rng = np.random.default_rng(12)
    ac = AcousticPhonon(S)
    ens = bulk_ensemble(2000, ev(0.1), rng, z=50 * NM)
    r = Simulation(S, [ac], t_max=200 * PS).run(ens, rng)
    a = r.arrivals
    assert len(a) > 0
    assert np.all(a.k[:, 2] < 0)                       # moving toward the surface
    assert np.allclose(a.E, ev(0.1))                   # elastic only
    assert np.all(a.t > 0) and np.all(np.isclose(a.z0, 50 * NM))
    assert len(a) + (r.ensemble.status == 2).sum() == 2000


def test_reproducible_with_seed():
    def run(seed):
        rng = np.random.default_rng(seed)
        ens = photoexcite(S, ev(1.6), 300, rng, 300 * NM)
        sim = Simulation(S, stage_a_mechanisms(S), SpinModel(S, stage_a_mechanisms(S)), t_max=50 * PS)
        return sim.run(ens, rng).arrivals
    a, b = run(42), run(42)
    assert np.array_equal(a.t, b.t) and np.array_equal(a.k, b.k) and np.array_equal(a.spin, b.spin)


class ConstantSpin:
    """Stub spin model with a constant 1/tau_s, to test the Eq. 54 bookkeeping."""
    def __init__(self, inv):
        self.inv = inv

    def total(self, E):
        return np.full_like(E, self.inv)


def test_eq54_gives_exponential_esp_decay():
    """With constant tau_s, Eq. 54 applied at real events (dt measured between real events,
    self-scatterings skipped) gives ESP(t) = ESP(0) exp(-t/tau_s) exactly in expectation."""
    rng = np.random.default_rng(13)
    tau_s = 5 * PS
    times = np.array([2, 5, 10]) * PS
    sim = Simulation(S, stage_a_mechanisms(S), ConstantSpin(1 / tau_s), surface="none",
                     t_max=10 * PS, snapshot_times=times)
    r = sim.run(bulk_ensemble(40_000, ev(0.1), rng), rng)
    # snapshots record the spin as of the last real event before ts; the flip test for the
    # partial interval is pending, so compare against the time of that last event.
    for i, ts in enumerate(times):
        esp = r.snapshots.esp(i)
        lo, hi = np.exp(-ts / tau_s), np.exp(-(ts - 1 * PS) / tau_s)
        assert lo - 0.02 < esp < hi + 0.02


def test_uniform_field_free_flight_conserves_energy():
    rng = np.random.default_rng(14)
    F = 1e6                                            # V/m, pushes electrons toward -z
    sim = Simulation(S, stage_a_mechanisms(S), field=UniformField(F), surface="none")
    ens = bulk_ensemble(50, ev(0.05), rng, z=1e-6)
    dt = np.full(50, 0.3 * PS)
    z, k, E, used, ev_ = sim.propagate(ens.z, ens.k, ens.E, ens.valley, dt)
    from gaas_mc.constants import HBAR, Q_E
    assert np.allclose(k[:, 2], ens.k[:, 2] - Q_E * F * dt / HBAR, rtol=1e-10)
    tot0 = ens.E + Q_E * F * ens.z
    tot1 = E + Q_E * F * z
    assert np.allclose(tot1, tot0, rtol=0, atol=1e-6 * 1.6e-19)   # < 1 ueV drift
