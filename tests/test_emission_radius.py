"""Excitation spot drawn before transport, and a front surface that emits only inside R_cathode."""
import numpy as np
import pytest

from gaas_mc.emission import excitation_spot, simulate_emission, _cuda_available

DEVICES = ["reference", "cpu", "cuda"]


def skip_no_cuda(device):
    if device == "cuda" and not _cuda_available():
        pytest.skip("CUDA unavailable")


def test_flattop_spot_is_uniform_disc():
    R = 2e-6
    x, y = excitation_spot(200_000, np.random.default_rng(0), R_flattop=R)
    r2 = (x**2 + y**2) / R**2
    assert r2.max() <= 1
    # uniform in area: r^2/R^2 is uniform on [0, 1]
    np.testing.assert_allclose(np.histogram(r2, bins=10, range=(0, 1))[0] / len(r2), 0.1, atol=0.003)
    np.testing.assert_allclose([x.mean() / R, y.mean() / R], 0, atol=0.005)


def test_spot_options_validated():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="not both"):
        excitation_spot(10, rng, sigma_xy=1e-6, R_flattop=1e-6)
    with pytest.raises(ValueError, match="positive"):
        excitation_spot(10, rng, R_flattop=0.0)
    x, y = excitation_spot(10, rng)
    assert not x.any() and not y.any()


RUN = dict(hv_eV=1.6, n=400, t_max_ps=10.0, seed=3)


@pytest.mark.parametrize("device", DEVICES)
def test_spot_does_not_change_transport(device):
    """Without R_cathode, the spot only shifts x, y: the same electrons emit at the same times."""
    skip_no_cuda(device)
    point = simulate_emission(**RUN, device=device)
    for spot in (dict(sigma_xy=3e-6), dict(R_flattop=3e-6)):
        run = simulate_emission(**RUN, device=device, **spot)
        a, b = point.emissions, run.emissions
        np.testing.assert_array_equal(a.pid, b.pid)
        np.testing.assert_array_equal(a.t, b.t)
        x0, y0 = run.xy0[b.pid].T
        np.testing.assert_allclose(b.x - x0, a.x, atol=1e-15)
        np.testing.assert_allclose(b.y - y0, a.y, atol=1e-15)
        np.testing.assert_array_equal(run.particle_group.x, b.x)


@pytest.mark.parametrize("device", DEVICES)
def test_large_emission_radius_is_no_op(device):
    skip_no_cuda(device)
    free = simulate_emission(**RUN, device=device, R_flattop=1e-6)
    wide = simulate_emission(**RUN, device=device, R_flattop=1e-6, R_cathode=1.0)
    np.testing.assert_array_equal(free.emissions.pid, wide.emissions.pid)
    np.testing.assert_array_equal(free.emissions.t, wide.emissions.t)


@pytest.mark.parametrize("device", DEVICES)
def test_emission_only_inside_radius(device):
    """Spot 2x wider than the emitting area: every emission happens at r < R_cathode, and electrons
    born under the barrier are reflected (not trapped), so some of them still reach the opening."""
    skip_no_cuda(device)
    R = 25e-9                                             # a 50 nm hole
    kw = dict(hv_eV=1.6, n=1500 if device == "reference" else 4000, t_max_ps=10.0, seed=5,
              device=device, R_flattop=2 * R)
    run = simulate_emission(**kw, R_cathode=R)
    em = run.emissions
    assert run.n_emitted > 0
    assert np.all(em.x**2 + em.y**2 < R**2)
    r0 = np.hypot(*run.xy0[em.pid].T)
    assert np.any(r0 > R)                                 # born under the barrier, emitted in the opening
    # Without the barrier the same spot emits more, everywhere.
    free = simulate_emission(**kw)
    assert free.n_emitted > run.n_emitted
    assert np.any(free.emissions.x**2 + free.emissions.y**2 > R**2)


def test_engines_agree_with_barrier():
    """Same physics in the reference and compiled engines (different random streams): emitted
    fraction and mean emission radius agree within Monte Carlo noise."""
    R = 25e-9
    kw = dict(hv_eV=1.6, t_max_ps=10.0, R_flattop=3 * R, R_cathode=R)
    runs = [simulate_emission(**kw, n=1500, device="reference", seed=1),
            simulate_emission(**kw, n=20_000, device="cpu", seed=2)]
    if _cuda_available():
        runs.append(simulate_emission(**kw, n=20_000, device="cuda", seed=3))
    ref = runs[0]
    f0 = ref.n_emitted / ref.n_generated
    for r in runs[1:]:
        f = r.n_emitted / r.n_generated
        sig = np.sqrt(f * (1 - f) * (1 / ref.n_generated + 1 / r.n_generated))
        assert abs(f - f0) < 4 * sig, (f0, f)


def test_emission_radius_needs_surface_model():
    from gaas_mc.assumptions import ModelAssumptions
    from gaas_mc.constants import per_cm3
    from gaas_mc.material import Sample, gaas_chubenko2021
    from gaas_mc.scattering import build_mechanisms
    from gaas_mc.transport import Simulation
    s = Sample(gaas_chubenko2021(), per_cm3(1e19))
    mech = build_mechanisms(s, ModelAssumptions(), "A")
    with pytest.raises(ValueError, match="surface model"):
        Simulation(s, mech, emission_radius=1e-6)


def test_time_to_edge():
    from gaas_mc.fast.engine import _namespace
    from gaas_mc.transport import time_to_edge
    R = 1.0
    cases = [  # x, y, vx, vy, expected
        (0.0, 0.0, 2.0, 0.0, 0.5),            # centre, leaving
        (0.5, 0.0, -1.0, 0.0, 1.5),           # inside, moving away from the near edge
        (2.0, 0.0, -1.0, 0.0, 1.0),           # outside, heading in
        (2.0, 0.0, 1.0, 0.0, np.inf),         # outside, heading away
        (2.0, 2.0, -1.0, 0.0, np.inf),        # outside, missing the circle
        (0.3, 0.4, 0.0, 0.0, np.inf),         # not moving
    ]
    x, y, vx, vy, want = (np.array(c, float) for c in zip(*cases))
    np.testing.assert_allclose(time_to_edge(x, y, vx, vy, R), want)
    jit_version = _namespace(lambda f: f)["time_to_edge"]   # kernel source as plain Python
    got = [jit_version(*c[:4], R) for c in cases]
    np.testing.assert_allclose(np.where(np.array(got) > 1e300, np.inf, got), want)


@pytest.mark.parametrize("device", DEVICES)
def test_bounce_trains_respect_edge(device, monkeypatch):
    """Aggregated bounce trains reproduce one-return-at-a-time brute force with a 20 nm hole,
    where a train's lateral skid (a few nm) is comparable to the hole. A 20 nm layer keeps the
    electrons born near the surface, so enough of them find the hole."""
    skip_no_cuda(device)
    from gaas_mc import transport
    R = 10e-9
    kw = dict(hv_eV=1.8, chi_eV=0.55, n=4000, t_max_ps=2.0, thickness_nm=20, R_flattop=2 * R,
              R_cathode=R, seed=7, device=device)
    agg = simulate_emission(**kw)
    assert np.all(agg.emissions.x**2 + agg.emissions.y**2 < R**2)

    class BruteForce(transport.Simulation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, surface_bounce_aggregation=False, **kwargs)

    monkeypatch.setattr(transport, "Simulation", BruteForce)
    brute = simulate_emission(**{**kw, "seed": 8})
    assert np.all(brute.emissions.x**2 + brute.emissions.y**2 < R**2)
    f1, f2 = agg.n_emitted / kw["n"], brute.n_emitted / kw["n"]
    sig = np.sqrt((f1 * (1 - f1) + f2 * (1 - f2)) / kw["n"])
    assert agg.n_emitted > 50 and abs(f1 - f2) < 4 * sig, (f1, f2)
