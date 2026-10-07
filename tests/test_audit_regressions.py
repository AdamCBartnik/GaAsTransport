"""Independent limits and output invariants found during the physics audit."""
from dataclasses import replace

import numpy as np
import pytest

from gaas_mc import bands
from gaas_mc.constants import EV, HBAR, M0, PS, ev, per_cm3
from gaas_mc.excitation import excess_energy
from gaas_mc.fields import UniformField
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import AcousticPhonon
from gaas_mc.surface_c21 import C21Surface, REFLECT
from gaas_mc.transport import Simulation

MAT = gaas_chubenko2021()


def test_emitted_vector_is_invariant_under_surface_reciprocal_translation():
    surf = C21Surface(ev(-1), MAT)
    E = ev(np.array([1.0]))
    k = np.array([[1e8, 2e8, -bands.k_of_E(E[0], MAT.gamma.m_eff, MAT.gamma.alpha)]])
    reciprocal = 2*np.pi/MAT.a_lat*np.array([[1.,1.,0.]])
    def momentum(K):
        return surf.interact(k,E,np.zeros(1,int),K,np.random.default_rng(1))[1]['p_vac']
    np.testing.assert_allclose(momentum(k+reciprocal)/HBAR,momentum(k)/HBAR,rtol=1e-12,atol=1e-4)


@pytest.mark.parametrize('band',[0,1,2])
@pytest.mark.parametrize('alpha_scale',[0.,1e-12,1.])
def test_vertical_excitation_energy_and_momentum_limit(band,alpha_scale):
    mat=replace(MAT,valleys=(replace(MAT.gamma,alpha=MAT.gamma.alpha*alpha_scale),)+MAT.valleys[1:])
    sample=Sample(mat,per_cm3(1e19)); hv=ev(2.)
    E=excess_energy(sample,hv,band)
    mh=(mat.m_hh,mat.m_lh,mat.m_so)[band]
    # Equal wavevectors: hole energy = (me/mh) E(1+alpha E).
    total=E+(mat.gamma.m_eff/mh)*E*(1+mat.gamma.alpha*E)
    target=hv-sample.Eg-(mat.Delta_so if band==2 else 0)
    assert np.isfinite(E)
    np.testing.assert_allclose(total/EV,target/EV,rtol=2e-13,atol=0)


class RejectingAcoustic(AcousticPhonon):
    def rate(self,E): return np.full_like(E,1e5)
    def scatter(self,k,E,rng): return k.copy(),np.zeros(len(E),bool),np.zeros(len(E),int)


class MirrorModel:
    is_surface_model=True
    def interact(self,k,E,valley,K,rng,pid=None):
        return np.full(len(E),REFLECT),dict(T=np.zeros(len(E)))


def test_snapshots_inside_bounce_train_are_not_lost():
    s=Sample(MAT,per_cm3(1e19))
    k=np.array([[1e9,0,1e7]])
    E=bands.E_of_k(np.linalg.norm(k,axis=1),MAT.gamma.m_eff,MAT.gamma.alpha)
    ens=Ensemble.create(z=np.zeros(1),k=k,E=E,spin=1)
    ts=np.array([.001,.002,.003])*PS
    sim=Simulation(s,[RejectingAcoustic(s)],field=UniformField(1e8),surface=MirrorModel(),
                   t_max=.004*PS,snapshot_times=ts)
    result=sim.run(ens,np.random.default_rng(2))
    assert np.isfinite(result.snapshots.E).all()
    assert (result.snapshots.valley==0).all()


def test_fast_engine_preserves_bounce_aggregation_switch():
    pytest.importorskip('numba')
    from gaas_mc.fast.engine import pack_tables,I
    s=Sample(MAT,per_cm3(1e19))
    sim=Simulation(s,[AcousticPhonon(s)],surface_bounce_aggregation=False)
    tables=pack_tables(sim)
    assert 'I_BOUNCE_AGGREGATION' in I
    assert tables['icfg'][I['I_BOUNCE_AGGREGATION']]==0


def test_folded_vector_kernel_matches_reciprocal_lattice():
    pytest.importorskip('numba')
    from gaas_mc.fast.testing import _ns
    ns=_ns(); g=2*np.pi/MAT.a_lat
    assert 'fold_kpar_vector' in ns
    x,y=ns['fold_kpar_vector'](.6*g,.55*g,MAT.a_lat)
    np.testing.assert_allclose(np.array([x,y])/g,[-.4,-.45],rtol=1e-14,atol=0)


@pytest.mark.parametrize('energy_eV',[.670001,.68,.9,1.5])
def test_triangular_barrier_against_independent_schrodinger_ode(energy_eV):
    from scipy.integrate import solve_ivp
    surf=C21Surface(ev(.67),MAT,n_slices=400)
    E=ev(energy_eV); m=MAT.gamma.m_eff
    kin=bands.k_of_E(E,m,MAT.gamma.alpha)
    kout=np.sqrt(2*M0*(E-surf.chi))/HBAR
    # u=x/L, state=(psi, dpsi/du); integrate continuum barrier backwards from vacuum.
    def ode(u,y):
        potential=surf.chi+surf.E_b*(1-u)
        return [y[1],-2*M0*(E-potential)*surf.L_b**2/HBAR**2*y[0]]
    sol=solve_ivp(ode,[1.,0.],np.array([1.+0j,1j*kout*surf.L_b]),rtol=1e-11,atol=1e-13)
    assert sol.success
    psi,du=sol.y[:,-1]
    incoming=.5*(psi+(m/M0)*du/(1j*kin*surf.L_b))
    expected=(kout/M0)/(kin/m)/abs(incoming)**2
    actual=surf.transmission(np.array([E]),np.array([kin]),np.array([m]),np.zeros(1))[0]
    assert 0 <= actual <= 1
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=0)


def test_fast_bounce_disable_changes_execution_and_preserves_statistics():
    pytest.importorskip('numba')
    from gaas_mc.fast import FastSimulation
    s=Sample(MAT,per_cm3(1e19))
    n=500
    E=np.full(n,ev(.7)); km=bands.k_of_E(E,MAT.gamma.m_eff,MAT.gamma.alpha)
    kz=np.full(n,1e7)
    k=np.column_stack([np.sqrt(km*km-kz*kz),np.zeros(n),kz])
    ens=Ensemble.create(z=np.zeros(n),k=k,E=E,spin=1)
    results=[]
    for enabled in (False,True):
        sim=Simulation(s,[AcousticPhonon(s)],field=UniformField(1e8),surface=C21Surface(ev(.55),MAT),
                       t_max=.004*PS,surface_bounce_aggregation=enabled)
        results.append(FastSimulation(sim,'cpu').run(ens,np.random.default_rng(171)))
    brute,aggregated=results
    assert brute.n_iterations > aggregated.n_iterations
    fb,fa=len(brute.emissions)/n,len(aggregated.emissions)/n
    assert fb>.1
    assert abs(fb-fa)<5*np.sqrt(2*fb*(1-fb)/n)
    # Both must consume actual elapsed time and preserve nonparabolic energy at emission.
    for result in results:
        np.testing.assert_allclose(result.ensemble.time_in_valley.sum(1)/PS,result.ensemble.t/PS,
                                   rtol=1e-12,atol=1e-16)
        em=result.emissions
        np.testing.assert_allclose(np.sum(em.p_vac**2,axis=1)/(2*M0*EV),em.E_vac/EV,rtol=1e-12,atol=0)


def test_zero_emission_run_returns_empty_particle_group():
    from gaas_mc.emission import simulate_emission
    run = simulate_emission(hv_eV=1.6, n=200, t_max_ps=2.0, chi_eV=3.0, device="reference", seed=1)
    assert run.n_emitted == 0 and run.qe == 0 and len(run.particle_group) == 0


@pytest.mark.parametrize("engine", ["ref", "cpu"])
def test_field_flight_leaving_the_rate_table_raises(engine):
    """An electron accelerated by the band bending beyond E_table_max during a flight must stop the
    run instead of scattering with clamped rates."""
    from gaas_mc.fields import C21BandBending
    s = Sample(MAT, per_cm3(1e19))
    f = C21BandBending(s)
    sim = Simulation(s, [AcousticPhonon(s)], field=f, E_table_max=ev(0.6), t_max=2 * PS)
    n = 50
    E = np.full(n, ev(0.45))
    k = np.zeros((n, 3)); k[:, 2] = -bands.k_of_E(E, MAT.gamma.m_eff, MAT.gamma.alpha)
    ens = Ensemble.create(z=np.full(n, 0.9 * f.W), k=k, E=E, spin=1)
    with pytest.raises(RuntimeError, match="rate table"):
        if engine == "ref":
            sim.run(ens, np.random.default_rng(0))
        else:
            from gaas_mc.fast import FastSimulation
            FastSimulation(sim, "cpu").run(ens, np.random.default_rng(0))
