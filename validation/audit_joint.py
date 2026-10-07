"""Reproducible audit measurements; outputs are separate from published benchmark caches."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstest

from gaas_mc import bands
from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import EV, HBAR, M0, PS, ev, per_cm3
from gaas_mc.excitation import photoexcite, layer_absorption
from gaas_mc.fast import FastSimulation
from gaas_mc.fields import C21BandBending
from gaas_mc.holes import HoleGas
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.particle import Ensemble
from gaas_mc.scattering import ElectronHole, build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.surface_c21 import C21Surface
from gaas_mc.transport import Simulation


def run(label, n=5000, device="cuda"):
    mat = gaas_chubenko2021()
    s = Sample(mat, per_cm3(1e19))
    a = ModelAssumptions(depletion_scattering="bulk", absorption_model="adachi1989")
    field = C21BandBending(s)
    mech = build_mechanisms(s, a, "C", field=field)
    spin = SpinModel(s, [m for m in mech if m.valley_from == 0])
    rows = []
    out = Path(__file__).parent / "out" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    for hv in (1.45, 1.6, 1.8, 2.2):
        surf = C21Surface(ev(.67), mat)
        sim = Simulation(s, mech, spin, field=field, surface=surf, assumptions=a)
        rng = np.random.default_rng(1841)
        ens = photoexcite(s, ev(hv), n, rng, assumptions=a)
        r = FastSimulation(sim, device).run(ens, rng)
        em = r.emissions
        p = em.p_vac
        E = np.sum(p*p, axis=1)/(2*M0)
        residual = np.max(np.abs(E-em.E_vac)/EV)
        assert residual < 1e-12
        row = dict(hv=hv, n=n, emitted=len(em), qe=layer_absorption(s, ev(hv), assumptions=a)*len(em)/n,
                   esp=float(em.spin.mean()), energy_eV=float(em.E_vac.mean()/EV),
                   mte_eV=float(np.mean(np.sum(p[:,:2]**2,axis=1)/(2*M0))/EV),
                   timing_ps=np.quantile(em.t/PS,[.1,.5,.9]).tolist(), energy_residual_eV=float(residual),
                   correlation=np.corrcoef(np.column_stack([em.t/PS,p/np.sqrt(2*M0*EV),em.E_vac/EV,em.spin,em.valley]).T).tolist())
        rows.append(row)
        np.savez(out/f"{label}_{hv:.2f}.npz",t=em.t,p=p,E=em.E_vac,spin=em.spin,valley=em.valley,pid=em.pid)
        print(json.dumps(row),flush=True)
    (out/f"{label}.json").write_text(json.dumps(rows,indent=2))


def equilibrium(n=20000, device="cuda"):
    """Compare the entire final energy CDF with independently integrated Kane DOS times MB."""
    mat=gaas_chubenko2021(); s=Sample(mat,per_cm3(1e19)); g=mat.gamma
    hg=HoleGas(s); rows=[]
    for pauli in ("fermi_dirac","step_c21"):
        mech=[ElectronHole(s,hg,b,pauli=pauli) for b in ("hh","lh")]
        rng=np.random.default_rng(203)
        E=np.full(n,ev(.15)); k=bands.random_unit_vectors(n,rng)*bands.k_of_E(E,g.m_eff,g.alpha)[:,None]
        ens=Ensemble.create(z=np.ones(n),k=k,E=E,spin=1)
        sim=Simulation(s,mech,surface="none",t_max=12*PS)
        r=FastSimulation(sim,device).run(ens,rng)
        x=np.linspace(0,40,40001); alpha=g.alpha*s.kT
        pdf=np.sqrt(x*(1+alpha*x))*(1+2*alpha*x)*np.exp(-x)
        cdf=cumulative_trapezoid(pdf,x,initial=0); cdf/=cdf[-1]
        final=r.ensemble.E/s.kT
        row=dict(pauli=pauli,n=n,mean=float(final.mean()),mean_se=float(final.std()/np.sqrt(n)),
                 expected=float(np.trapezoid(x*pdf,x)/np.trapezoid(pdf,x)),
                 ks=float(kstest(final,lambda q:np.interp(q,x,cdf)).statistic))
        rows.append(row); print(row,flush=True)
    Path('validation/out/audit/equilibrium.json').write_text(json.dumps(rows,indent=2))


def compare_engines(n=3000):
    mat=gaas_chubenko2021(); s=Sample(mat,per_cm3(1e19))
    a=ModelAssumptions(depletion_scattering='bulk'); field=C21BandBending(s)
    mech=build_mechanisms(s,a,'C',field=field)
    spin=SpinModel(s,[m for m in mech if m.valley_from==0])
    sim=Simulation(s,mech,spin,field=field,surface=C21Surface(ev(.67),mat),t_max=5*PS,assumptions=a)
    features={}; counts={}
    for j,device in enumerate(('reference','cpu','cuda')):
        rng=np.random.default_rng(311+j)
        ens=photoexcite(s,ev(1.9),n,rng,assumptions=a)
        ens.z[:]=np.minimum(ens.z,60e-9)
        r=sim.run(ens,rng) if device=='reference' else FastSimulation(sim,device).run(ens,rng)
        em=r.emissions; p=em.p_vac/np.sqrt(2*M0*EV)
        f=np.column_stack([em.t/PS,p,em.E_vac/EV,em.E_perp/EV,em.spin,em.valley==2])
        # First and second moments, including all cross products, of the joint distribution.
        features[device]=np.column_stack([f]+[f[:,i]*f[:,j] for i in range(f.shape[1]) for j in range(i,f.shape[1])])
        counts[device]=len(em)
    comparisons=[]
    for dev in ('cpu','cuda'):
        x,y=features['reference'],features[dev]
        se=np.sqrt(x.var(axis=0)/len(x)+y.var(axis=0)/len(y))
        z=np.divide(x.mean(axis=0)-y.mean(axis=0),se,out=np.zeros_like(se),where=se>0)
        comparisons.append(dict(device=dev,max_abs_z=float(np.max(np.abs(z))),
                                rms_z=float(np.sqrt(np.mean(z*z))),features=len(z)))
    output=dict(n_generated=n,emitted=counts,comparisons=comparisons,
                setup='1.9 eV, bulk, chi=.67 eV, depths capped at 60 nm, 5 ps')
    Path('validation/out/audit/engines.json').write_text(json.dumps(output,indent=2))
    print(output,flush=True)


if __name__ == "__main__":
    if sys.argv[1] == 'equilibrium': equilibrium()
    elif sys.argv[1] == 'engines': compare_engines()
    else: run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 5000)
