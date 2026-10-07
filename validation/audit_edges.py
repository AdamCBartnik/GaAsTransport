"""Independent boundary timing and nonparabolic e-h root-domain diagnostics."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from gaas_mc import bands
from gaas_mc.constants import EV, HBAR, Q_E, ev, per_cm3
from gaas_mc.fields import C21BandBending
from gaas_mc.holes import HoleGas
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.scattering import ElectronHole, AcousticPhonon
from gaas_mc.surface_c21 import C21Surface
from gaas_mc.transport import Simulation


def main():
    mat=gaas_chubenko2021(); s=Sample(mat,per_cm3(1e19)); g=mat.gamma
    field=C21BandBending(s); rows=[]
    for energy in (.01,.1,.5):
        for cosine in (.05,1.):
            km=float(bands.k_of_E(ev(energy),g.m_eff,g.alpha))
            kx=km*np.sqrt(1-cosine*cosine); kz=-km*cosine
            # State in nm and 1/nm; independent adaptive Hamiltonian ODE with time in fs.
            def rhs(t,y):
                z=y[0]*1e-9; kn=y[1]*1e9
                E=bands.E_of_k(np.hypot(kx,kn),g.m_eff,g.alpha)
                return [HBAR*kn/(g.m_eff*(1+2*g.alpha*E))*1e-6,
                        -Q_E*float(field.Ez(np.array([z]))[0])/HBAR*1e-24]
            def surface(t,y): return y[0]
            surface.terminal=True; surface.direction=-1
            ode=solve_ivp(rhs,[0,2000],[field.W/1e-9,kz/1e9],events=surface,
                          rtol=2e-11,atol=1e-12,max_step=.2)
            exact=ode.t_events[0][0]
            for step in (.25,.125):
                sim=Simulation(s,[AcousticPhonon(s)],field=field,dt_max_field=step*1e-15)
                _,_,_,dt,_=sim.propagate(np.array([field.W]),np.array([[kx,0,kz]]),
                                        ev(np.array([energy])),np.zeros(1,int),np.array([2e-12]))
                rows.append(dict(E_eV=energy,cosine=cosine,step_fs=step,exact_fs=exact,
                                 error_fs=float(dt[0]/1e-15-exact)))
    # Count false no-root diagnoses among actual thermal-hole proposals.
    mech=ElectronHole(s,HoleGas(s)); rng=np.random.default_rng(551); roots=[]
    for energy in (.01,.3,1.):
        n=100000; E=np.full(n,ev(energy))
        k=bands.random_unit_vectors(n,rng)*bands.k_of_E(E,mech.m,mech.alpha)[:,None]
        kh=mech.holes.sample_k('hh',n,rng); mr=mech.m_R(E)
        gv=2*mr[:,None]*(kh/mech.mh-k/mech.m); gg=np.linalg.norm(gv,axis=1)
        accepted=rng.random(n)<2*gg*mech.beta/(gg**2+mech.beta**2)
        k,kh,E,mr,gv,gg=(x[accepted] for x in (k,kh,E,mr,gv,gg))
        u=rng.random(len(E)); ct=1-2*u/(1+gg**2*(1-u)/mech.beta**2)
        direction=bands.rotate_about(gv/gg[:,None],ct,2*np.pi*rng.random(len(E)))
        K=k+kh; c=mr[:,None]/mech.mh*K
        total=E+HBAR**2*np.sum(kh*kh,axis=1)/(2*mech.mh)
        _,ok=mech._solve_s(c,K,direction,total,gg)
        false_reject=0
        for i in np.flatnonzero(~ok):
            def f(u):
                ke=c[i]-.5*u*gg[i]*direction[i]; hole=K[i]-ke
                return float((bands.E_of_k(np.linalg.norm(ke),mech.m,mech.alpha)
                              +HBAR**2*np.dot(hole,hole)/(2*mech.mh)-total[i])/EV)
            minimum=minimize_scalar(f,bounds=(0,20),method='bounded')
            false_reject+=int(minimum.fun< -1e-12)
        roots.append(dict(E_eV=energy,attempts=n,passed_eq40=len(E),rejected_domain=int((~ok).sum()),
                          demonstrably_valid_positive_roots=false_reject))
    # A deterministic counterexample: the incident direction itself has a known positive root.
    k=np.array([[1e9,0.,0.]])
    kh=k*mech.mh/mech.m*.99
    E=bands.E_of_k(np.linalg.norm(k,axis=1),mech.m,mech.alpha)
    mr=mech.m_R(E); K=k+kh; c=mr[:,None]/mech.mh*K
    gv=2*mr[:,None]*(kh/mech.mh-k/mech.m); gg=np.linalg.norm(gv,axis=1)
    direction=gv/gg[:,None]
    total=E+HBAR**2*np.sum(kh*kh,axis=1)/(2*mech.mh)
    _,ok=mech._solve_s(c,K,direction,total,gg)
    ke=c-.5*gg[:,None]*direction
    residual=bands.E_of_k(np.linalg.norm(ke,axis=1),mech.m,mech.alpha)+HBAR**2*np.sum((K-ke)**2,axis=1)/(2*mech.mh)-total
    counterexample=dict(known_positive_root=float(gg[0]),accepted_domain=bool(ok[0]),
                        root_residual_eV=float(residual[0]/EV),electron_eV=float(E[0]/EV))
    result=dict(timing=rows,roots=roots,root_counterexample=counterexample)
    Path('validation/out/audit/edges.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
