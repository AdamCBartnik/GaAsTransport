"""Chunked / multi-process runs: reproducible and independent of the number of workers."""
from functools import partial

import numpy as np

from gaas_mc.assumptions import ModelAssumptions
from gaas_mc.constants import NM, PS, ev, per_cm3
from gaas_mc.excitation import photoexcite
from gaas_mc.fields import C21BandBending
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.parallel import chunk_layout, load_result, merge_results, run_chunk, run_parallel, save_result
from gaas_mc.scattering import build_mechanisms
from gaas_mc.spin import SpinModel
from gaas_mc.surface_c21 import C21Surface
from gaas_mc.transport import Simulation

MAT = gaas_chubenko2021()


def make_sim(t_max_ps=3.0):
    s = Sample(MAT, per_cm3(1e19))
    f = C21BandBending(s)
    a = ModelAssumptions(depletion_scattering="bulk")
    mech = build_mechanisms(s, a, "C", field=f)
    sm = SpinModel(s, [m for m in mech if m.valley_from == 0])
    return Simulation(s, mech, sm, field=f, surface=C21Surface(chi=ev(0.64), material=MAT),
                      t_max=t_max_ps * PS, assumptions=a)


def make_ens(n, rng, hv=1.8):
    s = Sample(MAT, per_cm3(1e19))
    ens = photoexcite(s, ev(hv), n, rng, assumptions=ModelAssumptions(depletion_scattering="bulk"))
    ens.z[:] = np.minimum(ens.z, 40 * NM)       # short test: start near the surface
    return ens


def _same(r1, r2):
    for name in ("z", "t", "E", "k", "spin", "status", "pid", "n_surface"):
        assert np.array_equal(getattr(r1.ensemble, name), getattr(r2.ensemble, name)), name
    assert np.array_equal(r1.emissions.pid, r2.emissions.pid)
    assert np.array_equal(r1.emissions.t, r2.emissions.t)
    assert np.array_equal(r1.n_real, r2.n_real) and r1.n_self == r2.n_self


def test_layout():
    assert chunk_layout(10, 4) == [(0, 4), (4, 4), (8, 2)]
    assert chunk_layout(8, 4) == [(0, 4), (4, 4)]


def test_workers_do_not_change_results(tmp_path):
    kw = dict(n_total=360, seed=123, chunk_size=120)
    r1 = run_parallel(make_sim, make_ens, max_workers=1, **kw)
    r3 = run_parallel(make_sim, make_ens, max_workers=3, **kw)
    _same(r1, r3)
    assert len(r1.ensemble) == 360 and np.array_equal(np.sort(r1.ensemble.pid), np.arange(360))
    assert len(r1.emissions) > 0
    # cluster path: chunks run separately, saved, loaded, merged
    files = []
    for i in range(3):
        res = run_chunk(make_sim, make_ens, 360, i, 120, 123)
        files.append(tmp_path / f"c{i}.npz")
        save_result(res, files[-1])
    _same(r1, merge_results([load_result(f) for f in files]))


def test_partial_callables():
    r = run_parallel(partial(make_sim, 1.0), partial(make_ens, hv=1.6), n_total=50, seed=1,
                     max_workers=2, chunk_size=25)
    assert len(r.ensemble) == 50
