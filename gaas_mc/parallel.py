"""Parallel ensembles: split N electrons into independent chunks and run them in worker processes
(or one at a time, e.g. as tasks of a cluster job array), then merge.

Electrons do not interact (holes are a fixed bath), so an ensemble can be split freely. Results
are reproducible and independent of the number of workers:
  * chunk i always holds electrons [i * chunk_size, (i + 1) * chunk_size) of the run;
  * its random stream is SeedSequence(seed).spawn(n_chunks)[i], used for photoexcitation and
    transport;
  * particle ids are offset to the global index, and chunks are merged in index order.
So run_parallel(..., max_workers=1) and run_parallel(..., max_workers=64) give identical Results
for the same seed and chunk_size. (Changing chunk_size changes the streams: statistically
equivalent, not bitwise.)

The user supplies two picklable callables (module-level functions or functools.partial of them):
    make_simulation() -> transport.Simulation
    make_ensemble(n, rng) -> particle.Ensemble          (e.g. photoexcitation)
Each worker process calls make_simulation() once and reuses it for all of its chunks (building
the rate tables is the expensive part of the setup).

Cluster use (one chunk per task; any scheduler):
    res = run_chunk(make_simulation, make_ensemble, n_total, chunk_index=i, chunk_size=c, seed=s)
    save_result(res, f"chunk_{i:04d}.npz")
    ...
    merged = merge_results([load_result(f) for f in sorted(files)])
See examples/parallel_run.py.

Chunk size: per-electron cost falls with the batch size (per-call overhead is spread over more
electrons): on the CPU ~2.3x cheaper at 32k than at 2k electrons per chunk (docs/PERFORMANCE.md).
The default is ceil(n_total / max_workers), capped at 50 000. On the GPU use max_workers=1 and
chunks of 1e5-1e6.
"""
from __future__ import annotations

import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields

import numpy as np

from .particle import Ensemble
from .surface import Emissions, SurfaceArrivals

DEFAULT_MAX_CHUNK = 50_000


def chunk_layout(n_total, chunk_size):
    """[(start, n), ...] for every chunk."""
    n_chunks = max(1, math.ceil(n_total / chunk_size))
    return [(i * chunk_size, min(chunk_size, n_total - i * chunk_size)) for i in range(n_chunks)]


def chunk_rng(seed, n_chunks, i):
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(n_chunks)[i])


def default_chunk_size(n_total, max_workers):
    return max(1, min(DEFAULT_MAX_CHUNK, math.ceil(n_total / max(1, max_workers))))


# ---------------------------------------------------------------------------------------------
_WORKER = {}          # per-process cache: the Simulation built by make_simulation()


def _simulation(make_simulation):
    key = pickle.dumps(make_simulation)        # equal for equal functions / partials in every task
    sim = _WORKER.get(key)
    if sim is None:
        sim = make_simulation()
        _WORKER[key] = sim
    return sim


def run_chunk(make_simulation, make_ensemble, n_total, chunk_index, chunk_size, seed,
              run_kwargs=None):
    """Run chunk `chunk_index` of an n_total-electron run. Returns its transport.Result with
    particle ids offset to the global index."""
    layout = chunk_layout(n_total, chunk_size)
    start, n = layout[chunk_index]
    rng = chunk_rng(seed, len(layout), chunk_index)
    ens = make_ensemble(n, rng)
    ens.pid = ens.pid + start
    sim = _simulation(make_simulation)
    return sim.run(ens, rng, **(run_kwargs or {}))


def _task(args):
    make_simulation, make_ensemble, n_total, i, chunk_size, seed, run_kwargs = args
    t0 = time.time()
    res = run_chunk(make_simulation, make_ensemble, n_total, i, chunk_size, seed, run_kwargs)
    return i, res, time.time() - t0


def run_parallel(make_simulation, make_ensemble, n_total, seed, max_workers=None, chunk_size=None,
                 run_kwargs=None, progress=None):
    """Run n_total electrons in chunks on up to max_workers processes (None: os.cpu_count();
    1: in this process, no pool) and return the merged Result.
    progress: optional callable(done, total, chunk_index, seconds)."""
    max_workers = max_workers or os.cpu_count() or 1
    chunk_size = chunk_size or default_chunk_size(n_total, max_workers)
    layout = chunk_layout(n_total, chunk_size)
    tasks = [(make_simulation, make_ensemble, n_total, i, chunk_size, seed, run_kwargs)
             for i in range(len(layout))]
    results = [None] * len(layout)
    if max_workers == 1 or len(layout) == 1:
        for done, t in enumerate(tasks, 1):
            i, res, dt = _task(t)
            results[i] = res
            if progress:
                progress(done, len(layout), i, dt)
    else:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(layout))) as pool:
            for done, (i, res, dt) in enumerate(pool.map(_task, tasks), 1):
                results[i] = res
                if progress:
                    progress(done, len(layout), i, dt)
    return merge_results(results)


# ---------------------------------------------------------------------------------------------
def _cat_dataclass(cls, parts):
    return cls(**{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(cls)})


def merge_results(results):
    """Merge chunk Results (in the given order) into one Result."""
    from .transport import Result, Snapshots
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("no results to merge")
    r0 = results[0]
    for r in results[1:]:
        if r.mechanism_names != r0.mechanism_names:
            raise ValueError("chunks were run with different mechanism sets")
    ens = Ensemble.concatenate([r.ensemble for r in results])
    M = len(r0.mechanism_names)
    arr = SurfaceArrivals.concatenate([r.arrivals for r in results if len(r.arrivals)], M,
                                      r0.mechanism_names)
    if len(arr) == 0:
        arr = r0.arrivals
    arr.band_edge_at_surface = r0.arrivals.band_edge_at_surface
    em = None
    if r0.emissions is not None:
        em = Emissions.concatenate([r.emissions for r in results])
    snaps = None
    if r0.snapshots is not None:
        s = [r.snapshots for r in results]
        snaps = Snapshots(s[0].times, *(np.concatenate([getattr(x, k) for x in s], axis=1)
                                        for k in ("E", "z", "spin", "valley")))
    log = {}
    if r0.event_log:
        log = {k: np.concatenate([r.event_log[k] for r in results]) for k in r0.event_log}
    return Result(ensemble=ens, arrivals=arr, snapshots=snaps, mechanism_names=r0.mechanism_names,
                  n_iterations=max(r.n_iterations for r in results),
                  n_real=sum(r.n_real for r in results), n_self=sum(r.n_self for r in results),
                  n_rejected=sum(r.n_rejected for r in results), flight_mode=r0.flight_mode,
                  event_log=log, emissions=em)


# ---------------------------------------------------------------------------------------------
def save_result(res, path):
    """Save a Result (ensemble, arrivals, emissions, counters) to one .npz file."""
    d = {f"ens_{f.name}": getattr(res.ensemble, f.name) for f in fields(res.ensemble)}
    for f in fields(res.arrivals):
        v = getattr(res.arrivals, f.name)
        d[f"arr_{f.name}"] = np.asarray(v)
    if res.emissions is not None:
        for f in fields(res.emissions):
            d[f"em_{f.name}"] = getattr(res.emissions, f.name)
    if res.snapshots is not None:
        for k in ("times", "E", "z", "spin", "valley"):
            d[f"snap_{k}"] = getattr(res.snapshots, k)
    for k, v in res.event_log.items():
        d[f"log_{k}"] = v
    d.update(mechanism_names=np.array(res.mechanism_names), n_iterations=res.n_iterations,
             n_real=res.n_real, n_self=res.n_self, n_rejected=res.n_rejected,
             flight_mode=np.array(res.flight_mode))
    np.savez_compressed(path, **d)


def load_result(path):
    from .transport import Result, Snapshots
    d = dict(np.load(path, allow_pickle=False))
    pick = lambda pre: {k[len(pre):]: v for k, v in d.items() if k.startswith(pre)}
    names = tuple(str(x) for x in d["mechanism_names"])
    a = pick("arr_")
    a["band_edge_at_surface"] = float(a["band_edge_at_surface"])
    a["mechanism_names"] = names
    em = pick("em_")
    sn = pick("snap_")
    return Result(ensemble=Ensemble(**pick("ens_")), arrivals=SurfaceArrivals(**a),
                  snapshots=Snapshots(**sn) if sn else None, mechanism_names=names,
                  n_iterations=int(d["n_iterations"]), n_real=d["n_real"], n_self=int(d["n_self"]),
                  n_rejected=d["n_rejected"], flight_mode=str(d["flight_mode"]),
                  event_log=pick("log_"), emissions=Emissions(**em) if em else None)
