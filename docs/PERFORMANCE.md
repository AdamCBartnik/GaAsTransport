# Performance: engines, parallel runs, measurements

There is one physics setup and two implementations of the transport event loop.

| Engine | Where | What it is | Use it for |
|---|---|---|---|
| **Reference** `transport.Simulation` | NumPy (CPU); optionally CuPy arrays | Vectorized over electrons, one flight per loop iteration. The auditable implementation; every equation cites C21. | Validation, diagnostics (snapshots, event logs), any field or mechanism the fast engine does not support. |
| **Fast** `fast.FastSimulation(sim, device)` | Numba: `device="cpu"` (threads) or `"cuda"` (GPU) | One scalar per-electron event loop (`gaas_mc/fast/kernel_src.py`), compiled for both targets from the same source. Built from a reference `Simulation`: it reads that object's rate tables, mechanism parameters (every depletion variant), spin tables, field and boundaries, so no physics is set up twice. | Production runs. |

```python
sim = Simulation(sample, mechanisms, spin_model, field=field, surface=C21Surface(chi, mat), t_max=370 * PS)
res = FastSimulation(sim, "cuda").run(ensemble, rng)        # same Result as sim.run(ensemble, rng)
```

## Measured throughput (p = 1e19, C21 band bending, Stage C mechanisms)

Events = real + self-scattering events. This machine: 16 logical CPU cores, RTX 5070 Ti.

| Engine | Batch | Events/s | ns/event |
|---|---|---|---|
| Reference, 1 core | 2 000 electrons | ~0.16 M | 6 300 |
| Reference, 1 core | 100 000 electrons | ~0.6 M | 1 750 |
| Reference on CuPy (GPU) | 1 000 000 electrons | ~0.8 M | 1 200 |
| Fast CPU, 1 thread | 20 000 | 2.1 M | 475 |
| Fast CPU, 16 threads | 20 000 | 21 M | 47 |
| Fast GPU | 200 000 | 160 M | 6 |
| Fast GPU | 1 000 000 | 172-179 M | 6 |

* The reference is limited by per-call overhead: each loop iteration issues ~2000 small array
  operations (one masked pass per mechanism), so larger batches help (2.3x from 2k to 32k electrons
  per process). On the GPU each of those operations costs a kernel launch; the CuPy backend is
  therefore only about one CPU core fast. It is kept as a correctness option, not for speed.
* Fast engine, CPU: about 3.7x a reference core at its best batch size, linear in threads. Results
  do not depend on the number of threads.
* Fast engine, GPU: about 8x the 16-thread CPU build and ~20x the reference on 14 cores.
* Compilation: once per process, ~35 s (CPU) and ~10 s (GPU).
* `local` depletion scattering costs ~10% more than `bulk`.

## Validation of the fast engine (`tests/test_fast_engine.py`)

* Exact or round-off agreement with the reference: energy-grid and hole-CDF searches (lookup
  tables give identical indices), all 34 mechanism rates at bulk and band-bending positions (both
  depletion modes), spin-relaxation tables, flights (field and field-free, reflecting walls, image
  field), the C21 transmission (both matching masses) and K_par folding.
* Distributions: final states of every mechanism and depletion variant (acceptance, final energy,
  scattering angle, final valley) at two energies; hot-electron relaxation (valley populations,
  energy, spin, event counts); the C21 surface model end to end (emitted / trapped / timed-out
  fractions, emitted ESP, vacuum energy, transverse energy); host-path vs kernel-path surface model.
* Determinism: identical results for 1 and 4 threads; GPU matches CPU (the per-electron streams
  are the same; GPU arithmetic contracts multiply-adds, so long histories can diverge in the last
  bits, which is statistically irrelevant).

## Differences from the reference

* **Random numbers:** one xoroshiro128+ stream per electron, expanded from a 64-bit seed per
  electron with SplitMix64 (raw seeds give a biased first draw: found and fixed during
  validation). Results are statistically, not draw-for-draw, equal to the reference.
* **Surface encounters:** the absorbing surface and C21Surface models are evaluated inside the
  kernel (first-arrival and emission records in per-electron arrays). Any other surface model with
  the reference interface `interact(k, E, valley, K, rng, pid)` is called on the host between
  kernel launches (slower, but your own model needs no kernel code).
* **Surface branches:** `FastSimulation(sim, surface_models=[...])` and
  `run(..., surface_branch=b)` run several C21 models (e.g. several chi) in one launch.
* **Repeated bounces (aggregation of short returns).** An electron reflected at z = 0 while the surface field
  pushes it back returns after t_ret = 2 hbar k_z / |F(0)|. The reference treats each return as a
  flight plus a new surface-model call. For grazing incidence (k_z -> 0) the number of returns before
  the next scattering diverges, ~ 1 / (Gamma0 t_ret): one electron in 1e5 needed ~1e9 returns, which
  stalls any per-return algorithm (the reference included). Between returns the state repeats
  exactly, so each return is the same Bernoulli trial with the transmission T of the previous one,
  and by memorylessness one flight time covers the train: floor(dt / t_ret) returns, a geometric
  number of failures before an emission, then the remaining time is flown normally. This is exact in
  distribution; encounter counts stay exact bookkeeping. Tested against the reference's one-by-one
  bounces where those are affordable (`test_bounce_train_matches_brute_force`).
  Encounter-count *means* are dominated by these rare electrons; report medians and percentiles.
  The geometric aggregation is exact for the explicit short-return algorithm, whose use of F(0)
  is a constant-force approximation in the spatially varying C21 field. Both engines honor
  `surface_bounce_aggregation=False`. Reference runs with snapshots use explicit returns so that
  snapshot intervals are not skipped. See `docs/PHYSICS_AUDIT.md` for independent timing checks.
* **GPU launches** are capped by an adaptive flight budget (~0.15 s per launch), well below the
  Windows display-driver watchdog (~2 s).
* Not supported in the fast engine (raises; use the reference): snapshots, event logs, fields
  other than none / uniform / C21 band bending, mechanisms other than the five standard classes.
  `Result.n_iterations` is the largest number of flights of any electron.

## Parallel runs (`gaas_mc/parallel.py`)

`run_parallel(make_simulation, make_ensemble, n_total, seed, max_workers, chunk_size)` splits a run
into chunks with independent `SeedSequence` streams; results are independent of `max_workers`.
`make_simulation` may return a reference `Simulation` or a `FastSimulation` (both have
`run(ensemble, rng)`).

* Clusters, reference engine: one process per core (`max_workers` = cores), chunks of 20k-50k.
* Clusters, fast CPU engine: one process per node (or per socket) with Numba threads
  (`NUMBA_NUM_THREADS`), `max_workers` = 1 per node; several nodes via `run_chunk` per task (job
  array) + `merge_results`.
* GPU: one process, `FastSimulation(sim, "cuda")`, batches of 1e5-1e6 electrons.
