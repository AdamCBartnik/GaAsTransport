"""Fast engine: the transport event loop as per-electron compiled kernels (Numba), for CPU threads
and for CUDA GPUs, from ONE scalar source (gaas_mc/fast/kernel_src.py).

The vectorized NumPy code (gaas_mc/transport.py) is the reference implementation. The fast engine
does not set up any physics itself: it is constructed from a reference Simulation and reads its
rate tables, mechanism parameters (including every depletion variant), spin-relaxation tables,
field, boundaries and options. It returns the same transport.Result.

    sim = Simulation(sample, mechanisms, spin_model, field=..., surface=..., t_max=...)
    fast = FastSimulation(sim, device="cpu")          # or "cuda"
    res = fast.run(ensemble, rng)                      # same signature as Simulation.run

Differences from the reference (all documented in docs/PERFORMANCE.md):
  * random numbers: one xoroshiro128+ stream per electron (seeded from rng), so results do not
    depend on the number of threads or on CPU vs GPU, but differ from the reference draw by draw;
    the comparison is statistical (tests/test_fast_engine.py);
  * the surface model is applied on the host, in NumPy, between kernel launches: the kernel stops
    an electron at each surface encounter. Any surface model with the reference interface works;
  * not supported (use the reference): snapshots, event logs, fields other than none / uniform /
    C21 band bending, mechanisms other than the five standard classes;
  * Result.n_iterations is the largest number of flights of any electron.
"""
from __future__ import annotations

import cmath
import math
import time
import warnings
from pathlib import Path

import numpy as np

from ..constants import HBAR, M0, Q_E
from ..depletion import LocalMechanism
from ..fields import C21BandBending, NoField, UniformField
from ..particle import ALIVE, BACK, EMITTED, SURFACE, TIMEOUT, TRAPPED, Ensemble
from ..scattering import AcousticPhonon, ElectronHole, IonizedImpurity, Intervalley, PolarOptical
from ..surface import Emissions, SurfaceArrivals
from ..valleys import N_EQUIV, valley_center

PENDING = 6
ERR_RATE, ERR_FORBIDDEN, ERR_NONFINITE, ERR_NOMECH, ERR_TABLE = 10, 11, 12, 13, 14
ERRORS = {ERR_RATE: "total rate exceeds the flight rate; table/margin problem",
          ERR_FORBIDDEN: "forbidden intervalley transition selected",
          ERR_NONFINITE: "non-finite energy after scattering",
          ERR_NOMECH: "an electron is in a valley without mechanisms",
          ERR_TABLE: "electron energy left the rate table; raise E_table_max"}
MT_ACOUSTIC, MT_POP, MT_IMPURITY, MT_EH, MT_IV = 0, 1, 2, 3, 4
EV = dict(EV_NONE=0, EV_SURFACE=1, EV_BACK=2, EV_REGION=3, EV_WALL=4)
# parameter columns
PCOLS = ["P_HW", "P_EMISSION", "P_A", "P_SCREENED_ANGLE", "P_EBETA", "P_MH", "P_BETA", "P_KT",
         "P_GAS", "P_MASS_MODEL", "P_PAULI", "P_EFH", "P_ETA", "P_STATS", "P_DELTA", "P_VTO"]
P = {name: i for i, name in enumerate(PCOLS)}
CCOLS = ["C_TMAX", "C_EMAX", "C_ZFIELD", "C_DTMAX", "C_ZBACK", "C_PHI0", "C_DPHI", "C_INV_EMAX",
         "C_ALAT"]
ICOLS = ["I_SURFACE", "I_HAS_MODEL", "I_HAS_BACK", "I_BACK", "I_FLIGHT", "I_HAS_DEPL", "I_NPHI",
         "I_FLIP_ARRIVAL", "I_SURF_MODE"]
SURF_HOST, SURF_ABSORB, SURF_C21 = 0, 1, 2
C = {name: i for i, name in enumerate(CCOLS)}
I = {name: i for i, name in enumerate(ICOLS)}

_SRC = Path(__file__).with_name("kernel_src.py")
_BUILT = {}
_NS = {}            # compiled CPU namespace (used by gaas_mc/fast/testing.py)


def _namespace(jit):
    ns = dict(jit=jit, math=math, cmath=cmath, np=np, HBAR=HBAR, Q_E=Q_E, M0=M0, ALIVE=ALIVE,
              SURFACE=SURFACE, TIMEOUT=TIMEOUT, BACK=BACK, EMITTED=EMITTED, TRAPPED=TRAPPED,
              PENDING=PENDING, ERR_RATE=ERR_RATE,
              ERR_FORBIDDEN=ERR_FORBIDDEN, ERR_NONFINITE=ERR_NONFINITE, ERR_NOMECH=ERR_NOMECH,
              ERR_TABLE=ERR_TABLE, MT_ACOUSTIC=MT_ACOUSTIC, MT_POP=MT_POP, MT_IMPURITY=MT_IMPURITY,
              MT_EH=MT_EH, MT_IV=MT_IV, **EV, **P, **C, **I)
    exec(compile(_SRC.read_text(encoding="utf-8"), str(_SRC), "exec"), ns)
    return ns


def build(device):
    """Compile (lazily, once per process) the driver for device "cpu" or "cuda"."""
    if device in _BUILT:
        return _BUILT[device]
    import numba
    args = ", ".join(STATE_ORDER + TABLE_ORDER)
    if device == "cpu":
        ns = _namespace(numba.njit(error_model="numpy"))
        _NS["cpu"] = ns
        src = (f"def run_cpu(idx, budget, {args}):\n"
               f"    for q in prange(idx.size):\n"
               f"        advance(idx[q], budget, {args})\n")
        g = dict(advance=ns["advance"], prange=numba.prange)
        exec(src, g)
        driver = numba.njit(parallel=True, error_model="numpy")(g["run_cpu"])
    elif device == "cuda":
        from numba import cuda
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ns = _namespace(cuda.jit(device=True))
        src = (f"def run_gpu(idx, budget, {args}):\n"
               f"    q = cuda.grid(1)\n"
               f"    if q < idx.size:\n"
               f"        advance(idx[q], budget, {args})\n")
        g = dict(advance=ns["advance"], cuda=cuda)
        exec(src, g)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            driver = cuda.jit(g["run_gpu"])
    else:
        raise ValueError(f"device must be 'cpu' or 'cuda', not {device!r}")
    _BUILT[device] = driver
    return driver


# ------------------------------------------------------------------------------------- packing
N_LUT_E = 8192
N_LUT_CDF = 16384


def search_lut(xs, edges):
    """lut[b] = searchsorted(xs, edges[b], 'right') (int64), for kernel_src.lut_search."""
    return np.searchsorted(xs, edges, side="right").astype(np.int64)


def pack_tables(sim):
    """All constant inputs of the kernels, from a reference Simulation (host NumPy arrays)."""
    if sim.snapshot_times.size or sim.log_events:
        raise NotImplementedError("snapshots and event logs: use the reference engine")
    f = sim.field
    if isinstance(f, NoField):
        ftype, fpar = 0, [0.0, 0.0]
    elif isinstance(f, UniformField):
        ftype, fpar = 1, [f.E0, 0.0]
    elif isinstance(f, C21BandBending):
        ftype, fpar = 2, [f.E_bb, f.W]
    else:
        raise NotImplementedError(f"field {type(f).__name__}: use the reference engine")
    mechs = sim.mechanisms
    M = len(mechs)
    nE = sim.E_grid.size
    depl = sim.depl
    n_phi = depl.n if depl is not None else 1
    mech_local = np.full(M, -1, np.int32)
    local_tables = []
    for i, m in enumerate(mechs):
        if isinstance(m, LocalMechanism):
            mech_local[i] = len(local_tables)
            local_tables.append(m.table)
    local_tables = (np.array(local_tables, float) if local_tables else np.zeros((1, 1, nE)))
    # parameters, one row per (mechanism, variant)
    gases, gas_index = [], {}
    rows, mtype, par_offset = [], np.zeros(M, np.int32), np.zeros(M, np.int32)
    for i, m in enumerate(mechs):
        variants = m.variants if isinstance(m, LocalMechanism) else [m]
        par_offset[i] = len(rows)
        for vm in variants:
            p = np.zeros(len(PCOLS))
            if isinstance(vm, AcousticPhonon):
                t = MT_ACOUSTIC
            elif isinstance(vm, PolarOptical):
                t = MT_POP
                p[P["P_HW"]] = vm.hw
                p[P["P_EMISSION"]] = float(vm.emission)
                p[P["P_A"]] = vm.a
                p[P["P_SCREENED_ANGLE"]] = float(vm.angle == "screened")
            elif isinstance(vm, IonizedImpurity):
                t = MT_IMPURITY
                p[P["P_EBETA"]] = vm.E_beta
            elif isinstance(vm, ElectronHole):
                t = MT_EH
                g = vm.holes
                if id(g) not in gas_index:
                    gas_index[id(g)] = len(gases)
                    gases.append(g)
                p[P["P_MH"]] = vm.mh
                p[P["P_BETA"]] = vm.beta
                p[P["P_KT"]] = g.kT
                p[P["P_GAS"]] = gas_index[id(g)]
                p[P["P_MASS_MODEL"]] = {"band_edge": 0, "velocity_mass": 1}[vm.mass_model]
                p[P["P_PAULI"]] = {"fermi_dirac": 0, "step_c21": 1, "none": 2}[vm.pauli]
                p[P["P_EFH"]] = vm.sample.EF_h
                p[P["P_ETA"]] = g.eta
                p[P["P_STATS"]] = {"fermi_dirac": 0, "maxwell_boltzmann": 1}[g.statistics]
            elif isinstance(vm, Intervalley):
                t = MT_IV
                p[P["P_HW"]] = vm.hw
                p[P["P_EMISSION"]] = float(vm.emission)
                p[P["P_DELTA"]] = vm.Delta
                p[P["P_VTO"]] = vm.valley_to
            else:
                raise NotImplementedError(f"mechanism {type(vm).__name__}: use the reference engine")
            if vm.valley_from != m.valley_from:
                raise ValueError("variant valley mismatch")
            rows.append(p)
        mtype[i] = t
    params = np.array(rows, float)
    if gases:
        nx = {g._x.size for g in gases}
        if len(nx) != 1:
            raise ValueError("hole-gas samplers of different sizes")
        hole_x = np.array([g._x for g in gases], float)
        hole_cdf = np.array([g._cdf for g in gases], float)
    else:
        hole_x = hole_cdf = np.array([[0.0, 1.0]])
    u_edges = np.linspace(0.0, 1.0, N_LUT_CDF + 1)
    hole_lut = np.array([search_lut(c, u_edges) for c in hole_cdf])
    E_max_grid = float(sim.E_grid[-1])
    E_lut = search_lut(sim.E_grid, E_max_grid * np.linspace(0.0, 1.0, N_LUT_E + 1) ** 2)
    nv = len(sim.material.valleys)
    maxm = max(len(x) for x in sim.mech_by_valley)
    valley_mech = np.zeros((nv, max(maxm, 1)), np.int32)
    valley_nmech = np.zeros(nv, np.int32)
    for v, idx in enumerate(sim.mech_by_valley):
        valley_mech[v, :len(idx)] = idx
        valley_nmech[v] = len(idx)
    vpar = np.array([[v.m_eff, v.alpha] for v in sim.material.valleys], float)
    cfg = np.zeros(len(CCOLS))
    cfg[C["C_TMAX"]] = sim.t_max
    cfg[C["C_EMAX"]] = sim.E_table_max
    cfg[C["C_ZFIELD"]] = sim.z_field
    cfg[C["C_DTMAX"]] = sim.dt_max
    cfg[C["C_ZBACK"]] = sim.z_back if sim.z_back is not None else 0.0
    cfg[C["C_INV_EMAX"]] = 1.0 / E_max_grid
    cfg[C["C_ALAT"]] = sim.material.a_lat
    if depl is not None:
        cfg[C["C_PHI0"]] = depl.phi[0]
        cfg[C["C_DPHI"]] = depl.phi[1] - depl.phi[0]
        if not np.allclose(np.diff(depl.phi), depl.phi[1] - depl.phi[0], rtol=1e-9, atol=0):
            raise ValueError("the fast engine needs a uniform phi grid")
        if depl.field is not f:
            raise ValueError("depletion model and field differ")
    icfg = np.zeros(len(ICOLS), np.int32)
    icfg[I["I_SURFACE"]] = {"absorb": 0, "reflect": 1, "none": 2}[sim.surface]
    icfg[I["I_HAS_MODEL"]] = int(sim.surface_model is not None)
    icfg[I["I_HAS_BACK"]] = int(sim.z_back is not None)
    icfg[I["I_BACK"]] = {"absorb": 0, "reflect": 1}[sim.back]
    icfg[I["I_FLIGHT"]] = int(sim.flight_mode == "self_scattering")
    icfg[I["I_HAS_DEPL"]] = int(depl is not None)
    icfg[I["I_NPHI"]] = n_phi
    icfg[I["I_FLIP_ARRIVAL"]] = int(sim.spin_flip_at_arrival)
    if sim.inv_tau_s.shape[1] != n_phi:
        raise ValueError("spin table / depletion grid mismatch")
    return dict(E_grid=sim.E_grid, E_lut=E_lut, hole_lut=hole_lut, rate_table=sim.rate_table, local_tables=local_tables,
                mech_local=mech_local, thresholds=sim.thresholds, valley_mech=valley_mech,
                valley_nmech=valley_nmech, gamma0=sim.gamma0, mtype=mtype, par_offset=par_offset,
                params=params, hole_x=hole_x, hole_cdf=hole_cdf, vpar=vpar,
                n_equiv=np.array(N_EQUIV, np.int32), inv_tau=sim.inv_tau_s, ftype=ftype,
                fpar=np.array(fpar, float), cfg=cfg, icfg=icfg,
                voff=np.array([v.offset for v in sim.material.valleys], float))


def splitmix64(x):
    """SplitMix64 (Steele, Lea & Flood 2014) on uint64 arrays: the recommended way to expand a
    seed into xoroshiro128+ state (a raw seed gives correlated first outputs)."""
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        z = x.copy()
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
        return x, (z ^ (z >> np.uint64(31))).astype(np.uint64)


def stream_states(rng, n):
    """xoroshiro128+ states (n, 2) for n independent per-electron streams, from a NumPy Generator."""
    x = rng.integers(0, 2**64, size=n, dtype=np.uint64, endpoint=False)
    x, s0 = splitmix64(x)
    _, s1 = splitmix64(x)
    s = np.column_stack([s0, s1])
    s[(s[:, 0] == 0) & (s[:, 1] == 0), 0] = np.uint64(1)
    return s


TABLE_ORDER = ["E_grid", "E_lut", "rate_table", "local_tables", "mech_local", "thresholds",
               "valley_mech", "valley_nmech", "gamma0", "mtype", "par_offset", "params", "hole_x",
               "hole_cdf", "hole_lut", "vpar", "n_equiv", "inv_tau", "ftype", "fpar", "cfg", "icfg",
               "voff", "surf_par", "surf_V"]
STATE_ORDER = ["fs", "ist", "tiv", "vis", "n_events", "n_rej", "n_self", "n_flight", "rng", "branch",
               "start", "has_arr", "arr_f", "arr_i", "arr_vis", "arr_ev", "em_f", "em_i", "last_T"]


def surface_mode(sim, surface_models=None):
    """How encounters are handled: SURF_C21 (in the kernel) for C21Surface models that keep the
    standard interact(); SURF_ABSORB without a surface model; SURF_HOST otherwise."""
    from ..surface_c21 import C21Surface
    models = surface_models if surface_models is not None else (
        [sim.surface_model] if sim.surface_model is not None else [])
    if not models:
        return SURF_ABSORB, []
    if all(isinstance(m, C21Surface) and type(m).interact is C21Surface.interact for m in models):
        return SURF_C21, models
    if surface_models is not None and len(models) > 1:
        raise NotImplementedError("several surface branches need C21Surface models")
    return SURF_HOST, models


def pack_surface(sim, models):
    """Per-branch C21 parameters (chi, L_b, matching mass flag) and barrier slice potentials."""
    if not models:
        return np.zeros((1, 3)), np.zeros((1, 1))
    nsl = {m.n_slices for m in models}
    if len(nsl) != 1:
        raise ValueError("all surface branches must use the same n_slices")
    N = nsl.pop()
    par = np.zeros((len(models), 3))
    V = np.zeros((len(models), N))
    for b, m in enumerate(models):
        if m.material is not sim.material and m.material.a_lat != sim.material.a_lat:
            raise ValueError("surface model built for a different material")
        dx = m.L_b / N
        V[b] = m.barrier((np.arange(N) + 0.5) * dx)
        par[b] = (m.chi, m.L_b, 0.0 if m.matching_mass == "velocity" else 1.0)
    return par, V


# ------------------------------------------------------------------------------------- engine
class FastSimulation:
    """Compiled per-electron transport built from a reference Simulation (see module docstring).

    device:  "cpu" (Numba, parallel threads; numba.set_num_threads controls the count) or "cuda".
    budget:  flights per electron per kernel launch on the GPU (kept small so one launch stays well
             below the Windows display-driver watchdog of ~2 s); on the CPU one launch runs every
             electron to its end (or, with a host surface model, to its next encounter).
    surface_models: optional list of C21Surface models ("branches"); run(..., surface_branch=b)
             assigns electron i to branch b[i]. Default: [sim.surface_model]. C21 models (and the
             absorbing surface) are evaluated inside the kernel; any other surface model with the
             reference interface is called on the host between launches.
    """

    def __init__(self, sim, device="cpu", budget=None, surface_models=None, target_launch_s=0.15):
        self.sim = sim
        self.device = device
        self.adaptive = budget is None
        self.target_launch_s = target_launch_s
        self.mode, self.models = surface_mode(sim, surface_models)
        self.tables = pack_tables(sim)
        self.tables["surf_par"], self.tables["surf_V"] = pack_surface(
            sim, self.models if self.mode == SURF_C21 else [])
        self.tables["icfg"] = self.tables["icfg"].copy()
        self.tables["icfg"][I["I_SURF_MODE"]] = self.mode
        self.tables["icfg"][I["I_HAS_MODEL"]] = int(bool(self.models))
        self.driver = build(device)
        self.budget = budget if budget is not None else (2**62 if device == "cpu" else 256)
        if device == "cuda":
            import cupy as cp
            self.xp = cp
            self._dev = {k: (v if np.isscalar(v) else cp.asarray(v)) for k, v in self.tables.items()}
        else:
            self.xp = np
            self._dev = self.tables
        self.launch_seconds = []

    def _launch(self, idx, budget, state):
        args = [state[k] for k in STATE_ORDER] + [self._dev[k] for k in TABLE_ORDER]
        t0 = time.perf_counter()
        if self.device == "cpu":
            self.driver(idx, budget, *args)
        else:
            import cupy as cp
            threads = 128
            blocks = (idx.size + threads - 1) // threads
            self.driver[blocks, threads](idx, np.int64(budget), *args)
            cp.cuda.runtime.deviceSynchronize()
        self.launch_seconds.append(time.perf_counter() - t0)

    def run(self, ens: Ensemble, rng: np.random.Generator, start_at_surface=False, surface_branch=None):
        sim = self.sim
        xp = self.xp
        N = len(ens)
        M = len(sim.mechanisms)
        if np.any(ens.E > sim.E_table_max):
            raise ValueError("initial energies exceed the rate-table range; raise E_table_max")
        host = ens.copy()                       # static per-electron data + host-side bookkeeping
        fs = np.column_stack([ens.z, ens.t, ens.k[:, 0], ens.k[:, 1], ens.k[:, 2], ens.E, ens.dt_spin])
        ist = np.column_stack([ens.valley, ens.eqv, ens.spin, ens.status, ens.n_flips,
                               ens.n_surface]).astype(np.int32)
        branch = np.zeros(N, np.int32) if surface_branch is None else np.asarray(surface_branch, np.int32)
        if branch.shape != (N,) or (N and (branch.min() < 0 or branch.max() >= max(1, len(self.models)))):
            raise ValueError("surface_branch must give a valid branch index per electron")
        start = np.zeros(N, np.uint8)
        at_surface = (ist[:, 3] == ALIVE) & (ens.z <= 0) & (ens.k[:, 2] < 0)
        if start_at_surface:
            if not self.models:
                raise ValueError("start_at_surface needs a surface model")
            if self.mode != SURF_HOST:
                start[at_surface] = 1
        state = dict(fs=fs, ist=ist, tiv=ens.time_in_valley.astype(float).copy(),
                     vis=ens.visited.astype(np.uint8), n_events=np.zeros((N, M), np.int32),
                     n_rej=np.zeros((N, M), np.int32), n_self=np.zeros(N, np.int64),
                     n_flight=np.zeros(N, np.int64), rng=stream_states(rng, N), branch=branch,
                     start=start, has_arr=np.zeros(N, np.uint8), arr_f=np.zeros((N, 10)),
                     arr_i=np.zeros((N, 5), np.int32), arr_vis=np.zeros((N, 3), np.uint8),
                     arr_ev=np.zeros((N, M), np.int32), em_f=np.zeros((N, 12)),
                     em_i=np.zeros((N, 4), np.int32), last_T=np.full(N, -1.0))
        if xp is not np:
            state = {k: xp.asarray(v) for k, v in state.items()}
        srng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))     # host surface model
        arrivals, emissions = [], []
        if start_at_surface and self.mode == SURF_HOST:
            self._service_surface(np.flatnonzero(at_surface), state, host, srng, arrivals, emissions)

        budget = self.budget
        if self.device == "cuda" and self.adaptive:
            # first launch: ~target_launch_s at ~1.5e8 flights/s spread over N electrons
            budget = int(min(max(1.5e8 * self.target_launch_s / max(N, 1), 8), 2000))
        while True:
            ist_h = self._host(state["ist"])
            idx = np.flatnonzero(ist_h[:, 3] == ALIVE).astype(np.int64)
            if idx.size == 0:
                break
            self._launch(xp.asarray(idx) if xp is not np else idx, budget, state)
            if self.device == "cuda" and self.adaptive:
                # keep each launch near target_launch_s (far below the ~2 s display watchdog)
                dt = max(self.launch_seconds[-1], 1e-4)
                budget = int(min(max(budget * min(self.target_launch_s / dt, 4.0), 16), 10**7))
            ist_h = self._host(state["ist"])
            bad = ist_h[:, 3] >= 10
            if np.any(bad):
                code = int(ist_h[np.flatnonzero(bad)[0], 3])
                raise RuntimeError(ERRORS.get(code, f"kernel error {code}"))
            pend = np.flatnonzero(ist_h[:, 3] == PENDING)
            if pend.size:
                self._service_surface(pend, state, host, srng, arrivals, emissions)
        return self._result(state, host, arrivals, emissions)
    # ---------------------------------------------------------------------------------
    def _host(self, a):
        return a if self.xp is np else self.xp.asnumpy(a)

    def _gather(self, state, key, rows):
        a = state[key]
        if self.xp is np:
            return a[rows]
        return self.xp.asnumpy(a[self.xp.asarray(rows)])

    def _scatter_rows(self, state, key, rows, values):
        if self.xp is np:
            state[key][rows] = values
        else:
            state[key][self.xp.asarray(rows)] = self.xp.asarray(values)

    def _service_surface(self, pend, state, host, srng, arrivals, emissions):
        """Surface encounters of electrons `pend` (stopped at z = 0): first-arrival records,
        encounter counter, and the surface model (Simulation._surface_interaction)."""
        sim = self.sim
        fs = self._gather(state, "fs", pend)
        ist = self._gather(state, "ist", pend)
        k = fs[:, 2:5].copy()
        valley = ist[:, 0].astype(np.int8)
        eqv = ist[:, 1].astype(np.int8)
        first = ist[:, 5] == 0
        if np.any(first):
            sel = pend[first]
            ne = self._gather(state, "n_events", sel)
            tiv = self._gather(state, "tiv", sel)
            vis = self._gather(state, "vis", sel).astype(bool)
            arrivals.append(SurfaceArrivals(
                t=fs[first, 1].copy(), E=fs[first, 5].copy(), k=k[first].copy(), valley=valley[first],
                spin=ist[first, 2].astype(np.int8), spin0=host.spin0[sel].copy(), z0=host.z0[sel].copy(),
                E0=host.E0[sel].copy(), band=host.band[sel].copy(), n_flips=ist[first, 4].astype(np.int32),
                n_events=ne, pid=host.pid[sel].copy(), time_in_valley=tiv, visited=vis, eqv=eqv[first],
                dt_spin=fs[first, 6].copy(),
                K=k[first] + valley_center(valley[first], eqv[first], sim.material.a_lat),
                band_edge_at_surface=float(sim.field.band_edge(np.array([0.0]))[0]),
                mechanism_names=sim.names))
        ist[:, 5] += 1
        if sim.surface_model is None:
            ist[:, 3] = SURFACE
        else:
            from ..surface_c21 import EMIT, REFLECT, TRAP
            K = k + valley_center(valley, eqv, sim.material.a_lat)
            outcome, info = sim.surface_model.interact(k, fs[:, 5], valley, K, srng, pid=host.pid[pend])
            em = outcome == EMIT
            if np.any(em):
                ie = pend[em]
                emissions.append(Emissions(
                    t=fs[em, 1].copy(), E=fs[em, 5].copy(), k=k[em].copy(), K=K[em].copy(),
                    valley=valley[em], eqv=eqv[em], spin=ist[em, 2].astype(np.int8),
                    spin0=host.spin0[ie].copy(), z0=host.z0[ie].copy(), E0=host.E0[ie].copy(),
                    band=host.band[ie].copy(), n_surface=ist[em, 5].astype(np.int32),
                    pid=host.pid[ie].copy(), p_vac=np.asarray(info["p_vac"])[em].copy(),
                    E_vac=np.asarray(info["E_vac_kin"])[em].copy()))
                ist[em, 3] = EMITTED
            ist[outcome == TRAP, 3] = TRAPPED
            rf = outcome == REFLECT
            fs[rf, 4] = np.abs(fs[rf, 4])
            fs[rf, 0] = 0.0
            ist[rf, 3] = ALIVE
        self._scatter_rows(state, "fs", pend, fs)
        self._scatter_rows(state, "ist", pend, ist)

    def _result(self, state, host, arrivals, emissions):
        from ..transport import Result
        sim = self.sim
        h = {k: self._host(v) for k, v in state.items()}
        fs, ist = h["fs"], h["ist"]
        ens = Ensemble(z=fs[:, 0].copy(), t=fs[:, 1].copy(), k=fs[:, 2:5].copy(), E=fs[:, 5].copy(),
                       valley=ist[:, 0].astype(np.int8), spin=ist[:, 2].astype(np.int8),
                       status=ist[:, 3].astype(np.int8), z0=host.z0, E0=host.E0, spin0=host.spin0,
                       band=host.band, dt_spin=fs[:, 6].copy(), n_flips=ist[:, 4].astype(np.int32),
                       pid=host.pid, time_in_valley=h["tiv"], visited=h["vis"].astype(bool),
                       eqv=ist[:, 1].astype(np.int8), n_surface=ist[:, 5].astype(np.int32))
        M = len(sim.mechanisms)
        a_lat = sim.material.a_lat
        if self.mode != SURF_HOST:              # records written by the kernel
            sel = np.flatnonzero(h["has_arr"] == 1)
            af, ai = h["arr_f"][sel], h["arr_i"][sel]
            valley, eqv = ai[:, 0].astype(np.int8), ai[:, 1].astype(np.int8)
            arrivals = [SurfaceArrivals(
                t=af[:, 1].copy(), E=af[:, 5].copy(), k=af[:, 2:5].copy(), valley=valley,
                spin=ai[:, 2].astype(np.int8), spin0=host.spin0[sel].copy(), z0=host.z0[sel].copy(),
                E0=host.E0[sel].copy(), band=host.band[sel].copy(), n_flips=ai[:, 4].astype(np.int32),
                n_events=h["arr_ev"][sel], pid=host.pid[sel].copy(), time_in_valley=af[:, 7:10].copy(),
                visited=h["arr_vis"][sel].astype(bool), eqv=eqv, dt_spin=af[:, 6].copy(),
                K=af[:, 2:5] + valley_center(valley, eqv, a_lat), mechanism_names=sim.names)]
            if self.mode == SURF_C21:
                sel = np.flatnonzero(ist[:, 3] == EMITTED)
                ef, ei = h["em_f"][sel], h["em_i"][sel]
                emissions = [Emissions(
                    t=ef[:, 0].copy(), E=ef[:, 1].copy(), k=ef[:, 2:5].copy(), K=ef[:, 5:8].copy(),
                    valley=ei[:, 0].astype(np.int8), eqv=ei[:, 1].astype(np.int8),
                    spin=ei[:, 2].astype(np.int8), spin0=host.spin0[sel].copy(), z0=host.z0[sel].copy(),
                    E0=host.E0[sel].copy(), band=host.band[sel].copy(), n_surface=ei[:, 3].astype(np.int32),
                    pid=host.pid[sel].copy(), p_vac=ef[:, 8:11].copy(), E_vac=ef[:, 11].copy())]
        arr = SurfaceArrivals.concatenate(arrivals, M, sim.names)
        arr.band_edge_at_surface = float(sim.field.band_edge(np.array([0.0]))[0])
        em = Emissions.concatenate(emissions) if self.models else None
        self.last_branch = h["branch"]
        return Result(ensemble=ens, arrivals=arr, snapshots=None, mechanism_names=sim.names,
                      n_iterations=int(h["n_flight"].max()) if len(ens) else 0,
                      n_real=h["n_events"].sum(axis=0).astype(np.int64), n_self=int(h["n_self"].sum()),
                      n_rejected=h["n_rej"].sum(axis=0).astype(np.int64), flight_mode=sim.flight_mode,
                      event_log={}, emissions=em)
