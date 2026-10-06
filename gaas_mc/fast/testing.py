"""Access to individual fast-engine kernel routines (CPU build) for validation against the reference
implementation: deterministic pieces (flights, rates, spin tables, depletion index) are compared
exactly or to round-off, stochastic pieces (final states) by distribution. Used by
tests/test_fast_engine.py."""
from __future__ import annotations

import numpy as np

from . import engine


def _ns():
    engine.build("cpu")
    return engine._NS["cpu"]


_DRIVERS = {}


def _driver(name):
    if name in _DRIVERS:
        return _DRIVERS[name]
    import numba
    ns = _ns()
    if name == "final_state":
        fn = ns["final_state"]

        @numba.njit(error_model="numpy")
        def d(t, p, v, k, E, vpar, hole_x, hole_cdf, hole_lut, rng, out_k, out_acc, out_v):
            for i in range(E.size):
                nx, ny, nz, acc, vn = fn(t, p, v, k[i, 0], k[i, 1], k[i, 2], E[i], vpar, hole_x,
                                         hole_cdf, hole_lut, rng, i)
                out_k[i, 0] = nx
                out_k[i, 1] = ny
                out_k[i, 2] = nz
                out_acc[i] = acc
                out_v[i] = vn
    elif name == "propagate":
        pf, pfree, inf = ns["propagate_field"], ns["propagate_free"], ns["in_field"]

        @numba.njit(error_model="numpy")
        def d(z, k, E, m, a, dt, ftype, fpar, icfg, cfg, out):
            for i in range(z.size):
                if inf(z[i], k[i, 2], E[i], m[i], a[i], cfg):
                    z1, kz1, E1, du, ev = pf(z[i], k[i, 0], k[i, 1], k[i, 2], E[i], m[i], a[i], dt[i],
                                             ftype, fpar, icfg, cfg)
                else:
                    z1, kz1, du, ev = pfree(z[i], k[i, 2], E[i], m[i], a[i], dt[i], icfg, cfg)
                    E1 = E[i]
                out[i, 0] = z1
                out[i, 1] = kz1
                out[i, 2] = E1
                out[i, 3] = du
                out[i, 4] = ev
    elif name == "rates":
        mr, br, pidx = ns["mech_rate"], ns["bracket_lut"], ns["phi_index"]

        @numba.njit(error_model="numpy")
        def d(mi, E, z, E_grid, E_lut, rate_table, local_tables, mech_local, thresholds, ftype, fpar,
              cfg, icfg, out):
            for i in range(E.size):
                j, w = br(E_grid, E_lut, E[i], cfg[7])
                fi = pidx(z[i], ftype, fpar, cfg, icfg)
                out[i] = mr(mi, E[i], j, w, fi, rate_table, local_tables, mech_local, thresholds,
                            icfg[6])
    elif name == "inv_tau":
        f, pidx, br = ns["inv_tau_s"], ns["phi_index"], ns["bracket_lut"]

        @numba.njit(error_model="numpy")
        def d(E, v, z, inv_tau, E_grid, E_lut, ftype, fpar, cfg, icfg, out):
            for i in range(E.size):
                j, u = br(E_grid, E_lut, E[i], cfg[7])
                out[i] = f(v[i], pidx(z[i], ftype, fpar, cfg, icfg), inv_tau, j, u, icfg[6])
    elif name == "bracket":
        b1, b2 = ns["bracket"], ns["bracket_lut"]

        @numba.njit(error_model="numpy")
        def d(E, E_grid, E_lut, inv_emax, out):
            for i in range(E.size):
                j1, w1 = b1(E_grid, E[i])
                j2, w2 = b2(E_grid, E_lut, E[i], inv_emax)
                out[i, 0] = j1
                out[i, 1] = j2
                out[i, 2] = w1
                out[i, 3] = w2
    elif name == "interp":
        f = ns["interp_lut"]

        @numba.njit(error_model="numpy")
        def d(u, xs, ys, lut, out):
            for i in range(u.size):
                out[i] = f(u[i], xs, ys, lut)
    elif name == "transmission":
        f = ns["transmission"]

        @numba.njit(error_model="numpy")
        def d(E_tot, k_in, m_in, kpar, chi, V, L_b, out):
            for i in range(E_tot.size):
                out[i] = f(E_tot[i], k_in[i], m_in[i], kpar[i], chi, V, L_b)
    elif name == "fold":
        f = ns["fold_kpar"]

        @numba.njit(error_model="numpy")
        def d(K, a_lat, out):
            for i in range(K.shape[0]):
                out[i] = f(K[i, 0], K[i, 1], a_lat)
    else:
        raise KeyError(name)
    _DRIVERS[name] = d
    return d


def seeds(n, seed=0):
    return engine.stream_states(np.random.default_rng(seed), n)


def final_states(sim, mech_index, variant, k, E, seed=0):
    """Kernel final states for mechanism `mech_index` (variant row `variant` for a depletion
    mechanism) at wavevectors k (n, 3) and energies E (n,). Returns (k', accepted, valley')."""
    tab = engine.pack_tables(sim)
    m = sim.mechanisms[mech_index]
    row = tab["par_offset"][mech_index] + variant
    n = E.size
    out_k = np.zeros((n, 3))
    out_acc = np.zeros(n, np.bool_)
    out_v = np.zeros(n, np.int64)
    _driver("final_state")(int(tab["mtype"][mech_index]), tab["params"][row], int(m.valley_from),
                           np.ascontiguousarray(k, float), np.ascontiguousarray(E, float), tab["vpar"],
                           tab["hole_x"], tab["hole_cdf"], tab["hole_lut"], seeds(n, seed), out_k, out_acc,
                           out_v)
    return out_k, out_acc, out_v


def propagate(sim, z, k, E, valley, dt):
    """Kernel flight (field or free) for each electron: returns z, k_z, E, dt_used, event."""
    tab = engine.pack_tables(sim)
    m = tab["vpar"][valley, 0]
    a = tab["vpar"][valley, 1]
    out = np.zeros((z.size, 5))
    _driver("propagate")(np.asarray(z, float), np.ascontiguousarray(k, float), np.asarray(E, float), m, a,
                         np.asarray(dt, float), tab["ftype"], tab["fpar"], tab["icfg"], tab["cfg"], out)
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4].astype(np.int8)


def rates(sim, mech_index, E, z):
    tab = engine.pack_tables(sim)
    out = np.zeros(E.size)
    _driver("rates")(mech_index, np.asarray(E, float), np.asarray(z, float), tab["E_grid"], tab["E_lut"],
                     tab["rate_table"], tab["local_tables"], tab["mech_local"], tab["thresholds"],
                     tab["ftype"], tab["fpar"], tab["cfg"], tab["icfg"], out)
    return out


def inv_tau(sim, E, valley, z):
    tab = engine.pack_tables(sim)
    out = np.zeros(E.size)
    _driver("inv_tau")(np.asarray(E, float), np.asarray(valley, np.int64), np.asarray(z, float),
                       tab["inv_tau"], tab["E_grid"], tab["E_lut"], tab["ftype"], tab["fpar"], tab["cfg"], tab["icfg"], out)
    return out


def brackets(sim, E):
    """(j_binary, j_lut, w_binary, w_lut) for energies E: the lookup-table search must equal the
    plain binary search exactly."""
    tab = engine.pack_tables(sim)
    out = np.zeros((E.size, 4))
    _driver("bracket")(np.asarray(E, float), tab["E_grid"], tab["E_lut"], float(tab["cfg"][7]), out)
    return out


def hole_energy_quantiles(sim, gas, u):
    """Kernel inverse-CDF lookup (x = E/kT) of hole gas `gas` at uniform deviates u."""
    tab = engine.pack_tables(sim)
    out = np.zeros(u.size)
    _driver("interp")(np.asarray(u, float), tab["hole_cdf"][gas], tab["hole_x"][gas], tab["hole_lut"][gas], out)
    return out


def transmission(model, E_tot, k_in, m_in, kpar):
    """Kernel C21 transmission (barrier slices tabulated from model.barrier) for comparison with
    C21Surface.transmission."""
    N = model.n_slices
    V = np.asarray(model.barrier((np.arange(N) + 0.5) * model.L_b / N), float)
    out = np.zeros(np.size(E_tot))
    _driver("transmission")(*(np.asarray(x, float) for x in (E_tot, k_in, m_in, kpar)), float(model.chi), V,
                            float(model.L_b), out)
    return out


def fold_kpar(K, a_lat):
    out = np.zeros(K.shape[0])
    _driver("fold")(np.ascontiguousarray(K, float), float(a_lat), out)
    return out
