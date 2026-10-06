"""Per-electron event loop of the fast engine: scalar source compiled twice (gaas_mc/fast/engine.py).

This file is not imported as a module. engine.py executes it in two namespaces, with `jit` bound to
numba.njit (CPU threads) and to numba.cuda.jit(device=True) (GPU). Every function here is a
scalar transcription of the vectorized reference (gaas_mc/transport.py and the mechanism
classes); the docstrings name the reference routine. Rules that make one source compile for both
targets: scalar arithmetic and `math` only, no allocation, arrays only as arguments, small
vectors as separate scalars, tuples for multiple return values, explicit uint64 arithmetic in the
random-number generator.

The physics is not re-derived here: rate tables, mechanism parameters, depletion variants and
spin-relaxation tables are all produced by the reference objects (engine.py packs them).
"""
# The names below are injected by engine.py before this source is executed:
#   jit, math, cmath, np, HBAR, Q_E, M0, ALIVE, SURFACE, TIMEOUT, BACK, EMITTED, TRAPPED, PENDING,
#   ERR_*  (status codes),
#   MT_ACOUSTIC, MT_POP, MT_IMPURITY, MT_EH, MT_IV (mechanism types), P_* (parameter columns),
#   C_* / I_* (configuration indices), EV_* (flight events)


# ----------------------------------------------------------------------------------- RNG
# xoroshiro128+ (Blackman & Vigna), one independent stream per electron; state rng[i, 0:2].
@jit
def _rotl(x, k):
    return (x << np.uint64(k)) | (x >> np.uint64(64 - k))


@jit
def rand(rng, i):
    """Uniform double in [0, 1) from the 53 high bits."""
    s0 = rng[i, 0]
    s1 = rng[i, 1]
    result = s0 + s1
    s1 ^= s0
    rng[i, 0] = _rotl(s0, 24) ^ s1 ^ (s1 << np.uint64(16))
    rng[i, 1] = _rotl(s1, 37)
    return float(result >> np.uint64(11)) * (1.0 / 9007199254740992.0)


# ----------------------------------------------------------------------------------- bands
@jit
def E_of_k(k, m, a):
    """bands.E_of_k (Eq. 2, stable form)."""
    g = HBAR * HBAR * k * k / (2.0 * m)
    return 2.0 * g / (1.0 + math.sqrt(1.0 + 4.0 * a * g))


@jit
def k_of_E(E, m, a):
    """bands.k_of_E (Eq. 1)."""
    return math.sqrt(2.0 * m * E * (1.0 + a * E)) / HBAR


@jit
def gamma_of_E(E, a):
    return E * (1.0 + a * E)


@jit
def random_unit(rng, i):
    """bands.random_unit_vectors for one vector: cos_t = 1 - 2r, phi = 2 pi r."""
    cos_t = 1.0 - 2.0 * rand(rng, i)
    phi = 2.0 * math.pi * rand(rng, i)
    s = 1.0 - cos_t * cos_t
    sin_t = math.sqrt(s) if s > 0.0 else 0.0
    return sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t


@jit
def unit(x, y, z, rng, i):
    """bands.unit: normalized vector; a zero vector gets a random direction."""
    n = math.sqrt(x * x + y * y + z * z)
    if n > 0.0:
        return x / n, y / n, z / n
    return random_unit(rng, i)


@jit
def rotate_about(ux, uy, uz, cos_t, phi):
    """bands.rotate_about: unit vector at polar angle theta (cos_t), azimuth phi about u."""
    if abs(ux) < 0.9:            # helper axis x: e1 = u x (1, 0, 0)
        e1x, e1y, e1z = 0.0, uz, -uy
    else:                        # helper axis y: e1 = u x (0, 1, 0)
        e1x, e1y, e1z = -uz, 0.0, ux
    n = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    e1x /= n
    e1y /= n
    e1z /= n
    e2x = uy * e1z - uz * e1y
    e2y = uz * e1x - ux * e1z
    e2z = ux * e1y - uy * e1x
    s = 1.0 - cos_t * cos_t
    sin_t = math.sqrt(s) if s > 0.0 else 0.0
    a = sin_t * math.cos(phi)
    b = sin_t * math.sin(phi)
    return (cos_t * ux + a * e1x + b * e2x, cos_t * uy + a * e1y + b * e2y,
            cos_t * uz + a * e1z + b * e2z)


@jit
def new_k_from_angle(kx, ky, kz, E_new, cos_t, m, a, rng, i):
    """scattering.base.new_k_from_angle."""
    ux, uy, uz = unit(kx, ky, kz, rng, i)
    phi = 2.0 * math.pi * rand(rng, i)
    nx, ny, nz = rotate_about(ux, uy, uz, cos_t, phi)
    kk = k_of_E(E_new, m, a)
    return nx * kk, ny * kk, nz * kk


@jit
def new_k_isotropic(E_new, m, a, rng, i):
    """scattering.base.new_k_isotropic."""
    nx, ny, nz = random_unit(rng, i)
    kk = k_of_E(E_new, m, a)
    return nx * kk, ny * kk, nz * kk


# ----------------------------------------------------------------------------------- tables
@jit
def bracket(g, E):
    """j = clip(searchsorted(g, E, side='right'), 1, n - 1) and the clipped weight w, exactly as
    Simulation.rates_at / LocalMechanism.rates_at (plain binary search; see bracket_lut)."""
    n = g.shape[0]
    lo = 0
    hi = n
    while lo < hi:                    # first index with g[idx] > E
        mid = (lo + hi) // 2
        if g[mid] <= E:
            lo = mid + 1
        else:
            hi = mid
    return _clip_bracket(g, E, lo)


@jit
def lut_search(xs, lut, xq, s):
    """searchsorted(xs, xq, 'right') for increasing xs, narrowed by a lookup table: lut[b] =
    searchsorted(xs, edge_b, 'right') at the edges of nb uniform bins of s = s(xq) in [0, 1]
    (engine.search_lut). The answer lies between the counts at the edges of bin b; the window is
    widened by one bin on each side so that rounding of s at a bin edge cannot exclude it. Exact."""
    nb = lut.shape[0] - 1
    b = int(s * nb)
    if b < 0:
        b = 0
    if b > nb - 1:
        b = nb - 1
    lo = lut[b - 1] if b > 0 else 0
    hi = lut[b + 2] if b + 2 <= nb else xs.shape[0]
    while lo < hi:                    # first index in [lo, hi) with xs > xq (else hi)
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid + 1
        else:
            hi = mid
    return lo


@jit
def bracket_lut(g, lut, E, inv_emax):
    """bracket(g, E) via lut_search, with bins uniform in sqrt(E / E_max) (the energy grid is
    quadratic in that variable)."""
    x = E * inv_emax
    s = math.sqrt(x) if x > 0.0 else 0.0
    return _clip_bracket(g, E, lut_search(g, lut, E, s))


@jit
def _clip_bracket(g, E, j):
    n = g.shape[0]
    if j < 1:
        j = 1
    if j > n - 1:
        j = n - 1
    w = (E - g[j - 1]) / (g[j] - g[j - 1])
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    return j, w


@jit
def interp_sorted(xq, xs, ys, row):
    """np.interp(xq, xs[row], ys[row]) for increasing xs (clamped at both ends)."""
    n = xs.shape[1]
    if xq <= xs[row, 0]:
        return ys[row, 0]
    if xq >= xs[row, n - 1]:
        return ys[row, n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[row, mid] <= xq:
            lo = mid
        else:
            hi = mid
    x0 = xs[row, lo]
    x1 = xs[row, hi]
    if x1 == x0:
        return ys[row, hi]
    return ys[row, lo] + (ys[row, hi] - ys[row, lo]) * (xq - x0) / (x1 - x0)


@jit
def interp_lut(xq, xs, ys, lut):
    """np.interp(xq, xs, ys) for increasing xs on [0, 1] (a CDF), with lut_search."""
    n = xs.shape[0]
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[n - 1]:
        return ys[n - 1]
    hi = lut_search(xs, lut, xq, xq)
    lo = hi - 1
    x0 = xs[lo]
    x1 = xs[hi]
    if x1 == x0:
        return ys[hi]
    return ys[lo] + (ys[hi] - ys[lo]) * (xq - x0) / (x1 - x0)


@jit
def phi_split(fi, n_phi):
    """LocalMechanism.rates_at: i0 = clip(floor(fi)), i1 = min(i0 + 1, n - 1), w = clip(fi - i0)."""
    i0 = int(math.floor(fi))
    if i0 < 0:
        i0 = 0
    if i0 > n_phi - 1:
        i0 = n_phi - 1
    i1 = i0 + 1
    if i1 > n_phi - 1:
        i1 = n_phi - 1
    w = fi - i0
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    return i0, i1, w


@jit
def mech_rate(mi, E, j, wE, fi, rate_table, local_tables, mech_local, thresholds, n_phi):
    """Rate of mechanism mi at energy E (bracket j, wE): the Simulation.rates_at row, with the
    bilinear local-potential interpolation for depletion-dependent mechanisms when fi >= 0."""
    if not (E > thresholds[mi]):
        return 0.0
    li = mech_local[mi]
    if li < 0 or fi < 0.0:
        return rate_table[mi, j - 1] * (1.0 - wE) + rate_table[mi, j] * wE
    i0, i1, w = phi_split(fi, n_phi)
    r0 = local_tables[li, i0, j - 1] * (1.0 - wE) + local_tables[li, i0, j] * wE
    r1 = local_tables[li, i1, j - 1] * (1.0 - wE) + local_tables[li, i1, j] * wE
    return (1.0 - w) * r0 + w * r1


# ----------------------------------------------------------------------------------- fields
@jit
def field_Ez(z, ftype, fpar):
    """fields.*.Ez: 0 none, 1 uniform (E0), 2 C21 band bending (E_bb, W) Eq. 62."""
    if ftype == 1:
        return fpar[0]
    if ftype == 2:
        x = 1.0 - z / fpar[1]
        if x < 0.0:
            x = 0.0
        return 2.0 * fpar[0] / (Q_E * fpar[1]) * x
    return 0.0


@jit
def field_band_edge(z, ftype, fpar):
    """fields.*.band_edge (Eq. 61 for the C21 band bending)."""
    if ftype == 1:
        return Q_E * fpar[0] * z
    if ftype == 2:
        x = 1.0 - z / fpar[1]
        if x < 0.0:
            x = 0.0
        return -fpar[0] * x * x
    return 0.0


@jit
def Ez_ext(z, ftype, fpar, icfg, cfg):
    """Simulation._Ez_ext: field with mirror images across reflecting walls."""
    if icfg[I_SURFACE] == 1 and z < 0.0:
        Ez = -field_Ez(-z, ftype, fpar)
    else:
        Ez = field_Ez(z, ftype, fpar)
    zb = cfg[C_ZBACK]
    if icfg[I_HAS_BACK] == 1 and icfg[I_BACK] == 1 and z > zb:
        Ez = -field_Ez(2.0 * zb - z, ftype, fpar)
    return Ez


@jit
def phi_index(z, ftype, fpar, cfg, icfg):
    """DepletionModel.frac_index: fractional index of phi(z) on the uniform phi grid; -1 without a
    depletion model."""
    if icfg[I_HAS_DEPL] == 0:
        return -1.0
    zz = z if z > 0.0 else 0.0
    ph = field_band_edge(zz, ftype, fpar)
    n = icfg[I_NPHI]
    phi0 = cfg[C_PHI0]
    dphi = cfg[C_DPHI]
    fi = (ph - phi0) / dphi
    if fi < 0.0:
        fi = 0.0
    if fi > n - 1:
        fi = n - 1.0
    return fi


# ----------------------------------------------------------------------------------- flights
@jit
def in_field(z, kz, E, m, a, cfg):
    """Simulation.in_field."""
    zf = cfg[C_ZFIELD]
    if zf <= 0.0:
        return False
    if math.isinf(zf):
        return True
    vz = HBAR * kz / (m * (1.0 + 2.0 * a * E))
    return (z < zf) or ((z <= zf * (1.0 + 1e-12)) and (vz < 0.0))


@jit
def kdk(z, kx, ky, kz, m, a, h, ftype, fpar, icfg, cfg):
    """Simulation._kdk: one kick-drift-kick substep (Eqs. 17-18)."""
    F = -Q_E * Ez_ext(z, ftype, fpar, icfg, cfg)
    kz = kz + 0.5 * F * h / HBAR
    E = E_of_k(math.sqrt(kx * kx + ky * ky + kz * kz), m, a)
    vz = HBAR * kz / (m * (1.0 + 2.0 * a * E))
    z = z + vz * h
    F = -Q_E * Ez_ext(z, ftype, fpar, icfg, cfg)
    kz = kz + 0.5 * F * h / HBAR
    return z, kz


@jit
def propagate_free(z, kz, E, m, a, dt, icfg, cfg):
    """Simulation._propagate_free (exact straight flight). Returns z, kz, dt_used, event."""
    vz = HBAR * kz / (m * (1.0 + 2.0 * a * E))
    z1 = z + vz * dt
    event = EV_NONE
    t_ev = math.inf
    zf = cfg[C_ZFIELD]
    if zf > 0.0 and not math.isinf(zf):
        if z1 < zf and vz < 0.0:
            t_ev = (z - zf) / (-vz)
            event = EV_REGION
    if icfg[I_SURFACE] == 0 or icfg[I_SURFACE] == 1:
        if z1 <= 0.0 and vz < 0.0:
            ts = z / (-vz)
            if ts < t_ev:
                t_ev = ts
                event = EV_SURFACE
    zb = cfg[C_ZBACK]
    if icfg[I_HAS_BACK] == 1:
        if z1 >= zb and vz > 0.0:
            tb = (zb - z) / vz
            if tb < t_ev:
                t_ev = tb
                event = EV_BACK
    if event == EV_NONE:
        return z1, kz, dt, event
    dt_used = t_ev
    if dt_used < 0.0:
        dt_used = 0.0
    if dt_used > dt:
        dt_used = dt
    z1 = z + vz * dt_used
    if event == EV_SURFACE:
        z1 = 0.0
    if event == EV_REGION:
        z1 = zf
    if event == EV_BACK:
        z1 = zb
        if icfg[I_BACK] == 1:
            kz = -kz
            event = EV_WALL
    if icfg[I_SURFACE] == 1 and event == EV_SURFACE:
        kz = -kz
        event = EV_WALL
    return z1, kz, dt_used, event


@jit
def surface_crossing_time(z_old, kx, ky, kz, m, a, h, h_lin, ftype, fpar, icfg, cfg):
    """Simulation._surface_crossing_time: smallest root in [0, h] of z_old + v t + a_z t^2 / 2 = 0
    (force at z_old, velocity mass); linear interpolation h_lin if there is none."""
    E0 = E_of_k(math.sqrt(kx * kx + ky * ky + kz * kz), m, a)
    mv = m * (1.0 + 2.0 * a * E0)
    v = HBAR * kz / mv
    acc = -Q_E * Ez_ext(z_old, ftype, fpar, icfg, cfg) / mv
    A2 = 0.5 * acc
    D = v * v - 4.0 * A2 * z_old
    if D < 0.0:
        return h_lin
    sq = math.sqrt(D)
    q = -0.5 * (v + (sq if v >= 0.0 else -sq))
    best = math.inf
    hmax = h * (1.0 + 1e-9)
    if A2 != 0.0:
        t1 = q / A2
        if t1 >= 0.0 and t1 <= hmax and t1 < best:
            best = t1
    if q != 0.0:
        t2 = z_old / q
        if t2 >= 0.0 and t2 <= hmax and t2 < best:
            best = t2
    if not math.isfinite(best):
        return h_lin
    return best if best < h else h


@jit
def surface_crossing_kz(z_old, kx, ky, kz_old, kc, m, a, ftype, fpar):
    """Simulation._surface_crossing_k: k_z < 0 at the surface from total-energy conservation (k_par
    unchanged); the integrator's kc if there is no solution."""
    E0 = E_of_k(math.sqrt(kx * kx + ky * ky + kz_old * kz_old), m, a)
    zz = z_old if z_old > 0.0 else 0.0
    Es = E0 + field_band_edge(zz, ftype, fpar) - field_band_edge(0.0, ftype, fpar)
    if Es < 0.0:
        Es = 0.0
    kt = k_of_E(Es, m, a)
    kz2 = kt * kt - kx * kx - ky * ky
    if kz2 > 0.0:
        return -math.sqrt(kz2)
    return kc


@jit
def propagate_field(z, kx, ky, kz, E, m, a, dt, ftype, fpar, icfg, cfg):
    """Simulation._propagate_field: velocity-Verlet substeps; a substep crossing a boundary is
    redone with the interpolated fraction. Returns z, kz, E, dt_used, event."""
    dt_max = cfg[C_DTMAX]
    zb = cfg[C_ZBACK]
    zf = cfg[C_ZFIELD]
    has_zr = zf > 0.0 and not math.isinf(zf)
    event = EV_NONE
    remaining = dt
    elapsed = 0.0
    if icfg[I_HAS_MODEL] == 1 and z <= 0.0 and kz > 0.0:
        F0 = -Q_E * field_Ez(0.0, ftype, fpar)
        if F0 < 0.0:
            t_ret = 2.0 * HBAR * kz / (-F0)
            if t_ret <= min(dt, dt_max):
                kz = -kz
                z = 0.0
                elapsed = t_ret
                remaining = 0.0
                event = EV_SURFACE
    while remaining > 1e-30:
        h = remaining if remaining < dt_max else dt_max
        z_old = z
        kz_old = kz
        z_new, kz_new = kdk(z_old, kx, ky, kz_old, m, a, h, ftype, fpar, icfg, cfg)
        if icfg[I_SURFACE] == 1 and z_new < 0.0:
            z_new = -z_new
            kz_new = -kz_new
        if icfg[I_HAS_BACK] == 1 and icfg[I_BACK] == 1 and z_new > zb:
            z_new = 2.0 * zb - z_new
            kz_new = -kz_new
        cross_s = icfg[I_SURFACE] == 0 and z_new <= 0.0
        cross_b = icfg[I_HAS_BACK] == 1 and icfg[I_BACK] == 0 and z_new >= zb
        cross_r = has_zr and z_new >= zf and not cross_b
        if cross_s or cross_b or cross_r:
            if cross_s:
                zbd = 0.0
            elif cross_b:
                zbd = zb
            else:
                zbd = zf
            f = (z_old - zbd) / (z_old - z_new)
            if f < 0.0:
                f = 0.0
            if f > 1.0:
                f = 1.0
            hc = h * f
            if cross_s:
                hc = surface_crossing_time(z_old, kx, ky, kz_old, m, a, h, hc, ftype, fpar, icfg, cfg)
            _, kc = kdk(z_old, kx, ky, kz_old, m, a, hc, ftype, fpar, icfg, cfg)
            if cross_s:
                kc = surface_crossing_kz(z_old, kx, ky, kz_old, kc, m, a, ftype, fpar)
            z_new = zbd
            kz_new = kc
            h = hc
            if cross_s:
                event = EV_SURFACE
            if cross_b:
                event = EV_BACK
            if cross_r:
                event = EV_REGION
        z = z_new
        kz = kz_new
        elapsed += h
        remaining -= h
        if event != EV_NONE:
            remaining = 0.0
    dt_used = elapsed if event != EV_NONE else dt
    E = E_of_k(math.sqrt(kx * kx + ky * ky + kz * kz), m, a)
    return z, kz, E, dt_used, event


# ----------------------------------------------------------------------------------- spin
@jit
def inv_tau_s(v, fi, inv_tau, j, u, n_phi):
    """Simulation._inv_tau_s_at, at the energy bracket (j, u) of the electron's energy."""
    if fi < 0.0 or n_phi == 1:
        last = inv_tau.shape[1] - 1
        return inv_tau[v, last, j - 1] * (1.0 - u) + inv_tau[v, last, j] * u
    i0 = int(math.floor(fi))
    if i0 < 0:
        i0 = 0
    if i0 > n_phi - 1:
        i0 = n_phi - 1
    i1 = i0 + 1
    if i1 > n_phi - 1:
        i1 = n_phi - 1
    w = fi - i0
    r0 = inv_tau[v, i0, j - 1] * (1.0 - u) + inv_tau[v, i0, j] * u
    r1 = inv_tau[v, i1, j - 1] * (1.0 - u) + inv_tau[v, i1, j] * u
    return (1.0 - w) * r0 + w * r1


@jit
def spin_flip(i, fs, ist, inv_tau, j, u, ftype, fpar, cfg, icfg, rng):
    """Simulation._spin_flip (Eq. 54) for electron i, with (j, u) the bracket of its current energy
    on the transport grid; resets dt_spin."""
    fi = phi_index(fs[i, 0], ftype, fpar, cfg, icfg)
    inv = inv_tau_s(ist[i, 0], fi, inv_tau, j, u, icfg[I_NPHI])
    P = 0.5 * (-math.expm1(-fs[i, 6] * inv))
    if rand(rng, i) < P:
        ist[i, 2] = -ist[i, 2]
        ist[i, 4] += 1
    fs[i, 6] = 0.0


# ----------------------------------------------------------------------------------- mechanisms
@jit
def scatter_pop(kx, ky, kz, E, p, m, a, rng, i):
    """PolarOptical.scatter (Eq. 30 angle; optional screened rejection)."""
    hw = p[P_HW]
    Ep = E - hw if p[P_EMISSION] > 0.5 else E + hw
    kk = math.sqrt(2.0 * m * gamma_of_E(E, a)) / HBAR
    kp = math.sqrt(2.0 * m * gamma_of_E(Ep, a)) / HBAR
    xi = 2.0 * kk * kp / ((kk - kp) * (kk - kp))
    r = rand(rng, i)
    cos_t = ((1.0 + xi) - math.exp(r * math.log1p(2.0 * xi))) / xi
    aa = p[P_A]
    if p[P_SCREENED_ANGLE] > 0.5 and aa > 0.0:
        while True:
            u = kk * kk + kp * kp - 2.0 * kk * kp * cos_t
            acc = u / (u + aa)
            if rand(rng, i) < acc * acc:
                break
            r = rand(rng, i)
            cos_t = ((1.0 + xi) - math.exp(r * math.log1p(2.0 * xi))) / xi
    if cos_t < -1.0:
        cos_t = -1.0
    if cos_t > 1.0:
        cos_t = 1.0
    nx, ny, nz = new_k_from_angle(kx, ky, kz, Ep, cos_t, m, a, rng, i)
    return nx, ny, nz, True


@jit
def scatter_impurity(kx, ky, kz, E, p, m, a, rng, i):
    """IonizedImpurity.scatter (Eq. 36, elastic)."""
    r = rand(rng, i)
    cos_t = 1.0 - 2.0 * r / (1.0 + 4.0 * gamma_of_E(E, a) * (1.0 - r) / p[P_EBETA])
    nx, ny, nz = new_k_from_angle(kx, ky, kz, E, cos_t, m, a, rng, i)
    return nx, ny, nz, True


@jit
def eh_F(s, cx, cy, cz, Kx, Ky, Kz, gx, gy, gz, E_tot, m, a, mh):
    ex = cx - 0.5 * s * gx
    ey = cy - 0.5 * s * gy
    ez = cz - 0.5 * s * gz
    hx = Kx - ex
    hy = Ky - ey
    hz = Kz - ez
    return (E_of_k(math.sqrt(ex * ex + ey * ey + ez * ez), m, a)
            + HBAR * HBAR * (hx * hx + hy * hy + hz * hz) / (2.0 * mh) - E_tot)


@jit
def eh_solve_s(cx, cy, cz, Kx, Ky, Kz, gx, gy, gz, E_tot, s_guess, m, a, mh):
    """ElectronHole._solve_s: Newton from s = g, bisection fallback. Returns (s, ok)."""
    if not (eh_F(0.0, cx, cy, cz, Kx, Ky, Kz, gx, gy, gz, E_tot, m, a, mh) < 0.0):
        return 0.0, False
    tol = 1e-13 * abs(E_tot)
    s = s_guess
    done = False
    for _ in range(12):
        ex = cx - 0.5 * s * gx
        ey = cy - 0.5 * s * gy
        ez = cz - 0.5 * s * gz
        hx = Kx - ex
        hy = Ky - ey
        hz = Kz - ez
        kn = math.sqrt(ex * ex + ey * ey + ez * ez)
        Ee = E_of_k(kn, m, a)
        f = Ee + HBAR * HBAR * (hx * hx + hy * hy + hz * hz) / (2.0 * mh) - E_tot
        if abs(f) <= tol:
            done = True
            break
        dEdk = HBAR * HBAR * kn / (m * (1.0 + 2.0 * a * Ee))
        dk = -(ex * gx + ey * gy + ez * gz) / (2.0 * (kn if kn > 0.0 else 1.0))
        df = dEdk * dk + HBAR * HBAR * (hx * gx + hy * gy + hz * gz) / (2.0 * mh)
        step = f / df
        if math.isfinite(step):
            s = s - step
    if done and s >= 0.0:
        return s, True
    # bisection (ElectronHole._bisect)
    lo = 0.0
    hi = max(2.0 * s_guess, 1.0)
    for _ in range(200):
        if eh_F(hi, cx, cy, cz, Kx, Ky, Kz, gx, gy, gz, E_tot, m, a, mh) <= 0.0:
            hi *= 2.0
        else:
            break
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if eh_F(mid, cx, cy, cz, Kx, Ky, Kz, gx, gy, gz, E_tot, m, a, mh) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


@jit
def scatter_eh(kx, ky, kz, E, p, m, a, hole_x, hole_cdf, hole_lut, rng, i):
    """ElectronHole.scatter: sampled hole, Eq. 40 acceptance, Eq. 42 angle, exact two-body
    kinematics, Pauli blocking. Returns kx', ky', kz', accepted."""
    mh = p[P_MH]
    beta = p[P_BETA]
    kT = p[P_KT]
    gas = int(p[P_GAS])
    # hole state: HoleGas.sample_k (energy from the tabulated distribution, isotropic direction)
    Eh = kT * interp_lut(rand(rng, i), hole_cdf[gas], hole_x[gas], hole_lut[gas])
    kh = math.sqrt(2.0 * mh * Eh) / HBAR
    hx, hy, hz = random_unit(rng, i)
    k0x = hx * kh
    k0y = hy * kh
    k0z = hz * kh
    me = m if p[P_MASS_MODEL] < 0.5 else m * (1.0 + 2.0 * a * E)
    mR = me * mh / (me + mh)                                                   # Eq. 39
    gx = 2.0 * mR * (k0x / mh - kx / me)                                       # Eq. 41
    gy = 2.0 * mR * (k0y / mh - ky / me)
    gz = 2.0 * mR * (k0z / mh - kz / me)
    g = math.sqrt(gx * gx + gy * gy + gz * gz)
    if not (rand(rng, i) < 2.0 * g * beta / (g * g + beta * beta)):          # Eq. 40
        return kx, ky, kz, False
    r = rand(rng, i)
    cos_t = 1.0 - 2.0 * r / (1.0 + g * g * (1.0 - r) / (beta * beta))       # Eq. 42
    ux, uy, uz = unit(gx, gy, gz, rng, i)
    if cos_t < -1.0:
        cos_t = -1.0
    if cos_t > 1.0:
        cos_t = 1.0
    nx, ny, nz = rotate_about(ux, uy, uz, cos_t, 2.0 * math.pi * rand(rng, i))
    Kx = kx + k0x
    Ky = ky + k0y
    Kz = kz + k0z
    cx = mR / mh * Kx
    cy = mR / mh * Ky
    cz = mR / mh * Kz
    E_tot = E + HBAR * HBAR * (k0x * k0x + k0y * k0y + k0z * k0z) / (2.0 * mh)
    s, ok = eh_solve_s(cx, cy, cz, Kx, Ky, Kz, nx, ny, nz, E_tot, g, m, a, mh)
    if not ok:
        return kx, ky, kz, False
    px = cx - 0.5 * s * nx
    py = cy - 0.5 * s * ny
    pz = cz - 0.5 * s * nz
    qx = Kx - px
    qy = Ky - py
    qz = Kz - pz
    Eh_new = HBAR * HBAR * (qx * qx + qy * qy + qz * qz) / (2.0 * mh)
    pauli = int(p[P_PAULI])
    if pauli == 0:                                    # Fermi-Dirac (or MB) occupation of the gas
        x = Eh_new / kT - p[P_ETA]
        occ = 0.5 * (1.0 - math.tanh(0.5 * x)) if p[P_STATS] < 0.5 else math.exp(-x)
        free = rand(rng, i) < 1.0 - occ
    elif pauli == 1:                                  # C21 Eq. 44 step
        free = Eh_new >= p[P_EFH]
    else:
        free = True
    if not free:
        return kx, ky, kz, False
    return px, py, pz, True


# ----------------------------------------------------------------------------------- scattering
@jit
def final_state(t, p, v, kx, ky, kz, E, vpar, hole_x, hole_cdf, hole_lut, rng, i):
    """Final state of one event of a mechanism of type t with parameter row p, for an electron in
    valley v: (kx', ky', kz', accepted, valley'), valley' = -1 for a forbidden transition."""
    m = vpar[v, 0]
    a = vpar[v, 1]
    v_new = v
    if t == MT_ACOUSTIC:
        nx, ny, nz = new_k_isotropic(E, m, a, rng, i)
        acc = True
    elif t == MT_POP:
        nx, ny, nz, acc = scatter_pop(kx, ky, kz, E, p, m, a, rng, i)
    elif t == MT_IMPURITY:
        nx, ny, nz, acc = scatter_impurity(kx, ky, kz, E, p, m, a, rng, i)
    elif t == MT_EH:
        nx, ny, nz, acc = scatter_eh(kx, ky, kz, E, p, m, a, hole_x, hole_cdf, hole_lut, rng, i)
    else:                                              # MT_IV
        sign = -1.0 if p[P_EMISSION] > 0.5 else 1.0
        Ep = E + sign * p[P_HW] - p[P_DELTA]
        if Ep <= 0.0:
            return kx, ky, kz, False, -1
        v_new = int(p[P_VTO])
        nx, ny, nz = new_k_isotropic(Ep, vpar[v_new, 0], vpar[v_new, 1], rng, i)
        acc = True
    return nx, ny, nz, acc, v_new


@jit
def choose_equivalent(v_old, eqv_old, v_new, n_equiv, rng, i):
    """valleys.choose_equivalent_valley for one transfer."""
    n_eq = n_equiv[v_new]
    out = int(math.floor(rand(rng, i) * n_eq))
    if v_old == v_new and n_eq > 1:
        shift = 1 + int(math.floor(rand(rng, i) * (n_eq - 1)))
        out = (eqv_old + shift) % n_eq
    if v_new == 0:
        out = 0
    return out


@jit
def flight_rate(E, v, inside, gamma0, E_grid, E_lut, rate_table, local_tables, mech_local,
                thresholds, valley_mech, valley_nmech, cfg, icfg):
    """Simulation._flight_rate: Gamma0 in the field region / self-scattering mode, W_total(E)
    (bulk rates) in the field-free region."""
    if icfg[I_FLIGHT] == 1 or inside:
        return gamma0[v]
    j, w = bracket_lut(E_grid, E_lut, E, cfg[C_INV_EMAX])
    tot = 0.0
    for q in range(valley_nmech[v]):
        tot += mech_rate(valley_mech[v, q], E, j, w, -1.0, rate_table, local_tables, mech_local,
                         thresholds, 1)
    return tot


@jit
def scatter(i, G, fs, ist, vis, n_events, n_rej, n_self, rng,
            E_grid, E_lut, rate_table, local_tables, mech_local, thresholds, valley_mech, valley_nmech,
            mtype, par_offset, params, hole_x, hole_cdf, hole_lut, vpar, n_equiv, inv_tau,
            ftype, fpar, cfg, icfg):
    """Simulation._scatter for electron i (end of a flight drawn with rate G)."""
    v = ist[i, 0]
    E = fs[i, 5]
    fi = phi_index(fs[i, 0], ftype, fpar, cfg, icfg)
    n_phi = icfg[I_NPHI]
    j, wE = bracket_lut(E_grid, E_lut, E, cfg[C_INV_EMAX])
    nm = valley_nmech[v]
    r = rand(rng, i) * G
    choice = 0                       # number of cumulative rates below r (== nm: self-scattering)
    cum = 0.0
    for q in range(nm):
        cum += mech_rate(valley_mech[v, q], E, j, wE, fi, rate_table, local_tables, mech_local,
                         thresholds, n_phi)
        if r > cum:
            choice = q + 1
    tot = cum
    if tot > G * (1.0 + 1e-9):
        ist[i, 3] = ERR_RATE
        return
    if abs(G - tot) <= 1e-8 + 1e-9 * abs(tot) and choice > nm - 1:     # direct flight: no null
        choice = nm - 1
    if choice == nm:
        n_self[i] += 1
        return
    mi = valley_mech[v, choice]
    # depletion-dependent mechanism: choose the variant (LocalMechanism.choose_variant)
    row = par_offset[mi]
    li = mech_local[mi]
    if li >= 0 and fi >= 0.0:
        i0, i1, w = phi_split(fi, n_phi)
        W0 = local_tables[li, i0, j - 1] * (1.0 - wE) + local_tables[li, i0, j] * wE
        W1 = local_tables[li, i1, j - 1] * (1.0 - wE) + local_tables[li, i1, j] * wE
        totw = (1.0 - w) * W0 + w * W1
        p1 = w * W1 / totw if totw > 0.0 else w
        row += i1 if rand(rng, i) < p1 else i0
    elif li >= 0:
        row += icfg[I_NPHI] - 1                        # bulk variant (no depletion position)
    t = mtype[mi]
    nx, ny, nz, acc, v_new = final_state(t, params[row], v, fs[i, 2], fs[i, 3], fs[i, 4], E, vpar,
                                         hole_x, hole_cdf, hole_lut, rng, i)
    if v_new < 0:
        ist[i, 3] = ERR_FORBIDDEN
        return
    if not acc:
        n_rej[i, mi] += 1
        n_self[i] += 1
        return
    spin_flip(i, fs, ist, inv_tau, j, wE, ftype, fpar, cfg, icfg, rng)     # pre-scattering state
    fs[i, 2] = nx
    fs[i, 3] = ny
    fs[i, 4] = nz
    if t == MT_IV:
        ist[i, 1] = choose_equivalent(v, ist[i, 1], v_new, n_equiv, rng, i)
    ist[i, 0] = v_new
    vis[i, v_new] = 1
    E_new = E_of_k(math.sqrt(nx * nx + ny * ny + nz * nz), vpar[v_new, 0], vpar[v_new, 1])
    if not math.isfinite(E_new):
        ist[i, 3] = ERR_NONFINITE
        return
    fs[i, 5] = E_new
    n_events[i, mi] += 1


# ----------------------------------------------------------------------------------- surface
@jit
def valley_center(v, eqv, a_lat):
    """valleys.valley_center for one electron (L: (pi/a)(+-1, +-1, +-1) in L_DIRS order; X: (2pi/a) e_eqv)."""
    if v == 1:
        s = math.pi / a_lat
        if eqv == 0:
            return s, s, s
        if eqv == 1:
            return s, -s, -s
        if eqv == 2:
            return -s, s, -s
        return -s, -s, s
    if v == 2:
        s = 2.0 * math.pi / a_lat
        if eqv == 0:
            return s, 0.0, 0.0
        if eqv == 1:
            return 0.0, s, 0.0
        return 0.0, 0.0, s
    return 0.0, 0.0, 0.0


@jit
def fold_kpar(Kx, Ky, a_lat):
    """C21Surface.fold_kpar: |K_par| folded into the (001) surface Brillouin zone."""
    g = 2.0 * math.pi / a_lat
    u = (Kx + Ky) / (2.0 * g)
    w = (Kx - Ky) / (2.0 * g)
    best = math.inf
    for du in range(2):
        for dv in range(2):
            mm = math.floor(u) + du
            nn = math.floor(w) + dv
            rx = Kx - g * (mm + nn)
            ry = Ky - g * (mm - nn)
            d = rx * rx + ry * ry
            if d < best:
                best = d
    return math.sqrt(best)


@jit
def _region_k(E_tot, V, kpt):
    """Complex wavevector in a barrier slice: sqrt(2 m0 (E - V - E_par) / hbar^2)."""
    return cmath.sqrt(complex(2.0 * M0 * (E_tot - V - kpt) / (HBAR * HBAR), 0.0))


@jit
def transmission(E_tot, k_in, m_in, kpar, chi, V, L_b):
    """C21Surface.transmission for one electron: back-propagated transfer matrix through the
    barrier slices V (potential of each slice, relative to the Gamma edge at z = 0), from the
    vacuum (A = 1, B = 0) to the GaAs side; T = (k_out/m0) / (k_in/m_in) / |A|^2."""
    kpt = HBAR * HBAR * kpar * kpar / (2.0 * M0)
    eps = E_tot - chi - kpt
    if not (eps > 0.0 and k_in > 0.0):
        return 0.0
    N = V.shape[0]
    dx = L_b / N
    k_out = math.sqrt(2.0 * M0 * eps) / HBAR
    A = complex(1.0, 0.0)
    B = complex(0.0, 0.0)
    kb = complex(k_out, 0.0)
    mb = M0
    for i in range(N, -1, -1):              # interface between region i and i + 1, at x = i dx
        if i == 0:
            ka = complex(k_in, 0.0)
            ma = m_in
        else:
            ka = _region_k(E_tot, V[i - 1], kpt)
            ma = M0
        if abs(ka) < 1e-6:
            ka = complex(1e-6, 0.0)
        x = i * dx
        rho = (kb / mb) / (ka / ma)
        eb_p = cmath.exp(1j * kb * x)
        eb_m = cmath.exp(-1j * kb * x)
        ea_p = cmath.exp(1j * ka * x)
        ea_m = cmath.exp(-1j * ka * x)
        Aa = 0.5 * ((1.0 + rho) * A * eb_p + (1.0 - rho) * B * eb_m) / ea_p
        Ba = 0.5 * ((1.0 - rho) * A * eb_p + (1.0 + rho) * B * eb_m) / ea_m
        A = Aa
        B = Ba
        kb = ka
        mb = ma
    return (k_out / M0) / (k_in / m_in) / (A.real * A.real + A.imag * A.imag)


@jit
def record_arrival(i, fs, ist, tiv, vis, n_events, has_arr, arr_f, arr_i, arr_vis, arr_ev):
    """First surface arrival of electron i (Simulation._arrival_record)."""
    has_arr[i] = 1
    for c in range(7):
        arr_f[i, c] = fs[i, c]
    for c in range(3):
        arr_f[i, 7 + c] = tiv[i, c]
        arr_vis[i, c] = vis[i, c]
    for c in range(5):
        arr_i[i, c] = ist[i, c]
    for c in range(n_events.shape[1]):
        arr_ev[i, c] = n_events[i, c]


@jit
def emit(i, fs, ist, branch, em_f, em_i, voff, surf_par, cfg):
    """Record the emission of electron i in its current (incident) state; status EMITTED."""
    b = branch[i]
    chi = surf_par[b, 0]
    v = ist[i, 0]
    E = fs[i, 5]
    kx = fs[i, 2]
    ky = fs[i, 3]
    kz = fs[i, 4]
    a_lat = cfg[C_ALAT]
    cx, cy, cz = valley_center(v, ist[i, 1], a_lat)
    Kx = kx + cx
    Ky = ky + cy
    Kz = kz + cz
    E_tot = voff[v] + E
    kpar = fold_kpar(Kx, Ky, a_lat)
    eps = E_tot - chi - HBAR * HBAR * kpar * kpar / (2.0 * M0)
    nrm = math.sqrt(Kx * Kx + Ky * Ky)
    sc = kpar / nrm if nrm > 0.0 else 0.0
    em_f[i, 0] = fs[i, 1]
    em_f[i, 1] = E
    em_f[i, 2] = kx
    em_f[i, 3] = ky
    em_f[i, 4] = kz
    em_f[i, 5] = Kx
    em_f[i, 6] = Ky
    em_f[i, 7] = Kz
    em_f[i, 8] = HBAR * Kx * sc
    em_f[i, 9] = HBAR * Ky * sc
    em_f[i, 10] = -math.sqrt(2.0 * M0 * (eps if eps > 0.0 else 0.0))
    em_f[i, 11] = E_tot - chi
    em_i[i, 0] = v
    em_i[i, 1] = ist[i, 1]
    em_i[i, 2] = ist[i, 2]
    em_i[i, 3] = ist[i, 5]
    ist[i, 3] = EMITTED


@jit
def surface_c21(i, fs, ist, branch, em_f, em_i, last_T, rng, vpar, voff, surf_par, surf_V, cfg):
    """C21Surface.interact for electron i at z = 0 (surface branch branch[i]): trapped if the total
    energy is below the vacuum level; else emitted with probability T, else reflected (and T is
    kept in last_T[i] for bounce_train)."""
    b = branch[i]
    chi = surf_par[b, 0]
    v = ist[i, 0]
    E = fs[i, 5]
    a_lat = cfg[C_ALAT]
    cx, cy, cz = valley_center(v, ist[i, 1], a_lat)
    E_tot = voff[v] + E
    if E_tot < chi:                                      # below the vacuum level: trapped
        ist[i, 3] = TRAPPED
        return
    kpar = fold_kpar(fs[i, 2] + cx, fs[i, 3] + cy, a_lat)
    m = vpar[v, 0]
    al = vpar[v, 1]
    m_in = m * (1.0 + 2.0 * al * E) if surf_par[b, 2] < 0.5 else m
    T = transmission(E_tot, -fs[i, 4], m_in, kpar, chi, surf_V[b], surf_par[b, 1])
    if rand(rng, i) < T:
        emit(i, fs, ist, branch, em_f, em_i, voff, surf_par, cfg)
        return
    last_T[i] = T
    fs[i, 4] = abs(fs[i, 4])                             # reflected back into the GaAs
    fs[i, 0] = 0.0


@jit
def encounter(i, fs, ist, tiv, vis, n_events, branch, has_arr, arr_f, arr_i, arr_vis, arr_ev,
              em_f, em_i, last_T, rng, vpar, voff, surf_par, surf_V, cfg, icfg):
    """Surface encounter handled in the kernel (surface mode 1: absorbing, 2: C21 model): first-
    arrival record, encounter counter, then the surface model. Mode 0 (any other surface model):
    stop with status PENDING for the host."""
    mode = icfg[I_SURF_MODE]
    if mode == 0:
        ist[i, 3] = PENDING
        return
    if ist[i, 5] == 0:
        record_arrival(i, fs, ist, tiv, vis, n_events, has_arr, arr_f, arr_i, arr_vis, arr_ev)
    ist[i, 5] += 1
    if mode == 1:
        ist[i, 3] = SURFACE
        return
    surface_c21(i, fs, ist, branch, em_f, em_i, last_T, rng, vpar, voff, surf_par, surf_V, cfg)


@jit
def bounce_train(i, dt, T, fs, ist, tiv, n_flight, branch, em_f, em_i, rng, vpar, voff,
                 surf_par, ftype, fpar, icfg, cfg):
    """Exact aggregation of repeated surface bounces (fast-engine reformulation, docs/PERFORMANCE.md).

    Electron i was just reflected (z = 0, k_z > 0) and the surface field pushes it back. The
    reference engine treats each return separately: it draws a flight, bounces analytically when
    t_ret = 2 hbar k_z / |F(0)| <= min(dt, dt_max), and asks the surface model again. Between the
    bounces the state repeats exactly (same E, k_par, |k_z|), so every return is the same Bernoulli
    trial with the transmission T of the previous encounter (T >= 0; T < 0: none stored, i.e. the
    electron was not just reflected by the surface model), and by memorylessness one
    flight time dt (already drawn, capped at t_max) covers the whole train: n_b = floor(dt / t_ret)
    returns; the number of failed trials before an emission is geometric in T. Returns
    (handled, emitted, remaining time): handled = False when the bounce criterion does not hold
    (normal flight); otherwise the bounces are booked (time, valley time, encounter counter) and,
    without emission, the remaining dt - n_b t_ret < t_ret is to be flown normally."""
    if T < 0.0:
        return False, False, dt
    F0 = -Q_E * field_Ez(0.0, ftype, fpar)
    if not (F0 < 0.0):
        return False, False, dt
    kz = fs[i, 4]
    t_ret = 2.0 * HBAR * kz / (-F0)
    if not (t_ret <= min(dt, cfg[C_DTMAX])):
        return False, False, dt
    n_b = math.floor(dt / t_ret)
    nb_done = n_b
    emitted = False
    if T > 0.0:
        if T >= 1.0:
            k_fail = 0.0
        else:
            k_fail = math.floor(math.log1p(-rand(rng, i)) / math.log1p(-T))
        if k_fail < n_b:
            nb_done = k_fail + 1.0
            emitted = True
    elapsed = nb_done * t_ret
    v = ist[i, 0]
    fs[i, 1] += elapsed
    fs[i, 6] += elapsed
    tiv[i, v] += elapsed
    n_flight[i] += 1
    tot = ist[i, 5] + nb_done                            # saturating int32 encounter counter
    ist[i, 5] = int(tot) if tot < 2147483647.0 else 2147483647
    if emitted:
        fs[i, 4] = -kz                                   # incident (outward) at the emitting return
        fs[i, 0] = 0.0
        emit(i, fs, ist, branch, em_f, em_i, voff, surf_par, cfg)
        return True, True, 0.0
    return True, False, dt - elapsed


# ----------------------------------------------------------------------------------- driver
@jit
def advance(i, budget, fs, ist, tiv, vis, n_events, n_rej, n_self, n_flight, rng, branch, start,
            has_arr, arr_f, arr_i, arr_vis, arr_ev, em_f, em_i, last_T,
            E_grid, E_lut, rate_table, local_tables, mech_local, thresholds, valley_mech, valley_nmech,
            gamma0, mtype, par_offset, params, hole_x, hole_cdf, hole_lut, vpar, n_equiv, inv_tau,
            ftype, fpar, cfg, icfg, voff, surf_par, surf_V):
    """Advance electron i by up to `budget` flights (the body of the Simulation.run loop for one
    electron). Stops at time-out, back contact, a final surface outcome, an error, or (surface
    mode 0) at a surface encounter with status PENDING for the host. fs columns: z, t, kx, ky, kz,
    E, dt_spin. ist columns: valley, eqv, spin, status, n_flips, n_surface."""
    t_max = cfg[C_TMAX]
    if start[i] == 1:                                    # start_at_surface (no arrival spin test)
        start[i] = 0
        encounter(i, fs, ist, tiv, vis, n_events, branch, has_arr, arr_f, arr_i, arr_vis, arr_ev,
                  em_f, em_i, last_T, rng, vpar, voff, surf_par, surf_V, cfg, icfg)
    for _ in range(budget):
        if ist[i, 3] != ALIVE:
            return
        v = ist[i, 0]
        E = fs[i, 5]
        if not (gamma0[v] > 0.0):
            ist[i, 3] = ERR_NOMECH
            return
        if E > cfg[C_EMAX]:
            ist[i, 3] = ERR_TABLE
            return
        m = vpar[v, 0]
        a = vpar[v, 1]
        z = fs[i, 0]
        kx = fs[i, 2]
        ky = fs[i, 3]
        kz = fs[i, 4]
        inside = in_field(z, kz, E, m, a, cfg)
        G = flight_rate(E, v, inside, gamma0, E_grid, E_lut, rate_table, local_tables, mech_local,
                        thresholds, valley_mech, valley_nmech, cfg, icfg)
        u = rand(rng, i)
        tau = -math.log1p(-u) / G if G > 0.0 else math.inf
        t0 = fs[i, 1]
        rem = t_max - t0
        dt = tau if tau < rem else rem
        timeout = tau >= rem
        T_prev = last_T[i]                    # transmission of the encounter that just reflected i
        last_T[i] = -1.0                      # valid for this flight only
        if (inside and icfg[I_SURF_MODE] == 2 and icfg[I_FLIP_ARRIVAL] == 0 and z <= 0.0
                and kz > 0.0):
            handled, emitted, dt_rest = bounce_train(i, dt, T_prev, fs, ist, tiv, n_flight, branch,
                                                     em_f, em_i, rng, vpar, voff, surf_par, ftype,
                                                     fpar, icfg, cfg)
            if emitted:
                return
            if handled:                       # bounces booked; fly the remainder (< t_ret)
                t0 = fs[i, 1]
                dt = dt_rest
                z = fs[i, 0]
                kz = fs[i, 4]
        if inside:
            z1, kz1, E1, dt_used, event = propagate_field(z, kx, ky, kz, E, m, a, dt, ftype, fpar,
                                                          icfg, cfg)
        else:
            z1, kz1, dt_used, event = propagate_free(z, kz, E, m, a, dt, icfg, cfg)
            E1 = E
        fs[i, 0] = z1
        fs[i, 4] = kz1
        fs[i, 5] = E1
        fs[i, 1] = t0 + dt_used
        fs[i, 6] += dt_used
        tiv[i, v] += dt_used
        n_flight[i] += 1
        if event == EV_SURFACE:
            if icfg[I_HAS_MODEL] == 1 and kz1 > 0.0:
                continue                      # zero-length stop of a reflected electron
            if icfg[I_FLIP_ARRIVAL] == 1:
                ja, ua = bracket_lut(E_grid, E_lut, fs[i, 5], cfg[C_INV_EMAX])
                spin_flip(i, fs, ist, inv_tau, ja, ua, ftype, fpar, cfg, icfg, rng)
            encounter(i, fs, ist, tiv, vis, n_events, branch, has_arr, arr_f, arr_i, arr_vis, arr_ev,
                      em_f, em_i, last_T, rng, vpar, voff, surf_par, surf_V, cfg, icfg)
            continue                          # reflected electrons fly on; others stop above
        if event == EV_BACK:
            ist[i, 3] = BACK
            return
        if event != EV_NONE:                  # EV_REGION / EV_WALL: next flight, no scattering
            continue
        if timeout:
            ist[i, 3] = TIMEOUT
            return
        scatter(i, G, fs, ist, vis, n_events, n_rej, n_self, rng,
                E_grid, E_lut, rate_table, local_tables, mech_local, thresholds, valley_mech,
                valley_nmech, mtype, par_offset, params, hole_x, hole_cdf, hole_lut, vpar, n_equiv,
                inv_tau, ftype, fpar, cfg, icfg)
