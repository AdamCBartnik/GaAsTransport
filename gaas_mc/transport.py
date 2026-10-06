"""Event-driven ensemble Monte Carlo transport.

Every electron is an independent history with its own clock. Each loop iteration advances
every alive electron by one free flight followed by one scattering event.

Free flight (ModelAssumptions.flight_mode). Each field has a depth z_max beyond which it vanishes
(C21 band bending: z_max = W_bb).
  * Field-free region (z >= z_max), "auto" mode: E is constant during the flight, so
        tau = -ln(U) / W_total(E),   W_total = sum_i W_i(E),
    is exact, with no self-scattering. The only null events left are rejections inside a
    mechanism (e-h Eq. 40, kinematics, Pauli blocking).
  * Field region (z < z_max), or flight_mode = "self_scattering": tau = -ln(U) / Gamma0[valley],
    with a constant bound Gamma0 >= max_E W_total(E) over the rate table. U*Gamma0 > W_total means
    self-scattering, so the varying E(t) is handled exactly (null-collision method).
  * A flight that reaches z = z_max is stopped there without a scattering event, and the next
    flight is drawn in the new region. This is exact because both the W_total(E) process (E
    constant) and the Gamma0 process are memoryless Poisson processes, which may be restarted at
    any stopping time.
During the flight, [C21] Eqs. 17-19:
    dz/dt = v_z(k),   hbar dk/dt = -e E_z(z) z_hat.
Without a field this is exact (z += v_z tau). With a field, kick-drift-kick (velocity-Verlet)
substeps of at most dt_max are used ([C21] uses 1 fs in the band-bending region).

End of flight: the mechanism is chosen by comparing U * G (G = W_total or Gamma0) with the
cumulative rates at the current energy. A mechanism may also reject an event (e-h Eq. 40,
kinematics, Pauli), which then counts as self-scattering.

Upper valleys (user decision 4, ModelAssumptions.side_valley_spin = "frozen"): the spin is
preserved in L and X, with zero additional relaxation. Gamma-valley rates are never applied
there. The time spent in each valley and whether L or X was ever visited are recorded per
particle and in every surface-arrival record.

Spin, [C21] Eq. 54: at each *real* event the spin flips with probability
0.5 (1 - exp(-dt/tau_s(E))), where dt is the time since the previous real event (self-scatterings
excluded) and E is the energy just before the event.

Boundaries:
    surface = "absorb"  stop at z = 0 and store a SurfaceArrivals record (default)
              "reflect" specular reflection k_z -> -k_z (e.g. internal-ESP studies, Eq. 55)
              "none"    no boundary (infinite homogeneous medium)
    z_back  : optional back boundary at z = z_back (thin films): back = "absorb" (status BACK) or
              "reflect" (specular; the flight stops at the wall, k_z flips, and the next flight
              follows, which is exact by memorylessness).
dt_max_field: velocity-Verlet substep in the field region. Default (None): h = min(2 fs, 0.05/omega),
              omega = sqrt(e max|dE_z/dz| / m_Gamma) (the harmonic time scale of the potential). For the
              C21 band bending at 1e19 cm^-3 this gives 0.25 fs, and crossing it conserves energy to
              < 0.15 meV, versus ~2 meV for the 1 fs step used in C21 (second order, tested).
surface_bounce_aggregation (default True; surface models that report T in their info dict):
              an electron reflected at z = 0 while the field pushes it back returns after
              t_ret = 2 hbar k_z / |F(0)|. Each return repeats the same state, hence the same
              transmission trial; with one flight time dt (memoryless) the floor(dt / t_ret) returns
              are booked at once, with a geometric number of failed trials before an emission. Exact
              in distribution; prevents the stall of grazing electrons (k_z -> 0 needs ~1/(Gamma0
              t_ret) returns). False: one return per loop iteration (brute force, for tests).

Backend (gaas_mc/backend.py): backend="numpy" (default) or "cupy" (GPU). The loop below is the same
code for both; with "cupy" the ensemble, the tables and the random numbers live on the device, and
the Result is returned as NumPy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import bands
from .assumptions import DEFAULT
from .backend import add_at, asarray, dev, device_rng, get_xp, to_host
from .depletion import LocalMechanism
from .valleys import choose_equivalent_valley, valley_center
from .constants import EV, FS, HBAR, PS, Q_E
from .fields import NoField
from .particle import ALIVE, BACK, EMITTED, SURFACE, TIMEOUT, TRAPPED, Ensemble
from .spin import flip_probability
from .surface import Emissions, SurfaceArrivals

EV_NONE, EV_SURFACE, EV_BACK, EV_REGION, EV_WALL = 0, 1, 2, 3, 4


@dataclass
class Snapshots:
    times: np.ndarray
    E: np.ndarray          # (n_t, N) energy, nan if not inside at that time
    z: np.ndarray
    spin: np.ndarray       # 0 if not inside
    valley: np.ndarray     # -1 if not inside

    def inside(self, i):
        return self.valley[i] >= 0

    def esp(self, i):
        m = self.inside(i)
        return float(self.spin[i][m].mean()) if m.any() else np.nan


@dataclass
class Result:
    ensemble: Ensemble
    arrivals: SurfaceArrivals
    snapshots: Snapshots | None
    mechanism_names: tuple
    n_iterations: int
    n_real: np.ndarray                       # real events per mechanism
    n_self: int
    n_rejected: np.ndarray                   # rejected candidates per mechanism
    flight_mode: str = "self_scattering"
    event_log: dict = field(default_factory=dict)
    emissions: Emissions = None              # with a surface model (e.g. surface_c21.C21Surface)


class Simulation:
    def __init__(self, sample, mechanisms, spin_model=None, field=None, t_max=370 * PS,
                 surface="absorb", z_back=None, E_table_max=2.0 * EV, n_table=4001,
                 dt_max_field=None, snapshot_times=(), log_events=False, gamma0_margin=1.02,
                 assumptions=DEFAULT, back="absorb", backend="numpy", surface_bounce_aggregation=True):
        self.surface_bounce_aggregation = surface_bounce_aggregation
        self.backend = backend
        self.xp = get_xp(backend)
        self.sample = sample
        self.material = sample.material
        self.mechanisms = list(mechanisms)
        self.names = tuple(m.name for m in self.mechanisms)
        if len(set(self.names)) != len(self.names):
            raise ValueError("mechanism names must be unique")
        self.spin_model = spin_model
        self.field = field if field is not None else NoField()
        self.t_max = float(t_max)
        # surface: "absorb" | "reflect" | "none" | a surface model with
        # interact(k, E, valley, K, rng, pid) -> (outcome, info)   (see gaas_mc/surface_c21.py)
        self.surface_model = surface if getattr(surface, "is_surface_model", False) else None
        if self.surface_model is not None:
            surface = "absorb"                      # geometry: stop at z = 0, then ask the model
        if surface not in ("absorb", "reflect", "none"):
            raise ValueError(surface)
        self.surface = surface
        self.z_back = z_back
        if back not in ("absorb", "reflect"):
            raise ValueError(back)
        self.back = back
        self.assumptions = assumptions.validate()
        self.spin_flip_at_arrival = assumptions.spin_flip_at_arrival
        mode = assumptions.flight_mode
        if mode == "direct" and not self.field.is_zero:
            raise ValueError("direct W_total(E) flights are only exact without a field; use 'auto'")
        self.z_field = 0.0 if self.field.is_zero else float(getattr(self.field, "z_max", np.inf))
        if mode == "auto":
            mode = ("direct" if self.z_field <= 0 else
                    "self_scattering" if np.isinf(self.z_field) else "hybrid")
        self.flight_mode = mode
        self.dt_max = float(dt_max_field) if dt_max_field is not None else self._auto_field_step()
        self.snapshot_times = np.sort(np.asarray(snapshot_times, float))
        self.log_events = log_events
        depls = {id(m.depletion): m.depletion for m in self.mechanisms if isinstance(m, LocalMechanism)}
        if len(depls) > 1:
            raise ValueError("all local mechanisms must share one depletion model")
        self.depl = next(iter(depls.values())) if depls else None
        if self.depl is not None and self.depl.field is not self.field:
            raise ValueError("the mechanisms' depletion model was built for a different field")
        self._build_tables(E_table_max, n_table, gamma0_margin)

    def _auto_field_step(self, eta=0.05, cap=2 * FS):
        if self.field.is_zero:
            return cap
        zmax = self.z_field if np.isfinite(self.z_field) else 1e-6
        z = np.linspace(0.0, zmax, 2001)
        grad = np.abs(np.gradient(self.field.Ez(z), z)).max()
        if grad == 0:
            return cap
        omega = np.sqrt(Q_E * grad / self.material.gamma.m_eff)
        return float(min(cap, eta / omega))

    # ------------------------------------------------------------------------------------
    # Rate tables
    # ------------------------------------------------------------------------------------
    def _build_tables(self, E_max, n, margin):
        # quadratic grid: dense near E = 0 where rates vary fastest
        grid = E_max * np.linspace(0.0, 1.0, n) ** 2
        grid[0] = 1e-9 * EV
        # put every threshold (and a point just above it) on the grid so the onset is resolved
        thr = [m.threshold for m in self.mechanisms if 0 < m.threshold < E_max]
        extra = [t * (1 + d) for t in thr for d in (0.0, 1e-9, 1e-6, 1e-4)]
        self.E_grid = np.unique(np.concatenate([grid, extra]))
        n = self.E_grid.size
        self.thresholds = np.array([m.threshold for m in self.mechanisms])
        self.E_table_max = E_max
        nv = len(self.material.valleys)
        self._m_valley = np.array([v.m_eff for v in self.material.valleys])
        self._a_valley = np.array([v.alpha for v in self.material.valleys])
        self.mech_by_valley = [[i for i, m in enumerate(self.mechanisms) if m.valley_from == v]
                               for v in range(nv)]
        self.rate_table = np.array([m.rate(self.E_grid) for m in self.mechanisms]).reshape(
            len(self.mechanisms), n)
        self.is_local = np.array([isinstance(m, LocalMechanism) for m in self.mechanisms], bool)
        for i in np.flatnonzero(self.is_local):
            self.mechanisms[i].tabulate(self.E_grid)       # (n_phi, n_E); last row = bulk
        self.gamma0 = np.zeros(nv)
        for v, idx in enumerate(self.mech_by_valley):
            if not idx:
                continue
            tot = self.rate_table[idx].sum(axis=0)
            if self.depl is not None:                       # bound over every local potential too
                loc = [i for i in idx if self.is_local[i]]
                fixed = sum((self.rate_table[i] for i in idx if not self.is_local[i]), np.zeros(n))
                tot = (fixed[None, :] + sum(self.mechanisms[i].table for i in loc)).max(axis=0)
            self.gamma0[v] = margin * tot.max()
        # spin relaxation: Gamma valley only; L/X frozen (zero), see side_valley_spin.
        # With the depletion model the Gamma table depends on the local potential: (n_phi, n_E).
        n_phi = self.depl.n if self.depl is not None else 1
        self.inv_tau_s = np.zeros((nv, n_phi, n))
        if self.spin_model is not None:
            if self.depl is not None and hasattr(self.spin_model, "rebuild"):
                for j in range(n_phi):
                    mech_j = [m.variants[j] if isinstance(m, LocalMechanism) else m
                              for m in self.mechanisms if m.valley_from == 0]
                    self.inv_tau_s[0, j] = self.spin_model.rebuild(self.depl.samples[j], mech_j).total(self.E_grid)
            else:
                self.inv_tau_s[0, :] = self.spin_model.total(self.E_grid)[None, :]

    def rates_at(self, E, valley_index, fi=None):
        """Interpolated rate matrix (M_v, n) for the mechanisms of one valley. fi: fractional index
        of the local potential (depletion model) per particle; None or no depletion model = bulk."""
        idx = self.mech_by_valley[valley_index]
        xp = self.xp
        E = xp.atleast_1d(E)
        # one bracket search for all mechanisms (same as np.interp per row, with end clamping)
        g = dev(self.E_grid, xp)
        j = np.clip(np.searchsorted(g, E, side="right"), 1, g.size - 1)
        w = np.clip((E - g[j - 1]) / (g[j] - g[j - 1]), 0.0, 1.0)
        T = dev(self.rate_table, xp)[idx]
        R = T[:, j - 1] * (1 - w) + T[:, j] * w
        if fi is not None and self.depl is not None:
            for row, i in enumerate(idx):
                if self.is_local[i]:
                    r0, r1, ww, _, _ = self.mechanisms[i].rates_at(E, fi)
                    R[row] = (1 - ww) * r0 + ww * r1
        return np.where(E[None, :] > dev(self.thresholds, xp)[idx][:, None], R, 0.0)

    def phi_index(self, z):
        """Fractional local-potential index (None without a depletion model)."""
        return None if self.depl is None else self.depl.frac_index(z)

    # ------------------------------------------------------------------------------------
    # Free flight
    # ------------------------------------------------------------------------------------
    def _valley_params(self, valley):
        xp = self.xp
        return dev(self._m_valley, xp)[valley], dev(self._a_valley, xp)[valley]

    def in_field(self, z, k, E, valley):
        """True where the particle is inside the field region (z < z_max). A particle exactly at
        z_max counts as inside if it is moving toward the surface."""
        if self.z_field <= 0:
            return self.xp.zeros(z.size, bool)
        if np.isinf(self.z_field):
            return self.xp.ones(z.size, bool)
        m, a = self._valley_params(valley)
        vz = HBAR * k[:, 2] / (m * (1 + 2 * a * E))
        return (z < self.z_field) | ((z <= self.z_field * (1 + 1e-12)) & (vz < 0))

    def propagate(self, z, k, E, valley, dt):
        """Propagate copies of (z, k, E) for times dt (per particle). Returns
        (z, k, E, dt_used, event), event in {EV_NONE, EV_SURFACE, EV_BACK, EV_REGION}. A particle
        that reaches an absorbing boundary or the field-region boundary stops there, and dt_used
        is the time at which it did."""
        z = z.copy(); k = k.copy(); E = E.copy(); dt = asarray(dt, float)
        m, a = self._valley_params(valley)
        event = self.xp.zeros(z.size, np.int8)
        dt_used = dt.copy()
        inside = self.in_field(z, k, E, valley)
        out = ~inside
        if out.any():
            r = self._propagate_free(z[out], k[out], E[out], m[out], a[out], dt[out])
            z[out], k[out], E[out], dt_used[out], event[out] = r
        if inside.any():
            r = self._propagate_field(z[inside], k[inside], E[inside], m[inside], a[inside], dt[inside])
            z[inside], k[inside], E[inside], dt_used[inside], event[inside] = r
        return z, k, E, dt_used, event

    def _propagate_free(self, z, k, E, m, a, dt):
        """Straight-line flight where the field vanishes (exact)."""
        n = z.size
        xp = self.xp
        event = xp.zeros(n, np.int8)
        vz = HBAR * k[:, 2] / (m * (1 + 2 * a * E))
        z1 = z + vz * dt
        t_ev = xp.full(n, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            if 0 < self.z_field < np.inf:                   # entering the field region
                hr = (z1 < self.z_field) & (vz < 0)
                t_ev[hr] = (z[hr] - self.z_field) / (-vz[hr])
                event[hr] = EV_REGION
            if self.surface in ("absorb", "reflect"):
                hs = (z1 <= 0) & (vz < 0)
                ts = z / (-vz)
                upd = hs & (ts < t_ev)
                t_ev[upd] = ts[upd]
                event[upd] = EV_SURFACE
            if self.z_back is not None:
                hb = (z1 >= self.z_back) & (vz > 0)
                tb = (self.z_back - z) / vz
                upd = hb & (tb < t_ev)
                t_ev[upd] = tb[upd]
                event[upd] = EV_BACK
        stop = event != EV_NONE
        dt_used = np.where(stop, np.clip(t_ev, 0.0, dt), dt)
        z1 = np.where(stop, z + vz * dt_used, z1)
        z1[event == EV_SURFACE] = 0.0
        z1[event == EV_REGION] = self.z_field
        if self.z_back is not None:
            z1[event == EV_BACK] = self.z_back
            if self.back == "reflect":
                wall = event == EV_BACK
                k[wall, 2] = -k[wall, 2]
                event[wall] = EV_WALL
        if self.surface == "reflect":                  # stop on the surface, flip k_z, restart (exact)
            refl = event == EV_SURFACE
            k[refl, 2] = -k[refl, 2]
            event[refl] = EV_WALL
        return z1, k, E, dt_used, event

    def _Ez_ext(self, z):
        """Field with mirror images across reflecting walls (method of images): E_z(-z) = -E_z(z) at a
        reflecting surface, and likewise about a reflecting back wall. A substep that ends beyond the
        wall, mirrored back (z -> -z, k_z -> -k_z), is then the reflected trajectory to integrator
        order, and time always advances."""
        z = asarray(z, float)
        if self.surface == "reflect" and np.any(z < 0):
            zi = np.where(z < 0, -z, z)
            Ez = self.field.Ez(zi)
            Ez = np.where(z < 0, -Ez, Ez)
        else:
            Ez = self.field.Ez(z)
        if self.z_back is not None and self.back == "reflect" and np.any(z > self.z_back):
            beyond = z > self.z_back
            Ez = np.where(beyond, -self.field.Ez(np.where(beyond, 2 * self.z_back - z, z)), Ez)
        return Ez

    def _kdk(self, z, k, m, a, h):
        """One kick-drift-kick (velocity-Verlet) substep of length h, Eqs. 17-18."""
        F = -Q_E * self._Ez_ext(z)                      # force along z, Eq. 18
        k[:, 2] += 0.5 * F * h / HBAR
        E = bands.E_of_k(np.linalg.norm(k, axis=1), m, a)
        vz = HBAR * k[:, 2] / (m * (1 + 2 * a * E))
        z = z + vz * h
        F = -Q_E * self._Ez_ext(z)
        k[:, 2] += 0.5 * F * h / HBAR
        return z, k

    def _surface_crossing_time(self, z_old, k_old, m, a, h, h_lin):
        """Time to reach z = 0 within a substep that crosses the surface. Linear interpolation in z
        is badly biased for short arcs near the surface (an electron near the top of an arc of
        duration << h): it underestimates the crossing time, hence |k_z| at the surface, and
        repeated returns drift k_z -> 0. Here the motion over the substep is z(t) = z_old + v t +
        a_z t^2 / 2 with the force at z_old and the velocity mass m(1 + 2 alpha E); the smallest
        root in [0, h] is used (linear interpolation if there is none)."""
        E0 = bands.E_of_k(np.linalg.norm(k_old, axis=1), m, a)
        mv = m * (1 + 2 * a * E0)
        v = HBAR * k_old[:, 2] / mv
        acc = -Q_E * self._Ez_ext(z_old) / mv
        A2, B, Cc = 0.5 * acc, v, z_old
        with np.errstate(invalid="ignore", divide="ignore"):
            D = B * B - 4 * A2 * Cc
            sq = np.sqrt(np.where(D >= 0, D, 0.0))
            q = -0.5 * (B + np.where(B >= 0, sq, -sq))
            t1 = np.where(A2 != 0, q / A2, np.inf)
            t2 = np.where(q != 0, Cc / q, np.inf)
            t1 = np.where((t1 >= 0) & (t1 <= h * (1 + 1e-9)), t1, np.inf)
            t2 = np.where((t2 >= 0) & (t2 <= h * (1 + 1e-9)), t2, np.inf)
            t = np.minimum(t1, t2)
            ok = (D >= 0) & np.isfinite(t)
        return np.where(ok, np.minimum(t, h), h_lin)

    def _surface_crossing_k(self, z_old, k_old, kc, m, a, sel):
        """k at the surface crossing (rows sel): k_par unchanged, |k_z| from total-energy
        conservation E(z_old) + E_C(z_old) = E(0) + E_C(0) (exact, independent of the step), k_z < 0.
        Keeps the integrator's value where energy conservation has no solution."""
        E0 = bands.E_of_k(np.linalg.norm(k_old, axis=1), m, a)
        Es = E0 + self.field.band_edge(np.maximum(z_old, 0.0)) - self.field.band_edge(0.0 * z_old)
        kt2 = bands.k_of_E(np.maximum(Es, 0.0), m, a) ** 2
        kz2 = kt2 - k_old[:, 0] ** 2 - k_old[:, 1] ** 2
        use = sel & (kz2 > 0)
        kc = kc.copy()
        kc[use, 2] = -np.sqrt(kz2[use])
        return kc

    def _propagate_field(self, z, k, E, m, a, dt):
        """Velocity-Verlet substeps (<= dt_max) inside the field region. A substep that crosses a
        boundary (surface, back contact, or field-region boundary z_max) is redone with the
        interpolated fraction so the particle stops on the boundary."""
        n = z.size
        xp = self.xp
        event = xp.zeros(n, np.int8)
        remaining = dt.copy()
        elapsed = xp.zeros(n)
        zr = self.z_field if 0 < self.z_field < np.inf else None
        if self.surface_model is not None:
            # just reflected by the surface model (z = 0, moving inward) while the field pushes back:
            # if it returns within one substep, do the bounce analytically (constant force F at z = 0):
            # return time 2 hbar k_z / |F|, k_z -> -k_z (exact by time-reversal symmetry, any dispersion).
            at = (z <= 0) & (k[:, 2] > 0)
            if np.any(at):
                F0 = -Q_E * float(self.field.Ez(np.array([0.0]))[0])
                if F0 < 0:
                    t_ret = 2 * HBAR * k[:, 2] / (-F0)
                    bounce = at & (t_ret <= np.minimum(dt, self.dt_max))
                    k[bounce, 2] = -k[bounce, 2]
                    z[bounce] = 0.0
                    elapsed[bounce] = t_ret[bounce]
                    remaining[bounce] = 0.0
                    event[bounce] = EV_SURFACE
        active = remaining > 0
        while np.any(active):
            ia = xp.flatnonzero(active)
            h = np.minimum(remaining[ia], self.dt_max)
            z_old, k_old = z[ia].copy(), k[ia].copy()
            z_new, k_new = self._kdk(z_old.copy(), k_old.copy(), m[ia], a[ia], h)
            # reflecting walls: mirror the full substep (z -> -z, k_z -> -k_z), with the image field of
            # _Ez_ext. This always advances time; stopping on the wall could stall when the field pushes
            # the electron back onto it.
            if self.surface == "reflect":
                rs = z_new < 0
                z_new[rs] = -z_new[rs]
                k_new[rs, 2] = -k_new[rs, 2]
            if self.z_back is not None and self.back == "reflect":
                rb = z_new > self.z_back
                z_new[rb] = 2 * self.z_back - z_new[rb]
                k_new[rb, 2] = -k_new[rb, 2]
            cross_s = (z_new <= 0) if self.surface == "absorb" else xp.zeros(ia.size, bool)
            cross_b = ((z_new >= self.z_back) if (self.z_back is not None and self.back == "absorb")
                       else xp.zeros(ia.size, bool))
            cross_r = (z_new >= zr) & ~cross_b if zr is not None else xp.zeros(ia.size, bool)
            cr = cross_s | cross_b | cross_r
            if np.any(cr):
                zb = np.where(cross_s, 0.0, np.where(cross_b, self.z_back or 0.0, zr or 0.0))
                f = np.clip((z_old - zb) / (z_old - z_new), 0.0, 1.0)
                hc = h * f
                if np.any(cross_s):
                    hc = np.where(cross_s, self._surface_crossing_time(z_old, k_old, m[ia], a[ia], h, hc), hc)
                _, kc = self._kdk(z_old[cr].copy(), k_old[cr].copy(), m[ia][cr], a[ia][cr], hc[cr])
                if np.any(cross_s):
                    kc = self._surface_crossing_k(z_old[cr], k_old[cr], kc, m[ia][cr], a[ia][cr],
                                                  cross_s[cr])
                z_new[cr] = zb[cr]
                k_new[cr] = kc
                h = np.where(cr, hc, h)
                event[ia[cross_s]] = EV_SURFACE
                event[ia[cross_b]] = EV_BACK
                event[ia[cross_r]] = EV_REGION
            z[ia] = z_new
            k[ia] = k_new
            elapsed[ia] += h
            remaining[ia] -= h
            remaining[event != EV_NONE] = 0.0
            active = remaining > 1e-30
        stopped = event != EV_NONE
        dt_used = np.where(stopped, elapsed, dt)
        E = bands.E_of_k(np.linalg.norm(k, axis=1), m, a)
        return z, k, E, dt_used, event

    # ------------------------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------------------------
    def run(self, ens: Ensemble, rng: np.random.Generator, max_iterations=10_000_000,
            start_at_surface=False) -> Result:
        """start_at_surface: electrons at z = 0 with k_z < 0 (e.g. Ensemble.from_arrivals of an
        earlier run with an absorbing surface) meet the surface model first. The Eq. 54 arrival spin
        test is not repeated (it was applied when those arrivals were recorded).
        rng: a NumPy Generator; with backend="cupy" a device generator is seeded from it."""
        xp = self.xp
        ens = ens.copy() if xp is np else ens.to(xp)
        rng = device_rng(rng, xp)
        N = len(ens)
        M = len(self.mechanisms)
        n_events = xp.zeros((N, M), np.int32)
        n_real = np.zeros(M, np.int64)
        n_rej = np.zeros(M, np.int64)
        n_self = 0
        arrivals = []
        emissions = []
        last_T = xp.full(N, -1.0)            # transmission of the encounter that just reflected i
        log = {"mech": [], "E_before": [], "E_after": [], "valley_before": [], "valley_after": []}

        snaps = None
        if self.snapshot_times.size:
            nt = self.snapshot_times.size
            snaps = Snapshots(self.snapshot_times, xp.full((nt, N), np.nan), xp.full((nt, N), np.nan),
                              xp.zeros((nt, N), np.int8), xp.full((nt, N), -1, np.int8))

        if np.any(ens.E > self.E_table_max):
            raise ValueError("initial energies exceed the rate-table range; raise E_table_max")

        if start_at_surface:
            if self.surface_model is None:
                raise ValueError("start_at_surface needs a surface model")
            ih = xp.flatnonzero((ens.status == ALIVE) & (ens.z <= 0) & (ens.k[:, 2] < 0))
            arrivals.append(self._arrival_record(ens, ih, n_events))
            ens.n_surface[ih] += 1
            self._surface_interaction(ens, ih, rng, emissions, last_T)

        it = 0
        idx = xp.flatnonzero(ens.status == ALIVE)
        gamma0 = dev(self.gamma0, xp)
        while idx.size:
            it += 1
            if it > max_iterations:
                raise RuntimeError("max_iterations exceeded")
            v = ens.valley[idx]
            if np.any(gamma0[v] <= 0):
                raise RuntimeError("an electron is in a valley without mechanisms")
            if np.any(ens.E[idx] > self.E_table_max):
                raise RuntimeError("electron energy left the rate table; raise E_table_max")
            inside = self.in_field(ens.z[idx], ens.k[idx], ens.E[idx], v)
            G = self._flight_rate(ens.E[idx], v, inside)
            with np.errstate(divide="ignore"):
                tau = -np.log1p(-rng.random(idx.size)) / G          # G = 0 -> infinite flight
            t0 = ens.t[idx]
            dt = np.minimum(tau, self.t_max - t0)
            timeout = tau >= self.t_max - t0
            if self.surface_model is not None:
                lt = last_T[idx]
                last_T[idx] = -1.0                       # valid for this flight only
                if self.surface_bounce_aggregation and not self.spin_flip_at_arrival:
                    keep, dt = self._bounce_trains(ens, idx, v, inside, lt, dt, rng, emissions)
                    if not bool(np.all(keep)):
                        idx, v, inside, G, dt, timeout = (x[keep] for x in (idx, v, inside, G, dt, timeout))
                        if idx.size == 0:
                            idx = xp.flatnonzero(ens.status == ALIVE)
                            continue
                    t0 = ens.t[idx]
            state0 = (ens.z[idx].copy(), ens.k[idx].copy(), ens.E[idx].copy())

            z1, k1, E1, dt_used, event = self.propagate(ens.z[idx], ens.k[idx], ens.E[idx], v, dt)
            if snaps is not None:
                self._record_snapshots(ens, idx, state0, v, t0, dt_used, snaps)
            ens.z[idx], ens.k[idx], ens.E[idx] = z1, k1, E1
            ens.t[idx] = t0 + dt_used
            ens.dt_spin[idx] += dt_used
            add_at(ens.time_in_valley, (idx, v), dt_used)

            # --- boundary events
            hit = event == EV_SURFACE
            if self.surface_model is not None:
                # zero-length stop of an electron just reflected inward (k_z > 0, possible only at
                # integrator round-off near the analytic-bounce threshold): not an encounter
                hit &= ens.k[idx, 2] <= 0
            if np.any(hit):
                ih = idx[hit]
                if self.spin_flip_at_arrival:
                    self._spin_flip(ens, ih, rng)
                first = ih[ens.n_surface[ih] == 0]
                if first.size:
                    arrivals.append(self._arrival_record(ens, first, n_events))
                ens.n_surface[ih] += 1
                if self.surface_model is None:
                    ens.status[ih] = SURFACE
                else:
                    self._surface_interaction(ens, ih, rng, emissions, last_T)
            ens.status[idx[event == EV_BACK]] = BACK
            # EV_REGION / EV_WALL: the flight stopped at the field boundary or reflected off the back
            # wall; there is no scattering event, and the next flight follows
            done_t = timeout & (event == EV_NONE)
            ens.status[idx[done_t]] = TIMEOUT

            # --- scattering at the end of the flight
            sc = (event == EV_NONE) & ~timeout
            isc = idx[sc]
            if isc.size:
                n_self += self._scatter(ens, isc, G[sc], rng, n_events, n_real, n_rej, log)
            idx = xp.flatnonzero(ens.status == ALIVE)

        arr = SurfaceArrivals.concatenate(arrivals, M, self.names).to_host()
        if self.log_events:
            log = {key: (to_host(xp.concatenate(val)) if val else np.zeros(0)) for key, val in log.items()}
        else:
            log = {}
        if snaps is not None:
            snaps = Snapshots(snaps.times, to_host(snaps.E), to_host(snaps.z), to_host(snaps.spin),
                              to_host(snaps.valley))
        em = Emissions.concatenate(emissions).to_host() if self.surface_model is not None else None
        return Result(ensemble=ens.to_host(), arrivals=arr, snapshots=snaps, mechanism_names=self.names,
                      n_iterations=it, n_real=n_real, n_self=n_self, n_rejected=n_rej,
                      flight_mode=self.flight_mode, event_log=log, emissions=em)

    def _surface_interaction(self, ens, ih, rng, emissions, last_T=None):
        """Ask the surface model what happens to electrons ih at z = 0: emitted, trapped, or reflected
        back into the semiconductor (specularly, k_z -> -k_z). For reflected electrons the trial's
        transmission (info["T"], if the model reports it) is kept in last_T for _bounce_trains."""
        from .surface_c21 import EMIT, REFLECT, TRAP
        K = ens.k[ih] + valley_center(ens.valley[ih], ens.eqv[ih], self.material.a_lat)
        outcome, info = self.surface_model.interact(ens.k[ih], ens.E[ih], ens.valley[ih], K, rng,
                                                    pid=ens.pid[ih])
        em = outcome == EMIT
        if np.any(em):
            self._record_emission(ens, ih[em], K[em], info["p_vac"][em], info["E_vac_kin"][em], emissions)
        ens.status[ih[outcome == TRAP]] = TRAPPED
        refl = outcome == REFLECT
        rf = ih[refl]
        ens.k[rf, 2] = np.abs(ens.k[rf, 2])           # back into the semiconductor (+z)
        ens.z[rf] = 0.0
        if last_T is not None and "T" in info:
            last_T[rf] = info["T"][refl]

    def _record_emission(self, ens, ie, K, p_vac, E_vac, emissions):
        emissions.append(Emissions(
            t=ens.t[ie].copy(), E=ens.E[ie].copy(), k=ens.k[ie].copy(), K=K.copy(),
            valley=ens.valley[ie].copy(), eqv=ens.eqv[ie].copy(), spin=ens.spin[ie].copy(),
            spin0=ens.spin0[ie].copy(), z0=ens.z0[ie].copy(), E0=ens.E0[ie].copy(),
            band=ens.band[ie].copy(), n_surface=ens.n_surface[ie].copy(), pid=ens.pid[ie].copy(),
            p_vac=p_vac.copy(), E_vac=E_vac.copy()))
        ens.status[ie] = EMITTED

    def _bounce_trains(self, ens, idx, v, inside, lt, dt, rng, emissions):
        """Exact aggregation of repeated surface returns (see surface_bounce_aggregation in the module
        docstring). Electrons idx just reflected by the surface model (lt >= 0: that trial's T), at
        z = 0 moving inward in the field region, with t_ret <= min(dt, dt_max): the floor(dt / t_ret)
        returns of this flight are booked at once. Returns (keep, dt): keep is False for electrons
        emitted in the train; dt is the remaining flight time (< t_ret) of the others."""
        xp = self.xp
        keep = xp.ones(idx.size, bool)
        cand = inside & (lt >= 0) & (ens.z[idx] <= 0) & (ens.k[idx, 2] > 0)
        if not bool(np.any(cand)):
            return keep, dt
        F0 = -Q_E * float(self.field.Ez(np.array([0.0]))[0])
        if not F0 < 0:
            return keep, dt
        kz = ens.k[idx, 2]
        t_ret = 2 * HBAR * kz / (-F0)
        tr = cand & (t_ret <= np.minimum(dt, self.dt_max))
        if not bool(np.any(tr)):
            return keep, dt
        pos = xp.flatnonzero(tr)
        it = idx[pos]
        T = lt[pos]
        trt = t_ret[pos]
        n_b = np.floor(dt[pos] / trt)
        u = rng.random(it.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            k_fail = np.where(T >= 1, 0.0, np.where(T > 0, np.floor(np.log1p(-u) / np.log1p(-T)), np.inf))
        em = k_fail < n_b
        nb_done = np.where(em, k_fail + 1, n_b)
        elapsed = nb_done * trt
        ens.t[it] += elapsed
        ens.dt_spin[it] += elapsed
        add_at(ens.time_in_valley, (it, v[pos]), elapsed)
        # saturating int32 encounter counter (a grazing train can add ~1e9 returns)
        ens.n_surface[it] = np.minimum(ens.n_surface[it] + nb_done, 2**31 - 1).astype(np.int32)
        if bool(np.any(em)):
            ie = it[em]
            ens.k[ie, 2] = -ens.k[ie, 2]                 # incident (outward) at the emitting return
            ens.z[ie] = 0.0
            K = ens.k[ie] + valley_center(ens.valley[ie], ens.eqv[ie], self.material.a_lat)
            _, info = self.surface_model.interact(ens.k[ie], ens.E[ie], ens.valley[ie], K, rng,
                                                  pid=ens.pid[ie])       # vacuum state only
            self._record_emission(ens, ie, K, info["p_vac"], info["E_vac_kin"], emissions)
            keep[pos[em]] = False
        dt = dt.copy()
        dt[pos] = dt[pos] - elapsed
        return keep, dt

    def _flight_rate(self, E, valley, inside):
        """Rate used to draw the free flight: W_total(E) in the field-free region (direct), or
        Gamma0 inside the field region or when self-scattering is forced."""
        G = dev(self.gamma0, self.xp)[valley].astype(float)
        if self.flight_mode == "self_scattering":
            return G
        free = ~inside
        for vv in self.xp.unique(valley[free]).tolist():
            sel = free & (valley == vv)
            G[sel] = self.rates_at(E[sel], vv).sum(axis=0)
        return G

    def _spin_flip(self, ens, ii, rng):
        """Eq. 54 for particles ii, with dt = ens.dt_spin; resets dt_spin."""
        inv = self._inv_tau_s_at(ens.E[ii], ens.valley[ii], self.phi_index(ens.z[ii]))
        P = flip_probability(ens.dt_spin[ii], inv)
        flip = rng.random(ii.size) < P
        ens.spin[ii[flip]] *= -1
        ens.n_flips[ii[flip]] += 1
        ens.dt_spin[ii] = 0.0

    def _inv_tau_s_at(self, E, valley, fi=None):
        xp = self.xp
        out = xp.zeros_like(E)
        T = dev(self.inv_tau_s, xp)
        g = dev(self.E_grid, xp)
        for vv in xp.unique(valley).tolist():
            sel = valley == vv
            if fi is None or T.shape[1] == 1:
                out[sel] = np.interp(E[sel], g, T[vv, -1])
                continue
            f = fi[sel]
            i0 = np.clip(np.floor(f).astype(int), 0, T.shape[1] - 1)
            i1 = np.minimum(i0 + 1, T.shape[1] - 1)
            w = f - i0
            j = np.clip(np.searchsorted(g, E[sel], side="right"), 1, g.size - 1)
            u = np.clip((E[sel] - g[j - 1]) / (g[j] - g[j - 1]), 0.0, 1.0)
            r0 = T[vv, i0, j - 1] * (1 - u) + T[vv, i0, j] * u
            r1 = T[vv, i1, j - 1] * (1 - u) + T[vv, i1, j] * u
            out[sel] = (1 - w) * r0 + w * r1
        return out

    def _scatter(self, ens, isc, G, rng, n_events, n_real, n_rej, log):
        """Choose and apply the scattering event for particles isc, given the flight rates G
        used to draw their flights. Returns the number of self-scatterings."""
        n_self = 0
        # freeze pre-event valleys: an electron that transfers valleys inside this loop
        # must not be scattered again in the destination valley's pass
        xp = self.xp
        v_before = ens.valley[isc].copy()
        for vv in xp.unique(v_before).tolist():
            in_v = v_before == vv
            iv = isc[in_v]
            mech_idx = self.mech_by_valley[vv]
            fi_v = self.phi_index(ens.z[iv])
            R = self.rates_at(ens.E[iv], vv, fi_v)               # (M_v, n)
            cum = np.cumsum(R, axis=0)
            Gv = G[in_v]
            if np.any(cum[-1] > Gv * (1 + 1e-9)):
                raise RuntimeError("total rate exceeds the flight rate; table/margin problem")
            r = rng.random(iv.size) * Gv
            choice = (r[None, :] > cum).sum(axis=0)              # == M_v -> self-scattering
            # direct flights (G = W_total) have no null events; guard against round-off
            direct = np.isclose(Gv, cum[-1], rtol=1e-9)
            choice = np.where(direct, np.minimum(choice, len(mech_idx) - 1), choice)
            n_self += int((choice == len(mech_idx)).sum())
            for j, mi in enumerate(mech_idx):
                sel = iv[choice == j]
                if sel.size == 0:
                    continue
                mech = self.mechanisms[mi]
                E_before = ens.E[sel].copy()
                if self.is_local[mi]:
                    k_new, accepted, v_new = mech.scatter(ens.k[sel], E_before, rng,
                                                          phi_index=fi_v[choice == j])
                else:
                    k_new, accepted, v_new = mech.scatter(ens.k[sel], E_before, rng)
                acc = sel[accepted]
                n_rej[mi] += int((~accepted).sum())
                n_self += int((~accepted).sum())
                if acc.size == 0:
                    continue
                # spin flip (Eq. 54) evaluated with the pre-scattering state
                self._spin_flip(ens, acc, rng)
                ens.k[acc] = k_new[accepted]
                v_old = ens.valley[acc].copy()
                ens.valley[acc] = v_new[accepted]
                if getattr(mech, "valley_to", None) is not None:      # intervalley: which equivalent valley
                    ens.eqv[acc] = choose_equivalent_valley(v_old, ens.eqv[acc], ens.valley[acc], rng)
                ens.visited[acc, ens.valley[acc]] = True
                m, a = self._valley_params(ens.valley[acc])
                ens.E[acc] = bands.E_of_k(np.linalg.norm(ens.k[acc], axis=1), m, a)
                if not np.all(np.isfinite(ens.E[acc])):
                    raise RuntimeError(f"non-finite energy after {mech.name}")
                n_events[acc, mi] += 1
                n_real[mi] += acc.size
                if self.log_events:
                    log["mech"].append(xp.full(acc.size, mi))
                    log["E_before"].append(E_before[accepted])
                    log["E_after"].append(ens.E[acc].copy())
                    log["valley_before"].append(xp.full(acc.size, vv))
                    log["valley_after"].append(ens.valley[acc].copy())
        return n_self

    def _record_snapshots(self, ens, idx, state0, v, t0, dt_used, snaps):
        """Record the state at snapshot times inside this flight, [t0, t0 + dt_used], by
        re-propagating copies of the flight's initial state (no boundary is crossed before dt_used)."""
        z0, k0, E0 = state0
        t1 = t0 + dt_used
        for i, ts in enumerate(snaps.times):
            sel = (t0 < ts) & (ts <= t1) if ts > 0 else (t0 == 0)
            if not np.any(sel):
                continue
            ii = idx[sel]
            if ts == 0:
                z, E = z0[sel], E0[sel]
            else:
                z, _, E, _, _ = self.propagate(z0[sel], k0[sel], E0[sel], v[sel], ts - t0[sel])
            snaps.E[i, ii] = E
            snaps.z[i, ii] = z
            snaps.spin[i, ii] = ens.spin[ii]
            snaps.valley[i, ii] = ens.valley[ii]

    def _arrival_record(self, ens, ih, n_events):
        return SurfaceArrivals(
            t=ens.t[ih].copy(), E=ens.E[ih].copy(), k=ens.k[ih].copy(), valley=ens.valley[ih].copy(),
            spin=ens.spin[ih].copy(), spin0=ens.spin0[ih].copy(), z0=ens.z0[ih].copy(),
            E0=ens.E0[ih].copy(), band=ens.band[ih].copy(), n_flips=ens.n_flips[ih].copy(),
            n_events=n_events[ih].copy(), pid=ens.pid[ih].copy(),
            time_in_valley=ens.time_in_valley[ih].copy(), visited=ens.visited[ih].copy(),
            eqv=ens.eqv[ih].copy(), dt_spin=ens.dt_spin[ih].copy(),
            K=ens.k[ih] + valley_center(ens.valley[ih], ens.eqv[ih], self.material.a_lat),
            band_edge_at_surface=float(self.field.band_edge(np.array([0.0]))[0]),
            mechanism_names=self.names)
