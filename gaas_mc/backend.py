"""Array backend: NumPy (CPU) or CuPy (GPU) for the transport hot path.

There is ONE implementation of the physics. The vectorized routines take the array module from
their inputs (``xp_of``), so the same code runs on host (NumPy) or device (CuPy) arrays. NumPy
functions called on CuPy arrays dispatch to CuPy (NEP 18 / NEP 13), so most arithmetic is written
with ``np.`` as before; what must be backend-aware is:
  * array creation (zeros, full, arange, ...): ``xp = xp_of(x)``, then ``xp.zeros(...)``;
  * conversions: ``asarray(x, dtype)`` keeps the device of x (np.asarray would copy to host and
    fails on CuPy arrays);
  * constant tables built on the host (rate tables, samplers, valley constants): ``dev(table, xp)``
    returns a cached device copy (tables are never modified after they are built);
  * random numbers: a NumPy Generator on the host; a CuPy Generator on the device, seeded from the
    host generator (``device_rng``). The two streams differ, so GPU runs are statistically, not
    bitwise, equal to CPU runs. The NumPy path is unchanged, draw for draw.

Selection: ``Simulation(..., backend="numpy" | "cupy")``. Everything outside the transport loop
(material setup, rate tables, photoexcitation, analysis) stays on the host; results are returned
as NumPy arrays.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

BACKENDS = ("numpy", "cupy")
_cp = None          # CuPy is imported only when the "cupy" backend is requested (CPU-only machines
                    # and clusters never import it); until then no array can be a CuPy array.


def _import_cupy():
    global _cp
    if _cp is None:
        with warnings.catch_warnings():
            # CuPy warns when CUDA_PATH is unset even though pip-installed CUDA wheels work
            warnings.filterwarnings("ignore", message="CUDA path could not be detected")
            import cupy
        _cp = cupy
    return _cp


def available(name):
    if name == "numpy":
        return True
    if name != "cupy":
        return False
    try:
        return _import_cupy().cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_xp(name):
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            cp = _import_cupy()
        except ImportError as e:
            raise ImportError("backend 'cupy' requested but CuPy is not installed") from e
        try:
            ok = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            ok = False
        if not ok:
            raise RuntimeError("backend 'cupy' requested but no CUDA device is available")
        return cp
    raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")


def is_device(a):
    return _cp is not None and isinstance(a, _cp.ndarray)


def xp_of(*arrays):
    """Array module of the inputs: cupy if any input is a CuPy array, else numpy."""
    if _cp is not None:
        for a in arrays:
            if isinstance(a, _cp.ndarray):
                return _cp
    return np


def asarray(x, dtype=None):
    """np.asarray that keeps device arrays on the device."""
    if is_device(x):
        return x if dtype is None else x.astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def to_host(x):
    return _cp.asnumpy(x) if is_device(x) else x


_CACHE = {}


def dev(a, xp):
    """Device copy of a host constant array (cached; the host array must not be modified later).
    For xp = numpy the host array itself is returned."""
    if xp is np or is_device(a):
        return a
    key = id(a)
    hit = _CACHE.get(key)
    if hit is None or hit[0] is not a:
        hit = (a, xp.asarray(a))
        _CACHE[key] = hit
    return hit[1]


def device_rng(rng, xp):
    """Random generator for arrays of module xp. NumPy: the given Generator unchanged. CuPy: a
    CuPy Generator seeded from the host generator (advances it by one draw)."""
    if xp is np:
        return rng
    if is_device_rng(rng):
        return rng
    return xp.random.default_rng(int(rng.integers(0, 2**63 - 1)))


def is_device_rng(rng):
    return _cp is not None and isinstance(rng, _cp.random.Generator)


def add_at(a, index, values):
    """Unbuffered a[index] += values (np.add.at); cupyx.scatter_add on the device."""
    if is_device(a):
        import cupyx
        cupyx.scatter_add(a, index, values)
    else:
        np.add.at(a, index, values)
