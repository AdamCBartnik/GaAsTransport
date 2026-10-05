"""Equilibrium hole gas used by electron-hole scattering.

Holes are not transported. They are a thermal bath whose state is sampled whenever an
electron-hole collision is attempted (Karkare 2013 / thesis approach; user decision 3).

Model (docs/MODEL_ASSUMPTIONS.md):
  * Parabolic, isotropic hh and lh bands (C21 Eq. 3), sharing one chemical potential mu
    (measured from the VBM *into* the valence band, so mu > 0 means degenerate).
  * Hole kinetic energy E_h = hbar^2 k^2 / (2 m_b) >= 0.
  * Fermi-Dirac:      f_b(E) = 1 / (1 + exp((E - mu)/kT)),
                      p = sum_b N_b F_{1/2}(mu/kT),   N_b = 2 (m_b kT / 2 pi hbar^2)^(3/2)
    Maxwell-Boltzmann (option): f_b(E) = exp((mu - E)/kT), p = sum_b N_b exp(mu/kT).
    F_{1/2} is the normalized Fermi-Dirac integral (2/sqrt(pi)) Int sqrt(x)/(1 + e^(x-eta)) dx.
  * The split-off band is neglected (Delta_so >> kT).
This hole chemical potential is used only for e-h scattering. The C21 prescriptions for band
bending (Eq. 58, single band) and for BAP (Eq. 44) are kept unchanged where C21 uses them.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from . import bands
from .constants import HBAR

BAND_MASS_ATTR = {"hh": "m_hh", "lh": "m_lh"}


def fermi_integral_half(eta):
    """Normalized F_{1/2}(eta) = (2/sqrt(pi)) Int_0^inf sqrt(x)/(1+exp(x-eta)) dx."""
    # substitute x = t^2 (smooth integrand), split at the Fermi edge
    if eta < 0:     # factor out e^eta: f = e^eta e^-x / (1 + e^(eta - x))
        def g(t):
            return 2 * t * t * np.exp(-t * t) / (1 + np.exp(eta - t * t))
        val = integrate.quad(g, 0.0, np.sqrt(60.0), epsabs=0, epsrel=1e-11, limit=200)[0]
        return 2 / np.sqrt(np.pi) * np.exp(eta) * val
    t_edge, t_max = np.sqrt(eta), np.sqrt(eta + 60.0)

    def g(t):
        return 2 * t * t * 0.5 * (1 - np.tanh(0.5 * (t * t - eta)))
    val = sum(integrate.quad(g, a, b, epsabs=0, epsrel=1e-11, limit=200)[0]
              for a, b in ((0.0, t_edge), (t_edge, t_max)) if b > a)
    return 2 / np.sqrt(np.pi) * val


class HoleGas:
    def __init__(self, sample, bands_=("hh", "lh"), statistics="fermi_dirac"):
        self.sample = sample
        self.kT = sample.kT
        self.bands = tuple(bands_)
        self.statistics = statistics
        mat = sample.material
        self.mass = {b: getattr(mat, BAND_MASS_ATTR[b]) for b in self.bands}
        self.N_eff = {b: 2 * (m * self.kT / (2 * np.pi * HBAR**2)) ** 1.5 for b, m in self.mass.items()}
        Ntot = sum(self.N_eff.values())
        if statistics == "fermi_dirac":
            self.eta = optimize.brentq(lambda e: Ntot * fermi_integral_half(e) - sample.p, -60, 200,
                                       xtol=1e-13, rtol=1e-13)
            Fh = fermi_integral_half(self.eta)
        elif statistics == "maxwell_boltzmann":
            self.eta = float(np.log(sample.p / Ntot))
            Fh = np.exp(self.eta)
        else:
            raise ValueError(statistics)
        self.mu = self.eta * self.kT
        self.density = {b: self.N_eff[b] * Fh for b in self.bands}
        self._build_samplers()

    def occupation(self, E):
        """f(E_h) for hole kinetic energy E_h >= 0 [J]."""
        x = np.asarray(E, float) / self.kT - self.eta
        if self.statistics == "fermi_dirac":
            return 0.5 * (1 - np.tanh(0.5 * x))
        return np.exp(-x)

    def _build_samplers(self):
        # energy distribution P(x) dx ~ sqrt(x) f(x) dx, x = E/kT  (identical for both bands)
        x_max = max(self.eta, 0.0) + 50.0
        x = np.linspace(0.0, x_max, 20001)
        pdf = np.sqrt(x) * self.occupation(x * self.kT)
        cdf = integrate.cumulative_trapezoid(pdf, x, initial=0.0)
        self._x = x
        self._cdf = cdf / cdf[-1]
        self.mean_energy = self.kT * np.trapezoid(x * pdf, x) / np.trapezoid(pdf, x)

    def sample_energy(self, n, rng):
        """Hole kinetic energies [J] drawn from sqrt(E) f(E)."""
        return self.kT * np.interp(rng.random(n), self._cdf, self._x)

    def sample_k(self, band, n, rng):
        """Hole wavevectors (n, 3) for band 'hh' or 'lh': isotropic, |k| from the energy distribution."""
        E = self.sample_energy(n, rng)
        kmag = np.sqrt(2 * self.mass[band] * E) / HBAR
        return bands.random_unit_vectors(n, rng) * kmag[:, None]

    def summary(self):
        from .constants import to_ev, to_per_cm3
        return {"statistics": self.statistics, "eta": self.eta, "mu [meV]": to_ev(self.mu) * 1e3,
                **{f"p_{b} [cm^-3]": to_per_cm3(d) for b, d in self.density.items()},
                "<E_h> [meV]": to_ev(self.mean_energy) * 1e3}
