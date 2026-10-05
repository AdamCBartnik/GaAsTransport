"""Photoexcitation by monochromatic, circularly polarized light at normal incidence.

[C21] Sec. III A. Only direct (vertical) transitions into the Gamma valley; no multiphoton
or multielectron processes; delta-function pulse at t = 0.

Depth, Eq. 7:   z0 = -l ln(1 - r)      (l = absorption length; R does not affect the depth
                                         distribution, only the normalization of QE)
Energy, Eqs. 8-9 (energy + momentum conservation for a vertical transition from band h):
    dE_e = hw - Eg - (dE_h + Delta)
    dE_h = Gamma2/(2 alpha) [1 - sqrt(1 - 4 alpha (1 + alpha Gamma1) Gamma1 / Gamma2^2)]
    Gamma1 = hw - Eg - Delta,   Gamma2 = 1 + m_h/m_e + 2 alpha Gamma1
    Delta = 0 for hh and lh, Delta_so for so.  Eg = Eg(p), Eq. 10.
Broadening, Eq. 11:  E0 = dE_e +- (3/2) kT ln(1 - r), random sign; E0 <= 0 -> E0 = dE_e.
Spin, Eqs. 12-16 (D'yakonov & Perel 1971, spherical bands):
    ESP0 = sum_i P_i K_i / sum_i K_i,  i = hh, lh, so
    P_1 = 1/2,  K_1 = D sqrt(2x) (3 zeta - 3)^(-3/2)
    P_{2,3} = (1+zeta)(-+g - 6x + 9 zeta - 5) / (4 (+-zeta g + 6x - zeta - 3))
    K_{2,3} = D kappa (2x - 3 zeta kappa^2 - 1) / (3 zeta (2x - 3 zeta kappa^2 - 1) + 9 kappa^2 - 1)
    kappa^2 = (+-g + 6 zeta x - 3 zeta - 1) / (9 (zeta^2 - 1))      (upper: lh, lower: so)
    g = sqrt(36 x^2 - 12 x (zeta + 3) + (3 zeta + 1)^2),  x = (hw - Eg)/Delta_so
    zeta = (4/3) (1/m_e + 3/(4 m_lh) + 1/(4 m_hh)) / (1/m_lh - 1/m_hh)

Choices not fixed by the paper (see docs/IMPLEMENTATION_PLAN.md):
  * A3: the initial k direction is isotropic.
  * A4: band and spin assignment.
        spin_model="chubenko_text" (default): an hh fraction (1+ESP0)/2 with s = +1; the rest
        lh/so (split K_lh : K_so) with s = -1, as described in the text after Eq. 16. Of the two
        readings, this one reproduces the relative hh/lh/so peak heights of [C21] Fig. 5
        (validation/stage_a_excitation.py).
        spin_model="per_band": band i with probability K_i/sum K, then s = +1 with probability
        (1 + P_i)/2 (the D'yakonov-Perel 1971 per-band polarizations). Also reproduces Eq. 12.
  * A5: l(hw) must be supplied by the user (scalar [m] or callable hw[J] -> l[m]).
  * A16: hw < Eg raises unless below_gap_energy is given ([K13] uses 5 meV).
"""
from __future__ import annotations

import numpy as np

from . import bands
from .particle import Ensemble

HH, LH, SO = 0, 1, 2
BAND_NAMES = ("hh", "lh", "so")


def zeta_param(material):
    """Eq. 16, zeta."""
    me, mlh, mhh = material.gamma.m_eff, material.m_lh, material.m_hh
    return 4 / 3 * (1 / me + 3 / (4 * mlh) + 1 / (4 * mhh)) / (1 / mlh - 1 / mhh)


def dp71_weights(sample, hw):
    """Eqs. 13-16. Returns (P[3], K[3] normalized to sum 1, ESP0). Inaccessible bands get K = 0.

    The common constant D cancels in Eq. 12 and in the normalized weights.
    """
    mat = sample.material
    x = (hw - sample.Eg) / mat.Delta_so
    if x <= 0:
        raise ValueError("hw must exceed Eg for Eqs. 13-16")
    z = zeta_param(mat)
    g = np.sqrt(36 * x**2 - 12 * x * (z + 3) + (3 * z + 1) ** 2)
    P = np.zeros(3)
    K = np.zeros(3)
    P[HH] = 0.5
    K[HH] = np.sqrt(2 * x) * (3 * z - 3) ** -1.5
    for band, s in ((LH, +1.0), (SO, -1.0)):
        kap2 = (s * g + 6 * z * x - 3 * z - 1) / (9 * (z**2 - 1))
        if kap2 <= 0:            # band not reached (so below threshold x < 1)
            continue
        kap = np.sqrt(kap2)
        P[band] = (1 + z) * (-s * g - 6 * x + 9 * z - 5) / (4 * (s * z * g + 6 * x - z - 3))
        num = 2 * x - 3 * z * kap2 - 1
        K[band] = kap * num / (3 * z * num + 9 * kap2 - 1)
    if np.any(K < 0):
        raise ValueError(f"negative absorption weight from Eq. 14 at x={x}: {K}")
    K = K / K.sum()
    return P, K, float(np.dot(P, K))


def esp0(sample, hw):
    """Initial spin polarization, Eq. 12."""
    return dp71_weights(sample, hw)[2]


def excess_energy(sample, hw, band):
    """Eqs. 8-9: electron excess energy dE_e for a vertical transition from `band`.
    Returns <= 0 if the band is not accessible."""
    mat = sample.material
    a = mat.gamma.alpha
    me = mat.gamma.m_eff
    mh = (mat.m_hh, mat.m_lh, mat.m_so)[band]
    Delta = mat.Delta_so if band == SO else 0.0
    G1 = hw - sample.Eg - Delta
    if G1 <= 0:
        return G1
    G2 = 1 + mh / me + 2 * a * G1
    dEh = G2 / (2 * a) * (1 - np.sqrt(1 - 4 * a * (1 + a * G1) * G1 / G2**2))
    return G1 - dEh


def absorption_length(l, hw):
    return float(l(hw)) if callable(l) else float(l)


def photoexcite(sample, hw, n, rng, absorption_len, spin_model="chubenko_text",
                broadening=True, below_gap_energy=None, direction="isotropic"):
    """Generate n photoexcited electrons in the Gamma valley.

    Parameters
    ----------
    sample : material.Sample
    hw : photon energy [J]
    n : number of electrons
    rng : numpy.random.Generator
    absorption_len : l [m] or callable hw -> l
    spin_model : "chubenko_text" (default) | "per_band"   (ambiguity A4)
    broadening : apply Eq. 11
    below_gap_energy : if hw <= Eg, place electrons at this energy [J] (A16; [K13] uses 5 meV).
    direction : "isotropic" (A3)

    Returns
    -------
    Ensemble with z = z0, t = 0, valley = Gamma, band labels in ``ens.band``.
    """
    mat = sample.material
    gv = mat.gamma
    l_abs = absorption_length(absorption_len, hw)
    z0 = -l_abs * np.log1p(-rng.random(n))                                  # Eq. 7

    if hw <= sample.Eg:
        if below_gap_energy is None:
            raise ValueError("photon energy below Eg(p); pass below_gap_energy to override (A16)")
        E0 = np.full(n, float(below_gap_energy))
        band = np.full(n, HH, dtype=np.int8)
        spin = np.where(rng.random(n) < 0.75, 1, -1).astype(np.int8)       # 50% (Eq. 4)
    else:
        P, K, esp = dp71_weights(sample, hw)
        dE = np.array([excess_energy(sample, hw, b) for b in (HH, LH, SO)])
        if spin_model == "per_band":
            band = rng.choice(3, size=n, p=K).astype(np.int8)
            p_up = (1 + P[band]) / 2
            spin = np.where(rng.random(n) < p_up, 1, -1).astype(np.int8)
        elif spin_model == "chubenko_text":
            f_hh = (1 + esp) / 2
            rest = K[LH] + K[SO]
            w = np.array([f_hh, (1 - f_hh) * K[LH] / rest, (1 - f_hh) * K[SO] / rest])
            band = rng.choice(3, size=n, p=w).astype(np.int8)
            spin = np.where(band == HH, 1, -1).astype(np.int8)
        else:
            raise ValueError(spin_model)
        if np.any(dE[band] <= 0):
            raise RuntimeError("selected an energetically inaccessible band")
        E0 = dE[band].copy()
        if broadening:                                                     # Eq. 11
            sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
            E_try = E0 + sign * 1.5 * sample.kT * np.log1p(-rng.random(n))
            E0 = np.where(E_try > 0, E_try, E0)

    if direction != "isotropic":
        raise ValueError(direction)
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E0, gv.m_eff, gv.alpha)[:, None]
    ens = Ensemble.create(z=z0, k=k, E=E0, spin=spin, valley=0)
    ens.band[:] = band
    return ens
