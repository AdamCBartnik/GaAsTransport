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

Model assumptions (docs/MODEL_ASSUMPTIONS.md; selectable through ModelAssumptions):
  * initial_spin_rule = "per_band" (default; user decision 1). Eqs. 12-16 are authoritative:
        band i is chosen with probability K_i / sum K, and spin s = +1 with probability (1 + P_i)/2.
        The ensemble ESP equals Eq. 12 exactly in expectation.
    "chubenko_prose": the literal prose after Eq. 16 (hh -> s = +1; lh, so -> s = -1). To keep
        ESP0 equal to Eq. 12, the hh fraction is (1 + ESP0)/2 and the lh : so split is K_lh : K_so.
        This mode reproduces the relative peak heights of C21 Fig. 5 somewhat better.
  * initial_k_direction = "isotropic" (C21 is silent).
  * absorption_model = Adachi (1989) MDF for intrinsic GaAs (optics.py; user decision 2). Any
    object with absorption_coefficient(hv) can be passed instead. Doping-induced gap narrowing
    is NOT applied to the optical data. Photons with Eg(p) < hv but alpha(hv) = 0 are refused.
  * hv <= Eg(p): refused unless below_gap_energy is given ([K13] uses 5 meV).
"""
from __future__ import annotations

import numpy as np

from . import bands
from .assumptions import DEFAULT
from .optics import as_absorption_model
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


SPIN_RULE_ALIASES = {"per_band": "per_band", "chubenko_prose": "chubenko_prose",
                     "chubenko_text": "chubenko_prose"}


def band_weights(sample, hw, rule):
    """(weights over hh/lh/so, probability of s = +1 for each band) for the chosen rule."""
    P, K, esp = dp71_weights(sample, hw)
    rule = SPIN_RULE_ALIASES[rule]
    if rule == "per_band":
        return K, (1 + P) / 2
    f_hh = (1 + esp) / 2
    rest = K[LH] + K[SO]
    w = np.array([f_hh, (1 - f_hh) * K[LH] / rest, (1 - f_hh) * K[SO] / rest])
    return w, np.array([1.0, 0.0, 0.0])


def photoexcite(sample, hw, n, rng, absorption=None, assumptions=DEFAULT, spin_rule=None,
                broadening=True, below_gap_energy=None):
    """Generate n photoexcited electrons in the Gamma valley at t = 0.

    Parameters
    ----------
    sample : material.Sample
    hw : photon energy [J]
    n : number of electrons
    rng : numpy.random.Generator
    absorption : None / "adachi1989" (default model), an object with absorption_coefficient(hv)
                 [1/m], a scalar absorption length [m], or a callable hv -> length [m]
    assumptions : ModelAssumptions (initial_spin_rule, initial_k_direction, absorption_model)
    spin_rule : overrides assumptions.initial_spin_rule
    broadening : apply Eq. 11
    below_gap_energy : for hv <= Eg(p), place electrons at this energy [J]

    Returns
    -------
    Ensemble with z = z0, valley = Gamma, and band labels in ``ens.band``.
    """
    assumptions.validate()
    rule = SPIN_RULE_ALIASES[spin_rule or assumptions.initial_spin_rule]
    model = as_absorption_model(absorption if absorption is not None else assumptions.absorption_model)
    mat = sample.material
    gv = mat.gamma
    alpha_abs = float(np.squeeze(model.absorption_coefficient(hw)))
    if not alpha_abs > 0:
        raise ValueError(f"absorption model gives alpha = {alpha_abs} at hv = {hw / 1.602176634e-19:.4f} eV "
                         "(e.g. Adachi 1989 is zero below its E0 = 1.42 eV; gap narrowing is not applied)")
    z0 = -np.log1p(-rng.random(n)) / alpha_abs                              # Eq. 7

    if hw <= sample.Eg:
        if below_gap_energy is None:
            raise ValueError("photon energy below Eg(p); pass below_gap_energy to override")
        E0 = np.full(n, float(below_gap_energy))
        band = np.full(n, HH, dtype=np.int8)
        spin = np.where(rng.random(n) < 0.75, 1, -1).astype(np.int8)       # 50% (Eq. 4)
    else:
        w, p_up = band_weights(sample, hw, rule)
        dE = np.array([excess_energy(sample, hw, b) for b in (HH, LH, SO)])
        band = rng.choice(3, size=n, p=w).astype(np.int8)
        spin = np.where(rng.random(n) < p_up[band], 1, -1).astype(np.int8)
        if np.any(dE[band] <= 0):
            raise RuntimeError("selected an energetically inaccessible band")
        E0 = dE[band].copy()
        if broadening:                                                     # Eq. 11
            sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
            E_try = E0 + sign * 1.5 * sample.kT * np.log1p(-rng.random(n))
            E0 = np.where(E_try > 0, E_try, E0)

    if assumptions.initial_k_direction != "isotropic":
        raise ValueError(assumptions.initial_k_direction)
    k = bands.random_unit_vectors(n, rng) * bands.k_of_E(E0, gv.m_eff, gv.alpha)[:, None]
    ens = Ensemble.create(z=z0, k=k, E=E0, spin=spin, valley=0)
    ens.band[:] = band
    return ens
