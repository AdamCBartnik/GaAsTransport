"""Screened polar-optical-phonon (Froehlich) scattering, absorption or emission.

Rate, [C21] Eq. 25 (upper sign: absorption, lower: emission):

    W_pop(k) = e^2 (hbar w0) / (8 pi hbar^2 eps_p) * sqrt(m*) (1 + 2 alpha E') / sqrt(2 gamma(E))
               * (N0 + 1/2 -+ 1/2)
               * [ beta^2/(beta^2 + qmax^2) - beta^2/(beta^2 + qmin^2) + ln((beta^2+qmax^2)/(beta^2+qmin^2)) ]

    E' = E +- hbar w0,  eps_p = eps0 (1/eps_inf - 1/eps_s)^-1,  N0 = Bose(hbar w0)    (Eq. 26)
    qmin = sqrt(2 m gamma(E))/hbar * | sqrt(gamma(E')/gamma(E)) - 1 |                (Eq. 27)
    qmax = sqrt(2 m gamma(E))/hbar * ( sqrt(gamma(E')/gamma(E)) + 1 )

This is the golden-rule rate for |M_q|^2 = e^2 hbar w0 / (2 eps_p Omega) * q^2/(q^2+beta^2)^2 * N:
writing u = q^2, the square bracket equals  J = Int_{qmin^2}^{qmax^2} u/(u+beta^2)^2 du.

Momentum relaxation rate (Eq. 22). With 1 - (k'/k) cos(theta) = (k^2 - k'^2 + u)/(2 k^2), the
same integral gives the closed form used here:

    1/tau_m = C0/(2 k^2) * [ (k^2 - k'^2) I1(u) + I2(u) ]_{qmin^2}^{qmax^2}
    I1(u) = ln(u + a) + a/(u + a),   I2(u) = u - 2 a ln(u + a) - a^2/(u + a),   a = beta^2

where W = C0 * J. [C21] Eq. 31 gives an equivalent-looking expression with hard-to-audit
sign conventions (ambiguity A7). It is transcribed in ``tau_m_eq31`` for comparison only.

Final angle, [C21] Eq. 30 (default; this is the *unscreened* Froehlich distribution, ambiguity A6):
    cos(theta) = [(1 + xi) - (1 + 2 xi)^r] / xi,   xi = 2 sqrt(gamma gamma') / (sqrt(gamma) - sqrt(gamma'))^2
Option ``angle="screened"`` samples the distribution consistent with the screened matrix
element exactly, by rejection from Eq. 30 with acceptance (u/(u+beta^2))^2.
"""
from __future__ import annotations

import numpy as np

from ..backend import asarray, xp_of
from ..constants import HBAR, Q_E
from .base import Mechanism, bose, new_k_from_angle


class PolarOptical(Mechanism):
    spin_class = "pop"

    def __init__(self, sample, emission: bool, valley_from=0, angle="chubenko_eq30", screened=True):
        super().__init__(sample, valley_from)
        if angle not in ("chubenko_eq30", "screened"):
            raise ValueError(angle)
        self.emission = emission
        self.angle = angle
        self.name = f"pop_{'em' if emission else 'abs'}[{self.valley.name}]"
        mat = self.material
        self.hw = mat.hw0
        self.threshold = self.hw if emission else 0.0
        self.N0 = bose(self.hw, sample.kT)
        self.occupation = self.N0 + (1.0 if emission else 0.0)      # (N0 + 1/2 -+ 1/2)
        self.a = sample.beta**2 if screened else 0.0                  # beta^2
        # e^2 (hbar w0) / (8 pi hbar^2 eps_p) * sqrt(m) * occupation
        self._C = Q_E**2 * self.hw / (8 * np.pi * HBAR**2 * mat.eps_p) * np.sqrt(self.m) * self.occupation

    # ---- kinematics ---------------------------------------------------------------------
    def final_energy(self, E):
        return E - self.hw if self.emission else E + self.hw

    def allowed(self, E):
        return E > self.hw if self.emission else np.ones_like(E, dtype=bool)

    def _k_kp(self, E):
        """|k| and |k'| from Eq. 1 (nonparabolic). Requires allowed(E)."""
        Ep = self.final_energy(E)
        k = np.sqrt(2 * self.m * self.gamma(E)) / HBAR
        kp = np.sqrt(2 * self.m * self.gamma(Ep)) / HBAR
        return k, kp, Ep

    def _C0(self, E, Ep):
        """C0(E) such that W = C0 * J (Eq. 25 prefactor)."""
        return self._C * (1 + 2 * self.alpha * Ep) / np.sqrt(2 * self.gamma(E))

    # ---- rates --------------------------------------------------------------------------
    def rate(self, E):
        """Eq. 25."""
        E = np.atleast_1d(np.asarray(E, dtype=float))
        W = np.zeros_like(E)
        ok = self.allowed(E) & (E > 0)
        k, kp, Ep = self._k_kp(E[ok])
        qmin2, qmax2 = (kp - k) ** 2, (kp + k) ** 2                  # Eq. 27
        a = self.a
        J = np.log((a + qmax2) / (a + qmin2)) + a / (a + qmax2) - a / (a + qmin2)
        W[ok] = self._C0(E[ok], Ep) * J
        return W

    def momentum_rate(self, E):
        """Eq. 22 evaluated in closed form for the Eq. 25 matrix element (see module docstring)."""
        E = np.atleast_1d(np.asarray(E, dtype=float))
        R = np.zeros_like(E)
        ok = self.allowed(E) & (E > 0)
        k, kp, Ep = self._k_kp(E[ok])
        qmin2, qmax2 = (kp - k) ** 2, (kp + k) ** 2
        a = self.a
        # [I1]_{qmin^2}^{qmax^2} and [I2]_{qmin^2}^{qmax^2}; the logs combine into a ratio
        L = np.log((qmax2 + a) / (qmin2 + a))
        dI1 = L + a / (qmax2 + a) - a / (qmin2 + a)
        dI2 = (qmax2 - qmin2) - 2 * a * L - (a**2 / (qmax2 + a) - a**2 / (qmin2 + a))
        R[ok] = self._C0(E[ok], Ep) / (2 * k**2) * ((k**2 - kp**2) * dI1 + dI2)
        return R

    def tau_m_eq31(self, E):
        """Literal transcription of [C21] Eq. 31 (+ Eq. 32), for validation only (ambiguity A7).

        1/tau_m = e^2 hbar w0 / (16 pi hbar^2 eps_p) sqrt(m)(1+2 alpha E')/sqrt(2 gamma) (N0+1/2-+1/2)
                  * [ +- N1 / (E (qmin^2+beta^2)) -+ N2 / (E (qmax^2+beta^2)) ]
        N1 = hw beta^2 -+ E E_beta (qmin^4/beta^2 + 2 qmin^2)/gamma + (qmin^2+beta^2)(hw +- 2 E E_beta/gamma) ln|qmin^2+beta^2|
        N2 = same with qmax;   E_beta = hbar^2 beta^2 / (2 m)   (Eq. 32)
        Upper sign = absorption. The ln of a dimensional argument cancels between N1 and N2.
        """
        E = np.atleast_1d(np.asarray(E, dtype=float))
        R = np.zeros_like(E)
        ok = self.allowed(E) & (E > 0)
        Eo = E[ok]
        k, kp, Ep = self._k_kp(Eo)
        qmin2, qmax2 = (kp - k) ** 2, (kp + k) ** 2
        b2 = self.a
        Eb = HBAR**2 * b2 / (2 * self.m)
        g = self.gamma(Eo)
        s = -1.0 if self.emission else 1.0          # "upper sign" -> s = +1
        hw = self.hw

        def N(q2):
            return (hw * b2 - s * Eo * Eb * (q2**2 / b2 + 2 * q2) / g
                    + (q2 + b2) * (hw + s * 2 * Eo * Eb / g) * np.log(q2 + b2))

        bracket = s * N(qmin2) / (Eo * (qmin2 + b2)) - s * N(qmax2) / (Eo * (qmax2 + b2))
        R[ok] = 0.5 * self._C0(Eo, Ep) * bracket
        return R

    # ---- final state --------------------------------------------------------------------
    def _sample_cos_eq30(self, k, kp, rng):
        """Eq. 30, written with k ~ sqrt(gamma)."""
        xi = 2 * k * kp / (k - kp) ** 2
        r = rng.random(len(k))
        # (1+2 xi)^r computed as exp(r log1p(2 xi)) for accuracy
        return ((1 + xi) - np.exp(r * np.log1p(2 * xi))) / xi

    def scatter(self, k, E, rng):
        E = asarray(E, dtype=float)
        xp = xp_of(E)
        kk, kp, Ep = self._k_kp(E)
        cos_t = self._sample_cos_eq30(kk, kp, rng)
        if self.angle == "screened" and self.a > 0:
            # rejection: proposal density ~ 1/u (Eq. 30), target ~ u/(u+a)^2
            todo = xp.arange(len(E))
            while todo.size:
                u = kk[todo] ** 2 + kp[todo] ** 2 - 2 * kk[todo] * kp[todo] * cos_t[todo]
                acc = rng.random(todo.size) < (u / (u + self.a)) ** 2
                todo = todo[~acc]
                if todo.size:
                    cos_t[todo] = self._sample_cos_eq30(kk[todo], kp[todo], rng)
        cos_t = np.clip(cos_t, -1.0, 1.0)
        k_new = new_k_from_angle(k, Ep, cos_t, rng, self.m, self.alpha)
        return k_new, xp.ones(len(E), bool), xp.full(len(E), self.valley_from)
