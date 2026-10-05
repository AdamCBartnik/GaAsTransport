"""Physical constants (SI) and unit helpers.

All internal quantities in gaas_mc are SI: J, m, s, kg, m^-3.
Convert at the user boundary with the helpers below, e.g. ``ev(1.423)`` or
``per_cm3(1e19)``.
"""
from scipy import constants as _c

Q_E = _c.e                 # elementary charge [C]
M0 = _c.m_e                # free-electron mass [kg]
HBAR = _c.hbar             # reduced Planck constant [J s]
K_B = _c.k                 # Boltzmann constant [J/K]
EPS0 = _c.epsilon_0        # vacuum permittivity [F/m]

EV = Q_E                   # 1 eV in J
MEV = 1e-3 * EV
ANGSTROM = 1e-10
NM = 1e-9
PS = 1e-12
FS = 1e-15
CM3 = 1e-6                 # 1 cm^3 in m^3


def ev(x):
    """eV -> J"""
    return x * EV


def to_ev(x):
    """J -> eV"""
    return x / EV


def per_cm3(x):
    """cm^-3 -> m^-3"""
    return x / CM3


def to_per_cm3(x):
    """m^-3 -> cm^-3"""
    return x * CM3


def kT(T):
    """Thermal energy k_B T [J]."""
    return K_B * T
