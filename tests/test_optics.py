from pathlib import Path

import numpy as np
import pytest

from gaas_mc.constants import NM, ev, per_cm3
from gaas_mc.excitation import photoexcite
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.optics import (Adachi1989GaAs, ConstantAbsorptionLength, TabulatedAbsorption,
                            as_absorption_model)

DATA = Path(__file__).parent / "data" / "refractiveindex_GaAs_Adachi1989.yml"


def load_reference():
    txt = DATA.read_text(encoding="utf-8").split("data: |")[1]
    d = np.array([[float(v) for v in line.split()] for line in txt.strip().splitlines() if line.strip()])
    lam_um, n, k = d.T
    return 1.23984198 / lam_um, n, k


def test_adachi_matches_cc0_tabulation():
    """Independent CC0 tabulation of the same model (refractiveindex.info, 4-digit values)."""
    E, n_ref, k_ref = load_reference()
    n, k = Adachi1989GaAs().nk(E)
    assert np.max(np.abs(n - n_ref) / n_ref) < 5e-4
    assert np.max(np.abs(k - k_ref)) < 1e-3


def test_adachi_absorption_physics():
    m = Adachi1989GaAs()
    # below E0 only the E2 Lorentzian tail remains (model artifact, documented): ~2.5e3 /cm
    import dataclasses
    from gaas_mc.optics import Adachi1989Params
    no_e2 = Adachi1989GaAs(dataclasses.replace(Adachi1989Params(), C=0.0))
    assert no_e2.absorption_coefficient(ev(1.40)) == 0.0
    assert 2e5 < m.absorption_coefficient(ev(1.40)) < 3e5
    hv = ev(np.array([1.45, 1.6, 1.9, 2.5]))
    a = m.absorption_coefficient(hv)
    assert np.all(np.diff(a) > 0)                                    # monotonic in this range
    assert 0.8e-6 < 1 / a[0] < 1.6e-6                                # ~1.2 um at 1.45 eV
    assert 0.25 < m.reflectivity(ev(1.6)) < 0.4


def test_interfaces_and_photoexcite_integration():
    s = Sample(gaas_chubenko2021(), per_cm3(1e18))
    rng = np.random.default_rng(0)
    tab = TabulatedAbsorption(ev(np.array([1.4, 2.0])), np.array([1e6, 1e7]))
    assert np.isclose(tab.absorption_coefficient(ev(1.7)), np.sqrt(1e13))   # log-linear midpoint
    with pytest.raises(ValueError):
        tab.absorption_coefficient(ev(2.5))
    for model, l_expected in ((ConstantAbsorptionLength(300 * NM), 300 * NM),
                              (300 * NM, 300 * NM),
                              (lambda hv: 300 * NM, 300 * NM),
                              (None, 1 / Adachi1989GaAs().absorption_coefficient(ev(1.6)))):
        ens = photoexcite(s, ev(1.6), 100_000, rng, absorption=model)
        assert abs(ens.z.mean() / l_expected - 1) < 0.015
    assert isinstance(as_absorption_model("adachi1989"), Adachi1989GaAs)


def test_no_silent_bandgap_shift():
    """At 1e19, Eg(p) = 1.361 eV, but the Adachi model is only meaningful above its E0 = 1.42 eV:
    refuse rather than silently use the E2-tail artifact or shift the spectrum."""
    s = Sample(gaas_chubenko2021(), per_cm3(1e19))
    with pytest.raises(ValueError, match="validity"):
        photoexcite(s, ev(1.40), 10, np.random.default_rng(0))
    ens = photoexcite(s, ev(1.40), 10, np.random.default_rng(0), allow_optics_extrapolation=True)
    assert len(ens) == 10
