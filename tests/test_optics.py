import warnings
from pathlib import Path

import numpy as np
import pytest

from gaas_mc.constants import EV, NM, ev, per_cm3
from gaas_mc.excitation import photoexcite
from gaas_mc.material import Sample, gaas_chubenko2021
from gaas_mc.optical_data import DATA as CASEY_CSV
from gaas_mc.optical_data import casey1975_ptype
from gaas_mc.optics import (Adachi1989GaAs, CaseyAdachiAbsorption, ConstantAbsorptionLength,
                            DopingTabulatedAbsorption, TabulatedAbsorption, as_absorption_model)

DATA = Path(__file__).parent / "data" / "refractiveindex_GaAs_Adachi1989.yml"
ROOT = Path(__file__).resolve().parents[1]
MAT = gaas_chubenko2021()


# ------------------------------------------------------------------------------ Adachi ----

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


def test_adachi_sub_E0_tail_is_not_interband_absorption():
    with pytest.raises(ValueError, match="E2-oscillator"):
        Adachi1989GaAs().absorption_coefficient(ev(1.40))
    tail = Adachi1989GaAs(below_E0="model")
    import dataclasses
    from gaas_mc.optics import Adachi1989Params
    no_e2 = Adachi1989GaAs(dataclasses.replace(Adachi1989Params(), C=0.0), below_E0="model")
    assert no_e2.absorption_coefficient(ev(1.40)) == 0.0              # E0 term alone: no absorption
    assert 2e5 < tail.absorption_coefficient(ev(1.40)) < 3e5          # the documented artifact
    a = Adachi1989GaAs().absorption_coefficient(ev(np.array([1.45, 1.6, 1.9, 2.5])))
    assert np.all(np.diff(a) > 0)
    assert 0.25 < Adachi1989GaAs().reflectivity(ev(1.6)) < 0.4


# ------------------------------------------------------------- doping-dependent tables ----

def synthetic():
    hv = ev(np.linspace(1.30, 1.60, 31))
    return [dict(p=per_cm3(1e17), hv=hv, alpha=np.full(31, 1e4), label="a"),
            dict(p=per_cm3(1e19), hv=hv, alpha=np.full(31, 1e6), label="b")]


def test_doping_interpolation_is_linear_in_log10_p():
    ds = synthetic()
    assert np.isclose(DopingTabulatedAbsorption(ds, per_cm3(1e17)).absorption_coefficient(ev(1.45)), 1e4)
    assert np.isclose(DopingTabulatedAbsorption(ds, per_cm3(1e18)).absorption_coefficient(ev(1.45)), 1e5)
    assert np.isclose(DopingTabulatedAbsorption(ds, per_cm3(1e19)).absorption_coefficient(ev(1.45)), 1e6)


def test_doping_out_of_range_clamps_with_warning_or_raises():
    ds = synthetic()
    with pytest.warns(UserWarning, match="outside the measured range"):
        m = DopingTabulatedAbsorption(ds, per_cm3(5e19))
    assert np.isclose(m.absorption_coefficient(ev(1.45)), 1e6)
    with pytest.raises(ValueError):
        DopingTabulatedAbsorption(ds, per_cm3(5e19), out_of_range="raise")
    with pytest.raises(ValueError):
        DopingTabulatedAbsorption(ds, per_cm3(1e18)).absorption_coefficient(ev(1.65))


# ------------------------------------------------------------------ Casey 1975 data ----

def test_casey_dataset_provenance_and_coverage():
    head = [line for line in CASEY_CSV.read_text().splitlines() if line.startswith("#")]
    assert any("10.1063/1.321330" in line for line in head)
    assert any("digitize_casey1975.py" in line for line in head)
    ds = casey1975_ptype()
    ps = sorted(d["p"] * 1e-6 for d in ds)
    assert np.allclose(ps, [1.6e16, 2.2e17, 4.9e17, 1.2e18, 2.4e18, 1.6e19])
    for d in ds:
        assert d["hv"][-1] / EV > 1.59 and d["hv"][0] / EV < 1.38
        assert np.all(d["alpha"] > 0)


def test_casey_physics_statements():
    """Statements in Casey et al. (1975) Sec. IV B that the digitized data must reproduce:
    (1) below Eg, alpha at a fixed energy increases with p; (2) alpha ~ 1e2 cm^-1 near 1.35 eV
    for p = 1.6e19; (3) all p-type samples approach a similar alpha at 1.6 eV."""
    ds = sorted(casey1975_ptype(), key=lambda d: d["p"])
    a137 = [np.interp(ev(1.375), d["hv"], d["alpha"]) for d in ds]
    assert np.all(np.diff(a137) > 0)
    d19 = ds[-1]
    assert 80e2 < np.interp(ev(1.35), d19["hv"], d19["alpha"]) < 160e2       # 1/m (0.8-1.6e2 /cm)
    a159 = np.array([np.interp(ev(1.59), d["hv"], d["alpha"]) for d in ds])
    assert a159.max() / a159.min() < 1.3


@pytest.mark.parametrize("p", [1.5e17, 1e18, 1e19])
def test_composite_is_continuous_and_positive(p):
    m = CaseyAdachiAbsorption(per_cm3(p))
    hv = ev(np.arange(1.40, 1.75, 0.0001))
    a = m.absorption_coefficient(hv)
    assert np.all(np.isfinite(a)) and np.all(a > 0)
    d = 1e-7 * EV                                              # no jump at either blend boundary
    for b in (m.b0, m.b1):
        assert abs(np.log(m.absorption_coefficient(b + d) / m.absorption_coefficient(b - d))) < 1e-4
    hb = np.linspace(m.b0 - 0.005 * EV, m.b1 + 0.005 * EV, 2001)   # smooth through the window
    assert np.max(np.abs(np.diff(np.log(m.absorption_coefficient(hb))))) < 0.002
    assert 0.8 < m.seam_ratio[0] < 1.5 and 0.8 < m.seam_ratio[1] < 1.5
    assert np.isclose(m.absorption_coefficient(ev(1.7)), Adachi1989GaAs().absorption_coefficient(ev(1.7)))


def test_window_between_doped_and_intrinsic_gap_is_allowed():
    """At 1e19, Eg(p) = 1.361 eV; photons at 1.38-1.41 eV are absorbed according to the measured
    p-type spectrum. No hard 1.42 eV optical cutoff."""
    s = Sample(MAT, per_cm3(1e19))
    rng = np.random.default_rng(0)
    for hv in (1.38, 1.40, 1.41):
        ens = photoexcite(s, ev(hv), 20_000, rng)
        l_model = 1 / CaseyAdachiAbsorption(s.p).absorption_coefficient(ev(hv))
        assert abs(ens.z.mean() / l_model - 1) < 0.03
    with pytest.raises(ValueError, match="E2-oscillator"):
        photoexcite(s, ev(1.40), 10, rng, absorption="adachi1989")       # comparison model refuses
    with pytest.raises(ValueError, match="Eg"):
        photoexcite(s, ev(1.35), 10, rng)                                # energetics not satisfied


def test_interfaces_and_photoexcite_integration():
    s = Sample(MAT, per_cm3(1e18))
    rng = np.random.default_rng(1)
    tab = TabulatedAbsorption(ev(np.array([1.4, 2.0])), np.array([1e6, 1e7]))
    assert np.isclose(tab.absorption_coefficient(ev(1.7)), np.sqrt(1e13))
    with pytest.raises(ValueError):
        tab.absorption_coefficient(ev(2.5))
    for model, l_expected in ((ConstantAbsorptionLength(300 * NM), 300 * NM),
                              (300 * NM, 300 * NM),
                              (lambda hv: 300 * NM, 300 * NM),
                              (None, 1 / CaseyAdachiAbsorption(s.p).absorption_coefficient(ev(1.6)))):
        ens = photoexcite(s, ev(1.6), 100_000, rng, absorption=model)
        assert abs(ens.z.mean() / l_expected - 1) < 0.015
    assert isinstance(as_absorption_model("adachi1989"), Adachi1989GaAs)
    assert isinstance(as_absorption_model("casey1975+adachi1989", s), CaseyAdachiAbsorption)


@pytest.mark.skipif(not (ROOT / "refs" / "Casey1975_JAP46_250.pdf").exists(),
                    reason="publisher PDF not present (refs/ is not committed)")
def test_digitization_is_reproducible(tmp_path):
    """Re-running the digitizer from the PDF reproduces the committed data points exactly."""
    import subprocess
    import sys
    before = CASEY_CSV.read_text().splitlines()
    subprocess.run([sys.executable, str(ROOT / "tools" / "digitize_casey1975.py")], check=True,
                   capture_output=True, cwd=ROOT)
    after = CASEY_CSV.read_text().splitlines()
    assert [l for l in before if not l.startswith("# Produced")] == \
        [l for l in after if not l.startswith("# Produced")]
