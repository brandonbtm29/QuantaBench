"""quantabench.units: energy units and number format for the GUI (Settings > Display), and that the display
settings stay out of runs and exports."""

import dataclasses
import json

import pytest

from quantabench import settings as sm
from quantabench import units as u


def test_factors_are_the_ones_the_computations_use():
    from quantabench.chem import conformers, engines, surfaces, vibrations

    assert u.KCAL_PER_EV == conformers.KCAL_PER_EV == vibrations.KCAL_PER_EV
    assert u.HARTREE_EV == engines.HARTREE_EV
    assert u.factor(u.HARTREE, u.KCAL) == pytest.approx(surfaces.HARTREE_KCAL, rel=1e-6)
    assert u.factor(u.KCAL, u.KJ) == 4.184
    assert u.factor(u.EV, u.KJ) == pytest.approx(96.4853, rel=1e-5)
    assert u.factor(u.HARTREE, u.KJ) == pytest.approx(2625.4996, rel=1e-6)
    for a in u.UNITS:
        for b in u.UNITS:
            assert u.factor(a, b) * u.factor(b, a) == pytest.approx(1.0, rel=1e-12)
            assert u.convert(u.convert(1.2345, a, b), b, a) == pytest.approx(1.2345, rel=1e-12)
    assert u.convert(6.0, u.KCAL, u.KCAL) == 6.0                  # same unit: untouched, not multiplied by 1.0...
    assert u.convert(100.0, u.KCAL, u.KJ, power=2) == pytest.approx(100.0 * 4.184**2)   # a variance, σ²


def test_normalize_spellings_and_bad_values():
    assert [u.normalize(x) for x in ("kJ/mol", "kj", "KJ/MOL", "kcal", "EV", "Eh", "au", "Hartree", " kcal/mol ")] == [
        u.KJ, u.KJ, u.KJ, u.KCAL, u.EV, u.HARTREE, u.HARTREE, u.HARTREE, u.KCAL]
    assert u.normalize("furlongs") == u.normalize(None) == u.normalize(3) == u.DEFAULT_UNIT == "kJ/mol"
    assert u.normalize_style("SCIENTIFIC") == "scientific" and u.normalize_style("pretty") == "auto"
    assert [u.normalize_digits(x) for x in (0, 6, 15, 16, -1, "4", "x", None)] == [0, 6, 15, 0, 0, 4, 0, 0]


def test_resolution_decimals_keep_the_precision_shown_before():
    # 0.01 kcal/mol (the conformer table's ΔE) is 0.042 kJ/mol, 0.00043 eV, 0.0000159 hartree
    assert [u.resolution_decimals(2, u.KCAL, x) for x in u.UNITS] == [2, 2, 4, 5]
    # 0.00001 eV (the conformer table's E)
    assert [u.resolution_decimals(5, u.EV, x) for x in u.UNITS] == [4, 4, 5, 7]
    assert u.resolution_decimals(1, u.KCAL, u.KJ, power=2) == 0        # (kcal/mol)² x 17.5
    assert u.resolution_decimals(0, u.EV, u.KJ) == 0


def test_format_number_styles_and_digits():
    f = u.format_number
    assert f(5.16515, 2) == "5.17" and f(-2.0, 0) == "-2"
    assert f(None) == "" and f(float("nan")) == "nan" and f(float("-inf")) == "-inf"
    assert f(0.0, 2) == "0.00"
    assert f(4.184e-5, 2) == "4.18e-05"                   # auto: a nonzero value never reads as 0.00
    assert f(4.184e-5, 2, style="decimal") == "0.00"
    assert f(-4658004.30673, 4, style="scientific") == "-4.6580043067e+06"   # as many digits as the decimal form
    # significant digits
    assert f(-4658004.30673, 4, digits=6) == "-4.65800e+06"                  # auto: as the g format
    assert f(-4658004.30673, 4, digits=8) == "-4658004.3"
    assert f(-4658004.30673, 4, digits=3, style="decimal") == "-4660000"
    assert f(5.16515, 2, digits=3) == "5.17" and f(5.0, 2, digits=3) == "5.00" and f(12.5, 2, digits=2) == "12"
    assert f(4.184e-5, 2, digits=3, style="decimal") == "0.0000418"
    assert f(4.184e-5, 2, digits=3, style="scientific") == "4.18e-05"
    assert f(0.0, 2, digits=3, style="decimal") == "0.00"


def test_energy_display_defaults_units_and_text():
    d = u.EnergyDisplay.of(sm.Settings())
    assert (d.absolute, d.relative, d.digits, d.style) == ("kJ/mol", "kJ/mol", 0, "auto")
    assert d == u.EnergyDisplay() == u.EnergyDisplay.of(None) == u.EnergyDisplay.of(d)
    assert d.header("ΔE") == "ΔE (kJ/mol)" and d.header("E", "absolute") == "E (kJ/mol)"
    assert d.with_unit(1.2345, u.KCAL) == "5.17 kJ/mol"
    assert d.text(-48276.81234, u.EV, "absolute", 5) == "-4658004.3067"
    assert d.text(None, u.KCAL) == "" and d.with_unit(None, u.KCAL) == ""
    assert d.with_unit(120.0, u.KCAL, power=2, decimals=1) == "2101 (kJ/mol)²"
    assert d.entropy_text(45.1) == "188.7 J/(mol·K)"           # water's S°, 188.8 J/(mol·K)
    with pytest.raises(ValueError):
        d.unit("reaction")

    mixed = u.EnergyDisplay(sm.DisplaySettings(energy_unit="hartree", relative_energy_unit="kcal/mol",
                                               significant_digits=4, number_style="scientific"))
    assert mixed.header("E", "absolute") == "E (hartree)" and mixed.header("ΔE") == "ΔE (kcal/mol)"
    assert mixed.text(-48276.81234, u.EV, "absolute", 5) == "-1.774e+03"
    assert mixed.with_unit(1.2346, u.KCAL) == "1.235e+00 kcal/mol"
    assert mixed.entropy_text(45.1) == "4.510e+01 cal/(mol·K)"
    ev = u.EnergyDisplay(sm.DisplaySettings("eV", "eV"))
    assert ev.label(power=2) == "eV²" and ev.entropy_text(45.1) == "1.956 meV/K"
    assert u.EnergyDisplay(sm.DisplaySettings("hartree", "hartree")).entropy_text(45.1) == "0.0719 mEh/K"

    # a hand-edited settings.json: bad values fall back to the defaults
    bad = u.EnergyDisplay(sm.DisplaySettings(energy_unit="Btu", relative_energy_unit="KCAL", significant_digits=99,
                                             number_style="roman"))
    assert bad.key() == ("kJ/mol", "kcal/mol", 0, "auto")


def test_display_settings_round_trip_and_stay_out_of_runs(isolated_settings):
    from quantabench.chem import jobs
    from quantabench.chem.pipeline import BuildConfig

    s = sm.Settings()
    assert dataclasses.asdict(s.display) == {"energy_unit": "kJ/mol", "relative_energy_unit": "kJ/mol",
                                             "significant_digits": 0, "number_style": "auto"}
    s.display.energy_unit, s.display.relative_energy_unit = "hartree", "kcal/mol"
    s.display.significant_digits, s.display.number_style = 6, "decimal"
    path = sm.save(s)
    assert json.loads(path.read_text())["display"]["energy_unit"] == "hartree"
    back = sm.load()
    assert back.display == s.display
    # old settings files get the defaults; wrong types are ignored
    assert sm.Settings.from_dict({"build": {"ph": 6}}).display == sm.DisplaySettings()
    odd = sm.Settings.from_dict({"display": {"energy_unit": 4, "significant_digits": "6", "number_style": "decimal"}})
    assert (odd.display.energy_unit, odd.display.significant_digits, odd.display.number_style) == (
        "kJ/mol", 0, "decimal")
    # display only: not in the build config, not a science setting
    names = {f.name for f in dataclasses.fields(BuildConfig)}
    for field in dataclasses.fields(sm.DisplaySettings):
        assert field.name not in names and field.name not in jobs.SCIENCE_FIELDS
    assert BuildConfig.from_settings(back) == BuildConfig.from_settings(sm.Settings.from_dict(
        {k: v for k, v in back.to_dict().items() if k != "display"}))
