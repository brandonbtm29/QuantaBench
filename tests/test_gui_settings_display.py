"""Settings > Display: energy units and number format reach ``settings.display`` and every GUI place that shows an
energy (conformer table, molecule list, info and thermochemistry text), while the energy window, a science setting
stored in kcal/mol, keeps exactly its value."""

import copy
import os

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    import time

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_display_tab_round_trip():
    from PySide6.QtWidgets import QTabWidget

    from quantabench import settings as sm
    from quantabench.chem.pipeline import BuildConfig
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    s = sm.Settings()
    assert s.build.energy_window_kcal == 6.0
    d = SettingsDialog(s, None)
    tabs = d.findChild(QTabWidget)
    assert "Display" in [tabs.tabText(i) for i in range(tabs.count())]
    assert (d.energy_unit.currentData(), d.relative_unit.currentData()) == ("kJ/mol", "kJ/mol")
    assert d.sig_digits.value() == 0 and d.sig_digits.text() == "Automatic" and d.number_style.currentData() == "auto"
    assert d.display_preview.text() == "E = -4658004.3067 kJ/mol, ΔE = 5.17 kJ/mol"
    # Build defaults' energy window: shown in kJ/mol, still exactly 6.0 kcal/mol when untouched
    assert d.window.suffix() == " kJ/mol" and d.window.value() == pytest.approx(25.10)
    out = d.result_settings()
    assert out.build.energy_window_kcal == 6.0 and out.display == sm.DisplaySettings()

    d.relative_unit.setCurrentIndex(d.relative_unit.findData("hartree"))
    assert d.window.suffix() == " hartree" and d.window.value() == pytest.approx(0.009562)
    assert d.result_settings().build.energy_window_kcal == 6.0
    d.relative_unit.setCurrentIndex(d.relative_unit.findData("kcal/mol"))
    assert d.window.value() == 6.0 and d.window.decimals() == 2          # as before the units existed
    d.relative_unit.setCurrentIndex(d.relative_unit.findData("kJ/mol"))
    d.window.setValue(20.92)                                           # typed: 5 kcal/mol
    d.energy_unit.setCurrentIndex(d.energy_unit.findData("hartree"))
    d.sig_digits.setValue(6)
    d.number_style.setCurrentIndex(d.number_style.findData("scientific"))
    assert d.display_preview.text() == "E = -1.77414e+03 hartree, ΔE = 5.16515e+00 kJ/mol"
    out = d.result_settings()
    assert out.display == sm.DisplaySettings("hartree", "kJ/mol", 6, "scientific")
    assert out.build.energy_window_kcal == pytest.approx(5.0)
    assert BuildConfig.from_settings(out).energy_window_kcal == pytest.approx(5.0)

    again = SettingsDialog(sm.Settings.from_dict(out.to_dict()), None)     # the saved values come back
    assert again.energy_unit.currentData() == "hartree" and again.sig_digits.value() == 6
    assert again.number_style.currentData() == "scientific" and again.window.value() == pytest.approx(20.92)
    for dlg in (d, again):
        dlg.reject()
        dlg.deleteLater()
    app.processEvents()


def test_energy_spin_box_keeps_the_stored_value():
    from quantabench import units
    from quantabench.gui.energy_widgets import EnergySpinBox

    _app()
    box = EnergySpinBox(0.5, 50, units.KCAL, units.KJ)
    seen = []
    box.valueChanged.connect(seen.append)
    assert (box.minimum(), box.maximum()) == (pytest.approx(2.09), pytest.approx(209.2))
    box.set_stored(6.0)
    assert box.value() == pytest.approx(25.10) and box.stored() == 6.0 and seen == [pytest.approx(25.10)]
    for unit, shown in ((units.EV, 0.2602), (units.HARTREE, 0.009562), (units.KCAL, 6.0), (units.KJ, 25.10)):
        box.set_unit(unit)
        assert box.unit() == unit and box.suffix() == f" {unit}" and box.value() == pytest.approx(shown)
        assert box.stored() == 6.0                               # exactly: not 5.99904 from 25.10 kJ/mol
    box.setValue(31.38)
    assert box.stored() == pytest.approx(7.5, abs=1e-3)
    box.set_unit(units.KCAL)
    assert box.value() == pytest.approx(7.5) and box.stored() == pytest.approx(7.5, abs=1e-3)
    box.set_stored(100.0)                                        # clamped to the range, as a plain spin box would
    assert box.value() == 50.0 and box.stored() == 50.0


def _built():
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig, run_molecule

    cfg = BuildConfig(method="mmff", max_conformers=10, refine_top=4, rmsd_threshold=0.1, write_files=False,
                      xtb_properties=False, protonate=False)
    res = run_molecule(from_smiles("CCCCO", id="butanol"), cfg)
    assert res.status == "ok" and len(res.conformers) >= 2
    return res


def _cells(w, col):
    return [w.conf_table.item(r, col).text() for r in range(w.conf_table.rowCount())]


def test_main_window_shows_energies_in_the_display_units(tmp_path, monkeypatch):
    from PySide6.QtCore import QModelIndex, Qt

    from quantabench import settings as sm
    from quantabench import units
    from quantabench.gui import main_window
    from quantabench.gui.molecule_list import COL
    from quantabench.gui.settings_dialog import SettingsDialog

    def conv(x, src, dst, places):         # the expected text: converted, with the precision kept
        return f"{units.convert(x, src, dst):.{places}f}"

    res = _built()
    app = _app()
    w = main_window.MainWindow(sm.Settings(output_dir=str(tmp_path)))
    w.add_result(res)
    assert _wait(app, lambda: w.conf_table.rowCount() == len(res.conformers))
    heads = [w.conf_table.horizontalHeaderItem(i).text() for i in range(w.conf_table.columnCount())]
    assert heads[1] == "ΔE (kJ/mol)" and heads[3] == "E (kJ/mol)"
    assert _cells(w, 1) == [conv(c.rel_kcal, "kcal/mol", "kJ/mol", 2) for c in res.conformers]
    assert _cells(w, 3) == [conv(c.energy_ev, "eV", "kJ/mol", 4) for c in res.conformers]
    assert float(_cells(w, 1)[1]) == pytest.approx(res.conformers[1].rel_kcal * 4.184, abs=0.006)
    m = w.model
    col = COL["energy"]
    best = res.conformers[0].energy_ev
    assert m.headerData(col, Qt.Orientation.Horizontal) == "E (kJ/mol)"
    assert m.data(m.index(0, col, QModelIndex())) == conv(best, "eV", "kJ/mol", 3)
    assert float(m.data(m.index(0, col, QModelIndex()))) == pytest.approx(best * 96.4853, abs=6e-4)
    assert w.w_window.suffix() == " kJ/mol" and w.w_window.stored() == 6.0

    # mixed units; the selected conformer stays selected
    w.conf_table.setCurrentCell(1, 0)
    w.s.display.energy_unit, w.s.display.relative_energy_unit = "hartree", "kcal/mol"
    w.apply_display()
    heads = [w.conf_table.horizontalHeaderItem(i).text() for i in range(w.conf_table.columnCount())]
    assert heads[1] == "ΔE (kcal/mol)" and heads[3] == "E (hartree)" and w.conf_table.currentRow() == 1
    assert _cells(w, 1) == [f"{c.rel_kcal:.2f}" for c in res.conformers]                  # as before units
    assert _cells(w, 3) == [conv(c.energy_ev, "eV", "hartree", 7) for c in res.conformers]
    assert m.headerData(col, Qt.Orientation.Horizontal) == "E (hartree)"
    assert w.w_window.suffix() == " kcal/mol" and w.w_window.value() == 6.0
    assert w.current_config().energy_window_kcal == 6.0 and w.s.build.energy_window_kcal == 6.0

    # the thermochemistry and QM lines: H and G absolute, ZPE and corrections relative, S after the relative unit
    fake = copy.copy(res)
    fake.vibrations = {"method": "mmff", "hessian_method": "finite-difference", "n_modes": 39, "n_imaginary": 0,
                       "lowest_freq_cm1": 85.0,
                       "thermo": {"zpe_kcal": 80.5, "enthalpy_kcal": -150000.0, "h_corr_kcal": 85.25,
                                  "entropy_cal_molK": 80.0, "gibbs_kcal": -150020.0, "g_corr_kcal": 61.4,
                                  "temperature_k": 298.15, "qrrho": True}}
    fake.qm = {"level": "wb97x-d/def2-svp", "program": "psi4", "energy_ev": -6312.12345}
    w._show_info(fake)
    info, props = w.mol_info.text(), w.props_view.text()
    assert "ZPE 80.50 kcal/mol, S 80.0 cal/(mol·K), G − E 61.40 kcal/mol at 298 K" in info
    assert f"E = {conv(-6312.12345, 'eV', 'hartree', 7)} hartree" in info
    assert f"H: {conv(-150000.0, 'kcal/mol', 'hartree', 6)} hartree (H − E 85.250 kcal/mol)" in props
    assert f"G: {conv(-150020.0, 'kcal/mol', 'hartree', 6)} hartree (G − E 61.400 kcal/mol)" in props
    assert "S: 80.000 cal/(mol·K)" in props and "ZPE: 80.500 kcal/mol" in props

    # through the Settings dialog: OK applies the Display tab, the energy window keeps 6.0 kcal/mol exactly
    class Accepting(SettingsDialog):
        def exec(self):
            self.relative_unit.setCurrentIndex(self.relative_unit.findData("kJ/mol"))
            self.energy_unit.setCurrentIndex(self.energy_unit.findData("eV"))
            return 1

    monkeypatch.setattr(main_window, "SettingsDialog", Accepting)
    w.open_settings()
    assert (w.s.display.energy_unit, w.s.display.relative_energy_unit) == ("eV", "kJ/mol")
    assert sm.load().display.energy_unit == "eV"
    heads = [w.conf_table.horizontalHeaderItem(i).text() for i in range(w.conf_table.columnCount())]
    assert heads[1] == "ΔE (kJ/mol)" and heads[3] == "E (eV)"
    assert _cells(w, 3) == [f"{c.energy_ev:.5f}" for c in res.conformers]                 # as before units
    assert m.data(m.index(0, col, QModelIndex())) == f"{res.conformers[0].energy_ev:.4f}"
    w._show_info(fake)
    assert "H: " in w.props_view.text() and "(H − E 356.686 kJ/mol)" in w.props_view.text()
    assert "S: 334.720 J/(mol·K)" in w.props_view.text()
    assert w.s.build.energy_window_kcal == 6.0 and w.w_window.suffix() == " kJ/mol"
    w.close()
    assert _wait(app, lambda: w.closed)
    assert sm.load().build.energy_window_kcal == 6.0
