"""Settings > Surfaces & radii: the radii set, the per-element table (Spartan's Preferences > VDW Radii) and the
SASA probe reach ``settings.surfaces`` and, from there, the build config."""

import os

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _row(d, symbol):
    return next(r for r in range(d.radii_table.rowCount()) if d.radii_table.item(r, 0).text() == symbol)


def test_surfaces_tab_round_trip():
    from PySide6.QtCore import Qt

    from quantabench import settings as sm
    from quantabench.chem import pipeline
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    s = sm.Settings()
    s.surfaces.radii_overrides = {"F": 1.47, "cl": 1.9}
    d = SettingsDialog(s, None)
    t = d.radii_table
    assert t.rowCount() == 111 and d.radii_set.currentData() == "bondi" and d.sasa_probe.value() == 1.4
    c = _row(d, "C")
    assert (t.item(c, 1).text(), t.item(c, 2).text(), t.item(c, 3).text()) == ("6", "1.7000", "1.9200")
    assert t.item(_row(d, "F"), 4).text() == "1.47" and t.item(_row(d, "Cl"), 4).text() == "1.9"
    assert d.radii_overrides() == {"F": 1.47, "Cl": 1.9}
    assert "2 elements changed" in d.radii_status.text()

    # only the last column is editable; bad text is flagged and ignored
    assert not t.item(c, 2).flags() & Qt.ItemFlag.ItemIsEditable and t.item(c, 4).flags() & Qt.ItemFlag.ItemIsEditable
    t.item(c, 4).setText("1.80")
    t.item(_row(d, "N"), 4).setText("abc")
    assert t.item(_row(d, "N"), 4).toolTip().startswith("Not a radius")
    assert d.radii_overrides() == {"C": 1.8, "F": 1.47, "Cl": 1.9}

    # select the F row and reset it; filter finds rows by symbol prefix or Z
    t.clearSelection()
    t.selectRow(_row(d, "F"))
    d._reset_selected_radii()
    assert d.radii_overrides() == {"C": 1.8, "Cl": 1.9}
    d.radii_filter.setText("c")
    shown = {t.item(r, 0).text() for r in range(t.rowCount()) if not t.isRowHidden(r)}
    assert {"C", "Cl", "Ca", "Co"} <= shown and "N" not in shown
    d.radii_filter.setText("8")
    assert [t.item(r, 0).text() for r in range(t.rowCount()) if not t.isRowHidden(r)] == ["O"]

    d.radii_set.setCurrentIndex(d.radii_set.findData("spartan"))
    assert t.horizontalHeaderItem(3).font().bold() and not t.horizontalHeaderItem(2).font().bold()
    d.sasa_probe.setValue(1.0)
    out = d.result_settings()
    assert out.surfaces.radii_set == "spartan" and out.surfaces.sasa_probe_A == 1.0
    assert out.surfaces.radii_overrides == {"C": 1.8, "Cl": 1.9}
    opts = pipeline.surface_options(pipeline.BuildConfig.from_settings(out))
    assert opts.radii_set == "spartan" and opts.probe_A == 1.0 and dict(opts.overrides) == {6: 1.8, 17: 1.9}

    d._reset_all_radii()
    assert d.radii_overrides() == {} and d.result_settings().surfaces.radii_overrides == {}
    d.reject()
    d.deleteLater()
    app.processEvents()
