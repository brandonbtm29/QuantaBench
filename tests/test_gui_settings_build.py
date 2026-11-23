"""Settings > Build defaults (SETTINGS-1): Spartan's Import Filters as build filters (water, acids, named counter-ions
stripped before the largest-fragment rule) and its Conformer Rules (ring conformers) reach ``settings.build`` and,
from there, the build config: science settings, stored with each run."""

import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY,
                                reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _item(d, name):
    return next(d.strip_list.item(i) for i in range(d.strip_list.count()) if d.strip_list.item(i).text() == name)


def test_build_defaults_tab_round_trip():
    from PySide6.QtCore import Qt

    from quantabench import settings as sm
    from quantabench.chem import jobs, pipeline, standardize
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    d = SettingsDialog(sm.Settings(), None)
    try:
        assert not d.strip_water.isChecked() and not d.strip_acids.isChecked() and d.desalt.isChecked()
        assert d.strip_list.count() == len(standardize.NAMED_COUNTERIONS) and d.strip_counterions() == ""
        assert d.ring_conformers.currentData() == "normal" and d.ring_conformers.count() == 2
        assert d.twist_boat_seeds.value() == 20 and not d.twist_boat_seeds.isEnabled()     # only with Thorough
        assert d.strip_summary.text() == "Nothing stripped first; then the largest fragment is kept."
        assert "Science settings" in d.strip_note.text() and "before the largest-fragment" in d.strip_note.text()
        assert _item(d, "sodium").toolTip().startswith("[Na+]")

        d.strip_water.setChecked(True)
        _item(d, "sodium").setCheckState(Qt.CheckState.Checked)
        _item(d, "potassium").setCheckState(Qt.CheckState.Checked)
        d.desalt.setChecked(False)
        d.ring_conformers.setCurrentIndex(d.ring_conformers.findData("thorough"))
        assert d.twist_boat_seeds.isEnabled()
        d.twist_boat_seeds.setValue(12)
        assert d.strip_counterions() == "potassium,sodium"
        assert d.strip_summary.text() == ("Stripped first: water, potassium, sodium; then the remaining fragments are "
                                          "computed together.")
        out = d.result_settings()
    finally:
        d.reject()
        d.deleteLater()
        app.processEvents()
    b = out.build
    assert (b.strip_water, b.strip_acids, b.strip_counterions, b.keep_largest_fragment, b.ring_conformers) == \
        (True, False, "potassium,sodium", False, "thorough")
    assert b.twist_boat_seeds == 12 and pipeline.BuildConfig.from_settings(out).twist_boat_seeds == 12
    sm.save(out)

    s = sm.load()
    cfg = pipeline.BuildConfig.from_settings(s)
    assert (cfg.strip_water, cfg.strip_acids, cfg.strip_counterions, cfg.keep_largest_fragment, cfg.ring_conformers) \
        == (True, False, "potassium,sodium", False, "thorough")
    assert {"strip_water", "strip_counterions", "ring_conformers"} <= set(jobs.SCIENCE_FIELDS)
    d = SettingsDialog(s, None)                                  # the saved values come back
    try:
        assert d.strip_water.isChecked() and not d.desalt.isChecked()
        checked = [d.strip_list.item(i).text() for i in range(d.strip_list.count())
                   if d.strip_list.item(i).checkState() == Qt.CheckState.Checked]
        assert checked == ["sodium", "potassium"]                # in the table's order
        assert d.ring_conformers.currentData() == "thorough"
        _item(d, "sodium").setCheckState(Qt.CheckState.Unchecked)
        d.strip_acids.setChecked(True)
        assert d.result_settings().build.strip_counterions == "potassium" and d.result_settings().build.strip_acids
    finally:
        d.reject()
        d.deleteLater()
        app.processEvents()


def test_counterion_entries_checkbox_and_rows(tmp_path):
    """Build defaults' *Keep the counter-ions as separate entries* reaches the build config, and a counter-ion entry
    a run adds gets a row of its own in the molecule list."""
    import time

    from quantabench import settings as sm
    from quantabench.chem import pipeline
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.main_window import MainWindow
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    d = SettingsDialog(sm.Settings(), None)
    try:
        assert not d.counterion_entries.isChecked()
        d.counterion_entries.setChecked(True)
        assert d.strip_summary.text().endswith("The counter-ions removed become entries of their own.")
        out = d.result_settings()
    finally:
        d.reject()
        d.deleteLater()
        app.processEvents()
    assert out.build.counterion_entries and pipeline.BuildConfig.from_settings(out).counterion_entries

    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "runs")))
    try:
        w.add_inputs([from_smiles("[K+].[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", id="kpfbs")])
        ion = from_smiles("[K+]", id="kpfbs_K+", name="potassium")
        ion.meta["qb_counterion_of"] = ["kpfbs"]
        w._on_molecule_done(pipeline.MoleculeResult(input=ion, status="ok", smiles="[K+]", method="mmff"))
        row = w.model.row("kpfbs_K+")
        assert row is not None and row.name == "potassium" and row.status == "ok"
        assert w.model.ids() == ["kpfbs", "kpfbs_K+"]
        stray = pipeline.MoleculeResult(input=from_smiles("C", id="not-in-the-list"), status="ok")
        w._on_molecule_done(stray)                       # any other unknown id is still ignored
        assert w.model.row("not-in-the-list") is None
    finally:
        w.close()
        t = time.time()
        while not w.closed and time.time() - t < 30:
            app.processEvents()
            time.sleep(0.02)
