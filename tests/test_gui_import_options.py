"""File > Open with the Import Options dialog (UI-4), its "Don't ask again", the posted data-field columns of the
molecule list, and Settings > Import."""

import pytest
from conftest import HAS_DISPLAY, PFOA

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    import time

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


def _sdf(path):
    """Two 2D records with data fields; one is an amine hydrochloride."""
    from rdkit import Chem

    w = Chem.SDWriter(str(path))
    for title, smi, props in (("PFOA", PFOA, {"logKow": "4.81", "CAS": "335-67-1"}),
                              ("ethylamine HCl", "CCN.Cl", {"logKow": "-0.13"})):
        m = Chem.MolFromSmiles(smi)
        m.SetProp("_Name", title)
        for k, v in props.items():
            m.SetProp(k, v)
        w.write(m)
    w.close()
    return path


def test_dialog_shows_the_file_and_returns_the_choices(tmp_path):
    from quantabench.chem import inputs
    from quantabench.chem.import_options import ImportOptions, survey
    from quantabench.gui.import_options import ImportOptionsDialog, survey_text

    app = _app()
    items = inputs.read_raw(_sdf(tmp_path / "set.sdf"))
    s = survey(items)
    text = survey_text(s, ["set.sdf"])
    assert "2 structure(s) in set.sdf" in text and "2 drawn in 2D" in text and "2 data field(s) (logKow, CAS)" in text
    assert "1 with several fragments (HCl and other acids 1" in text and "RDKit (SD file)" in text
    d = ImportOptionsDialog(s, ImportOptions(), ["set.sdf"])
    f = d.form
    assert f.convert_2d.isChecked() and f.water.isChecked() and f.read_order.isChecked() and f.as_molecules.isChecked()
    assert [f.id_field.itemText(i) for i in range(f.id_field.count())] == ["(automatic)", "logKow", "CAS"]
    f.keep_coords.setChecked(True)
    f.post_fields.setChecked(True)
    f.id_field.setCurrentText("CAS")
    f.acids.setChecked(False)
    f.largest.setChecked(True)
    f.standard_order.setChecked(True)
    f.as_conformers.setChecked(True)
    o = d.options()
    assert (o.keep_coords, o.post_fields, o.id_field, o.name_field) == (True, True, "CAS", "")
    assert (o.remove_acids, o.keep_largest, o.label_order, o.same_constitution) == (False, True, "standard", "conformers")
    d.form.set_options(ImportOptions())
    assert d.options() == ImportOptions()
    d.deleteLater()
    app.processEvents()


def test_open_asks_once_applies_the_options_and_posts_fields(tmp_path, monkeypatch):
    from PySide6.QtCore import Qt

    from quantabench import settings as sm
    from quantabench.gui import import_options as gui_imp
    from quantabench.gui.main_window import MainWindow
    from quantabench.gui.molecule_list import COLUMNS

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "out")))
    asked = []

    def accept(dlg):
        asked.append(dlg)
        dlg.form.post_fields.setChecked(True)
        dlg.dont_ask.setChecked(True)
        return 1

    monkeypatch.setattr(gui_imp.ImportOptionsDialog, "exec", accept)
    path = _sdf(tmp_path / "set.sdf")
    w.open_files([str(path)])
    assert _wait(app, lambda: w.model.rowCount() == 2)
    assert len(asked) == 1
    rows = {r.id: r for r in w.model.rows()}
    inp = rows["ethylamine_HCl"].input
    assert inp.smiles == "CCN" and inp.mol.GetConformer().Is3D()          # HCl filtered, 2D converted to 3D
    assert w.model.extra_columns() == ["logKow", "CAS"]
    assert w.model.columnCount() == len(COLUMNS) + 2
    col = len(COLUMNS)
    assert w.model.headerData(col, Qt.Orientation.Horizontal) == "logKow"
    pos = w.model.position("PFOA")
    assert w.model.data(w.model.index(pos, col)) == "4.81" and w.model.data(w.model.index(pos, col + 1)) == "335-67-1"
    w.model.sort(col)                                                      # numbers in order
    assert [r.id for r in w.model.rows()] == ["ethylamine_HCl", "PFOA"]
    # the choices were stored, and "Don't ask again" stops the dialog
    saved = sm.load()
    assert saved.import_.ask is False and saved.import_.post_fields is True
    monkeypatch.setattr(gui_imp.ImportOptionsDialog, "exec", lambda dlg: pytest.fail("asked again"))
    w.open_files([str(path)])
    assert _wait(app, lambda: w.model.rowCount() == 4)
    _close(app, w)


def test_cancel_adds_nothing_and_outputs_go_to_the_importer(tmp_path, monkeypatch):
    from quantabench import settings as sm
    from quantabench.gui import import_action
    from quantabench.gui import import_options as gui_imp
    from quantabench.gui.main_window import MainWindow

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "out")))
    monkeypatch.setattr(gui_imp.ImportOptionsDialog, "exec", lambda dlg: 0)
    handed = []
    monkeypatch.setattr(import_action, "import_files", lambda win, paths: handed.append(paths))
    out = tmp_path / "calc.log"
    out.write_text("not read here")
    w.open_files([str(_sdf(tmp_path / "set.sdf")), str(out)])
    assert handed == [[str(out)]]
    assert _wait(app, lambda: not w.statusBar().currentMessage(), 10)
    app.processEvents()
    assert w.model.rowCount() == 0
    _close(app, w)


def test_settings_import_tab_round_trip():
    from quantabench import settings as sm
    from quantabench.chem.import_options import ImportOptions
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    s = sm.Settings()
    s.import_.ask = False
    s.import_.id_field = "CAS_RN"
    d = SettingsDialog(s, None)
    assert not d.import_ask.isChecked() and d.import_form.id_field.currentText() == "CAS_RN"
    d.import_ask.setChecked(True)
    d.import_form.keep_coords.setChecked(True)
    d.import_form.counterions.setChecked(True)
    d.import_form.standard_order.setChecked(True)
    d.import_form.id_field.setCurrentText("(automatic)")
    out = d.result_settings()
    assert out.import_.ask and out.import_.keep_coords and out.import_.strip_counterions
    assert out.import_.label_order == "standard" and out.import_.id_field == ""
    assert ImportOptions.from_settings(out).filters() == ["water", "acids", "counterions"]
    d.reject()
    d.deleteLater()
    app.processEvents()
