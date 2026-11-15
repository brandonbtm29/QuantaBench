"""UI-8: collapsible sections and pop-outs. The Pipeline dock's stages as an accordion with one-line summaries and ⓘ
pop-outs, the Molecules dock's folded details and notes, the Table toolbar's drop-down menus, and panels that pop
out into windows of their own."""

import time

import pytest
from conftest import HAS_DISPLAY

pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=10.0):
    from PySide6.QtCore import QEvent

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        time.sleep(0.02)
    return cond()


def _window(tmp_path, workspace="calculate"):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "runs")))
    w.resize(1500, 920)
    w.show()
    w.set_workspace(workspace)
    _wait(_app(), lambda: False, 0.3)
    return w


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


def test_collapsible_pieces():
    from PySide6.QtWidgets import QLabel, QWidget

    from quantabench.gui import collapsible as cl

    app = _app()
    acc = cl.Accordion(exclusive=True)
    a = acc.add(cl.Section("A & B", QLabel("body a"), key="a", info="what A does"))
    b = acc.add(cl.Section("B", QLabel("body b"), key="b"))
    assert a.title() == "A & B" and a.toggle.text() == "A && B"        # no mnemonic underline
    acc.show()
    acc.open("a")
    assert a.is_open() and a.body.isVisible() and not a.summary.isVisible()
    acc.open("b")
    assert b.is_open() and not a.is_open() and acc.open_key() == "b"     # one at a time
    a.set_summary("x · y")
    assert a.summary.text() == "x · y" and a.summary.isVisible()
    assert a.info_button.info_text() == "what A does" and a.info_button.text() == "ⓘ"
    # an InfoButton follows its InfoLabel, warns when it says something is missing, forwards links
    src = cl.InfoLabel()
    hits = []
    src.linkActivated.connect(hits.append)
    row = cl.with_info(QWidget(), src)
    btn = row.info_button
    src.setText("<b>GFN2</b>: fine. <a href='guide:gfn2'>More…</a>")
    assert btn.text() == "ⓘ" and not src.isVisible()
    src.setText("<b>GFN2</b>: Not available: tblite is not installed")
    assert btn.text() == "⚠" and "#d35400" in btn.styleSheet()
    popup = btn.show_popup()
    assert "Not available" in popup.label.text()
    popup.label.linkActivated.emit("guide:gfn2")
    assert hits == ["guide:gfn2"] and not popup.isVisible()
    notes = cl.Notes()
    notes.show()
    notes.set_notes([])
    assert not notes.isVisible()
    notes.set_notes(["rank auto → refine: AIMNet2 is not available", "g-xTB not found"])
    assert notes.isVisible() and notes.toggle.text() == "⚠ 2 notes" and not notes.label.isVisible()
    notes.toggle.setChecked(True)
    assert notes.label.isVisible() and "g-xTB not found" in notes.label.text()
    acc.close()
    notes.close()
    app.processEvents()


def test_pipeline_dock_is_an_accordion_with_summaries(tmp_path):
    from quantabench.gui.collapsible import InfoButton, Section
    from quantabench.gui.main_window import STAGES

    app = _app()
    w = _window(tmp_path)
    try:
        assert list(w.stage_items) == [k for k, *_ in STAGES]
        assert all(isinstance(s, Section) for s in w.stage_items.values())
        acc = w.stage_accordion
        assert acc.open_key() == "refine"                                # the default
        assert w.stage_items["refine"].body.isVisible() and not w.stage_items["qm"].body.isVisible()
        # the summaries follow the fields
        w.w_refine_top.setValue(7)
        assert "lowest 7" in w.stage_items["refine"].summary.text()
        w.w_qm_on.setChecked(False)
        assert w.stage_items["qm"].summary.text() == "off"
        w.w_qm_on.setChecked(True)
        assert "/" in w.stage_items["qm"].summary.text()
        assert w.stage_summary("input") == "0 molecule(s)"
        # the method guide's notes are behind ⓘ next to their field, the stage description behind the header's ⓘ
        btn = w.engine_info.parentWidget().info_button
        assert isinstance(btn, InfoButton) and btn.info_text() == w.engine_info.text() and not w.engine_info.isVisible()
        assert w.stage_items["refine"].info_button.info_text().startswith("Re-optimize the lowest conformers")
        assert "sdf" in w.stage_items["export"].info_button.info_text().lower()
        # status per stage for the current molecule
        w._update_stage_status(None)
        assert w.stage_items["refine"].status.text() == ""
        acc.open("qm")
        assert acc.open_key() == "qm" and not w.stage_items["refine"].is_open()
    finally:
        _close(app, w)
    w = _window(tmp_path)                                              # the open stage is remembered
    try:
        assert w.stage_accordion.open_key() == "qm"
    finally:
        _close(app, w)


def test_molecule_details_fold_and_notes(tmp_path):
    from quantabench.chem import pipeline
    from quantabench.chem.inputs import from_smiles

    app = _app()
    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=2, xtb_properties=False, protonate=False,
                               write_files=False, threads=1)
    res = pipeline.run_molecule(from_smiles("CCO", id="ethanol"), cfg)
    res.notes.append("rank auto → refine: AIMNet2 is not available (a long engine note)")
    w = _window(tmp_path, workspace="build")
    try:
        w.add_inputs([res.input], built=[res])
        w._show_info(res)
        assert "ethanol" in w.mol_info.text() and "a long engine note" not in w.mol_info.text()
        n = len(res.notes)                                              # the run's own notes and the one above
        assert w.mol_notes.toggle.text() == f"⚠ {n} note{'s' if n != 1 else ''}" and not w.mol_notes.label.isVisible()
        assert "a long engine note" in w.mol_notes.label.text()
        d = w.mol_details
        assert d.is_open()
        d.set_open(False)
        assert not w.mol_info.isVisible() and d.summary.text().startswith("ethanol · C2H6O")
        w._clear_notes()
        assert not w.mol_notes.isVisible()
    finally:
        _close(app, w)
    w = _window(tmp_path, workspace="build")
    try:
        assert not w.mol_details.is_open()                             # remembered closed
        w.mol_details.set_open(True)
    finally:
        _close(app, w)


def test_table_toolbar_menus_and_popout(tmp_path):
    from quantabench.gui import popout

    pytest.importorskip("pyqtgraph")
    app = _app()
    w = _window(tmp_path, workspace="analyze")
    try:
        p = w.table_panel
        names = [a.text() for a in p.columns_btn.menu().actions() if a.text()]
        assert names == ["Show or hide columns...", "New formula column...", "Join columns from a CSV file..."]
        assert [a.text() for a in p.models_btn.menu().actions()] == ["Fit a formula...", "Model a property..."]
        assert p.features_btn.menu() is p.features_menu and p.export_btn.menu() is not None
        shown = []
        p.model_dialog = lambda show=True: shown.append("model")
        p.a_model.trigger()
        assert shown == ["model"]
        # the plot pops out from its own button, the View menu lists it checked, Dock all panels brings it back
        dock = w.plot_dock
        w.plot_panel.popout_btn.setChecked(True)
        assert _wait(app, lambda: dock.isFloating())
        assert dock.width() >= popout.POPOUT_SIZE.width() and dock.height() >= popout.POPOUT_SIZE.height()
        popout._fill(w, w.popout_menu)
        checked = {a.text(): a.isChecked() for a in w.popout_menu.actions()}
        assert checked.get("Plot") is True and checked.get("Table") is False
        w.dock_all_action.trigger()
        assert _wait(app, lambda: not dock.isFloating()) and not w.plot_panel.popout_btn.isChecked()
        # a View menu check pops any panel out
        popout._fill(w, w.popout_menu)
        next(a for a in w.popout_menu.actions() if a.text() == "Table").setChecked(True)
        assert _wait(app, lambda: w.bottom_dock.isFloating() or any(
            d.isFloating() for d in popout.docks(w) if d.windowTitle() == "Table"))
        assert popout.dock_all(w) == 1
    finally:
        _close(app, w)
