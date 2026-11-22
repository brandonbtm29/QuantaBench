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


# ---------------------------------------------------------------------------------------------------- UI-8b
def test_surfaces_panel_sections_and_info(tmp_path):
    """UI-8b: the Surfaces panel's Surface / Color / Descriptors sections with one-line summaries, the colour bar in
    view while Color is folded, and the legend's sentence behind the ⓘ next to Color."""
    app = _app()
    w = _window(tmp_path, workspace="analyze")
    try:
        p = getattr(w, "surface_panel", None)
        if p is None or p.viewer.plotter is None:
            pytest.skip("no 3D view here")
        assert p.surface_section.is_open() and not p.color_section.is_open() and p.descriptors_section.is_open()
        assert not p.surface_section.summary.isVisible()                 # open: the summary is hidden
        p.surface_section.set_open(False)
        assert p.surface_section.summary.text().startswith("off · ")
        assert p.color_section.summary.text().startswith("ESP · bwr reversed · ±")
        assert p.legend_bar.isVisible()                                  # the bar stays while Color is folded
        btn = p.legend_text.parentWidget().info_button
        assert btn.info_text().startswith("ESP, ") and "red = negative" in btn.info_text()
        assert not p.legend_text.isVisible()
        p.opacity_box.setValue(0.5)
        assert "opacity 0.50" in p.color_section.summary.text()
        p.color_box.setCurrentIndex(p.color_box.findData("none"))
        assert p.color_section.summary.text() == "uniform · opacity 0.50"
        assert p.descriptors_section.summary.text() == "no surface"
        assert "Murray-Politzer" in p.descriptors_section.info_button.info_text()
        assert "Bondi" in p.source_info.text() and p.iso_info.text().startswith("The electron density")
    finally:
        _close(app, w)
    w = _window(tmp_path, workspace="analyze")                           # the open states are remembered
    try:
        assert not w.surface_panel.surface_section.is_open()
        w.surface_panel.surface_section.set_open(True)
    finally:
        _close(app, w)


def test_spectra_menu_info_and_modes_fold(tmp_path):
    """UI-8b: Load / Clear / Export in one Spectrum menu (the old button names are its actions), the broadening and
    scale explanation behind ⓘ, and the Normal modes pane folding so the plot takes the width."""
    pytest.importorskip("pyqtgraph")
    app = _app()
    w = _window(tmp_path, workspace="analyze")
    try:
        p = w.spectra_panel
        names = [a.text() for a in p.spectrum_btn.menu().actions() if a.text()]
        assert names == ["Load experimental...", "Clear the experimental spectrum", "Export the spectrum (CSV)..."]
        assert p.load_btn is p.a_load and p.clear_btn is p.a_clear and p.export_btn is p.a_export
        assert not p.clear_btn.isEnabled()
        assert "Lorentzian" in p.broadening_info.info_text()
        w.spectra_dock.show()
        w.spectra_dock.raise_()
        _wait(app, lambda: False, 0.3)
        assert p.modes_section.is_open() and not p.modes_section.summary.isVisible()
        before = p.splitter.sizes()
        p.modes_section.set_open(False)
        _wait(app, lambda: False, 0.1)
        folded = p.splitter.sizes()
        assert folded[1] < before[1] and folded[0] > before[0] and not p.table.isVisible()
        assert p.modes_section.summary.text() == "no modes"
        p.modes_section.set_open(True)
        _wait(app, lambda: False, 0.1)
        assert p.splitter.sizes()[1] > folded[1] and p.table.isVisible()
    finally:
        _close(app, w)


def test_settings_dialog_sections_and_info(tmp_path):
    """UI-8b: the Settings dialog's Compute tab in folding sections with summaries; the long explanations (run
    priority, import, surfaces, display, engines, settings folder) behind ⓘ buttons."""
    from quantabench import settings as sm
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    d = SettingsDialog(sm.Settings(), None)
    try:
        d.show()
        _wait(app, lambda: False, 0.2)
        sec = d.sections
        assert set(sec) == {"compute_threads", "compute_gpu", "compute_lanes", "compute_limits",
                            "compute_expert"}                                       # no report here
        assert sec["compute_threads"].is_open() and sec["compute_lanes"].is_open()
        assert not sec["compute_gpu"].is_open() and not sec["compute_limits"].is_open()
        assert not sec["compute_expert"].is_open()          # SETTINGS-1: lane spec, GPU linear algebra, dxtb
        assert sec["compute_gpu"].summary.text().startswith("GPU auto · device 0")
        d.molecule_timeout.setValue(900)
        assert sec["compute_limits"].summary.text().endswith("900 s per molecule")
        d.lanes.setCurrentText("auto")
        assert sec["compute_lanes"].summary.text().startswith("auto")
        btn = d.priority_info.parentWidget().info_button
        assert not d.priority_info.isVisible() and btn.info_text() == d.priority_info.text() != ""
        assert "rel_kcal" in d.display_preview.parentWidget().info_button.info_text()
        assert "wsl:" in d.engines_info.info_text()
        assert "SyncThing" in d.settings_label.parentWidget().info_button.info_text()
        assert "fixed thresholds" in d.surfaces_note.text()
        sec["compute_gpu"].set_open(True)
    finally:
        d.reject()
        d.deleteLater()
        app.processEvents()
    d = SettingsDialog(sm.Settings(), None)                              # remembered open
    try:
        assert d.sections["compute_gpu"].is_open()
        d.sections["compute_gpu"].set_open(False)
    finally:
        d.reject()
        d.deleteLater()
        app.processEvents()


def test_every_dock_has_a_popout_and_close_button(tmp_path):
    """UI-8b: the docks' title bar with ⧉ (pop out / back) and ✕ (close); the title follows the dock's, and a
    double-click on the title still floats the dock (the bar leaves that to Qt)."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from quantabench.gui import popout

    app = _app()
    w = _window(tmp_path, workspace="calculate")
    try:
        for d in popout.docks(w):
            assert isinstance(d.titleBarWidget(), popout.DockTitleBar), d.objectName()
        dock = w.results_dock
        bar = dock.titleBarWidget()
        assert bar.title.text() == "Results" and not bar.popout_btn.isChecked()
        bar.popout_btn.click()
        assert _wait(app, lambda: dock.isFloating()) and bar.popout_btn.isChecked()
        assert dock.width() >= popout.POPOUT_SIZE.width()
        bar.popout_btn.click()
        assert _wait(app, lambda: not dock.isFloating()) and not bar.popout_btn.isChecked()
        dock.setWindowTitle("Results (3)")
        assert bar.title.text() == "Results (3)"
        QTest.mouseDClick(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(20, bar.height() // 2))
        assert _wait(app, lambda: dock.isFloating(), 3) and bar.popout_btn.isChecked()
        assert popout.dock_all(w) >= 1 and not bar.popout_btn.isChecked()
        jobs = w.jobs_dock.titleBarWidget()
        jobs.close_btn.click()
        assert _wait(app, lambda: w.jobs_dock.isHidden())
        assert popout.add_title_bar(dock) is bar                         # once per dock
    finally:
        _close(app, w)


def test_results_dock_folds_for_one_conformer(tmp_path):
    """UI-8b: one conformer (or none): the Results dock shrinks to the table's header and row; more conformers bring
    its height back; the View menu's switch turns it off."""
    from PySide6.QtCore import Qt

    from quantabench.chem import pipeline
    from quantabench.chem.inputs import from_smiles

    app = _app()
    one = pipeline.run_molecule(from_smiles("C", id="methane"), pipeline.BuildConfig(
        method="mmff", max_conformers=1, refine_top=1, xtb_properties=False, protonate=False, write_files=False))
    many = pipeline.run_molecule(from_smiles("CCCCCCO", id="hexanol"), pipeline.BuildConfig(
        method="mmff", max_conformers=8, refine_top=8, xtb_properties=False, protonate=False, write_files=False))
    assert len(one.conformers) == 1 and len(many.conformers) > 2
    w = _window(tmp_path, workspace="build")
    try:
        f = w.results_folder
        assert f.action.isChecked() and f.action in next(
            a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "View").actions()
        dock = w.results_dock
        w.add_inputs([one.input, many.input], built=[one, many])
        w.bottom_tabs.setCurrentWidget(w.conf_table)
        w.resizeDocks([dock], [360], Qt.Orientation.Vertical)
        _wait(app, lambda: False, 0.3)
        tall = dock.height()
        w.mol_list.setCurrentRow(1)                                      # hexanol: many rows, nothing to fold
        _wait(app, lambda: False, 0.3)
        assert abs(dock.height() - tall) <= 2
        w.mol_list.setCurrentRow(0)                                      # methane: one row
        assert _wait(app, lambda: dock.height() < tall - 40, 5)
        assert w.conf_table.rowCount() == 1 and f.folded_from is not None
        w.mol_list.setCurrentRow(1)
        assert _wait(app, lambda: abs(dock.height() - tall) <= 4, 5) and f.folded_from is None
        f.action.setChecked(False)                                       # off: nothing folds
        w.mol_list.setCurrentRow(0)
        _wait(app, lambda: False, 0.4)
        assert abs(dock.height() - tall) <= 4
        f.action.setChecked(True)
    finally:
        _close(app, w)
