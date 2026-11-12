"""UI-1: the app shell. Task workspaces (activity bar, per-workspace dock arrangements kept across switches and
restarts), the conformer playback bar under the view, the molecule status line, and the Library's run folders."""

import os
import time
from dataclasses import replace

import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    from PySide6.QtCore import QEvent

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)    # Qt retires old tab bars with deleteLater
        time.sleep(0.02)
    return cond()


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


def _window(tmp_path, show=True):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "runs")))
    if show:
        w.resize(1400, 900)
        w.show()
        _wait(_app(), lambda: False, 0.3)
    return w


def _expected(w, key):
    from quantabench.gui import workspaces

    return {n for n in workspaces.BY_KEY[key].docks if w.workspaces.dock(n) is not None}


def _cfg(method="mmff"):
    from quantabench.chem import pipeline

    return pipeline.BuildConfig(method=method, max_conformers=6, refine_top=6, rmsd_threshold=0.1,
                                xtb_properties=False, protonate=False, write_files=False, threads=1)


@pytest.fixture(scope="module")
def octanoic():
    """A built molecule with several conformers (MMFF, a second or two)."""
    from quantabench.chem import pipeline
    from quantabench.chem.inputs import from_smiles

    res = pipeline.run_molecule(from_smiles("OC(=O)CCCCCCC", id="octanoic_acid"), _cfg())
    assert res.status == "ok" and len(res.conformers) >= 3
    return res


def _run_folder(path, template, n=6, built=4, failed=1, method="mmff"):
    """A run folder in the engine's format (its job table): ``built`` done, ``failed`` failed, the rest to do."""
    from quantabench.chem import jobs
    from quantabench.chem.inputs import from_smiles

    path.mkdir(parents=True, exist_ok=True)
    inputs = [from_smiles("CCO", id=f"M{i:03d}") for i in range(n)]
    with jobs.JobStore(path) as store:
        store.init_run(_cfg(method), None)
        store.add_inputs(inputs)
        for i, inp in enumerate(inputs[:built + failed]):
            res = replace(template, input=inp, files=[])
            if i >= built:
                res = replace(res, status="failed", error="synthetic failure", conformers=[], mol=None)
            store.record(res, "")
    return path


# ---------------------------------------------------------------------------------------------- workspaces
def test_workspace_definitions():
    from quantabench.gui import workspaces

    keys = [w.key for w in workspaces.WORKSPACES]
    assert keys == ["build", "calculate", "simulate", "analyze", "library"]
    assert workspaces.DEFAULT in keys
    assert len({w.shortcut for w in workspaces.WORKSPACES}) == len(keys)
    for ws in workspaces.WORKSPACES:
        assert ws.docks and len(set(ws.docks)) == len(ws.docks)          # a dock appears once per workspace
    assert set(workspaces.MANAGED) == {n for ws in workspaces.WORKSPACES for n in ws.docks}


@needs_display
def test_switching_shows_each_workspaces_docks(tmp_path):
    from PySide6.QtWidgets import QDockWidget, QTabBar

    from quantabench.gui import workspaces

    app = _app()
    w = _window(tmp_path)
    try:
        assert w.workspaces.current == workspaces.DEFAULT          # first start: the default workspace
        assert w.workspaces.showing_default
        assert _wait(app, lambda: abs(w.mol_dock.width() - 270) < 30, 3), w.mol_dock.width()     # its dock widths
        assert w.workspaces.bar.objectName() == "activity_bar" and not w.workspaces.bar.isHidden()
        for key in ("build", "calculate", "simulate", "analyze", "library", "calculate", "build"):
            w.set_workspace(key)
            _wait(app, lambda: False, 0.1)
            assert w.workspaces.current == key and w.workspaces.actions[key].isChecked()
            assert w.workspaces.visible_docks() == _expected(w, key), key
            # tab groups: the first dock of each area in front, the others behind its tabs
            ws = workspaces.BY_KEY[key]
            for area in (ws.left, ws.right, ws.bottom):
                docks = [d for n in area if (d := w.workspaces.dock(n)) is not None]
                if len(docks) > 1:
                    assert set(w.tabifiedDockWidgets(docks[0])) >= set(docks[1:])
                    assert not docks[0].visibleRegion().isEmpty()
            # no tab bar left over from another workspace's groups (stale tabs over the view)
            _wait(app, lambda: False, 0.1)
            shown = {d.windowTitle() for d in w.findChildren(QDockWidget) if d.isVisible()}
            for bar in w.findChildren(QTabBar):
                if bar.parentWidget() is w and bar.isVisible() and bar.geometry().right() >= 0:
                    assert {bar.tabText(i) for i in range(bar.count())} <= shown, key
        # the bar's buttons, the menu and the shortcuts switch too
        w.findChild(object, "workspace_analyze").click()
        assert w.workspaces.current == "analyze"
        from PySide6.QtGui import QKeySequence

        view = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "View")
        sub = next(a.menu() for a in view.actions() if a.menu() is not None and a.text() == "Workspace")
        assert [a.text() for a in sub.actions()] == [ws.label for ws in workspaces.WORKSPACES]
        assert w.workspaces.actions["library"].shortcut() == QKeySequence(f"{workspaces.SHORTCUT_MODIFIER}+5")
        sub.actions()[4].trigger()
        assert w.workspaces.current == "library"
        with pytest.raises(KeyError):
            w.set_workspace("nope")
    finally:
        _close(app, w)


@needs_display
def test_no_two_actions_share_a_shortcut(tmp_path):
    """Qt fires neither action of an ambiguous shortcut: the workspace keys must not take the mouse modes' Ctrl+1 ...
    Ctrl+3 or any other key the window uses."""
    from PySide6.QtGui import QAction, QKeySequence

    from quantabench.gui import workspaces

    app = _app()
    w = _window(tmp_path, show=False)
    try:
        seen = {}
        for a in w.findChildren(QAction):
            for seq in a.shortcuts():
                key = seq.toString(QKeySequence.SequenceFormat.PortableText)
                if key:
                    assert key not in seen or seen[key] is a, f"{key}: {seen.get(key).text()!r} and {a.text()!r}"
                    seen[key] = a
        assert {"Ctrl+1", "Ctrl+2", "Ctrl+3"} <= set(seen)                  # the mouse modes keep theirs
        assert {ws.shortcut for ws in workspaces.WORKSPACES} <= set(seen)
    finally:
        _close(app, w)


@needs_display
def test_each_workspace_keeps_its_own_arrangement(tmp_path):
    app = _app()
    w = _window(tmp_path)
    try:
        w.set_workspace("analyze")
        w.workspaces.dock("spectra").hide()                 # the user closes a dock in Analyze ...
        w.workspaces.dock("machines").show()                # ... and opens one Analyze does not show
        changed = (_expected(w, "analyze") - {"spectra"}) | {"machines"}
        assert w.workspaces.visible_docks() == changed
        w.set_workspace("calculate")
        assert w.workspaces.visible_docks() == _expected(w, "calculate")        # other workspaces unchanged
        w.set_workspace("analyze")
        assert w.workspaces.visible_docks() == changed                          # Analyze as it was left
        w.workspaces.reset()                                                    # View > Reset layout
        assert w.workspaces.visible_docks() == _expected(w, "analyze")
        w.set_workspace("build")
        w.workspaces.dock("results").hide()
        w.workspaces.reset_all()
        assert w.workspaces.visible_docks() == _expected(w, "build") and not w.workspaces.states
    finally:
        _close(app, w)


@needs_display
def test_workspace_and_arrangements_persist_across_restarts(tmp_path):
    from quantabench.gui import workspaces
    from quantabench.gui.main_window import _window_settings

    app = _app()
    w = _window(tmp_path)
    try:
        w.set_workspace("analyze")
        w.workspaces.dock("appearance").hide()
        w.set_workspace("build")
        w.workspaces.dock("results").hide()
        w.set_workspace("analyze")
    finally:
        _close(app, w)
    qs = _window_settings()
    assert qs.value("workspace/current") == "analyze" and int(qs.value("workspace/version")) == workspaces.STATE_VERSION
    assert qs.value("state") is None                       # the pre-workspace layout key is not written any more
    w = _window(tmp_path)
    try:
        assert w.workspaces.current == "analyze"
        assert w.workspaces.visible_docks() == _expected(w, "analyze") - {"appearance"}
        w.set_workspace("build")
        assert w.workspaces.visible_docks() == _expected(w, "build") - {"results"}
        w.set_workspace("simulate")                         # never arranged: its default
        assert w.workspaces.visible_docks() == _expected(w, "simulate")
    finally:
        _close(app, w)


@needs_display
def test_layouts_of_another_version_are_not_used(tmp_path):
    from PySide6.QtCore import QByteArray

    from quantabench.gui.main_window import _window_settings

    qs = _window_settings()
    qs.setValue("workspace/version", 0)                    # an older layout version (or garbage) ...
    qs.setValue("workspace/current", "simulate")
    qs.setValue("workspace/simulate", QByteArray(b"not a state"))
    qs.setValue("state", QByteArray(b"a layout from before workspaces"))
    qs.sync()
    app = _app()
    w = _window(tmp_path)
    try:
        assert w.workspaces.current == "simulate"            # ... the workspace is kept, the arrangement is not
        assert w.workspaces.visible_docks() == _expected(w, "simulate") and not w.workspaces.states
    finally:
        _close(app, w)
    qs = _window_settings()
    qs.setValue("workspace/current", "no-such-workspace")
    qs.sync()
    w = _window(tmp_path, show=False)
    try:
        from quantabench.gui import workspaces

        assert w.workspaces.current == workspaces.DEFAULT
    finally:
        _close(app, w)


@needs_display
def test_appearance_dock_mirrors_the_view_actions(tmp_path):
    app = _app()
    w = _window(tmp_path, show=False)
    try:
        assert w.findChild(object, "appearance") is w.appearance_dock
        w.a_show2d.setChecked(False)
        assert not w.show2d_box.isChecked()
        w.show2d_box.setChecked(True)
        assert w.a_show2d.isChecked()
        w.a_h2d.setChecked(True)
        assert w.h2d_box.isChecked()
        w.h2d_box.setChecked(False)
        assert not w.a_h2d.isChecked()
        w.style_box.setCurrentIndex(1)
        assert w.viewer.style == w.style_box.currentText()
    finally:
        _close(app, w)


# ---------------------------------------------------------------------------------------------- playback
@needs_display
def test_playback_bar_steps_plays_and_wraps():
    from quantabench.gui.playback import PlaybackBar

    _app()
    bar = PlaybackBar()
    seen = []
    bar.indexChanged.connect(seen.append)
    assert bar.count() == 0 and bar.index() == -1 and bar.label.text() == "No conformers"
    assert not bar.play_btn.isEnabled() and not bar.slider.isEnabled()
    bar.set_count(4, lambda i: f"ΔE {i}.0 kJ/mol")
    assert bar.index() == 0 and seen == [] and bar.label.text() == "Conformer 1 of 4, ΔE 0.0 kJ/mol"
    assert not bar.prev_btn.isEnabled() and bar.next_btn.isEnabled()
    bar.step(1)
    bar.next_btn.click()
    assert seen == [1, 2] and bar.slider.value() == 2 and bar.label.text().startswith("Conformer 3 of 4")
    bar.last()
    assert bar.index() == 3 and not bar.next_btn.isEnabled() and not bar.last_btn.isEnabled()
    bar.step(1)                                            # clamped at the end, nothing emitted
    assert seen == [1, 2, 3]
    bar.slider.setValue(1)                                 # dragging the slider moves the view
    assert seen[-1] == 1 and bar.index() == 1
    bar.set_index(2)                                       # the window follows a table click: no echo
    assert seen[-1] == 1 and bar.index() == 2
    bar.toggle_play()
    assert bar.is_playing()
    bar._tick()
    bar._tick()                                            # 2 -> 3 -> wraps to 0
    assert seen[-2:] == [3, 0]
    bar.rate_box.setCurrentIndex(bar.rate_box.findData(8))
    assert bar.timer.interval() == 125
    bar.set_count(2)                                       # another molecule: stops, back to the first
    assert not bar.is_playing() and bar.index() == 0 and bar.label.text() == "Conformer 1 of 2"
    bar.set_count(1)
    assert not bar.play_btn.isEnabled()
    bar.toggle_play()
    assert not bar.is_playing()
    bar.set_count(3, lambda i: 1 / 0)                      # a failing detail: the plain label
    assert bar.label.text() == "Conformer 1 of 3"


@needs_display
def test_playback_follows_the_conformer_table(tmp_path, octanoic):
    from quantabench.chem.inputs import from_smiles

    app = _app()
    w = _window(tmp_path)
    try:
        n = len(octanoic.conformers)
        w.add_inputs([from_smiles("CCO", id="ethanol"), octanoic.input], built=[octanoic])
        w.mol_list.setCurrentRow(1)
        assert _wait(app, lambda: w.playback.count() == n)
        assert w.playback.label.text().startswith(f"Conformer 1 of {n}, ΔE 0.00 kJ/mol, weight")
        status = lambda: " ".join(w.mol_status.text().split())                     # noqa: E731
        assert status() == f"2 molecules · octanoic_acid · conformer 1 of {n}"
        w.playback.step(1)
        assert w.conf_table.currentRow() == 1                       # the bar moves the table (and the views)
        w.conf_table.setCurrentCell(2, 0)
        assert w.playback.index() == 2 and status().endswith(f"conformer 3 of {n}")
        w.playback.toggle_play()
        assert _wait(app, lambda: w.conf_table.currentRow() == 3, 5)
        w.mol_list.setCurrentRow(0)                                 # a molecule not built: nothing to step
        assert _wait(app, lambda: w.playback.count() == 0)
        assert not w.playback.is_playing() and w.playback.label.text() == "No conformers"
        assert status() == "2 molecules · ethanol"
    finally:
        _close(app, w)


# ---------------------------------------------------------------------------------------------- library
def test_find_run_folders(tmp_path, octanoic):
    from quantabench.gui.library_panel import find_run_folders

    root = tmp_path / "runs"
    old = _run_folder(root / "qsar" / "b03", octanoic, n=6, built=5, failed=1)
    new = _run_folder(root / "qsar" / "b04", octanoic, n=6, built=2, failed=0, method="gfn2")
    top = _run_folder(root / "single", octanoic, n=1, built=1, failed=0)
    _run_folder(old / "M000" / "nested", octanoic, n=1, built=1, failed=0)   # inside a run folder: a molecule's
    _run_folder(root / "a" / "b" / "c" / "too-deep", octanoic, n=1, built=1, failed=0)
    (root / "qsar" / "notes").mkdir()
    (root / ".hidden").mkdir()
    _run_folder(root / ".hidden" / "x", octanoic, n=1, built=1, failed=0)
    broken = root / "broken"
    broken.mkdir()
    (broken / "jobs.sqlite").write_bytes(b"not a database")
    for i, folder in enumerate((top, old, new, broken)):                       # newest last
        t = time.time() - 1000 + 100 * i
        os.utime(folder / "jobs.sqlite", (t, t))

    infos = find_run_folders(root)
    assert [i.rel.replace(os.sep, "/") for i in infos] == ["broken", "qsar/b04", "qsar/b03", "single"]
    by = {i.rel.replace(os.sep, "/"): i for i in infos}
    b03, b04 = by["qsar/b03"], by["qsar/b04"]
    assert (b03.total, b03.done, b03.failed, b03.state, b03.method, b03.phase) == (6, 5, 1, "finished", "mmff", "gas")
    assert (b04.total, b04.done, b04.failed, b04.state, b04.method) == (6, 2, 0, "4 to do", "gfn2")
    assert by["broken"].state == "unreadable" and by["broken"].error
    assert [i.rel for i in find_run_folders(root, depth=1)] == ["broken", "single"]
    assert len(find_run_folders(root, limit=2)) == 2
    assert find_run_folders(root, should_stop=lambda: True) == []
    assert find_run_folders(tmp_path / "missing") == []
    assert [i.rel for i in find_run_folders(old)] == ["."]          # the output folder is itself a run folder


@needs_display
def test_library_lists_opens_and_resumes_run_folders(tmp_path, octanoic, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: pytest.fail(str(a[2]))))
    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: (asked.append(a[2]), QMessageBox.StandardButton.Yes)[1]))      # replace the list: yes
    root = tmp_path / "runs"
    done = _run_folder(root / "done", octanoic, n=3, built=3, failed=0)
    todo = _run_folder(root / "todo", octanoic, n=4, built=1, failed=1)
    t = time.time() - 500
    os.utime(done / "jobs.sqlite", (t, t))
    app = _app()
    w = _window(tmp_path)
    try:
        lib = w.library_panel
        assert not lib._scanned                                     # nothing scanned until the Library is shown
        w.set_workspace("library")
        assert _wait(app, lambda: lib.table.rowCount() == 2 and not lib._scanning)
        assert [lib.table.item(r, 0).text() for r in range(2)] == ["todo", "done"]
        assert lib.table.item(0, 7).text() == "2 to do" and lib.table.item(1, 7).text() == "finished"
        assert lib.note.text() == "2 run folders" and str(root) in lib.head.text()
        assert not lib.open_btn.isEnabled() and not lib.resume_btn.isEnabled()
        lib.filter.setText("don")
        assert lib.table.isRowHidden(0) and not lib.table.isRowHidden(1)
        lib.filter.setText("")
        lib.table.sortItems(2)                                      # by molecules: numbers, not text
        assert lib.table.item(0, 0).text() == "done"
        lib.table.selectRow(0)
        assert lib.open_btn.isEnabled() and not lib.resume_btn.isEnabled()     # finished: nothing to resume
        lib.open_selected()
        assert _wait(app, lambda: w.mol_list.count() == 3) and w.run_folder == done
        lib.table.selectRow(1)
        assert lib.selected_folder() == todo and lib.resume_btn.isEnabled()
        resumed = []
        monkeypatch.setattr(w, "resume_run", lambda folder=None: resumed.append(folder))
        lib.resume_selected()                                       # another folder: it is opened first
        assert _wait(app, lambda: w.run_folder == todo and w.mol_list.count() == 4) and resumed == []
        assert len(asked) == 1 and "Replace the 3 molecule(s)" in asked[0]
        lib.resume_selected()
        assert resumed == [todo]
        _run_folder(root / "third", octanoic, n=1, built=1, failed=0)
        lib.refresh()
        assert _wait(app, lambda: lib.table.rowCount() == 3 and not lib._scanning)
    finally:
        _close(app, w)
