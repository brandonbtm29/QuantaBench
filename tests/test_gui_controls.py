"""UI-9: Settings > Controls. The keymap registry (every shortcut is a named action), the presets, rebinding with
conflict detection, reset, persistence per computer and keymap export / import."""

import json
import re
import time
from pathlib import Path

import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")

GUI = Path(__file__).resolve().parents[1] / "src" / "quantabench" / "gui"


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def _window(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    return MainWindow(sm.Settings(output_dir=str(tmp_path / "runs")))


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


def _key(a) -> str:
    from quantabench.gui import keymap

    return keymap.portable(a.shortcut())


@pytest.fixture
def window(tmp_path):
    app = _app()
    w = _window(tmp_path)
    yield w
    from PySide6.QtWidgets import QWidget

    for page in w.findChildren(QWidget, "controls_page_test"):
        page.deleteLater()
    _close(app, w)


def _page(w):
    """A Controls page owned by the window (deleted with it), not shown."""
    from PySide6.QtCore import Qt

    from quantabench.gui.controls_page import ControlsPage

    page = ControlsPage(w.keymap, w)
    page.setWindowFlag(Qt.WindowType.Window)      # its own (hidden) window, not a child drawn over the main one
    page.setObjectName("controls_page_test")
    return page


# ---------------------------------------------------------------------------------------------- the registry
def test_registry_and_presets_are_consistent():
    """Names are unique, every scope and category is known, every preset key parses, presets only name actions of
    the registry, and no preset gives two actions the same key where both are active."""
    _app()
    from quantabench.gui import keymap as km

    assert len(km.BY_NAME) == len(km.SPECS)
    for s in km.SPECS:
        assert s.scope in km.SCOPES and s.category in km.CATEGORIES
        assert re.fullmatch(r"[a-z0-9]+(\.[a-z0-9_]+)+", s.name), s.name
    for p in km.PRESETS.values():
        assert p.key in km.PRESETS and p.label and p.summary
        for name, keys in p.keys.items():
            assert name in km.BY_NAME, f"{p.key}: {name}"
            for k in keys:
                seq = km.seq(k)
                assert seq.count() == 1 and km.portable(seq) == k, f"{p.key}: {name}: {k!r} does not parse"
        if p.key != km.DEFAULT_PRESET:
            assert p.sources and all(url.startswith("https://") for _t, url in p.sources)
        everything = [s.name for s in km.SPECS]
        assert km.all_conflicts(km.KeymapState(p.key), everything) == {}, p.key
    assert km.PRESETS[km.DEFAULT_PRESET].keys == {}       # the default is the registry's own keys


def test_scopes_overlap():
    from quantabench.gui import keymap as km

    assert km.scopes_overlap("window", "view3d") and km.scopes_overlap("view3d", "views")
    assert not km.scopes_overlap("view2d", "view3d")            # the same zoom keys in both views are fine
    assert not km.scopes_overlap("table", "views") and not km.scopes_overlap("table", "molecules")
    assert not km.scopes_overlap("guide", "window")             # the Methods guide is a window of its own


def test_no_literal_shortcuts_outside_the_registry():
    """No ``setShortcut``, ``QShortcut`` or key literal in the GUI modules: keys come from ``gui.keymap`` only."""
    offenders = []
    for path in GUI.glob("*.py"):
        if path.name == "keymap.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\bsetShortcuts?\(|\bQShortcut\(|QKeySequence\(\s*[\"']", code):
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert offenders == []


def test_state_overrides_on_top_of_a_preset():
    _app()
    from quantabench.gui import keymap as km

    st = km.KeymapState()
    st.set_keys("file.export", ["Ctrl+Shift+E"])
    assert st.overrides == {"file.export": ("Ctrl+Shift+E",)}
    st.set_keys("view.reset", ["R"])                       # equal to the preset: no override
    assert "view.reset" not in st.overrides
    st.set_preset("vmd")
    assert km.portable(st.keys("view.reset")[0]) == "Ctrl+R"           # the preset's key ...
    assert km.portable(st.keys("file.export")[0]) == "Ctrl+Shift+E"    # ... and the user's change kept on top
    st.set_keys("view.reset", [])
    assert st.keys("view.reset") == [] and st.overrides["view.reset"] == ()
    with pytest.raises(ValueError):
        st.set_preset("nope")


# ---------------------------------------------------------------------------------------------- the window
@needs_display
def test_every_shortcut_is_a_named_registered_action(window):
    from PySide6.QtGui import QAction

    from quantabench.gui import keymap as km

    w = window
    with_keys = [a for a in w.findChildren(QAction) if a.shortcuts()]
    assert len(with_keys) > 40
    names = [a.objectName() for a in with_keys]
    missing = [(a.objectName(), a.text()) for a in with_keys if a.objectName() not in km.BY_NAME]
    assert missing == []
    assert len(set(names)) == len(names), sorted(n for n in names if names.count(n) > 1)
    assert set(names) <= set(w.keymap.names())          # and the keymap keeps their keys
    assert w.keymap.conflicts() == {}
    # actions of this build are all bound (the ones from other PRs are not here yet)
    assert {s.name for s in km.SPECS if not s.note and s.scope != "guide"} <= set(w.keymap.names())
    # standard keys where the platform has them; the defaults otherwise
    from PySide6.QtGui import QKeySequence

    assert w.a_open.shortcut() == QKeySequence(QKeySequence.StandardKey.Open)
    assert w.a_paste.shortcut() == QKeySequence(QKeySequence.StandardKey.Paste)
    assert _key(w.a_run) == "F5" and _key(w.a_cancel) == "Shift+Esc" and _key(w.a_reset_view) == "R"
    assert [_key(w.mode_actions[k]) for k in ("rotate", "select", "move")] == ["Ctrl+1", "Ctrl+2", "Ctrl+3"]
    assert _key(w.project.a_open) == "Ctrl+Shift+O" and _key(w.a_open_run) == "Ctrl+Shift+R"


@needs_display
def test_switching_presets_changes_the_shortcuts(window):
    from quantabench.gui import keymap as km

    w = window
    w.keymap.set_state(km.KeymapState("vmd"))
    assert _key(w.mode_actions["rotate"]) == "R" and _key(w.mode_actions["move"]) == "8"
    assert _key(w.a_reset_view) == "Ctrl+R"
    assert [km.portable(k) for k in w.playback_actions["next"].shortcuts()] == ["+", "F", "Shift+F"]
    assert w.viewer2d.key_actions["view2d.zoom_in"].shortcuts() == []          # + steps frames in VMD
    w.keymap.set_state(km.KeymapState("avogadro"))
    assert _key(w.mode_actions["select"]) == "Ctrl+4" and _key(w.mode_actions["move"]) == "Ctrl+5"
    assert _key(w.viewer.key_actions["view3d.rotate_left"]) == "Left"
    assert w.playback_actions["next"].shortcuts() == []                        # Avogadro's arrows turn the view
    w.keymap.set_state(km.KeymapState("spartan"))
    assert _key(w.molecule_step_actions["next"]) == "PgDown" and _key(w.molecule_step_actions["first"]) == "Home"
    assert w.playback_actions["first"].shortcuts() == []
    w.keymap.set_state(km.KeymapState())
    assert _key(w.mode_actions["rotate"]) == "Ctrl+1" and _key(w.molecule_step_actions["next"]) == "Down"
    assert w.keymap.conflicts() == {}


@needs_display
def test_tooltips_follow_the_keys(window):
    from quantabench.gui import keymap as km

    w = window
    assert w.a_cancel.toolTip() == "Stop the build after the current step (Shift+Esc)"
    assert w.review_btn.toolTip().endswith("(F6)") and w.new_job_btn.toolTip().endswith("(Ctrl+J)")
    st = km.KeymapState()
    st.set_keys("build.review", ["F7"])
    st.set_keys("build.cancel", [])
    w.keymap.set_state(st)
    assert w.review_btn.toolTip().endswith("(F7)") and w.a_review.toolTip().endswith("(F7)")
    assert w.a_cancel.toolTip() == "Stop the build after the current step"


@needs_display
def test_rebinding_persists_across_a_restart(tmp_path):
    from quantabench.gui import keymap as km

    app = _app()
    w = _window(tmp_path)
    try:
        page = _page(w)
        assert page.assign("file.export", "Ctrl+Shift+E")
        page.set_preset("avogadro")
        w.keymap.set_state(page.result_state())          # what Settings > OK does
        assert _key(w.a_export) == "Ctrl+Shift+E"
    finally:
        _close(app, w)
    w2 = _window(tmp_path)                                # a restart: the same window settings file
    try:
        assert w2.keymap.state.preset == "avogadro"
        assert _key(w2.a_export) == "Ctrl+Shift+E" and _key(w2.mode_actions["move"]) == "Ctrl+5"
        assert w2.keymap.state.overrides == {"file.export": ("Ctrl+Shift+E",)}
        assert km.STORE_PRESET in w2.keymap.store.allKeys()
    finally:
        _close(app, w2)


@needs_display
def test_settings_dialog_has_controls_and_ok_applies_them(window, monkeypatch):
    from quantabench.gui.settings_dialog import SettingsDialog

    w = window
    d = SettingsDialog(w.s, None, w)
    try:
        assert d.controls.keymap is w.keymap and d.show_tab("Controls")
        assert d.tabs.tabText(d.tabs.currentIndex()) == "Controls"
        assert not d.show_tab("Nope")
    finally:
        d.deleteLater()
    # Tools > Keyboard controls... opens Settings on the Controls tab; OK applies the page's keymap
    from quantabench import settings as sm

    shown = []

    def fake_exec(dlg):
        shown.append(dlg.tabs.tabText(dlg.tabs.currentIndex()))
        dlg.controls.set_preset("vmd")
        return 1

    monkeypatch.setattr(SettingsDialog, "exec", fake_exec)
    monkeypatch.setattr(sm, "save", lambda *_a, **_k: None)
    w.a_controls.trigger()
    assert shown == ["Controls"] and w.keymap.state.preset == "vmd" and _key(w.a_reset_view) == "Ctrl+R"


def test_settings_dialog_without_a_window_lists_this_builds_actions():
    _app()
    from quantabench import settings as sm
    from quantabench.gui import keymap as km
    from quantabench.gui.settings_dialog import SettingsDialog

    d = SettingsDialog(sm.Settings(), None)
    try:
        names = d.controls.names
        assert "file.open" in names and "view3d.along_x" in names
        assert "edit.undo" not in names                        # the Model Kit's (another PR) is not in this build
        assert d.controls.result_state().preset == km.DEFAULT_PRESET
    finally:
        d.deleteLater()


# ---------------------------------------------------------------------------------------------- the page
@needs_display
def test_conflict_is_detected_and_swapped_or_cleared(window, monkeypatch):
    from quantabench.gui import keymap as km

    page = _page(window)
    asked = []
    choice = ["swap"]
    monkeypatch.setattr(page, "ask_conflict", lambda name, key, others: asked.append((name, km.portable(key),
                                                                                       others)) or choice[0])
    # Ctrl+O is Open's: swap gives Open the Export key
    assert page.assign("file.export", "Ctrl+O")
    assert asked == [("file.export", "Ctrl+O", ["file.open"])]
    assert page.keys_text("file.export") == "Ctrl+O" and page.keys_text("file.open") == "Ctrl+E"
    assert "Swapped" in page.message and km.all_conflicts(page.state, page.names) == {}
    # clear takes the key away from the other action
    choice[0] = "clear"
    assert page.assign("build.run", "Ctrl+E")
    assert page.keys_text("file.open") == "" and page.keys_text("build.run") == "Ctrl+E"
    # cancel changes nothing
    choice[0] = "cancel"
    before = page.result_state()
    assert not page.assign("build.review", "Ctrl+E")
    assert page.result_state() == before
    # scopes: X in the views (Reset view) clashes with the 3D view's X; F in the 3D view does not clash with the
    # 2D view's F (never both active)
    asked.clear()
    choice[0] = "swap"
    page.assign("view.reset", "X")
    assert asked[-1][2] == ["view3d.along_x"] and page.keys_text("view3d.along_x") == "R"
    asked.clear()
    assert page.assign("view3d.projection", "F") and asked == []


@needs_display
def test_reset_this_action_and_reset_all(window):
    from quantabench.gui import keymap as km

    page = _page(window)
    page.set_preset("spartan")
    page.assign("file.export", "F9")
    page.select("file.export")
    assert page.selected() == "file.export"
    page.clear_selected()
    assert page.keys_text("file.export") == "" and page.state.is_changed("file.export")
    page.reset_selected()
    assert page.keys_text("file.export") == "Ctrl+E" and not page.state.is_changed("file.export")
    page.select("molecules.next")
    page.assign("molecules.next", "N")
    page.reset_selected()                                     # back to the preset's key, not the default's
    assert page.keys_text("molecules.next") == "PgDown"
    page.assign("build.run", "F9")
    page.reset_all()
    assert page.state == km.KeymapState() and page.preset_box.currentData() == "quantabench"
    assert page.keys_text("build.run") == "F5"


@needs_display
def test_page_table_search_and_marks(window):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence


    page = _page(window)
    page.search.setText("zoom")
    shown = page.visible_names()
    assert shown and all("zoom" in n for n in shown)
    page.search.setText("ctrl+shift+o")                       # by key too
    assert page.visible_names() == ["file.open_project"]
    page.search.setText("")
    page.assign("file.export", "F9")
    r = page.row_of("file.export")
    assert page.table.item(r, 2).text() == "F9" and page.table.item(r, 2).font().bold()
    assert "Changed by you" in page.table.item(r, 2).toolTip()
    # key capture: select a row, the field shows its key; a captured key rebinds it
    page.select("build.run")
    assert page.capture.keySequence() == QKeySequence("F5")
    page.capture.setKeySequence(QKeySequence("F8"))
    page.capture.editingFinished.emit()
    assert page.keys_text("build.run") == "F8"
    assert page.table.item(page.row_of("file.open"), 0).data(Qt.ItemDataRole.UserRole) == "file.open"


@needs_display
def test_export_import_round_trip(window, tmp_path):
    from quantabench.gui import keymap as km

    page = _page(window)
    page.set_preset("vmd")
    page.assign("file.export", "Ctrl+Shift+E")
    page.state.set_keys("build.cancel", [])
    path = page.export_to(tmp_path / "keys.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["quantabench_keymap"] == 1 and data["preset"] == "vmd"
    assert data["overrides"] == {"build.cancel": [], "file.export": ["Ctrl+Shift+E"]}
    other = _page(window)
    assert other.state.preset == "quantabench"
    assert other.import_from(path) == []
    assert other.result_state() == page.result_state()
    assert other.preset_box.currentData() == "vmd" and other.keys_text("build.cancel") == ""
    # a file with problems: the unknown action and the bad key are left out and named
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"quantabench_keymap": 1, "preset": "spartan",
                               "overrides": {"no.such": ["F2"], "build.run": ["Ctrl+Nope+Q", "F4"]}}))
    problems = other.import_from(bad)
    assert any("no.such" in p for p in problems) and any("Ctrl+Nope+Q" in p for p in problems)
    assert other.state.preset == "spartan" and other.keys_text("build.run") == "F4"
    with pytest.raises(ValueError):
        km.KeymapState.from_dict({"preset": "vmd"})
    notjson = tmp_path / "x.json"
    notjson.write_text("{nope")
    with pytest.raises(ValueError):
        other.import_from(notjson)


# ---------------------------------------------------------------------------------------------- keys in the views
@needs_display
def test_view_keys_run_their_actions(window, tmp_path):
    """The views' keys reach their actions through ``keymap.dispatch_key`` (whatever the window's activation), and
    a rebound key replaces the old one."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from quantabench.chem.inputs import from_smiles
    from quantabench.gui import keymap as km

    app = _app()
    w = window
    w.add_inputs([from_smiles("CCO", id="a"), from_smiles("CCC", id="b"), from_smiles("CCN", id="c")])
    assert _wait(app, lambda: w.mol_list.count() == 3, 10)
    w.mol_list.setCurrentRow(0)
    view2d = w.viewer2d.view
    if view2d is None:
        pytest.skip(w.viewer2d.error)
    QTest.keyClick(view2d, Qt.Key.Key_Down)                    # molecules.next in the views
    assert w.mol_list.currentRow() == 1
    QTest.keyClick(view2d, Qt.Key.Key_End, Qt.KeyboardModifier.ControlModifier)
    assert w.mol_list.currentRow() == 2
    w.step_molecule(-10)
    assert w.mol_list.currentRow() == 0
    fits = []
    w.viewer2d.key_actions["view2d.fit"].triggered.connect(lambda *_: fits.append(1))
    QTest.keyClick(view2d, Qt.Key.Key_F)
    assert fits == [1]
    st = km.KeymapState()
    st.set_keys("view2d.fit", ["G"])
    w.keymap.set_state(st)
    QTest.keyClick(view2d, Qt.Key.Key_F)
    QTest.keyClick(view2d, Qt.Key.Key_G)
    assert fits == [1, 1]
    # Esc and Select all act on the shared selection
    assert _wait(app, lambda: w.viewer.atom_count() > 0, 10) or w.viewer.plotter is None
    if w.viewer.atom_count():
        w.a_select_all.trigger()
        assert len(w.selection.atoms) == w.viewer.atom_count()
        QTest.keyClick(view2d, Qt.Key.Key_Escape)
        assert w.selection.atoms == []
    # the 3D view's keys: through the VTK widget's dispatcher
    if w.viewer.plotter is not None:
        cam = w.viewer.plotter.renderer.GetActiveCamera()
        par = cam.GetParallelProjection()
        QTest.keyClick(w.viewer.plotter.interactor, Qt.Key.Key_P)
        assert cam.GetParallelProjection() != par


def test_dispatch_key_matches_shifted_symbols():
    """"+" is Shift+= on most keyboards: a key bound to "+" runs whether Shift is reported or not."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QWidget

    from quantabench.gui import keymap as km

    _app()
    w = QWidget()
    try:
        hits = []
        km.local_action(w, "view2d.zoom_in", lambda: hits.append("in"))
        km.local_action(w, "view3d.along_minus_x", lambda: hits.append("-x"))
        for key, mods in ((Qt.Key.Key_Plus, Qt.KeyboardModifier.ShiftModifier),
                          (Qt.Key.Key_Plus, Qt.KeyboardModifier.KeypadModifier),
                          (Qt.Key.Key_X, Qt.KeyboardModifier.ShiftModifier), (Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier)):
            km.dispatch_key(QKeyEvent(QEvent.Type.KeyPress, key, mods), w)
        assert hits == ["in", "in", "-x"]                     # a plain x is not Shift+X
    finally:
        w.deleteLater()
