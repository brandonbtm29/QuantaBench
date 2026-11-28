"""Clean up in the GUI (BUILD-CLEAN, gui.clean_up): the Model Kit's button as one undoable step with held atoms, the
Molecules dock's right-click entries, the 2D view's re-layout, the Sketch dock's button and the named actions.

    xvfb-run -a python -m pytest tests/test_gui_clean_up.py
"""

import time

import numpy as np
import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")

GENX_R = "OC(=O)[C@@](F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)F"
L_ALA = "N[C@@H](C)C(=O)O"


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()


@pytest.fixture
def win(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "runs")))
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    yield w
    w.close()
    assert _wait(app, lambda: w.closed, 30)
    w.sketch_panel.shutdown()


@pytest.fixture
def dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    seen = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, title, text, *a, **k: seen.append((title, text)))
    monkeypatch.setattr(QMessageBox, "information", lambda _p, title, text, *a, **k: seen.append((title, text)))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda _p, title, text, *a, **k: seen.append((title, text)) or QMessageBox.StandardButton.Yes)
    return seen


def _canon(smiles):
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _distorted(smiles, seed=3):
    from quantabench.chem.edit import EditableMolecule

    m = EditableMolecule.from_smiles(smiles)
    pos = m.positions + np.random.default_rng(seed).normal(0.0, 0.15, m.positions.shape)
    pos[1] += np.array([0.5, 0.0, 0.0])
    m.set_positions(pos)
    m.history.clear()
    return m


def _clean_menu(win):
    menu = win.mol_list.context_menu()
    sub = next((a.menu() for a in menu.actions() if a.menu() is not None and a.text() == "Clean up"), None)
    return menu, sub


def _key(action):
    from quantabench.gui import keymap as km

    return [km.portable(k) for k in action.shortcuts()]


def test_actions_keys_and_buttons(win):
    build = next(a.menu() for a in win.menuBar().actions() if a.text().replace("&", "") == "Build")
    assert win.a_clean_2d in build.actions() and win.a_clean_3d in build.actions()
    assert win.a_clean_2d.objectName() == "build.clean_2d" and win.a_clean_3d.objectName() == "build.clean_3d"
    assert _key(win.a_clean_2d) == [] and _key(win.a_clean_3d) == []          # no default keys, as Minimize
    from quantabench.gui import keymap as km

    state = km.KeymapState()
    state.set_keys("build.clean_2d", ["Ctrl+Shift+L"])                          # keys of the user's own work
    state.set_keys("build.clean_3d", ["Ctrl+Shift+M"])
    win.keymap.set_state(state, save=False)
    assert _key(win.a_clean_2d) == ["Ctrl+Shift+L"] and _key(win.a_clean_3d) == ["Ctrl+Shift+M"]
    assert win.keymap.conflicts() == {}
    assert win.model_kit.b_clean.toolTip().endswith("(Ctrl+Shift+M)")          # the kit's button shows its key
    kit = win.model_kit
    assert kit.b_clean.text() == "Clean up" and not kit.b_clean.isEnabled()       # no edit open
    kit.start(_distorted(GENX_R))
    assert kit.b_clean.isEnabled()
    p = win.sketch_panel
    assert p.b_clean.text() == "Clean up" and not p.b_clean.isEnabled()           # the sketcher is not running


def test_model_kit_clean_up_is_one_undoable_step(win):
    kit = win.model_kit
    kit.start(_distorted(GENX_R), name="GenX")
    before = kit.model.positions.copy()
    smiles = kit.model.to_smiles()
    win.viewer2d.rotate(40.0)
    kit.b_clean.click()
    after = kit.model.positions.copy()
    assert kit.stack.count() == 1 and kit.stack.undoText() == "Clean up"
    assert "Cleaned up with MMFF94: E " in kit.message and " → " in kit.message
    assert "bonds and stereo unchanged" in kit.message
    assert kit.model.to_smiles() == smiles == _canon(GENX_R)
    assert np.abs(after - before).max() > 0.3
    assert np.allclose(win.viewer2d._xform, np.eye(2))                           # the 2D view laid out again
    win.a_undo.trigger()
    assert np.array_equal(kit.model.positions, before)
    win.a_redo.trigger()
    assert np.array_equal(kit.model.positions, after)


def test_model_kit_holds_the_selected_atoms_fixed(win):
    kit = win.model_kit
    kit.start(_distorted(GENX_R))
    before = kit.model.positions.copy()
    win.selection.select([0, 1, 2])
    kit.freeze_box.setChecked(True)
    res = kit.clean_up()
    assert res is not None and res.held == 3 and "3 atom(s) held fixed" in kit.message
    assert np.array_equal(kit.model.positions[[0, 1, 2]], before[[0, 1, 2]])


def test_model_kit_refused_clean_up_changes_nothing(win, monkeypatch):
    from quantabench.chem.edit import EditableMolecule

    minimize = EditableMolecule._op_minimize

    def mirrored(self, frozen, max_iters, movable=None):
        out = minimize(self, frozen, max_iters, movable)
        p = self.positions.copy()
        p[:, 0] *= -1.0
        self._set_positions(p)
        return out

    kit = win.model_kit
    kit.start(_distorted(L_ALA))
    before = kit.model.positions.copy()
    monkeypatch.setattr(EditableMolecule, "_op_minimize", mirrored)
    assert kit.clean_up() is None
    assert kit.stack.count() == 0 and np.array_equal(kit.model.positions, before)
    assert "refused" in kit.message and "would turn from S to R" in kit.message


def test_molecule_list_right_click_cleans_up_in_the_model_kit(win):
    from quantabench.chem.inputs import from_smiles

    menu, sub = _clean_menu(win)
    assert sub is None                                       # nothing listed: no Clean up
    menu.deleteLater()
    win.add_inputs([from_smiles(GENX_R, id="GenX")])
    assert win.mol_list.select_id("GenX")
    win.mol_list.currentIdChanged.emit("GenX")
    menu, sub = _clean_menu(win)
    texts = [a.text() for a in sub.actions()]
    assert texts == ["2D layout", "3D structure (in the Model Kit)",
                     "3D structure, holding the selected atoms fixed (select atoms first)"]
    assert [a.isEnabled() for a in sub.actions()] == [True, True, False]
    sub.actions()[1].trigger()
    menu.deleteLater()
    kit = win.model_kit
    assert kit.active and kit.source_id == "GenX" and win.model_kit_dock.isVisible()
    assert kit.stack.count() == 1 and kit.stack.undoText() == "Clean up"
    assert kit.model.to_smiles() == _canon(GENX_R)
    menu, sub = _clean_menu(win)
    assert sub.actions()[1].text() == "3D structure (the open Model Kit edit)"
    menu.deleteLater()


def test_right_click_holds_the_atoms_selected_in_the_views(win):
    from quantabench.chem.edit import EditableMolecule
    from quantabench.chem.inputs import from_smiles

    win.add_inputs([from_smiles(L_ALA, id="L-Ala")])
    assert win.mol_list.select_id("L-Ala")
    win.mol_list.currentIdChanged.emit("L-Ala")
    win.selection.select([0, 1])
    start = EditableMolecule.from_input(win._input_of(win.model.row("L-Ala"))).positions   # what the edit opens with
    menu, sub = _clean_menu(win)
    held = sub.actions()[2]
    assert held.text() == "3D structure, holding the 2 selected atoms fixed" and held.isEnabled()
    held.trigger()
    menu.deleteLater()
    kit = win.model_kit
    assert kit.active and kit.selected() == [0, 1]
    assert np.array_equal(kit.model.positions[[0, 1]], start[[0, 1]])
    assert not np.allclose(kit.model.positions, start)
    assert kit.model.to_smiles() == _canon(L_ALA)


def test_clean_2d_lays_the_2d_view_out_again(win):
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.viewer2d import build_depiction

    win.add_inputs([from_smiles(GENX_R, id="GenX")])
    assert win.mol_list.select_id("GenX")
    win.mol_list.currentIdChanged.emit("GenX")
    v = win.viewer2d
    fresh = build_depiction(v._full)[0].GetConformer().GetPositions()[:, :2]
    v.rotate(30.0)
    v.flip_horizontal()
    assert not np.allclose(v.coords_2d, fresh)
    win.a_clean_2d.trigger()
    assert np.allclose(v._xform, np.eye(2)) and np.allclose(v.coords_2d, fresh)
    assert "2D layout redone (CoordGen)" in win.statusBar().currentMessage()
    win.viewer2d.clear()
    win.a_clean_2d.trigger()
    assert "No molecule is shown" in win.statusBar().currentMessage()


def test_sketch_dock_clean_up_of_a_drawing(win, dialogs):
    from rdkit import Chem

    from quantabench.chem import clean

    m = Chem.MolFromSmiles(GENX_R)
    clean.layout_2d(m)
    conf = m.GetConformer()
    rng = np.random.default_rng(2)
    for i in range(m.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + rng.normal(0, 0.3), p.y + rng.normal(0, 0.3), 0.0))
    messy = Chem.MolToV3KMolBlock(m)
    assert Chem.MolToSmiles(Chem.MolFromMolBlock(messy)) == _canon(GENX_R)
    p = win.sketch_panel
    block = p.clean_block(messy)
    assert block is not None and Chem.MolToSmiles(Chem.MolFromMolBlock(block)) == _canon(GENX_R)
    assert "Cleaned up" in p.status.text() and "stereo unchanged" in p.status.text()
    assert p.clean_block("") is None
    assert dialogs[-1][0] == "Clean up" and "no structure" in dialogs[-1][1]
