"""The Model Kit (gui.model_kit, UI-7): the 3D view's Build mode driven through the GUI layer's handlers (no real
mouse picks, except one test of the view's VTK-free click path), the QUndoStack wrapper, the edit tools, Minimize
and the hand-off to the molecule list.

The handlers do not need a working 3D view (its methods are no-ops without one), so these tests run where VTK
cannot draw, such as Qt's offscreen platform on Windows; what needs the drawn view (stubs, picking) is checked only
where it works (``drawn``).

    xvfb-run -a python -m pytest tests/test_gui_model_kit.py
"""

import time

import numpy as np
import pytest
from conftest import HAS_DISPLAY, PFOA

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


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


@pytest.fixture
def dialogs(monkeypatch):
    """QMessageBox.warning/information/question without dialogs: records (title, text); question answers Yes."""
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


def drawn(win) -> bool:
    """The 3D view draws (VTK works here): stubs and picking can be checked."""
    return win.viewer.plotter is not None


def _rows(kit, atom, hydrogen=True):
    """Stub rows of ``atom``'s free valences (its hydrogens when ``hydrogen``, else its open directions)."""
    return [r for r, v in enumerate(kit.valences) if v.atom == atom and (v.h is not None) == hydrogen]


def _state(kit):
    return kit.model.to_smiles(), kit.model.positions.copy()


def build_pfoa(kit, record=None):
    """PFOA the way a user builds it in Build mode: an sp2 carbon in empty space, =O and O sp3 on its stubs, then
    seven sp3 carbons, each on a hydrogen stub of the last one, whose other hydrogen stubs get fluorines."""
    def step(result):
        if record is not None:
            record.append(_state(kit))
        return result

    assert kit.choose("C sp2") is not None
    assert step(kit.click_space((0.0, 0.0, 0.0))) == 0
    kit.choose("=O")
    step(kit.click_stub(_rows(kit, 0)[0]))
    kit.choose("O sp3")
    step(kit.click_stub(_rows(kit, 0)[0]))
    cur = 0
    for k in range(8):
        if k < 7:
            kit.choose("C sp3")
            nxt = step(kit.click_stub(_rows(kit, cur)[0]))
            assert nxt is not None, kit.message
        kit.choose("F")
        while _rows(kit, cur):
            assert step(kit.click_stub(_rows(kit, cur)[0])) is not None, kit.message
        cur = nxt
    return kit.model


# ====================================================================== the dock, the mode and the menus
def test_dock_mode_action_and_undo_menu(win):
    from PySide6.QtGui import QKeySequence

    from quantabench.gui import viewer3d, workspaces

    assert "build" in viewer3d.MODES
    assert win.mode_actions["build"].shortcut().toString() == "Ctrl+4"
    assert win.model_kit_dock.objectName() == "modelkit"
    assert workspaces.BY_KEY["build"].right[0] == "modelkit"
    win.set_workspace("build")
    assert "modelkit" in win.workspaces.visible_docks()
    edit = next(a.menu() for a in win.menuBar().actions() if a.text().replace("&", "") == "Edit")
    assert edit.actions()[:2] == [win.a_undo, win.a_redo]
    assert win.a_undo.shortcut() == QKeySequence("Ctrl+Z") and QKeySequence("Ctrl+Y") in win.a_redo.shortcuts()
    assert not win.a_undo.isEnabled() and not win.model_kit.active and not win.model_kit.b_add.isEnabled()
    assert len(win.model_kit.atom_buttons) == 19 and win.model_kit.atom_buttons["C sp3"].isChecked()


def test_build_mode_starts_an_edit_and_owns_the_views(win):
    from quantabench.chem.inputs import from_smiles

    kit = win.model_kit
    win.set_view_mode("build")                       # nothing selected: an empty model
    assert kit.active and kit.model.n_atoms == 0 and kit.source_id is None
    assert win.viewer.mode == "build" and win.mode_actions["build"].isChecked()
    assert kit.close_edit() and not kit.active and win.viewer.mode == "rotate"
    win.add_inputs([from_smiles("CCO", id="ethanol", name="ethanol")])
    win.set_view_mode("build")                       # the selected molecule is edited
    assert kit.active and kit.source_id == "ethanol" and kit.model.to_smiles() == "CCO"
    assert kit.b_replace.text() == "Replace ethanol" and kit.b_replace.isEnabled()
    assert len(win.viewer.atoms) == kit.model.n_atoms == 9
    assert len(kit.valences) == 6
    if drawn(win):     # build mode: hydrogens are stubs, not atoms; the stubs are the model's free valences, in order
        assert set(win.viewer._vis) == {i for i in range(9) if kit.model.symbol(i) != "H"}
        assert len(win.viewer.stub_tips) == len(kit.valences) and "build_stubs" in win.viewer._names
    win.set_view_mode("rotate")                      # still the model, hydrogens back as atoms, no stubs
    assert kit.active
    if drawn(win):
        assert len(win.viewer._vis) == 9 and "build_stubs" not in win.viewer._names
    # selecting another molecule in the list does not take the views from the edit
    win.add_inputs([from_smiles("CCCC", id="butane")])
    assert len(win.viewer.atoms) == 9
    kit.end()
    assert len(win.viewer.atoms) == 14              # butane, selected, is shown again
    kit.b_new.click()                               # New: an empty model, in Build mode
    assert kit.active and kit.model.n_atoms == 0 and win.viewer.mode == "build" and win.viewer.atoms is None
    win.clear_all()                                 # clearing the list leaves the open edit alone
    assert kit.active and win.viewer.mode == "build"


# ====================================================================== building through the handlers
def test_pfoa_built_with_build_mode_handlers(win):
    win.set_view_mode("build")
    kit = win.model_kit
    model = build_pfoa(kit)
    assert model.to_smiles(isomeric=False) == _canon(PFOA)
    assert model.problems() == []
    assert kit.stack.count() == 25 and win.a_undo.isEnabled()     # 1 + 1 + 1 + 7 carbons + 15 fluorines
    assert win.a_undo.text() == "Undo Attach F"
    assert len(win.viewer.atoms) == model.n_atoms == 26
    assert len(kit.valences) == 1 and kit.valences[0].h is not None        # the acid's OH hydrogen


def test_undo_redo_round_trips(win):
    win.set_view_mode("build")
    kit = win.model_kit
    states = []
    build_pfoa(kit, states)
    assert kit.stack.count() == len(states)
    for expected in reversed(states[:-1]):
        win.a_undo.trigger()                          # Edit > Undo (Ctrl+Z)
        smiles, pos = _state(kit)
        assert smiles == expected[0] and np.array_equal(pos, expected[1])
        assert len(win.viewer.atoms) == kit.model.n_atoms
    win.a_undo.trigger()
    assert kit.model.n_atoms == 0 and not kit.stack.canUndo() and win.viewer.atoms is None
    for expected in states:
        win.a_redo.trigger()                          # Edit > Redo (Ctrl+Y)
        smiles, pos = _state(kit)
        assert smiles == expected[0] and np.array_equal(pos, expected[1])
    assert not kit.stack.canRedo() and kit.model.to_smiles(isomeric=False) == _canon(PFOA)
    assert np.allclose(win.viewer.atoms.positions, kit.model.positions)


def test_failed_edit_leaves_model_and_stack_unchanged(win):
    win.set_view_mode("build")
    kit = win.model_kit
    kit.choose("C sp3")
    kit.click_space()
    before, n = _state(kit), kit.stack.count()
    kit.choose("=O")                                  # an sp3 carbon has no double free valence
    assert kit.click_stub(_rows(kit, 0)[0]) is None
    assert "double" in kit.message and kit.stack.count() == n
    assert _state(kit)[0] == before[0] and np.array_equal(_state(kit)[1], before[1])
    kit.choose("COOH")                                # a group needs a free valence, not an atom
    assert kit.click_atom(0) is None and "free valence" in kit.message
    assert kit.click_space((5.0, 0, 0)) is None       # the model is not empty: Shift adds a separate piece
    assert kit.click_space((5.0, 0, 0), shift=True) is not None
    assert len(kit.model.to_smiles().split(".")) == 2
    assert kit.choose("no such thing") is None and "No atom type" in kit.message


def test_minimize_bond_lengths_match_mmff(win):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    win.set_view_mode("build")
    kit = win.model_kit
    build_pfoa(kit)
    res = kit.minimize()
    assert res is not None and res.forcefield == "MMFF94" and res.converged
    assert "Minimized with MMFF94" in kit.message and kit.stack.undoText() == "Minimize"
    ref = Chem.AddHs(Chem.MolFromSmiles(PFOA))
    AllChem.EmbedMolecule(ref, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(ref, mmffVariant="MMFF94", maxIters=5000)
    mol = kit.model.to_mol()
    match = mol.GetSubstructMatch(ref)
    assert len(match) == ref.GetNumAtoms() == mol.GetNumAtoms()
    rp, mp = ref.GetConformer().GetPositions(), mol.GetConformer().GetPositions()
    worst = max(abs(np.linalg.norm(rp[b.GetBeginAtomIdx()] - rp[b.GetEndAtomIdx()])
                    - np.linalg.norm(mp[match[b.GetBeginAtomIdx()]] - mp[match[b.GetEndAtomIdx()]]))
                for b in ref.GetBonds())
    assert worst < 0.05
    # frozen atoms stay where they are
    kit.model.set_distance(0, 1, 1.30)              # stretch C=O a little, then minimize with C and O frozen
    win.selection.select([0, 1])
    kit.freeze_box.setChecked(True)
    kit.minimize()
    assert abs(kit.model.distance(0, 1) - 1.30) < 1e-6 and "2 atom(s) frozen" in kit.message


# ====================================================================== groups, rings, text
def test_groups_rings_pfas_and_text(win, dialogs):
    win.set_view_mode("build")
    kit = win.model_kit
    kit.choose("C sp3")
    kit.click_space()
    assert kit.atom_buttons["C sp3"].isChecked()
    kit.choose("COOH")
    placed = kit.click_stub(_rows(kit, 0)[0])
    assert placed is not None and not any(b.isChecked() for b in kit.atom_buttons.values())
    assert kit.model.to_smiles() == "CC(=O)O"
    kit.choose("benzene")
    kit.click_stub(_rows(kit, 0)[0])
    kit.choose("trifluoromethyl")
    kit.click_stub(_rows(kit, 0)[0])
    assert kit.model.to_smiles(isomeric=False) == "O=C(O)C(c1ccccc1)C(F)(F)F"   # a stereocentre now
    # the PFAS menu's (CF2)n uses the n box; the menus hold the library's groups and rings
    kit.n_box.setValue(3)
    pfas = kit.menus["PFAS"]
    next(a for a in pfas.actions() if a.text().startswith("(CF2)n")).trigger()
    assert kit.choice.caption == "(CF2)3"
    labels = {a.text().split("  ")[0] for a in kit.menus["Groups"].actions()}
    assert {"COOH", "COO-", "SO3-", "SO2NH2", "COO", "CONH", "NO2", "OPO3H2", "CF3"} <= labels
    rings = " ".join(a.text() for a in kit.menus["Rings"].actions())
    assert all(r in rings for r in ("phenyl", "cyclohexyl", "pyridyl", "cyclopentyl", "furanyl", "thiophenyl"))
    assert {a.text() for a in kit.menus["More"].actions()} >= {"N+ sp3", "O-", "S(VI)", "P(V)"}
    # SMILES and InChI from the text box
    frag = kit.use_text("[*:1]C#N")
    assert frag is not None and kit.choice is frag
    frag = kit.use_text("InChI=1S/CH4O/c1-2/h2H,1H3")       # methanol: its first atom bonds
    assert frag is not None and "C" in frag.smiles
    assert kit.use_text("not a smiles((") is None and kit.message
    # a fragment on an empty model (Shift not needed); New asks before it drops the changes
    assert kit.new() and dialogs and dialogs[-1][0] == "New"
    kit.choose("sulfonate")
    assert kit.click_space() is not None and "[O-]" in kit.model.to_smiles() and kit.model.problems() == []


# ====================================================================== the edit tools
def test_edit_tools_on_the_selection(win):
    kit = win.model_kit
    win.set_view_mode("build")
    def carbons():
        return [i for i in range(kit.model.n_atoms) if kit.model.symbol(i) == "C"]

    kit.choose("C sp3")
    kit.click_space()
    c1 = kit.click_stub(_rows(kit, 0)[0])
    assert kit.model.to_smiles() == "CC"
    win.selection.select([0, c1])
    assert kit.value_label.text() == "Distance" and abs(kit.value_box.value() - kit.model.distance(0, c1)) < 1e-3
    kit.cycle_bond_order()                            # single -> double: both carbons become sp2
    assert kit.model.to_smiles() == "C=C"
    for smiles in ("C#C", "CC"):
        win.selection.select(carbons())
        kit.cycle_bond_order()
        assert kit.model.to_smiles() == smiles
    win.selection.select(carbons())
    kit.set_value(1.60)
    assert abs(kit.model.distance(*carbons()) - 1.60) < 1e-6
    win.selection.select([0])
    kit.choose("N sp3")
    kit.click_atom(0)
    assert kit.model.to_smiles() == "CN"
    n = next(i for i in range(kit.model.n_atoms) if kit.model.symbol(i) == "N")
    win.selection.select([n])
    kit.change_charge(1)
    assert kit.model.to_smiles() == "C[NH3+]"
    n = next(i for i in range(kit.model.n_atoms) if kit.model.symbol(i) == "N")
    win.selection.select([n])
    kit.change_charge(-1)
    assert kit.model.to_smiles() == "CN"
    # delete needs a selection; then deletes the atom with its hydrogens
    win.selection.clear()
    assert kit.delete_selected() is None and "Select" in kit.message
    n = next(i for i in range(kit.model.n_atoms) if kit.model.symbol(i) == "N")
    win.selection.select([n])
    kit.delete_selected()
    assert kit.model.to_smiles() == "C"
    # two methanes: make a bond, break it, delete a whole piece
    kit.choose("C sp3")
    kit.click_space((4.0, 0, 0), shift=True)
    assert kit.model.to_smiles() == "C.C"
    win.selection.select([0, 5])
    kit.make_bond()
    assert kit.model.to_smiles() == "CC"
    a, b = [i for i in range(kit.model.n_atoms) if kit.model.symbol(i) == "C"]
    win.selection.select([a, b])
    kit.break_bond()
    assert kit.model.to_smiles() == "C.C"
    win.selection.select([0])
    kit.delete_selected(whole=True)
    assert kit.model.to_smiles() == "C"
    # a Move-mode drag is one undoable step
    n_before = kit.stack.count()
    pos = kit.model.positions + [1.0, 0, 0]
    win.viewer.set_positions(pos)                     # what a finished drag emits
    assert np.allclose(kit.model.positions, pos) and kit.stack.count() == n_before + 1
    assert kit.stack.undoText() == "Move atoms"
    kit.stack.undo()
    assert np.allclose(kit.model.positions, pos - [1.0, 0, 0])


@pytest.mark.parametrize("ortho", [False, True], ids=["perspective", "orthographic"])
def test_clicks_through_the_view(win, ortho):
    """The view's own click path (VTK-free mouse methods): a stub click attaches, an atom click replaces, a
    Ctrl+click selects, a click in empty space of an empty model places the first atom under the cursor."""
    kit = win.model_kit
    viewer = win.viewer
    if not drawn(win):
        pytest.skip(viewer.error)
    win.set_view_mode("build")
    if ortho:
        viewer.toggle_projection()
    w, h = viewer.plotter.renderer.GetSize()

    def click(x, y, ctrl=False):
        viewer._mouse_press(x, y, ctrl=ctrl)
        viewer._mouse_release(x + 1, y)               # < 4 px jitter is still a click

    kit.choose("C sp3")
    expected = viewer.focal_plane_point(w / 2 + 31, h / 2 - 20)     # where the click is released
    click(w / 2 + 30, h / 2 - 20)
    assert kit.model.n_atoms == 5 and len(viewer.stub_tips) == 4
    assert np.allclose(kit.model.positions[0], expected, atol=1e-6)    # then the view centres on it
    kit.choose("Cl")
    tips = viewer.world_to_display(viewer.stub_tips)
    front = int(np.argmin(tips[:, 2]))                # the stub nearest to the camera
    click(*tips[front, :2])
    assert kit.model.to_smiles() == "CCl" and kit.stack.undoText() == "Attach Cl"
    kit.choose("Br")
    cl = next(i for i in range(kit.model.n_atoms) if kit.model.symbol(i) == "Cl")
    x, y, _ = viewer.world_to_display(kit.model.positions[cl])[0]
    click(x, y, ctrl=True)
    assert win.selection.atoms == [cl] and kit.model.to_smiles() == "CCl"   # Ctrl+click selects, builds nothing
    click(x, y)                                       # a plain click on Cl with Br chosen replaces it
    assert kit.model.to_smiles() == "CBr"


# ====================================================================== hand-off
def test_add_to_list_and_replace(win, dialogs):
    from quantabench.chem import jobs
    from quantabench.gui import project_actions

    kit = win.model_kit
    win.set_view_mode("build")
    build_pfoa(kit)
    assert kit.add_to_list()
    assert not kit.active and win.viewer.mode == "rotate"
    row = win.model.rows()[-1]
    assert row.id == "kit_C8HF15O2" and row.source == "model kit" and row.smiles == _canon(PFOA)
    assert row.input.mol is not None and row.input.mol.GetConformer().Is3D()
    assert win.mol_list.current_id() == row.id
    assert len(win.viewer.atoms) == 26 and np.allclose(win.viewer.atoms.positions,
                                                       row.input.mol.GetConformer().GetPositions())
    # the project document keeps the built coordinates
    again = project_actions.row_from_dict(project_actions.row_to_dict(row))
    assert again.input.source == "model kit" and np.allclose(again.input.mol.GetConformer().GetPositions(),
                                                             row.input.mol.GetConformer().GetPositions(), atol=1e-4)
    assert "molblock" in jobs.input_to_json(row.input)
    # Build mode on the listed molecule edits it; Replace puts the edit back under the same id
    win.model.update(row.id, status="ok", n_conformers=3, method="mmff")
    win.set_view_mode("build")
    assert kit.active and kit.source_id == row.id
    f = next(i for i in range(kit.model.n_atoms) if kit.model.symbol(i) == "F")
    kit.choose("Cl")
    kit.click_atom(f)
    assert kit.replace_original()
    again = win.model.row(row.id)
    assert again.status == "pending" and again.n_conformers is None and "Cl" in again.smiles
    assert any(title == "Replace" for title, _t in dialogs)                # a built molecule asks first
    assert len(win.model.rows()) == 1 and not kit.active
    # Close with changes asks; Add to list with nothing built says so
    win.set_view_mode("build")
    kit.new()
    assert kit.add_to_list() is False and dialogs[-1][0] == "Add to list"
    kit.choose("C sp2")
    kit.click_space()
    n = len(dialogs)
    assert kit.close_edit() and len(dialogs) == n + 1 and not kit.active
