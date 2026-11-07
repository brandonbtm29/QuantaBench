"""The Sketch dock (gui.sketcher_dock): menus, what it says without Qt WebEngine or the editor's files, the MOL-block
hand-off to the molecule list, and (QB_TEST_SKETCHER=1, with Qt WebEngine and the downloaded editor) the real
Schrödinger Sketcher round trip of PFAS structures through its JavaScript API.

    QB_TEST_SKETCHER=1 xvfb-run -a python -m pytest tests/test_gui_sketcher.py    # after quantabench-cli sketcher --install
"""

import os
import time

import pytest
from conftest import HAS_DISPLAY, PFOA

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")

PFOS = "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
GENX = "OC(=O)C(F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)F"
FTS_62 = "OS(=O)(=O)CCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
# neopentane with a fifth methyl, as the sketcher exports a drawing with too many bonds on one carbon
PENTAVALENT_C = """
     RDKit          2D

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 6 5 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C -1.299038 -0.750000 0.000000 0
M  V30 2 C 0.000000 0.000000 0.000000 0
M  V30 3 C 0.311868 -1.467221 0.000000 0
M  V30 4 C 0.610105 1.370318 0.000000 0
M  V30 5 C 1.491783 -0.156793 0.000000 0
M  V30 6 C -1.114717 1.003696 0.000000 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 2 3
M  V30 3 1 2 4
M  V30 4 1 2 5
M  V30 5 1 2 6
M  V30 END BOND
M  V30 END CTAB
M  END
$$$$
"""


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
    assert _wait(app, lambda: w.closed, 30)
    w.sketch_panel.shutdown()


def _canon(smiles):
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _block(smiles):
    from rdkit import Chem

    from quantabench.chem import sketch

    return sketch.molblock_for_mol(Chem.MolFromSmiles(smiles))


@pytest.fixture
def warnings(monkeypatch):
    """QMessageBox.warning/information/question without dialogs: records (title, text); question answers Yes."""
    from PySide6.QtWidgets import QMessageBox

    seen = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, title, text, *a, **k: seen.append((title, text)))
    monkeypatch.setattr(QMessageBox, "information", lambda _p, title, text, *a, **k: seen.append((title, text)))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda _p, title, text, *a, **k: seen.append((title, text)) or QMessageBox.StandardButton.Yes)
    return seen


def test_dock_and_menus(tmp_path):
    app = _app()
    w = _window(tmp_path)
    assert w.sketch_dock.objectName() == "sketch" and not w.sketch_dock.isVisible()
    build = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "Build")
    view = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "View")
    assert w.a_sketch in build.actions()
    assert w.sketch_dock.toggleViewAction() in view.actions()
    assert w.sketch_panel.view is None              # nothing of Qt WebEngine before the dock is opened
    _close(app, w)


def test_without_webengine_the_dock_says_what_to_install(tmp_path, monkeypatch):
    from quantabench.gui import sketcher_dock, sketcher_web

    monkeypatch.setattr(sketcher_web, "webengine_installed", lambda: False)
    app = _app()
    w = _window(tmp_path)
    sketcher_dock.open_dock(w)
    p = w.sketch_panel
    app.processEvents()
    text = p.message.text()
    assert "quantabench[editor]" in text and "Qt WebEngine" in text and "works without it" in text
    assert p.view is None and not p.b_download.isVisible()
    assert not any(b.isEnabled() for b in (p.b_new, p.b_edit, p.b_add, p.b_replace, p.b_fragment))
    from quantabench.chem.inputs import from_smiles

    w.add_inputs([from_smiles("CCO", id="ethanol")])
    assert w.mol_list.count() == 1                   # the rest of the window works as before
    _close(app, w)


def test_without_the_editor_files_it_offers_the_download(tmp_path, monkeypatch):
    from quantabench import sketcher_assets
    from quantabench.gui import sketcher_dock, sketcher_web

    monkeypatch.setattr(sketcher_web, "problem", lambda: "")
    monkeypatch.setattr(sketcher_assets, "programs_dir", lambda: tmp_path / "programs")
    monkeypatch.delenv(sketcher_assets.ENV_DIR, raising=False)
    app = _app()
    w = _window(tmp_path)
    sketcher_dock.open_dock(w)
    p = w.sketch_panel
    app.processEvents()
    text = p.message.text()
    assert "not downloaded yet" in text and sketcher_assets.URL in text and "--from-zip" in text
    assert p.b_download.isVisible() and p.b_zip.isVisible() and p.view is None
    _close(app, w)


def test_install_failure_is_shown(tmp_path, monkeypatch):
    from quantabench import sketcher_assets
    from quantabench.gui import sketcher_dock, sketcher_web

    def offline(**_kw):
        raise sketcher_assets.SketcherAssetError("Could not download Schrödinger's Sketcher (offline).\n"
                                                 + sketcher_assets.manual_instructions())

    monkeypatch.setattr(sketcher_web, "problem", lambda: "")
    monkeypatch.setattr(sketcher_assets, "programs_dir", lambda: tmp_path / "programs")
    monkeypatch.setattr(sketcher_assets, "install", offline)
    monkeypatch.delenv(sketcher_assets.ENV_DIR, raising=False)
    app = _app()
    w = _window(tmp_path)
    sketcher_dock.open_dock(w)
    p = w.sketch_panel
    p.start_install()
    assert _wait(app, lambda: not p._installing, 20)
    assert "Could not download" in p.message.text() and sketcher_assets.URL in p.message.text()
    assert p.b_download.isVisible()
    _close(app, w)


def test_hand_off_adds_and_replaces(tmp_path, warnings):
    from quantabench.chem.inputs import from_smiles

    app = _app()
    w = _window(tmp_path)
    p = w.sketch_panel
    assert p.add_block(_block(PFOA))
    rows = w.model.rows()
    assert [r.id for r in rows] == ["sketch_C8HF15O2"] and rows[0].smiles == _canon(PFOA)
    assert rows[0].source == "sketch" and w.mol_list.current_id() == "sketch_C8HF15O2"
    assert p.add_block(_block(PFOA)) and w.model.ids()[-1] == "sketch_C8HF15O2_2"    # ids stay unique
    # a structure problem is reported and nothing is added
    assert not p.add_block(PENTAVALENT_C)
    assert warnings and warnings[-1][0] == "Add to list" and "has 5 bonds" in warnings[-1][1]
    assert w.mol_list.count() == 2
    # Replace selected keeps the id, name and place, and resets the row
    w.add_inputs([from_smiles("CCO", id="mol1", name="first")])
    w.model.update("mol1", status="ok", n_conformers=3, method="mmff", folder=str(tmp_path))
    assert w.mol_list.select_id("mol1")
    assert p.replace_block(_block(PFOS))
    row = w.model.row("mol1")
    assert (row.name, row.smiles, row.status, row.n_conformers, row.folder) == ("first", _canon(PFOS), "pending", None, "")
    assert row.input.smiles == _canon(PFOS) and w.model.position("mol1") == 2
    assert any("Replace" in title for title, _t in warnings)          # a built molecule asks first
    assert w.selected_input().smiles == _canon(PFOS)
    _close(app, w)


def test_replace_needs_a_selection(tmp_path, warnings):
    app = _app()
    w = _window(tmp_path)
    assert not w.sketch_panel.replace_block(_block(PFOA))
    assert warnings[-1][0] == "Replace selected"
    _close(app, w)


# ---------------------------------------------------------------------------------------- the real sketcher
def test_real_sketcher_round_trip():
    """QB_TEST_SKETCHER=1: run ``test_real_sketcher_in_child`` in a child pytest, which registers the sketcher's
    URL scheme before its QApplication (conftest.pytest_configure) and takes Chromium's processes with it when it
    ends. Skipped without Qt WebEngine or the downloaded editor."""
    import subprocess
    import sys
    from pathlib import Path

    if os.environ.get("QB_TEST_SKETCHER") != "1":
        pytest.skip("set QB_TEST_SKETCHER=1 to drive the real sketcher (needs Qt WebEngine and its download)")
    from quantabench import sketcher_assets
    from quantabench.gui import sketcher_web

    if not sketcher_web.webengine_installed():
        pytest.skip("Qt WebEngine is not installed")
    loc = sketcher_assets.locate()
    if not loc.ok:
        pytest.skip(f"the sketcher is not downloaded ({loc.problem}): quantabench-cli sketcher --install")
    here = Path(__file__).resolve()
    env = {k: v for k, v in os.environ.items() if k != "QB_TEST_SKETCHER"}
    env["QB_TEST_SKETCHER_CHILD"] = "1"
    env.setdefault(sketcher_assets.ENV_DIR, str(loc.path))     # the child's settings are a fresh test folder
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
                           f"{here}::test_real_sketcher_in_child"], cwd=here.parents[1], env=env,
                          capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "1 passed" in out, out[-6000:]


def _js(app, call, timeout=30.0):
    box = []
    call(lambda ok, value: box.append((ok, value)))
    assert _wait(app, lambda: box, timeout), "the sketcher did not answer"
    return box[0]


def test_real_sketcher_in_child(tmp_path, warnings):
    if os.environ.get("QB_TEST_SKETCHER_CHILD") != "1":
        pytest.skip("runs inside test_real_sketcher_round_trip's child pytest")
    from quantabench.gui import sketcher_web

    assert not sketcher_web.problem(), sketcher_web.problem()
    from quantabench.chem import sketch
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui import sketcher_dock

    app = _app()
    w = _window(tmp_path)
    w.show()
    sketcher_dock.open_dock(w)
    p = w.sketch_panel
    assert _wait(app, lambda: p.ready, 240), p.message.text() or p.status.text()
    v = p.view
    for smiles in (PFOA, PFOS, GENX, FTS_62):
        ok, _ = _js(app, lambda cb, s=smiles: v.import_text(s, clear=True, callback=cb))
        assert ok
        ok, block = _js(app, lambda cb: v.export_text("MDL_MOLV3000", cb))
        assert ok and "V3000" in block
        assert sketch.input_from_molblock(block).input.smiles == _canon(smiles), smiles
    # the change callback reaches the dock: Add to list is enabled once something is drawn
    assert _wait(app, lambda: not p.empty and p.b_add.isEnabled(), 10)
    # an error from the C++ side comes back as text
    ok, message = _js(app, lambda cb: v.import_text("C1CC is not a molecule", callback=cb))
    assert not ok and "format" in message
    # Insert fragment: added next to the drawing as a second piece
    p.insert_fragment("C(F)(F)F")
    ok, smiles = _js(app, lambda cb: v.export_text("SMILES", cb))
    assert ok and len(smiles.split(".")) == 2
    # Edit selected loads the molecule's input; Add to list hands it back
    w.add_inputs([from_smiles("C[C@H](N)C(=O)O", id="L-Ala")])
    assert w.mol_list.select_id("L-Ala")
    p.edit_selected()
    ok, block = _js(app, lambda cb: v.export_text("MDL_MOLV3000", cb))
    assert ok and sketch.input_from_molblock(block).input.smiles == _canon("C[C@H](N)C(=O)O")
    n = w.mol_list.count()
    p.add_to_list()
    assert _wait(app, lambda: w.mol_list.count() == n + 1, 20)
    assert w.model.rows()[-1].smiles == _canon("C[C@H](N)C(=O)O")
    p.new()
    assert _wait(app, lambda: p.empty and not p.b_add.isEnabled(), 10)
    _close(app, w)
    assert p.view is None
