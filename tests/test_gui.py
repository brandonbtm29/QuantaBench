import os

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                reason="needs a display (run under xvfb-run on Linux)")


def test_main_window_builds(tmp_path):
    from PySide6.QtWidgets import QApplication

    from quantabench import settings as sm
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    s = sm.Settings(output_dir=str(tmp_path))
    w = MainWindow(s)
    w.add_inputs([from_smiles("CCO", id="ethanol")])
    cfg = w.current_config()
    assert cfg.output_dir == str(tmp_path)
    assert w.mol_list.count() == 1
    _close(app, w)


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


def _close(app, w, timeout=30.0):
    """close() no longer blocks while workers run (hardware detection, lookups): wait until it has closed,
    so no QThread is destroyed while running when the window is garbage-collected."""
    w.close()
    assert _wait(app, lambda: w.closed, timeout)


def test_shared_selection_and_measurement(tmp_path):
    from quantabench import settings as sm
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.main_window import MainWindow

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    w.add_inputs([from_smiles("OC(=O)C(F)(F)F", id="TFA")])
    assert w.viewer.atoms is not None and w.viewer2d.depicted_to_full
    assert w.viewer.selection is w.viewer2d.selection is w.selection
    w.selection.select([0, 1])
    app.processEvents()
    assert set(w.viewer2d.highlighted_atoms) == {0, 1}
    assert w.measure_label.text().startswith("distance:")
    w.label_box.setCurrentIndex(w.label_box.findData("index"))
    assert w.viewer.label_mode == "index" and w.viewer2d.atom_indices
    w.set_view_mode("select")
    assert w.viewer.mode == "select" and w.mode_actions["select"].isChecked()
    w._set_2d_visible(False)
    assert not w.viewer2d.isVisibleTo(w) and not w.a_show2d.isChecked()
    w.w_freq.setChecked(True)
    cfg = w.current_config()
    assert cfg.frequencies and "spectrum" in cfg.formats and cfg.batched == "auto"
    assert cfg.symmetry_number == 0                      # auto: detected per molecule
    _close(app, w)


def test_surface_builder_move_mode_updates_system(tmp_path):
    import numpy as np
    from ase.build import molecule

    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    mol = molecule("CH3CH2OH")
    w._selected_atoms = lambda: (mol, "ethanol", None)
    w.open_surface_builder()
    dlg = w.surface_dialog
    dlg.nx.setValue(6)
    dlg.ny.setValue(4)
    dlg.layers.setValue(2)
    dlg.move_btn.setChecked(True)          # builds, previews, switches the 3D view to move mode
    assert w.viewer.mode == "move" and w.mode_actions["move"].isChecked()
    assert w.viewer.color_mode == "tag"
    group = w.viewer.move_group()
    assert len(group) == len(mol) and np.all(w.viewer.atoms.get_tags()[group] == 1)
    before = dlg.system.positions.copy()
    w.viewer.rotate_atoms(group, [0, 0, 1], 40.0)
    assert np.allclose(dlg.system.positions, w.viewer.atoms.positions)
    assert not np.allclose(dlg.system.positions[group], before[group])
    assert np.allclose(dlg.system.positions[: len(before) - len(mol)], before[: len(before) - len(mol)])
    assert "moved by hand" in dlg.status.text()
    out = tmp_path / "smd"
    dlg.export_smd(str(out))
    assert _wait(app, lambda: (out / "in.smd").exists() and dlg.smd_btn.isEnabled())
    dlg.close()
    app.processEvents()
    assert w.surface_dialog is None and w.viewer.mode == "rotate"
    _close(app, w)


def test_worker_callbacks_run_on_gui_thread():
    """PySide6 runs a closure connected to a worker signal in the worker thread; workers.start relays
    on_done/on_failed to the GUI thread so they may touch widgets."""
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QWidget

    from quantabench.gui import workers

    app = _app()
    owner = QWidget()
    main = QThread.currentThread()
    seen = {}
    workers.start(workers.FunctionWorker(lambda: 1), owner,
                  on_done=lambda v: seen.__setitem__("done", (v, QThread.currentThread() is main)))
    workers.start(workers.FunctionWorker(lambda: 1 / 0), owner,
                  on_failed=lambda m: seen.__setitem__("failed", ("ZeroDivisionError" in m,
                                                                  QThread.currentThread() is main)))
    assert _wait(app, lambda: len(seen) == 2 and not owner.__dict__.get("_live_workers"), 10)
    assert seen == {"done": (1, True), "failed": (True, True)}


def test_esc_does_not_cancel_and_close_does_not_block(tmp_path, monkeypatch):
    import time

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from quantabench import settings as sm
    from quantabench.chem import pipeline
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.main_window import MainWindow

    def slow_run(inputs, cfg, should_stop=lambda: False, **_kw):     # an uninterruptible step
        time.sleep(1.5)
        return [], None

    monkeypatch.setattr(pipeline, "run", slow_run)
    from PySide6.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(a) or QMessageBox.StandardButton.Yes))
    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    w.show()
    w.add_inputs([from_smiles("CCO", id="e")])
    fired = []
    w.a_cancel.triggered.connect(lambda *_: fired.append("cancel"))
    w.run_build()
    assert w.worker is not None and w.a_cancel.isEnabled()
    assert w.a_cancel.shortcut().toString() != "Esc"
    w.viewer.plotter.interactor.setFocus()
    QTest.keyClick(w.viewer.plotter.interactor, Qt.Key.Key_Escape)
    app.processEvents()
    assert fired == [] and w.worker is not None
    t = time.perf_counter()
    w.close()
    assert time.perf_counter() - t < 0.5                 # returns at once, closes when the step ends
    assert len(asked) == 1 and not w.closed and "Closing" in w.statusBar().currentMessage()
    assert _wait(app, lambda: w.closed, 30)
    assert w.worker is None


def test_surface_preview_survives_list_selection_and_selection_resets_on_build(tmp_path):
    from ase.build import molecule

    from quantabench import settings as sm
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.main_window import MainWindow

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    w.add_inputs([from_smiles("OC(=O)CCCC(=O)O", id="glutaric"), from_smiles("CCO", id="eth")])
    # the input preview and the built molecule number their atoms differently: the selection is cleared
    w.mol_list.setCurrentRow(0)
    w.selection.select([10, 11])
    w.w_method.setCurrentIndex(w.w_method.findData("mmff"))
    w.w_maxconf.setValue(2)
    w.w_xtbprops.setChecked(False)
    w.run_build(selected_only=False)
    assert _wait(app, lambda: w.worker is None, 240)
    assert w.results["glutaric"].status == "ok"
    assert w.selection.atoms == []                       # preview indices do not carry over to the build
    w.mol_list.setCurrentRow(1)
    w.mol_list.setCurrentRow(0)
    w.selection.select([10, 11])
    w._on_molecule_done(w.results["glutaric"])           # same built result shown again: selection kept
    assert w.selection.atoms == [10, 11]

    # an open Surface Builder keeps the 3D view while the list selection changes
    mol = molecule("CH3CH2OH")
    w._selected_atoms = lambda: (mol, "ethanol", None)
    w.open_surface_builder()
    dlg = w.surface_dialog
    dlg.nx.setValue(4)
    dlg.ny.setValue(3)
    dlg.layers.setValue(1)
    dlg.move_btn.setChecked(True)
    n_system = len(dlg.system)
    w.mol_list.setCurrentRow(1)
    app.processEvents()
    assert len(w.viewer.atoms) == n_system and w.viewer.mode == "move"
    # a move that splits the molecule is flagged
    ads = list(range(n_system - len(mol), n_system))
    w.viewer.translate_atoms([ads[0]], [3.0, 0, 0])
    assert "Warning: a bond" in dlg.status.text()
    dlg.close()
    app.processEvents()
    assert len(w.viewer.atoms) != n_system                # back to the molecule
    _close(app, w)
