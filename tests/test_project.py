"""UI-2: project documents. The file format without Qt (round trip, relative paths, a moved project, versions),
then the window: save and reopen with built results and the Analyze table, results that were only in memory, the
unsaved-changes question, resuming a half-finished project, and the playback bar's Molecules mode."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import HAS_DISPLAY

from quantabench import project as proj

needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")
TFA = "OC(=O)C(F)(F)F"


# ============================================================================================ the format
def test_round_trip_relative_paths_and_a_moved_project(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from quantabench.analysis.sheet import FormulaSpec, JoinSpec, TableState
    from quantabench.chem import jobs
    from quantabench.chem.inputs import MoleculeInput, from_smiles

    root = proj.project_path(tmp_path / "study")
    assert root.name == "study.qbproj" and proj.project_path(root) == root
    inside = root / "runs" / "2026-11-13_batch_2"
    outside = tmp_path / "elsewhere" / "run"
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    structure = MoleculeInput(id="ethanol_3d", mol=m, source="ethanol.sdf")
    (root / "targets.csv").parent.mkdir(parents=True)
    (root / "targets.csv").write_text("id,sigma\nA,1\n", encoding="utf-8")
    table = TableState(columns={"tpsa": False}, formulas=[FormulaSpec("x2", "mol_weight*2")],
                       plot={"x": "mol_weight", "parity": True}, joins=[JoinSpec(str(root / "targets.csv"), "id")])
    data = proj.ProjectData(
        molecules=[{"id": "A", "smiles": "CCO", "status": "ok", "method": "mmff", "n_conformers": 2, "energy_ev": -1.5,
                    "folder": str(inside / "A"), "run_dir": str(inside), "seq": 0,
                    "fields": {"st25": np.float64(22.0), "bad": float("nan"), "n": np.int64(3), "note": "x"},
                    "input": jobs.input_to_json(from_smiles("CCO", id="A"))},
                   {"id": "B", "status": "pending", "run_dir": str(outside), "seq": 1},
                   {"id": "ethanol_3d", "status": "pending", "seq": 2, "input": jobs.input_to_json(structure)}],
        build={"refine_method": "mmff", "max_conformers": 7}, table=table.to_json(), workspace="analyze", current="B")
    proj.save(root, data)
    assert proj.is_project(root) and (root / "runs").is_dir()
    head = json.loads((root / proj.PROJECT_FILE).read_text(encoding="utf-8"))
    assert head["format"] == proj.FORMAT and head["version"] == proj.VERSION and head["n_molecules"] == 3
    assert json.loads(head["table"])["joins"][0]["path"] == "targets.csv"           # inside: relative
    lines = [json.loads(x) for x in (root / proj.MOLECULES_FILE).read_text(encoding="utf-8").splitlines()]
    assert lines[0]["folder"] == "runs/2026-11-13_batch_2/A" and lines[0]["run_dir"] == "runs/2026-11-13_batch_2"
    assert lines[0]["fields"] == {"st25": 22.0, "bad": None, "n": 3, "note": "x"}
    assert lines[1]["run_dir"] == str(outside)                                        # outside: absolute
    # moved (or synced to another machine): the paths inside follow the folder
    moved = tmp_path / "moved" / "study.qbproj"
    shutil.copytree(root, moved)
    back = proj.load(moved)
    assert back.name == "study" and back.workspace == "analyze" and back.current == "B"
    assert back.build == {"refine_method": "mmff", "max_conformers": 7}
    assert back.molecules[0]["folder"] == str(moved / "runs" / "2026-11-13_batch_2" / "A")
    assert back.molecules[1]["run_dir"] == str(outside)
    st = TableState.from_json(back.table)
    assert st.formulas == table.formulas and st.plot == table.plot and st.columns == table.columns
    assert st.joins[0].path == str(moved / "targets.csv")
    inp = jobs.input_from_json(back.molecules[2]["input"])
    assert inp.mol is not None and inp.mol.GetNumAtoms() == m.GetNumAtoms() and inp.source == "ethanol.sdf"
    assert proj.find_project(moved / "runs") == moved and proj.find_project(tmp_path) is None


def test_windows_paths_versions_and_damage(tmp_path):
    root = tmp_path / "p.qbproj"
    assert proj.from_stored("runs\\b\\A", root) == str(root / "runs" / "b" / "A")     # saved on Windows
    assert proj.from_stored("C:\\Users\\Brand\\run", root) == str(Path("C:\\Users\\Brand\\run"))
    assert proj.from_stored("", root) == "" and proj.from_stored(".", root) == str(root)
    proj.save(root, proj.ProjectData(name="p"))
    assert proj.load(root).molecules == []
    head = json.loads((root / proj.PROJECT_FILE).read_text(encoding="utf-8"))
    head["version"] = proj.VERSION + 1
    head["future_key"] = {"kept": True}
    (root / proj.PROJECT_FILE).write_text(json.dumps(head), encoding="utf-8")
    with pytest.raises(proj.ProjectError, match="newer QuantaBench"):
        proj.load(root)
    head["version"] = proj.VERSION
    (root / proj.PROJECT_FILE).write_text(json.dumps(head), encoding="utf-8")
    data = proj.load(root)
    assert data.extra == {"future_key": {"kept": True}}
    proj.save(root, data)                                     # a key of a newer version survives a save
    assert json.loads((root / proj.PROJECT_FILE).read_text(encoding="utf-8"))["future_key"] == {"kept": True}
    (root / proj.MOLECULES_FILE).write_text('{"id": "A"}\nnot json\n', encoding="utf-8")
    with pytest.raises(proj.ProjectError, match="line 2"):
        proj.load(root)
    with pytest.raises(proj.ProjectError, match="not a QuantaBench project"):
        proj.load(tmp_path)


# ============================================================================================ the window
def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    from PySide6.QtCore import QEvent

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        time.sleep(0.02)
    return cond()


def _window(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    pytest.importorskip("pyqtgraph")
    s = sm.Settings(output_dir=str(tmp_path / "runs"))
    s.build.refine_method = "mmff"
    s.build.max_conformers = 3
    w = MainWindow(s)
    w.w_xtbprops.setChecked(False)
    w.resize(1400, 900)
    w.show()
    _wait(_app(), lambda: False, 0.2)
    return w


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


@pytest.fixture
def answers(monkeypatch):
    """The unsaved-changes question answers from this list (Save / Discard / Cancel), and records each question."""
    from PySide6.QtWidgets import QMessageBox

    asked: list[str] = []
    queue: list = []

    def exec_(box):
        asked.append(box.text())
        return queue.pop(0) if queue else QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    return asked, queue


@needs_display
def test_save_and_reopen_a_built_project(tmp_path, answers):
    from PySide6.QtWidgets import QMessageBox

    from quantabench.chem.inputs import from_smiles

    asked, queue = answers
    app = _app()
    path = tmp_path / "docs" / "acids"
    w = _window(tmp_path)
    try:
        ctl = w.project
        assert not ctl.is_open and not ctl.is_dirty()
        assert ctl.save_as(path=path)                        # an empty project first: then builds go into it
        root = proj.project_path(path)
        assert ctl.path == root and "acids" in w.windowTitle() and not w.isWindowModified()
        w.add_inputs([from_smiles(TFA, id="TFA"), from_smiles("CCO", id="ethanol")])
        assert ctl.is_dirty() and w.isWindowModified()
        w.run_build()
        assert _wait(app, lambda: w.worker is None, 240)
        rows = {r.id: r for r in w.model.rows()}
        assert all(r.status == "ok" for r in rows.values())
        assert Path(rows["TFA"].run_dir).parent == root / "runs"               # the project's runs/ folder
        w.set_workspace("analyze")                          # the table refreshes while it is in view
        assert _wait(app, lambda: "mol_weight" in w.table_panel.model.aliases(), 5)
        w.table_panel.model.add_formula("mw2", "mol_weight*2")
        w.mol_list.select_id("ethanol")
        assert ctl.save() and not ctl.is_dirty()
        w.add_inputs([from_smiles("CCC", id="propane")])                        # not saved: discarded below
        queue.append(QMessageBox.StandardButton.Cancel)
        w.close()
        assert not w.closed and asked and "acids" in asked[-1] and "3 molecules" in asked[-1]
        queue.append(QMessageBox.StandardButton.Discard)
    finally:
        _close(app, w)
    w = _window(tmp_path)
    try:
        assert w.project.open_project(path=root / proj.PROJECT_FILE)            # the file or the folder both work
        assert [r.id for r in w.model.rows()] == ["TFA", "ethanol"]
        assert all(r.status == "ok" and r.n_conformers for r in w.model.rows())
        assert w.mol_list.current_id() == "ethanol" and w.workspaces.current == "analyze"
        assert not w.project.is_dirty() and w.windowTitle().endswith("acids[*]")
        model = w.table_panel.model
        assert [f.name for f in model.formulas()] == ["mw2"]
        assert _wait(app, lambda: not w.table_panel.reading() and np.isfinite(model.values("fx:mw2")).all(), 10)
        assert _wait(app, lambda: w.current_result() is not None)               # the result, from its folder
        assert w.current_result().input.id == "ethanol" and w.conf_table.rowCount() >= 1
        assert w.project.recent()[0] == str(root)
        # Close project: an empty, untitled window again
        assert w.project.close_project() and w.model.rowCount() == 0 and not w.project.is_open
    finally:
        _close(app, w)


@needs_display
def test_results_only_in_memory_are_written_into_the_project(tmp_path, answers):
    from quantabench.chem import pipeline
    from quantabench.chem.inputs import from_smiles

    app = _app()
    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, xtb_properties=False, protonate=False,
                               write_files=False, threads=1)
    res = pipeline.run_molecule(from_smiles(TFA, id="imported"), cfg)       # as an imported output: no folder
    w = _window(tmp_path)
    try:
        w.add_inputs([res.input], built=[res])
        assert w.model.row("imported").folder == "" and "imported" in w.results.pinned
        assert w.project.save_as(path=tmp_path / "mem")
        row = w.model.row("imported")
        assert Path(row.folder) == tmp_path / "mem.qbproj" / "results" / "imported"
        assert (Path(row.folder) / "imported_summary.json").is_file()
        assert not w.project.is_dirty()                     # writing the results is part of the save
    finally:
        _close(app, w)
    w = _window(tmp_path)
    try:
        assert w.project.open_project(path=tmp_path / "mem.qbproj")
        model = w.table_panel.model
        assert _wait(app, lambda: not w.table_panel.reading() and "mol_weight" in [c.key for c in model.columns()], 10)
        assert model.values("mol_weight")[0] == pytest.approx(res.descriptors["mol_weight"])
        assert _wait(app, lambda: w.current_result() is not None)
        assert w.current_result().input.id == "imported"
    finally:
        _close(app, w)


@needs_display
def test_resume_a_half_finished_project(tmp_path, answers):
    from quantabench.chem import export, jobs, pipeline
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui import project_actions as pa
    from quantabench.gui.molecule_list import read_run_folder

    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=2, xtb_properties=False, threads=1)
    root = proj.project_path(tmp_path / "half")
    run = root / "runs" / "2026-11-13_batch_3"
    inputs = [from_smiles(TFA, id=f"M{i}") for i in range(3)]
    done = pipeline.run_molecule(inputs[0], replace(cfg, write_files=False))
    with jobs.JobStore(run) as store:
        store.init_run(cfg, None)
        store.add_inputs(inputs)
        done.files = [str(p) for p in export.write_molecule(done, run / "M0", export.DEFAULT_FORMATS)]
        store.record(done, "M0")
    jobs.rebuild_tables(run)
    rows, _st = read_run_folder(run)
    proj.save(root, proj.ProjectData(molecules=[pa.row_to_dict(r) for r in rows]))
    # a resume from the command line after the save finished M1: the window picks that up when it opens
    with jobs.JobStore(run) as store:
        r1 = pipeline.run_molecule(inputs[1], replace(cfg, write_files=False))
        r1.files = [str(p) for p in export.write_molecule(r1, run / "M1", export.DEFAULT_FORMATS)]
        store.record(r1, "M1")
    app = _app()
    w = _window(tmp_path)
    try:
        assert w.project.open_project(path=root)
        assert _wait(app, lambda: w.model.row("M1").status == "ok", 10)          # news from the job table
        assert w.model.row("M2").status == "pending" and w.last_run_dir == run
        assert w.project.is_dirty()                                            # the list changed since the save
        w.resume_run()
        assert w.worker is not None
        assert _wait(app, lambda: w.worker is None, 240)
        assert w.model.row("M2").status == "ok" and jobs.open_run(run).remaining == 0
        assert w.project.save()
        assert [m["status"] for m in proj.load(root).molecules] == ["ok", "ok", "ok"]
    finally:
        _close(app, w)


@needs_display
def test_new_project_asks_and_the_playback_bar_steps_through_molecules(tmp_path, answers):
    from PySide6.QtWidgets import QMessageBox

    from quantabench.chem.inputs import from_smiles

    asked, queue = answers
    app = _app()
    w = _window(tmp_path)
    try:
        w.add_inputs([from_smiles(s, id=i) for i, s in (("a", "C"), ("b", "CC"), ("c", "CCC"))])
        bar = w.playback
        bar.set_mode("Molecule")
        assert bar.kind == "Molecule" and bar.count() == 3 and bar.index() == 2         # the last one added
        bar.first()
        assert w.mol_list.current_id() == "a" and "Molecule 1 of 3, a (not built)" in bar.label.text()
        bar.step(1)
        assert w.mol_list.current_id() == "b"
        w.mol_list.select_id("c")
        assert bar.index() == 2
        w.search.setText("b")
        assert _wait(app, lambda: bar.count() == 1, 5)
        w.search.clear()
        assert _wait(app, lambda: bar.count() == 3, 5)
        bar.set_mode("Conformer")
        assert bar.kind == "Conformer" and bar.count() == 0                        # not built: no conformers
        # New project on an untitled list with molecules asks; Cancel keeps them, Discard clears them
        queue.append(QMessageBox.StandardButton.Cancel)
        assert not w.project.new_project() and w.model.rowCount() == 3 and "untitled" in asked[-1]
        queue.append(QMessageBox.StandardButton.Discard)
        assert w.project.new_project() and w.model.rowCount() == 0 and not w.project.is_dirty()
        n = len(asked)
        w.add_inputs([from_smiles("C", id="d")])
    finally:
        _close(app, w)
    assert len(asked) == n            # closing an untitled window does not ask (as before projects existed)
