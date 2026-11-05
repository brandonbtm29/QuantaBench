"""SCALE-1: the virtualized molecule list, results read back from run folders, Open run folder and Resume."""

import os
from dataclasses import replace

import pytest

pytest.importorskip("PySide6")
needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                   reason="needs a display (run under xvfb-run on Linux)")

TFA = "OC(=O)C(F)(F)F"


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


def _cfg():
    from quantabench.chem import pipeline

    return pipeline.BuildConfig(method="mmff", max_conformers=3, refine_top=3, xtb_properties=False, threads=1)


@pytest.fixture(scope="module")
def template():
    """One real MMFF result, cloned under other ids for the synthetic run folders (as scratch/ does at 14,000)."""
    from quantabench.chem import pipeline
    from quantabench.chem.inputs import from_smiles

    res = pipeline.run_molecule(from_smiles(TFA, id="T"), _cfg())
    assert res.status == "ok" and len(res.conformers) >= 1
    return res


def _run_folder(path, template, n=30, pending=(5, 17), failed=(3,)):
    """A run folder in the engine's own format (job table, per-molecule folders, summary tables), written with
    the engine's writers: n molecules, some failed, some left pending."""
    from quantabench.chem import export, jobs
    from quantabench.chem.inputs import from_smiles

    inputs = [from_smiles(TFA, id=f"M{i:03d}", name=f"molecule {i}") for i in range(n)]
    with jobs.JobStore(path) as store:
        store.init_run(_cfg(), None)
        store.add_inputs(inputs)
        for i, inp in enumerate(inputs):
            if i in pending:
                continue
            res = replace(template, input=inp, files=[], descriptors=dict(template.descriptors))
            if i in failed:
                res = replace(res, status="failed", error="synthetic failure", conformers=[], mol=None)
                store.record(res, "")
                continue
            res.files = [str(p) for p in export.write_molecule(res, path / inp.id, export.DEFAULT_FORMATS)]
            store.record(res, inp.id)
    jobs.rebuild_tables(path)
    return inputs


def test_results_read_back_from_a_run_folder(tmp_path, template):
    from quantabench.chem import export, jobs

    run = tmp_path / "run"
    _run_folder(run, template)
    rows = jobs.molecule_rows(run)
    assert [r["id"] for r in rows][:3] == ["M000", "M001", "M002"] and len(rows) == 30
    ok = rows[0]
    assert ok["status"] == "ok" and ok["method"] == "mmff" and ok["n_conformers"] == len(template.conformers)
    assert ok["best_energy_eV"] == pytest.approx(template.conformers[0].energy_ev)
    assert ok["formula"] == template.descriptors["formula"] and ok["result_dir"] == "M000"
    assert rows[3]["status"] == "failed" and rows[3]["error"] == "synthetic failure"
    assert rows[5]["status"] == "pending" and rows[5]["best_energy_eV"] is None and rows[5]["name"] == "molecule 5"

    res = jobs.load_result(run, "M001")
    assert res.input.id == "M001" and res.input.name == "molecule 1" and res.status == "ok"
    assert [c.conf_id for c in res.conformers] == [c.conf_id for c in template.conformers]
    assert res.mol.GetNumConformers() == len(template.conformers) and res.mol.GetNumAtoms() == template.mol.GetNumAtoms()
    for c in res.conformers:      # coordinates of every conformer come back (SDF: 4 decimals)
        a = res.mol.GetConformer(c.conf_id).GetPositions()
        b = template.mol.GetConformer(c.conf_id).GetPositions()
        assert abs(a - b).max() < 1e-3
    assert res.descriptors == pytest.approx(template.descriptors) and res.method == "mmff"
    assert any(f.endswith("M001_summary.json") for f in res.files)
    failed = jobs.load_result(run, "M003")
    assert failed.status == "failed" and failed.error == "synthetic failure" and failed.mol is None
    assert [x.id for x in jobs.load_inputs(run, ["M004", "nope", "M002"])] == ["M004", "M002"]

    # without the SDF: the best conformer from the XYZ file
    (run / "M002" / "M002_conformers.sdf").unlink()
    res = export.read_molecule(run / "M002", "M002")
    assert len(res.conformers) == 1 and res.mol.GetNumConformers() == 1 and res.input.name == "molecule 2"


@needs_display
def test_model_sort_filter_and_batched_rows():
    from PySide6.QtCore import Qt

    from quantabench.gui.molecule_list import COL, MoleculeListView, MoleculeRow, MoleculeTableModel

    _app()
    model = MoleculeTableModel()
    view = MoleculeListView(model)
    inserts = []
    model.rowsInserted.connect(lambda *a: inserts.append(a))
    rows = [MoleculeRow(id=f"CAS-{i}", name=f"PFAS {i}", smiles="OC(=O)C(F)(F)F" if i % 2 else "CCO",
                        status="ok" if i % 3 else "failed", energy_ev=float(-i) if i % 3 else None,
                        n_conformers=2 if i % 3 else None, method="mmff" if i % 3 else "")
            for i in range(5000)]
    model.add_rows(rows)
    assert len(inserts) == 1 and view.count() == 5000            # one insert for the whole batch
    assert view.row_text(1) == "✔ CAS-1  mmff, 2 conf." and view.row_text(0) == "✖ CAS-0"

    view.sortByColumn(COL["energy"], Qt.SortOrder.AscendingOrder)
    assert view.row_at(0).id == "CAS-4999" and view.row_at(4999).energy_ev is None   # missing energies last
    view.setCurrentRow(0)
    view.sortByColumn(COL["seq"], Qt.SortOrder.AscendingOrder)   # the current molecule follows the sort
    assert view.current_id() == "CAS-4999" and view.row_at(0).id == "CAS-0"

    view.proxy.set_text("cas-12")
    assert view.count() == 111 and all(view.row_at(i).id.startswith("CAS-12") for i in range(view.count()))
    view.proxy.set_text("cco")                                     # SMILES, case-insensitive
    assert view.count() == 2500
    view.proxy.set_status("failed")
    assert view.count() == len([i for i in range(0, 5000, 2) if i % 3 == 0])
    view.proxy.set_text("")
    view.proxy.set_status("")
    assert view.count() == 5000

    model.update("CAS-2", stage="conformers")
    assert view.row_text(2) == "▶ CAS-2  (conformers)" and model.counts()["running"] == 1
    model.remove_ids(["CAS-1", "CAS-2", "CAS-3", "CAS-10"])
    assert view.count() == 4996 and model.row("CAS-4") is not None and model.position("CAS-4") == 1
    assert view.select_id("CAS-4") and view.current_id() == "CAS-4"


@needs_display
def test_result_store_keeps_a_few_and_pins_imports(tmp_path, template):
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.molecule_list import MoleculeRow, MoleculeTableModel, ResultStore

    _app()
    run = tmp_path / "run"
    _run_folder(run, template, n=20, pending=(), failed=(3,))
    model = MoleculeTableModel()
    from quantabench.gui.molecule_list import read_run_folder

    rows, status = read_run_folder(run)
    assert status.total == 20 and len(rows) == 20 and rows[0].folder.endswith("M000") and rows[0].input is None
    model.set_rows(rows)
    store = ResultStore(model, size=4)
    loaded = [store[f"M{i:03d}"] for i in (0, 1, 2, 4, 5, 6)]   # synchronous reads, LRU of 4
    assert all(r.mol is not None for r in loaded) and list(store.cache) == ["M002", "M004", "M005", "M006"]
    assert store["M006"] is loaded[-1]                             # cached: the same object
    assert store["M003"].status == "failed" and store["M003"].error == "synthetic failure"
    imported = replace(template, input=from_smiles(TFA, id="imported"))
    model.add_rows([MoleculeRow(id="imported", status="ok")])
    store.pin(imported)
    for i in (7, 8, 9, 10, 11):
        store[f"M{i:03d}"]
    assert store.cached("imported") is imported and len(store.cache) == 4   # pinned results are never evicted
    assert len(store) == 21 and "M019" in store and "nope" not in store
    with pytest.raises(KeyError):
        store["nope"]


@needs_display
def test_open_run_folder_select_search_and_resume(tmp_path, template, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from quantabench import settings as sm
    from quantabench.chem import jobs
    from quantabench.gui.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: pytest.fail(str(a[2]))))
    run = tmp_path / "run"
    _run_folder(run, template, n=30, pending=(5, 17), failed=(3,))
    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "out")))
    try:
        w.open_run_folder(folder=str(run))
        assert _wait(app, lambda: w.mol_list.count() == 30)
        assert w.run_folder == run and "30 molecules: 27 built, 1 failed, 2 not built" in w.count_label.text()
        assert w.mol_list.current_id() == "M000"
        assert _wait(app, lambda: w.current_result() is not None)              # read off the GUI thread
        res = w.current_result()
        assert res.input.id == "M000" and w.conf_table.rowCount() == len(template.conformers)
        assert "formula" in w.props_view.text() and w.viewer.atoms is not None
        assert len(w.viewer.atoms) == template.mol.GetNumAtoms()

        # any molecule: arrow through a few quickly, the last one wins
        for i in (10, 11, 12, 13):
            w.mol_list.setCurrentRow(i)
        assert _wait(app, lambda: w.current_result() is not None and w.current_result().input.id == "M013")
        assert len(w.results.cache) <= w.results.size
        w.mol_list.setCurrentRow(3)                                              # failed: from its row
        assert "synthetic failure" in w.mol_info.text() and w.current_result() is None

        w.search.setText("M01")
        assert _wait(app, lambda: w.mol_list.count() == 10)
        assert "(10 shown)" in w.count_label.text() or _wait(app, lambda: "(10 shown)" in w.count_label.text())
        w.search.clear()
        w.status_filter.setCurrentIndex(w.status_filter.findData("todo"))
        assert _wait(app, lambda: w.mol_list.count() == 2)
        assert {w.mol_list.row_at(i).id for i in range(2)} == {"M005", "M017"}
        w.status_filter.setCurrentIndex(0)
        assert _wait(app, lambda: w.mol_list.count() == 30)

        # Resume: the two pending molecules, with the folder's stored settings (MMFF)
        w.resume_run()
        assert w.worker is not None and w.model.row("M005").stage == "queued"
        assert _wait(app, lambda: w.worker is None, 240)
        st = jobs.open_run(run)
        assert st.remaining == 0 and st.counts["ok"] == 29
        assert w.model.row("M005").status == "ok" and w.model.row("M017").n_conformers
        assert not w.model.row("M005").stage
        assert _wait(app, lambda: "29 built" in w.count_label.text(), 5)
        w.mol_list.select_id("M017")
        assert _wait(app, lambda: w.current_result() is not None and w.current_result().input.id == "M017")
        assert w.current_result().method == "mmff"
    finally:
        _close(app, w)


@needs_display
def test_one_3d_render_per_selection(tmp_path, template):
    """Selecting a molecule rebuilds the 3D scene, clears the atom selection and lets panels add actors, each
    asking for a render; they share one (~65-150 ms each under software OpenGL). No idle re-renders either."""
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    run = tmp_path / "run"
    _run_folder(run, template, n=8, pending=(), failed=())
    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "out")))
    try:
        w.show()
        w.open_run_folder(folder=str(run))
        assert _wait(app, lambda: w.current_result() is not None)
        if w.viewer.plotter is None:
            pytest.skip(w.viewer.error)
        calls = []
        real = w.viewer.plotter.render
        w.viewer.plotter.render = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        _wait(app, lambda: False, 0.5)
        assert calls == []                                       # idle: nothing re-renders on a timer
        for i in (2, 5, 1, 6):
            before = len(calls)
            w.mol_list.setCurrentRow(i)
            ident = w.mol_list.current_id()
            assert _wait(app, lambda ident=ident: w.current_result() is not None
                         and w.current_result().input.id == ident)
            _wait(app, lambda: False, 0.3)
            assert len(calls) - before == 1, f"{len(calls) - before} renders for one selection"
            assert w.viewer.atoms is not None and len(w.viewer.atoms) == template.mol.GetNumAtoms()
        before = len(calls)
        w.conf_table.setCurrentCell(min(1, w.conf_table.rowCount() - 1), 0)   # another conformer: one more
        _wait(app, lambda: False, 0.3)
        assert len(calls) - before <= 1
    finally:
        _close(app, w)


@needs_display
def test_gui_batch_keeps_summary_rows_not_results(tmp_path, monkeypatch):
    """A batch's finished molecules become rows; only the selected molecule's full result stays in memory."""
    from PySide6.QtWidgets import QMessageBox

    from quantabench import settings as sm
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.main_window import MainWindow

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    try:
        w.add_inputs([from_smiles(s, id=f"m{i}") for i, s in enumerate(("CCO", "CCN", "CCC", TFA))])
        w.mol_list.setCurrentRow(1)
        w.w_method.setCurrentIndex(w.w_method.findData("mmff"))
        w.w_maxconf.setValue(2)
        w.w_xtbprops.setChecked(False)
        w.run_build()
        assert _wait(app, lambda: w.worker is None, 240)
        assert [w.model.row(f"m{i}").status for i in range(4)] == ["ok"] * 4
        assert list(w.results.cache) == ["m1"] and not w.results.pinned     # only the selected one
        assert w.current_result().input.id == "m1" and w.model.row("m3").folder
        assert w.model.row("m0").run_dir == str(w.last_run_dir)
        w.mol_list.setCurrentRow(3)                                          # read back from its folder
        assert _wait(app, lambda: w.current_result() is not None and w.current_result().input.id == "m3")
        assert w.results["m0"].mol is not None and len(w.results) == 4
        said = []
        monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: said.append(a[2])))
        w.resume_run()                                                        # nothing left: says so
        assert w.worker is None and "Every molecule" in said[0]
    finally:
        _close(app, w)
