"""UI-5: the job dialog, the job queue and the Jobs dock. The dialog's recipes, task and fields give the job's
configuration; jobs run one at a time in the order submitted, a queued job can be cancelled or moved, the running one
cancelled; Run all goes through the queue too."""

import time
from dataclasses import replace

import pytest
from conftest import HAS_DISPLAY

pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")

MOLS = (("ethanol", "CCO"), ("hexane", "CCCCCC"), ("propanol", "CCCO"), ("butanone", "CCC(=O)C"),
        ("toluene", "Cc1ccccc1"), ("pentanol", "CCCCCO"), ("anisole", "COc1ccccc1"), ("octane", "CCCCCCCC"))


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=60.0):
    from PySide6.QtCore import QEvent

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        time.sleep(0.02)
    return cond()


def _window(tmp_path, n=2):
    from quantabench import settings as sm
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.main_window import MainWindow

    s = sm.Settings(output_dir=str(tmp_path / "runs"))
    s.build.refine_method = "mmff"
    s.build.max_conformers = 3
    w = MainWindow(s)
    w.resize(1500, 920)
    w.show()
    w.set_workspace("calculate")
    w.add_inputs([from_smiles(smi, id=i) for i, smi in MOLS[:n]])
    _wait(_app(), lambda: False, 0.3)
    return w


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed, 30)


def _dialog(w, **kw):
    from quantabench.gui.job_dialog import JobDialog

    return JobDialog(w, available={"mmff", "tblite-gfn2"}, programs=set(), **kw)


def _mmff_job(w, ids, name, max_confs=2):
    from quantabench.chem import recipes
    from quantabench.gui.job_dialog import submit_spec

    cfg = replace(w.current_config(), method="mmff", max_conformers=max_confs, refine_top=1, xtb_properties=False)
    return submit_spec(w, recipes.JobSpec(name=name, cfg=cfg), ids)


def test_the_dialog_makes_the_jobs_configuration(tmp_path):
    from quantabench.chem import recipes

    app = _app()
    w = _window(tmp_path)
    try:
        assert w.jobs_dock.isVisible() and w.new_job_btn.text() == "New job..."
        assert w.a_new_job.shortcut().toString() == "Ctrl+J"
        d = _dialog(w)
        # filled from the Pipeline dock; the dock is not changed by the dialog
        assert d.w_recipe.currentData() == recipes.CUSTOM and d.config().method == w.current_config().method
        names = [d.w_recipe.itemText(i) for i in range(d.w_recipe.count())]
        assert names == ["The Pipeline dock's settings", "Screen", "Standard", "Careful"]
        # a recipe sets the fields (the gfn2 alias resolved here), the task its own on top
        d.w_recipe.setCurrentIndex(d.w_recipe.findData("screen"))
        cfg = d.spec().config()
        assert (cfg.method, cfg.max_conformers, cfg.refine_top, cfg.rank_method) == ("tblite-gfn2", 30, 10, "auto")
        assert not d.recipe_modified() and "Steps:" in d.recipe_info.text()
        d.w_refine_top.setValue(5)
        assert d.recipe_modified() and "Changed from the recipe" in d.recipe_info.text()
        d.w_task.setCurrentIndex(d.w_task.findData("single_point"))
        sp = d.spec().config()
        assert (sp.max_conformers, sp.max_steps, sp.method) == (1, 0, "tblite-gfn2") and not d.conf_box.isEnabled()
        d.w_task.setCurrentIndex(d.w_task.findData("frequencies"))
        assert d.spec().config().frequencies and "spectrum" in d.spec().cfg.formats
        scan = d.w_task.model().item(d.w_task.findData("scan"))
        assert not scan.isEnabled()
        # Careful needs CREST and Psi4: said in the dialog; without Psi4 the DFT box cannot be set
        d.w_recipe.setCurrentIndex(d.w_recipe.findData("careful"))
        assert "Needs crest, psi4" in d.recipe_info.text() and not d.w_qm_on.isEnabled()
        assert d.spec().config().conformer_search == "auto" and d.spec().config().crest_with_rdkit
        # engines that are not installed are listed, disabled; family headers are not choices
        model = d.w_method.model()
        heads = [d.w_method.itemText(i) for i in range(d.w_method.count()) if d.w_method.itemData(i) is None]
        assert "— Force field —" in heads and all(not model.item(d.w_method.findText(h)).isEnabled() for h in heads)
        # charge, multiplicity, resources
        d.w_charge.setValue(-1)
        d.w_mult.setValue(2)
        d.w_threads.setValue(3)
        d.w_qm_mem.setValue(14)
        d.w_timeout.setValue(1200)
        spec = d.spec()
        assert (spec.charge, spec.multiplicity) == (-1, 2)
        assert (spec.cfg.threads, spec.cfg.qm_memory_gb, spec.cfg.molecule_timeout_s) == (3, 14.0, 1200.0)
        d.w_charge.setValue(d.w_charge.minimum())
        d.w_mult.setValue(0)
        assert d.spec().charge is None and d.spec().multiplicity is None
        assert d.w_run_on.currentData() == recipes.LOCAL and d.submit_btn.isEnabled()
        # back to the Pipeline dock's settings
        d.w_recipe.setCurrentIndex(0)
        assert d.config().conformer_search == w.current_config().conformer_search
        d.close()
    finally:
        _close(app, w)


def test_save_and_delete_a_recipe(tmp_path):
    from quantabench import settings as sm
    from quantabench.chem import recipes

    app = _app()
    w = _window(tmp_path)
    try:
        d = _dialog(w)
        d.w_recipe.setCurrentIndex(d.w_recipe.findData("screen"))
        d.w_refine_top.setValue(4)
        d.w_task.setCurrentIndex(d.w_task.findData("optimize"))
        r = d.save_recipe("My screen")
        assert r is not None and d.w_recipe.currentData() == "user:My screen"
        assert "My screen (yours)" in [d.w_recipe.itemText(i) for i in range(d.w_recipe.count())]
        saved = recipes.user_recipes(sm.load())                      # written to the settings file
        assert [x.name for x in saved] == ["My screen"] and saved[0].task == "optimize"
        assert saved[0].changes["refine_top"] == 4 and saved[0].changes["method"] == "tblite-gfn2"
        assert d.w_recipe_del.isEnabled() and d.delete_recipe()
        assert recipes.user_recipes(sm.load()) == [] and d.w_recipe.currentData() == recipes.CUSTOM
        d.close()
    finally:
        _close(app, w)


def test_jobs_run_in_order_and_a_queued_one_can_be_cancelled(tmp_path):
    app = _app()
    w = _window(tmp_path, n=4)
    try:
        q = w.job_queue
        j1 = _mmff_job(w, ["ethanol", "hexane"], "first")
        j2 = _mmff_job(w, ["propanol"], "second")
        j3 = _mmff_job(w, ["butanone"], "third")
        assert j1.status == "running" and j2.status == "queued" and j3.status == "queued"
        assert [j.number for j in q.queued()] == [j2.number, j3.number]
        mon = w.job_monitor
        assert mon.tree.topLevelItemCount() == 3
        mon.select(j3.number)
        assert mon.up_btn.isEnabled() and not mon.down_btn.isEnabled()
        assert mon.move_selected(-1) and [j.number for j in q.queued()] == [j3.number, j2.number]
        assert q.move(j3.number, 1)                                  # back
        assert mon.cancel_selected() and j3.status == "cancelled"   # queued: it never starts
        assert _wait(app, lambda: j2.status in ("done", "failed", "cancelled"), 120)
        assert (j1.status, j2.status, j3.status) == ("done", "done", "cancelled")
        assert j1.started <= j1.finished <= j2.started and j3.started == 0
        assert (j1.ok, j1.failed, j2.ok) == (2, 0, 1) and j1.progress_text() == "2 / 2"
        assert j1.output_dir is not None and j1.output_dir.is_dir()   # a batch: its run folder, made at its start
        assert any("ethanol" in line for line in j1.log)
        assert w.model.row("propanol").status == "ok"
        mon.select(j1.number)
        assert "first" in mon.detail.text() and mon.open_btn.isEnabled()
        assert "ethanol" in mon.log.toPlainText()
        assert mon.clear_btn.isEnabled() and q.clear_finished() == 3 and mon.tree.topLevelItemCount() == 0
    finally:
        _close(app, w)


def test_cancelling_the_running_job_starts_the_next(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    app = _app()
    w = _window(tmp_path, n=8)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    try:
        ids = [i for i, _ in MOLS]
        big = _mmff_job(w, ids[1:], "big", max_confs=20)
        nxt = _mmff_job(w, ids[:1], "next")
        assert _wait(app, lambda: big.done >= 1 or big.status != "running", 120)
        w.job_monitor.select(big.number)
        if big.status == "running":                                  # (a fast machine may have finished it)
            assert w.job_monitor.cancel_selected() and big.status == "cancelling"
        assert _wait(app, lambda: nxt.status in ("done", "failed", "cancelled"), 120)
        assert big.status in ("cancelled", "done") and nxt.status == "done"
        assert big.finished <= nxt.started
        if big.status == "cancelled":                                # some molecules left undone (in one lane;
            assert big.ok + big.failed < len(ids) - 1                # with lanes the ones in flight still finish)
        else:
            assert big.ok + big.failed == len(ids) - 1
    finally:
        _close(app, w)


def test_run_all_goes_through_the_queue_and_the_usage_line(tmp_path):
    from quantabench.gui.job_monitor import usage_html

    app = _app()
    w = _window(tmp_path)
    try:
        assert "No job running" in usage_html(w, None)
        w.run_build()
        job = w.job_queue.running
        assert job is not None and job.name == "Run all: 2 molecules"
        assert "thread" in usage_html(w, job) and f"Job {job.number}" in usage_html(w, job)
        assert _wait(app, lambda: job.status == "done", 120)
        assert w.worker is None and job.ok == 2
    finally:
        _close(app, w)


def test_the_review_page_and_a_preset_set_the_dialogs_fields(tmp_path):
    from quantabench.gui.job_dialog import REVIEW_TAB

    app = _app()
    w = _window(tmp_path)
    try:
        d = _dialog(w)
        d.show()
        d.tabs.setCurrentIndex(REVIEW_TAB)                          # opening the page starts the review
        assert _wait(app, lambda: d.review is not None, 120)
        assert d.review.n == 2 and "Review of 2 molecules" in d.review_title.text()
        enabled = [k for k, b in d.preset_buttons.items() if b.isEnabled()]
        assert enabled == ["fast", "recommended", "careful"]
        p = d.review.preset("fast")
        d.apply_preset("fast")
        cfg = d.config()
        assert all(getattr(cfg, k) == v for k, v in p.changes.items() if k in ("max_conformers", "refine_top",
                                                                                 "rank_method", "conformer_search"))
        assert _wait(app, lambda: d.review is not None, 120)        # reviewed again for the new settings
        job = d.submit()
        assert job is not None and job.status in ("running", "queued") and not d.isVisible()
        assert _wait(app, lambda: job.status in ("done", "failed"), 120)
    finally:
        _close(app, w)


def test_conformer_search_auto_in_the_pipeline_dock_and_the_dialog(tmp_path):
    """CONF-AUTO in the GUI: the Pipeline dock's Conformer search page and the job dialog offer "auto" with its
    threshold and the merge box; the review's Apply sets them."""
    from quantabench.gui.review_panel import apply_changes

    app = _app()
    w = _window(tmp_path)
    try:
        w.w_confsearch.setCurrentIndex(w.w_confsearch.findData("auto"))
        assert w.w_crest_min_rotors.isEnabled() and w.w_crest_with_rdkit.isEnabled() and w.w_crest_method.isEnabled()
        w.w_crest_min_rotors.setValue(7)
        w.w_crest_with_rdkit.setChecked(True)
        cfg = w.current_config()
        assert (cfg.conformer_search, cfg.crest_min_rotors, cfg.crest_with_rdkit) == ("auto", 7, True)
        assert w.s.build.crest_min_rotors == 7 and w.s.build.crest_with_rdkit
        assert w.stage_summary("conformers").startswith("Auto: CREST") and "from 7 rotors" in w.stage_summary("conformers")
        w.w_confsearch.setCurrentIndex(w.w_confsearch.findData("crest"))
        assert not w.w_crest_min_rotors.isEnabled() and w.w_crest_with_rdkit.isEnabled()
        w.w_confsearch.setCurrentIndex(w.w_confsearch.findData("rdkit"))
        assert not w.w_crest_with_rdkit.isEnabled()
        assert apply_changes(w, {"conformer_search": "auto", "crest_min_rotors": 10, "crest_with_rdkit": False}) == []
        cfg = w.current_config()
        assert (cfg.conformer_search, cfg.crest_min_rotors, cfg.crest_with_rdkit) == ("auto", 10, False)
        d = _dialog(w)
        assert d.config().conformer_search == "auto" and d.w_crest_min_rotors.value() == 10   # from the dock
        d.w_crest_with_rdkit.setChecked(True)
        d.w_crest_min_rotors.setValue(12)
        spec = d.spec().config()
        assert (spec.conformer_search, spec.crest_min_rotors, spec.crest_with_rdkit) == ("auto", 12, True)
        assert "CREST from 12 rotatable bonds" in d.spec().summary()
        d.w_search.setCurrentIndex(d.w_search.findData("rdkit"))
        assert not d.w_crest_min_rotors.isEnabled() and not d.w_crest_with_rdkit.isEnabled()
        d.close()
    finally:
        _close(app, w)
