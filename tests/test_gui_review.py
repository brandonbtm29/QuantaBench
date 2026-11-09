"""The Review window (gui.review_panel, ADV-1): shown before a run and on demand, the batch profile with
click-through to the molecules, findings with Apply and Explain, presets, Run anyway / Cancel run."""

import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")

PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
AVAILABLE = {"mmff", "tblite-gfn2", "tblite-gfn1", "psi4"}


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


@pytest.fixture
def window(tmp_path):
    from quantabench import settings as sm
    from quantabench.chem import advisor
    from quantabench.chem.inputs import MoleculeInput
    from quantabench.gui.main_window import MainWindow

    app = _app()
    real = advisor.review
    with pytest.MonkeyPatch.context() as mp:   # the installed engines of the test machine do not matter here
        mp.setattr(advisor, "review", lambda inputs, cfg, **kw: real(inputs, cfg, **{**kw, "available": AVAILABLE}))
        w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
        w.add_inputs([MoleculeInput(id="PFOA", smiles=PFOA), MoleculeInput(id="octane", smiles="CCCCCCCC")])
        w._set_combo(w.w_solvent, "water")
        w._set_combo(w.w_method, "tblite-gfn2")
        started = []
        mp.setattr(w, "run_build", lambda selected_only=False: started.append(selected_only))
        w.started_runs = started
        yield w
        dlg = getattr(w, "review_dialog", None)
        if dlg is not None:
            _wait(app, lambda: not dlg.running_review)
        w.close()
        assert _wait(app, lambda: w.closed)


def _reviewed(w):
    app = _app()
    dlg = w.review_dialog
    assert _wait(app, lambda: dlg.review is not None and not dlg.running_review)
    return dlg


def test_run_shows_the_review_first_and_run_anyway_starts_it(window):
    w = window
    w.run_btn.click()                                # "Run all": the review, not the run
    assert w.started_runs == []
    dlg = _reviewed(w)
    assert dlg.isVisible() and dlg.before_run
    assert dlg.title.text().startswith("Review of 2 molecules: water phase")
    assert dlg.run_btn.isVisible() and dlg.run_btn.text() == "Run anyway"     # a problem: GFN2 ranks PFAS ions
    assert "1 problems" in dlg.status.text()
    keys = [dlg.findings_tree.topLevelItem(i).data(0, 256) for i in range(dlg.findings_tree.topLevelItemCount())]
    assert keys[0] == "rank.gfn2.pfas_ion"
    assert "rank.gfn2.pfas_ion" in dlg.apply_buttons and "rank.gfn2.pfas_ion" in dlg.explain_buttons
    assert dlg.findings_tree.topLevelItem(0).icon(0).isNull() is False
    assert "Why:" in dlg.detail.toPlainText() and "PFOA" in dlg.detail.toPlainText()
    dlg.run_btn.click()
    assert w.started_runs == [False] and not dlg.isVisible()


def test_apply_changes_the_pipeline_and_reviews_again(window):
    w = window
    dlg = __import__("quantabench.gui.review_panel", fromlist=["show_review"]).show_review(w)
    dlg = _reviewed(w)
    assert not dlg.before_run and not dlg.run_btn.isVisible()                 # on demand: no Run button
    dlg.apply_buttons["rank.gfn2.pfas_ion"].click()
    assert w.w_qm_on.isChecked() and w.qm_level() == "wb97x-d/def2-svpd"
    assert w.w_qm_top.value() == 8 and w.w_window.stored() == 15.0 and w.w_refine_top.value() == 20
    dlg = _reviewed(w)
    assert dlg.review.counts()["problem"] == 0                                 # the DFT re-rank is configured
    cfg = w.current_config()
    assert cfg.qm_method == "wb97x-d/def2-svpd" and cfg.solvent == "water"


def test_explain_opens_the_methods_guide_and_ids_select_molecules(window):
    from PySide6.QtCore import QUrl

    w = window
    w.request_run(False)
    dlg = _reviewed(w)
    dlg.explain_buttons["rank.gfn2.pfas_ion"].click()
    assert w.methods_guide.isVisible() and w.methods_guide.current_key == "conformer-ranking"
    dlg._on_link(QUrl("guide:def2-svpd"))
    assert w.methods_guide.current_key == "def2-svpd"
    # the profile: Charge class > anion > PFOA; double-clicking the id selects it in the Molecules list
    w.mol_list.select_id("octane")
    charge = next(dlg.profile_tree.topLevelItem(i) for i in range(dlg.profile_tree.topLevelItemCount())
                  if dlg.profile_tree.topLevelItem(i).text(0).startswith("Charge class"))
    anion = next(charge.child(i) for i in range(charge.childCount()) if charge.child(i).text(0) == "anion")
    assert anion.text(1) == "1" and anion.child(0).text(0) == "PFOA"
    dlg._on_profile_item(anion.child(0))
    assert w.mol_list.current_id() == "PFOA"
    dlg._on_link(QUrl("mol:octane"))
    assert w.mol_list.current_id() == "octane"
    dlg.cancel_run()
    assert w.started_runs == [] and not dlg.isVisible()


def test_presets_and_the_before_run_switch(window):
    w = window
    w.mol_list.select_id("octane")
    w.request_run(True)                                 # Run selected: the review covers the selection only
    dlg = _reviewed(w)
    assert dlg.review.n == 1 and dlg.review.profile.group("flags", "hydrocarbon").ids == ["octane"]
    assert not dlg.review.preset("recommended").config.qm_method      # hydrocarbons: no DFT re-rank
    dlg.start(selected_only=False)
    dlg = _reviewed(w)
    assert dlg.review.n == 2
    assert all(b.isEnabled() for b in dlg.preset_buttons.values())
    assert "Estimated" in dlg.preset_buttons["recommended"].toolTip()
    dlg.preset_buttons["fast"].click()
    assert w.w_refine_top.value() == 10 and w.w_maxconf.value() == 30 and not w.w_qm_on.isChecked()
    dlg = _reviewed(w)
    dlg.preset_buttons["recommended"].click()
    assert w.qm_level() == "wb97x-d/def2-svpd" and w.w_qm_top.value() == 8
    dlg = _reviewed(w)
    assert dlg.run_btn.text() in ("Run", "Run anyway")
    dlg.before_box.setChecked(False)                    # off: Run goes straight to the build
    assert not w.a_review_before.isChecked()
    dlg.close()
    w.run_sel_btn.click()
    assert w.started_runs == [True]
    w.a_review_before.setChecked(True)
    w.a_run.trigger()
    assert w.started_runs == [True]                     # the review again
    _reviewed(w)


def test_review_button_and_menu_are_installed(window):
    w = window
    assert w.review_btn.text() == "Review..." and w.a_review.shortcut().toString() == "F6"
    build = next(a.menu() for a in w.menuBar().actions() if a.menu() is not None and a.text().replace("&", "") == "Build")
    texts = [a.text() for a in build.actions()]
    assert texts[:2] == ["Review molecules...", "Review before each run"]
    w.review_btn.click()
    dlg = _reviewed(w)
    assert not dlg.before_run


def test_apply_changes_sets_every_field(window):
    from quantabench.gui.review_panel import apply_changes

    w = window
    skipped = apply_changes(w, {"method": "tblite-gfn1", "solvent": "", "conformer_search": "crest",
                                "crest_method": "gfnff", "crest_quick": False, "max_conformers": 77, "refine_top": 9,
                                "energy_window_kcal": 12.5, "qm_method": "b3lyp-d3bj/def2-svpd", "qm_top": 4,
                                "qm_task": "optimize", "keep_largest_fragment": False, "protonate": False, "ph": 5.0,
                                "frequencies": True, "not_a_field": 1})
    assert skipped == ["not_a_field"]
    cfg = w.current_config()
    assert (cfg.method, cfg.solvent, cfg.conformer_search, cfg.crest_method, cfg.crest_quick) == (
        "tblite-gfn1", "", "crest", "gfnff", False)
    assert (cfg.max_conformers, cfg.refine_top, cfg.energy_window_kcal) == (77, 9, 12.5)
    assert (cfg.qm_method, cfg.qm_top, cfg.qm_task) == ("b3lyp-d3bj/def2-svpd", 4, "optimize")
    assert (cfg.keep_largest_fragment, cfg.protonate, cfg.ph, cfg.frequencies) == (False, False, 5.0, True)
    apply_changes(w, {"qm_method": ""})
    assert w.current_config().qm_method == ""
