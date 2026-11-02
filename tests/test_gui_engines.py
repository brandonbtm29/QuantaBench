"""GUI wiring of the external engines: pipeline pages (CREST, solvent, QM) and Settings > Engines."""

import os

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                reason="needs a display (run under xvfb-run on Linux)")


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


def test_pipeline_pages_map_onto_build_config_and_settings(tmp_path):
    from quantabench import settings as sm
    from quantabench.chem import engines
    from quantabench.gui.main_window import STAGES, MainWindow

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    assert "qm" in w.stage_items and [k for k, *_ in STAGES].index("qm") < [k for k, *_ in STAGES].index("export")
    methods = [w.w_method.itemData(i) for i in range(w.w_method.count())]
    assert methods == engines.method_keys()          # plugin engines appear without GUI changes
    assert {"xtb-gfn2", "tblite-gfn2", "mopac-pm7", "gfnff"} <= set(methods)

    w._set_combo(w.w_confsearch, "crest")
    assert w.w_crest_method.isEnabled() and w.w_crest_quick.isEnabled()
    w._set_combo(w.w_crest_method, "gfnff")
    w._set_combo(w.w_solvent, "water")
    w._set_combo(w.w_optimizer, "native")
    assert not w.w_qm_func.isEnabled()
    w.w_qm_on.setChecked(True)
    assert w.w_qm_func.isEnabled()
    w.w_qm_func.setCurrentText("B3LYP-D3BJ")
    w.w_qm_basis.setCurrentText("def2-TZVP")
    w._set_combo(w.w_qm_task, "optimize")
    w.w_qm_top.setValue(3)
    w.w_qm_freq.setChecked(True)
    w.w_qm_cubes.setChecked(True)
    cfg = w.current_config()
    assert (cfg.conformer_search, cfg.crest_method, cfg.crest_quick) == ("crest", "gfnff", True)
    assert cfg.solvent == "water" and cfg.optimizer == "native"
    assert cfg.qm_method == "b3lyp-d3bj/def2-tzvp" and cfg.qm_task == "optimize" and cfg.qm_top == 3
    assert cfg.qm_frequencies and cfg.qm_cubes and "spectrum" in cfg.formats
    # the persistent choices land in the build defaults and come back
    b = w.s.build
    assert (b.conformer_search, b.solvent, b.qm_method) == ("crest", "water", "b3lyp-d3bj/def2-tzvp")
    w.w_qm_on.setChecked(False)
    w._load_stage_values()
    assert w.w_qm_on.isChecked() and w.w_qm_func.currentText() == "b3lyp-d3bj"
    assert w.w_solvent.currentData() == "water" and w.w_confsearch.currentData() == "crest"
    w.w_qm_on.setChecked(False)
    assert w.current_config().qm_method == ""
    _close(app, w)


def test_settings_engines_tab_round_trip_and_detection(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QTabWidget

    from quantabench import settings as sm
    from quantabench.chem import external
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    s = sm.Settings()
    s.engines.paths = {"xtb": "~/bin/xtb"}
    d = SettingsDialog(s, None)
    tabs = d.findChild(QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Compute", "Build defaults", "Engines", "Folders"]
    assert d.program_paths["xtb"].text() == "~/bin/xtb"
    d.program_paths["xtb"].setText("")
    d.program_paths["crest"].setText("wsl:/home/u/miniforge3/envs/qb-tools/bin/crest")
    d.scratch.setText(str(tmp_path / "scratch"))
    d.keep_scratch.setChecked(True)
    d.stacksize.setText("2G")
    d.qm_memory.setValue(8)
    out = d.result_settings()
    assert out.engines.paths == {"crest": "wsl:/home/u/miniforge3/envs/qb-tools/bin/crest"}
    assert out.engines.scratch_dir == str(tmp_path / "scratch") and out.engines.keep_scratch
    assert out.engines.omp_stacksize == "2G" and out.engines.qm_memory_gb == 8

    # "Detect now" checks typed paths without touching the cache or the environment
    monkeypatch.setattr(external, "find_program", lambda name, **k: None)
    monkeypatch.setattr(external, "find_python_with", lambda *a, **k: None)
    env_before = dict(os.environ)
    d._detect_programs()
    assert _wait(app, lambda: "Searching" not in d.program_report.toPlainText())
    text = d.program_report.toPlainText()
    assert "xtb: not found" in text and "Psi4: not found" in text
    assert "crest: version ?" in text and "wsl:" in text          # a wsl: path is taken as given
    assert dict(os.environ) == env_before
    d.reject()
    d.deleteLater()
    app.processEvents()
