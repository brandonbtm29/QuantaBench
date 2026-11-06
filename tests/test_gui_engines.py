"""GUI wiring of the external engines: pipeline pages (CREST, solvent, QM) and Settings > Engines."""

import os

import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY,
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
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Compute", "Build defaults", "Surfaces && radii",
                                                                 "Display", "Engines", "Folders"]
    assert d.program_paths["xtb"].text() == "~/bin/xtb"
    d.program_paths["xtb"].setText("")
    d.program_paths["crest"].setText("wsl:/home/u/miniforge3/envs/qb-tools/bin/crest")
    d.scratch.setText(str(tmp_path / "scratch"))
    d.keep_scratch.setChecked(True)
    d.stacksize.setText("2G")
    d.qm_memory.setValue(8)
    out = d.result_settings()
    # the shared default stays for the other machines; this machine gets its own paths ("" = automatic here)
    me = sm.this_machine()
    assert out.engines.paths == {"xtb": "~/bin/xtb"}
    assert out.engines.machine_paths == {me: {"xtb": "", "crest": "wsl:/home/u/miniforge3/envs/qb-tools/bin/crest"}}
    assert out.engine_paths_for() == {"crest": ("wsl:/home/u/miniforge3/envs/qb-tools/bin/crest", "this machine")}
    assert out.engine_paths_for("another-machine") == {"xtb": ("~/bin/xtb", "all machines")}
    assert out.engines.scratch_dir == str(tmp_path / "scratch") and out.engines.keep_scratch
    assert out.engines.omp_stacksize == "2G" and out.engines.qm_memory_gb == 8

    # "Detect now" checks typed paths without touching the cache or the environment
    monkeypatch.setattr(external, "find_program", lambda name, **k: None)
    monkeypatch.setattr(external, "find_python_with", lambda *a, **k: None)
    monkeypatch.setattr(external, "_probe_version", lambda *a, **k: "")   # never start a real WSL from a test
    env_before = dict(os.environ)
    d._detect_programs()
    assert _wait(app, lambda: "Searching" not in d.program_report.toPlainText())
    text = d.program_report.toPlainText()
    assert "xtb: not found" in text and "Psi4: not found" in text
    assert "crest: version ?" in text and "wsl:" in text          # a wsl: path is taken as given
    assert dict(os.environ) == env_before
    assert _wait(app, lambda: not d.__dict__.get("_live_workers"))   # the detection thread has ended
    d.reject()
    d.deleteLater()
    app.processEvents()


_STARTUP_CHECK = r"""
import json, sys, time
from PySide6.QtWidgets import QApplication
app = QApplication([])
from quantabench import settings as sm
from quantabench.gui.main_window import MainWindow
w = MainWindow(sm.Settings(output_dir=sys.argv[1]))
w.show()
t = time.time()
while time.time() - t < 1.5:              # the hardware probe starts at 50 ms (in a child process)
    app.processEvents()
    time.sleep(0.02)
labels = [w.w_method.itemText(i) for i in range(w.w_method.count())]
heavy = [m for m in ("torch", "dxtb", "aimnet", "ovito", "tad_libcint", "tad_mctc") if m in sys.modules]
w.close()                                 # kills the probe's child if it still runs
while not w.closed and time.time() - t < 30:
    app.processEvents()
    time.sleep(0.02)
print("RESULT " + json.dumps({"heavy": heavy, "labels": labels, "closed": w.closed}))
"""


def test_main_window_starts_without_torch(tmp_path):
    """SCALE-1: the window imports no torch, dxtb, aimnet or ovito (~500 MB) just to label engines and GPUs.
    Checked in a fresh process: this one may have imported torch already."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    import quantabench
    from quantabench.chem import engines
    from quantabench.gui.settings_dialog import method_label

    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(quantabench.__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]))
    cp = subprocess.run([sys.executable, "-c", _STARTUP_CHECK, str(tmp_path / "out")], capture_output=True,
                        text=True, timeout=240, env=env)
    lines = [ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")]
    assert cp.returncode == 0 and lines, cp.stderr[-3000:]
    out = json.loads(lines[-1][len("RESULT "):])
    assert out["heavy"] == [] and out["closed"]
    # and the labels still say which engines are usable (here from the full check; this process may import torch)
    engines.libcint_status()
    assert out["labels"] == [method_label(k) for k in engines.method_keys()]


def test_gfn2_label_follows_the_full_libcint_check(tmp_path, monkeypatch):
    """The version-based guess labels GFN2 at start-up; the probe's full check replaces it, with the reason."""
    from quantabench import resources
    from quantabench import settings as sm
    from quantabench.chem import engines
    from quantabench.gui.main_window import MainWindow

    if not (engines.has_package("torch") and engines.has_package("dxtb")):
        pytest.skip("GFN2 (dxtb) not installed")
    monkeypatch.setattr(engines, "_LIBCINT_STATUS", None)
    monkeypatch.setattr(engines, "_LIBCINT_PROBED", None)
    monkeypatch.setattr(engines, "libcint_status_from_versions", lambda: (True, ""))
    probe = (resources.detect_light(), {"libcint": [False, "tad-libcint 9.9 does not load (synthetic)"]})
    monkeypatch.setattr(resources, "detect_isolated", lambda **_k: probe)
    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    try:
        i = w.w_method.findData("gfn2-xtb")
        assert w.w_method.itemText(i) == engines.ENGINES["gfn2-xtb"].label        # the guess: usable
        assert _wait(app, lambda: w.report is not None)
        assert w.w_method.itemText(i).endswith("(not installed)")
        w.w_method.setCurrentIndex(i)
        assert "Not available: GFN2 needs dxtb's libcint backend: tad-libcint 9.9 does not load" in w.engine_info.text()
    finally:
        _close(app, w)
