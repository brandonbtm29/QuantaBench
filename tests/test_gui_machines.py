"""Machines dock (DIST-3, minimal) and the GPU policy settings of the Compute tab."""

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


def test_machines_dock_lists_registry_and_checks_workers(tmp_path, monkeypatch):
    from quantabench import settings as sm
    from quantabench.dist import client, profile
    from quantabench.gui.main_window import MainWindow

    here = profile.machine_name()
    assert here not in ("nova-test", "mac-test")
    profs = [profile.MachineProfile(name="nova-test", os="Windows 11", cpu="Ryzen 9 7900X", physical_cores=12,
                                    logical_cores=24, ram_gb=32, gpus=[{"name": "RX 9060 XT"}],
                                    throughput={"mmff@cpu": 9000.0, "aimnet2@cuda:0": 4000.0},
                                    address="http://100.101.1.2:8765"),
             profile.MachineProfile(name=here, physical_cores=4, logical_cores=8)]
    monkeypatch.setattr(profile, "load_profiles", lambda *a, **k: list(profs))
    nova = client.WorkerRef(name="nova-test", url="http://100.101.1.2:8765", source="registry",
                            info={"queue": {"running": ["j1"], "queued": ["j2"]},
                                  "quantabench_version": "0.0.0-other"})
    mac = client.WorkerRef(name="mac-test", url="http://100.101.1.3:8765", source="settings", error="timed out")
    monkeypatch.setattr(client, "discover", lambda *a, **k: {"nova-test": nova, "mac-test": mac})

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    panel = w.machines_panel
    assert w.machines_dock.windowTitle() == "Machines"
    t = panel.table

    def rows():
        return {t.item(r, 0).text(): [t.item(r, c).text() for c in range(t.columnCount())]
                for r in range(t.rowCount())}

    got = rows()
    assert got[f"{here}  (this computer)"][1] == "this computer"
    assert got["nova-test"][1] == "worker (not checked)"
    assert "RX 9060 XT" in got["nova-test"][2] and got["nova-test"][3].startswith("mmff@cpu 9,000")
    assert "Parallel lanes" in panel.here.text()

    panel.check_workers()
    assert _wait(app, lambda: not panel._busy)
    got = rows()
    assert got["nova-test"][1].startswith("busy, 1 queued, version 0.0.0-other")
    assert got["mac-test"][1] == "unreachable"             # known from settings only: listed after the check
    w.close()
    assert _wait(app, lambda: w.closed)


def test_compute_tab_gpu_policies_round_trip():
    from quantabench import settings as sm
    from quantabench.chem.pipeline import BuildConfig
    from quantabench.gui.settings_dialog import SettingsDialog

    _app()
    s = sm.Settings()
    s.resources.dxtb_gpu, s.resources.linalg = "on", "cpu"
    d = SettingsDialog(s, None)
    assert d.dxtb_gpu.currentText() == "on" and d.linalg.currentText() == "cpu"
    d.dxtb_gpu.setCurrentText("off")
    d.linalg.setCurrentText("device")
    out = d.result_settings()
    assert (out.resources.dxtb_gpu, out.resources.linalg) == ("off", "device")
    cfg = BuildConfig.from_settings(out)
    assert (cfg.dxtb_gpu, cfg.linalg) == ("off", "device")
    d.reject()
    d.deleteLater()


def test_compute_tab_memory_budget_and_time_limit_round_trip():
    from quantabench import settings as sm
    from quantabench.chem.pipeline import BuildConfig
    from quantabench.gui.settings_dialog import SettingsDialog

    _app()
    s = sm.Settings()
    d = SettingsDialog(s, None)
    assert d.memory_budget.value() == 0.0 and d.molecule_timeout.value() == 1800          # auto, 30 min
    assert "0 = auto" in d.memory_budget.toolTip() and "Off" in d.molecule_timeout.toolTip()
    d.memory_budget.setValue(3.5)
    d.molecule_timeout.setValue(600)
    out = d.result_settings()
    assert (out.resources.memory_budget_gb, out.resources.molecule_timeout_s) == (3.5, 600.0)
    cfg = BuildConfig.from_settings(out)
    assert (cfg.memory_budget_gb, cfg.molecule_timeout_s) == (3.5, 600.0)
    d.reject()
    d.deleteLater()
    # off: below 0 GB is no guard, 0 s is no limit; stored values come back into the dialog
    s.resources.memory_budget_gb, s.resources.molecule_timeout_s = -1.0, 0.0
    d = SettingsDialog(s, None)
    assert d.memory_budget.text() == "Off (no memory guard)" and d.molecule_timeout.text() == "Off (no time limit)"
    out = d.result_settings()
    assert (out.resources.memory_budget_gb, out.resources.molecule_timeout_s) == (-1.0, 0.0)
    d.reject()
    d.deleteLater()


def test_compute_tab_speed_aware_lanes_round_trip():
    from quantabench import settings as sm
    from quantabench.gui.settings_dialog import SettingsDialog

    _app()
    s = sm.Settings()
    assert s.resources.speed_aware_lanes is True
    d = SettingsDialog(s, None)
    assert d.speed_aware.isChecked()
    d.speed_aware.setChecked(False)
    out = d.result_settings()
    assert out.resources.speed_aware_lanes is False
    sm.save(out)
    assert sm.load().resources.speed_aware_lanes is False
    d.reject()
    d.deleteLater()
    d = SettingsDialog(sm.load(), None)
    assert not d.speed_aware.isChecked()
    d.reject()
    d.deleteLater()


def test_benchmark_this_computer_button(tmp_path, monkeypatch):
    from quantabench import settings as sm
    from quantabench.dist import profile
    from quantabench.gui.main_window import MainWindow

    here = profile.machine_name()
    calls = []

    def fake(bench=False, should_stop=None, **kw):
        calls.append(bench)
        return profile.MachineProfile(name=here, physical_cores=4, logical_cores=8,
                                      throughput={"mmff@cpu": 4321.0} if bench else {})

    saved = []
    monkeypatch.setattr(profile, "local_profile", fake)
    monkeypatch.setattr(profile, "save_profile", lambda p: saved.append(p) or tmp_path / "x.json")
    monkeypatch.setattr(profile, "load_profiles", lambda *a, **k: list(saved))
    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    panel = w.machines_panel
    t = panel.table
    assert t.item(0, 3).text() == "not measured"
    panel.benchmark_here()
    assert _wait(app, lambda: not panel._benching)
    assert calls == [True] and saved and "mmff@cpu 4,321" in panel.note.text()
    assert t.item(0, 3).text().startswith("mmff@cpu 4,321") and panel.bench_btn.isEnabled()
    w.close()
    assert _wait(app, lambda: w.closed)


def test_app_collects_cycles_in_the_gui_thread():
    import gc

    from quantabench.gui import app as app_mod

    qapp = _app()
    try:
        app_mod._collect_in_gui_thread(qapp)
        assert not gc.isenabled() and qapp._qb_gc_timer.isActive() and qapp._qb_gc_timer.interval() == 5000
    finally:
        qapp._qb_gc_timer.stop()
        gc.enable()
