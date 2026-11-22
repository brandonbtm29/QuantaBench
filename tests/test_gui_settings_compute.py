"""Settings > Compute (Spartan's Preferences > Jobs, SETTINGS-1): the "N cores, M logical" line, "Molecules at once"
and the Expert box (the lane spec in full, GPU linear algebra, dxtb on the GPU). The tab edits the same per-machine
values as the Machines dock and the job dialog (``Settings.lanes_for``, ``priority_for``, ``free_threads_for``) and
the shared ``resources.linalg`` / ``dxtb_gpu``; none of them is a science setting."""

import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY,
                                reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def bench_pc(monkeypatch):
    """This machine is "bench-pc" with 12 cores and 24 logical threads."""
    from quantabench import resources
    from quantabench import settings as sm

    monkeypatch.setattr(sm, "_this_machine", lambda: "bench-pc")
    monkeypatch.setattr(resources, "logical_cores", lambda: 24)
    monkeypatch.setattr(resources, "physical_cores", lambda: 12)
    return "bench-pc"


def _close(d, app):
    d.reject()
    d.deleteLater()
    app.processEvents()


def test_cores_line_from_the_report_or_psutil(bench_pc):
    from quantabench import resources
    from quantabench.gui.settings_dialog import cores_line

    assert cores_line(None) == "12 cores, 24 logical"
    mac = resources.HardwareReport(os="macOS", python="3.12", cpu_name="Apple M3 Max", physical_cores=14,
                                   logical_cores=14, ram_gb=36.0, wsl=False, performance_cores=10)
    assert cores_line(mac) == "14 cores (10 performance), 14 logical"


def test_compute_tab_expert_box(bench_pc):
    """The lane spec, GPU linear algebra and dxtb on the GPU sit in a folded Expert box whose header counts what is
    not at its default; the GPU section keeps the GPU itself."""
    from quantabench import settings as sm
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    d = SettingsDialog(sm.Settings(), None)
    try:
        d.show()
        app.processEvents()
        assert d.cores_label.text() == "This computer: 12 cores, 24 logical"
        expert, gpu = d.sections["compute_expert"], d.sections["compute_gpu"]
        assert not expert.is_open() and expert.info_button is not None
        for widget in (d.lanes, d.linalg, d.dxtb_gpu):
            assert expert.body.isAncestorOf(widget) and not widget.isVisible()
            assert not gpu.body.isAncestorOf(widget)
        assert gpu.body.isAncestorOf(d.gpu) and gpu.body.isAncestorOf(d.mem)
        assert d.sections["compute_lanes"].body.isAncestorOf(d.lane_count)
        assert expert.summary.text() == "lane spec 1 · GPU linear algebra auto · dxtb on the GPU auto"
        assert expert.status.text() == "" and d.expert_changes() == []
        assert "dxtb" not in gpu.summary.text()

        d.linalg.setCurrentText("cpu")
        assert expert.status.text() == "1 changed" and d.expert_changes() == ["GPU linear algebra cpu"]
        d.dxtb_gpu.setCurrentText("on")
        d.lanes.setCurrentText("gpu:1x4,cpu:5x2")
        assert expert.status.text() == "3 changed"
        assert d.expert_changes() == ["lane spec gpu:1x4,cpu:5x2", "GPU linear algebra cpu", "dxtb on the GPU on"]
        assert expert.status.toolTip() == ", ".join(d.expert_changes())
        d.lanes.setCurrentText("auto")                  # one of the plain choices: not an expert change
        assert expert.status.text() == "2 changed"
    finally:
        _close(d, app)


def test_molecules_at_once_and_the_lane_spec_are_one_value(bench_pc):
    from quantabench import settings as sm
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    d = SettingsDialog(sm.Settings(), None)
    try:
        choices = [d.lane_count.itemData(i) for i in range(d.lane_count.count())]
        assert choices == ["auto", "1", *[str(n) for n in range(2, 13)], "custom"]     # 2 to the physical cores
        assert d.lane_count.currentData() == "1" and d.lanes.currentText() == "1"
        d.lane_count.setCurrentIndex(d.lane_count.findData("4"))
        assert d.lanes.currentText() == "4" and "4 CPU lanes" in d.lane_plan.text()
        d.lanes.setCurrentText("Auto ")
        assert d.lane_count.currentData() == "auto"
        d.lanes.setCurrentText("gpu:1x4,cpu:5x2")
        assert d.lane_count.currentData() == "custom" and not d.sections["compute_expert"].is_open()
        d.lane_count.setCurrentIndex(d.lane_count.findData("2"))
        assert d.lanes.currentText() == "2"
        d.lane_count.setCurrentIndex(d.lane_count.findData("custom"))   # Custom: type a spec in the Expert box
        assert d.sections["compute_expert"].is_open() and d.lanes.currentText() == "2"
        assert d.result_settings().lanes_for() == "2"
        d.lanes.setCurrentText("16")                    # more lanes than cores: a spec only the Expert box shows
        assert d.lane_count.currentData() == "custom" and d.result_settings().lanes_for() == "16"
    finally:
        _close(d, app)


def test_compute_tab_round_trip(bench_pc):
    """What the tab sets is saved, comes back in a new dialog, and is what the job dialog's config, the Machines dock
    and ``machines set`` read; the per-machine values are stamped for the SyncThing conflict merge."""
    from quantabench import settings as sm
    from quantabench.chem.pipeline import BuildConfig
    from quantabench.dist import control
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    d = SettingsDialog(sm.load(), None)
    try:
        d.priority.setCurrentIndex(d.priority.findData("background"))
        d.free_threads.setValue(3)
        d.lane_count.setCurrentIndex(d.lane_count.findData("6"))
        d.linalg.setCurrentText("cpu")
        d.dxtb_gpu.setCurrentText("off")
        out = d.result_settings()
    finally:
        _close(d, app)
    stamps = out.dist.machine_updated[bench_pc]
    assert set(stamps) == {"resources.machine_priority", "resources.machine_free_threads", "dist.machine_lanes"}
    sm.save(out)

    s = sm.load()
    assert s.lanes_for() == "6" and s.dist.lanes == "1"                     # this machine only
    assert s.priority_for(scope=True) == ("background", "this machine")
    assert s.free_threads_for(scope=True) == (3, "this machine")
    assert (s.resources.priority, s.resources.free_threads) == ("low", 1)   # the shared defaults are untouched
    assert (s.resources.linalg, s.resources.dxtb_gpu) == ("cpu", "off")
    cfg = BuildConfig.from_settings(s)                  # the job dialog starts from this (win.current_config)
    assert (cfg.priority, cfg.free_threads, cfg.linalg, cfg.dxtb_gpu) == ("background", 3, "cpu", "off")
    line = control.machine_lines(s, bench_pc)[0]        # machines set / grid machines, and the Machines dock's box
    assert "run priority background (this machine); free threads 3 (this machine); lanes 6" in line

    d = SettingsDialog(s, None)
    try:
        assert d.priority.currentData() == "background" and d.free_threads.value() == 3
        assert d.lane_count.currentData() == "6" and d.lanes.currentText() == "6"
        assert (d.linalg.currentText(), d.dxtb_gpu.currentText()) == ("cpu", "off")
        assert d.sections["compute_expert"].status.text() == "2 changed"
        again = d.result_settings()
        assert again.dist.machine_updated == s.dist.machine_updated         # nothing changed: no new stamp
    finally:
        _close(d, app)


def test_an_unchanged_compute_tab_keeps_the_machines_docks_newer_values(bench_pc):
    """The dialog was opened, then the Machines dock saved this machine's priority and lanes (update_machine); OK in
    the dialog without touching them does not write the old values back."""
    from quantabench import settings as sm
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    d = SettingsDialog(sm.load(), None)
    try:
        sm.update_machine(bench_pc, priority="normal", lanes="3")
        out = d.result_settings()
    finally:
        _close(d, app)
    sm.save(out)
    s = sm.load()
    assert s.priority_for(scope=True) == ("normal", "this machine") and s.lanes_for() == "3"
