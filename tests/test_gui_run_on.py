"""UI-5 Run on a worker (DIST-3): the job dialog sends a job to a worker (here: on 127.0.0.1, in this process), the
Jobs dock follows it, Cancel goes there, and the results come back into the Molecules list, through the synced output
folder or the worker's archive. "Auto" chooses by speed per waiting job, this computer included."""

import socket
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import HAS_DISPLAY

from quantabench import settings as settings_mod
from quantabench.dist import client as client_mod
from quantabench.dist import remote_job
from quantabench.dist import worker as worker_mod

TOKEN = "test-token-" + "x" * 32
needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


@pytest.fixture(autouse=True)
def qb_settings(isolated_settings, monkeypatch):
    s = settings_mod.Settings(output_dir=str(isolated_settings / "client_runs"))
    s.resources.threads, s.resources.gpu = 1, "off"
    s.dist.worker_token = TOKEN
    s.build.refine_method = "mmff"
    s.build.max_conformers = 2
    settings_mod.save(s)
    monkeypatch.setattr("quantabench.chem.properties.available", lambda: False)  # no dxtb step in tests
    return s


@pytest.fixture
def make_worker(tmp_path):
    started = []

    def make(name, output=None):
        w = worker_mod.Worker("127.0.0.1", 0, TOKEN, output or tmp_path / f"{name}_runs",
                              state_dir=tmp_path / f"{name}_state", name=name, awake=False)
        started.append(w)
        s = settings_mod.load()
        s.dist.workers = {**s.dist.workers, name: f"http://127.0.0.1:{w.port}"}
        settings_mod.save(s)
        return w.start()

    yield make
    for w in started:
        w.stop(timeout=30)


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=90.0):
    from PySide6.QtCore import QEvent

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        time.sleep(0.02)
    return cond()


def _window(qb_settings):
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.main_window import MainWindow

    w = MainWindow(settings_mod.load())
    w.resize(1400, 900)
    w.show()
    w.set_workspace("calculate")
    w.add_inputs([from_smiles("CCO", id="ethanol"), from_smiles("OCC(F)(F)F", id="TFE")])
    _wait(_app(), lambda: False, 0.3)
    return w


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed, 30)


def _spec(w, run_on, name="remote"):
    from quantabench.chem import recipes

    cfg = replace(w.current_config(), method="mmff", max_conformers=2, refine_top=1, xtb_properties=False,
                  formats=("xyz",))
    return recipes.JobSpec(name=name, cfg=cfg, run_on=run_on)


# ------------------------------------------------------------------------------------------------ no GUI
def test_auto_picks_the_fastest_machine_per_waiting_job(monkeypatch):
    from quantabench import __version__
    from quantabench.chem.pipeline import BuildConfig

    refs = {"mac": client_mod.WorkerRef("mac", "http://127.0.0.1:1", "settings",
                                        info={"quantabench_version": __version__, "queue": {"running": [], "queued": []}}),
            "old": client_mod.WorkerRef("old", "http://127.0.0.1:2", "settings",
                                        info={"quantabench_version": "0.0.1", "queue": {}}),
            "gone": client_mod.WorkerRef("gone", "http://127.0.0.1:3", "settings", error="cannot reach")}
    why = remote_job.worker_choices(refs)
    assert why["mac"] == "" and "0.0.1 there" in why["old"] and why["gone"] == "cannot reach"
    speeds = {"local": 1.0, "mac": 3.0}
    monkeypatch.setattr(client_mod, "resolve_weights", lambda spec, names, *a, **k: {n: speeds[n] for n in names})
    cfg = BuildConfig(method="mmff")
    assert remote_job.choose_machine(cfg, refs, local_busy=0)[0] == "mac"
    refs["mac"].info["queue"] = {"running": ["a"], "queued": ["b", "c"]}            # 3 / (1 + 3) < 1 / 1
    assert remote_job.choose_machine(cfg, refs, local_busy=0)[0] == "local"
    assert remote_job.choose_machine(cfg, {}, local_busy=5) == ("local", "no worker can take it")
    assert remote_job.label_for("Screen: conformers, 2 molecules", 7).startswith("job7_Screen")
    assert worker_mod._LABEL_RE.fullmatch(remote_job.label_for("Ünïcode / spaces & stuff" * 5, 12))


# ------------------------------------------------------------------------------------------------ GUI
@needs_display
def test_a_job_runs_on_a_worker_and_its_results_come_back(make_worker, qb_settings):
    from quantabench.gui.job_dialog import JobDialog, submit_spec

    app = _app()
    out = Path(qb_settings.output_dir)
    w1 = make_worker("w1", output=out)          # its run folder appears in this computer's output folder (synced)
    w2 = make_worker("w2")                      # its own folder: the results come back as the worker's archive
    w = _window(qb_settings)
    try:
        d = JobDialog(w, available={"mmff"}, programs=set())
        assert _wait(app, lambda: d.worker_refs != {}, 30)
        names = {d.w_run_on.itemData(i): (d.w_run_on.itemText(i), d.w_run_on.model().item(i).isEnabled())
                 for i in range(d.w_run_on.count())}
        assert names["w1"] == ("w1 (idle)", True) and names["w2"][1] and names["auto"][1]
        d.w_run_on.setCurrentIndex(d.w_run_on.findData("w1"))
        assert not d.w_threads.isEnabled() and "w1" in d.run_on_info.text()
        d.close()
        q = w.job_queue
        j1 = submit_spec(w, _spec(w, "w1", "on w1"), ["ethanol", "TFE"])
        assert j1.is_remote and j1.machine == "w1" and q.running is None      # the local queue is not used
        assert _wait(app, lambda: j1.status in ("done", "failed", "cancelled"))
        assert j1.status == "done", list(j1.log)
        assert (j1.ok, j1.failed, j1.progress_text()) == (2, 0, "2 / 2") and j1.remote_id
        assert j1.output_dir is not None and j1.output_dir.parent == out             # the synced copy, in place
        assert any("Sent to w1" in line for line in j1.log) and any("(synced)" in line for line in j1.log)
        for ident in ("ethanol", "TFE"):
            row = w.model.row(ident)
            assert row.status == "ok" and row.stage == "" and Path(row.run_dir) == j1.output_dir
        # the archive way: w2's run folder is not here
        w.model.update_many(["ethanol", "TFE"], status="pending", run_dir="", folder="")
        j2 = submit_spec(w, _spec(w, "w2", "on w2"), ["ethanol", "TFE"])
        assert _wait(app, lambda: j2.status in ("done", "failed", "cancelled"))
        assert j2.status == "done", list(j2.log)
        assert j2.output_dir.parent == out / remote_job.DOWNLOAD_DIR / "w2"
        assert any("(archive)" in line for line in j2.log)
        assert w.model.row("TFE").status == "ok" and Path(w.model.row("TFE").run_dir) == j2.output_dir
        mon = w.job_monitor
        mon.select(j2.number)
        assert mon.tree.currentItem().text(2) == "w2" and "on w2" in mon.detail.text()
        assert w.worker is None and w2.manager.list()[0].state == "done" and w1.manager.list()[0].state == "done"
    finally:
        _close(app, w)


@needs_display
def test_cancel_goes_to_the_worker_and_an_unreachable_one_is_reported(make_worker, qb_settings, monkeypatch):
    from quantabench.chem import pipeline
    from quantabench.gui.job_dialog import submit_spec

    app = _app()
    gate = threading.Event()
    real = pipeline.run_molecule

    def held(inp, cfg, device, threads, should_stop, *a, **k):
        while not gate.is_set() and not should_stop():
            time.sleep(0.02)
        return real(inp, cfg, device, threads, should_stop, *a, **k)

    monkeypatch.setattr(pipeline, "run_molecule", held)
    monkeypatch.setattr(remote_job, "POLL_S", 0.2)
    w1 = make_worker("w1")
    s = settings_mod.load()
    s.dist.workers = {**s.dist.workers, "ghost": f"http://127.0.0.1:{_closed_port()}"}
    settings_mod.save(s)
    w = _window(qb_settings)
    try:
        q = w.job_queue
        job = submit_spec(w, _spec(w, "w1", "held"), ["ethanol", "TFE"])
        assert _wait(app, lambda: job.status == "running", 60)
        assert w.model.row("ethanol").stage == "on w1"
        w.job_monitor.select(job.number)
        assert w.job_monitor.cancel_btn.isEnabled() and w.job_monitor.cancel_selected()
        assert job.status == "cancelling"
        assert _wait(app, lambda: job.status in ("cancelled", "done", "failed"), 60)
        assert job.status == "cancelled", list(job.log)
        assert w1.manager.list()[0].state == "cancelled"
        assert w.model.row("ethanol").stage == ""
        # a worker that does not answer: the job is not taken, and says why
        ghost = submit_spec(w, _spec(w, "ghost", "ghost"), ["ethanol"])
        assert _wait(app, lambda: ghost.status == "failed", 60)
        assert "ghost did not take the job" in ghost.error
        # a worker that stops answering while following: reported, retried
        gate.set()
        job2 = submit_spec(w, _spec(w, "w1", "then gone"), ["ethanol", "TFE"])
        assert _wait(app, lambda: job2.remote_id != "", 60)
        w1.stop(timeout=30)
        assert _wait(app, lambda: job2.unreachable, 60)
        assert "unreachable" in job2.current
        assert w.job_monitor.items[job2.number].text(2) == "w1 (unreachable)"
        assert q.stop_remote() == 1
    finally:
        gate.set()
        _close(app, w)


@needs_display
def test_auto_that_picks_this_computer_runs_the_job_here(qb_settings):
    from quantabench.gui.job_dialog import submit_spec

    app = _app()
    w = _window(qb_settings)
    try:
        job = submit_spec(w, _spec(w, "auto", "auto"), ["ethanol", "TFE"])     # no worker known: this computer
        assert _wait(app, lambda: job.status in ("done", "failed", "cancelled"), 120)
        assert job.status == "done" and not job.is_remote and job.ok == 2
        assert any("Auto: this computer" in line for line in job.log)
        assert job.output_dir is not None and job.output_dir.is_dir()            # a local run folder
    finally:
        _close(app, w)
