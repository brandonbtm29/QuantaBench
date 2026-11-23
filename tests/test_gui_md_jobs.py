"""SIM-1: MD runs as jobs in the window. An MD job runs in its own supervisor process beside the window's builds, one MD
job at a time; the Jobs dock shows its ns, ns/day and ETA; Cancel pauses it; Resume continues it; Simulate > Run MD
folder... submits one (a real tiny run where LAMMPS is found)."""

import time
from pathlib import Path

import pytest
from conftest import HAS_DISPLAY

pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


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


def _window(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "runs")))
    w.resize(1500, 920)
    w.show()
    w.set_workspace("simulate")
    _wait(_app(), lambda: False, 0.3)
    return w


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed, 30)


class FakeProc:
    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


def test_md_jobs_run_beside_builds_pause_and_resume(tmp_path, monkeypatch):
    from quantabench.gui import job_queue as JQ
    from quantabench.md import supervisor as SV

    app = _app()
    w = _window(tmp_path)
    try:
        q, mon = w.job_queue, w.job_monitor
        q.md_timer.setInterval(50)
        status: dict[str, dict] = {}
        procs: dict[str, FakeProc] = {}
        started = []

        def fake_start(job):
            started.append((job.output_dir.name, job.md_resume))
            procs[job.output_dir.name] = FakeProc()
            return procs[job.output_dir.name]

        q.md_start = fake_start
        q.md_status = lambda folder: dict(status.get(Path(folder).name, {"state": "written", "segments": []}))
        pauses = []
        monkeypatch.setattr(SV, "request_pause", lambda folder, now=False: pauses.append(folder) or "Pausing")
        a = q.submit(JQ.md_job(tmp_path / "a", name="MD: a", resume=False))
        b = q.submit(JQ.md_job(tmp_path / "b", name="MD: b", resume=False))
        assert a.status == "running" and b.status == "queued" and q.md_running is a and started == [("a", False)]
        assert q.running is None and q.local_load() == 0          # builds are not held up by an MD job
        status["a"] = {"state": "running", "alive": True, "ns_total": 13.06, "ns_done": 7.4,
                       "segments": [{"n": 1, "part": "bulk", "status": "done", "ns_per_day": 13.4}],
                       "progress": {"part": "slab", "stage": "nvt", "step": 6_340_000, "part_end": 12_000_000,
                                    "ns_per_day": 21.0, "eta_text": "6 h 27 min", "machine": "here",
                                    "variant": "gpu-cutoff16", "threads": 6, "alerts": ["temperature off target"]}}
        assert _wait(app, lambda: a.md.get("ns_done") == 7.4, 5)
        assert a.progress_text() == "7.4 / 13.1 ns · 21 ns/day · ETA 6 h 27 min"
        mon.select(a.number)
        _wait(app, lambda: False, 0.2)
        it = mon.items[a.number]
        assert it.text(4).startswith("7.4 / 13.1 ns") and mon.cancel_btn.text() == "Pause"
        detail = mon.detail.text()
        assert "step 6,340,000 of 12,000,000" in detail and "temperature off target" in detail
        assert not mon.resume_btn.isEnabled() and mon.resume_btn.isVisible()
        from quantabench.gui.job_monitor import usage_html

        assert f"MD job {a.number}</b>: gpu-cutoff16 x 6 threads, beside the builds" in usage_html(w, None)
        # Pause: the STOP file (request_pause); the job shows "pausing" until the supervisor ends
        assert mon.cancel_selected() and pauses and a.status == "cancelling"
        assert it.text(3) == "pausing" and "(pausing)" in mon.detail.text()
        status["a"] = {**status["a"], "state": "paused", "alive": False,
                       "segments": status["a"]["segments"] + [{"n": 2, "part": "slab", "status": "paused"}]}
        procs["a"].code = 0
        assert _wait(app, lambda: a.status == "paused", 5)
        assert _wait(app, lambda: b.status == "running", 5) and started[-1] == ("b", False)
        assert "Resume continues it" in "\n".join(a.log)
        mon.select(a.number)
        _wait(app, lambda: False, 0.1)
        assert mon.resume_btn.isEnabled() and mon.cancel_btn.text() == "Cancel"
        c = mon.resume_selected()
        assert c is not None and c.md_resume and c.status == "queued" and c.output_dir == a.output_dir
        # b fails to start (no segment appeared): the supervisor log says why
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "supervisor.log").write_text("LAMMPS: LAMMPS not found (test)\n", encoding="utf-8")
        status["b"] = {"state": "written", "alive": False, "segments": []}
        procs["b"].code = 2
        assert _wait(app, lambda: b.status == "failed", 5) and "LAMMPS not found" in b.error
        assert _wait(app, lambda: c.status == "running", 5) and started[-1] == ("a", True)
        # closing the window stops following: the run goes on in its own process
        q.cancel_queued()
        q.stop_remote()
        assert not q.md_timer.isActive() and c.status == "running"
    finally:
        _close(app, w)


def test_run_md_folder_runs_a_real_tiny_slab(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    from quantabench.md import bench
    from quantabench.md import forcefield as F
    from quantabench.md import liquid as L

    info, why = bench.find_lammps()
    if info is None:
        pytest.skip(why)
    folder = tmp_path / "hexane"
    tm = F.type_molecule("CCCCCC", "hexane", "oplsaa")
    cfg = L.LiquidSlabConfig(lateral=18.0, slab_thickness=18.0, packing="lattice", dispersion="cutoff8",
                             thermo_every=100, warmup_ps=0.5, compress_ps=1.0, bulk_ns=0.002, settle_ns=0.001,
                             production_ns=0.002, pressure_every=10, profile_every=10, profile_repeat=10,
                             profile_freq=100, restart_every=1000, dump_every=0, vacuum=20.0)
    L.write_liquid_slab(tm, folder, cfg, density_estimate=0.66)
    asked = []
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: (asked.append(a[3]) or a[3][0], True)))
    app = _app()
    w = _window(tmp_path)
    try:
        assert w.run_md_btn.isVisible() and w.jobs_dock.isVisible()        # the Simulate workspace shows the Jobs
        from quantabench.gui import job_queue as JQ

        def one_thread(job):                  # a tiny run on a shared machine: 1 thread (the default is the budget)
            job.md_threads, job.md_variant = 1, "cpu"
            return JQ._start_md_supervisor(job)

        w.job_queue.md_start = one_thread
        job = w.run_md_folder(str(folder))
        assert job is not None and job.is_md and job.status == "running"
        assert asked and asked[0][0].startswith("LAMMPS")          # OPENMM-1: the engine asked, the folder's first
        assert w.run_md_folder(str(folder)) is None                       # the same folder is not submitted twice
        assert _wait(app, lambda: job.status in ("done", "failed"), 240), (job.status, list(job.log))
        assert job.status == "done", (job.error, list(job.log))
        assert job.progress_text().startswith("0.0065 / 0.0065 ns")
        assert job.md.get("state") == "done" and len(job.md.get("segments")) == 2
        assert (folder / "pressure_tensor.txt").is_file() and (folder / "supervisor.log").is_file()
    finally:
        _close(app, w)


def test_run_md_folder_asks_the_engine_and_switches_to_openmm(tmp_path, monkeypatch):
    """OPENMM-1: an unstarted liquid-slab folder is run on the engine chosen in the dialog (with the note on what
    OpenMM gives); Cancel submits nothing; a started run is not asked again."""
    from PySide6.QtWidgets import QInputDialog

    from quantabench.gui import job_monitor as JM
    from quantabench.md import forcefield as F
    from quantabench.md import liquid as L
    from quantabench.md import sim

    folder = tmp_path / "hexane"
    tm = F.type_molecule("CCCCCC", "hexane", "oplsaa")
    L.write_liquid_slab(tm, folder, L.LiquidSlabConfig(lateral=18.0, slab_thickness=18.0, packing="lattice",
                                                       dispersion="cutoff8", dump_every=0), density_estimate=0.66)
    seen = {}

    def pick(answer, ok=True):
        def get_item(parent, title, label, items, current=0, editable=False):
            seen.update(label=label, items=list(items))
            return next(i for i in items if i.startswith(answer)), ok
        return staticmethod(get_item)

    app = _app()
    w = _window(tmp_path)
    try:
        monkeypatch.setattr(QInputDialog, "getItem", pick("OpenMM", ok=False))
        assert w.run_md_folder(str(folder)) is None and sim.Run.load(folder).engine == "lammps"
        assert " ".join(seen["label"].split()).endswith(L.ENGINE_NOTE) and seen["items"][0].startswith("LAMMPS")
        monkeypatch.setattr(QInputDialog, "getItem", pick("OpenMM"))
        assert JM.choose_md_engine(w, folder) == "openmm"
        run = sim.Run.load(folder)
        assert run.engine == "openmm" and run.protocol.parts[1].restart_files == ["slab.state.a.xml",
                                                                                  "slab.state.b.xml"]
        assert L.load_folder(folder)[1].engine == "openmm"
        run.segments.append(sim.Segment(n=1, part="bulk", status="paused"))
        run.save(folder)
        monkeypatch.setattr(QInputDialog, "getItem", pick("LAMMPS"))
        seen.clear()
        assert JM.choose_md_engine(w, folder) == "openmm" and not seen      # started: not asked, not switched
    finally:
        _close(app, w)
