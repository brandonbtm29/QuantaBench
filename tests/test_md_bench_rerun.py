"""MD-BENCH-3: a release that changes the MD benchmark asks whether to run it again, and nothing re-runs it on its own.

``md.bench.stored_version`` / ``outdated`` / ``rerun_note`` / ``notice_due`` / ``status``; the note under a machine in
``quantabench-cli machines`` (and ``--json``); the worker that logs it and leaves the benchmark alone, reports it in
``GET /v1/info`` and runs it when asked (``POST /v1/bench``); the Machines dock's notices (this computer and other
machines' workers) with their Run and Not now buttons (Not now remembered per machine and benchmark version), the Run
MD benchmark button, and the "older benchmark" mark of other machines."""

import json
import threading
import time
from types import SimpleNamespace

import pytest
from conftest import HAS_DISPLAY

from quantabench import cli
from quantabench.dist import profile
from quantabench.dist.profile import MachineProfile
from quantabench.md import bench

CPU_EWALD = {"ns_per_day": 6.5, "atoms": 9780, "threads": 6, "accelerator": "omp", "steps": 2000}


def md_record(version, **variants) -> dict:
    rec = {"date": "2026-11-20T12:00:00+00:00", "lammps": "4 Jul 2026", "variants": variants, "notes": [],
           "skipped": ""}
    if version is not None:
        rec["bench_version"] = version
    return rec


OLD = md_record(1, **{"cpu-ewald": {**CPU_EWALD, "bench_version": 1}})
NEW = md_record(bench.MD_BENCH_VERSION, **{"cpu-ewald@slab": {**CPU_EWALD, "atoms": 3800,
                                                              "bench_version": bench.MD_BENCH_VERSION}})


def test_version_changes_name_the_current_version():
    assert bench.MD_BENCH_VERSION in bench.VERSION_CHANGES
    assert sorted(bench.VERSION_CHANGES) == list(range(1, bench.MD_BENCH_VERSION + 1))


def test_outdated_compares_the_stored_version_with_this_release():
    assert bench.MD_BENCH_VERSION > 1
    assert bench.stored_version(OLD) == 1 and bench.outdated(OLD) == 1
    assert bench.stored_version(NEW) == bench.MD_BENCH_VERSION and bench.outdated(NEW) is None
    assert bench.outdated(None) is None and bench.outdated({}) is None
    # nothing measured: no question (the "not measured" text says what to do)
    assert bench.outdated(md_record(1)) is None
    assert bench.outdated({"bench_version": 1, "skipped": "LAMMPS not found", "variants": {}}) is None
    assert bench.outdated(md_record(1, **{"cpu-ewald": {"error": "boom"}})) is None
    # no top-level version: the newest of the measured variants'; none at all: unknown, so no question
    assert bench.stored_version(md_record(None, a={**CPU_EWALD, "bench_version": 1},
                                          b={**CPU_EWALD, "bench_version": 2})) == 2
    assert bench.outdated(md_record(None, **{"cpu-ewald": {**CPU_EWALD, "bench_version": 1}})) == 1
    assert bench.outdated(md_record(None, **{"cpu-ewald": CPU_EWALD})) is None
    assert bench.stored_version({"bench_version": True}) is None          # not a version
    # a newer benchmark than this release's (another machine updated first): not "older"
    assert bench.outdated(md_record(bench.MD_BENCH_VERSION + 1, **{"cpu-ewald": CPU_EWALD})) is None


def test_a_rerun_clears_the_question_and_a_failed_one_keeps_it():
    assert bench.outdated(bench.merge(OLD, NEW)) is None          # the old bulk records stay, the question goes
    failed = {"bench_version": bench.MD_BENCH_VERSION, "skipped": "LAMMPS not found", "variants": {}}
    kept = bench.merge(OLD, failed)
    assert bench.outdated(kept) == 1 and kept["last_attempt"]["skipped"] == "LAMMPS not found"


def test_rerun_note_asks_and_names_the_machine():
    here = bench.rerun_note(OLD)
    assert here.startswith("This computer's MD benchmark is from benchmark version 1. This QuantaBench's is "
                           f"version {bench.MD_BENCH_VERSION}: ")
    assert bench.VERSION_CHANGES[bench.MD_BENCH_VERSION] in here
    assert "Run it again? `quantabench-cli machines --bench md` (a few minutes, best while" in here
    assert here.endswith("Nothing re-runs it on its own.")
    other = bench.rerun_note(OLD, "mac-test", here=False)
    assert other.startswith("mac-test's MD benchmark") and "`quantabench-cli machines --bench md` on mac-test" in other
    assert bench.rerun_note(NEW) == "" and bench.rerun_note({}) == ""


def test_notice_due_remembers_not_now_for_one_version():
    v = bench.MD_BENCH_VERSION
    assert bench.notice_due(OLD) and bench.notice_due(OLD, 0)
    assert not bench.notice_due(OLD, v)                     # Not now for this version: no notice
    assert bench.notice_due(OLD, v, version=v + 1)          # a later release changes the benchmark again: it returns
    assert not bench.notice_due(NEW) and not bench.notice_due({}) and not bench.notice_due(None, 0)
    assert not bench.notice_due(md_record(1))                # nothing measured: no question
    # another machine's worker reports its own version: compared with that, not with this QuantaBench's
    assert not bench.notice_due(OLD, 0, version=1) and bench.notice_due(OLD, 1, version=2)
    st = bench.status(OLD)
    assert st == {"version": v, "stored": 1, "outdated": 1, "note": bench.rerun_note(OLD)}
    assert bench.status(NEW)["outdated"] is None and bench.status(NEW)["note"] == ""
    assert bench.status(None) == {"version": v, "stored": None, "outdated": None, "note": ""}


def test_not_now_is_a_per_machine_setting():
    from quantabench import settings as sm

    assert sm.load().machine_control_for("mac-test")["md_bench_not_now"] == 0
    sm.update_machine("mac-test", md_bench_not_now=2)
    s = sm.load()
    assert s.machine_control_for("mac-test")["md_bench_not_now"] == 2
    assert s.machine_control_for("tower-test")["md_bench_not_now"] == 0
    for bad in (True, -1, 2.5, "2"):
        with pytest.raises(ValueError):
            sm.update_machine("mac-test", md_bench_not_now=bad)
    s.dist.machine_control["mac-test"]["md_bench_not_now"] = "x"          # a hand-edited file: ignored
    assert s.machine_control_for("mac-test")["md_bench_not_now"] == 0
    sm.update_machine("mac-test", md_bench_not_now=None)
    assert sm.load().machine_control_for("mac-test")["md_bench_not_now"] == 0


def test_machines_cli_shows_the_note_under_each_machine(capsys):
    me = profile.machine_name()
    profile.save_profile(MachineProfile(name=me, md_throughput=OLD))
    profile.save_profile(MachineProfile(name="mac-test", md_throughput=OLD))
    profile.save_profile(MachineProfile(name="tower-test", md_throughput=NEW))
    assert cli.main(["machines"]) == 0
    out = capsys.readouterr().out
    assert "  note: This computer's MD benchmark is from benchmark version 1" in out
    assert "  note: mac-test's MD benchmark is from benchmark version 1" in out
    assert "--bench md` on mac-test" in out and "tower-test's MD benchmark" not in out
    assert cli.main(["machines", "--json"]) == 0
    rows = {m["name"]: m for m in json.loads(capsys.readouterr().out)["machines"]}
    assert rows[me]["md_bench_rerun"].startswith("This computer's MD benchmark")
    assert rows["mac-test"]["md_bench_rerun"].startswith("mac-test's") and rows["tower-test"]["md_bench_rerun"] == ""
    text = "\n".join(profile.describe(MachineProfile(name="pc", md_throughput=NEW), this=True))
    assert "note:" not in text


def test_the_worker_logs_the_note_and_never_reruns_it():
    from quantabench.dist.worker import Worker

    def worker(mode, methods, prof):
        logs: list[str] = []
        w = SimpleNamespace(bench=mode, bench_methods=methods, profile=prof, log=logs.append, logs=logs,
                            manager=SimpleNamespace(set_hold=lambda *_a: pytest.fail("no benchmark may start")))
        w._methods_to_measure = lambda: Worker._methods_to_measure(w)
        w.needs_benchmark = lambda: Worker.needs_benchmark(w)
        w.md_rerun_note = lambda: Worker.md_rerun_note(w)
        return w

    old = MachineProfile(name="w", throughput={"mmff@cpu": 5.0}, md_throughput=OLD)
    for mode, methods in (("auto", ["md"]), ("auto", ["default", "md"]), ("auto", None), ("never", ["md"])):
        w = worker(mode, methods, old)
        assert not Worker.needs_benchmark(w), (mode, methods)
        Worker._start_benchmark(w)                         # logs, starts nothing (set_hold would fail the test)
        assert len(w.logs) == 1 and w.logs[0].startswith("Note: This computer's MD benchmark is from benchmark "
                                                         "version 1") and w.logs[0].endswith("leaves it as it is.")
    current = worker("auto", ["md"], MachineProfile(name="w", throughput={"mmff@cpu": 5.0}, md_throughput=NEW))
    Worker._start_benchmark(current)
    assert current.logs == []
    # --bench always measures everything again, because it was asked for: no note then
    assert Worker._methods_to_measure(worker("always", ["default", "md"], old)) == ["default", "md"]


def _wait_for(cond, timeout=30.0):
    t = time.monotonic()
    while not cond():
        assert time.monotonic() - t < timeout, "timed out"
        time.sleep(0.02)


def test_the_worker_reports_and_runs_the_md_benchmark_when_asked(tmp_path, monkeypatch):
    """GET /v1/info's "md_bench"; POST /v1/bench runs it (jobs held meanwhile) and refuses while paused, while a job
    runs, while it already runs, and any other benchmark."""
    from quantabench.dist import worker as worker_mod
    from quantabench.dist.client import WorkerClient, WorkerError

    token = "test-token-" + "b" * 32
    gate = threading.Event()
    calls = []
    failed = bench.merge(OLD, {"bench_version": bench.MD_BENCH_VERSION, "skipped": "LAMMPS not found",
                               "variants": {}})
    results = [failed, bench.merge(OLD, NEW)]

    def fake_local_profile(bench=False, methods=None, should_stop=None, **kw):
        if not bench:
            return MachineProfile(name="bench-w", throughput={"mmff@cpu": 5.0}, md_throughput=OLD)
        calls.append(methods)
        while not gate.is_set() and not (should_stop and should_stop()):
            time.sleep(0.01)
        return MachineProfile(name="bench-w", throughput={"mmff@cpu": 5.0}, md_throughput=results.pop(0))

    monkeypatch.setattr(profile, "local_profile", fake_local_profile)
    w = worker_mod.Worker("127.0.0.1", 0, token, tmp_path / "runs", state_dir=tmp_path / "state", name="bench-w",
                          awake=False).start()
    try:
        c = WorkerClient(w.url, token)
        mb = c.info()["md_bench"]
        assert mb["version"] == bench.MD_BENCH_VERSION and mb["stored"] == 1 and mb["outdated"] == 1
        assert mb["note"].startswith("This computer's MD benchmark") and mb["running"] is False and mb["last"] is None
        assert calls == []                                    # nothing ran on its own
        with pytest.raises(WorkerError) as e:                 # only the MD benchmark can be asked for
            c._json("POST", "/v1/bench", {"methods": ["mmff"]})
        assert e.value.status == 400
        c.pause(True, "drain", by="test")
        with pytest.raises(WorkerError) as e:
            c.bench(["md"], by="dock-test")
        assert e.value.status == 409 and "paused" in str(e.value) and calls == []
        c.pause(False)
        # a job running: refused, and the hold it took is released
        real_state = w.manager.queue_state
        monkeypatch.setattr(w.manager, "queue_state", lambda: {**real_state(), "running": ["0123456789ab"]})
        with pytest.raises(WorkerError) as e:
            c.bench(["md"], by="dock-test")
        assert e.value.status == 409 and "0123456789ab" in str(e.value) and w.manager.hold == ""
        monkeypatch.setattr(w.manager, "queue_state", real_state)
        # a run that measures nothing: "failed", the older numbers and the question stay
        st = c.bench(["md"], by="dock-test")
        assert st["running"] is True and st["last"]["state"] == "running" and st["last"]["by"] == "dock-test"
        assert c.info()["queue"]["hold"].startswith("benchmarking")          # new jobs wait meanwhile
        with pytest.raises(WorkerError) as e:
            c.bench(["md"])
        assert e.value.status == 409 and "already running" in str(e.value)
        gate.set()
        _wait_for(lambda: not c.info()["md_bench"]["running"])
        mb = c.info()["md_bench"]
        assert calls == [["md"]] and mb["outdated"] == 1 and mb["last"]["state"] == "failed"
        assert mb["last"]["text"] == "LAMMPS not found; the older numbers stay" and mb["last"]["ended"]
        assert c.info()["queue"]["hold"] == ""
        # a run that measures: "done", the question goes, the engines' numbers stay
        c.bench(["md"], by="dock-test")
        _wait_for(lambda: not c.info()["md_bench"]["running"])
        info = c.info()
        assert info["md_bench"]["outdated"] is None and info["md_bench"]["note"] == ""
        assert info["md_bench"]["last"]["state"] == "done" and info["profile"]["throughput"] == {"mmff@cpu": 5.0}
    finally:
        gate.set()
        w.stop(timeout=30)


# ------------------------------------------------------------------------------------------- the Machines dock
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


@pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")
def test_machines_dock_asks_and_runs_the_md_benchmark(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from quantabench import settings as sm
    from quantabench.gui.machines_panel import COL
    from quantabench.gui.main_window import MainWindow

    here = profile.machine_name()
    registry = {here: MachineProfile(name=here, physical_cores=4, logical_cores=8, throughput={"mmff@cpu": 99.0},
                                     md_throughput=OLD),
                "mac-test": MachineProfile(name="mac-test", md_throughput=OLD)}
    calls = []
    outcome = {"md": NEW}

    def fake_local_profile(bench=False, methods=None, should_stop=None, **kw):
        calls.append((bench, methods))
        return MachineProfile(name=here, physical_cores=4, logical_cores=8, throughput={"mmff@cpu": 99.0},
                              md_throughput=outcome["md"])

    def fake_save(p):
        registry[p.name] = p
        return tmp_path / "x.json"

    monkeypatch.setattr(profile, "load_profiles", lambda *a, **k: list(registry.values()))
    monkeypatch.setattr(profile, "local_profile", fake_local_profile)
    monkeypatch.setattr(profile, "save_profile", fake_save)
    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    panel = w.machines_panel
    try:
        assert "older benchmark version" in w.statusBar().currentMessage()
        assert not panel.md_notice.isHidden() and panel.md_run_btn.isEnabled()
        assert panel.md_notice_text.text().startswith("This computer's MD benchmark is from benchmark version 1")
        t, md = panel.table, COL["MD (ns/day)"]
        rows = {t.item(r, 0).text(): r for r in range(t.rowCount())}
        assert t.item(rows["mac-test"], md).text() == "cpu-ewald 6.5; older benchmark (version 1)"
        assert "--bench md` on mac-test" in t.item(rows["mac-test"], md).toolTip()
        assert t.item(rows[f"{here}  (this computer)"], md).text().endswith("; older benchmark (version 1)")
        # a re-run that measures nothing keeps the old numbers and the question
        outcome["md"] = bench.merge(OLD, {"bench_version": bench.MD_BENCH_VERSION, "skipped": "LAMMPS not found",
                                          "variants": {}})
        panel.benchmark_md_here()
        assert _wait(app, lambda: not panel._benching)
        assert calls == [(True, ["md"])] and "LAMMPS not found; the older numbers stay" in panel.note.text()
        assert not panel.md_notice.isHidden() and panel.md_run_btn.isEnabled() and panel.bench_btn.isEnabled()
        # Run MD benchmark: the MD benchmark only (the engines' numbers stay), saved, and the question goes
        outcome["md"] = bench.merge(OLD, NEW)
        assert panel.md_bench_btn.isEnabled()
        panel.md_run_btn.click()
        assert _wait(app, lambda: not panel._benching)
        assert calls[-1] == (True, ["md"]) and registry[here].throughput == {"mmff@cpu": 99.0}
        assert panel.note.text() == "MD benchmark saved: cpu-ewald@slab 6.5, cpu-ewald 6.5"
        assert panel.md_notice.isHidden()
        assert "older benchmark" not in t.item(rows[f"{here}  (this computer)"], md).text()
        assert t.item(rows["mac-test"], md).text().endswith("(version 1)")          # still asks about the Mac
        assert panel.md_bench_btn.isEnabled()
        # the button under the table runs it any time, on the selected machine (none selected: this computer)
        panel.table.clearSelection()
        panel.md_bench_btn.click()
        assert _wait(app, lambda: not panel._benching)
        assert len(calls) == 3 and panel.note.text().startswith("MD benchmark saved")
    finally:
        w.close()
        assert _wait(app, lambda: w.closed)


@pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")
def test_not_now_is_remembered_until_the_benchmark_changes_again(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    here = profile.machine_name()
    monkeypatch.setattr(profile, "load_profiles", lambda *a, **k: [MachineProfile(name=here, md_throughput=OLD)])
    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    try:
        panel = w.machines_panel
        assert not panel.md_notice.isHidden()
        panel.md_later_btn.click()
        assert panel.md_notice.isHidden() and panel.note.text().startswith("MD benchmark: not now for this computer")
        assert sm.load().machine_control_for(here)["md_bench_not_now"] == bench.MD_BENCH_VERSION
        panel.refresh()
        assert panel.md_notice.isHidden()
    finally:
        w.close()
        assert _wait(app, lambda: w.closed)
    # the next start: still hidden, and no status-bar message
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    try:
        assert w.machines_panel.md_notice.isHidden() and not w.machines_panel.md_notice_due()
        assert "older benchmark" not in w.statusBar().currentMessage()
        assert w.machines_panel.md_bench_btn.isEnabled()          # it can still be run any time
        # a later release changes the benchmark again: the notice comes back
        monkeypatch.setattr(bench, "MD_BENCH_VERSION", bench.MD_BENCH_VERSION + 1)
        monkeypatch.setitem(bench.VERSION_CHANGES, bench.MD_BENCH_VERSION, "a test change")
        w.machines_panel.refresh()
        assert not w.machines_panel.md_notice.isHidden()
    finally:
        w.close()
        assert _wait(app, lambda: w.closed)


@pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")
def test_the_dock_offers_the_rerun_for_another_machine(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from quantabench import settings as sm
    from quantabench.dist import client
    from quantabench.gui.main_window import MainWindow

    here = profile.machine_name()
    v = bench.MD_BENCH_VERSION
    registry = [MachineProfile(name=here, md_throughput=NEW),
                MachineProfile(name="mac-test", md_throughput=OLD, address="http://100.101.1.3:8765"),
                MachineProfile(name="old-test", md_throughput=OLD, address="http://100.101.1.4:8765"),
                MachineProfile(name="tower-test", md_throughput=OLD, address="http://100.101.1.2:8765")]
    monkeypatch.setattr(profile, "load_profiles", lambda *a, **k: list(registry))
    state = {"mac-test": {"version": v, "stored": 1, "outdated": 1, "note": "x", "running": False, "last": None}}

    def refs():
        q = {"running": [], "queued": [], "paused": False}
        return {"mac-test": client.WorkerRef(name="mac-test", url="http://100.101.1.3:8765", source="registry",
                                             info={"queue": q, "md_bench": dict(state["mac-test"])}),
                "old-test": client.WorkerRef(name="old-test", url="http://100.101.1.4:8765", source="registry",
                                             info={"queue": q}),                   # a worker from before MD-BENCH-3
                "tower-test": client.WorkerRef(name="tower-test", url="http://100.101.1.2:8765", source="registry",
                                               info={"queue": q, "md_bench": {"version": v, "stored": v,
                                                                              "outdated": None, "note": "",
                                                                              "running": False, "last": None}})}

    monkeypatch.setattr(client, "discover", lambda *a, **k: refs())
    asked = []
    answer = {"error": None}

    def fake_bench(self, methods=("md",), by=""):
        asked.append((self.url, list(methods), by))
        if answer["error"]:
            raise client.WorkerError(answer["error"], 409)
        state["mac-test"] = {**state["mac-test"], "running": True}
        return dict(state["mac-test"])

    monkeypatch.setattr(client.WorkerClient, "bench", fake_bench)
    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    panel = w.machines_panel
    try:
        assert panel.md_notice.isHidden() and panel.md_remote == {}          # nothing asked before Check workers
        panel.check_workers()
        assert _wait(app, lambda: not panel._busy)
        assert set(panel.md_remote) == {"mac-test"}                          # tower is current, old-test can't say
        row = panel.md_remote["mac-test"]
        assert not panel.md_remote_box.isHidden() and row.run.text() == "Run on mac-test"
        assert row.text.text().startswith("mac-test's MD benchmark is from benchmark version 1")
        # a refusal (busy) is shown, and the notice stays
        answer["error"] = "http://100.101.1.3:8765/v1/bench: HTTP 409: mac-test is running job(s) j1"
        row.run.click()
        assert _wait(app, lambda: "not started" in panel.note.text())
        assert "running job(s) j1" in panel.note.text() and row.run.isEnabled()
        # Run on mac-test: asked through its worker; it runs there, so the notice goes
        answer["error"] = None
        row.run.click()
        assert _wait(app, lambda: "running there now" in panel.note.text() and not panel._busy)
        assert asked[-1][0] == "http://100.101.1.3:8765" and asked[-1][1] == ["md"]
        assert asked[-1][2].endswith("(Machines dock)") and panel.md_remote == {}
        # the button under the table, on a worker that cannot do it from here: the command to run there
        panel.select("old-test")
        panel.md_bench_btn.click()
        assert "older than MD-BENCH-3" in panel.note.text() and "--bench md` on old-test" in panel.note.text()
        # done there, but still older (e.g. no LAMMPS): offered again; Not now remembers it for that version
        state["mac-test"] = {**state["mac-test"], "running": False}
        panel.check_workers()
        assert _wait(app, lambda: not panel._busy and "mac-test" in panel.md_remote)
        panel.md_remote["mac-test"].later.click()
        assert "mac-test" not in panel.md_remote
        assert sm.load().machine_control_for("mac-test")["md_bench_not_now"] == v
        panel.check_workers()
        assert _wait(app, lambda: not panel._busy)
        assert panel.md_remote == {} and panel.md_remote_box.isHidden()
        # its worker updates to a later benchmark version: offered again
        state["mac-test"] = {**state["mac-test"], "version": v + 1}
        panel.check_workers()
        assert _wait(app, lambda: not panel._busy and "mac-test" in panel.md_remote)
    finally:
        w.close()
        assert _wait(app, lambda: w.closed)


@pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")
def test_no_notice_when_the_benchmark_is_current_or_missing(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    here = profile.machine_name()
    for md in (NEW, {}):
        monkeypatch.setattr(profile, "load_profiles",
                            lambda *a, md=md, **k: [MachineProfile(name=here, md_throughput=md)])
        app = _app()
        w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
        try:
            assert w.machines_panel.md_notice.isHidden() and w.machines_panel.md_rerun_note() == ""
            assert "older benchmark" not in w.statusBar().currentMessage()
        finally:
            w.close()
            assert _wait(app, lambda w=w: w.closed)
