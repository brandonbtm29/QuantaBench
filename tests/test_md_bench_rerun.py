"""MD-BENCH-3: a release that changes the MD benchmark asks whether to run it again, and nothing re-runs it on its own.

``md.bench.stored_version`` / ``outdated`` / ``rerun_note``; the note under a machine in ``quantabench-cli machines``
(and ``--json``); the worker that logs it and leaves the benchmark alone; the Machines dock's notice with its Run and
Not now buttons and the "older benchmark" mark of other machines."""

import json
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
        # Not now: hidden until the next start, the marks stay
        panel.md_not_now()
        panel.refresh()
        assert panel.md_notice.isHidden() and t.item(rows["mac-test"], md).text().endswith("(version 1)")
        panel._md_later = False
        panel.refresh()
        assert not panel.md_notice.isHidden()
        # a re-run that measures nothing keeps the old numbers and the question
        outcome["md"] = bench.merge(OLD, {"bench_version": bench.MD_BENCH_VERSION, "skipped": "LAMMPS not found",
                                          "variants": {}})
        panel.benchmark_md_here()
        assert _wait(app, lambda: not panel._benching)
        assert calls == [(True, ["md"])] and "LAMMPS not found; the older numbers stay" in panel.note.text()
        assert not panel.md_notice.isHidden() and panel.md_run_btn.isEnabled() and panel.bench_btn.isEnabled()
        # Run MD benchmark: the MD benchmark only (the engines' numbers stay), saved, and the question goes
        outcome["md"] = bench.merge(OLD, NEW)
        panel.md_run_btn.click()
        assert _wait(app, lambda: not panel._benching)
        assert calls[-1] == (True, ["md"]) and registry[here].throughput == {"mmff@cpu": 99.0}
        assert panel.note.text() == "MD benchmark saved: cpu-ewald@slab 6.5, cpu-ewald 6.5"
        assert panel.md_notice.isHidden()
        assert "older benchmark" not in t.item(rows[f"{here}  (this computer)"], md).text()
        assert t.item(rows["mac-test"], md).text().endswith("(version 1)")          # still asks about the Mac
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
