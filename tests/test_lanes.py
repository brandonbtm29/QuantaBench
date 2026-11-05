"""Lanes: lane specs, and parallel runs in worker processes (dist.lanes + jobs.run_resumable).

Spawned lane processes take a second or two to start, so each test starts as few pools as it can."""

import csv
import os
import sqlite3

import psutil
import pytest

from quantabench import resources
from quantabench.chem import jobs, pipeline
from quantabench.chem.inputs import from_smiles
from quantabench.dist import lanes as L
from quantabench.dist.lanes import Lane

MOLS = [("ethanol", "CCO"), ("acetic_acid", "CC(=O)O"), ("TFE", "OCC(F)(F)F"), ("fluoroform", "FC(F)F")]
PFAS = [("PFBA", "OC(=O)C(F)(F)C(F)(F)C(F)(F)F"), ("PFPeA", "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
        ("PFHxA", "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
        ("PFHpA", "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F")]


def cfg(tmp_path, **kw):
    base = dict(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, threads=2, gpu="off",
                output_dir=str(tmp_path / "runs"), formats=("xyz", "sdf"))
    base.update(kw)
    return pipeline.BuildConfig(**base)


def inputs(mols=MOLS):
    return [from_smiles(s, id=i) for i, s in mols]


def report(physical=12, logical=24, ram=64.0, gpus=0, backend="rocm"):
    gpu_list = [resources.GPUInfo(i, "Radeon RX 9060 XT", backend, 16.0) for i in range(gpus)]
    return resources.HardwareReport(os="test", python="3.12", cpu_name="test", physical_cores=physical,
                                    logical_cores=logical, ram_gb=ram, wsl=False, torch_version="2.9",
                                    torch_backend=backend if gpus else "cpu", gpus=gpu_list)


@pytest.fixture
def cores(monkeypatch):
    """Pretend to be a 12-core/24-thread machine (resolve_threads uses logical_cores)."""
    monkeypatch.setattr(resources, "logical_cores", lambda: 24)
    monkeypatch.setattr(resources, "physical_cores", lambda: 12)


def lane_processes():
    """Live children of this process except multiprocessing's resource tracker (one per parent, by design)."""
    out = []
    for p in psutil.Process().children(recursive=True):
        try:
            if p.status() == psutil.STATUS_ZOMBIE or "resource_tracker" in " ".join(p.cmdline()):
                continue
        except psutil.Error:
            continue
        out.append(p.pid)
    return out


def summary_rows(run_dir):
    with open(run_dir / "summary.csv", newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


# ------------------------------------------------------------------------------------- lane specs
def test_parse_simple_specs(cores):
    c = pipeline.BuildConfig(threads=12)
    assert L.parse_lanes("1", c) == [Lane("main", "auto", 12)]
    assert L.parse_lanes(None, c) == L.parse_lanes("", c) == L.parse_lanes(1, c)
    assert L.parse_lanes("6", c) == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 7)]
    assert [x.threads for x in L.parse_lanes("5", c)] == [2] * 5           # 12 // 5, never more than asked
    assert [x.threads for x in L.parse_lanes(24, c)] == [1] * 24           # at least one thread each
    with pytest.raises(ValueError, match="logical cores"):
        L.parse_lanes("25", c)
    with pytest.raises(ValueError, match="cannot read"):
        L.parse_lanes("cpu:2y3", c)
    with pytest.raises(ValueError, match="at least one lane"):
        L.parse_lanes("0", c)


def test_parse_explicit_lists(cores):
    c = pipeline.BuildConfig(threads=12, gpu_index=0)
    lanes = L.parse_lanes("gpu:1x4,cpu:5x2", c, report(gpus=1))
    assert lanes == [Lane("gpu0", "cuda:0", 4)] + [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 6)]
    # GPU lanes go first wherever they are written; items without threads share what is left
    lanes = L.parse_lanes("cpu:2,gpu1:2x3", c, report(gpus=2))
    assert [(x.name, x.device, x.threads) for x in lanes] == [
        ("gpu1", "cuda:1", 3), ("gpu1.2", "cuda:1", 3), ("cpu1", "cpu", 3), ("cpu2", "cpu", 3)]
    assert L.parse_lanes("3x2", c) == [Lane(f"cpu{i}", "cpu", 2) for i in (1, 2, 3)]
    assert L.parse_lanes("gpu", c, report(gpus=1, backend="mps")) == [Lane("mps", "mps", 12)]
    with pytest.raises(ValueError, match="no usable GPU"):
        L.parse_lanes("gpu:1x4,cpu:4x2", c, report(gpus=0))
    with pytest.raises(ValueError, match="switched off"):
        L.parse_lanes("gpu:1x4", pipeline.BuildConfig(gpu="off"), report(gpus=1))
    with pytest.raises(ValueError, match="does not exist"):
        L.parse_lanes("gpu3", c, report(gpus=1))
    with pytest.raises(ValueError, match="more than the 24 logical cores"):
        L.parse_lanes("cpu:13x2", c)
    assert "1 GPU lane (cuda:0, 4 threads) + 5 CPU lanes (2 threads each) = 14 threads" == \
        L.describe(L.parse_lanes("gpu:1x4,cpu:5x2", c, report(gpus=1)))


@pytest.mark.parametrize("hw", [report(), report(gpus=1), report(physical=4, logical=8, ram=8), report(1, 2, 7),
                                report(physical=16, logical=16, ram=4.0, gpus=1)])
@pytest.mark.parametrize("threads", [0, 1, 3, 12])
@pytest.mark.parametrize("method,extra", [("mmff", {}), ("gfn1-xtb", {}), ("gfn1-xtb", {"qm_method": "b3lyp/def2-svp"}),
                                          ("gfn1-xtb", {"conformer_search": "crest"})])
def test_auto_never_oversubscribes(monkeypatch, hw, threads, method, extra):
    monkeypatch.setattr(resources, "logical_cores", lambda: hw.logical_cores)
    c = pipeline.BuildConfig(method=method, threads=threads, gpu="auto", **extra)
    lanes, reason = L.plan_auto(c, hw)
    total = resources.resolve_threads(threads)
    assert sum(x.threads for x in lanes) <= total and all(x.threads >= 1 for x in lanes), (lanes, reason)
    assert len({x.name for x in lanes}) == len(lanes)
    assert len(lanes) == 1 or not any(x.device == "auto" for x in lanes)


def test_auto_plans(cores, monkeypatch):
    nova = report(gpus=1)
    monkeypatch.setattr(L, "gpu_linalg_available", lambda: True)    # independent of this machine's torch
    mmff = L.plan_auto(pipeline.BuildConfig(method="mmff", threads=12), nova)[0]
    assert mmff == [Lane(f"cpu{i}", "cpu", 1) for i in range(1, 13)]             # lanes = physical cores
    cpu_xtb = L.plan_auto(pipeline.BuildConfig(method="gfn1-xtb", threads=12, gpu="off"), nova)[0]
    assert cpu_xtb == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 7)]
    gpu_xtb, reason = L.plan_auto(pipeline.BuildConfig(method="gfn1-xtb", threads=0), nova)
    assert gpu_xtb[0] == Lane("gpu0", "cuda:0", 4) and {x.threads for x in gpu_xtb[1:]} == {2}
    assert len(gpu_xtb) == 5 and "GPU lane" in reason       # 4 GPU-lane threads + 4 x 2 on the other 8 cores
    monkeypatch.setattr(L, "gpu_linalg_available", lambda: False)    # AMD's Windows ROCm wheels: no MAGMA
    no_linalg, reason = L.plan_auto(pipeline.BuildConfig(method="gfn1-xtb", threads=12), nova)
    assert no_linalg == cpu_xtb and "linear algebra" in reason
    external = L.plan_auto(pipeline.BuildConfig(method="mmff", threads=12, conformer_search="crest"), nova)[0]
    assert external == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 7)]
    # Psi4 needs memory per job: 8 GB of RAM leaves room for one 4 GB job, so auto stays at one lane
    qm, reason = L.plan_auto(pipeline.BuildConfig(method="mmff", threads=12, qm_method="b3lyp/def2-svp"),
                             report(ram=8.0))
    assert qm == [Lane("main", "auto", 12)] and "memory" in reason
    # settings.resources.parallel_molecules fixes the number of lanes
    pm = L.plan_auto(pipeline.BuildConfig(method="mmff", threads=12), nova, parallel_molecules=3)[0]
    assert pm == [Lane(f"cpu{i}", "cpu", 4) for i in (1, 2, 3)]
    from quantabench import settings as sm

    s = sm.Settings()
    s.resources.parallel_molecules = 4
    sm.save(s)
    assert len(L.parse_lanes("auto", pipeline.BuildConfig(method="mmff", threads=12), nova)) == 4
    # the settings dialog's text never raises and does not import torch for the GPU
    assert L.plan_text("gpu:1x4", pipeline.BuildConfig(threads=12)).startswith("Not usable here")
    assert "6 CPU lanes" in L.plan_text("6", pipeline.BuildConfig(threads=12))


# ------------------------------------------------------------------------------------ parallel runs
def test_two_lanes_give_the_same_results_as_one(tmp_path):
    c = cfg(tmp_path, max_conformers=4, refine_top=2)
    one, _ = jobs.run_resumable(inputs(), c, tmp_path / "one")
    logs, done, seen = [], [], []
    two, out = jobs.run_resumable(inputs(), c, tmp_path / "two", log=logs.append, on_result=done.append,
                                  progress=lambda i, n, stage, ident: seen.append((i, n, stage, ident)),
                                  lanes=L.parse_lanes("2", c))
    assert [r.input.id for r in two] == [r.input.id for r in one]                  # returned in input order
    assert all(r.status == "ok" for r in two), [r.error for r in two]
    for a, b in zip(one, two):
        assert [x.energy_ev for x in a.conformers] == pytest.approx([x.energy_ev for x in b.conformers], abs=1e-9)
        assert b.mol is not None and b.mol.GetNumConformers() == a.mol.GetNumConformers()   # the Mol pickles
        assert b.descriptors.keys() == a.descriptors.keys()
    assert sorted(r.input.id for r in done) == sorted(i for i, _ in MOLS)        # on_result once per molecule
    assert any(m.startswith("Lanes: 2 CPU lanes") for m in logs)
    assert any(m.startswith("[cpu") and "MMFF94s" in m for m in logs)          # lane logs, prefixed
    finished = [s for s in seen if s[2] == "ok"]
    assert [s[0] for s in finished] == [1, 2, 3, 4] and all(s[1] == 4 for s in seen)
    assert set(summary_rows(out)) == {i for i, _ in MOLS}
    assert sorted(p.name for p in (out / "TFE").iterdir()) == sorted(p.name for p in (tmp_path / "one" / "TFE").iterdir())
    assert not (out / jobs.LOCK_NAME).exists()
    assert jobs.open_run(out).counts["ok"] == 4
    assert lane_processes() == []


def test_lane_crash_is_counted_and_the_run_goes_on(tmp_path, monkeypatch):
    """A lane that dies (os._exit stands in for a segfault/OOM kill) is restarted; its molecule is tried again
    at the end and marked failed at MAX_CRASHES, like a crash of the single-process loop."""
    monkeypatch.setenv(L.TEST_CRASH_ENV, "acetic_acid")
    c = cfg(tmp_path)
    logs = []
    results, out = jobs.run_resumable(inputs(), c, tmp_path / "run", log=logs.append, lanes=L.parse_lanes("2", c))
    by_id = {r.input.id: r for r in results}
    assert by_id["acetic_acid"].status == "failed" and by_id["acetic_acid"].error == "crashed the process 2 times"
    assert all(by_id[i].status == "ok" for i in ("ethanol", "TFE", "fluoroform"))
    assert sum("lane died (exit code 86)" in m for m in logs) == jobs.MAX_CRASHES
    st = jobs.open_run(out)
    assert st.counts["ok"] == 3 and st.failures == [("acetic_acid", "crashed the process 2 times")]
    with sqlite3.connect(out / jobs.DB_NAME) as con:
        assert con.execute("SELECT crashes FROM molecules WHERE id='acetic_acid'").fetchone()[0] == 2
    assert summary_rows(out)["acetic_acid"]["status"] == "failed"
    assert lane_processes() == []
    # once the cause is gone, --retry-failed redoes it (single process here)
    monkeypatch.delenv(L.TEST_CRASH_ENV)
    jobs.run_resumable(None, c, out, retry_failed=True)
    assert jobs.open_run(out).counts["ok"] == 4


def test_cancel_reaches_running_lanes_and_leaves_no_processes(tmp_path):
    """Cancel while lanes are in the conformer search: their molecules come back cancelled, nothing new
    starts, every lane process ends; a parallel resume finishes the rest."""
    c = cfg(tmp_path, max_conformers=300, refine_top=1, rmsd_threshold=0.01)
    stop = {"now": False}
    started = []

    def progress(i, n, stage, ident):
        if stage == "start":
            started.append(ident)
        if stage == "conformers":
            stop["now"] = True

    run_dir = tmp_path / "run"
    results, _ = jobs.run_resumable(inputs(PFAS), c, run_dir, progress=progress, should_stop=lambda: stop["now"],
                                    lanes=L.parse_lanes("2", c))
    assert lane_processes() == []
    assert results and all(r.status in ("ok", "cancelled") for r in results)
    assert any(r.status == "cancelled" for r in results)          # the stop Event reached a running lane
    assert len(started) <= 2                                      # nothing new started after the cancel
    st = jobs.open_run(run_dir)
    assert st.counts["running"] == 0 and st.remaining == len(PFAS) - st.counts["ok"] and not st.active
    with sqlite3.connect(run_dir / jobs.DB_NAME) as con:
        assert con.execute("SELECT MAX(crashes) FROM molecules").fetchone()[0] == 0

    # (fewer conformers now, so the rest is quick: force accepts the changed settings)
    results, _ = jobs.run_resumable(None, cfg(tmp_path), run_dir, force=True, lanes=L.parse_lanes("2", c))
    assert jobs.open_run(run_dir).counts["ok"] == len(PFAS)
    assert set(summary_rows(run_dir)) == {i for i, _ in PFAS}
    assert lane_processes() == []


def test_interrupted_parallel_run_resumes(tmp_path):
    """A KeyboardInterrupt (or any exception) in the parent stops every lane; the molecules they held are
    'cancelled', not crashed, and the next call finishes the run."""
    c = cfg(tmp_path)
    run_dir = tmp_path / "run"

    def on_result(res):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        jobs.run_resumable(inputs(), c, run_dir, on_result=on_result, lanes=L.parse_lanes("2", c))
    assert lane_processes() == [] and not (run_dir / jobs.LOCK_NAME).exists()
    with sqlite3.connect(run_dir / jobs.DB_NAME) as con:
        rows = dict(con.execute("SELECT status, COUNT(*) FROM molecules GROUP BY status").fetchall())
        assert rows.get("ok") == 1 and "running" not in rows
        assert con.execute("SELECT MAX(crashes) FROM molecules").fetchone()[0] == 0
    logs = []
    results, _ = jobs.run_resumable(None, c, run_dir, log=logs.append)
    assert len(results) == 3 and not any("interrupted by a crash" in m for m in logs)
    assert jobs.open_run(run_dir).counts["ok"] == 4


needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                   reason="needs a display (xvfb-run)")


@needs_display
def test_settings_dialog_parallel_lanes():
    from PySide6.QtWidgets import QApplication

    from quantabench import settings as sm
    from quantabench.gui.settings_dialog import SettingsDialog

    app = QApplication.instance() or QApplication([])
    dlg = SettingsDialog(sm.Settings(), None)
    assert dlg.lanes.currentText() == "1" and dlg.lane_plan.text().startswith("1 lane (one process)")
    dlg.lanes.setCurrentText("2")
    assert "2 CPU lanes" in dlg.lane_plan.text()
    assert dlg.result_settings().dist.lanes == "2"
    dlg.lanes.setCurrentText("gpu:1x4")      # well formed: stored, though no GPU has been detected here
    assert "Not usable here" in dlg.lane_plan.text() and "not detected yet" in dlg.lane_plan.text()
    assert dlg.result_settings().dist.lanes == "gpu:1x4"
    dlg.lanes.setCurrentText("two")          # malformed: the plan line says why, the setting is kept
    assert "cannot read" in dlg.lane_plan.text()
    assert dlg.result_settings().dist.lanes == "gpu:1x4"
    dlg.reject()
    dlg.deleteLater()
    app.processEvents()


@needs_display
def test_gui_build_worker_with_two_lanes(tmp_path):
    """The GUI's BuildWorker (a QThread) spawns the lanes and gets one molecule_done per molecule."""
    import time

    from PySide6.QtWidgets import QApplication, QWidget

    from quantabench.gui import workers

    app = QApplication.instance() or QApplication([])
    owner = QWidget()
    worker = workers.BuildWorker(inputs(), cfg(tmp_path), lanes="2")
    got, logs, finished = [], [], []
    worker.molecule_done.connect(lambda r: got.append(r.input.id))
    worker.log.connect(logs.append)
    worker.finished.connect(lambda results, run_dir: finished.append((results, run_dir)))
    worker.failed.connect(lambda msg: finished.append(msg))
    workers.start(worker, owner)
    t = time.time()
    while not (finished and not owner.__dict__.get("_live_workers")) and time.time() - t < 120:
        app.processEvents()
        time.sleep(0.02)
    assert finished and not isinstance(finished[0], str), finished
    results, run_dir = finished[0]
    assert sorted(got) == sorted(i for i, _ in MOLS) and all(r.status == "ok" for r in results)
    assert any(m.startswith("Lanes: 2 CPU lanes") for m in logs)
    assert jobs.open_run(run_dir).counts["ok"] == 4
    assert lane_processes() == []
