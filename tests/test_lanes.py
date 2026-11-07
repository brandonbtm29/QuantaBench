"""Lanes: lane specs, and parallel runs in worker processes (dist.lanes + jobs.run_resumable).

Spawned lane processes take a second or two to start, so each test starts as few pools as it can."""

import contextlib
import csv
import os
import shutil
import sqlite3
import time

import psutil
import pytest
from conftest import HAS_DISPLAY

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


def report(physical=12, logical=24, ram=64.0, gpus=0, backend="rocm", perf=0):
    gpu_list = [resources.GPUInfo(i, "Radeon RX 9060 XT", backend, 16.0) for i in range(gpus)]
    return resources.HardwareReport(os="test", python="3.12", cpu_name="test", physical_cores=physical,
                                    logical_cores=logical, ram_gb=ram, wsl=False, torch_version="2.9",
                                    torch_backend=backend if gpus else "cpu", gpus=gpu_list, performance_cores=perf)


#: The MacBook: M3 Max, 10 performance + 4 efficiency cores, no SMT
M3_MAX = dict(physical=14, logical=14, ram=36.0, perf=10)


def fake_mac(monkeypatch, perf_level0=10, physical=14, machine="arm64"):
    """This machine is a Mac (sysctl answers hw.perflevel0.physicalcpu with ``perf_level0``, None = unreadable)."""
    monkeypatch.setattr(resources, "_perf_cores", None)              # detected again, restored after the test
    monkeypatch.setattr(resources, "_os_arch", lambda: ("darwin", machine))
    monkeypatch.setattr(resources, "_sysctl_int",
                        lambda name: perf_level0 if name == "hw.perflevel0.physicalcpu" else None)
    monkeypatch.setattr(resources, "logical_cores", lambda: physical)
    monkeypatch.setattr(resources, "physical_cores", lambda: physical)


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
                                report(physical=16, logical=16, ram=4.0, gpus=1), report(**M3_MAX),
                                report(**M3_MAX, gpus=1, backend="mps"), report(8, 8, 16.0, perf=4)])
@pytest.mark.parametrize("threads", [0, 1, 3, 12])
@pytest.mark.parametrize("method,extra", [("mmff", {}), ("tblite-gfn2", {}), ("gfn1-xtb", {}),
                                          ("gfn1-xtb", {"qm_method": "b3lyp/def2-svp"}),
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
    # tblite (no torch, CPU only): 1-thread lanes like MMFF
    assert L.workload(pipeline.BuildConfig(method="tblite-gfn2"))[1] == "native"
    tb, reason = L.plan_auto(pipeline.BuildConfig(method="tblite-gfn2", threads=12), nova)
    assert tb == [Lane(f"cpu{i}", "cpu", 1) for i in range(1, 13)] and "1 thread each" in reason
    tb24, reason = L.plan_auto(pipeline.BuildConfig(method="tblite-gfn2", threads=24), nova)   # SMT: 12 x 2
    assert tb24 == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 13)] and "SMT" in reason
    # a GPU-capable engine gets one GPU lane + CPU lanes on the rest
    gpu_ml, reason = L.plan_auto(pipeline.BuildConfig(method="aimnet2", threads=0), nova)
    assert gpu_ml[0] == Lane("gpu0", "cuda:0", 4) and {x.threads for x in gpu_ml[1:]} == {2}
    assert len(gpu_ml) == 5 and "GPU lane" in reason        # 4 GPU-lane threads + 4 x 2 on the other 8 cores
    # dxtb never mixes a float32 GPU lane with float64 CPU lanes: CPU lanes, or with dxtb GPU = on one GPU lane
    gpu_xtb, reason = L.plan_auto(pipeline.BuildConfig(method="gfn1-xtb", threads=12), nova)
    assert gpu_xtb == cpu_xtb and "one precision per run" in reason
    on, reason = L.plan_auto(pipeline.BuildConfig(method="gfn2-xtb", threads=12, dxtb_gpu="on"), nova)
    assert on == [Lane("main", "auto", 12)] and "same precision" in reason
    pm_xtb = L.plan_auto(pipeline.BuildConfig(method="gfn1-xtb", threads=12), nova, parallel_molecules=3)[0]
    assert {x.device for x in pm_xtb} == {"cpu"}
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


def crest_cfg(**kw):
    return pipeline.BuildConfig(**{"method": "tblite-gfn2", "threads": 0, "conformer_search": "crest", **kw})


def test_crest_auto_plan_on_apple_silicon(monkeypatch):
    """CREST on the M3 Max (10 performance + 4 efficiency cores): 3 lanes x 4 threads (measured 4x faster than
    the 6 x 2 of the 2-thread rule). The rule follows the thread setting and leaves other engines alone."""
    monkeypatch.setattr(resources, "logical_cores", lambda: 14)       # resolve_threads(0) = 13
    mac = report(**M3_MAX)
    lanes, reason = L.plan_auto(crest_cfg(), mac)
    assert lanes == [Lane(f"cpu{i}", "cpu", 4) for i in (1, 2, 3)], reason
    assert "4-thread lanes" in reason and "10 performance cores" in reason
    assert L.plan_auto(crest_cfg(threads=8), mac)[0] == [Lane("cpu1", "cpu", 4), Lane("cpu2", "cpu", 4)]
    assert L.plan_auto(crest_cfg(threads=3), mac)[0] == [Lane("main", "auto", 3)]        # room for one lane
    # 4 performance cores (M1): one lane, all the threads
    assert L.plan_auto(crest_cfg(threads=7), report(8, 8, 16.0, perf=4))[0] == [Lane("main", "auto", 7)]
    # an xtb engine with CREST conformers is planned for CREST too; without CREST nothing changes
    assert L.plan_auto(crest_cfg(method="xtb-gfn2"), mac)[0] == lanes
    tb = L.plan_auto(crest_cfg(conformer_search="rdkit"), mac)[0]
    assert tb == [Lane(f"cpu{i}", "cpu", 1) for i in range(1, 14)]                   # one lane per core
    assert L.plan_auto(pipeline.BuildConfig(method="xtb-gfn2", threads=0), mac)[0] == \
        [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 7)]                              # external: 2-thread rule
    # parallel_molecules (Compute settings) still fixes the number of lanes
    assert len(L.plan_auto(crest_cfg(), mac, parallel_molecules=6)[0]) == 6


def test_crest_auto_plan_elsewhere_keeps_the_two_thread_rule(monkeypatch):
    """A 12-core/24-thread x86 machine (NOVA) knows no performance cores: CREST keeps 2-thread lanes on all
    threads (23 threads: 11 x 2). On NOVA's WSL the layout mattered little (1 x 23 756 s, 4 x 5 769 s)."""
    monkeypatch.setattr(resources, "logical_cores", lambda: 24)
    lanes, reason = L.plan_auto(crest_cfg(), report())
    assert lanes == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 12)]
    assert reason.startswith("CREST: 2-thread lanes on all threads")
    assert L.plan_auto(crest_cfg(threads=12), report())[0] == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 7)]


def test_crest_auto_plan_from_this_machines_cores(monkeypatch):
    """plan_auto without a report reads this machine: sysctl's hw.perflevel0.physicalcpu on a Mac, the physical
    cores when an Apple Silicon Mac cannot say, none on an Intel Mac or a PC (no sysctl call there)."""
    fake_mac(monkeypatch)
    assert resources.performance_cores() == 10 and L.cpu_report().performance_cores == 10
    assert L.plan_auto(crest_cfg())[0] == [Lane(f"cpu{i}", "cpu", 4) for i in (1, 2, 3)]
    assert L.parse_lanes("auto", crest_cfg()) == [Lane(f"cpu{i}", "cpu", 4) for i in (1, 2, 3)]
    assert "3 CPU lanes (4 threads each) = 12 threads of 13" in L.plan_text("auto", crest_cfg())
    # unknown performance cores: the physical cores (8 here, 8 threads: 2 x 4)
    fake_mac(monkeypatch, perf_level0=None, physical=8)
    assert resources.performance_cores() == 8
    assert L.plan_auto(crest_cfg(threads=8))[0] == [Lane("cpu1", "cpu", 4), Lane("cpu2", "cpu", 4)]
    fake_mac(monkeypatch, perf_level0=None, machine="x86_64")                       # Intel Mac
    assert resources.performance_cores() == 0
    monkeypatch.setattr(resources, "_perf_cores", None)
    monkeypatch.setattr(resources, "_os_arch", lambda: ("linux", "x86_64"))

    def no_sysctl(name):
        raise AssertionError("sysctl asked on Linux")

    monkeypatch.setattr(resources, "_sysctl_int", no_sysctl)
    assert resources.performance_cores() == 0
    monkeypatch.setattr(resources, "logical_cores", lambda: 24)
    monkeypatch.setattr(resources, "physical_cores", lambda: 12)
    lanes, reason = L.plan_auto(crest_cfg())       # this machine's RAM may cap the 11 lanes
    assert {x.threads for x in lanes} == {2} and reason.startswith("CREST: 2-thread lanes on all threads")


def test_dispatch_order_is_largest_first():
    todo = [(i, None) for i in ("a", "b", "c", "d", "e")]
    size = {"a": 5, "b": 30, "c": 12, "d": 30, "e": 40}
    assert L.dispatch_order(todo, size.get) == [4, 1, 3, 2, 0]                 # e, then b and d in input order
    assert L.dispatch_order(todo, size.get, {"e": 1, "a": 2}) == [1, 3, 2, 4, 0]   # crashed ones last
    assert L.dispatch_order([], size.get) == []


def test_windows_caps_lanes_at_31(monkeypatch):
    """Windows waits on at most 63 handles and a pool waits on two per lane: 32+ lanes would fail every wait."""
    monkeypatch.setattr(resources, "logical_cores", lambda: 64)
    monkeypatch.setattr(resources, "physical_cores", lambda: 64)
    c = pipeline.BuildConfig(method="mmff", threads=64)
    monkeypatch.setattr(L, "_platform", lambda: "linux")
    assert len(L.parse_lanes("40", c)) == 40 and L.max_lanes() is None
    monkeypatch.setattr(L, "_platform", lambda: "win32")
    assert L.max_lanes() == 31 and len(L.parse_lanes("31", c)) == 31
    for spec in ("32", "cpu:40x1"):
        with pytest.raises(ValueError, match="31 one process can run on Windows"):
            L.parse_lanes(spec, c)
    lanes, reason = L.plan_auto(c, report(physical=64, logical=64))
    assert len(lanes) == 31 and "capped at 31 lanes" in reason
    assert len(L.plan_auto(c, report(physical=64, logical=64), parallel_molecules=50)[0]) == 31


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


def test_lanes_start_the_largest_molecules_first(tmp_path):
    """PFAS listed smallest first: the lanes get them largest first (the end of the batch is the short ones);
    the results still come back in input order and progress counts up in dispatch order."""
    c = cfg(tmp_path)
    started = []
    results, _ = jobs.run_resumable(inputs(PFAS), c, tmp_path / "run", lanes=L.parse_lanes("2", c),
                                    progress=lambda i, n, stage, ident: stage == "start" and started.append((i, ident)))
    assert [ident for _, ident in started] == ["PFHpA", "PFHxA", "PFPeA", "PFBA"]
    assert [i for i, _ in started] == [0, 1, 2, 3]
    assert [r.input.id for r in results] == [i for i, _ in PFAS] and all(r.status == "ok" for r in results)
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


def test_callback_errors_stop_the_run_instead_of_counting_crashes(tmp_path):
    """An OSError from a callback (print to a closed pipe, a full disk under the worker's job log) is not a
    lane death: it ends the run like in the single-process loop, with no crash counted and no lane left."""
    c = cfg(tmp_path)

    def log(msg):
        if msg.startswith("[cpu") and "MMFF94s" in msg:
            raise BrokenPipeError(32, "Broken pipe")

    run_dir = tmp_path / "run"
    with pytest.raises(BrokenPipeError):
        jobs.run_resumable(inputs(), c, run_dir, log=log, lanes=L.parse_lanes("2", c))
    assert lane_processes() == []
    with sqlite3.connect(run_dir / jobs.DB_NAME) as con:
        assert con.execute("SELECT MAX(crashes) FROM molecules").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM molecules WHERE status IN ('failed', 'running')").fetchone()[0] == 0
    results, _ = jobs.run_resumable(None, c, run_dir, lanes=L.parse_lanes("2", c))
    assert jobs.open_run(run_dir).counts["ok"] == 4 and lane_processes() == []


def test_lane_dead_while_idle_costs_no_crash(tmp_path, monkeypatch):
    """A molecule handed to a lane that already died (the send fails) goes back to the queue uncounted."""
    started = []
    real_start = L.LanePool.start

    class DeadOnSend:
        def __init__(self, conn, proc):
            self._conn, self._proc = conn, proc

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def send(self, obj):
            if obj is not None:
                self._proc.kill()
                self._proc.join(5)
                raise BrokenPipeError(32, "Broken pipe")
            return self._conn.send(obj)

    def start(self, w):
        real_start(self, w)
        if w.lane.name == "cpu1" and not started:
            started.append(w.lane.name)
            w.conn = DeadOnSend(w.conn, w.proc)

    monkeypatch.setattr(L.LanePool, "start", start)
    c = cfg(tmp_path)
    logs = []
    results, out = jobs.run_resumable(inputs(), c, tmp_path / "run", log=logs.append, lanes=L.parse_lanes("2", c))
    assert started and all(r.status == "ok" for r in results) and len(results) == 4
    assert any("lane died while idle" in m for m in logs)
    with sqlite3.connect(out / jobs.DB_NAME) as con:
        assert con.execute("SELECT MAX(crashes) FROM molecules").fetchone()[0] == 0
    assert lane_processes() == []


def _lane_that_dies_running_a_program(folder, marker):
    """A lane stand-in: starts an external program through run_program, then dies (segfault/OOM kill)."""
    import threading
    import time

    from quantabench.chem import external

    external.set_program_registry(folder)
    threading.Thread(target=lambda: external.run_program(["sleep", marker], folder, check=False),
                     daemon=True).start()
    time.sleep(1.5)
    os._exit(86)


def program_pids(marker):
    out = []
    for p in psutil.process_iter(["cmdline"]):
        with contextlib.suppress(psutil.Error):
            if p.info["cmdline"] and marker in p.info["cmdline"] and "sleep" in " ".join(p.info["cmdline"]):
                out.append(p.pid)
    return out


@pytest.mark.skipif(os.name == "nt" or not shutil.which("sleep"), reason="needs a POSIX sleep program")
def test_programs_of_a_dead_lane_are_killed(tmp_path):
    """External programs run in their own session and outlived a lane that died; the recorded ones are
    killed by the parent, records removed; a pid that now belongs to another process is left alone."""
    import multiprocessing

    marker = "41.25"
    folder = tmp_path / "programs"
    proc = multiprocessing.get_context("spawn").Process(target=_lane_that_dies_running_a_program,
                                                        args=(str(folder), marker), daemon=True)
    proc.start()
    proc.join(20)
    assert proc.exitcode == 86
    assert program_pids(marker) and len(list(folder.glob("*.json"))) == 1     # orphaned, and recorded
    from quantabench.chem import external

    assert external.kill_recorded_programs(folder) == 1
    deadline = time.monotonic() + 5
    while program_pids(marker) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert program_pids(marker) == [] and list(folder.glob("*.json")) == []
    # a program that ended normally leaves no record; a stale record of a reused pid kills nothing
    external.set_program_registry(folder)
    try:
        external.run_program(["true"], tmp_path, check=False)
    finally:
        external.set_program_registry(None)
    assert list(folder.glob("*.json")) == []
    (folder / "1.json").write_text('{"pid": 1, "started": 12345.0}')
    assert external.kill_recorded_programs(folder) == 0 and psutil.pid_exists(1)


def test_lane_death_kills_its_recorded_programs(tmp_path, monkeypatch):
    """lane_died asks the pool to kill what the dead lane recorded (and close() does for every lane)."""
    monkeypatch.setenv(L.TEST_CRASH_ENV, "acetic_acid")
    killed, dirs = [], set()
    real = L.LanePool.kill_programs

    def spy(self, w):
        killed.append(w.lane.name)
        dirs.add(self.programs_dir)
        assert self.programs_of(w).parent == self.programs_dir
        return real(self, w)

    monkeypatch.setattr(L.LanePool, "kill_programs", spy)
    c = cfg(tmp_path)
    jobs.run_resumable(inputs(), c, tmp_path / "run", lanes=L.parse_lanes("2", c))
    assert len(killed) >= jobs.MAX_CRASHES + 2          # each death, then every lane at close
    assert len(dirs) == 1 and not next(iter(dirs)).exists()              # the records folder is removed


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


# ------------------------------------------------------------------------------- speed-aware lanes
SMALL = [(f"m{i:02d}", smi) for i, smi in enumerate(
    ["CCO", "CC(=O)O", "OCC(F)(F)F", "FC(F)F", "CO", "CCC", "CC(F)F", "OC(F)(F)F", "CC#N", "CN", "CCN", "OCCO",
     "FCC(F)(F)F", "CC(C)O", "OC=O", "CCF"])]


def two_kinds():
    """Two kinds of lane (the thread count differs), so the speed-aware policy applies; the test hooks make
    cpu2 the slow one."""
    return [Lane("cpu1", "cpu", 1), Lane("cpu2", "cpu", 2)]


def seeds(monkeypatch, fast=0.001, slow=0.1):
    """A machine benchmark that says cpu2 is 100x slower than cpu1."""
    from quantabench.dist import speed

    monkeypatch.setattr(speed, "registry_seeds", lambda lanes, method, profile=None, solvent="": {"cpu1": fast, "cpu2": slow})


def record_decisions(monkeypatch) -> list:
    """(lane, molecules queued, took it, speed source) for every time the pool asks the policy."""
    from quantabench.dist import speed

    out = []
    real = speed.SpeedPolicy.allow

    def allow(self, name, sizes, states):
        sizes = list(sizes)
        ok = real(self, name, sizes, states)
        out.append((name, len(sizes), ok, self.source(name)))
        return ok

    monkeypatch.setattr(speed.SpeedPolicy, "allow", allow)
    return out


def test_speed_aware_slow_lane_leaves_the_batch_to_the_fast_lane(tmp_path, monkeypatch):
    """The benchmark says cpu2 is 100x slower: it leaves all 4 molecules to cpu1 (one log line says so), and
    the run completes."""
    seeds(monkeypatch)
    monkeypatch.setenv(L.TEST_DELAY_ENV, "cpu2=2.0")
    c = cfg(tmp_path)
    logs = []
    results, out = jobs.run_resumable(inputs(), c, tmp_path / "run", log=logs.append, lanes=two_kinds())
    assert len(results) == 4 and all(r.status == "ok" for r in results), [r.error for r in results]
    assert any(m.startswith("Lanes: ") and "speed-aware, benchmark speeds until measured" in m for m in logs)
    held = [m for m in logs if m.startswith("Speed-aware lanes:")]
    assert len(held) == 1 and "cpu2 (" in held[0] and "to cpu1 (" in held[0], held
    assert "Molecules per lane: cpu1 4" in logs
    assert any(m.startswith("Lane speeds: cpu1 ") and "(measured)" in m for m in logs)
    assert jobs.open_run(out).counts["ok"] == 4 and lane_processes() == []


def test_speed_aware_measured_slow_lane_stops_at_the_end(tmp_path, monkeypatch):
    """No benchmark: both lanes start at unknown speed and cpu2 takes molecules; once timed (~10x slower than
    cpu1) it never takes the last molecule, cpu1 is never held back, and the run completes."""
    monkeypatch.setenv(L.TEST_DELAY_ENV, "cpu1=0.2,cpu2=3.0")
    decisions = record_decisions(monkeypatch)
    c = cfg(tmp_path)
    logs = []
    results, out = jobs.run_resumable(inputs(SMALL), c, tmp_path / "run", log=logs.append, lanes=two_kinds())
    assert len(results) == len(SMALL) and all(r.status == "ok" for r in results), [r.error for r in results]
    assert any(name == "cpu2" and ok and src == "unknown" for name, _q, ok, src in decisions)    # helped early
    assert all(ok for name, _q, ok, _src in decisions if name == "cpu1")
    timed = [(q, ok) for name, q, ok, src in decisions if name == "cpu2" and src == "measured"]
    assert all(not ok for q, ok in timed if q <= 1), timed
    if any(not ok for _q, ok in timed):
        assert sum(m.startswith("Speed-aware lanes: cpu2") for m in logs) == 1
    per_lane = next(m for m in logs if m.startswith("Molecules per lane"))
    assert "cpu2" in per_lane and jobs.open_run(out).counts["ok"] == len(SMALL)
    assert lane_processes() == []


def test_speed_aware_fast_lane_crash_releases_the_slow_lane(tmp_path, monkeypatch):
    """cpu2 holds back for cpu1, then cpu1 dies running TFE: while cpu1 restarts, cpu2 takes the molecules
    held for it; TFE is counted and marked failed at the second crash; the run completes."""
    seeds(monkeypatch)
    monkeypatch.setenv(L.TEST_CRASH_ENV, "TFE")
    decisions = record_decisions(monkeypatch)
    c = cfg(tmp_path)
    logs = []
    results, out = jobs.run_resumable(inputs(), c, tmp_path / "run", log=logs.append, lanes=two_kinds())
    by_id = {r.input.id: r for r in results}
    assert by_id["TFE"].status == "failed" and by_id["TFE"].error == "crashed the process 2 times"
    assert all(by_id[i].status == "ok" for i in ("ethanol", "acetic_acid", "fluoroform"))
    assert any(name == "cpu2" and not ok for name, _q, ok, _src in decisions)     # held back first
    assert any(name == "cpu2" and ok for name, _q, ok, _src in decisions)         # released by the crash
    assert "cpu2" in next(m for m in logs if m.startswith("Molecules per lane"))
    assert sum("lane died (exit code 86)" in m for m in logs) == jobs.MAX_CRASHES
    st = jobs.open_run(out)
    assert st.counts["ok"] == 3 and st.failures == [("TFE", "crashed the process 2 times")]
    assert lane_processes() == []


def test_speed_aware_off_is_todays_dispatch(tmp_path, monkeypatch):
    """resources.speed_aware_lanes = off: the policy is never asked (every free lane takes the next
    molecule), even with a benchmark that would hold cpu2 back. Lanes of one kind never get a policy."""
    from quantabench import settings as sm
    from quantabench.dist import speed

    seeds(monkeypatch)

    def boom(*_a, **_k):
        raise AssertionError("the speed-aware policy was asked")

    monkeypatch.setattr(speed.SpeedPolicy, "allow", boom)
    s = sm.Settings()
    s.resources.speed_aware_lanes = False
    sm.save(s)
    c = cfg(tmp_path)
    logs = []
    results, _ = jobs.run_resumable(inputs(), c, tmp_path / "run", log=logs.append, lanes=two_kinds())
    assert all(r.status == "ok" for r in results) and len(results) == 4
    assert any(m.endswith("molecules go to whichever lane is free") for m in logs)
    assert not any(m.startswith(("Speed-aware", "Lane speeds")) for m in logs)
    assert L.speed_policy(L.parse_lanes("2", c), c) is None
    assert L.speed_policy(two_kinds(), c) is not None
    assert lane_processes() == []


def test_lane_past_the_time_limit_is_killed_and_restarted(tmp_path, monkeypatch):
    """Both lanes hold every molecule 60 s (TEST_DELAY_ENV) against a 2 s limit: the parent kills each lane at the
    limit, records its molecule failed "time limit (2 s)" (no crash counted, not retried in the run), restarts a
    lane for the third molecule, and the run completes. --retry-failed (no delay) then computes them."""
    monkeypatch.setenv(L.TEST_DELAY_ENV, "cpu1=60,cpu2=60")
    c = cfg(tmp_path, molecule_timeout_s=2.0)
    three = inputs(MOLS[:3])
    logs = []
    t0 = time.monotonic()
    results, out = jobs.run_resumable(three, c, tmp_path / "run", log=logs.append, lanes=L.parse_lanes("2", c))
    assert time.monotonic() - t0 < 40
    assert [r.status for r in results] == ["failed"] * 3
    assert all(r.error.startswith("time limit (2 s)") for r in results), [r.error for r in results]
    killed = [m for m in logs if "time limit (2 s)" in m and "lane killed" in m]
    assert len(killed) == 3 and any("restarting it" in m for m in killed)
    assert any(m.startswith("Per-molecule time limit: 2 s (a lane past it is killed") for m in logs)
    assert not any("lane died" in m for m in logs)
    with sqlite3.connect(out / jobs.DB_NAME) as con:
        assert con.execute("SELECT MAX(crashes) FROM molecules").fetchone()[0] == 0
    assert {r["status"] for r in summary_rows(out).values()} == {"failed"}
    assert summary_rows(out)["ethanol"]["error"].startswith("time limit (2 s)")
    assert lane_processes() == []
    # a resume without --retry-failed leaves them failed; with it (and no slow lanes) they are computed
    again, _ = jobs.run_resumable(None, c, out, lanes=L.parse_lanes("2", c))
    assert again == []
    monkeypatch.delenv(L.TEST_DELAY_ENV)
    redone, _ = jobs.run_resumable(None, c, out, retry_failed=True, lanes=L.parse_lanes("2", c))
    assert sorted(r.status for r in redone) == ["ok"] * 3
    assert jobs.open_run(out).counts["ok"] == 3 and lane_processes() == []


needs_display = pytest.mark.skipif(not HAS_DISPLAY,
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
    assert dlg.result_settings().lanes_for() == "2"
    assert dlg.result_settings().dist.lanes == "1"      # the shared default is left for the other machines
    dlg.lanes.setCurrentText("gpu:1x4")      # well formed: stored, though no GPU has been detected here
    assert "Not usable here" in dlg.lane_plan.text() and "not detected yet" in dlg.lane_plan.text()
    assert dlg.result_settings().lanes_for() == "gpu:1x4"
    dlg.lanes.setCurrentText("two")          # malformed: the plan line says why, the setting is kept
    assert "cannot read" in dlg.lane_plan.text()
    assert dlg.result_settings().lanes_for() == "gpu:1x4"
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


# ------------------------------------------------------------------- lanes auto from this machine's benchmark
def test_auto_gpu_lane_only_where_the_benchmark_says_it_is_faster(monkeypatch):
    from quantabench.dist import profile as profile_mod

    monkeypatch.setattr(L, "gpu_linalg_available", lambda: True)
    aimnet = pipeline.BuildConfig(method="aimnet2", threads=0)
    # the MacBook: AIMNet2 6,387/h on the 14-core CPU, 2,978/h on MPS -> CPU lanes only
    monkeypatch.setattr(resources, "logical_cores", lambda: 15)       # threads 0 = all but one: 14
    mac = report(physical=14, logical=15, ram=36.0, gpus=1, backend="mps")
    mac_bench = profile_mod.MachineProfile(name="mac", throughput={"aimnet2@cpu": 6387.0, "aimnet2@mps": 2978.0})
    lanes, reason = L.plan_auto(aimnet, mac, profile=mac_bench)
    assert all(x.device == "cpu" for x in lanes) and len(lanes) == 7, lanes      # 7 x 2 on the 14 cores
    assert "no GPU lane: benchmark aimnet2@mps 2,978/h vs @cpu 6,387/h" in reason, reason
    # NOVA: AIMNet2 about 10x faster on the RX 9060 XT -> its GPU lane
    monkeypatch.setattr(resources, "logical_cores", lambda: 24)
    nova = report(gpus=1)
    nova_bench = profile_mod.MachineProfile(name="nova", throughput={"aimnet2@cpu": 4100.0,
                                                                     "aimnet2@cuda:0": 41000.0})
    lanes, reason = L.plan_auto(aimnet, nova, profile=nova_bench)
    assert lanes[0] == Lane("gpu0", "cuda:0", 4) and all(x.device == "cpu" for x in lanes[1:])
    assert "(benchmark aimnet2@cuda:0 41,000/h vs @cpu 4,100/h)" in reason, reason
    # no benchmark (or only one of the two devices): today's GPU lane, and the reason says why
    for prof in (None, profile_mod.MachineProfile(name="nova", throughput={"aimnet2@cpu": 4100.0})):
        lanes, reason = L.plan_auto(aimnet, nova, profile=prof)
        assert lanes[0].device == "cuda:0" and "GPU lane by default" in reason, reason
    # water keys are not gas-phase keys: AIMNet2 has no water model, so its runs (and keys) are gas phase
    water = profile_mod.MachineProfile(name="nova", throughput={"aimnet2+water@cpu": 9.0,
                                                                "aimnet2+water@cuda:0": 1.0})
    lanes, reason = L.plan_auto(pipeline.BuildConfig(method="aimnet2", threads=0, solvent="water"), nova,
                                profile=water)
    assert lanes[0].device == "cuda:0" and "no benchmark aimnet2@cuda:0 and aimnet2@cpu" in reason, reason


def test_auto_reads_this_machines_registry(monkeypatch):
    from quantabench.dist import profile as profile_mod

    monkeypatch.setattr(resources, "logical_cores", lambda: 15)
    mac = report(physical=14, logical=15, ram=36.0, gpus=1, backend="mps")
    profile_mod.save_profile(profile_mod.MachineProfile(
        name=profile_mod.machine_name(), throughput={"aimnet2@cpu": 6387.0, "aimnet2@mps": 2978.0}))
    aimnet = pipeline.BuildConfig(method="aimnet2", threads=0)
    lanes, reason = L.plan_auto(aimnet, mac)
    assert all(x.device == "cpu" for x in lanes) and "no GPU lane: benchmark" in reason
    assert "no GPU lane: benchmark aimnet2@mps" in L.plan_text("auto", aimnet, mac, 0)
