"""Run priority and free threads (quantabench.priority, PRIO-1): the per-machine settings, the thread budgets, the
lane plan and dynamic CREST threads with free threads, how run_program starts a program per OS, the WSL wrapper,
and real POSIX checks that lanes, programs and their children run with a raised nice value.

The test process itself is never lowered (conftest: ``_keep_the_test_process_priority``): what lowers a whole
process runs in a child process here."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from quantabench import priority as P
from quantabench import resources
from quantabench import settings as sm
from quantabench.chem import external, jobs, pipeline
from quantabench.chem.inputs import from_smiles
from quantabench.dist import lanes as L
from quantabench.dist import threads as T
from quantabench.dist import worker as W
from quantabench.dist.client import config_for_workers
from quantabench.dist.lanes import Lane

POSIX = os.name != "nt"
LINUX = sys.platform.startswith("linux")
posix_only = pytest.mark.skipif(not POSIX, reason="nice values are POSIX")
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def report(physical=12, logical=24, ram=64.0, perf=0):
    return resources.HardwareReport(os="test", python="3.12", cpu_name="test", physical_cores=physical,
                                    logical_cores=logical, ram_gb=ram, wsl=False, torch_version="",
                                    torch_backend="cpu", gpus=[], performance_cores=perf)


NOVA = dict(physical=12, logical=24, ram=64.0)
M3_MAX = dict(physical=14, logical=14, ram=36.0, perf=10)


def crest_cfg(**kw):
    return pipeline.BuildConfig(**{"method": "tblite-gfn2", "threads": 0, "conformer_search": "crest", **kw})


def machine(monkeypatch, logical, physical):
    monkeypatch.setattr(resources, "logical_cores", lambda: logical)
    monkeypatch.setattr(resources, "physical_cores", lambda: physical)


def nice_of(pid: int) -> int:
    return os.getpriority(os.PRIO_PROCESS, pid)


# ---------------------------------------------------------------------------------- settings, per machine
def test_defaults_and_budgets():
    """Defaults: priority low, 1 thread free (the old 'all but one'); background leaves 25 % (at least 2)."""
    s = sm.Settings()
    assert s.resources.priority == "low" and s.resources.free_threads == 1
    assert P.DEFAULT_PRIORITY == "low" and pipeline.BuildConfig().priority == "low"
    assert P.effective_free_threads(1, "low", 24) == 1 and P.auto_thread_cap(1, "normal", 24) == 23
    assert P.effective_free_threads(1, "background", 24) == 6 and P.auto_thread_cap(1, "background", 24) == 18
    assert P.effective_free_threads(1, "background", 14) == 4                     # M3 Max: ceil(3.5)
    assert P.effective_free_threads(1, "background", 4) == 2                      # at least 2
    assert P.effective_free_threads(8, "background", 24) == 8                     # a larger setting wins
    assert P.effective_free_threads(99, "low", 8) == 7                            # never every thread
    assert P.effective_free_threads(-1, "low", 24) == 1                           # negative: the default
    assert P.thread_budget(0, 1, "low", 24) == 23 and P.thread_budget(0, 1, "low") == resources.resolve_threads(0)
    assert P.thread_budget(0, 4, "low", 24) == 20 and P.thread_budget(12, 4, "low", 24) == 12
    assert P.thread_budget(64, 1, "low", 24) == 24                                # explicit: as before
    assert P.check("Background") == "background" and P.normalize("turbo") == "low"
    with pytest.raises(ValueError, match="normal, low, background"):
        P.check("turbo")
    assert P.lower_of("low", "background") == "background" and P.lower_of("low", "normal") == "low"


def test_priority_and_free_threads_round_trip_per_machine(isolated_settings, monkeypatch):
    """Like the CREST path and the lanes: each machine keeps its own value in the shared file, and the defaults of
    every machine (resources.priority, resources.free_threads) are never written."""
    monkeypatch.setattr(sm, "_this_machine", lambda: "BTM-NOVA")
    s = sm.load()
    s.set_priority_for("background")
    s.set_free_threads_for(4)
    s.set_priority_for("normal", "Brandons-MacBook-Pro")
    sm.save(s)
    raw = json.loads((sm.settings_dir() / sm.SETTINGS_FILE).read_text(encoding="utf-8"))["resources"]
    assert raw["priority"] == "low" and raw["free_threads"] == 1                  # the shared keys: untouched
    assert raw["machine_priority"] == {"btm-nova": "background", "brandons-macbook-pro": "normal"}
    assert raw["machine_free_threads"] == {"btm-nova": 4}
    again = sm.load()
    assert again.priority_for() == "background" and again.free_threads_for() == 4
    assert again.priority_for(scope=True) == ("background", "this machine")
    assert again.priority_for("brandons_macbook_pro") == "normal"                 # names as in machine_lanes
    assert again.free_threads_for("brandons-macbook-pro", scope=True) == (1, "all machines")
    assert again.priority_for("linux-box", scope=True) == ("low", "all machines")
    cfg = pipeline.BuildConfig.from_settings(again)
    assert (cfg.priority, cfg.free_threads) == ("background", 4)
    # the default of every machine is not stored for a machine; "default" goes back to it
    again.set_priority_for("low")
    again.set_free_threads_for("default")
    assert "btm-nova" not in again.resources.machine_priority and again.resources.machine_free_threads == {}
    with pytest.raises(ValueError):
        again.set_priority_for("turbo")
    with pytest.raises(ValueError):
        again.set_free_threads_for(-2)


def test_save_keeps_another_machines_priority(isolated_settings, monkeypatch):
    """A GUI left open on NOVA does not write back its old copy of the Mac's entries."""
    monkeypatch.setattr(sm, "_this_machine", lambda: "btm-nova")
    nova = sm.load()                                     # NOVA's GUI opens the file
    monkeypatch.setattr(sm, "_this_machine", lambda: "brandons-macbook-pro")
    mac = sm.load()
    mac.set_priority_for("background")
    mac.set_free_threads_for(3)
    sm.save(mac)                                         # the Mac saves its own
    monkeypatch.setattr(sm, "_this_machine", lambda: "btm-nova")
    nova.set_priority_for("normal")
    sm.save(nova)                                        # NOVA saves later
    again = sm.load()
    assert again.resources.machine_priority == {"brandons-macbook-pro": "background", "btm-nova": "normal"}
    assert again.resources.machine_free_threads == {"brandons-macbook-pro": 3}


def test_bad_priority_values_fall_back(isolated_settings):
    (sm.settings_dir()).mkdir(parents=True, exist_ok=True)
    (sm.settings_dir() / sm.SETTINGS_FILE).write_text(json.dumps({"resources": {
        "priority": "turbo", "free_threads": "many",
        "machine_priority": {"nova": "warp", "mac": 3, "box": "Background"},
        "machine_free_threads": {"nova": -3, "mac": "2", "box": 5}}}), encoding="utf-8")
    s = sm.load()
    assert s.priority_for("nova") == "low" and s.priority_for("mac") == "low" and s.priority_for("box") == "background"
    assert s.free_threads_for("nova") == 1 and s.free_threads_for("mac") == 1 and s.free_threads_for("box") == 5


def test_settings_cli_sets_this_machine_and_others(isolated_settings, monkeypatch, capsys):
    from quantabench import cli

    monkeypatch.setattr(sm, "_this_machine", lambda: "btm-nova")
    assert cli.main(["settings", "--priority", "background", "--free-threads", "2"]) == 0
    out = capsys.readouterr().out
    assert "priority background (set for this machine); free threads 2 (set for this machine)" in out
    assert cli.main(["settings", "--priority", "normal", "--machine", "Brandons-MacBook-Pro"]) == 0
    s = sm.load()
    assert s.resources.machine_priority == {"btm-nova": "background", "brandons-macbook-pro": "normal"}
    assert s.resources.machine_free_threads == {"btm-nova": 2} and s.resources.priority == "low"
    assert cli.main(["settings", "--priority", "default", "--free-threads", "default"]) == 0
    s = sm.load()
    assert s.resources.machine_priority == {"brandons-macbook-pro": "normal"} and not s.resources.machine_free_threads
    capsys.readouterr()
    assert cli.main(["settings", "--free-threads", "-1"]) == 2
    assert "0 or more" in capsys.readouterr().err


# ----------------------------------------------------------------------------- lane plan, free threads
def test_lane_plans_keep_todays_layout_by_default(monkeypatch):
    """The default (1 free thread) is today's plan: NOVA 11 x 2 = 22 of 24 for CREST, the M3 Max 3 x 4."""
    machine(monkeypatch, 24, 12)
    assert L.plan_auto(crest_cfg(), report(**NOVA))[0] == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 12)]
    assert L.plan_auto(crest_cfg(free_threads=2), report(**NOVA))[0] == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 12)]
    assert "left free" not in L.plan_auto(crest_cfg(), report(**NOVA))[1]
    assert L.parse_lanes("1", crest_cfg()) == [Lane("main", "auto", 23)]
    machine(monkeypatch, 14, 14)
    mac = report(**M3_MAX)
    assert L.plan_auto(crest_cfg(), mac)[0] == [Lane(f"cpu{i}", "cpu", 4) for i in (1, 2, 3)]
    assert L.plan_auto(crest_cfg(free_threads=2), mac)[0] == [Lane(f"cpu{i}", "cpu", 4) for i in (1, 2, 3)]
    assert len(L.plan_auto(crest_cfg(conformer_search="rdkit"), mac)[0]) == 13          # tblite: 13 x 1


def test_lane_plans_leave_the_free_threads(monkeypatch):
    machine(monkeypatch, 24, 12)
    nova = report(**NOVA)
    lanes, reason = L.plan_auto(crest_cfg(free_threads=4), nova)
    assert lanes == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 11)] and "4 of 24 threads left free" in reason
    lanes, reason = L.plan_auto(crest_cfg(priority="background"), nova)       # 25 % of 24 = 6 free: 9 x 2
    assert lanes == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 10)]
    assert "6 of 24 threads left free (background)" in reason
    # an explicit thread count is used as given, as before (the free threads are for the automatic budget)
    assert sum(x.threads for x in L.plan_auto(crest_cfg(threads=24, free_threads=4), nova)[0]) == 24
    assert L.plan_auto(crest_cfg(threads=12, free_threads=4), nova)[0] == [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 7)]
    # MMFF: one lane per physical core, within the budget
    assert len(L.plan_auto(pipeline.BuildConfig(method="mmff", free_threads=16), nova)[0]) == 8
    # threads 0 = the budget for "1" and "N" too; an explicit list is used as given
    assert L.parse_lanes("1", crest_cfg(free_threads=8)) == [Lane("main", "auto", 16)]
    assert L.parse_lanes("4", crest_cfg(free_threads=8)) == [Lane(f"cpu{i}", "cpu", 4) for i in range(1, 5)]
    assert sum(x.threads for x in L.parse_lanes("6x4", crest_cfg(free_threads=8), nova)) == 24
    assert "= 18 threads of 18" in L.plan_text("auto", crest_cfg(priority="background"), nova, 0)
    machine(monkeypatch, 14, 14)
    mac = report(**M3_MAX)
    assert L.plan_auto(crest_cfg(priority="background"), mac)[0] == [Lane("cpu1", "cpu", 4), Lane("cpu2", "cpu", 4)]


# ------------------------------------------------------------------------- dynamic threads, free threads
def test_dynamic_threads_never_borrow_the_free_threads():
    two = [Lane("cpu1", "cpu", 2), Lane("cpu2", "cpu", 2)]
    # a 24-thread machine, 11 x 2 planned (22), two molecules: the default borrows up to the plan
    assert L.dynamic_threads_for(crest_cfg(), two, True, 22, logical=24) == 22
    # 6 threads free: the lanes the run does not use may lend only 18 - 4
    assert L.dynamic_threads_for(crest_cfg(free_threads=6), two, True, 22, logical=24) == 18
    assert L.dynamic_threads_for(crest_cfg(priority="background"), two, True, 22, logical=24) == 18
    # GPU lanes' threads count against the cap too
    gpu = [Lane("gpu0", "cuda:0", 4), *two]
    assert L.dynamic_threads_for(crest_cfg(free_threads=6), gpu, True, 18, logical=24) == 14
    # an explicit plan above the cap keeps its own lanes, and borrows nothing more
    six = [Lane(f"cpu{i}", "cpu", 4) for i in range(1, 7)]
    assert L.dynamic_threads_for(crest_cfg(free_threads=4), six, True, 24, logical=24) == 24
    assert L.dynamic_threads_for(crest_cfg(free_threads=4), six[:2], True, 24, logical=24) == 20
    # free threads leave no more than the lanes' own: nothing to borrow
    assert L.dynamic_threads_for(crest_cfg(free_threads=20), two, True, 22, logical=24) == 4
    # an explicit thread count keeps the plan's budget (as before the free threads existed)
    assert L.dynamic_threads_for(crest_cfg(threads=22, free_threads=6), two, True, 22, logical=24) == 22
    # the policy then never hands out more than that budget
    budget = L.dynamic_threads_for(crest_cfg(free_threads=6), two, True, 22, logical=24)
    dyn = T.DynamicThreads(budget, lambda m: 1000.0 if m == "big" else 1.0)
    width = dyn.width(2, "big", [])
    assert width == 18 and width <= 24 - 6
    dyn.start("cpu1", 2, width, "big")
    assert dyn.pool.in_use() == 18 and not dyn.can_start(2)      # the other lane waits for the threads


def test_one_molecule_run_borrows_only_up_to_the_cap(monkeypatch):
    """A run of one molecule on a plan of 11 lanes (jobs._run_todo's one-lane path) borrows within the free threads
    of this machine (24 threads here)."""
    machine(monkeypatch, 24, 12)
    lanes = [Lane(f"cpu{i}", "cpu", 2) for i in range(1, 12)]
    assert L.dynamic_threads_for(crest_cfg(free_threads=6), lanes[:1], True, 22) == 18


# ------------------------------------------------------------------------- how run_program starts programs
def test_popen_args_on_windows():
    cmd = ["C:\\xtb\\bin\\xtb.exe", "in.xyz", "--opt"]
    for level, cls in (("low", P.BELOW_NORMAL_PRIORITY_CLASS), ("background", P.IDLE_PRIORITY_CLASS)):
        prio = P.child_options(level, system="win32")
        argv, kw = external.popen_args(cmd, prio=prio, system="win32")
        assert argv == cmd and kw == {"creationflags": external.CREATE_NO_WINDOW | cls}
        assert prio.windows_io == (1 if level == "low" else 0)
    argv, kw = external.popen_args(cmd, prio=P.child_options("normal", system="win32"), system="win32")
    assert argv == cmd and kw == {"creationflags": external.CREATE_NO_WINDOW}
    # a process that already runs lower is never raised: 'low' asked in an idle-class process stays idle
    prio = P.child_options("low", system="win32", current="background")
    assert prio.creationflags == P.IDLE_PRIORITY_CLASS and prio.windows_io == 0


def test_popen_args_on_linux_and_macos():
    cmd = ["/opt/xtb/bin/xtb", "in.xyz"]
    argv, kw = external.popen_args(cmd, prio=P.child_options("low", system="linux", nice_now=0), system="linux")
    assert argv[:2] == ["/bin/sh", "-c"] and argv[3:] == cmd and kw == {"start_new_session": True}
    wrapper = argv[2]
    assert wrapper.startswith("ulimit -s unlimited") and wrapper.endswith('exec "$0" "$@"')
    assert "ionice -c 2 -n 7 -p $$" in wrapper and 'exec nice -n 10 "$0" "$@"' in wrapper
    # the thread already runs at nice 10: no second nice, the I/O priority is still set
    w10 = external.popen_args(cmd, prio=P.child_options("low", system="linux", nice_now=10), system="linux")[0][2]
    assert "nice -n" not in w10 and "ionice -c 2 -n 7" in w10
    wbg = external.popen_args(cmd, prio=P.child_options("background", system="linux", nice_now=10),
                              system="linux")[0][2]
    assert "ionice -c 3 -p $$" in wbg and 'exec nice -n 9 "$0" "$@"' in wbg
    # normal: exactly the old wrapper
    assert external.popen_args(cmd, prio=P.child_options("normal", system="linux"), system="linux")[0] == \
        ["/bin/sh", "-c", external.STACK_WRAPPER, *cmd]
    assert external.popen_args(cmd, system="linux")[0][2] == external.STACK_WRAPPER
    mac = external.popen_args(cmd, prio=P.child_options("background", system="darwin", taskpolicy=True),
                              system="darwin")[0][2]
    assert 'exec taskpolicy -d throttle nice -n 19 "$0" "$@"' in mac and "ionice" not in mac
    mac_low = external.popen_args(cmd, prio=P.child_options("low", system="darwin", taskpolicy=False),
                                  system="darwin")[0][2]
    assert "taskpolicy" not in mac_low and 'exec nice -n 10 "$0" "$@"' in mac_low


def test_wsl_wrapper_gets_nice_and_ionice():
    """Windows priorities do not reach the Linux VM: wsl.exe gets the class, the wrapper nice and ionice."""
    info = external.ProgramInfo("crest", "wsl:/home/b/micromamba/envs/qb-tools/bin/crest", "settings")
    cmd = [info.command()[0], "--cd", "/mnt/c/Users/Brand/job", *info.command()[1:], "in.xyz", "-T", "4"]
    prio = P.child_options("low", system="win32", via_wsl=True)
    argv, kw = external.popen_args(cmd, via_wsl=True, prio=prio, system="win32")
    assert kw == {"creationflags": external.CREATE_NO_WINDOW | P.BELOW_NORMAL_PRIORITY_CLASS}
    assert argv[:6] == ["wsl.exe", "--cd", "/mnt/c/Users/Brand/job", "-e", "sh", "-c"]
    wrapper = argv[6]
    assert wrapper.startswith(external.WSL_WRAPPER[:-len(P.EXEC_TAIL)]) and "echo $$ > .qb_pid" in wrapper
    assert "ionice -c 2 -n 7 -p $$" in wrapper and 'exec nice -n 10 "$0" "$@"' in wrapper
    assert argv[7:] == ["/home/b/micromamba/envs/qb-tools/bin/crest", "in.xyz", "-T", "4"]
    assert external.program_name(cmd) == "crest"            # the log and telemetry still name the program
    bg = external.popen_args(cmd, via_wsl=True, prio=P.child_options("background", system="win32", via_wsl=True),
                             system="win32")[0][6]
    assert "ionice -c 3 -p $$" in bg and 'exec nice -n 19 "$0" "$@"' in bg
    assert external.popen_args(cmd, via_wsl=True, system="win32")[0][6] == external.WSL_WRAPPER


@posix_only
def test_the_wsl_wrapper_runs_in_a_posix_shell(tmp_path):
    """The wrapper as WSL's sh runs it: the program keeps the shell's PID (.qb_pid, cancel) and gets the nice."""
    wrapper = P.child_options("background", system="win32", via_wsl=True).wrap(external.WSL_WRAPPER)
    code = "import os; print(os.getpid(), os.getpriority(os.PRIO_PROCESS, 0))"
    out = subprocess.run(["sh", "-c", wrapper, sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         text=True, check=True).stdout.split()
    assert out[0] == (tmp_path / ".qb_pid").read_text().strip()
    assert int(out[1]) == min(20 if sys.platform == "darwin" else 19, P.current_nice() + 19)   # macOS caps at 20


@posix_only
def test_wrapper_without_ionice_or_nice_still_runs(tmp_path, monkeypatch):
    """A machine without ionice (or nice) runs the program anyway."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sh", "cat"):
        os.symlink(shutil.which(name), bindir / name)
    wrapper = P.child_options("low", system="linux", nice_now=0).wrap(external.STACK_WRAPPER)
    (tmp_path / "f.txt").write_text("ran\n")
    out = subprocess.run([shutil.which("sh"), "-c", wrapper, "cat", "f.txt"], cwd=tmp_path, capture_output=True,
                         text=True, env={"PATH": str(bindir)}, check=False)
    assert out.returncode == 0 and out.stdout == "ran\n"


# ----------------------------------------------------------------------- real priorities (POSIX children)
CHILD = ("import os, subprocess, sys; print('child', os.getpriority(os.PRIO_PROCESS, 0)); "
         "subprocess.run([sys.executable, '-c', \"import os; print('grandchild', os.getpriority(os.PRIO_PROCESS, 0))\"]); "
         "import psutil; io = psutil.Process().ionice() if hasattr(psutil.Process, 'ionice') else None; "
         "print('io', int(io[0]) if io is not None and not isinstance(io, int) else -1, "
         "int(io[1]) if io is not None and not isinstance(io, int) else -1)")


def _run_child(tmp_path, level):
    with P.children_at(level):
        run = external.run_program([sys.executable, "-c", CHILD], tmp_path / level)
    vals = {}
    for line in run.text().splitlines():
        parts = line.split()
        if parts and parts[0] in ("child", "grandchild", "io"):
            vals[parts[0]] = [int(x) for x in parts[1:]]
    return vals


@posix_only
def test_programs_get_the_raised_nice_value_and_their_children_inherit_it(tmp_path):
    now = P.current_nice()
    low = _run_child(tmp_path, "low")
    assert low["child"] == [max(now, 10)] and low["grandchild"] == low["child"]       # CREST's xtb inherits it
    bg = _run_child(tmp_path, "background")
    assert bg["child"] == [19] and bg["grandchild"] == [19]
    assert _run_child(tmp_path, "normal")["child"] == [now]                           # normal: unchanged
    assert P.current_nice() == now                                                    # this process: unchanged
    if LINUX and shutil.which("ionice"):
        assert low["io"] == [2, 7] and bg["io"][0] == 3


@posix_only
def test_apply_to_process_lowers_every_thread_and_never_raises(tmp_path):
    """In a child Python (the test process stays as it is): a CLI build or a lane lowers itself, all its threads."""
    code = textwrap.dedent("""
        import os, threading, sys
        from quantabench import priority
        ev = threading.Event(); tid = []
        def wait():
            tid.append(threading.get_native_id()); ev.wait()
        t = threading.Thread(target=wait); t.start()
        while not tid: pass
        print("line", priority.apply_to_process("low"))
        print("main", os.getpriority(os.PRIO_PROCESS, 0))
        if sys.platform.startswith("linux"):
            print("thread", os.getpriority(os.PRIO_PROCESS, tid[0]))
        priority.apply_to_process("background")
        print("bg", os.getpriority(os.PRIO_PROCESS, 0), priority.in_force())
        print("again", priority.apply_to_process("low"))
        print("after", os.getpriority(os.PRIO_PROCESS, 0), priority.in_force())
        ev.set(); t.join()
    """)
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True,
                         timeout=60).stdout
    lines = {ln.split()[0]: ln.split(None, 1)[1] for ln in out.splitlines() if ln.strip()}
    base = P.current_nice()
    assert int(lines["main"]) == max(base, 10) and lines["line"].startswith("low")
    if LINUX:
        assert int(lines["thread"]) == max(base, 10)          # nice is per thread on Linux: every one is lowered
    assert lines["bg"].split() == ["19", "background"]
    assert lines["after"].split() == ["19", "background"] and "already ran at background" in lines["again"]


@posix_only
def test_a_lane_process_lowers_itself(tmp_path):
    """dist.lanes: a lane process applies the run's priority at start (its programs inherit it)."""
    cfg = pipeline.BuildConfig(method="mmff", threads=1, gpu="off", priority="background",
                               output_dir=str(tmp_path / "runs"))
    before = P.current_nice()
    pool = L.LanePool([Lane("cpu1", "cpu", 1)], cfg, tmp_path / "run")
    try:
        pool.start_all()
        w = pool.workers[0]
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not w.ready:
            for obj in pool.wait(0.2):
                if pool.owner(obj) is w:
                    msg = w.conn.recv()
                    if msg[1] == "ready":
                        w.ready = True
                    assert msg[1] != "fatal", msg
        assert w.ready
        assert nice_of(w.pid) == 19
        if LINUX and shutil.which("ionice"):
            import psutil

            assert int(psutil.Process(w.pid).ionice()[0]) == P.IOPRIO_CLASS_IDLE
    finally:
        pool.close(graceful=True)
    assert P.current_nice() == before                            # the parent (here: the test process) is untouched


@pytest.mark.skipif(not LINUX, reason="per-thread nice values are Linux's")
def test_thread_lowered_lowers_only_the_build_thread(tmp_path):
    """The GUI's build thread (gui.workers.BuildWorker): only that thread is lowered, its programs follow it."""
    main_before = P.current_nice()
    seen = {}

    def build_thread():
        with P.thread_lowered("low") as level:
            seen["level"] = level
            seen["thread"] = os.getpriority(os.PRIO_PROCESS, threading.get_native_id())
            seen["in_force"] = P.in_force()
            seen["child"] = _run_child(tmp_path, "low")["child"]
        seen["after"] = P.in_force()

    t = threading.Thread(target=build_thread)
    t.start()
    t.join(60)
    assert seen["level"] == "low" and seen["thread"] == max(main_before, 10) and seen["in_force"] == "low"
    assert seen["child"] == [max(main_before, 10)] and seen["after"] == "normal"
    assert P.current_nice() == main_before                    # the GUI thread (here: the test's) is untouched


# ----------------------------------------------------------------------------- worker, telemetry, CLI
def test_workers_use_their_own_priority_unless_the_client_sets_one(isolated_settings, monkeypatch):
    monkeypatch.setattr(sm, "_this_machine", lambda: "btm-nova")
    s = sm.load()
    s.set_priority_for("background")
    s.set_free_threads_for(3)
    sm.save(s)
    cfg = W.build_config({}, isolated_settings / "out")
    assert (cfg.priority, cfg.free_threads) == ("background", 3)
    assert W.build_config({}, isolated_settings / "out", priority="normal", free_threads=0).priority == "normal"
    assert W.build_config({"priority": "low"}, isolated_settings / "out").priority == "low"
    # a client leaves its own machine's priority out unless asked (grid build --priority)
    sent = config_for_workers(pipeline.BuildConfig(priority="normal", free_threads=5))
    assert "priority" not in sent and "free_threads" not in sent
    kept = config_for_workers(pipeline.BuildConfig(priority="normal"), keep_resources=("priority",))
    assert kept["priority"] == "normal"
    assert W.validate_config({"priority": "background", "free_threads": 2}) == {"priority": "background",
                                                                                 "free_threads": 2}
    with pytest.raises(W.RequestError, match="priority"):
        W.validate_config({"priority": "turbo"})
    with pytest.raises(W.RequestError):
        W.validate_config({"free_threads": -1})


def test_runs_record_the_priority_in_their_timings(tmp_path):
    import sqlite3

    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, threads=1,
                               gpu="off", output_dir=str(tmp_path / "runs"), formats=("xyz",),
                               priority="background", free_threads=0)
    logs: list[str] = []
    results, run_dir = jobs.run_resumable([from_smiles("CCO", id="ethanol"), from_smiles("CC", id="ethane")], cfg,
                                          tmp_path / "run", log=logs.append)
    assert all(r.status == "ok" for r in results)
    con = sqlite3.connect(run_dir / jobs.DB_NAME)
    config = json.loads(con.execute("SELECT config_json FROM telemetry_runs").fetchone()[0])
    con.close()
    assert config["priority"] == "background" and config["free_threads"] == 0
    assert config["free_threads_effective"] == P.effective_free_threads(0, "background")
    assert config["priority_effective"] == "background"
    assert any(line.startswith("Run priority: background") for line in logs)


def test_build_cli_takes_priority_and_free_threads(tmp_path, monkeypatch, capsys):
    from quantabench import cli

    seen = {}
    real = jobs.run_resumable

    def spy(inputs, cfg, run_dir, **kw):
        seen["cfg"] = cfg
        return real(inputs, cfg, run_dir, **kw)

    monkeypatch.setattr(jobs, "run_resumable", spy)
    code = cli.main(["build", "CCO ethanol", "CC ethane", "--method", "mmff", "--max-confs", "2", "--refine-top", "1",
                     "--threads", "1", "--gpu", "off", "--out", str(tmp_path / "runs"), "--priority", "background",
                     "--free-threads", "3"])
    assert code == 0
    assert (seen["cfg"].priority, seen["cfg"].free_threads) == ("background", 3)
    out = capsys.readouterr().out
    assert "Priority: background" in out and "Run priority: background" in out
    assert P._process_level == "background"           # conftest: recorded, the test process itself not lowered
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["build", "CCO", "--priority", "turbo"])


def test_doctor_reports_this_machines_priority(isolated_settings, monkeypatch, capsys):
    from quantabench import cli

    monkeypatch.setattr(sm, "_this_machine", lambda: "btm-nova")
    s = sm.load()
    s.set_priority_for("background")
    sm.save(s)
    info = cli.priority_info(sm.load(), 24)
    assert info == {"machine": "btm-nova", "priority": "background", "priority_scope": "this machine",
                    "free_threads": 1, "free_threads_scope": "all machines", "free_threads_effective": 6,
                    "auto_thread_budget": 18}
    line = cli.priority_lines(sm.load(), None, 24)
    assert line.startswith("priority background (set for this machine); free threads 1 (default of every machine)")
    assert "6 of 24 threads left free" in line and "up to 18" in line


def test_grid_and_worker_cli_take_the_options():
    from quantabench import cli

    p = cli.build_parser()
    a = p.parse_args(["grid", "build", "CCO", "--method", "mmff", "--priority", "low", "--free-threads", "2"])
    assert (a.priority, a.free_threads) == ("low", 2)
    a = p.parse_args(["worker", "--priority", "background"])
    assert a.priority == "background" and a.free_threads is None


# ------------------------------------------------------------------------------------------------ GUI
needs_display = pytest.mark.skipif(not (os.name == "nt" or sys.platform == "darwin" or os.environ.get("DISPLAY")),
                                   reason="needs a display (xvfb-run)")


@needs_display
def test_settings_dialog_run_priority_and_leave_free(isolated_settings, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from quantabench.gui.settings_dialog import SettingsDialog

    monkeypatch.setattr(sm, "_this_machine", lambda: "btm-nova")
    machine(monkeypatch, 24, 12)
    app = QApplication.instance() or QApplication([])
    dlg = SettingsDialog(sm.Settings(), None)
    assert dlg.priority.currentData() == "low" and dlg.free_threads.value() == 1
    assert "Below-normal CPU and low I/O" in dlg.priority_info.text()
    assert dlg.threads.specialValueText() == "Auto (23 of 24)"
    dlg.priority.setCurrentIndex(dlg.priority.findData("background"))
    assert "6 threads stay free" in dlg.priority_info.text() and dlg.threads.specialValueText() == "Auto (18 of 24)"
    dlg.lanes.setCurrentText("auto")
    assert "6 of 24 threads left free" in dlg.lane_plan.text()
    dlg.free_threads.setValue(3)
    s = dlg.result_settings()
    assert s.resources.machine_priority == {"btm-nova": "background"}
    assert s.resources.machine_free_threads == {"btm-nova": 3}
    assert (s.resources.priority, s.resources.free_threads) == ("low", 1)    # the shared defaults: untouched
    dlg.reject()
    dlg.deleteLater()
    app.processEvents()


@needs_display
@pytest.mark.skipif(not LINUX, reason="per-thread nice values are Linux's")
def test_gui_build_worker_lowers_only_its_own_thread(tmp_path, monkeypatch):
    """One molecule in the GUI runs in the BuildWorker's QThread: that thread is lowered, the GUI thread is not."""
    from PySide6.QtWidgets import QApplication, QWidget

    from quantabench.gui import workers

    seen = {}
    real = pipeline.run

    def spy(inputs, cfg, **kw):
        seen["nice"] = os.getpriority(os.PRIO_PROCESS, threading.get_native_id())
        seen["in_force"] = P.in_force()
        return real(inputs, cfg, **kw)

    monkeypatch.setattr(pipeline, "run", spy)
    app = QApplication.instance() or QApplication([])
    owner = QWidget()
    before = P.current_nice()
    cfg = pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, threads=1,
                               gpu="off", write_files=False, priority="low")
    worker = workers.BuildWorker([from_smiles("CCO", id="ethanol")], cfg)
    finished = []
    worker.finished.connect(lambda results, run_dir: finished.append(results))
    worker.failed.connect(finished.append)
    workers.start(worker, owner)
    t = time.time()
    while not (finished and not owner.__dict__.get("_live_workers")) and time.time() - t < 60:
        app.processEvents()
        time.sleep(0.02)
    assert finished and not isinstance(finished[0], str), finished
    assert seen == {"nice": max(before, 10), "in_force": "low"}
    assert P.current_nice() == before                         # the GUI thread keeps its priority
