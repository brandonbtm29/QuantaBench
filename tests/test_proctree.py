"""Programs cannot outlive QuantaBench, and probes cannot hang (ENG-3: ``quantabench.proctree``, ``chem.external``).

The hard-kill tests start a stand-in "GUI" (a Python process that runs ``external.run_program``), kill it outright
(TerminateProcess on Windows, SIGKILL on POSIX), and check that the stand-in engine and the process it started are
gone. Every process they start carries a ``qbtest-<id>`` tag in its command line.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import psutil
import pytest

import quantabench
from quantabench import proctree
from quantabench.chem import external

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell guard")
SRC = str(Path(quantabench.__file__).resolve().parent.parent)


def _tag() -> str:
    return f"qbtest-{uuid.uuid4().hex[:10]}"


def _script(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _read_pid(path: Path, timeout: float = 30.0) -> int:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"{path.name} never appeared")


def _running(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _wait_ended(pids, timeout: float = 15.0) -> list[int]:
    """The pids still running after up to ``timeout`` s ([] when all ended)."""
    t0 = time.monotonic()
    left = [pid for pid in pids if _running(pid)]
    while left and time.monotonic() - t0 < timeout:
        time.sleep(0.1)
        left = [pid for pid in left if _running(pid)]
    return left


def _kill_survivors(pids) -> None:
    for pid in pids:
        if _running(pid):
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass


#: the stand-in engine: writes its pid, starts a grandchild (which writes its own), then waits. With ``ignore_hup``
#: both ignore SIGHUP (a program WSL's hang-up does not stop).
ENGINE = '''
import os, signal, subprocess, sys, time
d, tag = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3 and hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
code = ("import os, signal, sys, time\\n"
        "if len(sys.argv) > 3 and hasattr(signal, 'SIGHUP'): signal.signal(signal.SIGHUP, signal.SIG_IGN)\\n"
        "open(sys.argv[1] + '.tmp', 'w').write(str(os.getpid())); os.replace(sys.argv[1] + '.tmp', sys.argv[1])\\n"
        "time.sleep(300)\\n")
subprocess.Popen([sys.executable, "-c", code, os.path.join(d, "grandchild.pid"), tag, *sys.argv[3:]])
open(os.path.join(d, "engine.pid.tmp"), "w").write(str(os.getpid()))
os.replace(os.path.join(d, "engine.pid.tmp"), os.path.join(d, "engine.pid"))
time.sleep(300)
'''

#: the stand-in GUI: runs the engine through external.run_program (in a thread, as the GUI's build worker does)
GUI = '''
import os, sys, threading, time
sys.path.insert(0, {src!r})
from quantabench import proctree
from quantabench.chem import external
d, cmd_file = sys.argv[1], sys.argv[2]
if os.environ.get("QBTEST_NO_JOB") == "1":
    proctree.Job.create = classmethod(lambda cls: None)
cmd = open(cmd_file, encoding="utf-8").read().split("\\x00")
open(os.path.join(d, "gui.pid"), "w").write(str(os.getpid()))
t = threading.Thread(target=external.run_program, args=(cmd, os.path.join(d, "job")), kwargs={{"timeout": 600}},
                     daemon=True)
t.start()
time.sleep(600)
'''


def _start_gui(tmp_path: Path, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
    gui = _script(tmp_path / "gui.py", GUI.format(src=SRC))
    (tmp_path / "cmd.txt").write_text("\x00".join(cmd), encoding="utf-8")
    return subprocess.Popen([sys.executable, str(gui), str(tmp_path), str(tmp_path / "cmd.txt")],
                            env={**os.environ, **(env or {})})


def _kill_gui_outright(tmp_path: Path, launcher: subprocess.Popen) -> None:
    """TerminateProcess / SIGKILL on the process that runs run_program (a venv's python.exe may be a launcher)."""
    psutil.Process(_read_pid(tmp_path / "gui.pid")).kill()
    try:
        launcher.wait(15)
    except subprocess.TimeoutExpired:
        launcher.kill()


# ----------------------------------------------------------------------------------------------- probes
HOLDER = '''
import os, subprocess, sys, time
# a launcher's child that inherits stdout and keeps it open long after the launcher has gone
subprocess.Popen([sys.executable, "-c", "import os, sys, time; open(sys.argv[1] + '.tmp', 'w').write(str(os.getpid())); "
                  "os.replace(sys.argv[1] + '.tmp', sys.argv[1]); time.sleep(120)", sys.argv[1], sys.argv[2]])
t0 = time.monotonic()
while not os.path.exists(sys.argv[1]) and time.monotonic() - t0 < 20:
    time.sleep(0.02)
print("* holder version 9.8.7 (stand-in)", flush=True)
'''


def test_probe_returns_while_a_child_keeps_stdout_open(tmp_path):
    """subprocess.run(capture_output=True, timeout=...) would wait for the child's 120 s; run_probe returns at once
    (the launcher has exited) and kills the child."""
    holder = _script(tmp_path / "holder.py", HOLDER)
    pid_file = tmp_path / "child.pid"
    t0 = time.monotonic()
    probe = proctree.run_probe([sys.executable, str(holder), str(pid_file), _tag()], timeout=10)
    assert time.monotonic() - t0 < 15
    assert probe.returncode == 0 and not probe.timed_out and "holder version 9.8.7" in probe.stdout
    child = _read_pid(pid_file)
    assert not _wait_ended([child], timeout=10), "the probe left its child running"


def test_probe_of_a_program_that_never_ends_keeps_its_output(tmp_path):
    hang = _script(tmp_path / "hang.py", 'import sys, time\nprint("xtb version 6.7.1 (stand-in)", flush=True)\n'
                                         'time.sleep(120)\n')
    t0 = time.monotonic()
    probe = proctree.run_probe([sys.executable, str(hang), _tag()], timeout=2)
    assert time.monotonic() - t0 < 12
    assert probe.timed_out and probe.returncode is None and "6.7.1" in probe.stdout


def test_version_probe_cannot_hang(tmp_path):
    """external._probe_version through a launcher whose child keeps the pipe: bounded by its timeout."""
    holder = _script(tmp_path / "holder.py", HOLDER)
    if sys.platform == "win32":
        exe = tmp_path / "xtb.bat"
        exe.write_text(f'@"{sys.executable}" "{holder}" "{tmp_path / "child.pid"}" {_tag()}\n', encoding="utf-8")
    else:
        exe = tmp_path / "xtb"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{holder}" "{tmp_path / "child.pid"}" {_tag()}\n')
        exe.chmod(0o755)
    info = external.ProgramInfo("xtb", str(exe), "test")
    spec = external.ProgramSpec("xtb", ("xtb",), ("--version",), r"holder version (\S+)")
    t0 = time.monotonic()
    assert external._probe_version(info, spec, timeout=10) == "9.8.7"
    assert time.monotonic() - t0 < 15


def test_probe_of_a_missing_program(tmp_path):
    probe = proctree.run_probe([str(tmp_path / "no_such_program")], timeout=5)
    assert probe.returncode is None and probe.error and not probe.timed_out


# ----------------------------------------------------------------------------------------------- Windows jobs
@windows_only
def test_programs_start_in_a_kill_on_close_job(tmp_path):
    proc = proctree.start([sys.executable, "-c", "import time; time.sleep(30)", _tag()])
    try:
        job = proctree.job_of(proc)
        assert job is not None and proc.pid in job.pids()
    finally:
        proctree.kill_tree(proc)
        proctree.release(proc)
    assert proc.returncode is not None


@windows_only
def test_kill_tree_reaches_a_child_whose_parent_has_exited(tmp_path):
    """taskkill /T walks parent links and misses an orphan; the job does not."""
    code = ("import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
            "'import os, sys, time; open(sys.argv[1], \"w\").write(str(os.getpid())); time.sleep(120)', "
            f"{str(tmp_path / 'orphan.pid')!r}, {_tag()!r}])")
    proc = proctree.start([sys.executable, "-c", code])
    orphan = _read_pid(tmp_path / "orphan.pid")
    proc.wait(30)                                   # the parent is gone, the child runs on in the job
    try:
        assert _running(orphan)
        proctree.kill_tree(proc)
        assert not _wait_ended([orphan], timeout=10)
    finally:
        _kill_survivors([orphan])
        proctree.release(proc)


@windows_only
def test_hard_killed_gui_takes_its_programs_along(tmp_path):
    """The ENG-3 check: a stand-in GUI runs a stand-in engine (which starts a grandchild) through run_program and is
    killed outright; Windows ends both when the dead process's job handle closes."""
    tag = _tag()
    engine = _script(tmp_path / "engine.py", ENGINE)
    gui = _start_gui(tmp_path, [sys.executable, str(engine), str(tmp_path), tag])
    procs = []
    try:
        for name in ("engine.pid", "grandchild.pid"):
            procs.append(_read_pid(tmp_path / name))
        assert all(_running(pid) for pid in procs)
        _kill_gui_outright(tmp_path, gui)
        assert not _wait_ended(procs, timeout=15), "the engine or its child outlived the killed GUI"
    finally:
        _kill_survivors(procs)
        if gui.poll() is None:
            gui.kill()


@windows_only
def test_adopted_lane_ends_with_its_parent(tmp_path):
    """dist.lanes: a running process (a multiprocessing lane) adopted into the parent's job, and what it starts, end
    when the parent is killed outright."""
    tag = _tag()
    engine = _script(tmp_path / "engine.py", ENGINE)
    parent_code = textwrap.dedent(f'''
        import os, subprocess, sys, time
        sys.path.insert(0, {SRC!r})
        from quantabench import proctree
        job = proctree.Job.create()
        p = subprocess.Popen([sys.executable, {str(engine)!r}, {str(tmp_path)!r}, {tag!r}])
        assert job.adopt(p.pid)
        open({str(tmp_path / "gui.pid")!r}, "w").write(str(os.getpid()))
        time.sleep(600)
    ''')
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    procs = []
    try:
        for name in ("engine.pid", "grandchild.pid"):
            procs.append(_read_pid(tmp_path / name))
        _kill_gui_outright(tmp_path, parent)
        assert not _wait_ended(procs, timeout=15)
    finally:
        _kill_survivors(procs)
        if parent.poll() is None:
            parent.kill()


# ----------------------------------------------------------------------------------------------- WSL
def _wsl_works() -> bool:
    if sys.platform != "win32" or shutil.which("wsl.exe") is None:
        return False
    try:
        return subprocess.run(["wsl.exe", "-e", "true"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _wsl_tagged(tag: str) -> list[str]:
    """Linux processes whose command line has ``tag`` (the [q] keeps pgrep from matching its own shell)."""
    out = subprocess.run(["wsl.exe", "-e", "sh", "-c", f"pgrep -af '[{tag[0]}]{tag[1:]}' || true"],
                         capture_output=True, text=True, timeout=60).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


#: a program that ignores SIGHUP, with a child that ignores it too: WSL's hang-up alone does not end them
WSL_PROGRAM = ('trap "" HUP; sh -c "trap \\"\\" HUP; while :; do sleep 1; done" "$0-child" & echo started > ready; '
               'wait')


@windows_only
@pytest.mark.skipif(not _wsl_works(), reason="needs a working WSL distribution")
@pytest.mark.parametrize("with_job", [True, False], ids=["job", "lifeline-only"])
def test_wsl_program_ends_with_a_hard_killed_gui(tmp_path, with_job):
    """A ``wsl:`` program (and its child) that ignore SIGHUP end when the GUI is killed outright: with the job, wsl.exe
    is ended and the guard's watchdog kills the group on WSL's hang-up; without a job (wsl.exe lives on), the
    lifeline's EOF does it."""
    tag = _tag()
    cmd = ["wsl.exe", "-e", "sh", "-c", external.WSL_WRAPPER, "sh", "-c", WSL_PROGRAM, tag]
    gui = _start_gui(tmp_path, cmd, env={"QBTEST_NO_JOB": "0" if with_job else "1"})
    try:
        t0 = time.monotonic()
        while not (tmp_path / "job" / "ready").is_file() and time.monotonic() - t0 < 60:
            time.sleep(0.1)
        assert (tmp_path / "job" / "ready").is_file(), "the program never started inside WSL"
        assert len(_wsl_tagged(tag)) >= 2
        _kill_gui_outright(tmp_path, gui)
        t0 = time.monotonic()
        left = _wsl_tagged(tag)
        while left and time.monotonic() - t0 < 30:
            time.sleep(1)
            left = _wsl_tagged(tag)
        assert not left, f"left running inside WSL: {left}"
    finally:
        subprocess.run(["wsl.exe", "-e", "sh", "-c", f"pkill -KILL -f '[{tag[0]}]{tag[1:]}' || true"],
                       capture_output=True, timeout=60)
        if gui.poll() is None:
            gui.kill()


# ----------------------------------------------------------------------------------------------- POSIX guard
@posix_only
def test_wsl_guard_lifeline_kills_the_group_when_the_gui_dies(tmp_path, monkeypatch):
    """The guard's lifeline on Linux, through a stand-in wsl.exe: the stand-in GUI is SIGKILLed, wsl.exe's stdin
    reaches EOF, and the watchdog kills the program and its child (both ignore SIGHUP)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "wsl.exe"
    fake.write_text('#!/bin/sh\n[ "$1" = "--cd" ] && { cd "$2"; shift 2; }\n[ "$1" = "-e" ] && shift\nexec "$@"\n')
    fake.chmod(0o755)
    tag = _tag()
    engine = _script(tmp_path / "engine.py", ENGINE)
    cmd = [*external.ProgramInfo("crest", "wsl:" + sys.executable, "test").command(), str(engine), str(tmp_path),
           tag, "ignore-hup"]
    gui = _start_gui(tmp_path, cmd, env={"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"})
    procs = []
    try:
        for name in ("engine.pid", "grandchild.pid"):
            procs.append(_read_pid(tmp_path / name))
        _kill_gui_outright(tmp_path, gui)
        assert not _wait_ended(procs, timeout=15), "the guard's watchdog did not end the program"
    finally:
        _kill_survivors(procs)
        if gui.poll() is None:
            gui.kill()


@posix_only
def test_wsl_guard_keeps_the_exit_code_and_ends_leftovers(tmp_path):
    """Normal end under the guard: the program's exit code comes back, and what it left in its group is ended."""
    left = tmp_path / "left.pid"
    script = f'sh -c "echo \\$\\$ > {left}; exec sleep 120" & sleep 0.3; exit 7'
    proc = subprocess.Popen(["sh", "-c", external.WSL_GUARD, "sh", "-c", script], stdin=subprocess.PIPE,
                            start_new_session=True)
    try:
        assert proc.wait(15) == 7
        assert not _wait_ended([_read_pid(left)], timeout=10), "the guard left the program's child running"
    finally:
        proc.stdin.close()
