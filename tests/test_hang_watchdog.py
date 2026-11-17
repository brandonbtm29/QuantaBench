"""FIX-3: the CI hang watchdog (tests/hang_watchdog.py) dumps the native stacks of a process stuck in C code that holds
the GIL (where pytest-timeout's thread cannot run) and kills it."""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

WATCHDOG = Path(__file__).with_name("hang_watchdog.py")
#: catastrophic backtracking: the regex engine runs in C for hours and never releases the GIL
STUCK = "import re; print('stuck', flush=True); re.match(r'(a+)+$', 'a' * 40 + 'b')"


def _run(dump_after: float, kill_after: float):
    target = subprocess.Popen([sys.executable, "-c", STUCK], stdout=subprocess.PIPE, text=True)
    assert target.stdout.readline().strip() == "stuck"
    dog = subprocess.Popen([sys.executable, str(WATCHDOG), str(target.pid), str(dump_after), str(kill_after)],
                           stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return target, dog


def test_the_watchdog_kills_a_process_stuck_in_c_and_says_why():
    target, dog = _run(dump_after=1, kill_after=4)
    try:
        dog.stdin.write("start tests/test_x.py::test_stuck\n")
        dog.stdin.flush()
        t = time.time()
        target.wait(timeout=60)                                   # killed by the watchdog, not by the test
        assert time.time() - t < 30
        err = dog.communicate(timeout=30)[1]
        assert "test_stuck has run" in err and "stacks of pid" in err and "killing pid" in err
        if shutil.which("py-spy"):
            assert "py-spy is not installed" not in err
    finally:
        for p in (target, dog):
            if p.poll() is None:
                p.kill()


#: as in CI: the stuck process started the watchdog itself, so the watchdog is its child
INSIDE = """import os, re, subprocess, sys
dog = subprocess.Popen([sys.executable, {watchdog!r}, str(os.getpid()), "60", "3"], stdin=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, text=True)
dog.stdin.write("start tests/test_x.py::test_inside\\n"); dog.stdin.flush()
print(dog.pid, flush=True)
re.match(r"(a+)+$", "a" * 40 + "b")
"""


def test_the_watchdog_kills_the_process_it_is_a_child_of():
    """FIX-5: as in CI, the watchdog is a child of the process it watches. It kills that process (first, then the other
    processes it started, never itself) and then ends, its input gone."""
    import psutil

    target = subprocess.Popen([sys.executable, "-c", INSIDE.format(watchdog=str(WATCHDOG))], stdout=subprocess.PIPE,
                              text=True)
    try:
        dog_pid = int(target.stdout.readline())
        t = time.time()
        target.wait(timeout=60)                                   # killed by its own watchdog child
        assert time.time() - t < 30

        def gone():
            try:
                return psutil.Process(dog_pid).status() == psutil.STATUS_ZOMBIE
            except psutil.Error:
                return True

        deadline = time.time() + 15
        while not gone() and time.time() < deadline:
            time.sleep(0.2)
        assert gone()                                             # and the watchdog ends with it
    finally:
        if target.poll() is None:
            target.kill()


def test_a_test_that_ends_in_time_is_left_alone():
    target, dog = _run(dump_after=3, kill_after=4)
    try:
        dog.stdin.write("start tests/test_x.py::test_quick\nend\n")
        dog.stdin.flush()
        time.sleep(7)
        assert target.poll() is None                             # still running: nobody killed it
        dog.stdin.close()                                        # end of input ends the watchdog
        assert dog.wait(timeout=10) == 0
        assert "hang watchdog" not in dog.stderr.read()
    finally:
        for p in (target, dog):
            if p.poll() is None:
                p.kill()


@pytest.mark.skipif(not shutil.which("py-spy"), reason="needs py-spy (CI installs it)")
def test_the_dump_has_the_native_frames():
    """The point of it: frames below the Python call (here the regex engine's C function). macOS: py-spy's Python
    frames and ``sample``'s native call graph."""
    target, dog = _run(dump_after=1, kill_after=8)
    try:
        dog.stdin.write("start tests/test_x.py::test_native\n")
        dog.stdin.flush()
        target.wait(timeout=60)
        err = dog.communicate(timeout=30)[1]
        if "Permission" in err or "sudo" in err.lower() and "password" in err.lower():
            pytest.skip("py-spy cannot read the process here (no root, ptrace_scope)")
        assert "Thread" in err and "match" in err                              # py-spy: the threads, the Python frame
        if sys.platform == "darwin":
            assert "Call graph" in err                                          # sample: the native frames
    finally:
        for p in (target, dog):
            if p.poll() is None:
                p.kill()
