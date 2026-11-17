"""CI hang watchdog (FIX-3): a separate process that watches the pytest process which started it.

A test that hangs inside Qt's C++ code while its thread holds Python's GIL is out of reach from inside the process:
pytest-timeout's thread needs the GIL to dump the stacks and exit, so on Windows such a hang ran into the job's 30 min
limit; faulthandler's dump (C, no GIL) shows only the Python frames, ending at the call into Qt (2026-11-16, Windows
and macOS: ``collapsible.Notes.set_notes`` -> ``setVisible``). This process has its own interpreter and GIL. When one
test has run longer than ``dump_after`` seconds it runs ``py-spy dump --native`` on the pytest process (every thread,
native and Python frames; on macOS, where py-spy has no native mode, also the system's ``sample``) into the job log,
and after ``kill_after`` seconds it kills that process, so the job fails in minutes and with the frames that matter.

Started by tests/conftest.py when ``QB_HANG_WATCHDOG=1`` (the CI workflow sets it and installs py-spy). Protocol on
stdin, one line each: ``start <test id>`` and ``end``; end of input (the pytest process closed the pipe or died) ends
the watchdog.

    python tests/hang_watchdog.py PID DUMP_AFTER_S KILL_AFTER_S
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

POLL_S = 2.0


def log(text: str) -> None:
    sys.stderr.write(f"\n=== hang watchdog: {text} ===\n")
    sys.stderr.flush()


def _root(cmd: list[str]) -> list[str]:
    """``sudo -n`` where reading another process needs root. macOS: always. Linux: Yama's ptrace_scope 1 (Ubuntu's
    default) lets a process read only its descendants, and the watchdog is the pytest process's child. GitHub's
    runners have passwordless sudo; Windows needs neither."""
    if os.name != "nt" and os.geteuid() != 0 and shutil.which("sudo"):
        return ["sudo", "-n", *cmd]
    return cmd


def dump_commands(pid: int) -> list[list[str]]:
    """The commands that dump ``pid``'s stacks: py-spy (Python frames, which thread holds the GIL; with native frames
    where py-spy can read them: Linux and Windows) and, on macOS, where py-spy has no ``--native``, the system's
    ``sample`` (every thread's native call graph for 2 s)."""
    out = []
    exe = shutil.which("py-spy")
    if exe is not None:
        native = [] if sys.platform == "darwin" else ["--native"]
        out.append(_root([exe, "dump", *native, "--pid", str(pid)]))
    if sys.platform == "darwin" and shutil.which("sample"):
        out.append(_root(["sample", str(pid), "2"]))
    return out


def dump(pid: int, test: str, seconds: float) -> None:
    log(f"{test} has run {seconds:.0f} s; stacks of pid {pid} (py-spy dump --native, or sample on macOS)")
    cmds = dump_commands(pid)
    if not cmds:
        log("py-spy is not installed: no stacks (pip install py-spy)")
        return
    for cmd in cmds:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            sys.stderr.write(f"$ {' '.join(cmd)}\n{out.stdout}{out.stderr}")
        except (OSError, subprocess.SubprocessError) as exc:
            log(f"{cmd[0]} failed: {type(exc).__name__}: {exc}")
        sys.stderr.flush()


def kill(pid: int) -> None:
    """Kill the test process first, then the processes it started, except this watchdog (its child) and the
    watchdog's own children, and say so if the test process is still there 15 s later. On main's Windows CI
    (2026-11-16) the job went on for 20 min after "killing pid" (``taskkill /F /T``, run from inside that process
    tree): the process had aborted and was stuck in Windows' process exit (VTK's DLL teardown waiting for the GIL).
    Whether anything can kill it there is unknown; the next time the log will say."""
    try:
        import psutil
    except ImportError:                       # not in this environment: the test process alone
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)   # Windows: TerminateProcess
        return
    try:
        target = psutil.Process(pid)
        me = psutil.Process()
        mine = {me.pid} | {p.pid for p in me.children(recursive=True)}
        others = [p for p in target.children(recursive=True) if p.pid not in mine]
    except psutil.Error:
        return
    for proc in (target, *others):
        with contextlib.suppress(psutil.Error):
            proc.kill()
    gone, alive = psutil.wait_procs([target], timeout=15)
    if alive:
        log(f"pid {pid} is still there 15 s after the kill")


def main(argv: list[str]) -> int:
    pid, dump_after, kill_after = int(argv[1]), float(argv[2]), float(argv[3])
    state = {"test": None, "t0": 0.0, "dumped": False}
    lock = threading.Lock()

    def read() -> None:
        for line in sys.stdin:
            line = line.strip()
            with lock:
                if line.startswith("start "):
                    state.update(test=line[6:], t0=time.monotonic(), dumped=False)
                elif line == "end":
                    state["test"] = None
        os._exit(0)                       # the pytest process closed the pipe (or is gone)

    threading.Thread(target=read, daemon=True).start()
    while True:
        time.sleep(POLL_S)
        with lock:
            test, t0, dumped = state["test"], state["t0"], state["dumped"]
            if test is not None and not dumped and time.monotonic() - t0 > dump_after:
                state["dumped"] = True
                dumped = None             # dump below, outside the lock
        if test is None:
            continue
        seconds = time.monotonic() - t0
        if dumped is None:
            dump(pid, test, seconds)
        if seconds > kill_after:
            log(f"{test} has run {seconds:.0f} s; killing pid {pid}")
            kill(pid)
            return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
