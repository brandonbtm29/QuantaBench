"""CI hang watchdog (FIX-3): a separate process that watches the pytest process which started it.

A test that hangs inside Qt's C++ code while its thread holds Python's GIL is out of reach from inside the process:
pytest-timeout's thread needs the GIL to dump the stacks and exit, so on Windows such a hang ran into the job's 30 min
limit; faulthandler's dump (C, no GIL) shows only the Python frames, ending at the call into Qt (2026-11-16, Windows
and macOS: ``collapsible.Notes.set_notes`` -> ``setVisible``). This process has its own interpreter and GIL. When one
test has run longer than ``dump_after`` seconds it runs ``py-spy dump --native`` on the pytest process (every thread,
native and Python frames) into the job log, and after ``kill_after`` seconds it kills that process, so the job fails
in minutes and with the frames that matter.

Started by tests/conftest.py when ``QB_HANG_WATCHDOG=1`` (the CI workflow sets it and installs py-spy). Protocol on
stdin, one line each: ``start <test id>`` and ``end``; end of input (the pytest process closed the pipe or died) ends
the watchdog.

    python tests/hang_watchdog.py PID DUMP_AFTER_S KILL_AFTER_S
"""

from __future__ import annotations

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


def dump_command(pid: int) -> list[str] | None:
    """py-spy's dump of ``pid`` with native frames (``sudo -n`` where reading the parent needs root)."""
    exe = shutil.which("py-spy")
    if exe is None:
        return None
    cmd = [exe, "dump", "--native", "--pid", str(pid)]
    # macOS: reading another process needs root. Linux: Yama's ptrace_scope 1 (Ubuntu's default) lets a process read
    # only its descendants, and the watchdog is the pytest process's child. GitHub's runners have passwordless sudo
    if os.name != "nt" and os.geteuid() != 0 and shutil.which("sudo"):
        cmd = ["sudo", "-n", *cmd]
    return cmd


def dump(pid: int, test: str, seconds: float) -> None:
    log(f"{test} has run {seconds:.0f} s; py-spy dump --native of pid {pid}")
    cmd = dump_command(pid)
    if cmd is None:
        log("py-spy is not installed: no native stacks (pip install py-spy)")
        return
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        sys.stderr.write(out.stdout + out.stderr)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"py-spy failed: {type(exc).__name__}: {exc}")
    sys.stderr.flush()


def kill(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


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
