"""The ``quantabench-cli worker`` on this computer, as an update sees it (DIST-1).

An update changes the files under a running worker: a job that imports a module after the pull mixes the old
code with the new (docs/WORKERS.md). So Help > Check for updates and ``quantabench-cli update`` ask here:

    find(repo) -> LocalWorker | None
        The worker that holds this computer's worker lock (``<state folder>/worker.lock``, ``dist.worker``), when it
        is alive: its process, whether it runs from ``repo`` (its working folder, its interpreter or the venv
        launcher it came from lies in the checkout), its jobs (``jobs.json``: running, queued, paused), whether the
        machine is paused (``pause.json``) and what keeps it running: the macOS LaunchAgent
        (``LAUNCHD_LABEL``), the Windows scheduled task (``WINDOWS_TASK``, docs/WORKERS.md), or nothing ("manual").
        Never raises; None when no worker runs (or the lock cannot be read).

    restart(worker) -> (ok, message)
        Restarts an idle worker that a service keeps running: ``launchctl kickstart -k`` (launchd stops it with
        SIGTERM, which ends it after the current molecule, and starts it again), or ``schtasks /End`` then
        ``/Run`` (Task Scheduler ends the process at once, so only when it is idle). It waits for the new worker
        to take the lock. A worker started by hand, or a busy one, is not touched: the message says what to do.

Only the default state folder is looked at (``QUANTABENCH_WORKER_STATE`` or the platform's), so a worker started
with ``--state-dir`` elsewhere is not found. No Qt here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

LAUNCHD_LABEL = "io.github.brandonbtm29.quantabench-worker"
WINDOWS_TASK = "QuantaBench worker"
LOCK = "worker.lock"
JOBS = "jobs.json"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass
class LocalWorker:
    pid: int
    state_dir: Path
    cwd: str = ""
    exe: str = ""
    venv: str = ""                  # the venv it runs from, when it can be told ('' otherwise)
    same_checkout: bool | None = None   # None: could not tell (another user's process, access denied)
    running: list[str] = field(default_factory=list)   # job labels (or ids)
    queued: list[str] = field(default_factory=list)
    paused_jobs: list[str] = field(default_factory=list)
    machine_paused: bool = False
    manager: str = "manual"         # launchd | task | manual

    @property
    def busy(self) -> bool:
        """A job runs, or one waits and the machine is not paused (it would start any moment)."""
        return bool(self.running) or (bool(self.queued) and not self.machine_paused)

    @property
    def affected(self) -> bool:
        """An update of the checkout changes this worker's code (or it cannot be told that it does not)."""
        return self.same_checkout is not False

    @property
    def can_restart(self) -> bool:
        return self.affected and not self.busy and self.manager in ("launchd", "task")

    def describe(self) -> str:
        """One line: what it is and what it does."""
        how = {"launchd": "LaunchAgent", "task": f"scheduled task '{WINDOWS_TASK}'"}.get(self.manager,
                                                                                         "started by hand")
        where = f", from {self.venv}" if self.venv else ""
        jobs = []
        if self.running:
            jobs.append(f"{len(self.running)} job(s) running")
        if self.queued:
            jobs.append(f"{len(self.queued)} queued")
        if self.paused_jobs:
            jobs.append(f"{len(self.paused_jobs)} paused")
        state = ("has " + ", ".join(jobs)) if jobs else "is idle"
        return (f"The worker on this computer (pid {self.pid}, {how}{where}) {state}"
                + ("; the machine is paused" if self.machine_paused else "") + ".")

    def busy_reason(self) -> str:
        """Why an update has to wait ('' when it need not)."""
        if not self.affected or not self.busy:
            return ""
        what = (f"{len(self.running)} job(s) running" if self.running else "") + \
               (" and " if self.running and self.queued else "") + \
               (f"{len(self.queued)} waiting" if self.queued else "")
        names = ", ".join((self.running + self.queued)[:3])
        return (f"The worker on this computer runs this checkout and has {what} ({names}). Updating now would "
                "change the code under them. Let them finish, or pause the machine (Machines dock, or "
                "`quantabench-cli worker pause`) and update when the running molecules are done.")

    def restart_hint(self) -> str:
        """What the user does to restart it (the app does it for a service; see ``restart``)."""
        if self.manager == "launchd":
            return f"launchctl kickstart -k gui/{_uid()}/{LAUNCHD_LABEL}"
        if self.manager == "task":
            return f'Stop-ScheduledTask "{WINDOWS_TASK}"; Start-ScheduledTask "{WINDOWS_TASK}"'
        return ("stop it with Ctrl+C in its window (or `pkill -INT -f \"quantabench-cli worker\"`) when it is idle, "
                "then start it again")


def _uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else 0


def _run(cmd: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",  # noqa: S603
                          timeout=timeout, stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)


def state_dir() -> Path:
    from .dist.worker import default_state_dir

    return default_state_dir()


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _jobs(folder: Path) -> tuple[list[str], list[str], list[str]]:
    """(running, queued, paused) job labels from the worker's saved job list."""
    data = _read_json(folder / JOBS) or _read_json(folder / (JOBS + ".bak")) or {}
    out: dict[str, list[str]] = {"running": [], "queued": [], "paused": []}
    for job in data.get("jobs") or []:
        if isinstance(job, dict) and job.get("state") in out:
            out[job["state"]].append(str(job.get("label") or job.get("id") or "?"))
    return out["running"], out["queued"], out["paused"]


def _inside(path: str | os.PathLike | None, root: Path) -> bool:
    if not path:
        return False
    try:
        p = Path(path).resolve()
        r = Path(root).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return p == r or r in p.parents


def _venv_of(path: str) -> str:
    """The venv folder an interpreter path lies in ('' when it is not in one)."""
    if not path:
        return ""
    p = Path(path)
    for parent in list(p.parents)[:3]:
        if (parent / "pyvenv.cfg").is_file():
            return str(parent)
    return ""


def _describe_process(pid: int) -> dict:
    """cwd, exe, the venv launcher and command line of ``pid`` (what psutil can read; never raises)."""
    import psutil

    out: dict = {"cwd": "", "exe": "", "cmd0": "", "launcher": "", "parent_exe": "", "readable": False}
    try:
        proc = psutil.Process(pid)
    except (psutil.Error, ValueError):
        return out
    for key, getter in (("cwd", proc.cwd), ("exe", proc.exe)):
        try:
            out[key] = getter() or ""
            out["readable"] = True
        except (psutil.Error, OSError):
            pass
    try:
        cmd = proc.cmdline()
        out["cmd0"] = cmd[0] if cmd else ""
    except (psutil.Error, OSError):
        pass
    try:   # Windows: a venv's python.exe is a launcher that starts the base interpreter and says so here
        out["launcher"] = proc.environ().get("__PYVENV_LAUNCHER__", "")
    except (psutil.Error, OSError, AttributeError):
        pass
    try:
        parent = proc.parent()
        out["parent_exe"] = parent.exe() if parent is not None else ""
    except (psutil.Error, OSError):
        pass
    return out


def _alive(info: dict | None) -> bool:
    from .dist.worker import _lock_owner_alive

    try:
        return _lock_owner_alive(info)
    except Exception:
        return False


def detect_manager(pid: int, run: Runner = _run, platform: str | None = None) -> str:
    """'launchd' when the LaunchAgent runs ``pid``, 'task' when the Windows scheduled task is running, else
    'manual'. Never raises."""
    platform = platform or sys.platform
    try:
        if platform == "darwin":
            cp = run(["launchctl", "print", f"gui/{_uid()}/{LAUNCHD_LABEL}"])
            if cp.returncode == 0 and any(line.strip() == f"pid = {pid}" for line in (cp.stdout or "").splitlines()):
                return "launchd"
        elif platform == "win32":
            cp = run(["schtasks", "/Query", "/TN", WINDOWS_TASK, "/FO", "CSV", "/NH"])
            if cp.returncode == 0 and '"Running"' in (cp.stdout or ""):
                return "task"
    except (OSError, subprocess.SubprocessError):
        pass
    return "manual"


def find(repo: Path | None = None, folder: Path | None = None, run: Runner = _run,
         platform: str | None = None) -> LocalWorker | None:
    """The live worker of this computer (see the module docstring), or None. Never raises."""
    try:
        folder = Path(folder) if folder is not None else state_dir()
        info = _read_json(folder / LOCK)
        if not _alive(info):
            return None
        pid = int(info["pid"])
        proc = _describe_process(pid)
        paths = [proc["cwd"], proc["exe"], proc["cmd0"], proc["launcher"], proc["parent_exe"]]
        same: bool | None = None
        if repo is not None:
            same = any(_inside(p, repo) for p in paths) if proc["readable"] else None
        venv = next((v for v in (_venv_of(proc["launcher"]), _venv_of(proc["cmd0"]), _venv_of(proc["parent_exe"]),
                                 _venv_of(proc["exe"])) if v), "")
        running, queued, paused = _jobs(folder)
        from .dist import control as control_mod

        machine_paused = control_mod.read_pause(folder)["paused"]
        return LocalWorker(pid=pid, state_dir=folder, cwd=proc["cwd"], exe=proc["exe"], venv=venv,
                           same_checkout=same, running=running, queued=queued, paused_jobs=paused,
                           machine_paused=machine_paused, manager=detect_manager(pid, run, platform))
    except Exception:          # a probe must never stop an update check
        return None


def _wait_gone(pid: int, timeout: float) -> bool:
    import psutil

    try:
        psutil.Process(pid).wait(timeout)
    except psutil.NoSuchProcess:
        return True
    except psutil.TimeoutExpired:
        return False
    except psutil.Error:
        return True
    return True


def _wait_new(folder: Path, old_pid: int, timeout: float) -> int | None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        info = _read_json(folder / LOCK)
        if info and _alive(info) and int(info.get("pid", -1)) != old_pid:
            return int(info["pid"])
        time.sleep(0.5)
    return None


def restart(worker: LocalWorker, run: Runner = _run, wait: float = 60.0, platform: str | None = None,
            wait_gone: Callable[[int, float], bool] = _wait_gone,
            wait_new: Callable[[Path, int, float], int | None] = _wait_new) -> tuple[bool, str]:
    """Restart an idle worker kept by launchd or the scheduled task; see the module docstring. Never raises."""
    platform = platform or sys.platform
    if worker.busy:
        return False, worker.busy_reason() or "The worker is busy: restart it when its jobs are done."
    if worker.manager not in ("launchd", "task"):
        return False, f"The worker on this computer was started by hand: {worker.restart_hint()}."
    try:
        if worker.manager == "launchd" and platform == "darwin":
            cp = run(["launchctl", "kickstart", "-k", f"gui/{_uid()}/{LAUNCHD_LABEL}"], timeout=60)
            if cp.returncode != 0:
                return False, f"launchctl kickstart failed: {(cp.stderr or cp.stdout).strip()}"
        elif worker.manager == "task" and platform == "win32":
            cp = run(["schtasks", "/End", "/TN", WINDOWS_TASK], timeout=60)
            if cp.returncode != 0:
                return False, f"schtasks /End failed: {(cp.stderr or cp.stdout).strip()}"
            if not wait_gone(worker.pid, 30.0):
                return False, (f"The worker (pid {worker.pid}) did not stop; start it again with "
                               f'Start-ScheduledTask "{WINDOWS_TASK}" once it has.')
            cp = run(["schtasks", "/Run", "/TN", WINDOWS_TASK], timeout=60)
            if cp.returncode != 0:
                return False, (f"The worker stopped, but schtasks /Run failed: {(cp.stderr or cp.stdout).strip()}. "
                               f'Start it with Start-ScheduledTask "{WINDOWS_TASK}".')
        else:
            return False, f"Cannot restart a {worker.manager} worker on {platform}: {worker.restart_hint()}."
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not restart the worker ({type(exc).__name__}: {exc}): {worker.restart_hint()}."
    new = wait_new(worker.state_dir, worker.pid, wait)
    if new is None:
        return False, (f"The worker was restarted but has not come back within {wait:.0f} s; check its log "
                       "(docs/WORKERS.md) or the Machines dock.")
    return True, f"Restarted the worker on this computer (pid {worker.pid} -> {new}); it runs the new code."
