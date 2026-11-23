"""Programs QuantaBench starts cannot outlive it, and short probes of them cannot hang (ENG-3).

**Windows: kill-on-close job objects.** ``start`` creates the program suspended, puts it in a job object of its own
with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, then lets it run, so it cannot start anything before it is in the job.
Whatever it starts later joins the job (a venv's ``python.exe`` launcher and the real interpreter, a conda ``.bat``
wrapper, Psi4's or MOPAC's helpers). Only this process holds the job's handle: when it ends in any way (a crash, End
task, a lane killed at its time limit) Windows closes the handle and ends every process in the job. ``release``
closes the handle once the program has ended, which also ends what it left running. The job sets no other limit:
priorities, affinity and memory stay as the program was started (``quantabench.priority``). Jobs nest (Windows 8 and
later): a lane process can be in its pool's job (``Job.adopt``) while each program it starts has its own.

A program through WSL: ``wsl.exe`` is in the job. When it ends, WSL hangs up the Linux session it started (SIGHUP),
and the guard script in ``chem.external`` (``WSL_GUARD``) also kills the program's process group, for programs that
ignore the hang-up or when ``wsl.exe`` outlives a dead parent (``chem.external`` explains it).

**POSIX** has no job objects. Programs run in their own session (``start_new_session``), so a cancel or a time limit
kills the whole process group, and a supervised lane's programs are recorded for its parent to kill
(``external.set_program_registry``). Since ENG-3b, ``run_program`` also starts every native program under the lifeline
guard (``chem.external.GUARD``): its stdin is a pipe this process holds open, and when this process dies, even of
SIGKILL, the kernel closes it and the guard's watchdog kills the program's process group.

**Probes** (``run_probe``): ``subprocess.run(timeout=...)`` waits for the program's pipes to close after it kills a
program that timed out, and on Windows a launcher's child that inherited them keeps them open, so the probe can block
far past its timeout. ``run_probe`` writes the output to temporary files instead (nothing to wait for), starts the
program like ``start`` (a job on Windows, its own session on POSIX) and kills the whole tree when it ends or times out.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_BASIC_ACCOUNTING = 1               # JobObjectBasicAccountingInformation
_JOB_PROCESS_ID_LIST = 3                # JobObjectBasicProcessIdList
_JOB_EXTENDED_LIMITS = 9                # JobObjectExtendedLimitInformation
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
#: the attribute ``start`` sets on the Popen object: its ``Job``, or None (POSIX, or no job could be made)
JOB_ATTR = "qb_job"


# ------------------------------------------------------------------------------------------- Windows API
_API: dict = {}


def _api():
    """kernel32/ntdll functions and the job structures (ctypes; Windows only)."""
    if _API:
        return _API
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                              ctypes.POINTER(wintypes.DWORD)]
    k32.QueryInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.TerminateJobObject.restype = wintypes.BOOL
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long

    class BasicLimits(ctypes.Structure):          # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class ExtendedLimits(ctypes.Structure):       # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class Accounting(ctypes.Structure):           # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

    _API.update(ctypes=ctypes, k32=k32, ntdll=ntdll, ExtendedLimits=ExtendedLimits, Accounting=Accounting)
    return _API


class Job:
    """A Windows job object whose processes end when its last handle closes (``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``).
    The handle is not inheritable, so only this process holds it."""

    def __init__(self, handle: int):
        self.handle = handle

    @classmethod
    def create(cls) -> Job | None:
        """A new kill-on-close job, or None (not Windows, or the system refused)."""
        if not IS_WINDOWS:
            return None
        try:
            api = _api()
            k32 = api["k32"]
            handle = k32.CreateJobObjectW(None, None)
            if not handle:
                return None
            info = api["ExtendedLimits"]()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ctypes = api["ctypes"]
            if not k32.SetInformationJobObject(handle, _JOB_EXTENDED_LIMITS, ctypes.byref(info), ctypes.sizeof(info)):
                k32.CloseHandle(handle)
                return None
        except Exception:  # noqa: BLE001 - no job: the program still runs, as before ENG-3
            return None
        return cls(handle)

    def assign(self, process_handle: int) -> bool:
        """Put a process (its handle, e.g. ``Popen._handle``) in the job."""
        if not self.handle:
            return False
        try:
            return bool(_api()["k32"].AssignProcessToJobObject(self.handle, int(process_handle)))
        except Exception:  # noqa: BLE001
            return False

    def adopt(self, pid: int) -> bool:
        """Put a running process (by pid) in the job: a ``multiprocessing`` lane, which cannot be started
        suspended. Whatever it starts afterwards is in the job too."""
        if not self.handle:
            return False
        try:
            k32 = _api()["k32"]
            h = k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, int(pid))
            if not h:
                return False
            try:
                return bool(k32.AssignProcessToJobObject(self.handle, h))
            finally:
                k32.CloseHandle(h)
        except Exception:  # noqa: BLE001
            return False

    def terminate(self, exit_code: int = 1) -> bool:
        """End every process in the job now (nested jobs included)."""
        if not self.handle:
            return False
        try:
            return bool(_api()["k32"].TerminateJobObject(self.handle, int(exit_code)))
        except Exception:  # noqa: BLE001
            return False

    def pids(self) -> list[int]:
        """Processes in the job now (for tests and diagnostics)."""
        if not self.handle:
            return []
        try:
            api = _api()
            ctypes = api["ctypes"]

            class IdList(ctypes.Structure):       # JOBOBJECT_BASIC_PROCESS_ID_LIST with room for 1024
                _fields_ = [("NumberOfAssignedProcesses", ctypes.c_uint32), ("NumberOfProcessIdsInList", ctypes.c_uint32),
                            ("ProcessIdList", ctypes.c_size_t * 1024)]

            ids = IdList()
            if not api["k32"].QueryInformationJobObject(self.handle, _JOB_PROCESS_ID_LIST, ctypes.byref(ids),
                                                        ctypes.sizeof(ids), None):
                return []
            return [int(ids.ProcessIdList[i]) for i in range(ids.NumberOfProcessIdsInList)]
        except Exception:  # noqa: BLE001
            return []

    def usage(self) -> tuple[float | None, int, float | None]:
        """(CPU s of every process the job ever held, how many it held, the largest one's peak private memory MB)."""
        cpu, n, peak = None, 0, None
        if not self.handle:
            return cpu, n, peak
        try:
            api = _api()
            ctypes, k32 = api["ctypes"], api["k32"]
            acc = api["Accounting"]()
            if k32.QueryInformationJobObject(self.handle, _JOB_BASIC_ACCOUNTING, ctypes.byref(acc),
                                             ctypes.sizeof(acc), None):
                cpu = (acc.TotalUserTime + acc.TotalKernelTime) * 1e-7          # 100 ns units
                n = int(acc.TotalProcesses)
            ext = api["ExtendedLimits"]()
            if k32.QueryInformationJobObject(self.handle, _JOB_EXTENDED_LIMITS, ctypes.byref(ext),
                                             ctypes.sizeof(ext), None):
                peak = ext.PeakProcessMemoryUsed / 2**20 or None
        except Exception:  # noqa: BLE001 - what could not be read stays None
            return cpu, n, peak
        return cpu, n, peak

    def close(self) -> None:
        """Close the handle: Windows ends whatever is still in the job."""
        handle, self.handle = self.handle, 0
        if handle:
            with contextlib.suppress(Exception):
                _api()["k32"].CloseHandle(handle)

    def __bool__(self) -> bool:
        return bool(self.handle)


def _resume(proc: subprocess.Popen) -> bool:
    try:
        return _api()["ntdll"].NtResumeProcess(int(proc._handle)) >= 0     # noqa: SLF001 - Popen's process handle
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------- programs
def start(cmd: Sequence[str], **popen_kwargs) -> subprocess.Popen:
    """``subprocess.Popen(cmd, **popen_kwargs)`` that cannot outlive this process on Windows: created suspended, put
    in a new kill-on-close ``Job`` (``job_of``), then resumed. When no job can be made (or the process cannot be put
    in one), it runs as before, without one. Elsewhere a plain Popen (pass ``start_new_session=True`` for a process
    group of its own). Raises what Popen raises."""
    if not IS_WINDOWS:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        setattr(proc, JOB_ATTR, None)
        return proc
    job = Job.create()
    if job is None:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        setattr(proc, JOB_ATTR, None)
        return proc
    flags = int(popen_kwargs.pop("creationflags", 0) or 0)
    try:
        proc = subprocess.Popen(cmd, creationflags=flags | CREATE_SUSPENDED, **popen_kwargs)
    except BaseException:
        job.close()
        raise
    try:
        in_job = job.assign(proc._handle)              # noqa: SLF001 - Popen's process handle
    finally:
        resumed = _resume(proc)                        # always: a suspended process would never end
    if not resumed:
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(5)
        job.close()
        raise OSError(f"could not resume {cmd[0] if cmd else 'the program'} after starting it")
    if not in_job:
        job.close()
        job = None
    setattr(proc, JOB_ATTR, job)
    return proc


def job_of(proc) -> Job | None:
    """The kill-on-close job ``start`` put ``proc`` in (None: POSIX, or none could be made)."""
    job = getattr(proc, JOB_ATTR, None)
    return job if job else None


def release(proc) -> None:
    """Close ``proc``'s job once it has ended: Windows ends whatever it left running in the job."""
    job = job_of(proc)
    if job is not None:
        job.close()
    with contextlib.suppress(AttributeError):
        setattr(proc, JOB_ATTR, None)


def kill_tree(proc: subprocess.Popen, wait_s: float = 10.0) -> None:
    """End ``proc`` and everything it started, then reap it. Windows: its job (``TerminateJobObject``), else
    ``taskkill /T /F``, else the process alone. POSIX: its process group when it leads one (SIGTERM, then SIGKILL
    after 5 s), else the process. Never raises."""
    if IS_WINDOWS:
        job = job_of(proc)
        done = job is not None and job.terminate()
        if not done and proc.poll() is None:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                cp = subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                                    creationflags=CREATE_NO_WINDOW, timeout=20)
                done = cp.returncode == 0
            if not done:
                with contextlib.suppress(OSError):
                    proc.kill()
    else:
        group = False
        with contextlib.suppress(OSError):
            group = os.getpgid(proc.pid) == proc.pid
        try:
            if group:
                os.killpg(proc.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(5)
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)   # members that ignored SIGTERM, or a stuck leader
            elif proc.poll() is None:
                proc.kill()
        except (OSError, subprocess.SubprocessError):
            with contextlib.suppress(OSError):
                proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(wait_s)


# ------------------------------------------------------------------------------------------- probes
@dataclass(frozen=True)
class Probe:
    """What ``run_probe`` got: the exit code (None when it timed out or could not start) and the output."""
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str = ""                  # why it could not start (OSError), "" otherwise

    @property
    def text(self) -> str:
        return self.stdout + self.stderr


def _end_probe(proc: subprocess.Popen) -> None:
    """Kill what the probe left (it is over, or out of time): the whole tree, whatever is still in it."""
    if IS_WINDOWS:
        job = job_of(proc)
        if job is not None:
            job.terminate()
        elif proc.poll() is None:
            kill_tree(proc, wait_s=5.0)
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)       # its session (start_new_session): grandchildren too
        with contextlib.suppress(OSError):
            if proc.poll() is None:
                proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(5)


def _read(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_probe(cmd: Sequence[str], timeout: float = 15.0, *, cwd: str | Path | None = None,
              env: Mapping[str, str] | None = None) -> Probe:
    """Run a short program (``xtb --version``, ``python -c "import psi4"``) and return its output within about
    ``timeout`` seconds (plus a few for the kill), whatever its children do with stdout. Never raises for the
    program: a missing executable comes back as ``Probe(None, error=...)``."""
    with tempfile.TemporaryDirectory(prefix="qb_probe_", ignore_cleanup_errors=True) as tmp:
        out_path, err_path = Path(tmp) / "stdout", Path(tmp) / "stderr"
        timed_out = False
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": err,
                            "cwd": str(cwd) if cwd else tempfile.gettempdir(),
                            "env": dict(env) if env is not None else None}
            if IS_WINDOWS:
                kwargs["creationflags"] = CREATE_NO_WINDOW
            else:
                kwargs["start_new_session"] = True
            try:
                proc = start([str(c) for c in cmd], **kwargs)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return Probe(None, error=f"{type(exc).__name__}: {exc}")
            try:
                proc.wait(max(0.0, float(timeout)))
            except subprocess.TimeoutExpired:
                timed_out = True
            except BaseException:
                _end_probe(proc)
                release(proc)
                raise
            _end_probe(proc)
            release(proc)
        return Probe(None if timed_out else proc.returncode, _read(out_path), _read(err_path), timed_out)
