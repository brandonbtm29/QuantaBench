"""Run priority and free threads: runs that keep the PC responsive (PRIO-1).

A 5-lane x 4-thread CREST run on BTM NOVA (Ryzen 9 7900X, 24 threads) at below-normal CPU priority still made
the desktop hard to use, even deleting files: CPU priority alone does not stop a run's disk I/O (Psi4's scratch
files) from crowding out the user's. Two settings per machine (``Settings.priority_for``,
``Settings.free_threads_for``; docs/ENGINES.md "Run priority and free threads"):

``priority``      ``"normal"`` | ``"low"`` (the default) | ``"background"``: CPU and I/O priority of the work. It is
                  applied where the work runs and never to the GUI process: the lane processes (``dist.lanes``), a
                  CLI ``build``/``grid build`` or a grid ``worker`` process, the GUI's build thread for in-process
                  work (``thread_lowered``), and every program ``external.run_program`` starts (xtb, CREST, MOPAC,
                  Psi4, also through WSL). Programs those programs start (CREST's xtb, Psi4's threads) inherit it.
``free_threads``  logical threads the automatic thread budgets leave free for the user: with CPU threads 0 (auto,
                  the default) the thread budget, the ``auto`` lane plan in it and the threads dynamic CREST threads
                  may borrow (``dist.threads``) never use them. Default 1, today's "all threads but one".
                  ``background`` leaves at least 25 % of the logical threads (at least 2) free. An explicit CPU
                  thread count, or an explicit lane spec, is used as given.

Mechanism (``apply_to_process``, ``thread_lowered``, ``child_options``):

    level        Windows                               Linux (and inside WSL)          macOS
    low          BELOW_NORMAL_PRIORITY_CLASS, I/O low  nice 10, I/O best-effort 7      nice 10, disk I/O utility
    background   IDLE_PRIORITY_CLASS, I/O very low     nice 19, I/O idle class         nice 19, disk I/O throttle

A priority is only ever lowered, never raised: a lane of a run at ``low`` inside a worker at ``background`` stays at
``background`` (POSIX cannot raise a nice value without privileges anyway). macOS's own background policy
(``taskpolicy -b``) is not used: on Apple Silicon it keeps a process on the efficiency cores, which would make a
CREST run several times slower even on an idle Mac.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

PRIORITIES = ("normal", "low", "background")
DEFAULT_PRIORITY = "low"
#: today's automatic thread budget: every logical thread but one (``resources.resolve_threads(0)``)
DEFAULT_FREE_THREADS = 1
#: ``background`` leaves at least this share of the logical threads free, and at least ``BACKGROUND_FREE_MIN``
BACKGROUND_FREE_FRACTION = 0.25
BACKGROUND_FREE_MIN = 2
#: one line per level for the GUI and ``doctor``
DESCRIPTIONS = {
    "normal": "normal priority: as fast as the machine allows, the desktop competes with the run",
    "low": "below-normal CPU and low I/O priority: the run gives way to what you do",
    "background": "idle CPU and very low I/O priority, more threads left free: the run uses what is idle",
}
_RANK = {"normal": 0, "low": 1, "background": 2}

#: POSIX nice values
NICE = {"normal": 0, "low": 10, "background": 19}
#: Windows priority classes (WinBase.h); psutil has them only on Windows
NORMAL_PRIORITY_CLASS = 0x00000020
IDLE_PRIORITY_CLASS = 0x00000040
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
WINDOWS_CLASS = {"normal": NORMAL_PRIORITY_CLASS, "low": BELOW_NORMAL_PRIORITY_CLASS,
                 "background": IDLE_PRIORITY_CLASS}
#: Windows I/O priority hints (psutil.IOPRIO_VERYLOW = 0, IOPRIO_LOW = 1, IOPRIO_NORMAL = 2)
WINDOWS_IO = {"normal": 2, "low": 1, "background": 0}
#: Windows thread priorities of the GUI's build thread (THREAD_PRIORITY_NORMAL, _LOWEST, _IDLE)
WINDOWS_THREAD = {"normal": 0, "low": -2, "background": -15}
THREAD_MODE_BACKGROUND_BEGIN = 0x00010000
THREAD_MODE_BACKGROUND_END = 0x00020000
#: Linux I/O priority (ioprio class, level): best effort level 7 (the lowest), the idle class
LINUX_IO = {"low": (2, 7), "background": (3, 0)}
IOPRIO_CLASS_IDLE = 3
#: macOS disk I/O policies (sys/resource.h IOPOL_*), as setiopolicy_np takes them and taskpolicy -d names them
MAC_IOPOL = {"low": 4, "background": 3}            # IOPOL_UTILITY, IOPOL_THROTTLE
MAC_TASKPOLICY = {"low": "utility", "background": "throttle"}
IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, IOPOL_SCOPE_THREAD = 0, 0, 1
QOS_CLASS_UTILITY = 0x11                           # sys/qos.h: the GUI's build thread on macOS
#: the end of every ``sh -c`` wrapper ``external.run_program`` uses (``external.STACK_WRAPPER``, ``WSL_WRAPPER``)
EXEC_TAIL = 'exec "$0" "$@"'


# ------------------------------------------------------------------------------------- the settings
def check(level) -> str:
    """``level`` as one of ``PRIORITIES`` ('Low' -> 'low'); ValueError for anything else."""
    text = str(level or "").strip().lower()
    if text not in PRIORITIES:
        raise ValueError(f"run priority must be one of {', '.join(PRIORITIES)}, not {level!r}")
    return text


def normalize(level, default: str = DEFAULT_PRIORITY) -> str:
    """``level`` as one of ``PRIORITIES``, ``default`` for anything else (a hand-edited settings file)."""
    try:
        return check(level)
    except ValueError:
        return default


def lower_of(a: str, b: str) -> str:
    """The lower of two levels ('background' is the lowest)."""
    return a if _RANK.get(a, 0) >= _RANK.get(b, 0) else b


def _logical(logical: int | None) -> int:
    if logical:
        return max(1, int(logical))
    from . import resources

    return max(1, int(resources.logical_cores()))


def effective_free_threads(free_threads, level: str = DEFAULT_PRIORITY, logical: int | None = None) -> int:
    """Logical threads left free: ``free_threads`` (negative = the default, 1), at least 25 % of the logical
    threads and at least 2 for ``background``; never all of them (one thread is always left to work with)."""
    n = _logical(logical)
    try:
        free = int(free_threads)
    except (TypeError, ValueError):
        free = DEFAULT_FREE_THREADS
    if free < 0:
        free = DEFAULT_FREE_THREADS
    if normalize(level) == "background":
        free = max(free, BACKGROUND_FREE_MIN, math.ceil(BACKGROUND_FREE_FRACTION * n))
    return max(0, min(free, n - 1))


def auto_thread_cap(free_threads, level: str = DEFAULT_PRIORITY, logical: int | None = None) -> int:
    """The most threads an automatic budget may use: the logical threads minus the free ones (at least 1)."""
    n = _logical(logical)
    return max(1, n - effective_free_threads(free_threads, level, n))


def thread_budget(threads: int, free_threads, level: str = DEFAULT_PRIORITY, logical: int | None = None) -> int:
    """The CPU-thread budget of a run: ``threads`` 0 (auto) = the logical threads minus the free ones; an explicit
    count is kept (at most the logical threads), as ``resources.resolve_threads`` always did."""
    n = _logical(logical)
    try:
        threads = int(threads or 0)
    except (TypeError, ValueError):
        threads = 0
    if threads <= 0:
        return auto_thread_cap(free_threads, level, n)
    return max(1, min(threads, n))


def of(cfg) -> tuple[str, int]:
    """(priority, free threads) of a ``BuildConfig`` (or anything with those attributes; defaults otherwise)."""
    level = normalize(getattr(cfg, "priority", DEFAULT_PRIORITY))
    try:
        free = int(getattr(cfg, "free_threads", DEFAULT_FREE_THREADS))
    except (TypeError, ValueError):
        free = DEFAULT_FREE_THREADS
    return level, free


def budget_of(cfg, logical: int | None = None) -> int:
    """``thread_budget`` of a ``BuildConfig``: what CPU threads 0 means for this run."""
    level, free = of(cfg)
    return thread_budget(getattr(cfg, "threads", 0), free, level, logical)


def cap_of(cfg, logical: int | None = None) -> int | None:
    """The most threads automatic choices may use for a ``BuildConfig``, borrowed CREST threads included
    (``auto_thread_cap``), when its CPU threads are 0 (auto); None for an explicit thread count, which is used as
    given (as the lane spec it plans)."""
    if int(getattr(cfg, "threads", 0) or 0) > 0:
        return None
    level, free = of(cfg)
    return auto_thread_cap(free, level, logical)


def describe(level: str, free_threads, logical: int | None = None) -> str:
    """'low (below-normal CPU and low I/O priority ...); 1 of 24 threads left free (auto plans use up to 23)'."""
    level = normalize(level)
    n = _logical(logical)
    free = effective_free_threads(free_threads, level, n)
    extra = ""
    try:
        asked = int(free_threads)
    except (TypeError, ValueError):
        asked = DEFAULT_FREE_THREADS
    if free != max(0, asked):
        extra = f", {asked} set; background leaves at least {BACKGROUND_FREE_MIN} and 25 %"
    return (f"{level} ({DESCRIPTIONS[level]}); {free} of {n} threads left free{extra} "
            f"(automatic budgets use up to {n - free})")


# ------------------------------------------------------------------------------------ the state here
_process_level: str | None = None       # set by apply_to_process (a lane, a CLI build, a worker)
_local = threading.local()               # .level: set by children_at / thread_lowered (a build thread)


def in_force() -> str:
    """The level the programs this thread starts should run at: the thread's (``children_at``), else the
    process's (``apply_to_process``), else 'normal'."""
    return getattr(_local, "level", None) or _process_level or "normal"


@contextlib.contextmanager
def children_at(level):
    """Programs ``external.run_program`` starts in this thread inside the block run at ``level`` (or lower, when
    this thread already runs lower). Does not change the thread itself."""
    old = getattr(_local, "level", None)
    _local.level = normalize(level)
    try:
        yield
    finally:
        _local.level = old


def current_nice() -> int:
    """POSIX: the nice value of the calling thread (Linux keeps one per thread; macOS one per process)."""
    try:
        return int(os.getpriority(os.PRIO_PROCESS, 0))
    except (AttributeError, OSError):
        return 0


def current_level(system: str | None = None) -> str:
    """The level this thread already runs at: Windows from the process's priority class, POSIX from the nice
    value (10 or more: low, 19: background)."""
    system = system or sys.platform
    if system == "win32":
        try:
            import psutil

            cls = int(psutil.Process().nice())
        except Exception:  # noqa: BLE001
            return "normal"
        return {IDLE_PRIORITY_CLASS: "background", BELOW_NORMAL_PRIORITY_CLASS: "low"}.get(cls, "normal")
    nice = current_nice()
    return "background" if nice >= NICE["background"] else "low" if nice >= NICE["low"] else "normal"


# ------------------------------------------------------------------------------ lowering a process
def _libc():
    import ctypes
    import ctypes.util

    return ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)


def _linux_threads() -> list[int]:
    try:
        return [int(t) for t in os.listdir("/proc/self/task")]
    except (OSError, ValueError):
        return [threading.get_native_id()]


def _linux_io_lower(pid: int, level: str) -> bool:
    """Set the I/O priority of task ``pid`` (a thread on Linux) to ``level``'s, unless it is lower already."""
    import psutil

    want_class, want_value = LINUX_IO[level]
    p = psutil.Process(pid)
    cur = p.ionice()
    cur_class, cur_value = int(cur[0]), int(cur[1])
    if cur_class == IOPRIO_CLASS_IDLE or (cur_class == want_class and cur_value >= want_value):
        return False
    if want_class == IOPRIO_CLASS_IDLE:
        p.ionice(psutil.IOPRIO_CLASS_IDLE)
    else:
        p.ionice(psutil.IOPRIO_CLASS_BE, value=want_value)
    return True


def apply_to_process(level) -> str:
    """Lower this whole process (every thread) to ``level``, never raising it, and make it the level of the programs
    it starts (``in_force``). For processes that only do work: a lane, a CLI build, a grid worker; never the GUI
    (use ``thread_lowered``). Returns a line for the log; never raises."""
    global _process_level
    want = normalize(level)
    level = lower_of(want, current_level())
    _process_level = level
    if level == "normal":
        return "normal"
    done: list[str] = []
    try:
        if sys.platform == "win32":
            import psutil

            p = psutil.Process()
            if current_level("win32") != level:
                p.nice(WINDOWS_CLASS[level])
            done.append("below-normal class" if level == "low" else "idle class")
            with contextlib.suppress(Exception):
                if int(p.ionice()) > WINDOWS_IO[level]:
                    p.ionice(WINDOWS_IO[level])
                done.append("I/O " + ("low" if level == "low" else "very low"))
        else:
            target = NICE[level]
            if sys.platform.startswith("linux"):
                for tid in _linux_threads():                 # nice is per thread on Linux
                    with contextlib.suppress(OSError):
                        if os.getpriority(os.PRIO_PROCESS, tid) < target:
                            os.setpriority(os.PRIO_PROCESS, tid, target)
                done.append(f"nice {target}")
                with contextlib.suppress(Exception):
                    for tid in _linux_threads():
                        _linux_io_lower(tid, level)
                    done.append("I/O " + ("best-effort 7" if level == "low" else "idle class"))
            else:
                with contextlib.suppress(OSError):
                    if current_nice() < target:
                        os.setpriority(os.PRIO_PROCESS, 0, target)
                done.append(f"nice {target}")
                if sys.platform == "darwin":
                    with contextlib.suppress(Exception):
                        _libc().setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, MAC_IOPOL[level])
                        done.append(f"disk I/O {MAC_TASKPOLICY[level]}")
    except Exception as exc:  # noqa: BLE001 - a priority is best effort; the run goes on
        done.append(f"not fully applied: {type(exc).__name__}: {exc}")
    note = "" if want == level else f" (asked {want}; this process already ran at {level})"
    return f"{level} ({', '.join(done)}){note}"


def _pid_level(p, system: str) -> str:
    """The level process ``p`` (psutil) runs at now."""
    if system == "win32":
        cls = int(p.nice())
        return {IDLE_PRIORITY_CLASS: "background", BELOW_NORMAL_PRIORITY_CLASS: "low"}.get(cls, "normal")
    nice = int(p.nice())
    return "background" if nice >= NICE["background"] else "low" if nice >= NICE["low"] else "normal"


def apply_to_pids(pids, level, *, children: bool = True, system: str | None = None) -> str:
    """Set running processes (``pids``, and with ``children`` the programs they started: CREST, xtb, Psi4) to
    ``level`` now, for a run priority changed while a batch runs (DIST-7: the Machines dock, ``machines set``).

    Lowering works everywhere. Raising works on Windows (a process may raise its own children's class up to its own),
    not on POSIX without privileges (a nice value only goes up): there a raised level is reported and takes effect
    for the next lanes the batch starts. Returns a line for the log; never raises."""
    system = system or sys.platform
    want = normalize(level)
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a dependency
        return "not applied (no psutil)"
    procs: dict[int, object] = {}
    for pid in pids or ():
        try:
            p = psutil.Process(int(pid))
        except (psutil.Error, ValueError, TypeError):
            continue
        procs[p.pid] = p
        if children:
            with contextlib.suppress(psutil.Error):
                for c in p.children(recursive=True):
                    procs[c.pid] = c
    changed = kept = failed = 0
    for p in procs.values():
        try:
            now = _pid_level(p, system)
            if now == want:
                kept += 1
                continue
            if _RANK[want] < _RANK[now] and system != "win32":
                failed += 1                     # POSIX: a nice value cannot be lowered (priority raised) again
                continue
            if system == "win32":
                p.nice(WINDOWS_CLASS[want])
                with contextlib.suppress(Exception):
                    p.ionice(WINDOWS_IO[want])
            else:
                p.nice(NICE[want])
                if system.startswith("linux") and want != "normal":
                    with contextlib.suppress(Exception):
                        _linux_io_lower(p.pid, want)
            changed += 1
        except (psutil.Error, OSError, ValueError):
            failed += 1
    if not procs:
        return f"{want}: no running lane to change"
    out = f"{want}: {changed} process(es) changed, {kept} already there"
    if failed:
        out += (f", {failed} left as they are (a priority cannot be raised on this OS without privileges; the next "
                "lanes start at the new level)" if system != "win32" else f", {failed} could not be changed")
    return out


# ------------------------------------------------------------------------- lowering the GUI's thread
def _lower_thread(level: str):
    """Lower the calling thread; returns a function that restores it where the OS allows."""
    if level == "normal":
        return lambda: None
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentThread.restype = ctypes.c_void_p
        k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        k32.GetThreadPriority.argtypes = [ctypes.c_void_p]
        h = k32.GetCurrentThread()
        before = int(k32.GetThreadPriority(h))
        background = level == "background" and bool(k32.SetThreadPriority(h, THREAD_MODE_BACKGROUND_BEGIN))
        if before > WINDOWS_THREAD[level]:
            k32.SetThreadPriority(h, WINDOWS_THREAD[level])

        def restore() -> None:
            if background:
                k32.SetThreadPriority(h, THREAD_MODE_BACKGROUND_END)
            k32.SetThreadPriority(h, before)
        return restore
    if sys.platform.startswith("linux"):
        tid = threading.get_native_id()
        with contextlib.suppress(OSError):
            if os.getpriority(os.PRIO_PROCESS, tid) < NICE[level]:
                os.setpriority(os.PRIO_PROCESS, tid, NICE[level])
        with contextlib.suppress(Exception):
            _linux_io_lower(tid, level)
        return lambda: None          # an unprivileged thread cannot raise its nice value again; it ends with the run
    if sys.platform == "darwin":
        import ctypes

        libc = _libc()
        before_io: list[int] = []
        with contextlib.suppress(Exception):
            before_io.append(int(libc.getiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD)))
            libc.setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, MAC_IOPOL[level])
        qos, relative = ctypes.c_uint(0), ctypes.c_int(0)
        with contextlib.suppress(Exception):
            libc.pthread_self.restype = ctypes.c_void_p
            libc.pthread_get_qos_class_np(ctypes.c_void_p(libc.pthread_self()), ctypes.byref(qos),
                                          ctypes.byref(relative))
            libc.pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0)

        def restore() -> None:
            with contextlib.suppress(Exception):
                if qos.value:
                    libc.pthread_set_qos_class_self_np(qos.value, relative.value)
            with contextlib.suppress(Exception):
                if before_io and before_io[0] >= 0:
                    libc.setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, before_io[0])
        return restore
    return lambda: None


@contextlib.contextmanager
def thread_lowered(level):
    """Run the block on a lowered calling thread, for in-process work in the GUI (``gui.workers.BuildWorker``), and
    start its programs at ``level`` (``children_at``). The GUI process itself keeps its priority. Limits: Windows
    lowers only this thread (thread priority LOWEST or IDLE, plus background I/O mode for ``background``); threads
    an in-process engine started before (PyTorch's and OpenMP's pools) keep theirs. Linux lowers this thread's nice
    and I/O priority, and threads it starts inherit them; macOS gives it the utility QoS class and disk I/O
    policy. Lanes lower whole processes, so a long in-process run is best given lanes."""
    level = normalize(level)
    try:
        restore = _lower_thread(level)
    except Exception:  # noqa: BLE001 - best effort
        def restore() -> None:
            return None
    try:
        with children_at(level):
            yield level
    finally:
        with contextlib.suppress(Exception):
            restore()


# ------------------------------------------------------------------------------ external programs
@dataclass(frozen=True)
class ChildPriority:
    """How ``external.run_program`` starts one program at ``level`` (``child_options``)."""
    level: str = "normal"
    creationflags: int = 0                      # Windows: the priority class flag (0 = inherit)
    windows_io: int | None = None               # Windows: psutil I/O priority set on the program once it started
    nice_increment: int = 0                     # POSIX/WSL: ``nice -n`` in the wrapper (0 = none)
    linux_io: tuple[int, int] | None = None     # Linux/WSL: ionice class and level in the wrapper
    mac_io: str = ""                            # macOS: ``taskpolicy -d`` policy in the wrapper

    def wrap(self, base: str) -> str:
        """``base``, a ``sh -c`` script ending in ``exec "$0" "$@"`` (``external.STACK_WRAPPER`` or
        ``WSL_WRAPPER``), with this priority applied before the exec. A missing ``ionice``/``nice`` is skipped,
        never fatal; the program keeps the shell's PID (exec), so cancelling and the ``.qb_pid`` record work."""
        if not base.endswith(EXEC_TAIL) or not (self.nice_increment > 0 or self.linux_io or self.mac_io):
            return base
        out = base[:-len(EXEC_TAIL)]
        if self.linux_io:
            cls, value = self.linux_io
            io = f"-c {cls}" + (f" -n {value}" if cls == 2 else "")
            out += f"command -v ionice >/dev/null 2>&1 && ionice {io} -p $$ >/dev/null 2>&1; "
        launcher = f"taskpolicy -d {self.mac_io} " if self.mac_io else ""
        if self.nice_increment > 0:
            out += f'command -v nice >/dev/null 2>&1 && exec {launcher}nice -n {self.nice_increment} "$0" "$@"; '
        return out + f'exec {launcher}"$0" "$@"'


def child_options(level, *, system: str | None = None, via_wsl: bool = False, current: str = "normal",
                  nice_now: int = 0, taskpolicy: bool = False) -> ChildPriority:
    """How to start a program at ``level`` (a pure function of its arguments, for tests; ``options_here`` fills
    them in for this thread). ``current``: the level this thread already runs at (never raised); ``nice_now``: its
    nice value (POSIX); ``taskpolicy``: macOS's ``taskpolicy -d`` works here.

    Windows: the priority class as a creation flag (CREST's xtb and anything else the program starts inherits
    BELOW_NORMAL and IDLE) and the I/O priority set once it started. A ``wsl:`` program: the same for wsl.exe, and
    inside WSL ``nice``/``ionice`` in the wrapper, since Windows priorities do not reach the Linux VM. Linux:
    ``ionice`` and ``nice`` in the wrapper (inherited by the program's children). macOS: ``nice`` and
    ``taskpolicy -d``."""
    system = system or sys.platform
    level = lower_of(normalize(level, "normal"), normalize(current, "normal"))
    if level == "normal":
        return ChildPriority()
    if system == "win32":
        return ChildPriority(level, creationflags=WINDOWS_CLASS[level], windows_io=WINDOWS_IO[level],
                             nice_increment=NICE[level] if via_wsl else 0,
                             linux_io=LINUX_IO[level] if via_wsl else None)
    increment = max(0, NICE[level] - int(nice_now))
    if system.startswith("linux"):
        return ChildPriority(level, nice_increment=increment, linux_io=LINUX_IO[level])
    return ChildPriority(level, nice_increment=increment,
                         mac_io=MAC_TASKPOLICY[level] if system == "darwin" and taskpolicy else "")


_TASKPOLICY: list[bool] = []


def taskpolicy_works() -> bool:
    """macOS: whether ``taskpolicy -d utility`` runs a program here (checked once per process)."""
    if not _TASKPOLICY:
        exe = shutil.which("taskpolicy") or ("/usr/sbin/taskpolicy" if os.path.exists("/usr/sbin/taskpolicy") else "")
        ok = False
        if exe:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                ok = subprocess.run([exe, "-d", "utility", "/usr/bin/true"], capture_output=True,  # noqa: S603
                                    timeout=10, check=False).returncode == 0
        _TASKPOLICY.append(ok)
    return _TASKPOLICY[0]


def options_here(via_wsl: bool = False) -> ChildPriority:
    """``child_options`` for a program this thread starts now (``in_force``, this thread's level and nice)."""
    level = in_force()
    if level == "normal":
        return ChildPriority()
    win = sys.platform == "win32"
    return child_options(level, via_wsl=via_wsl, current=current_level(), nice_now=0 if win else current_nice(),
                         taskpolicy=sys.platform == "darwin" and taskpolicy_works())


def after_start(pid: int, opts: ChildPriority) -> None:
    """Windows: the program's I/O priority (no creation flag sets it). Never raises."""
    if opts.windows_io is None:
        return
    with contextlib.suppress(Exception):
        import psutil

        p = psutil.Process(pid)
        if int(p.ionice()) > opts.windows_io:
            p.ionice(opts.windows_io)
