"""The job queue (UI-5): every build of the window is a job, run one at a time in the order submitted.

A job is what ``MainWindow.run_build`` (Run all / Run selected / F5), ``resume_run`` and the job dialog's Submit hand
over: the molecules, the configuration and the run folder. The queue starts the first job at once when nothing runs,
and the next one when the window's build ends (``MainWindow._on_finished`` / ``_on_failed`` call ``ended``). The
window still runs a job as before (``MainWindow._start_worker``: one ``BuildWorker``), so a job is resumable and its
run folder is like any other.

Per job the queue keeps its status, progress, molecule counts, its own log and its times; the job monitor dock
(``gui.job_monitor``) shows them. A queued job can be cancelled (it never starts) or moved; cancelling the running job
is the window's Cancel (the build stops after the current step and the next job starts).

**Jobs on a worker** (``Job.run_on``: a worker's name or "auto"; DIST-3) do not wait in this queue: ``make_follower``
(``dist.remote_job.RemoteJob``, set by ``gui.job_monitor.install``) sends each one at once and follows it on a thread
of its own. Its updates reach the GUI thread through ``_relay``; when it ends, the rows of its run folder (read on that
thread) go out in ``remote_rows`` for the Molecules list. "auto" that picks this computer turns the job into a local
one, queued here like any other.

**MD jobs** (``Job.kind == "md"``, SIM-1) do not run in the window: the queue starts the run supervisor
(``md.supervisor.start_detached``: ``quantabench-cli md run|resume DIR`` as a process of its own) and follows the run
folder's ``run.json`` and ``progress.json`` every few seconds (``md_poll``). They run beside the window's builds, one
MD job at a time on this computer (one per GPU, the design's per-resource rule until SCHED-3's ledger), and their
progress is in ns, with ns/day and the ETA. Cancel on a running MD job pauses it at its next pause point (a STOP file;
the run keeps a checkpoint); ``resume_md`` queues the folder again to continue. Closing the window leaves an MD run
going; ``attach_md`` follows one that is already running.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

#: job states; "cancelling": the running job was told to stop and has not ended yet (an MD job: pausing);
#: "paused": an MD job that ended at a pause point, resumable
STATES = ("queued", "running", "cancelling", "done", "failed", "cancelled", "paused")
FINISHED = ("done", "failed", "cancelled", "paused")
LOG_LINES = 2000            # per job (the window's Log keeps its own 5,000)
MD_POLL_MS = 2000           # how often an MD job's run folder is read


@dataclass
class Job:
    name: str
    ids: list[str]
    cfg: object                                   # BuildConfig
    todo: object = None                           # list of MoleculeInput, a callable returning them, or None (resume)
    run_dir: object = None                        # Path, a callable making it when the job starts, or None
    total: int = 0                                # molecules (a resume: the run folder's total)
    done_before: int = 0                          # a resume: already finished in the run folder
    summary: str = ""                             # level · phase · task
    recipe: str = ""
    task: str = ""
    run_on: str = "local"
    number: int = 0
    status: str = "queued"
    done: int = 0                                 # molecules ended in this job (ok, failed or cancelled)
    ok: int = 0
    failed: int = 0
    error: str = ""
    submitted: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    log: deque = field(default_factory=lambda: deque(maxlen=LOG_LINES))
    current: str = ""                             # molecule and stage running now
    machine: str = ""                             # a job on a worker: the worker ("auto" until it is chosen)
    remote_id: str = ""                           # the worker's job id
    unreachable: bool = False                     # the worker did not answer the last poll
    follower: object = None                       # dist.remote_job.RemoteJob of a job on a worker
    kind: str = "build"                           # "build" | "md" (SIM-1: a run folder run by its supervisor)
    md: dict = field(default_factory=dict)        # an MD job: its latest status (md.supervisor.read_status)
    md_resume: bool = False                       # an MD job: continue the run (md resume) rather than start it
    md_variant: str | None = None
    md_threads: int | None = None
    md_proc: object = None                        # the supervisor's Popen, when this window started it
    md_segments_before: int = 0                   # an MD job: the run's segments when it started

    @property
    def is_remote(self) -> bool:
        return self.run_on not in ("local", "")

    @property
    def is_md(self) -> bool:
        return self.kind == "md"

    @property
    def active(self) -> bool:
        return self.status in ("running", "cancelling")

    @property
    def output_dir(self) -> Path | None:
        if isinstance(self.run_dir, (str, Path)) and self.run_dir:
            return Path(self.run_dir)
        return None

    def elapsed(self, now: float | None = None) -> float:
        if not self.started:
            return 0.0
        return (self.finished or (now if now is not None else time.time())) - self.started

    def progress_text(self) -> str:
        if self.is_md:
            return md_progress_text(self.md, self.status)
        n = self.total or len(self.ids)
        done = self.done_before + self.done
        if self.status == "queued":
            return f"{n:,} molecule{'s' if n != 1 else ''}"
        text = f"{done:,} / {n:,}"
        if self.failed:
            text += f" ({self.failed:,} failed)"
        return text


def md_progress_text(st: dict, status: str = "") -> str:
    """"7.4 / 13 ns · 21 ns/day · ETA 6 h 24 min" from ``md.supervisor.read_status``."""
    total = st.get("ns_total") or 0.0
    done = st.get("ns_done") or 0.0
    if not total:
        return "MD run"
    text = f"{done:.3g} / {total:.3g} ns"
    prog = st.get("progress") or {}
    if status in ("running", "cancelling") and prog:
        if prog.get("ns_per_day"):
            text += f" · {prog['ns_per_day']:.3g} ns/day"
        if prog.get("eta_text"):
            text += f" · ETA {prog['eta_text']}"
    return text


class JobQueue(QObject):
    """Jobs in submission order (finished ones stay listed until cleared). ``start`` runs one job: the window's
    ``_start_worker`` with the job's molecules, configuration and run folder; ``can_start`` says whether the window
    is free to (no build running, not closing)."""

    added = Signal(int)          # job number
    changed = Signal(int)        # job number (status, progress, log)
    removed = Signal(int)
    remote_rows = Signal(int, object)     # job number, molecule_list.MoleculeRow list of a worker job's run folder
    remote_marked = Signal(int, str)      # job number, the molecules' stage text while on a worker ("" = clear)
    _relay = Signal(int, object)          # a follower thread's update -> the GUI thread

    def __init__(self, start: Callable[[Job], None], can_start: Callable[[], bool], parent=None,
                 make_follower: Callable | None = None):
        super().__init__(parent)
        self._start = start
        self._can_start = can_start
        self.jobs: list[Job] = []
        self._next_number = 1
        self.running: Job | None = None
        #: ``make_follower(job, on_update, on_local, local_busy)`` -> a started follower (``RemoteJob``)
        self.make_follower = make_follower
        self._relay.connect(self._apply_remote)
        #: MD jobs (SIM-1): the one running on this computer, how it starts and how its folder is read (replaceable
        #: in tests): ``md_start(job)`` -> a Popen or None (followed, not started), ``md_status(folder)`` -> dict
        self.md_running: Job | None = None
        self.md_start: Callable[[Job], object] = _start_md_supervisor
        self.md_status: Callable[[Path], dict] = _read_md_status
        self.md_timer = QTimer(self)
        self.md_timer.setInterval(MD_POLL_MS)
        self.md_timer.timeout.connect(self.md_poll)

    # ------------------------------------------------------------------ submitting
    def submit(self, job: Job) -> Job:
        job.number = self._next_number
        self._next_number += 1
        job.status = "queued"
        job.submitted = time.time()
        if not job.total:
            job.total = len(job.ids)
        self.jobs.append(job)
        if job.is_md:
            job.run_on = "local"
            self.added.emit(job.number)
            self.start_next_md()
            return job
        if job.is_remote:
            job.machine = job.run_on
            self.added.emit(job.number)
            self._start_remote(job)
            return job
        self.added.emit(job.number)
        self.start_next()
        return job

    # ------------------------------------------------------------------ jobs on a worker
    def local_load(self) -> int:
        """Local jobs running or waiting here ("auto" weighs this computer by it)."""
        return sum(1 for j in self.jobs if not j.is_remote and not j.is_md
                   and j.status in ("queued", "running", "cancelling"))

    def _start_remote(self, job: Job) -> None:
        if self.make_follower is None:
            self._end(job, "failed", "jobs on workers need the job monitor (gui.job_monitor.install)")
            return
        number = job.number

        def on_update(fields: dict) -> None:      # the follower's thread
            run_dir = fields.get("run_dir")
            if run_dir:
                from .molecule_list import read_run_folder  # plain Python: fine on this thread

                try:
                    fields["rows"] = read_run_folder(run_dir)[0]
                except (OSError, ValueError) as exc:
                    fields.setdefault("log", []).append(f"Reading {run_dir} failed: {exc}")
            self._relay.emit(number, fields)

        def on_local(why: str) -> None:           # the follower's thread: "auto" picked this computer
            self._relay.emit(number, {"go_local": why})

        job.log.append(f"Run on {job.run_on}" + (": choosing a machine" if job.run_on == "auto" else ""))
        try:
            job.follower = self.make_follower(job, on_update, on_local, self.local_load)
        except Exception as exc:  # noqa: BLE001 - reported in the job
            self._end(job, "failed", f"{type(exc).__name__}: {exc}")
            return
        self.remote_marked.emit(number, f"on {job.machine}" if job.machine != "auto" else "queued")

    def _apply_remote(self, number: int, fields: dict) -> None:
        """A follower's update, on the GUI thread."""
        job = self.job(number)
        if job is None:
            return
        if "go_local" in fields:          # "auto" chose this computer: an ordinary local job from here on
            job.run_on, job.machine, job.follower = "local", "", None
            job.log.append(f"Auto: this computer ({fields['go_local']})")
            job.status = "queued"
            self.remote_marked.emit(number, "")
            self.changed.emit(number)
            self.start_next()
            return
        for line in fields.get("log") or ():
            job.log.append(str(line))
        for key in ("machine", "remote_id", "current", "ok", "failed", "total", "unreachable"):
            if key in fields and fields[key] is not None:
                setattr(job, key, fields[key])
        if "machine" in fields and job.machine not in ("", "auto"):
            self.remote_marked.emit(number, f"on {job.machine}")
        job.done = job.ok + job.failed
        state = fields.get("state")
        if state in ("queued", "running", "cancelling") and job.status not in FINISHED:
            if state == "running" and not job.started:
                job.started = time.time()
            if not (job.status == "cancelling" and state in ("queued", "running")):
                job.status = state
        elif state in FINISHED and job.status not in FINISHED:
            if fields.get("run_dir"):
                job.run_dir = Path(fields["run_dir"])
            job.follower = None
            job.unreachable = False
            self._end(job, state, str(fields.get("error") or ""))
            if fields.get("rows"):
                self.remote_rows.emit(number, fields["rows"])
            self.remote_marked.emit(number, "")
            return
        self.changed.emit(number)

    def stop_remote(self) -> int:
        """Stop following the jobs on workers (the window closes); they go on there."""
        n = 0
        for job in self.jobs:
            if job.follower is not None and job.status not in FINISHED:
                job.follower.halt()
                n += 1
        self.md_timer.stop()                # MD runs go on in their own processes; this window stops following them
        return n

    def job(self, number: int) -> Job | None:
        return next((j for j in self.jobs if j.number == number), None)

    def queued(self) -> list[Job]:
        """Local jobs waiting here (a job on a worker waits there)."""
        return [j for j in self.jobs if j.status == "queued" and not j.is_remote]

    def start_next(self) -> Job | None:
        """Start the first queued job if nothing runs and the window can take one."""
        if self.running is not None or not self._can_start():
            return None
        job = next((j for j in self.jobs if j.status == "queued" and not j.is_remote and not j.is_md), None)
        if job is None:
            return None
        if callable(job.run_dir):
            try:
                job.run_dir = job.run_dir()
            except OSError as exc:
                self._end(job, "failed", f"could not make the run folder: {exc}")
                return self.start_next()
        job.status = "running"
        job.started = time.time()
        self.running = job
        self.changed.emit(job.number)
        try:
            self._start(job)
        except Exception as exc:  # noqa: BLE001 - a job that cannot start must not stop the queue
            self.running = None
            self._end(job, "failed", f"{type(exc).__name__}: {exc}")
            QTimer.singleShot(0, self.start_next)
        return job

    # ------------------------------------------------------------------ the running job's signals
    def attach(self, worker) -> None:
        """Follow ``worker`` (the running job's ``BuildWorker``): connected before the worker thread starts."""
        worker.log.connect(self.on_log)
        worker.progress.connect(self.on_progress)
        worker.molecule_done.connect(self.on_molecule)

    def on_log(self, line: str) -> None:
        job = self.running
        if job is not None:
            job.log.append(str(line))
            self.changed.emit(job.number)

    def on_progress(self, i: int, n: int, stage: str, ident: str) -> None:
        job = self.running
        if job is None:
            return
        if n and job.total != n and job.done_before == 0:
            job.total = int(n)
        if stage not in ("ok", "failed", "cancelled"):
            job.current = f"{ident}: {stage}"
        self.changed.emit(job.number)

    def on_molecule(self, res) -> None:
        job = self.running
        if job is None:
            return
        status = getattr(res, "status", "")
        job.done += 1
        job.ok += status == "ok"
        job.failed += status == "failed"
        self.changed.emit(job.number)

    def cancel_requested(self) -> None:
        """The window's Cancel stopped the running job (it ends after the current step)."""
        job = self.running
        if job is not None and job.status == "running":
            job.status = "cancelling"
            job.log.append("Cancelling after the current step...")
            self.changed.emit(job.number)

    def ended(self, ok: bool, error: str = "", run_dir=None) -> None:
        """The window's build ended (finished, cancelled or failed); the next queued job starts after this event."""
        job = self.running
        self.running = None
        if job is not None:
            if run_dir:
                job.run_dir = Path(run_dir)
            # cancelled only if something was left undone: molecules already in flight (lanes) may all finish
            finished_all = job.total and job.done_before + job.ok + job.failed >= job.total
            status = "failed" if not ok else ("cancelled" if job.status == "cancelling" and not finished_all
                                             else "done")
            self._end(job, status, error)
        QTimer.singleShot(0, self.start_next)

    def _end(self, job: Job, status: str, error: str = "") -> None:
        job.status = status
        job.error = error.splitlines()[0] if error else ""
        job.finished = time.time()
        job.current = ""
        if error:
            job.log.append(error)
        self.changed.emit(job.number)

    # ------------------------------------------------------------------ the user's actions
    def cancel(self, number: int) -> bool:
        """Cancel a queued job (it never starts) or a job on a worker (the cancel goes there). The running local job
        is the window's to cancel (``MainWindow.cancel_build``). A running MD job pauses (``pause_md``)."""
        job = self.job(number)
        if job is not None and job.is_md and job.status == "running":
            return self.pause_md(number)
        if job is not None and job.is_remote and job.status in ("queued", "running") and job.follower is not None:
            job.follower.cancel()
            job.status = "cancelling"
            job.log.append(f"Cancelling on {job.machine}...")
            self.changed.emit(number)
            return True
        if job is None or job.status != "queued" or job.is_remote:
            return False
        self._end(job, "cancelled")
        job.log.append("Cancelled before it started.")
        return True

    def cancel_queued(self) -> int:
        """Cancel every queued job (the window is closing)."""
        n = 0
        for job in self.queued():
            n += self.cancel(job.number)
        return n

    def move(self, number: int, by: int) -> bool:
        """Move a queued job up (``by`` < 0) or down among the queued jobs."""
        job = self.job(number)
        if job is None or job.status != "queued":
            return False
        queued = self.queued()
        i = queued.index(job)
        j = max(0, min(len(queued) - 1, i + by))
        if i == j:
            return False
        other = queued[j]
        a, b = self.jobs.index(job), self.jobs.index(other)
        self.jobs[a], self.jobs[b] = self.jobs[b], self.jobs[a]
        self.changed.emit(job.number)
        self.changed.emit(other.number)
        return True

    def clear_finished(self) -> int:
        gone = [j for j in self.jobs if j.status in FINISHED]
        self.jobs = [j for j in self.jobs if j.status not in FINISHED]
        for j in gone:
            self.removed.emit(j.number)
        return len(gone)

    # ------------------------------------------------------------------ MD jobs (SIM-1)
    def start_next_md(self) -> Job | None:
        """Start the first queued MD job when no MD job runs on this computer."""
        if self.md_running is not None:
            return None
        job = next((j for j in self.jobs if j.is_md and j.status == "queued"), None)
        if job is None:
            return None
        job.status = "running"
        job.started = time.time()
        self.md_running = job
        try:
            job.md_proc = self.md_start(job)
        except Exception as exc:  # noqa: BLE001 - reported in the job
            self.md_running = None
            self._end(job, "failed", f"{type(exc).__name__}: {exc}")
            QTimer.singleShot(0, self.start_next_md)
            return job
        job.log.append(("Following the run that is already going in " if job.md_proc is None else
                        "Started the run supervisor for ") + str(job.output_dir))
        self.changed.emit(job.number)
        self.md_timer.start()
        QTimer.singleShot(0, self.md_poll)
        return job

    def md_poll(self) -> None:
        """Read the running MD job's folder: progress, state, the end of the run."""
        job = self.md_running
        if job is None:
            self.md_timer.stop()
            return
        try:
            st = self.md_status(job.output_dir)
        except Exception as exc:  # noqa: BLE001 - a folder that cannot be read now may be readable next time
            job.log.append(f"Reading the run folder failed: {exc}")
            self.changed.emit(job.number)
            return
        old_state = (job.md or {}).get("state")
        job.md = st
        state = st.get("state", "unknown")
        prog = st.get("progress") or {}
        job.current = (f"{prog.get('part')}: {prog.get('stage')} step {prog.get('step'):,}"
                       if prog.get("step") is not None else "")
        if state != old_state:
            job.log.append(f"MD run: {state}" + (f" ({st.get('error')})" if st.get("error") else ""))
            for a in prog.get("alerts") or []:
                job.log.append(f"alert: {a}")
        proc = job.md_proc
        proc_done = proc is None or getattr(proc, "poll", lambda: 0)() is not None
        if st.get("alive"):
            if state == "pausing" and job.status == "running":
                job.status = "cancelling"
            self.changed.emit(job.number)
            return
        if not proc_done:
            self.changed.emit(job.number)        # the supervisor is starting (no lock yet)
            return
        # no supervisor: the run ended (or never started: no new segment and not done)
        ended = {"done": "done", "paused": "paused", "stopped": "paused", "failed": "failed"}.get(state)
        error = ""
        if proc is not None and state != "done" and len(st.get("segments") or []) <= job.md_segments_before:
            ended = None
        if ended is None or (ended == "failed" and not st.get("segments")):
            ended = "failed"
            error = _supervisor_tail(job.output_dir) or f"the run supervisor ended with the run {state}"
        elif ended == "failed":
            segs = st.get("segments") or []
            error = (segs[-1].get("error") if segs else "") or "the MD run failed"
        self.md_running = None
        self._end(job, ended, error)
        if ended == "paused":
            job.log.append("Paused with a checkpoint: Resume continues it." if state == "paused" else
                           "Stopped: Resume continues it from its last restart file.")
        QTimer.singleShot(0, self.start_next_md)

    def pause_md(self, number: int, now: bool = False) -> bool:
        """Pause a running MD job at its next pause point (``md pause``): the job ends "paused" there."""
        job = self.job(number)
        if job is None or not job.is_md or job.status not in ("running", "cancelling"):
            return False
        try:
            from ..md import supervisor as SV

            job.log.append(SV.request_pause(job.output_dir, now=now))
        except Exception as exc:  # noqa: BLE001 - e.g. the supervisor has not taken the folder yet
            job.log.append(f"Pause failed: {exc}")
            self.changed.emit(number)
            return False
        job.status = "cancelling"
        self.changed.emit(number)
        return True

    def resume_md(self, number: int) -> Job | None:
        """Queue a finished MD job's folder again, to continue the run (``md resume``)."""
        job = self.job(number)
        if job is None or not job.is_md or job.status not in FINISHED or job.output_dir is None:
            return None
        return self.submit(md_job(job.output_dir, name=job.name, resume=True, variant=job.md_variant,
                                  threads=job.md_threads))

    def attach_md(self, folder) -> Job:
        """Follow an MD run that a supervisor already runs (started from the command line, or before the window
        was closed)."""
        return self.submit(md_job(folder, resume=True))


def md_job(folder, *, name: str = "", resume: bool | None = None, variant: str | None = None,
           threads: int | None = None) -> Job:
    """A kind-"md" job for a run folder. ``resume`` None: continue when the run has segments and is not done."""
    folder = Path(folder)
    if resume is None or not name:
        st = _read_md_status(folder)
        if resume is None:
            resume = bool(st.get("segments")) and st.get("state") != "done"
        name = name or f"MD: {st.get('name') or folder.name} ({st.get('template') or 'run'})"
    return Job(name=name, ids=[], cfg=None, run_dir=folder, kind="md", md_resume=bool(resume), md_variant=variant,
               md_threads=threads, summary=f"MD run folder {folder}")


def _read_md_status(folder) -> dict:
    from ..md import supervisor as SV

    return SV.read_status(folder)


def _start_md_supervisor(job: Job):
    """Start the supervisor for an MD job, or None when one already runs the folder (the job follows it)."""
    from ..md import sim
    from ..md import supervisor as SV

    folder = job.output_dir
    if folder is None or not sim.is_run_folder(folder):
        raise sim.RunError(f"{folder}: not an MD run folder")
    st = SV.read_status(folder)
    job.md_segments_before = len(st.get("segments") or [])
    if SV.lock_holder(folder) is not None:
        return None
    if st.get("state") == "done":
        raise sim.RunError("this run is already done")
    resume = job.md_resume or (bool(st.get("segments")) and st.get("state") in sim.RESUMABLE)
    job.md_resume = resume
    return SV.start_detached(folder, resume=resume, variant=job.md_variant, threads=job.md_threads)


def _supervisor_tail(folder, n: int = 3) -> str:
    """The last lines of a run folder's supervisor.log (why a supervisor ended without a run state)."""
    from ..md import sim

    try:
        lines = (Path(folder) / sim.SUPERVISOR_LOG).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return " / ".join(ln.strip() for ln in lines[-n:] if ln.strip())
