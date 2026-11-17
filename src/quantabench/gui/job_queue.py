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
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

#: job states; "cancelling": the running job was told to stop and has not ended yet
STATES = ("queued", "running", "cancelling", "done", "failed", "cancelled")
FINISHED = ("done", "failed", "cancelled")
LOG_LINES = 2000            # per job (the window's Log keeps its own 5,000)


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

    @property
    def is_remote(self) -> bool:
        return self.run_on not in ("local", "")

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
        n = self.total or len(self.ids)
        done = self.done_before + self.done
        if self.status == "queued":
            return f"{n:,} molecule{'s' if n != 1 else ''}"
        text = f"{done:,} / {n:,}"
        if self.failed:
            text += f" ({self.failed:,} failed)"
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

    # ------------------------------------------------------------------ submitting
    def submit(self, job: Job) -> Job:
        job.number = self._next_number
        self._next_number += 1
        job.status = "queued"
        job.submitted = time.time()
        if not job.total:
            job.total = len(job.ids)
        self.jobs.append(job)
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
        return sum(1 for j in self.jobs if not j.is_remote and j.status in ("queued", "running", "cancelling"))

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
        job = next((j for j in self.jobs if j.status == "queued" and not j.is_remote), None)
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
        is the window's to cancel (``MainWindow.cancel_build``)."""
        job = self.job(number)
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
