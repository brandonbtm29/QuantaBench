"""A job sent to a worker from the job dialog, and followed until its results are here (UI-5 Run on, DIST-3).

``RemoteJob.run`` does it all on a thread of its own, never the GUI's:

1. **Where:** "auto" chooses among this computer and the reachable workers (``choose_machine``): each machine's
   speed for the job's method as ``grid build --weights auto`` sees it (the Machines dock's split settings, the
   newest grid of the same protocol, else the benchmark), divided by the jobs it already has. When this computer
   wins, the job goes back to the local queue (``on_local``).
2. **Submit:** the whole job as one share, the way ``grid build`` submits one (``client.job_body``: the inputs with
   the dialog's charge, multiplicity and starting coordinates, the science settings and run options; the worker
   uses its own threads, GPU and priority).
3. **Follow:** ``GET /v1/jobs/<id>`` every ``poll_s``: state, counts, the molecule and stage running, the log tail.
   A worker that does not answer is retried (backing off to ``MAX_BACKOFF_S``) and reported as unreachable; a worker
   that answers it does not know the job (restarted with its state gone) or refuses the token ends the job.
4. **Cancel:** ``cancel()`` sends ``POST /v1/jobs/<id>/cancel`` (a job not submitted yet is simply not sent).
5. **Results:** once the worker's job has ended, its run folder: the copy SyncThing brought into this computer's
   output folder once its ``worker.json`` vouches for its tables (``client.final_tables``), else the worker's zip
   archive, downloaded into ``<output folder>/from_workers/<worker>/``, as ``grid merge`` does for a share.

Updates go to ``on_update(dict)``, called on the thread: the caller (``gui.job_queue``) reads the results' rows
there (``run_dir``) and relays everything to the GUI thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import client as client_mod
from .client import FINAL_STATES, LOCAL, WorkerClient, WorkerError

AUTO = "auto"
POLL_S = 3.0
MAX_BACKOFF_S = 30.0
SYNC_WAIT_S = 60.0            # how long a finished job's synced copy may take to arrive before the archive is fetched
DOWNLOAD_DIR = "from_workers"
LOG_TAIL = 200


def _noop(*_a) -> None:
    pass


def label_for(name: str, number: int = 0) -> str:
    """A worker job label (``worker.check_label``: a folder-safe name of at most 60 characters) for a GUI job."""
    from ..chem.inputs import safe_id

    base = safe_id(name or "job", "job").strip("._+-") or "job"
    base = base[:40].rstrip("._+-") or "job"
    if not base[0].isalnum():
        base = "job" + base
    return f"job{number}_{base}" if number else base


def worker_choices(refs: dict, version: str | None = None) -> dict[str, str]:
    """{worker name: "" if a job can go there, else why not} for ``client.discover`` refs: reachable, same
    QuantaBench version (a worker refuses jobs of another one)."""
    from .. import __version__

    version = version or __version__
    out = {}
    for name, ref in refs.items():
        if ref.info is None:
            out[name] = ref.error or "not reachable"
        elif str(ref.info.get("quantabench_version", "")) != version:
            out[name] = f"QuantaBench {ref.info.get('quantabench_version') or '?'} there, {version} here"
        else:
            out[name] = ""
    return out


def busy_jobs(info: dict | None) -> int:
    """Jobs a worker is running or has queued (its ``/v1/info`` queue)."""
    q = (info or {}).get("queue") or {}
    return len(q.get("running") or []) + len(q.get("queued") or [])


def choose_machine(cfg, refs: dict, local_busy: int = 0, log: Callable[[str], None] = _noop) -> tuple[str, str]:
    """(LOCAL or a worker's name, why) for a job of ``cfg``: among this computer and the workers a job can go to
    (``worker_choices``), the one with the highest speed for the job's protocol (``client.resolve_weights`` "auto",
    as ``grid build`` splits a list) per job it already has. Ties and an unknown speed favour this computer."""
    from .. import settings as settings_mod
    from . import profile

    me = settings_mod.machine_key(profile.machine_name())
    ok = [n for n, why in worker_choices(refs).items() if not why and settings_mod.machine_key(n) != me]
    names = [LOCAL, *ok]
    if len(names) == 1:
        return LOCAL, "no worker can take it"
    method = profile.bench_method(cfg.method, cfg.solvent)
    try:
        weights = client_mod.resolve_weights("auto", names, method, refs, log=log,
                                             config=client_mod.config_for_workers(cfg))
    except (client_mod.GridError, ValueError, OSError) as exc:
        log(f"Auto: no speeds to compare ({exc}); equal")
        weights = dict.fromkeys(names, 1.0)
    busy = {LOCAL: int(local_busy), **{n: busy_jobs(refs[n].info) for n in ok}}
    score = {n: float(weights.get(n, 0.0)) / (1 + busy[n]) for n in names}
    best = max(names, key=lambda n: (score[n], n == LOCAL))
    why = ", ".join(f"{'this computer' if n == LOCAL else n} {weights.get(n, 0.0):.3g}"
                    + (f" / {1 + busy[n]}" if busy[n] else "") for n in names)
    return best, f"speed per waiting job: {why}"


class RemoteJob:
    """See the module docstring. ``machine``: a worker's name, or ``AUTO``. ``inputs``: the molecules, or a
    callable returning them (called on the thread). ``refs``: name -> ``client.WorkerRef``; None: looked up on the
    thread (``client.discover`` for "auto", which needs each worker's queue; else ``client.known_workers``).
    ``out_dir``: this computer's output folder, where a worker's run folder arrives through SyncThing;
    ``download_dir``: where an archive goes otherwise (default ``out_dir``), under ``from_workers/<worker>/``."""

    def __init__(self, machine: str, inputs, cfg, label: str, *, token: str, out_dir: str | Path,
                 refs: dict | None = None, download_dir: str | Path | None = None,
                 on_update: Callable[[dict], None] = _noop, on_local: Callable[[str], None] | None = None,
                 local_busy: Callable[[], int] = lambda: 0, poll_s: float = POLL_S, sync_wait: float = SYNC_WAIT_S,
                 client_name: str = ""):
        self.machine = machine
        self.inputs = inputs
        self.cfg = cfg
        self.label = label
        self.refs = refs
        self.token = token
        self.out_dir = Path(out_dir)
        self.download_dir = Path(download_dir) if download_dir else self.out_dir
        self.on_update = on_update
        self.on_local = on_local
        self.local_busy = local_busy
        self.poll_s = poll_s
        self.sync_wait = sync_wait
        self.client_name = client_name
        self.job_id = ""
        self.run_folder = ""
        self._cancel = threading.Event()
        self._halt = threading.Event()
        self._wake = threading.Event()
        self._cancel_sent = False
        self._seen_log: list[str] = []
        self.thread: threading.Thread | None = None

    # ------------------------------------------------------------------ control (any thread)
    def start(self) -> RemoteJob:
        self.thread = threading.Thread(target=self.run, name=f"quantabench-remote-{self.label}", daemon=True)
        self.thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()
        self._wake.set()

    def halt(self) -> None:
        """Stop following (the window closes); the job goes on on the worker."""
        self._halt.set()
        self._wake.set()

    # ------------------------------------------------------------------ the thread
    def _emit(self, **fields) -> None:
        if self._halt.is_set():
            return
        try:
            self.on_update(fields)
        except RuntimeError:          # the receiver was deleted (the window closed)
            self._halt.set()

    def _sleep(self, s: float) -> None:
        self._wake.wait(s)
        self._wake.clear()

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 - reported in the job, never a crash of the thread
            self._emit(state="failed", error=f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        if self.refs is None:
            self.refs = (client_mod.discover(token=self.token, timeout=3.0) if self.machine == AUTO
                         else client_mod.known_workers())
        if self.machine == AUTO:
            name, why = choose_machine(self.cfg, self.refs, self.local_busy())
            self._emit(machine=name, log=[f"Auto: {'this computer' if name == LOCAL else name} ({why})"])
            if name == LOCAL:
                if self.on_local is not None and not self._cancel.is_set():
                    self.on_local(why)
                elif self._cancel.is_set():
                    self._emit(state="cancelled", log=["Cancelled before it was sent."])
                return
            self.machine = name
        ref = self.refs.get(self.machine)
        if ref is None:
            self._emit(state="failed", error=f"no worker named {self.machine!r} (Machines dock, settings)")
            return
        c = WorkerClient(ref.url, self.token, timeout=20.0)
        if self._cancel.is_set():
            self._emit(state="cancelled", log=["Cancelled before it was sent."])
            return
        self._emit(state="queued", current=f"sending to {self.machine}")
        inputs = self.inputs() if callable(self.inputs) else list(self.inputs or [])
        if not inputs:
            self._emit(state="failed", error="no molecules to send")
            return
        try:
            body = client_mod.job_body(inputs, self.cfg, self.label, client=self.client_name)
            answer = c.submit_body(body)
        except WorkerError as exc:
            self._emit(state="failed", error=f"{self.machine} did not take the job: {exc}")
            return
        self.job_id = str(answer.get("id", ""))
        job = answer.get("job") or {}
        self.run_folder = str(job.get("run_folder") or "")
        self._emit(remote_id=self.job_id, total=len(inputs), current="",
                   log=[f"Sent to {self.machine} as job {self.job_id} ({len(inputs)} molecule(s))"])
        st = self._follow(c)
        if st is None:
            return
        self._collect(c, st)

    def _follow(self, c: WorkerClient) -> dict | None:
        backoff = self.poll_s
        down_since = 0.0
        while not self._halt.is_set():
            if self._cancel.is_set() and not self._cancel_sent:
                try:
                    c.cancel(self.job_id)
                    self._cancel_sent = True
                    self._emit(state="cancelling", log=[f"Cancel sent to {self.machine}"])
                except WorkerError as exc:
                    if exc.status:
                        self._cancel_sent = True
                    self._emit(log=[f"Cancel not sent yet ({exc}); retrying"])
            try:
                st = c.status(self.job_id, tail=LOG_TAIL)
            except WorkerError as exc:
                if exc.status:        # the worker answered: it does not know the job, or refuses us
                    self._emit(state="failed", error=f"{self.machine}: {exc}")
                    return None
                down_since = down_since or time.time()
                self._emit(current=f"{self.machine} unreachable since {time.strftime('%H:%M', time.localtime(down_since))}"
                                   f" (retrying)", unreachable=True)
                backoff = min(MAX_BACKOFF_S, backoff * 1.6)
                self._sleep(backoff)
                continue
            down_since, backoff = 0.0, self.poll_s
            self._report(st)
            if st.get("state") in FINAL_STATES:
                return st
            self._sleep(self.poll_s)
        return None

    def _report(self, st: dict) -> None:
        counts = st.get("counts") or {}
        progress = st.get("progress") or {}
        state = str(st.get("state", ""))
        mapped = {"queued": "queued", "running": "running", "paused": "running", "interrupted": "running"}.get(
            state, state)
        if self._cancel_sent and mapped in ("queued", "running"):
            mapped = "cancelling"
        current = ""
        if state == "running" and progress.get("molecule"):
            current = f"{progress.get('molecule')}: {progress.get('stage', '')}"
        elif state == "queued":
            current = f"queued on {self.machine}"
        elif state in ("paused", "interrupted"):
            current = f"{state} on {self.machine}"
        self.run_folder = str(st.get("run_folder") or self.run_folder)
        self._emit(state=mapped if mapped not in FINAL_STATES else None, ok=int(counts.get("ok", 0) or 0),
                   failed=int(counts.get("failed", 0) or 0), current=current, unreachable=False,
                   log=self._new_log(st.get("log_tail") or []))

    def _new_log(self, tail: list) -> list[str]:
        """The lines of ``tail`` not reported yet (the tail is the worker's last lines; they overlap)."""
        tail = [str(x) for x in tail]
        seen = self._seen_log
        new = tail
        if seen:
            last = seen[-1]
            idx = max((i for i, x in enumerate(tail) if x == last), default=-1)
            new = tail[idx + 1:] if idx >= 0 else tail
        if new:
            self._seen_log = (seen + new)[-LOG_TAIL:]
        return new

    def _collect(self, c: WorkerClient, st: dict) -> None:
        state = str(st.get("state"))
        error = str(st.get("error") or "")
        self._emit(current="bringing the results here")
        try:
            run_dir, how = self.fetch(c)
        except WorkerError as exc:
            self._emit(state=state, error=error or f"results not fetched: {exc}",
                       log=[f"The results could not be fetched: {exc}"])
            return
        self._emit(state=state, error=error, run_dir=str(run_dir) if run_dir else "",
                   log=[f"Results: {run_dir} ({how})" if run_dir else "No finished molecule to bring back"])

    def fetch(self, c: WorkerClient) -> tuple[Path | None, str]:
        """(run folder here, "synced" | "archive") of the ended job; see the module docstring."""
        if not client_mod._FOLDER_RE.fullmatch(self.run_folder or ""):
            raise WorkerError(f"run folder name {self.run_folder!r} is not a single folder name")
        if not client_mod._safe_name(self.machine):
            raise WorkerError(f"machine name {self.machine!r} cannot name a folder")
        here = self.out_dir / self.run_folder
        tables = client_mod._synced(here, self.job_id, self.sync_wait)
        if tables is not None:
            return (here if "summary.csv" in tables["files"] else None), "synced"
        dest = self.download_dir / DOWNLOAD_DIR / self.machine
        folder = c.download_archive(self.job_id, dest, expected_top=self.run_folder)
        tables = (client_mod._job_file(folder) or {}).get("tables") or {}
        has_summary = "summary.csv" in (tables.get("files") or {}) or (folder / "summary.csv").is_file()
        return (folder if has_summary else None), "archive"
