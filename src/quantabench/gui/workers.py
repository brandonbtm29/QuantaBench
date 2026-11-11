"""Background workers so the window stays responsive while chemistry runs."""

from __future__ import annotations

import contextlib
import threading
import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot


class BuildWorker(QObject):
    progress = Signal(int, int, str, str)      # index, total, stage, molecule id
    log = Signal(str)
    molecule_done = Signal(object)             # MoleculeResult
    finished = Signal(list, object)            # results, run_dir (Path | None)
    failed = Signal(str)

    def __init__(self, inputs, cfg, lanes=None, run_dir=None):
        """``lanes``: a lane spec ("1", "auto", "4", "gpu:1x4,cpu:5x2"), a list of ``dist.lanes.Lane``, or
        None for this machine's "Parallel lanes" setting (``Settings.lanes_for``).

        ``inputs``: the molecules, a callable returning them (called on the worker thread, e.g. to read
        thousands of stored inputs from a job table), or None with ``run_dir`` to resume that run folder.
        ``run_dir``: run (or continue) the batch in this folder through the job table."""
        super().__init__()
        self.inputs = inputs
        self.cfg = cfg
        self.lanes = lanes
        self.run_dir = run_dir
        self._stop = threading.Event()
        self.control = None          # jobs.BatchControl of a running batch: this machine's pause (DIST-7)

    def cancel(self) -> None:
        self._stop.set()

    def _lanes(self) -> list:
        """The lanes for a batch (parsed here, off the GUI thread: "auto" may detect the GPU). A setting that
        cannot be used on this machine is reported and the batch runs in one lane."""
        from ..dist import lanes as lanes_mod
        from ..dist import profile

        if isinstance(self.lanes, list):
            return self.lanes
        if self.lanes is None:
            return profile.resolve_lanes(None, self.cfg, log=self.log.emit)[0]
        try:
            return lanes_mod.parse_lanes(self.lanes, self.cfg)
        except ValueError as exc:
            self.log.emit(f"Parallel lanes {self.lanes!r} not usable here ({exc}); running in one lane")
            return lanes_mod.parse_lanes("1", self.cfg)

    def run(self) -> None:
        """One molecule runs directly; a batch goes through the resumable job table (``chem.jobs``), so an
        interrupted GUI batch can be finished with ``quantabench-cli build --resume RUN_DIR``. A batch runs in
        this machine's lanes from the settings (worker processes; ``dist.lanes``); results still arrive one molecule at a
        time through ``molecule_done``.

        Run priority (``quantabench.priority``): lanes lower their own processes; work done here, in this process
        (one lane, or one molecule), runs on this thread lowered to the run's priority (``priority.thread_lowered``),
        never the GUI thread or the GUI process."""
        from .. import priority
        from ..chem import jobs, pipeline
        from ..settings import Settings

        try:
            from .. import resources

            resources.set_molecule_limits(self.cfg)   # memory budget and time limit of embedding and Hessian
            inputs = self.inputs() if callable(self.inputs) else self.inputs
            run_dir = self.run_dir
            if run_dir is None and inputs is not None and len(inputs) > 1 and self.cfg.write_files:
                run_dir = pipeline.new_run_dir(self.cfg.output_dir or Settings().output_path(), f"batch_{len(inputs)}")
            if run_dir is not None:
                from ..dist import control as control_mod

                lanes = self._lanes()
                in_process = len(lanes) <= 1          # several lanes: this thread only hands out the molecules
                # this machine's pause (Machines dock, `worker pause`): the batch holds and goes on at resume
                self.control = jobs.BatchControl(source=control_mod.pause_gate)
                with priority.thread_lowered(self.cfg.priority) if in_process else contextlib.nullcontext():
                    results, run_dir = jobs.run_resumable(
                        inputs or None, self.cfg, run_dir,
                        progress=self.progress.emit, should_stop=self._stop.is_set, log=self.log.emit,
                        on_result=self.molecule_done.emit, lanes=lanes, keep_results=False, control=self.control)
            else:
                with priority.thread_lowered(self.cfg.priority):
                    results, run_dir = pipeline.run(
                        inputs, self.cfg,
                        progress=self.progress.emit, should_stop=self._stop.is_set, log=self.log.emit,
                        on_result=self.molecule_done.emit)
            self.finished.emit(results, run_dir)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


class FunctionWorker(QObject):
    """Run any callable off the GUI thread."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self) -> None:
        try:
            self.done.emit(self.fn(*self.args, **self.kwargs))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _Relay(QObject):
    """Calls plain Python callbacks on the GUI thread.

    PySide6 runs a closure or lambda connected to a worker signal in the *emitting* (worker) thread,
    even with a queued connection; only slots of a QObject that lives in the GUI thread are delivered
    there. This object lives in the owner's thread, so callbacks routed through it can touch widgets."""

    def __init__(self, owner: QObject, on_done=None, on_failed=None):
        super().__init__(owner)
        self._on_done, self._on_failed = on_done, on_failed

    @Slot(object)
    def done(self, value) -> None:
        if self._on_done is not None:
            self._on_done(value)

    @Slot(str)
    def failed(self, msg: str) -> None:
        if self._on_failed is not None:
            self._on_failed(msg)


def start(worker: QObject, owner: QObject, on_done=None, on_failed=None) -> QThread:
    """Move ``worker`` to a new QThread owned by ``owner`` and start it. Cleans up when done.

    ``on_done``/``on_failed`` may be any callables (closures, lambdas): they are routed through a relay
    in the GUI thread. Connect worker signals directly only to bound methods of GUI-thread QObjects."""
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_done is not None or on_failed is not None:
        relay = _Relay(owner, on_done, on_failed)
        if on_done is not None:
            worker.done.connect(relay.done)
        if on_failed is not None:
            worker.failed.connect(relay.failed)
        thread.finished.connect(relay.deleteLater)
        worker._relay = relay
    for sig in ("finished", "failed", "done"):
        if hasattr(worker, sig):
            getattr(worker, sig).connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    worker._thread = thread
    # Python must hold the worker until its thread ends, or the wrapper is garbage-collected mid-run
    alive = owner.__dict__.setdefault("_live_workers", set())
    alive.add(worker)
    thread.finished.connect(lambda: alive.discard(worker))
    thread.start()
    return thread
