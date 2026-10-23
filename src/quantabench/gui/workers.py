"""Background workers so the window stays responsive while chemistry runs."""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot


class BuildWorker(QObject):
    progress = Signal(int, int, str, str)      # index, total, stage, molecule id
    log = Signal(str)
    molecule_done = Signal(object)             # MoleculeResult
    finished = Signal(list, object)            # results, run_dir (Path | None)
    failed = Signal(str)

    def __init__(self, inputs, cfg):
        super().__init__()
        self.inputs = inputs
        self.cfg = cfg
        self._stop = threading.Event()

    def cancel(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """One molecule runs directly; a batch goes through the resumable job table (``chem.jobs``), so an
        interrupted GUI batch can be finished with ``quantabench-cli build --resume RUN_DIR``."""
        from ..chem import jobs, pipeline
        from ..settings import Settings

        try:
            if len(self.inputs) > 1 and self.cfg.write_files:
                run_dir = pipeline.new_run_dir(self.cfg.output_dir or Settings().output_path(),
                                               f"batch_{len(self.inputs)}")
                results, run_dir = jobs.run_resumable(
                    self.inputs, self.cfg, run_dir,
                    progress=self.progress.emit, should_stop=self._stop.is_set, log=self.log.emit,
                    on_result=self.molecule_done.emit)
            else:
                results, run_dir = pipeline.run(
                    self.inputs, self.cfg,
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
