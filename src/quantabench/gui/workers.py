"""Background workers so the window stays responsive while chemistry runs."""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QThread, Signal


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
        from ..chem import pipeline

        try:
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


def start(worker: QObject, owner: QObject) -> QThread:
    """Move ``worker`` to a new QThread owned by ``owner`` and start it. Cleans up when done."""
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
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
