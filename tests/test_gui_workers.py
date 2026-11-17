"""FIX-3: a worker started with gui.workers.start runs in its own thread and is destroyed in the GUI thread. Destroyed
in its own thread at the thread's end, its ~QObject held a Qt signal-slot lock while PySide waited for the GIL, and the
GUI thread (holding the GIL inside a layout's QObject::connect) waited for that lock: the CI hang of Nov 13-16."""

import threading
import time

import pytest
from conftest import HAS_DISPLAY

pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=10.0):
    from PySide6.QtCore import QEvent

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        time.sleep(0.01)
    return cond()


def test_a_worker_runs_in_its_thread_and_is_destroyed_in_the_gui_thread():
    from PySide6.QtCore import QObject

    from quantabench.gui import workers

    app = _app()
    owner = QObject()
    gui = threading.get_ident()
    ran_in, done, destroyed_in = [], [], []

    def job():
        ran_in.append(threading.get_ident())
        return 42

    w = workers.FunctionWorker(job)
    w.destroyed.connect(lambda *_a: destroyed_in.append(threading.get_ident()))
    thread = workers.start(w, owner, on_done=lambda v: done.append((v, w.thread() is app.thread())))
    assert _wait(app, lambda: bool(destroyed_in))
    assert ran_in and ran_in[0] != gui                              # the work ran off the GUI thread
    assert done == [(42, True)]                                     # and the worker was back home when it reported
    assert destroyed_in == [gui]                                    # destroyed in the GUI thread
    assert not owner.__dict__.get("_live_workers")                  # let go of, in the GUI thread too
    assert _wait(app, lambda: not _alive(thread))


def _alive(thread) -> bool:
    try:
        return thread.isRunning()
    except RuntimeError:                                            # deleted (deleteLater): it ended
        return False


def test_many_workers_while_the_gui_thread_lays_out_widgets():
    """The shape of the hang, many times over: workers ending while the GUI thread shows and hides widgets in a
    layout (each show runs QLayout::activate, which connects signals). Before the fix this could deadlock."""
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    from quantabench.gui import workers

    app = _app()
    win = QWidget()
    lay = QVBoxLayout(win)
    labels = [QLabel(f"note {i}") for i in range(20)]
    for lab in labels:
        lab.setWordWrap(True)
        lay.addWidget(lab)
    win.show()
    finished = []
    for i in range(60):
        w = workers.FunctionWorker(lambda i=i: i)
        workers.start(w, win, on_done=finished.append)
        for lab in labels[i % 5::5]:
            lab.setVisible(not lab.isVisible())
        app.processEvents()
    assert _wait(app, lambda: len(finished) == 60, 30)
    assert sorted(finished) == list(range(60))
    assert _wait(app, lambda: not win.__dict__.get("_live_workers"), 10)
    win.close()
