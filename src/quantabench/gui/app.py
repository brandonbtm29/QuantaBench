"""GUI entry point: ``quantabench`` or ``python -m quantabench``."""

from __future__ import annotations

import os
import sys
import traceback


def main() -> int:
    from .. import resources
    from .. import settings as settings_mod

    s = settings_mod.load()
    # Thread caps must be in the environment before NumPy/PyTorch load their thread pools.
    resources.apply_thread_limits(s.resources.threads)
    from ..chem import torch_compat

    torch_compat.apply_settings(s, strict=False)       # dxtb linear algebra / GPU modes for everything the GUI runs
    os.environ.setdefault("QT_API", "pyside6")  # pyvistaqt/qtpy: use PySide6 even if PyQt is installed

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    from . import sketcher_web

    # The Sketch dock's URL scheme (Qt WebEngine, optional) can only be registered before the QApplication exists
    sketcher_web.prepare_before_app()
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("QuantaBench")
    app.setOrganizationName("QuantaBench")
    app.setStyle("Fusion")  # follows the system light/dark scheme on Windows 11, Linux and macOS

    def excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.__stderr__.write(text)
        QMessageBox.critical(None, "QuantaBench error", text[-3000:])

    sys.excepthook = excepthook
    try:
        from .main_window import MainWindow
    except (ImportError, OSError) as exc:  # e.g. a chemistry DLL blocked by Smart App Control
        msg = f"{type(exc).__name__}: {exc}"
        hint = resources.explain_import_error(msg)
        QMessageBox.critical(None, "QuantaBench cannot start",
                             f"A required library failed to load:\n\n{msg}\n\n{hint}\n\n"
                             "Run `quantabench-cli doctor` for details.")
        return 1
    win = MainWindow(s)
    win.show()
    _collect_in_gui_thread(app)
    return app.exec()


def _collect_in_gui_thread(app) -> None:
    """Python's cyclic garbage collector runs in whichever thread happens to allocate; when that is a worker
    thread (a batch, a benchmark) and the garbage holds a closed dialog or panel, Qt destroys a widget outside
    the GUI thread and aborts the process. So automatic collection is off and cycles are collected here, in
    the GUI thread, every few seconds."""
    import gc

    from PySide6.QtCore import QTimer

    gc.disable()
    timer = QTimer(app)
    timer.setInterval(5000)
    timer.timeout.connect(gc.collect)
    timer.start()
    app._qb_gc_timer = timer


def entry() -> None:
    """GUI-script entry point (``quantabench``): ``main`` plus a hard exit where the GPU runtime would hang
    after the window closes (``resources.hard_exit_if_needed``)."""
    code = main()
    from .. import resources

    resources.hard_exit_if_needed(code)
    raise SystemExit(code)


if __name__ == "__main__":
    entry()
