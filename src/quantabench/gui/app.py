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
    os.environ.setdefault("QT_API", "pyside6")  # pyvistaqt/qtpy: use PySide6 even if PyQt is installed

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

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
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
