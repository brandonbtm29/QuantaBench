"""The Results dock folds to its header row when the conformer table has one row or none (UI-8b).

A molecule with one conformer (or none built yet) leaves the Results dock mostly empty. While View > Fold Results for
one conformer is on (the default, remembered per machine), the dock shrinks to the table's header and its one row, and
the 3D view gets the height. A molecule with more conformers brings the height back, unless the dock was resized by
hand in between (then the hand-made height stays).

Only the Conformers tab of a docked, visible Results dock in front of its tab group is folded: a dock in a window of
its own, behind another tab (Spectra, Jobs, Table) or showing the Log is left as it is.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QAction

SETTING = "results/fold_single"
MIN_GAIN_PX = 12             # fold only when it frees at least this much


def _setting(key, default):
    try:
        from .main_window import _window_settings

        value = _window_settings().value(key)
    except Exception:  # noqa: BLE001 - a preference
        return default
    return default if value is None else str(value).lower() in ("true", "1")


def _set_setting(key, value) -> None:
    try:
        from .main_window import _window_settings

        _window_settings().setValue(key, value)
    except Exception:  # noqa: BLE001, S110
        pass


class ResultsFolder(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.dock = win.results_dock
        self.table = win.conf_table
        self.folded_from: int | None = None     # the dock's height before it was folded
        self.folded_to: int | None = None       # the height it was folded to
        self.timer = QTimer(self)               # setRowCount(0) then setRowCount(n): look once both are done
        self.timer.setSingleShot(True)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self.check)
        model = self.table.model()
        for sig in (model.rowsInserted, model.rowsRemoved, model.modelReset):
            sig.connect(self._later)               # a method: disconnected when the folder goes
        win.bottom_tabs.currentChanged.connect(self._later)
        self.action = QAction("Fold Results for one conformer", win)
        self.action.setObjectName("fold_results")
        self.action.setCheckable(True)
        self.action.setChecked(_setting(SETTING, True))
        self.action.setToolTip("A molecule with one conformer: the Results dock shrinks to the table's header and "
                               "its row, and the 3D view gets the height")
        self.action.toggled.connect(self._on_toggled)

    def _later(self, *_args) -> None:
        self.timer.start()

    def enabled(self) -> bool:
        return self.action.isChecked()

    def can_resize(self) -> bool:
        d = self.dock
        return (d.isVisible() and not d.isFloating() and not d.visibleRegion().isEmpty()
                and self.win.bottom_tabs.currentWidget() is self.table)

    def folded_height(self) -> int:
        """The dock's height with the table showing its header and rows (at most one) and nothing below."""
        t = self.table
        rows = t.verticalHeader().length() if t.rowCount() else 0
        chrome = self.dock.height() - t.viewport().height()
        return chrome + rows + 2

    def check(self) -> None:
        if not self.enabled() or not self.can_resize():
            return
        if self.table.rowCount() <= 1:
            h = self.dock.height()
            target = self.folded_height()
            if target < h - MIN_GAIN_PX:
                if self.folded_from is None:
                    self.folded_from = h
                self.win.resizeDocks([self.dock], [target], Qt.Orientation.Vertical)
                # the layout applies the height later, and no lower than the dock's minimum
                self.folded_to = max(target, self.dock.minimumHeight(), self.dock.minimumSizeHint().height())
                QTimer.singleShot(0, self._record)
        else:
            self.unfold()

    def _record(self) -> None:
        if self.folded_from is not None and not self.dock.isFloating():
            self.folded_to = self.dock.height()

    def unfold(self) -> None:
        """Back to the height before folding, unless the dock was resized by hand since."""
        if self.folded_from is None:
            return
        if self.folded_to is not None and abs(self.dock.height() - self.folded_to) <= 2 and not self.dock.isFloating():
            self.win.resizeDocks([self.dock], [self.folded_from], Qt.Orientation.Vertical)
        self.folded_from = self.folded_to = None

    def _on_toggled(self, on: bool) -> None:
        _set_setting(SETTING, on)
        if on:
            self.timer.start()
        else:
            self.unfold()


def install(win) -> None:
    """The folder (``win.results_folder``) and its View menu entry, after the docks' toggles (before Open in OVITO
    viewport)."""
    folder = ResultsFolder(win)
    win.results_folder = folder
    view = next((a.menu() for a in win.menuBar().actions()
                 if a.menu() is not None and a.text().replace("&", "") == "View"), None)
    if view is not None:
        before = getattr(win, "a_ovito", None)
        if before is not None and before in view.actions():
            view.insertAction(before, folder.action)
        else:
            view.addAction(folder.action)
    folder.timer.start()
