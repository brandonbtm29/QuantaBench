"""Pop-out panels (UI-8): any dock can leave the window and come back.

    View > Pop out panel     the docks in view, each a check: checked = in its own window (``pop_out``), unchecked =
                             back in the main window. The dock's own ⧉ title button and a double-click on its title do
                             the same.
    View > Dock all panels   every popped-out panel back into the main window (Ctrl+Shift+D)
    pop_out(dock)            float a dock as a window of a usable size (at least POPOUT_SIZE), next to the main
                             window, raised

A popped-out panel is part of its workspace's arrangement (``QMainWindow.saveState``, ``gui.workspaces``): Analyze
keeps its Plot in a window of its own while Calculate keeps its docks in place, and the next start brings them back
so. The Plot dock has a ⧉ button of its own (``gui.plot_panel``): a scatter plot is the panel most worth a big window.
"""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QDockWidget, QMainWindow, QMenu

POPOUT_SIZE = QSize(640, 520)


def docks(win: QMainWindow) -> list[QDockWidget]:
    """The window's own docks (an OVITO viewport and other later docks too), in creation order."""
    return [d for d in win.findChildren(QDockWidget) if d.parent() is win]


def pop_out(dock: QDockWidget, on: bool = True) -> None:
    """Float ``dock`` in a window of its own (``on``) or put it back."""
    if not on:
        dock.setFloating(False)
        return
    win = dock.parent()
    if dock.isHidden():
        dock.show()
    dock.setFloating(True)
    size = dock.size().expandedTo(POPOUT_SIZE)
    dock.resize(size)
    if isinstance(win, QMainWindow):
        g = win.frameGeometry()
        screen = win.screen().availableGeometry() if win.screen() is not None else None
        x = g.right() - size.width() - 40
        y = g.top() + 80
        if screen is not None:
            x = max(screen.left(), min(x, screen.right() - size.width()))
            y = max(screen.top(), min(y, screen.bottom() - size.height()))
        dock.move(x, y)
    dock.raise_()
    dock.activateWindow()


def dock_all(win: QMainWindow) -> int:
    """Every floating dock back into the window; how many came back."""
    n = 0
    for d in docks(win):
        if d.isFloating():
            d.setFloating(False)
            n += 1
    return n


def _fill(win: QMainWindow, menu: QMenu) -> None:
    menu.clear()
    shown = [d for d in docks(win) if d.isVisible() or d.isFloating()]
    for d in shown:
        a = menu.addAction(d.windowTitle() or d.objectName())
        a.setCheckable(True)
        a.setChecked(d.isFloating())
        a.toggled.connect(lambda on, dd=d: pop_out(dd, on))
    if not shown:
        menu.addAction("No panel in view").setEnabled(False)


def install(win: QMainWindow) -> None:
    """View > Pop out panel and View > Dock all panels."""
    view = next((a.menu() for a in win.menuBar().actions()
                 if a.menu() is not None and a.text().replace("&", "") == "View"), None)
    if view is None:
        raise RuntimeError("no View menu")
    view.addSeparator()
    sub = view.addMenu("Pop out panel")
    sub.setObjectName("popout_menu")
    sub.setToolTipsVisible(True)
    sub.aboutToShow.connect(lambda: _fill(win, sub))
    back = QAction("Dock all panels", win)
    back.setObjectName("dock_all")
    back.setShortcut(QKeySequence("Ctrl+Shift+D"))
    back.setToolTip("Every panel in a window of its own back into the main window")
    back.triggered.connect(lambda _c=False: _dock_all_said(win))
    view.addAction(back)
    win.popout_menu = sub
    win.dock_all_action = back


def _dock_all_said(win) -> None:
    n = dock_all(win)
    if hasattr(win, "statusBar"):
        win.statusBar().showMessage(f"{n} panel(s) docked" if n else "Every panel is docked", 5000)
