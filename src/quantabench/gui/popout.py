"""Pop-out panels (UI-8): any dock can leave the window and come back.

    View > Pop out panel     the docks in view, each a check: checked = in its own window (``pop_out``), unchecked =
                             back in the main window. The ⧉ in the dock's title bar and a double-click on its title do
                             the same.
    DockTitleBar             the docks' title bar (UI-8b): the title, ⧉ (pop out / dock back) and ✕ (close), instead
                             of Qt's own float button, which is easy to miss. Dragging and double-clicking the title
                             still move and float the dock (the bar leaves those mouse events to Qt).
    View > Dock all panels   every popped-out panel back into the main window (Ctrl+Shift+D)
    pop_out(dock)            float a dock as a window of a usable size (at least POPOUT_SIZE), next to the main
                             window, raised

A popped-out panel is part of its workspace's arrangement (``QMainWindow.saveState``, ``gui.workspaces``): Analyze
keeps its Plot in a window of its own while Calculate keeps its docks in place, and the next start brings them back
so. The Plot dock has a ⧉ button of its own (``gui.plot_panel``): a scatter plot is the panel most worth a big window.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from . import keymap

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


class DockTitleBar(QWidget):
    """A dock's title bar: its title, ⧉ and ✕. Clicks on the title and the empty part are left to QDockWidget (Qt
    then drags, docks and floats it as with its own bar), so only the two buttons take mouse events."""

    BUTTON_STYLE = "QToolButton { border: none; padding: 0 4px; } QToolButton:hover { background: palette(midlight); }"

    def __init__(self, dock: QDockWidget):
        super().__init__(dock)
        self.dock = dock
        self.setObjectName("dock_title_bar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 1, 2, 1)
        lay.setSpacing(0)
        self.title = QLabel(dock.windowTitle())
        self.title.setObjectName("dock_title")
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(self.title, 1)
        self.popout_btn = QToolButton()
        self.popout_btn.setObjectName("dock_popout_btn")
        self.popout_btn.setText("⧉")
        self.popout_btn.setCheckable(True)
        self.popout_btn.setAutoRaise(True)
        self.popout_btn.setStyleSheet(self.BUTTON_STYLE)
        self.popout_btn.clicked.connect(lambda on: pop_out(self.dock, on))
        self.close_btn = QToolButton()
        self.close_btn.setObjectName("dock_close_btn")
        self.close_btn.setText("✕")
        self.close_btn.setAutoRaise(True)
        self.close_btn.setStyleSheet(self.BUTTON_STYLE)
        self.close_btn.setToolTip("Close the panel (View menu brings it back)")
        self.close_btn.clicked.connect(lambda _c=False: self.dock.close())
        lay.addWidget(self.popout_btn)
        lay.addWidget(self.close_btn)
        dock.windowTitleChanged.connect(self.title.setText)
        dock.topLevelChanged.connect(self._sync)
        dock.featuresChanged.connect(lambda _f: self._sync())
        self._sync()

    def _sync(self, *_args) -> None:
        f = self.dock.features()
        floating = self.dock.isFloating()
        self.popout_btn.setVisible(bool(f & QDockWidget.DockWidgetFeature.DockWidgetFloatable))
        self.close_btn.setVisible(bool(f & QDockWidget.DockWidgetFeature.DockWidgetClosable))
        self.popout_btn.blockSignals(True)
        self.popout_btn.setChecked(floating)
        self.popout_btn.blockSignals(False)
        self.popout_btn.setToolTip("Back into the main window (View > Dock all panels: every panel)" if floating
                                   else "Pop out into a window of its own (or double-click the title)")


def add_title_bar(dock: QDockWidget) -> DockTitleBar:
    """Give ``dock`` the ⧉/✕ title bar (once)."""
    bar = dock.titleBarWidget()
    if isinstance(bar, DockTitleBar):
        return bar
    bar = DockTitleBar(dock)
    dock.setTitleBarWidget(bar)
    return bar


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
    """View > Pop out panel and View > Dock all panels, and the ⧉/✕ title bar on every dock the window has now (a
    later dock can call ``add_title_bar``)."""
    for d in docks(win):
        add_title_bar(d)
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
    keymap.bind(win, back, "view.dock_all", "Every panel in a window of its own back into the main window")
    back.triggered.connect(lambda _c=False: _dock_all_said(win))
    view.addAction(back)
    win.popout_menu = sub
    win.dock_all_action = back


def _dock_all_said(win) -> None:
    n = dock_all(win)
    if hasattr(win, "statusBar"):
        win.statusBar().showMessage(f"{n} panel(s) docked" if n else "Every panel is docked", 5000)
