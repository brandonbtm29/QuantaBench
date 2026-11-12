"""Task workspaces (UI-1): the activity bar on the left switches the window between Build, Calculate, Simulate,
Analyze and Library (Alt+1 ... Alt+5; Control+1 ... Control+5 on macOS, ``SHORTCUT_MODIFIER``).

A workspace is a set of docks around the shared 2D/3D view and playback bar: which docks are shown, where, and
which tab of a group is in front. The view, the molecule list, the selection and every panel are the same objects
in every workspace; only the arrangement changes, so nothing is rebuilt or reloaded on a switch.

- **Defaults** (``WORKSPACES``): each workspace lists its docks by object name per dock area, the first of a group
  in front. A dock that is not listed is hidden in that workspace (its View menu toggle still shows it).
- **The user's arrangement** is kept per workspace: leaving a workspace stores ``QMainWindow.saveState()`` under its
  key, coming back restores it, so a dock moved, closed or tabbed in Analyze stays that way in Analyze only.
  ``WorkspaceManager.save`` / ``load`` keep the states in the window settings (QSettings: per machine and user, the
  registry on Windows; an INI file under the tests), with the current workspace. View > Reset layout drops the
  current workspace's arrangement and applies its default again. A dock layout saved before workspaces existed is
  not carried over: the first start shows the defaults.
- Docks are found by object name, so a panel that failed to install (``MainWindow._install_panels``) is simply
  missing from the layout. Docks the window creates later (an OVITO viewport) are left alone.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from PySide6.QtCore import QByteArray, QObject, QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import QDockWidget, QMainWindow, QToolBar, QWidget

#: bump when the default layouts change incompatibly: saved arrangements of an older version are dropped
STATE_VERSION = 1


@dataclass(frozen=True)
class Workspace:
    key: str
    label: str
    shortcut: str
    tip: str
    left: tuple[str, ...] = ()
    right: tuple[str, ...] = ()
    bottom: tuple[str, ...] = ()

    @property
    def docks(self) -> tuple[str, ...]:
        return self.left + self.right + self.bottom


#: the workspace shortcuts' modifier: Alt+1 ... Alt+5 (Ctrl+1 ... Ctrl+3 are the mouse modes). On macOS Control+1 ...
#: (Qt's "Meta"): Command+Shift+3/4/5 are the system's screenshots, and Option+digit types characters ([ on a German
#: keyboard), which the SMILES box needs.
SHORTCUT_MODIFIER = "Meta" if sys.platform == "darwin" else "Alt"


def _key(n: int) -> str:
    return f"{SHORTCUT_MODIFIER}+{n}"


WORKSPACES: tuple[Workspace, ...] = (
    Workspace("build", "Build", _key(1),
              "Build: add molecules (SMILES, names, CAS, files, the 2D sketcher) and set how they look",
              left=("molecules",), right=("appearance",), bottom=("results",)),
    Workspace("calculate", "Calculate", _key(2),
              "Calculate: the pipeline (conformers, refinement, ranking, properties, QM), the review before a run "
              "and the machines that run it",
              left=("molecules",), right=("pipeline", "machines", "appearance"), bottom=("results",)),
    Workspace("simulate", "Simulate", _key(3),
              "Simulate: molecular dynamics set-up (the surface builder and LAMMPS export today)",
              left=("molecules",), right=("simulate", "appearance"), bottom=("results",)),
    Workspace("analyze", "Analyze", _key(4),
              "Analyze: conformers, spectra and molecular surfaces of the selected molecule",
              left=("molecules",), right=("surfaces", "appearance"), bottom=("results", "spectra")),
    Workspace("library", "Library", _key(5),
              "Library: run folders on this computer and the synced ones; open or resume them",
              left=("molecules",), right=("library",), bottom=("results",)),
)
BY_KEY = {w.key: w for w in WORKSPACES}
DEFAULT = "calculate"          # the arrangement closest to the window before workspaces existed
#: every dock a workspace arranges (by object name)
MANAGED = tuple(dict.fromkeys(name for w in WORKSPACES for name in w.docks))

#: where a dock goes when a workspace does not use it (hidden)
_HOME = {"molecules": Qt.DockWidgetArea.LeftDockWidgetArea, "results": Qt.DockWidgetArea.BottomDockWidgetArea,
         "spectra": Qt.DockWidgetArea.BottomDockWidgetArea}
_AREAS = (("left", Qt.DockWidgetArea.LeftDockWidgetArea), ("right", Qt.DockWidgetArea.RightDockWidgetArea),
          ("bottom", Qt.DockWidgetArea.BottomDockWidgetArea))

# 24 x 24 line icons, drawn in the palette's text colour ({c})
_ICONS = {
    "build": ('<polygon points="12,3 19.8,7.5 19.8,16.5 12,21 4.2,16.5 4.2,7.5" fill="none" stroke="{c}" '
              'stroke-width="1.8" stroke-linejoin="round"/><circle cx="12" cy="12" r="4" fill="none" stroke="{c}" '
              'stroke-width="1.6"/>'),
    "calculate": ('<path d="M9 3h6M10 3v6l-5.5 9.5A1.6 1.6 0 0 0 5.9 21h12.2a1.6 1.6 0 0 0 1.4-2.5L14 9V3" fill="none" '
                  'stroke="{c}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
                  '<path d="M7.6 15h8.8" stroke="{c}" stroke-width="1.6" stroke-linecap="round"/>'),
    "simulate": ('<path d="M2.5 13c2.5-7 5-7 7.5 0s5 7 7.5 0 3-4 4-3" fill="none" stroke="{c}" stroke-width="1.8" '
                 'stroke-linecap="round"/><circle cx="6.2" cy="8.2" r="1.7" fill="{c}"/>'
                 '<circle cx="13.8" cy="17.8" r="1.7" fill="{c}"/>'),
    "analyze": ('<path d="M4 3v17h17" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" '
                'stroke-linejoin="round"/><circle cx="8.5" cy="15" r="1.6" fill="{c}"/><circle cx="12" cy="11.5" '
                'r="1.6" fill="{c}"/><circle cx="15.5" cy="12.5" r="1.6" fill="{c}"/><circle cx="18.5" cy="6.5" '
                'r="1.6" fill="{c}"/><path d="M7 17.5L20 5" stroke="{c}" stroke-width="1.2" stroke-dasharray="2 2"/>'),
    "library": ('<rect x="3.5" y="4" width="4" height="16" rx="0.8" fill="none" stroke="{c}" stroke-width="1.7"/>'
                '<rect x="9.5" y="4" width="4" height="16" rx="0.8" fill="none" stroke="{c}" stroke-width="1.7"/>'
                '<path d="M15.6 5.2l3.7-1 3.8 14.6-3.7 1z" fill="none" stroke="{c}" stroke-width="1.7" '
                'stroke-linejoin="round"/>'),
}


def icon(key: str, color: str, size: int = 28) -> QIcon:
    """The workspace's line icon in ``color`` (QIcon(), no icon, when QtSvg is not installed)."""
    try:
        from PySide6.QtSvg import QSvgRenderer
    except ImportError:          # a minimal Qt without the SVG module: text-only buttons
        return QIcon()
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">' + _ICONS[key].format(c=color)
           + "</svg>")
    out = QIcon()
    for scale in (1, 2):
        pm = QPixmap(size * scale, size * scale)
        pm.fill(Qt.GlobalColor.transparent)
        pm.setDevicePixelRatio(scale)
        p = QPainter(pm)
        QSvgRenderer(QByteArray(svg.encode())).render(p)
        p.end()
        out.addPixmap(pm)
    return out


class WorkspaceManager(QObject):
    """Switches ``win`` between the workspaces and keeps each one's arrangement (module docstring)."""

    changed = Signal(str)

    def __init__(self, win: QMainWindow):
        super().__init__(win)
        self.win = win
        self.current = ""
        self.states: dict[str, QByteArray] = {}
        #: the current workspace shows its default arrangement (not one restored from ``states``)
        self.showing_default = False
        self.bar = QToolBar("Workspaces", win)
        self.bar.setObjectName("activity_bar")
        self.bar.setOrientation(Qt.Orientation.Vertical)
        self.bar.setMovable(False)
        self.bar.setFloatable(False)
        self.bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.bar.setIconSize(QSize(28, 28))
        self.bar.toggleViewAction().setEnabled(False)       # the bar is how one gets around: always there
        win.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.bar)
        self.group = QActionGroup(self)
        self.group.setExclusive(True)
        self.actions: dict[str, QAction] = {}
        color = win.palette().windowText().color().name()
        for ws in WORKSPACES:
            a = QAction(icon(ws.key, color), ws.label, win)
            a.setCheckable(True)
            a.setShortcut(QKeySequence(ws.shortcut))
            a.setToolTip(f"{ws.tip} ({a.shortcut().toString(QKeySequence.SequenceFormat.NativeText)})")
            a.setStatusTip(ws.tip)
            a.triggered.connect(lambda _checked=False, k=ws.key: self.switch(k))
            self.group.addAction(a)
            self.bar.addAction(a)
            self.actions[ws.key] = a
        for ws in WORKSPACES:     # a button per workspace, all as wide as the widest label
            btn = self.bar.widgetForAction(self.actions[ws.key])
            if btn is not None:
                btn.setObjectName(f"workspace_{ws.key}")
                btn.setMinimumWidth(76)

    # ------------------------------------------------------------------ docks
    def dock(self, name: str) -> QDockWidget | None:
        d = self.win.findChild(QDockWidget, name)
        return d if isinstance(d, QDockWidget) and d.parent() is self.win else None

    def visible_docks(self) -> set[str]:
        """Object names of the managed docks that are shown now (shown = not hidden, even if behind a tab)."""
        return {n for n in MANAGED if (d := self.dock(n)) is not None and not d.isHidden()}

    def apply_default(self, key: str) -> None:
        """Arrange the docks as workspace ``key``'s default: its docks in their areas, a tab group per area with the
        first in front, every other managed dock hidden."""
        ws = BY_KEY[key]
        # Take every managed dock out of the layout first: re-tabifying docks that are still in other tab groups
        # leaves those groups' tab bars behind (empty strips over the view)
        for name in MANAGED:
            d = self.dock(name)
            if d is not None:
                d.setFloating(False)
                self.win.removeDockWidget(d)
        for attr, area in _AREAS:
            docks = [d for n in getattr(ws, attr) if (d := self.dock(n)) is not None]
            if not docks:
                continue
            self.win.addDockWidget(area, docks[0])
            docks[0].show()
            for d in docks[1:]:
                self.win.tabifyDockWidget(docks[0], d)
                d.show()
            docks[0].raise_()
        # ... and put the others back hidden, each in its home area: saveState only records docks that are in the
        # layout, and restoreState leaves an unrecorded dock as it is, so a dock missing from one workspace's state
        # would stay visible after a switch
        for name in MANAGED:
            d = self.dock(name)
            if d is not None and name not in ws.docks:
                self.win.addDockWidget(_HOME.get(name, Qt.DockWidgetArea.RightDockWidgetArea), d)
                d.hide()
        sizes = getattr(self.win, "default_dock_sizes", None)
        if callable(sizes):
            sizes(key)

    # ------------------------------------------------------------------ switching
    def switch(self, key: str) -> None:
        """Show workspace ``key``: the arrangement it had when it was left, else its default."""
        if key not in BY_KEY:
            raise KeyError(f"no workspace {key!r}; one of {', '.join(BY_KEY)}")
        if key == self.current:
            self.actions[key].setChecked(True)
            return
        if self.current:
            self.states[self.current] = self.win.saveState(STATE_VERSION)
        self.current = key
        state = self.states.get(key)
        self.showing_default = state is None or not self.win.restoreState(state, STATE_VERSION)
        if self.showing_default:
            self.apply_default(key)
        self.actions[key].setChecked(True)
        self.changed.emit(key)

    def reset(self) -> None:
        """View > Reset layout: the current workspace's default arrangement (its saved one is dropped)."""
        if not self.current:
            return
        self.states.pop(self.current, None)
        self.apply_default(self.current)
        self.showing_default = True

    def reset_all(self) -> None:
        self.states.clear()
        if self.current:
            self.apply_default(self.current)
            self.showing_default = True

    # ------------------------------------------------------------------ persistence
    def save(self, qs) -> None:
        """Store every workspace's arrangement (the current one as it is now) and the current workspace."""
        if self.current:
            self.states[self.current] = self.win.saveState(STATE_VERSION)
            qs.setValue("workspace/current", self.current)
        qs.setValue("workspace/version", STATE_VERSION)
        for ws in WORKSPACES:
            if ws.key in self.states:
                qs.setValue(f"workspace/{ws.key}", self.states[ws.key])
            else:
                qs.remove(f"workspace/{ws.key}")

    def load(self, qs) -> str:
        """Read the stored arrangements and switch to the stored workspace (``DEFAULT`` the first time). Window
        states saved before workspaces existed, or by an older ``STATE_VERSION``, are not used: the defaults are."""
        try:
            version = int(qs.value("workspace/version") or 0)
        except (TypeError, ValueError):
            version = 0
        if version == STATE_VERSION:
            for ws in WORKSPACES:
                value = qs.value(f"workspace/{ws.key}")
                if isinstance(value, (QByteArray, bytes)) and len(value):
                    self.states[ws.key] = QByteArray(value)
        key = str(qs.value("workspace/current") or DEFAULT)
        self.switch(key if key in BY_KEY else DEFAULT)
        return self.current


def placeholder(text: str) -> QWidget:
    """A plain explanatory page for a workspace whose tools are still to come."""
    from PySide6.QtWidgets import QLabel, QVBoxLayout

    w = QWidget()
    lay = QVBoxLayout(w)
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setOpenExternalLinks(True)
    lay.addWidget(label)
    lay.addStretch(1)
    return w
