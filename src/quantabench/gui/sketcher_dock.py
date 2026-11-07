"""The Sketch dock (UI-6): draw a molecule in Schrödinger's Sketcher and hand it to the molecule list.

    install(win)    adds the dock (hidden until opened: Build > Sketch a molecule..., View > Sketch) to a MainWindow

Buttons: **New** clears the canvas; **Edit selected** loads the selected molecule's input (as a MOL V3000 block with
2D coordinates); **Add to list** and **Replace selected** export the drawing as MOL V3000 and hand it to
``chem.sketch.input_from_molblock``, which reports structure problems before anything is added; the molecule then
goes the way of a SMILES typed into the Molecules box (``win.add_inputs``), and the 3D view previews it. **Insert
fragment** adds a PFAS piece (CF3, a (CF2)n chain, acid head groups, ...) next to the drawing, ready to be joined
with a bond.

Nothing of Qt WebEngine is loaded until the dock is first shown. Without WebEngine (``pip install
"quantabench[editor]"``) the dock says what to install; without the editor's files (``sketcher_assets``) it offers
to download them (17.5 MB, pinned by sha256) or to install a zip downloaded by hand.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import sketcher_assets
from ..chem import sketch
from . import sketcher_web

TITLE = "Sketch"


class SketcherPanel(QWidget):
    """The dock's widget: a button row, then either a message page (what to install, download progress, errors)
    or the editor (``gui.sketcher_view.SketcherView``)."""

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.view = None                     # SketcherView, created when the dock is first shown
        self.ready = False
        self.empty = True
        self.activated = False
        self._cancel = threading.Event()
        self._progress: tuple[int, int] | None = None
        self._installing = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.b_new = QPushButton("New")
        self.b_new.setToolTip("Clear the canvas")
        self.b_edit = QPushButton("Edit selected")
        self.b_edit.setToolTip("Load the selected molecule's structure into the sketcher")
        self.b_add = QPushButton("Add to list")
        self.b_add.setToolTip("Add the drawing to the molecule list (checked with RDKit first)")
        self.b_replace = QPushButton("Replace selected")
        self.b_replace.setToolTip("Replace the selected molecule's structure with the drawing (same ID and name)")
        self.b_fragment = QToolButton()
        self.b_fragment.setText("Insert fragment")
        self.b_fragment.setToolTip("Add a PFAS piece next to the drawing; draw a bond to join it")
        self.b_fragment.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.fragment_menu = QMenu(self.b_fragment)
        for label, smiles in sketch.FRAGMENTS:
            act = self.fragment_menu.addAction(label)
            act.triggered.connect(lambda _checked=False, s=smiles: self.insert_fragment(s))
        self.b_fragment.setMenu(self.fragment_menu)
        self.b_new.clicked.connect(self.new)
        self.b_edit.clicked.connect(self.edit_selected)
        self.b_add.clicked.connect(self.add_to_list)
        self.b_replace.clicked.connect(self.replace_selected)
        for b in (self.b_new, self.b_edit, self.b_add, self.b_replace, self.b_fragment):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.stack = QStackedWidget()
        self.message_page = QWidget()
        mlay = QVBoxLayout(self.message_page)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                             | Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.message.setOpenExternalLinks(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        mlay.addWidget(self.message)
        brow = QHBoxLayout()
        self.b_download = QPushButton("Download and open")
        self.b_download.setToolTip(f"Download {sketcher_assets.URL} ({sketcher_assets.SIZE / 1e6:.1f} MB), check its "
                                   "sha256 and unpack it into the programs folder")
        self.b_download.clicked.connect(lambda _checked=False: self.start_install())
        self.b_zip = QPushButton("Install from a zip file...")
        self.b_zip.setToolTip(f"Use {sketcher_assets.ZIP_NAME} downloaded by hand")
        self.b_zip.clicked.connect(self.install_from_zip)
        self.b_retry = QPushButton("Reload")
        self.b_retry.clicked.connect(self.reload)
        for b in (self.b_download, self.b_zip, self.b_retry):
            brow.addWidget(b)
        brow.addStretch(1)
        mlay.addLayout(brow)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        mlay.addWidget(self.progress)
        mlay.addStretch(1)
        self.stack.addWidget(self.message_page)
        lay.addWidget(self.stack, 1)
        self.status = QLabel("")
        self.status.setStyleSheet("color: gray;")
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.status)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(200)
        self._progress_timer.timeout.connect(self._show_progress)
        win.mol_list.currentIdChanged.connect(self._update_buttons)
        win.model.modelReset.connect(self._update_buttons)
        self._show_message("Open this dock to start the sketcher.", buttons=())
        self._update_buttons()

    # ------------------------------------------------------------------ state
    def _selected_row(self):
        ident = self.win.mol_list.current_id()
        return self.win.model.row(ident) if ident else None

    def _update_buttons(self, *_args) -> None:
        on = self.ready and self.view is not None
        row = self._selected_row()
        self.b_new.setEnabled(on)
        self.b_fragment.setEnabled(on)
        self.b_edit.setEnabled(on and row is not None)
        self.b_add.setEnabled(on and not self.empty)
        self.b_replace.setEnabled(on and not self.empty and row is not None and row.id not in self.win.running)

    def _show_message(self, html: str, buttons=("download", "zip")) -> None:
        self.message.setText(html)
        self.b_download.setVisible("download" in buttons)
        self.b_zip.setVisible("zip" in buttons)
        self.b_retry.setVisible("retry" in buttons)
        self.stack.setCurrentWidget(self.message_page)
        self.ready = False
        self._update_buttons()

    def _say(self, text: str) -> None:
        self.status.setText(text)

    # ------------------------------------------------------------------ start-up
    def activate(self) -> None:
        """First time the dock is shown: find the editor and start it (or say what is missing)."""
        if self.activated:
            return
        self.activated = True
        self.refresh()

    def refresh(self) -> None:
        problem = sketcher_web.problem()
        if problem:
            self._show_message(_html(problem + "\n\nEverything else in QuantaBench works without it."), buttons=())
            self._say("Sketcher unavailable")
            return
        loc = sketcher_assets.locate(self.win.s)
        if loc.ok:
            self._start_view(loc.path)
            return
        if loc.source in ("environment", "settings"):
            self._show_message(_html(f"The sketcher's folder cannot be used: {loc.problem}."), buttons=("retry",))
            return
        placed = sketcher_assets.programs_dir() / sketcher_assets.ZIP_NAME
        if placed.is_file() and not self._installing:
            self.start_install()                 # a zip put into the programs folder by hand
            return
        if not self._installing:
            self._show_message(_html(loc.problem + " Press Download and open (the zip is checked against its "
                                     "pinned sha256).\n\nOffline? " + sketcher_assets.manual_instructions()))

    def _start_view(self, root: Path) -> None:
        if self.view is None:
            from .sketcher_view import SketcherView

            self.view = SketcherView(self)
            self.view.ready.connect(self._on_ready)
            self.view.failed.connect(self._on_failed)
            self.view.changed.connect(self._on_changed)
            self.view.status.connect(self._say)
            self.stack.addWidget(self.view)
        self.ready = False
        self.stack.setCurrentWidget(self.view)
        self._update_buttons()
        self.view.start(root)

    def reload(self) -> None:
        self.refresh()

    def _on_ready(self, seconds: float) -> None:
        self.ready = True
        self.empty = True
        self._say(f"Sketcher ready ({seconds:.1f} s). Draw, then Add to list or Replace selected.")
        self._on_changed()

    def _on_failed(self, message: str) -> None:
        self._show_message(_html(message), buttons=("retry",))
        self._say("Sketcher failed to start")
        self.win.log(f"Sketch: {message}")

    def _on_changed(self) -> None:
        if self.view is None or not self.view.is_ready:
            return

        def got(ok, value):
            self.empty = bool(value) if ok else True
            self._update_buttons()
        self.view.is_empty(got)

    # ------------------------------------------------------------------ download / install
    def start_install(self, zip_path: str | None = None) -> None:
        if self._installing:
            return
        from . import workers

        self._installing = True
        self._cancel.clear()
        self._progress = None
        self._show_message(_html(f"Installing Schrödinger's Sketcher {sketcher_assets.RELEASE} into "
                                 f"{sketcher_assets.release_dir()}..."), buttons=())
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._progress_timer.start()

        def progress(done, total):           # worker thread: only stores the numbers
            self._progress = (done, total)

        worker = workers.FunctionWorker(sketcher_assets.install, zip_path=zip_path, progress=progress,
                                        cancel=self._cancel)
        workers.start(worker, self.win, on_done=self._installed, on_failed=self._install_failed)

    def _show_progress(self) -> None:
        if self._progress is None:
            return
        done, total = self._progress
        if total > 0:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(1000 * min(done, total) / total))
            self.progress.setFormat(f"{done / 1e6:.1f} of {total / 1e6:.1f} MB")

    def _install_done(self) -> None:
        self._installing = False
        self._progress_timer.stop()
        self.progress.setVisible(False)

    def _installed(self, folder) -> None:
        self._install_done()
        self.win.log(f"Sketch: Schrödinger's Sketcher {sketcher_assets.RELEASE} installed in {folder}")
        self.refresh()

    def _install_failed(self, message: str) -> None:
        self._install_done()
        self._show_message(_html(message.removeprefix("SketcherAssetError: ")))
        self._say("Download failed")

    def install_from_zip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Install Schrödinger's Sketcher", "",
                                              f"Sketcher release ({sketcher_assets.ZIP_NAME} *.zip);;All files (*)")
        if path:
            self.start_install(zip_path=path)

    def shutdown(self) -> None:
        """Before the window goes: stop a download and delete the web view (its page must go before Qt's
        default profile, or Qt WebEngine warns at exit)."""
        self._cancel.set()
        if self.view is not None:
            import shiboken6

            view, self.view = self.view, None
            if shiboken6.isValid(view):
                view.stop()
                shiboken6.delete(view)           # now, with its page: deleteLater may not run after the loop
        self.ready = False

    # ------------------------------------------------------------------ buttons
    def _js_failed(self, what: str, message) -> None:
        QMessageBox.warning(self, what, f"The sketcher reported an error: {message}")

    def new(self) -> None:
        if self.view is not None and self.ready:
            self.view.clear()

    def edit_selected(self) -> None:
        if self.view is None or not self.ready:
            return
        inp = self.win.selected_input()
        if inp is None:
            return
        try:
            block = sketch.molblock_for_input(inp)
        except sketch.SketchError as exc:
            QMessageBox.warning(self, "Edit selected", f"{inp.id}: {exc}")
            return
        if not self.empty and QMessageBox.question(
                self, "Edit selected", f"Replace the drawing with {inp.id}?") != QMessageBox.StandardButton.Yes:
            return
        self.view.import_text(block, clear=True, callback=lambda ok, v: ok or self._js_failed("Edit selected", v))
        self._say(f"Editing {inp.id}: Replace selected puts the changes back, Add to list adds a new molecule.")

    def insert_fragment(self, smiles: str | None) -> None:
        if self.view is None or not self.ready:
            return
        if smiles is None:
            n, ok = QInputDialog.getInt(self, "(CF2)n chain", "Number of CF2 units:", 4, 1, sketch.MAX_CHAIN)
            if not ok:
                return
            smiles = sketch.cf2_chain(n)
        self.view.import_text(smiles, callback=lambda ok, v: ok or self._js_failed("Insert fragment", v))
        self._say("Fragment added next to the drawing: draw a bond from it to the atom it belongs on.")

    def _export(self, what: str, then) -> None:
        if self.view is None or not self.ready:
            return

        def got(ok, value):
            if not ok:
                self._js_failed(what, value)
                return
            then(str(value or ""))
        self.view.export_text("MDL_MOLV3000", got)

    def add_to_list(self) -> None:
        self._export("Add to list", self.add_block)

    def replace_selected(self) -> None:
        self._export("Replace selected", self.replace_block)

    # ------------------------------------------------------------------ hand-off (no WebEngine needed)
    def _log_notes(self, ident: str, notes: list[str]) -> None:
        for note in notes:
            self.win.log(f"Sketch {ident}: {note}")

    def add_block(self, block: str) -> bool:
        """Add the drawing (a MOL block) to the molecule list, like a SMILES typed into the Molecules box."""
        try:
            res = sketch.input_from_molblock(block)
        except sketch.SketchError as exc:
            QMessageBox.warning(self, "Add to list", str(exc))
            return False
        self.win.add_inputs([res.input])          # may rename the id to keep ids unique
        self._log_notes(res.input.id, res.notes)
        self._say(f"Added {res.input.id} ({res.formula}).")
        return True

    def replace_block(self, block: str) -> bool:
        """Give the selected molecule the drawing's structure, keeping its id, name and place in the list. Its
        results leave the list (their files stay on disk); it is built again at the next Run."""
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "Replace selected", "Select a molecule in the list first.")
            return False
        if row.id in self.win.running:
            QMessageBox.information(self, "Replace selected", f"{row.id} is being built; replace it afterwards.")
            return False
        try:
            res = sketch.input_from_molblock(block, id=row.id, name=row.name)
        except sketch.SketchError as exc:
            QMessageBox.warning(self, "Replace selected", str(exc))
            return False
        if (row.status != "pending" or row.folder) and QMessageBox.question(
                self, "Replace selected", f"Replace {row.id}'s structure with the drawing? Its results are removed "
                "from the list (the files on disk stay) and it is built again at the next Run.") \
                != QMessageBox.StandardButton.Yes:
            return False
        from .molecule_list import MoleculeRow

        fresh = MoleculeRow(id=row.id, name=row.name, smiles=res.input.smiles, source=res.input.source,
                            input=res.input)
        self.win.model.update(row.id, **{f.name: getattr(fresh, f.name) for f in dataclasses.fields(MoleculeRow)
                                         if f.name not in ("id", "seq")})
        self.win.results.discard(row.id)
        self.win.selection.clear()
        self.win.mol_list.currentIdChanged.emit(row.id)     # the views and panels show the new structure
        self.win.log(f"Sketch: replaced {row.id}'s structure with {res.input.smiles}")
        self._log_notes(row.id, res.notes)
        self._say(f"Replaced {row.id} ({res.formula}).")
        return True


def _html(text: str) -> str:
    """Plain text (with URLs and line breaks) as label HTML."""
    import html
    import re

    out = html.escape(text).replace("\n", "<br>")
    return re.sub(r"(https?://[^\s<]+)", r'<a href="\1">\1</a>', out)


class _Watcher(QObject):
    """Starts the panel the first time its dock is shown; deletes the web view once the window has closed."""

    def __init__(self, panel: SketcherPanel, dock: QDockWidget, win):
        super().__init__(panel)
        self.panel = panel
        self.win = win
        dock.visibilityChanged.connect(self._on_visible)
        win.installEventFilter(self)
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(panel.shutdown)

    def _on_visible(self, visible: bool) -> None:
        if visible:
            self.panel.activate()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt's name
        if obj is self.win and event.type() == QEvent.Type.Close:
            QTimer.singleShot(0, self._after_close)      # closeEvent may still refuse (a build is running)
        return False

    def _after_close(self) -> None:
        if getattr(self.win, "closed", False):
            self.panel.shutdown()


def _menu(win, title: str):
    for action in win.menuBar().actions():
        if action.menu() is not None and action.text().replace("&", "") == title:
            return action.menu()
    return None


def open_dock(win) -> None:
    """Show the Sketch dock. Docked narrower than the editor needs, it opens as its own window instead."""
    dock = win.sketch_dock
    if not dock.isVisible() and not dock.isFloating() and dock.width() < 560:
        dock.setFloating(True)
        dock.resize(980, 680)
        center = win.geometry().center()
        dock.move(center.x() - 490, center.y() - 340)
    dock.show()
    dock.raise_()
    win.sketch_panel.activate()


def install(win) -> None:
    panel = SketcherPanel(win)
    dock = QDockWidget(TITLE, win)
    dock.setObjectName("sketch")
    dock.setWidget(panel)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    dock.hide()
    win.sketch_dock = dock
    win.sketch_panel = panel
    panel._watcher = _Watcher(panel, dock, win)
    toggle = dock.toggleViewAction()
    toggle.setText(TITLE)
    toggle.setToolTip("Schrödinger's 2D Sketcher: draw a molecule and add it to the list")
    view = _menu(win, "View")
    if view is not None:
        view.addAction(toggle)
    action = QAction("Sketch a molecule...", win)
    action.setToolTip("Open the 2D sketcher (Schrödinger's Sketcher)")
    action.triggered.connect(lambda _checked=False: open_dock(win))
    build = _menu(win, "Build")
    if build is not None:
        build.addAction(action)
    win.a_sketch = action
