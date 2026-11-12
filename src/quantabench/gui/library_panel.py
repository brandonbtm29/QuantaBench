"""Library workspace (UI-1): the run folders under the output folder, newest first.

Each row is a run folder found by its job table (``jobs.sqlite``), read with ``jobs.open_run`` (read-only: a folder
that synced from another machine is not changed): when it last changed, its molecules (built, failed), method, phase,
and whether a process runs it now. Double-click (or Open) lists its molecules in the Molecules dock, as File > Open
run folder does; Resume continues its unfinished molecules with the settings stored in it.

The scan runs off the GUI thread (``find_run_folders``): it walks at most ``DEPTH`` levels below the output folder
and does not descend into a run folder (its subfolders are molecules). It runs when the Library is first shown and
on Refresh. The project database and its search (UI-2's project document, a substructure search) come later.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

DEPTH = 3
LIMIT = 2000           # folders listed at most (the newest)
COLUMNS = ("Run folder", "Changed", "Molecules", "Built", "Failed", "Method", "Phase", "State")


@dataclass
class RunFolderInfo:
    path: Path
    rel: str
    modified: float
    total: int = 0
    done: int = 0
    failed: int = 0
    method: str = ""
    phase: str = ""
    state: str = ""
    error: str = ""


def _info(path: Path, root: Path) -> RunFolderInfo:
    from ..chem import jobs

    mtime = 0.0
    for name in (jobs.DB_NAME, jobs.DB_NAME + "-wal"):
        # a run writes its latest molecules to the -wal first; an empty -wal is what a reader (this scan's
        # open_run) leaves behind, not a change
        try:
            st = (path / name).stat()
        except OSError:
            continue
        if name == jobs.DB_NAME or st.st_size:
            mtime = max(mtime, st.st_mtime)
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    info = RunFolderInfo(path=path, rel=rel, modified=mtime)
    try:
        st = jobs.open_run(path)
    except Exception as exc:  # noqa: BLE001 - a damaged or foreign table: listed with its error
        info.error = f"{type(exc).__name__}: {exc}"[:200]
        info.state = "unreadable"
        return info
    cfg = st.config or {}
    info.total, info.done, info.failed = st.total, st.counts.get("ok", 0), st.counts.get("failed", 0)
    info.method = str(cfg.get("method") or "")
    info.phase = str(cfg.get("solvent") or "") or "gas"
    info.state = "running" if st.active else ("finished" if not st.remaining else f"{st.remaining} to do")
    return info


def find_run_folders(root: str | Path, depth: int = DEPTH, limit: int = LIMIT, should_stop=None) -> list[RunFolderInfo]:
    """The run folders (those with a job table) at most ``depth`` levels below ``root``, newest first. Plain
    Python and read-only: safe on a worker thread."""
    from ..chem import jobs

    root = Path(root).expanduser()
    found: list[Path] = []
    stack = [(root, 0)]
    while stack:
        if should_stop is not None and should_stop():
            break
        folder, level = stack.pop()
        try:
            if (folder / jobs.DB_NAME).is_file():
                found.append(folder)
                continue                         # its subfolders are molecules
            if level >= depth:
                continue
            with os.scandir(folder) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith("."):
                        stack.append((Path(e.path), level + 1))
        except OSError:
            continue
    infos = [_info(p, root) for p in found]
    infos.sort(key=lambda i: i.modified, reverse=True)
    return infos[:limit]


def _age(t: float) -> str:
    if not t:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


class _Item(QTableWidgetItem):
    """A cell that sorts by its number (or time), not its text."""

    def __init__(self, text: str, key=None):
        super().__init__(text)
        self.key = key if key is not None else text

    def __lt__(self, other):
        try:
            return self.key < other.key
        except TypeError:
            return str(self.key) < str(other.key)


class LibraryPanel(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.infos: list[RunFolderInfo] = []
        self._scanning = False
        self._scanned = False
        lay = QVBoxLayout(self)
        head = QLabel()
        head.setWordWrap(True)
        head.setStyleSheet("color: gray;")
        self.head = head
        lay.addWidget(head)
        row = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter run folders")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        row.addWidget(self.filter, 1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self.refresh_btn)
        lay.addLayout(row)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("library_table")
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(COLUMNS)):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSortingEnabled(True)
        self.table.doubleClicked.connect(lambda _i: self.open_selected())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        lay.addWidget(self.table, 1)
        btns = QHBoxLayout()
        self.open_btn = QPushButton("Open")
        self.open_btn.setToolTip("List this run folder's molecules (File > Open run folder)")
        self.open_btn.clicked.connect(self.open_selected)
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setToolTip("Continue its unfinished molecules with the settings stored in it")
        self.resume_btn.clicked.connect(self.resume_selected)
        self.browse_btn = QPushButton("Other folder...")
        self.browse_btn.clicked.connect(lambda: self.win.open_run_folder())
        for b in (self.open_btn, self.resume_btn, self.browse_btn):
            btns.addWidget(b)
        lay.addLayout(btns)
        self.note = QLabel()
        self.note.setStyleSheet("color: gray;")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self._update_head()
        self._update_buttons()

    def root(self) -> Path:
        return Path(self.win.s.output_path())

    def _update_head(self) -> None:
        self.head.setText(f"Run folders under <b>{self.root()}</b> (Settings > Output folder), newest first. "
                          "Double-click one to list its molecules.")

    # ------------------------------------------------------------------ scanning
    def ensure_scanned(self) -> None:
        if not self._scanned and not self._scanning:
            self.refresh()

    def refresh(self) -> None:
        if self._scanning:
            return
        from . import workers

        self._scanning = True
        self._scanned = True
        self._update_head()
        self.refresh_btn.setEnabled(False)
        self.note.setText("Looking for run folders...")
        w = workers.FunctionWorker(find_run_folders, self.root())
        workers.start(w, self, on_done=self._on_scanned, on_failed=self._on_failed)

    def _on_failed(self, msg: str) -> None:
        self._scanning = False
        self.refresh_btn.setEnabled(True)
        self.note.setText(f"Could not list the run folders: {msg}")

    def _on_scanned(self, infos) -> None:
        self._scanning = False
        self.refresh_btn.setEnabled(True)
        self.infos = list(infos or [])
        self._fill()
        n = len(self.infos)
        running = sum(1 for i in self.infos if i.state == "running")
        self.note.setText(f"{n} run folder{'s' if n != 1 else ''}" + (f", {running} running" if running else "")
                          + (f" (the newest {LIMIT} shown)" if n >= LIMIT else ""))

    def _fill(self) -> None:
        t = self.table
        t.setSortingEnabled(False)
        t.setRowCount(len(self.infos))
        for r, i in enumerate(self.infos):
            cells = [_Item(i.rel, i.rel.lower()), _Item(_age(i.modified), i.modified),
                     _Item(f"{i.total:,}", i.total), _Item(f"{i.done:,}", i.done), _Item(f"{i.failed:,}", i.failed),
                     _Item(i.method), _Item(i.phase), _Item(i.state)]
            cells[0].setData(Qt.ItemDataRole.UserRole, str(i.path))
            cells[0].setToolTip(str(i.path) + (f"\n{i.error}" if i.error else ""))
            for c, item in enumerate(cells):
                if c in (2, 3, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, item)
        t.setSortingEnabled(True)
        self._apply_filter(self.filter.text())
        self._update_buttons()

    def _apply_filter(self, text: str) -> None:
        text = (text or "").strip().lower()
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            hit = not text or any(text in (self.table.item(r, c).text().lower())
                                  for c in (0, 5, 6, 7) if self.table.item(r, c) is not None)
            self.table.setRowHidden(r, not hit if item is not None else True)

    # ------------------------------------------------------------------ actions
    def selected_folder(self) -> Path | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return Path(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

    def _info_of(self, folder: Path) -> RunFolderInfo | None:
        return next((i for i in self.infos if i.path == folder), None)

    def _update_buttons(self) -> None:
        folder = self.selected_folder()
        info = self._info_of(folder) if folder is not None else None
        self.open_btn.setEnabled(info is not None and not info.error)
        self.resume_btn.setEnabled(info is not None and not info.error and info.state not in ("running", "finished"))

    def open_selected(self) -> None:
        folder = self.selected_folder()
        if folder is not None:
            self.win.open_run_folder(folder=folder)

    def resume_selected(self) -> None:
        folder = self.selected_folder()
        if folder is None:
            return
        if self.win.run_folder != folder:     # resume works on the opened folder's rows
            self.win.open_run_folder(folder=folder)
            self.note.setText(f"Opened {folder.name}; press Resume again once its molecules are listed.")
            return
        self.win.resume_run(folder=folder)
