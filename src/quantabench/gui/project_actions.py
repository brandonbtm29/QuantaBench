"""Project documents in the window (UI-2): File > New / Open / Open recent / Save / Save as / Close project.

A project (``quantabench.project``) is a ``<name>.qbproj`` folder with the molecule list, the results, the build
settings, the Analyze table's state, the workspace and the selected molecule. This module turns the window's
state into ``project.ProjectData`` and back:

- **Save** writes the molecule list (each molecule's light row and, for one added here, its input), the build
  settings of the Pipeline dock, the table's state (columns, formula columns, joined files, plot) and the workspace.
  Results that exist only in memory (an imported output, a build with file writing off) are written to the
  project's ``results/`` folder first, so the project holds everything. **Save as** writes a new project folder;
  run folders stay where they are (inside the old project they are referenced by their absolute paths).
- **Open** restores the list at once (light rows only: a 14,000-molecule project opens in about a second), then,
  off the GUI thread, reads each run folder's job table for statuses that changed since the save (a resume from
  the command line) and the summary and descriptor tables for the Analyze table. Full results are read when a
  molecule is selected, as for a run folder. **Build > Resume run folder** continues the project's unfinished
  molecules (its newest run folder with molecules left).
- **Builds in a project** write into its ``runs/`` folder, every build through a run folder (one molecule too), so
  the project can always be resumed and its tables read.
- **Unsaved changes** (molecules added, removed or built; the table's columns, formulas, joined files and plot)
  mark the window title (``[*]``). New, Open and Close project ask whether to save them; closing the window asks
  when a project is open. Without a project, closing the window does not ask (as before projects existed): the
  results are in their run folders.
- **Recent projects** are kept in the window settings (``RECENT_KEY``).
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from .. import __version__
from .. import project as proj
from ..analysis.sheet import TableState
from .molecule_list import FINISHED, MoleculeRow

RECENT_KEY = "projects/recent"
MAX_RECENT = 8


def row_to_dict(r: MoleculeRow) -> dict:
    from ..chem import jobs

    d = {k: getattr(r, k) for k in proj.ROW_FIELDS}
    if r.input is not None:
        d["input"] = jobs.input_to_json(r.input)
    return d


def row_from_dict(d: dict) -> MoleculeRow:
    from ..chem import jobs

    inp = None
    if d.get("input"):
        try:
            inp = jobs.input_from_json(d["input"])
        except (ValueError, TypeError, KeyError):
            inp = None
    status = d.get("status") if d.get("status") in FINISHED else "pending"
    n_conf = d.get("n_conformers")
    energy = d.get("energy_ev")
    return MoleculeRow(
        id=str(d["id"]), name=str(d.get("name") or ""), smiles=str(d.get("smiles") or ""),
        source=str(d.get("source") or ""), status=status, method=str(d.get("method") or ""),
        n_conformers=int(n_conf) if isinstance(n_conf, (int, float)) and status == "ok" else None,
        energy_ev=float(energy) if isinstance(energy, (int, float)) else None, formula=str(d.get("formula") or ""),
        error=str(d.get("error") or ""), folder=str(d.get("folder") or ""), run_dir=str(d.get("run_dir") or ""),
        seq=int(d.get("seq") or 0), input=inp, fields=d.get("fields") if isinstance(d.get("fields"), dict) else None)


def refresh_from_runs(run_dirs: list[str]) -> dict[str, dict[str, dict]]:
    """For each run folder (worker thread): {run_dir: {id: the row fields its job table has now}}; a folder that
    cannot be read is left out."""
    from .molecule_list import read_run_folder

    out: dict[str, dict[str, dict]] = {}
    for run_dir in run_dirs:
        try:
            rows, _status = read_run_folder(run_dir)
        except (OSError, ValueError, FileNotFoundError):
            continue
        out[run_dir] = {r.id: {k: getattr(r, k) for k in ("status", "method", "n_conformers", "energy_ev",
                                                          "formula", "error", "folder")} for r in rows}
    return out


class ProjectController(QObject):
    """The window's project: its folder (None: untitled), whether it has unsaved changes, the File menu."""

    #: the project was opened, saved, closed or its unsaved state changed
    changed = Signal()

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.path: Path | None = None
        self.data: proj.ProjectData | None = None
        self._dirty = False
        self._restoring = False
        for sig in (win.model.rowsInserted, win.model.rowsRemoved, win.model.modelReset, win.model.dataChanged):
            sig.connect(self._on_list_changed)
        table = getattr(win, "table_panel", None)
        if table is not None and hasattr(table, "stateSaved"):
            table.stateSaved.connect(self.mark_dirty)
        win.runFolderOpened.connect(lambda _f: self.update_title())

    # ------------------------------------------------------------------ state
    @property
    def is_open(self) -> bool:
        return self.path is not None

    def is_dirty(self) -> bool:
        return self._dirty

    def name(self) -> str:
        return (self.data.name if self.data and self.data.name else None) or (self.path.stem if self.path else "")

    def runs_dir(self) -> Path | None:
        """Where builds of the open project write (None: no project)."""
        if self.path is None:
            return None
        d = proj.runs_dir(self.path)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _on_list_changed(self, *_args) -> None:
        self.mark_dirty()

    def mark_dirty(self, *_args) -> None:
        if self._restoring or self.win.__dict__.get("_closing"):
            return
        if not self._dirty and (self.is_open or self.win.model.rowCount()):
            self._dirty = True
            self.update_title()
            self.changed.emit()

    def _set_clean(self) -> None:
        self._dirty = False
        self.update_title()
        self.changed.emit()

    def update_title(self) -> None:
        w = self.win
        if self.is_open:
            w.setWindowTitle(f"QuantaBench {__version__} - {self.name()}[*]")
            w.setWindowModified(self._dirty)
        else:
            w.setWindowModified(False)
            if "[*]" in w.windowTitle():
                w.setWindowTitle(f"QuantaBench {__version__}")

    # ------------------------------------------------------------------ asking
    def maybe_save(self, what: str = "close", ask_untitled: bool = True) -> bool:
        """True when it is fine to go on (saved, discarded, or nothing to save); False to stay (Cancel)."""
        if not self._dirty:
            return True
        if not self.is_open and (not ask_untitled or not self.win.model.rowCount()):
            return True
        name = self.name() if self.is_open else "the untitled project"
        n = self.win.model.rowCount()
        box = QMessageBox(QMessageBox.Icon.Question, "Unsaved changes",
                          f"Save the changes to {name} ({n:,} molecule{'s' if n != 1 else ''}) before you {what}?",
                          QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                          | QMessageBox.StandardButton.Cancel, self.win)
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        answer = box.exec()
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def _idle(self, title: str) -> bool:
        if self.win.worker is not None:
            QMessageBox.information(self.win, title, "Wait for the build to finish (or cancel it) first.")
            return False
        return True

    # ------------------------------------------------------------------ new, close
    def new_project(self, _checked=False) -> bool:
        if not self._idle("New project") or not self.maybe_save("start a new project"):
            return False
        self._reset()
        self.win.log("New project (untitled): File > Save project names its folder.")
        return True

    def close_project(self, _checked=False) -> bool:
        if not self._idle("Close project") or not self.maybe_save("close it"):
            return False
        self._reset()
        return True

    def _reset(self) -> None:
        self._restoring = True
        try:
            self.win.clear_all()
            table = getattr(self.win, "table_panel", None)
            if table is not None:
                table.restore(TableState())
                table.store.clear()
                table.model.refresh()
            self.win.last_run_dir = None
            self.path, self.data = None, None
        finally:
            self._restoring = False
        self._set_clean()

    # ------------------------------------------------------------------ save
    def collect(self) -> proj.ProjectData:
        """The window's state as project data (pinned results are written by ``save``, not here)."""
        w = self.win
        w._store_stage_values()
        table = getattr(w, "table_panel", None)
        ws = getattr(w, "workspaces", None)
        data = self.data or proj.ProjectData()
        return proj.ProjectData(
            name=data.name, molecules=[row_to_dict(r) for r in w.model.rows()], build=asdict(w.s.build),
            table=table.state().to_json() if table is not None else "", workspace=getattr(ws, "current", "") or "",
            current=w.mol_list.current_id() or "", created=data.created, extra=dict(data.extra))

    def _write_pinned(self, root: Path) -> int:
        """Write results that exist only in memory into the project's results/ folder; their rows point there."""
        from ..chem import export

        n = 0
        for ident, res in list(self.win.results.pinned.items()):
            row = self.win.model.row(ident)
            if row is None or row.folder or getattr(res, "status", "") != "ok":
                continue
            dest = proj.results_dir(root) / ident
            files = export.write_molecule(res, dest, export.DEFAULT_FORMATS)
            res.files = [str(p) for p in files]
            self.win.model.update(ident, folder=str(dest))
            n += 1
        return n

    def save(self, _checked=False) -> bool:
        if self.path is None:
            return self.save_as()
        return self._save_to(self.path)

    def save_as(self, _checked=False, path: str | Path | None = None) -> bool:
        if path is None:
            base = self.path.parent if self.path else self.win.s.output_path()
            name = self.name() or "project"
            path, _ = QFileDialog.getSaveFileName(self.win, "Save project as", str(Path(base) / f"{name}{proj.SUFFIX}"),
                                                  f"QuantaBench project (*{proj.SUFFIX})")
            if not path:
                return False
        root = proj.project_path(path)
        if root.exists() and any(root.iterdir()) and not proj.is_project(root) and root != self.path:
            QMessageBox.warning(self.win, "Save project as", f"{root} exists and is not a QuantaBench project: choose "
                                "another name.")
            return False
        if root != self.path:
            self.data = proj.ProjectData(name=root.stem)
        return self._save_to(root)

    def _save_to(self, root: Path) -> bool:
        self._restoring = True                 # writing results updates rows: not a change to save again
        try:
            root.mkdir(parents=True, exist_ok=True)
            written = self._write_pinned(root)
            data = self.collect()
            data.name = data.name or root.stem
            proj.save(root, data)
        except OSError as exc:
            QMessageBox.warning(self.win, "Save project", f"Could not save {root}: {exc}")
            return False
        finally:
            self._restoring = False
        self.path, self.data = root, proj.load(root)
        self.add_recent(root)
        self._set_clean()
        self.win.statusBar().showMessage(f"Saved {root}" + (f" ({written} result(s) written to results/)"
                                                             if written else ""), 8000)
        self.win.log(f"Saved project {root} ({len(data.molecules):,} molecules).")
        return True

    # ------------------------------------------------------------------ open
    def open_project(self, _checked=False, path: str | Path | None = None) -> bool:
        if not self._idle("Open project"):
            return False
        if path is None:
            base = self.path.parent if self.path else self.win.s.output_path()
            path, _ = QFileDialog.getOpenFileName(self.win, "Open project", str(base),
                                                  f"QuantaBench project ({proj.PROJECT_FILE});;All files (*)")
            if not path:
                return False
        root = proj.find_project(path)
        if root is None:
            QMessageBox.warning(self.win, "Open project", f"{path} is not a QuantaBench project (no {proj.PROJECT_FILE} "
                                "in it or above it).")
            self.remove_recent(path)
            return False
        if not self.maybe_save("open another project"):
            return False
        try:
            data = proj.load(root)
        except proj.ProjectError as exc:
            QMessageBox.warning(self.win, "Open project", str(exc))
            return False
        self.apply(root, data)
        return True

    def apply(self, root: Path, data: proj.ProjectData) -> None:
        """Show the project ``data`` of folder ``root`` (the list at once, run folders' news after)."""
        from . import workers

        w = self.win
        self._restoring = True
        try:
            w.clear_all()
            rows = [row_from_dict(d) for d in data.molecules]
            w.model.set_rows(rows)
            self._apply_build(data.build)
            table = getattr(w, "table_panel", None)
            if table is not None:
                table.store.clear()
                table.restore(TableState.from_json(data.table))
                plot = getattr(w, "plot_panel", None)
                if plot is not None:
                    plot.fill_boxes()
            from .workspaces import WORKSPACES

            if data.workspace in {x.key for x in WORKSPACES} and getattr(w, "workspaces", None) is not None:
                w.set_workspace(data.workspace)
            self.path, self.data = root, data
            run_dirs = sorted({r.run_dir for r in rows if r.run_dir})
            unfinished = [r.run_dir for r in rows if r.run_dir and r.status not in FINISHED]
            w.last_run_dir = Path(max(unfinished or run_dirs, key=lambda d: Path(d).name)) if run_dirs else None
            if rows:
                if not (data.current and w.mol_list.select_id(data.current)):
                    w.mol_list.setCurrentRow(0)
            w._update_counts()
        finally:
            self._restoring = False
        self.add_recent(root)
        self._set_clean()
        w.log(f"Opened project {root}: {len(rows):,} molecule(s), {len(run_dirs)} run folder(s).")
        if run_dirs:
            workers.start(workers.FunctionWorker(refresh_from_runs, run_dirs), w,
                          on_done=lambda got, r=root: self._on_runs_read(r, got))
        if table is not None:
            table.read_tables(run_dirs, [r for r in rows if r.folder and not r.run_dir and r.status == "ok"])

    def _apply_build(self, build: dict) -> None:
        if not build:
            return
        b = self.win.s.build
        known = {f.name for f in fields(b)}
        for k, v in build.items():
            if k in known:
                try:
                    setattr(b, k, type(getattr(b, k))(v) if getattr(b, k) is not None and v is not None else v)
                except (TypeError, ValueError):
                    pass
        self.win._load_stage_values()

    def _on_runs_read(self, root: Path, got: dict) -> None:
        """Statuses from the project's job tables (a resume outside the window): newer than the saved rows."""
        if self.path != root or self.win.worker is not None:
            return
        changed = 0
        self._restoring = True
        try:
            for r in self.win.model.rows():
                now = (got.get(r.run_dir) or {}).get(r.id)
                if now is None:
                    continue
                upd = {k: v for k, v in now.items() if getattr(r, k) != v and not (k == "folder" and not v)}
                if upd:
                    self.win.model.update(r.id, **upd)
                    changed += 1
        finally:
            self._restoring = False
        if changed:
            self.win._update_counts()
            self.win.log(f"  {changed} molecule(s) changed in their run folders since the project was saved.")
            self.mark_dirty()

    # ------------------------------------------------------------------ recent
    def _settings(self):
        from .main_window import _window_settings

        return _window_settings()

    def recent(self) -> list[str]:
        v = self._settings().value(RECENT_KEY) or []
        if isinstance(v, str):
            v = [v]
        return [str(p) for p in v if p]

    def add_recent(self, root: Path) -> None:
        items = [str(root)] + [p for p in self.recent() if Path(p) != Path(root)]
        qs = self._settings()
        qs.setValue(RECENT_KEY, items[:MAX_RECENT])
        qs.sync()

    def remove_recent(self, path) -> None:
        qs = self._settings()
        qs.setValue(RECENT_KEY, [p for p in self.recent() if Path(p) != Path(path)])
        qs.sync()

    def _fill_recent(self, menu: QMenu) -> None:
        menu.clear()
        items = self.recent()
        for p in items:
            a = menu.addAction(Path(p).stem, lambda _c=False, q=p: self.open_project(path=q))
            a.setToolTip(p)
            a.setStatusTip(p)
        if not items:
            menu.addAction("(none yet)").setEnabled(False)
        else:
            menu.addSeparator()
            menu.addAction("Clear the list", lambda: (self._settings().setValue(RECENT_KEY, []), menu.clear()))


def install(win) -> None:
    """``win.project`` and its File menu entries (above Open...)."""
    ctl = ProjectController(win)
    win.project = ctl

    def act(text, slot, shortcut=None, tip=None):
        a = QAction(text, win)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        return a

    ctl.a_new = act("New project", ctl.new_project, "Ctrl+N", "An empty, untitled project")
    ctl.a_open = act("Open project...", ctl.open_project, "Ctrl+Alt+O",
                     "Open a .qbproj folder: its molecules, results, build settings and Analyze table")
    ctl.a_save = act("Save project", ctl.save, "Ctrl+S",
                     "Save the molecules, build settings and Analyze table to the project folder")
    ctl.a_save_as = act("Save project as...", ctl.save_as, "Ctrl+Shift+S", "Save to a new .qbproj folder")
    ctl.a_close = act("Close project", ctl.close_project, None, "Close the project (asks to save changes)")
    ctl.recent_menu = QMenu("Open recent project", win)
    ctl.recent_menu.setToolTipsVisible(True)
    ctl.recent_menu.aboutToShow.connect(lambda: ctl._fill_recent(ctl.recent_menu))
    menu = next((a.menu() for a in win.menuBar().actions()
                 if a.menu() is not None and a.text().replace("&", "") == "File"), None)
    if menu is None:
        return
    first = menu.actions()[0] if menu.actions() else None
    for a in (ctl.a_new, ctl.a_open):
        menu.insertAction(first, a)
    menu.insertMenu(first, ctl.recent_menu)
    for a in (ctl.a_save, ctl.a_save_as, ctl.a_close):
        menu.insertAction(first, a)
    menu.insertSeparator(first)
    ctl.update_title()
