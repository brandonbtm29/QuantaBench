"""The Table dock (UI-3): the molecules x properties spreadsheet of the Analyze workspace.

One row per molecule of the Molecules dock, the same rows in the same order (until a column of the table is sorted):

- **Columns:** the light row fields (#, ID, name, status, formula, conformers, best energy, method, SMILES), the data
  fields an imported file posted, the summary and descriptor values of built molecules (``analysis.sheet``), and
  formula columns (``analysis.formula``). The values come from results in memory (a batch's finished molecules, an
  import, a result read from its folder: ``MainWindow.moleculeResult``) and, for a folder opened with File > Open
  run folder, from its summary.csv and descriptors.csv (and its job table for molecules those miss), read on a
  worker thread.
- **Columns... (the chooser)** shows or hides each column and lists the name a formula uses for it (its alias).
  **New column...** adds a formula column, **Fit...** fits a formula's parameters to a target column
  (``gui.table_dialogs``). Right-click a header for Hide, Edit formula, Remove.
- **Features** (the feature builder, DESC-3) adds ready-made formula columns (``analysis.features``): the
  dimensionless liquid-state groups of ``chem.liquid`` (Lorentz-Lorenz, refractive index, dipole orientation ratio,
  packing fraction, sphericity, the Lifshitz Hamaker constant and surface tension, the kT/v_m^⅔ unit), ratios of α,
  μ, V and A, and the reduced target σ* of any column. A group the run already has as a descriptor is not added
  again; one that needs another is added after it.
- **Join...** brings in columns of a CSV file by id (measured surface tensions or densities, ``pubchem targets``
  output): the file, its id column and the columns are part of the table's state, read again at the next start.
- **Sort** by any column (numbers as numbers, missing values last), **filter** by ID, name, SMILES, formula or
  method. **Selecting a row selects the molecule** in the Molecules dock, and the other way round.
- **Ctrl+C** copies the selected cells as tab-separated text (Excel, LibreOffice, Origin), **Ctrl+Shift+C** with the
  column titles. **Export** writes the rows and columns shown to CSV, or to an Excel workbook when openpyxl is
  installed.
- **Energies** (the best energy) are shown, copied and exported in the unit of Settings > Display; formulas see them
  in eV, the unit in their alias (``best_energy_eV``).
- **Persistence:** the column choices, the formula columns, the joined files and the plot's settings are a
  ``sheet.TableState``, kept in
  the window settings (``STATE_KEY``) until the project document (UI-2) holds them: ``TablePanel.state()`` and
  ``TablePanel.restore()`` are what it will call.

    SheetModel   QAbstractTableModel: column list, values aligned with the Molecules dock's rows, own sort and filter
    SheetView    the QTableView (copy, context menus)
    TablePanel   the dock's widget
    install      the "table" dock (an optional panel, ``MainWindow._install_panels``)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import units
from ..analysis import features as feat
from ..analysis import formula as fm
from ..analysis import sheet
from ..analysis.sheet import ColumnStore, FormulaSpec, JoinRead, JoinSpec, TableState

#: the window settings key of the table's state (until UI-2's project document keeps it)
STATE_KEY = "analyze/table"
STATUS_TEXT = {"ok": "built", "failed": "failed", "cancelled": "cancelled", "pending": "not built"}
XLSX_MISSING = ("Excel export needs openpyxl: pip install openpyxl (or the excel extra, "
                "pip install \"quantabench[excel]\"). CSV export works without it.")


def set_clipboard_text(text: str, tries: int = 6) -> bool:
    """``QClipboard.setText``, retried. On Windows another program (clipboard history, a clipboard manager) can hold
    the clipboard for a moment; Qt's write then fails ("OleSetClipboard: ... OpenClipboard Failed") and the old
    content stays. Seen on BTM NOVA about one write in ten when two come quickly. At most ~0.6 s in all."""
    clip = QApplication.clipboard()
    for i in range(tries):
        clip.setText(text)
        if clip.text() == text:
            return True
        time.sleep(0.03 * (i + 1))
    return False


@dataclass
class Column:
    key: str                 # unique: row field name, "field:<name>", a summary/descriptor name, "fx:<name>",
                             # "join:<file label>::<name>"
    title: str
    alias: str               # how a formula names it
    source: str              # "row" | "field" | "data" | "joined" | "formula"
    numeric: bool
    width: int = 96
    energy: bool = False     # stored in eV, shown in the display unit
    expression: str = ""
    note: str = ""
    error: str = ""

    @property
    def group(self) -> str:
        return {"row": "Molecule", "field": "Imported data fields", "data": "Summary and descriptors",
                "joined": "Joined from files", "formula": "Formula columns"}[self.source]


#: (key, title, numeric, width): the light row fields, as the Molecules dock has them
ROW_COLUMNS = (("seq", "#", True, 46), ("id", "ID", False, 120), ("name", "Name", False, 150),
               ("status", "Status", False, 76), ("formula", "Formula", False, 90),
               ("n_conformers", "Conf.", True, 50), ("best_energy_eV", "E", True, 110),
               ("method", "Method", False, 90), ("input_smiles", "SMILES", False, 160))
ROW_KEYS = frozenset(k for k, *_ in ROW_COLUMNS)
HIDDEN_ROW_COLUMNS = frozenset(("input_smiles",))
_ROW_TIPS = {"seq": "Input order (sorting by it restores the order of the Molecules dock)",
             "status": "built, failed, cancelled, not built, or the stage a running molecule is at",
             "n_conformers": "Conformers kept", "input_smiles": "The SMILES as entered or imported"}


def _row_value(r, key: str):
    if key == "seq":
        return float(r.seq + 1)
    if key == "status":
        return f"running ({r.stage})" if r.stage else STATUS_TEXT.get(r.status, r.status)
    if key == "n_conformers":
        return np.nan if r.n_conformers is None else float(r.n_conformers)
    if key == "best_energy_eV":
        return np.nan if r.energy_ev is None else float(r.energy_ev)
    if key == "input_smiles":
        return r.smiles or ""
    return getattr(r, key) or ""


def _parses(text: str) -> bool:
    try:
        fm.parse(text)
    except fm.FormulaError:
        return False
    return True


def _numeric_values(values) -> bool:
    seen = False
    for v in values:
        kind, _v = sheet.classify(v)
        if kind == "txt":
            return False
        seen = seen or kind == "num"
    return seen


class _Env:
    """alias -> values for a formula, with circular references caught."""

    def __init__(self, model: SheetModel, stack: tuple[str, ...] = ()):
        self.model, self.stack = model, stack

    def __getitem__(self, alias: str):
        col = self.model._by_alias.get(alias)
        if col is None:
            raise KeyError(alias)
        if col.source == "formula":
            if col.key in self.stack:
                raise fm.FormulaError(f"circular reference: {' -> '.join(self.stack + (col.key[3:],))}")
            values = self.model._values(col.key, self.stack)
            if col.error:
                raise fm.FormulaError(f"{alias} has an error ({col.error})")
            return values
        if not col.numeric:
            raise fm.FormulaError(f"{alias} is a text column: a formula needs numbers")
        return self.model._values(col.key, self.stack)

    def __contains__(self, alias) -> bool:
        return alias in self.model._by_alias


class SheetModel(QAbstractTableModel):
    """The table (module docstring). ``refresh`` recomputes rows, columns and the view; changes of the Molecules
    dock's model and of the store are coalesced into one ``refresh`` a few tens of ms later (``schedule``)."""

    #: the column choices or the formula columns changed (the panel saves the state)
    stateChanged = Signal()
    #: rows, columns or values were recomputed (the plot redraws)
    refreshed = Signal()

    def __init__(self, molecules, store: ColumnStore, display=None, parent=None, lazy: bool = False):
        """``lazy``: changes are only applied while some consumer is active (``set_active``: a dock that is shown),
        so a batch running in another workspace costs the table nothing; they are applied when one becomes active."""
        super().__init__(parent)
        self.molecules = molecules
        self.store = store
        self._energy = units.EnergyDisplay.of(display)
        self._rows: list = []
        self._slots = np.empty(0, dtype=np.int64)
        self._all: list[Column] = []
        self._by_key: dict[str, Column] = {}
        self._by_alias: dict[str, Column] = {}
        self._cols: list[Column] = []
        self._choices: dict[str, bool] = {}
        self._formulas: list[FormulaSpec] = []
        self.joined = ColumnStore()                  # joined files' columns, by norm_id; stored as "<label>::<name>"
        self._joins: list[JoinSpec] = []
        self._join_reads: dict[str, JoinRead] = {}   # label -> what was read (None-free: unreadable files are left out)
        self.join_errors: dict[str, str] = {}        # label -> why a joined file could not be read
        self._jslots = np.empty(0, dtype=np.int64)
        self._cache: dict[str, np.ndarray] = {}
        self._view = np.empty(0, dtype=np.int64)       # positions in _rows, filtered and sorted
        self._view_pos: dict[str, int] | None = None
        self._sort_key: str | None = None
        self._sort_desc = False
        self._needle = ""
        self._lazy = lazy
        self._active: set[str] = set()
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self.refresh)
        for sig in (molecules.rowsInserted, molecules.rowsRemoved, molecules.modelReset, molecules.layoutChanged,
                    molecules.columnsInserted, molecules.dataChanged):
            sig.connect(self.schedule)
        self.refresh()

    # ------------------------------------------------------------------ refresh
    def schedule(self, *_args) -> None:
        if self._lazy and not self._active:
            self._dirty = True
            return
        if not self._timer.isActive():
            self._timer.start()

    def set_active(self, who: str, on: bool) -> None:
        """A consumer (the Table or Plot dock) is shown (``on``) or hidden; see ``lazy``."""
        if on:
            self._active.add(who)
            if self._dirty:
                self._dirty = False
                self.schedule()
        else:
            self._active.discard(who)

    def refresh(self) -> None:
        """Rows from the Molecules dock's model, values from the store, columns, the formula values, the view."""
        self._timer.stop()
        self._dirty = False
        rows = self.molecules.rows()
        ids = [r.id for r in rows]
        self.store.prune(ids)
        old_ids = self.view_ids()
        old_keys = [c.key for c in self._cols]
        old_titles = [self._title(c) for c in self._cols]
        self._rows = rows
        self._slots = self.store.slots(ids)
        self._jslots = self.joined.slots([sheet.norm_id(i) for i in ids]) if self._joins else \
            np.full(len(ids), -1, dtype=np.int64)
        self._cache = {}
        self._build_columns()
        for c in self._all:
            if c.source == "formula":
                self._values(c.key)                    # sets c.error
        cols = [c for c in self._all if self.is_shown(c.key)]
        view = self._compute_view()
        new_ids = [rows[p].id for p in view]
        if [c.key for c in cols] == old_keys and len(new_ids) == len(old_ids) and \
                [self._title(c) for c in cols] == old_titles:
            if new_ids != old_ids:
                self._relayout(view, new_ids, old_ids)
            else:
                self._view = view
                self._view_pos = None
            if len(view) and cols:
                self.dataChanged.emit(self.index(0, 0), self.index(len(view) - 1, len(cols) - 1))
        else:
            self.beginResetModel()
            self._cols = cols
            self._view = view
            self._view_pos = None
            self.endResetModel()
        self.refreshed.emit()

    def _relayout(self, view: np.ndarray, new_ids: list[str], old_ids: list[str]) -> None:
        self.layoutAboutToBeChanged.emit()
        old = self.persistentIndexList()
        self._view = view
        self._view_pos = None
        pos = {i: n for n, i in enumerate(new_ids)}
        new = []
        for ix in old:
            r = pos.get(old_ids[ix.row()]) if 0 <= ix.row() < len(old_ids) else None
            new.append(self.index(r, ix.column()) if r is not None else QModelIndex())
        self.changePersistentIndexList(old, new)
        self.layoutChanged.emit()

    def _build_columns(self) -> None:
        out: list[Column] = []
        for key, title, numeric, width in ROW_COLUMNS:
            out.append(Column(key, title, key, "row", numeric, width, energy=key == "best_energy_eV"))
        reserved = set(ROW_KEYS) | {f.name for f in self._formulas}
        data = [n for n in self.store.names() if n not in ROW_KEYS]
        data_alias = fm.aliases(data, reserved)
        reserved |= set(data_alias.values())
        fields = self.molecules.extra_columns() if hasattr(self.molecules, "extra_columns") else []
        field_alias = fm.aliases(fields, reserved)
        for name in fields:
            numeric = _numeric_values((r.fields or {}).get(name) for r in self._rows)
            out.append(Column(f"field:{name}", name, field_alias[name], "field", numeric))
        for name in data:
            out.append(Column(name, name, data_alias[name], "data", self.store.is_numeric(name)))
        joined = [(spec, name) for spec in self._joins if spec.label in self._join_reads
                  for name in self._join_reads[spec.label].block.names]
        plain = [name for _spec, name in joined]
        wanted = [name if plain.count(name) == 1 else f"{spec.label}_{name}" for spec, name in joined]
        join_alias = fm.aliases(wanted, reserved)
        reserved |= set(join_alias.values())
        for (spec, name), want in zip(joined, wanted):
            stored = f"{spec.label}::{name}"
            out.append(Column(f"join:{stored}", name if plain.count(name) == 1 else f"{name} ({spec.label})",
                              join_alias[want], "joined", self.joined.is_numeric(stored),
                              note=f"joined from {spec.path} by {spec.id_column}"))
        for f in self._formulas:
            out.append(Column(f"fx:{f.name}", f.name, f.name, "formula", True, expression=f.expression,
                              note=f.note))
        self._all = out
        self._by_key = {c.key: c for c in out}
        self._by_alias = {c.alias: c for c in out}

    def _compute_view(self) -> np.ndarray:
        n = len(self._rows)
        pos = np.arange(n, dtype=np.int64)
        if self._needle:
            needle = self._needle
            pos = np.fromiter((i for i, r in enumerate(self._rows) if needle in r.id.lower()
                               or needle in (r.name or "").lower() or needle in (r.smiles or "").lower()
                               or needle in (r.formula or "").lower() or needle in (r.method or "").lower()),
                              dtype=np.int64)
        key = self._sort_key if self._sort_key in self._by_key else None
        if key is None or not len(pos):
            return pos
        v = self._values(key)[pos]
        col = self._by_key[key]
        if col.numeric:
            v = np.asarray(v, dtype=float)
            order = np.argsort(-v if self._sort_desc else v, kind="stable")     # NaN last either way
            return pos[order]
        filled = [i for i, s in enumerate(v) if s is not None and str(s) != ""]
        empty = [i for i, s in enumerate(v) if s is None or str(s) == ""]
        filled.sort(key=lambda i: str(v[i]).lower(), reverse=self._sort_desc)
        return pos[np.array(filled + empty, dtype=np.int64)]

    # ------------------------------------------------------------------ values
    def _values(self, key: str, stack: tuple[str, ...] = ()) -> np.ndarray:
        """Values of column ``key`` for every row of the Molecules dock's model (in its order), as stored."""
        got = self._cache.get(key)
        if got is not None:
            return got
        col = self._by_key[key]
        n = len(self._rows)
        if col.source == "row":
            vals = [_row_value(r, key) for r in self._rows]
            if col.numeric:
                out = np.array(vals, dtype=float)
            else:
                out = np.empty(n, dtype=object)
                out[:] = vals
        elif col.source == "field":
            name = key.split(":", 1)[1]
            vals = [(r.fields or {}).get(name) for r in self._rows]
            if col.numeric:
                out = np.full(n, np.nan)
                for i, v in enumerate(vals):
                    kind, x = sheet.classify(v)
                    if kind == "num":
                        out[i] = x
            else:
                out = np.empty(n, dtype=object)
                out[:] = [None if v is None else str(v) for v in vals]
        elif col.source == "data":
            out = self.store.column(key, self._slots)
        elif col.source == "joined":
            out = self.joined.column(key[len("join:"):], self._jslots)
        else:
            col.error = ""
            try:
                out = fm.parse(col.expression).evaluate(_Env(self, stack + (key,)), n=n)
            except fm.FormulaError as exc:
                col.error = str(exc)
                out = np.full(n, np.nan)
        self._cache[key] = out
        return out

    def values(self, key: str) -> np.ndarray:
        """Column ``key`` for every row of the Molecules dock (its order), in the stored units (energies in eV)."""
        return self._values(key)

    def display_values(self, key: str) -> np.ndarray:
        """``values`` with energies in the display unit (what the table shows, copies and exports)."""
        v = self._values(key)
        if self._by_key[key].energy:
            return np.asarray(v, dtype=float) * units.factor(units.EV, self._energy.unit("absolute"))
        return v

    def formula_env(self) -> _Env:
        """alias -> values of every numeric column (``Formula.evaluate`` and ``formula.fit`` take it)."""
        return _Env(self)

    # ------------------------------------------------------------------ columns
    def columns(self) -> list[Column]:
        return list(self._all)

    def shown_columns(self) -> list[Column]:
        return list(self._cols)

    def column(self, key: str) -> Column | None:
        return self._by_key.get(key)

    def column_by_alias(self, alias: str) -> Column | None:
        return self._by_alias.get(alias)

    def numeric_columns(self) -> list[Column]:
        return [c for c in self._all if c.numeric]

    def aliases(self) -> list[str]:
        return [c.alias for c in self._all]

    def default_shown(self, col: Column) -> bool:
        if col.source == "row":
            return col.key not in HIDDEN_ROW_COLUMNS
        if col.source == "data":
            return col.key in sheet.DEFAULT_SHOWN
        return True

    def is_shown(self, key: str) -> bool:
        col = self._by_key.get(key)
        if col is None:
            return False
        choice = self._choices.get(key)
        return self.default_shown(col) if choice is None else choice

    def set_shown(self, key: str, on: bool) -> None:
        col = self._by_key.get(key)
        if col is None or self.is_shown(key) == on:
            return
        if on == self.default_shown(col):
            self._choices.pop(key, None)
        else:
            self._choices[key] = on
        self.refresh()
        self.stateChanged.emit()

    def set_shown_many(self, choices: dict[str, bool]) -> None:
        """Several ``set_shown`` at once (one refresh)."""
        changed = False
        for key, on in choices.items():
            col = self._by_key.get(key)
            if col is None or self.is_shown(key) == on:
                continue
            changed = True
            if on == self.default_shown(col):
                self._choices.pop(key, None)
            else:
                self._choices[key] = on
        if changed:
            self.refresh()
            self.stateChanged.emit()

    def reset_choices(self) -> None:
        self._choices.clear()
        self.refresh()
        self.stateChanged.emit()

    def _title(self, col: Column) -> str:
        if col.energy:
            return self._energy.header(col.title, "absolute")
        return col.title + (" ⚠" if col.error else "")

    def title(self, key: str) -> str:
        col = self._by_key.get(key)
        return self._title(col) if col is not None else key

    def set_display(self, display) -> None:
        d = units.EnergyDisplay.of(display)
        if d != self._energy:
            self._energy = d
            for i, c in enumerate(self._cols):
                if c.energy:
                    self.headerDataChanged.emit(Qt.Orientation.Horizontal, i, i)
                    if len(self._view):
                        self.dataChanged.emit(self.index(0, i), self.index(len(self._view) - 1, i))
            self.refreshed.emit()

    # ------------------------------------------------------------------ formula columns
    def formulas(self) -> list[FormulaSpec]:
        return [replace(f) for f in self._formulas]

    def check_formula(self, name: str, expression: str, old_name: str | None = None) -> fm.Formula:
        """Raise ``FormulaError`` if ``name`` cannot name a new formula column or ``expression`` uses names that
        are not columns; return the parsed formula."""
        name = name.strip()
        if not fm.is_name(name):
            raise fm.FormulaError(f"{name!r} cannot name a column: letters, digits and _ (a letter or _ first), not "
                                  "a Python keyword or a function name")
        taken = {c.alias for c in self._all if not (c.source == "formula" and c.alias == old_name)}
        if name in taken:
            raise fm.FormulaError(f"{name} is already the name of a column")
        f = fm.check(expression, [a for a in self.aliases() if a != old_name])
        if name in f.names:
            raise fm.FormulaError(f"{name} cannot use itself")
        # a cycle through other formula columns (a uses b, b uses a)
        uses = {spec.name: set(fm.parse(spec.expression).names) for spec in self._formulas
                if spec.name != old_name and _parses(spec.expression)}
        uses[name] = set(f.names)
        seen: set[str] = set()
        todo = [n for n in f.names if n in uses]
        while todo:
            n = todo.pop()
            if n == name:
                raise fm.FormulaError(f"circular reference: {name} would use itself through other formula columns")
            if n not in seen:
                seen.add(n)
                todo += [m for m in uses.get(n, ()) if m in uses]
        f.evaluate(self.formula_env(), n=len(self._rows))      # a text column, a column with an error: refused now
        return f

    def add_formula(self, name: str, expression: str, note: str = "") -> Column:
        f = self.check_formula(name, expression)
        self._formulas.append(FormulaSpec(name.strip(), f.text, note))
        self._choices.pop(f"fx:{name.strip()}", None)
        self.refresh()
        self.stateChanged.emit()
        return self._by_key[f"fx:{name.strip()}"]

    def update_formula(self, old_name: str, name: str, expression: str, note: str | None = None) -> Column:
        f = self.check_formula(name, expression, old_name=old_name)
        for k, spec in enumerate(self._formulas):
            if spec.name == old_name:
                self._formulas[k] = FormulaSpec(name.strip(), f.text, spec.note if note is None else note)
                if old_name != name.strip() and f"fx:{old_name}" in self._choices:
                    self._choices[f"fx:{name.strip()}"] = self._choices.pop(f"fx:{old_name}")
                break
        else:
            raise KeyError(old_name)
        if self._sort_key == f"fx:{old_name}":
            self._sort_key = f"fx:{name.strip()}"
        self.refresh()
        self.stateChanged.emit()
        return self._by_key[f"fx:{name.strip()}"]

    def remove_formula(self, name: str) -> None:
        self._formulas = [f for f in self._formulas if f.name != name]
        self._choices.pop(f"fx:{name}", None)
        self.refresh()
        self.stateChanged.emit()

    def add_formulas(self, items) -> list[Column]:
        """Several formula columns at once, in order (a later one may use an earlier one), one refresh: [(name,
        expression, note)]. All or none: a formula that is refused takes the ones before it back out."""
        before = list(self._formulas)
        try:
            for name, expression, note in items:
                f = self.check_formula(name, expression)
                self._formulas.append(FormulaSpec(name.strip(), f.text, note))
                self._choices.pop(f"fx:{name.strip()}", None)
                self._build_columns()
        except Exception:
            self._formulas = before
            self._build_columns()
            raise
        self.refresh()
        self.stateChanged.emit()
        return [self._by_key[f"fx:{name.strip()}"] for name, _e, _n in items]

    # ------------------------------------------------------------------ feature presets (analysis.features)
    def feature_status(self, name: str) -> str:
        """"" when the preset ``name`` can be added; else why not."""
        return feat.status(name, self.aliases(), [f.name for f in self._formulas])

    def add_features(self, names) -> list[Column]:
        """The formula columns of the presets ``names`` (their dependencies first; ``FeatureError`` when an input
        column is missing)."""
        return self.add_formulas(feat.plan(list(names), self.aliases(), [f.name for f in self._formulas]))

    def add_reduced_target(self, target_alias: str, name: str | None = None) -> list[Column]:
        """σ* = target / (kT/v_m^⅔) as a formula column (and the scale's, when the run has no descriptor for it)."""
        return self.add_formulas(feat.reduced_target(target_alias, self.aliases(), [f.name for f in self._formulas],
                                                     name))

    # ------------------------------------------------------------------ joined files
    def joins(self) -> list[JoinSpec]:
        return [replace(j, columns=list(j.columns)) for j in self._joins]

    def _rebuild_joined(self) -> None:
        self.joined = ColumnStore()
        for spec in self._joins:
            got = self._join_reads.get(spec.label)
            if got is None:
                continue
            b = got.block
            self.joined.merge(sheet.Block(ids=list(b.ids), columns={f"{spec.label}::{n}": v
                                                                    for n, v in b.columns.items()}))

    def set_joins(self, pairs, refresh: bool = True) -> None:
        """The joined files: [(spec, JoinRead or an error text)]; an error keeps the spec (it is part of the
        state) without its columns, and ``join_errors`` says why."""
        self._joins, self._join_reads, self.join_errors = [], {}, {}
        for spec, got in pairs:
            if any(j.label == spec.label for j in self._joins):
                continue
            self._joins.append(replace(spec, columns=list(spec.columns)))
            if isinstance(got, JoinRead):
                self._join_reads[spec.label] = got
            else:
                self.join_errors[spec.label] = str(got)
        self._rebuild_joined()
        if refresh:
            self.refresh()

    def add_join(self, spec: JoinSpec, got: JoinRead) -> int:
        """Join ``spec``'s columns (replacing a file of the same label); the number of rows that matched."""
        pairs = [(j, self._join_reads.get(j.label, self.join_errors.get(j.label, ""))) for j in self._joins
                 if j.label != spec.label]
        self.set_joins(pairs + [(spec, got)])
        self.stateChanged.emit()
        return self.join_matches(spec.label)

    def remove_join(self, label: str) -> None:
        pairs = [(j, self._join_reads.get(j.label, self.join_errors.get(j.label, ""))) for j in self._joins
                 if j.label != label]
        for key in [k for k in self._choices if k.startswith(f"join:{label}::")]:
            self._choices.pop(key)
        self.set_joins(pairs)
        self.stateChanged.emit()

    def join_matches(self, label: str) -> int:
        """How many of the table's rows have a row in the joined file ``label``."""
        got = self._join_reads.get(label)
        if got is None:
            return 0
        have = set(got.block.ids)
        return sum(1 for r in self._rows if sheet.norm_id(r.id) in have)

    # ------------------------------------------------------------------ state
    def state(self) -> TableState:
        return TableState(columns=dict(self._choices), formulas=self.formulas(), joins=self.joins())

    def restore(self, state: TableState, joins=None) -> None:
        """``joins``: [(spec, JoinRead or error)] for ``state.joins`` (the panel reads the files); None keeps no
        joined file."""
        self.set_joins(list(joins or ()), refresh=False)
        self._choices = dict(state.columns)
        self._formulas = []
        seen = set()
        for f in state.formulas:              # names must stay unique and usable (a hand-edited state)
            if fm.is_name(f.name) and f.name not in seen and f.name not in ROW_KEYS:
                seen.add(f.name)
                self._formulas.append(replace(f))
        self.refresh()

    # ------------------------------------------------------------------ sort, filter, rows
    def set_filter(self, text: str) -> None:
        needle = text.strip().lower()
        if needle != self._needle:
            self._needle = needle
            self.refresh()

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        key = self._cols[column].key if 0 <= column < len(self._cols) else None
        desc = order == Qt.SortOrder.DescendingOrder
        if key == self._sort_key and desc == self._sort_desc and column >= 0:
            return
        self._sort_key, self._sort_desc = key, desc
        view = self._compute_view()
        ids = [self._rows[p].id for p in view]
        old = self.view_ids()
        if len(ids) == len(old):
            self._relayout(view, ids, old)
        else:
            self.beginResetModel()
            self._view, self._view_pos = view, None
            self.endResetModel()
        self.refreshed.emit()

    def sort_state(self) -> tuple[str | None, bool]:
        return self._sort_key, self._sort_desc

    def view_positions(self) -> np.ndarray:
        """The rows shown (positions in the Molecules dock's model order), in the table's order."""
        return self._view

    def view_ids(self) -> list[str]:
        return [self._rows[p].id for p in self._view] if len(self._rows) else []

    def row_id(self, view_row: int) -> str:
        return self._rows[self._view[view_row]].id if 0 <= view_row < len(self._view) else ""

    def view_row(self, ident: str) -> int:
        if self._view_pos is None:
            self._view_pos = {self._rows[p].id: n for n, p in enumerate(self._view)}
        return self._view_pos.get(ident, -1)

    def total_rows(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------------ copy and export
    def export_rows(self, view_rows=None, keys=None) -> tuple[list[str], list[list]]:
        """(titles, rows) of ``view_rows`` (default: every row shown) and columns ``keys`` (default: those shown),
        as shown: energies in the display unit, numbers as floats (NaN missing), text as str or None."""
        cols = [self._by_key[k] for k in keys] if keys is not None else list(self._cols)
        rows = self._view if view_rows is None else self._view[np.asarray(list(view_rows), dtype=np.int64)]
        arrays = []
        for c in cols:
            a = self.display_values(c.key)[rows]
            arrays.append(a.tolist())
        return [self._title(c).replace(" ⚠", "") for c in cols], [list(r) for r in zip(*arrays)] if arrays else []

    # ------------------------------------------------------------------ Qt
    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: B008, N802 - Qt's signature
        return 0 if parent.isValid() else len(self._view)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else len(self._cols)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation != Qt.Orientation.Horizontal or not 0 <= section < len(self._cols):
            return None
        col = self._cols[section]
        if role == Qt.ItemDataRole.DisplayRole:
            return self._title(col)
        if role == Qt.ItemDataRole.ToolTipRole:
            lines = [self._title(col)]
            if col.energy:
                lines.append(f"Energy of the best conformer, shown in {self._energy.label('absolute')}; formulas "
                             "use it in eV")
            elif col.source == "row" and col.key in _ROW_TIPS:
                lines.append(_ROW_TIPS[col.key])
            elif col.source == "field":
                lines.append("A data field of the imported file")
            if col.numeric:
                lines.append(f"In formulas: {col.alias}")
            else:
                lines.append("Text (formulas need numbers)")
            if col.expression:
                lines.append(f"= {col.expression}")
            if col.note:
                lines.append(col.note)
            if col.error:
                lines.append(f"Error: {col.error}")
            return "\n".join(lines)
        if role == Qt.ItemDataRole.TextAlignmentRole and col.numeric:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._view):
            return None
        col = self._cols[index.column()]
        p = self._view[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            v = self._values(col.key)[p]
            if col.numeric:
                v = float(v)
                if v != v:                               # NaN: missing
                    return ""
                if col.energy:
                    return self._energy.text(v, units.EV, "absolute", 4)
                if col.key in ("seq", "n_conformers"):
                    return str(int(v))
                return f"{v:.6g}"
            return "" if v is None else str(v)
        if role == Qt.ItemDataRole.TextAlignmentRole and col.numeric:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            r = self._rows[p]
            if col.error:
                return f"{col.alias}: {col.error}"
            if col.key in ("id", "name", "status"):
                bits = [f"{r.id}  {r.name}".strip(), r.smiles, r.error]
                return "\n".join(b for b in bits if b)
        return None


class SheetView(QTableView):
    """The table: Ctrl+C copies the selected cells as TSV, Ctrl+Shift+C with the column titles."""

    def __init__(self, model: SheetModel, parent=None):
        super().__init__(parent)
        self.setModel(model)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setWordWrap(False)
        self.setAlternatingRowColors(True)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        vh = self.verticalHeader()
        vh.setVisible(False)
        vh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)      # uniform rows: scrolling costs O(visible rows)
        vh.setDefaultSectionSize(self.fontMetrics().height() + 6)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)   # never ResizeToContents: O(all rows)
        hh.setHighlightSections(False)
        hh.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)        # the Molecules dock's order until a click
        self.setSortingEnabled(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def selected_block(self) -> tuple[list[int], list[int], set[tuple[int, int]]]:
        idx = self.selectionModel().selectedIndexes()
        if not idx and self.currentIndex().isValid():
            idx = [self.currentIndex()]
        cells = {(i.row(), i.column()) for i in idx}
        return sorted({r for r, _ in cells}), sorted({c for _, c in cells}), cells

    def copy_text(self, headers: bool = False) -> str:
        """The selected cells as tab-separated text (cells outside the selection in its rows and columns blank)."""
        rows, cols, cells = self.selected_block()
        if not rows:
            return ""
        model: SheetModel = self.model()
        keys = [model.shown_columns()[c].key for c in cols]
        titles, values = model.export_rows(rows, keys)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if (r, c) not in cells:
                    values[i][j] = None
        return sheet.tsv(values, titles if headers else None)

    def copy(self, headers: bool = False) -> None:
        text = self.copy_text(headers)
        if text:
            set_clipboard_text(text)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy(headers=False)
            return
        mods = event.modifiers()
        if event.key() == Qt.Key.Key_C and mods & Qt.KeyboardModifier.ControlModifier \
                and mods & Qt.KeyboardModifier.ShiftModifier:
            self.copy(headers=True)
            return
        super().keyPressEvent(event)


class TablePanel(QWidget):
    """The Table dock's widget (module docstring)."""

    #: the table's state was saved (its columns, formulas, joined files or the plot changed): a project has changes
    stateSaved = Signal()

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.store = ColumnStore()
        self.model = SheetModel(win.model, self.store, display=win.s, parent=self, lazy=True)
        self.plot_settings: dict = {}
        self.last_export: Path | None = None
        self._syncing = False
        self._reading: Path | None = None
        self._read_token = 0                          # project reads (read_tables): only the newest one merges
        self._widths: dict[str, int] = {}
        self._kept: tuple[str, str, int, int] | None = None

        self.filter = QLineEdit()
        self.filter.setObjectName("table_filter")
        self.filter.setPlaceholderText("Filter ID, name, SMILES, formula, method")
        self.filter.setClearButtonEnabled(True)
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(150)
        self._filter_timer.timeout.connect(lambda: self.model.set_filter(self.filter.text()))
        self.filter.textChanged.connect(self._filter_timer.start)
        self.filter.returnPressed.connect(lambda: self.model.set_filter(self.filter.text()))
        self.count = QLabel()
        self.count.setStyleSheet("color: gray;")
        self.columns_btn = QPushButton("Columns...")
        self.columns_btn.setToolTip("Show or hide columns; lists the name each column has in formulas")
        self.columns_btn.clicked.connect(lambda: self.choose_columns())
        self.formula_btn = QPushButton("New column...")
        self.formula_btn.setToolTip("A formula column: arithmetic on other columns (+ - * / **, log, exp, sqrt, ...)")
        self.formula_btn.clicked.connect(lambda: self.edit_formula())
        self.fit_btn = QPushButton("Fit...")
        self.fit_btn.setToolTip("Fit the parameters of a formula to a column (least squares), e.g. the quantum "
                                "parachor ((a*alpha + b*mu**2 + c)/Vm)**4")
        self.fit_btn.clicked.connect(lambda: self.open_fit())
        self.features_btn = QToolButton()
        self.features_btn.setObjectName("table_features")
        self.features_btn.setText("Features")
        self.features_btn.setToolTip("Ready-made formula columns: dimensionless liquid-state groups (Lorentz-Lorenz, "
                                     "packing fraction, sphericity, Lifshitz σ, ...), ratios of α, μ, V and A, and "
                                     "the reduced target σ* = σ·v_m^⅔/kT")
        self.features_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.features_menu = QMenu(self.features_btn)
        self.features_menu.setToolTipsVisible(True)
        self.features_menu.aboutToShow.connect(self._fill_features_menu)
        self.features_btn.setMenu(self.features_menu)
        self.join_btn = QPushButton("Join...")
        self.join_btn.setObjectName("table_join")
        self.join_btn.setToolTip("Bring in columns of a CSV file by id: measured surface tensions, densities, "
                                 "`pubchem targets` output")
        self.join_btn.clicked.connect(lambda: self.join_dialog())
        self.export_btn = QToolButton()
        self.export_btn.setText("Export")
        self.export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.export_btn)
        self.a_csv = QAction("CSV...", self)
        self.a_csv.setToolTip("The rows and columns shown, as a CSV file (UTF-8)")
        self.a_csv.triggered.connect(lambda: self.export_dialog("csv"))
        self.a_xlsx = QAction("Excel workbook (.xlsx)...", self)
        self.a_xlsx.triggered.connect(lambda: self.export_dialog("xlsx"))
        if sheet.xlsx_available():
            self.a_xlsx.setToolTip("The rows and columns shown, as an Excel workbook")
        else:
            self.a_xlsx.setEnabled(False)
            self.a_xlsx.setToolTip(XLSX_MISSING)
            self.a_xlsx.setText("Excel workbook (.xlsx): pip install openpyxl")
        menu.setToolTipsVisible(True)
        menu.addAction(self.a_csv)
        menu.addAction(self.a_xlsx)
        self.export_btn.setMenu(menu)
        top = QHBoxLayout()
        top.setContentsMargins(4, 2, 4, 2)
        top.addWidget(self.filter, 1)
        top.addWidget(self.count)
        for b in (self.columns_btn, self.formula_btn, self.fit_btn, self.features_btn, self.join_btn,
                  self.export_btn):
            top.addWidget(b)

        self.view = SheetView(self.model)
        self.view.setObjectName("table_view")
        self.view.customContextMenuRequested.connect(self._cell_menu)
        hh = self.view.horizontalHeader()
        hh.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        hh.customContextMenuRequested.connect(self._header_menu)
        hh.sectionResized.connect(self._on_resized)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)

        self.model.modelAboutToBeReset.connect(self._before_reset)
        self.model.modelReset.connect(self._after_reset)
        self.model.refreshed.connect(self._update_count)
        self.model.stateChanged.connect(self.save_state)
        self.view.selectionModel().currentChanged.connect(self._on_current)
        win.mol_list.currentIdChanged.connect(self.follow)
        win.moleculeResult.connect(self.add_result)
        win.runFolderOpened.connect(self.read_run_folder)
        win.displayChanged.connect(self.model.set_display)
        self.restore(load_state())
        self._apply_widths()
        self._update_count()

    # ------------------------------------------------------------------ data in
    def add_result(self, res) -> None:
        """A molecule's result arrived (a batch finished it, an import, a read from its folder)."""
        ident = getattr(getattr(res, "input", None), "id", None)
        if not ident:
            return
        if res.status == "ok":
            self.store.set_row(ident, sheet.result_values(res))
        else:
            self.store.clear_row(ident)
        self.model.schedule()

    def read_run_folder(self, folder) -> None:
        """Read an opened run folder's summary and descriptor tables on a worker thread."""
        from . import workers

        folder = Path(folder)
        self._reading = folder
        self._update_count()

        def done(block) -> None:
            if self._reading != folder:
                return                                # another folder was opened meanwhile
            self._reading = None
            if Path(getattr(self.win, "run_folder", None) or "") == folder:
                self.store.merge(block)
                self.model.refresh()
            self._update_count()

        def failed(msg) -> None:
            if self._reading == folder:
                self._reading = None
                self.win.log(f"Table: could not read the values of {folder}: {msg}")
                self._update_count()

        workers.start(workers.FunctionWorker(sheet.read_run_tables, folder), self.win, on_done=done, on_failed=failed)

    def read_tables(self, run_dirs, rows=()) -> None:
        """A project's values (UI-2), on a worker thread: the summary and descriptor tables of each run folder in
        ``run_dirs`` and the results of ``rows`` that live in a folder of their own (a project's results/), merged
        into the table when all are read. A newer call (another project) makes an older one's values be dropped."""
        from . import workers

        self._read_token += 1
        token = self._read_token
        folders = [str(d) for d in run_dirs]
        singles = [(r.id, r.folder, r.input) for r in rows]
        if not folders and not singles:
            return
        self._reading = Path(folders[0]) if folders else Path(singles[0][1])
        self._update_count()

        def read():
            from ..chem import export

            blocks = []
            for d in folders:
                try:
                    blocks.append(sheet.read_run_tables(d))
                except (OSError, ValueError) as exc:
                    blocks.append(f"{d}: {exc}")
            got_ids, got_rows = [], []
            for ident, folder, inp in singles:
                try:
                    res = export.read_molecule(Path(folder), ident, inp)
                except (OSError, ValueError, KeyError):
                    continue
                got_ids.append(ident)
                got_rows.append(sheet.result_values(res))
            if got_ids:
                blocks.append(sheet.Block.from_rows(got_ids, got_rows, parse=False))
            return blocks

        def done(blocks) -> None:
            if token != self._read_token:
                return
            self._reading = None
            for b in blocks:
                if isinstance(b, str):
                    self.win.log(f"Table: could not read {b}")
                else:
                    self.store.merge(b)
            self.model.refresh()
            self._update_count()

        def failed(msg) -> None:
            if token == self._read_token:
                self._reading = None
                self.win.log(f"Table: could not read the project's values: {msg}")
                self._update_count()

        workers.start(workers.FunctionWorker(read), self.win, on_done=done, on_failed=failed)

    def reading(self) -> bool:
        return self._reading is not None

    # ------------------------------------------------------------------ selection
    def select_molecule(self, ident: str) -> bool:
        """Make ``ident`` the Molecules dock's current molecule (clearing that dock's search if it hides it)."""
        ml = self.win.mol_list
        if not ident:
            return False
        if ml.current_id() == ident:
            return True
        if ml.select_id(ident):
            return True
        win = self.win
        if hasattr(win, "search"):
            win.search.blockSignals(True)
            win.search.clear()
            win.search.blockSignals(False)
            win.status_filter.blockSignals(True)
            win.status_filter.setCurrentIndex(0)
            win.status_filter.blockSignals(False)
            win._apply_search()
        return ml.select_id(ident)

    def follow(self, ident: str) -> None:
        """The Molecules dock's current molecule changed: select its row here (no echo)."""
        row = self.model.view_row(ident)
        if row < 0:
            return
        cur = self.view.currentIndex()
        if cur.isValid() and cur.row() == row:
            return
        col = cur.column() if cur.isValid() else max(0, self._column_index("id"))
        ix = self.model.index(row, col)
        sm = self.view.selectionModel()
        self._syncing = True
        try:
            sm.setCurrentIndex(ix, sm.SelectionFlag.ClearAndSelect | sm.SelectionFlag.Rows)
            self.view.scrollTo(ix)
        finally:
            self._syncing = False

    def _on_current(self, cur, _prev) -> None:
        if self._syncing or not cur.isValid():
            return
        self.select_molecule(self.model.row_id(cur.row()))

    def current_id(self) -> str:
        cur = self.view.currentIndex()
        return self.model.row_id(cur.row()) if cur.isValid() else ""

    def _column_index(self, key: str) -> int:
        for i, c in enumerate(self.model.shown_columns()):
            if c.key == key:
                return i
        return -1

    def _before_reset(self) -> None:
        cur = self.view.currentIndex()
        cols = self.model.shown_columns()
        key = cols[cur.column()].key if cur.isValid() and cur.column() < len(cols) else ""
        self._kept = (self.current_id(), key, self.view.verticalScrollBar().value(),
                      self.view.horizontalScrollBar().value())

    def _after_reset(self) -> None:
        self._apply_widths()
        key, desc = self.model.sort_state()
        idx = self._column_index(key) if key else -1
        hh = self.view.horizontalHeader()
        hh.blockSignals(True)
        hh.setSortIndicator(idx, Qt.SortOrder.DescendingOrder if desc else Qt.SortOrder.AscendingOrder)
        hh.blockSignals(False)
        kept, self._kept = self._kept, None
        if not kept:
            return
        ident, ckey, vs, hs = kept
        row = self.model.view_row(ident) if ident else -1
        if row >= 0:
            col = max(0, self._column_index(ckey) if ckey else 0)
            ix = self.model.index(row, col)
            self._syncing = True
            try:
                sm = self.view.selectionModel()
                sm.setCurrentIndex(ix, sm.SelectionFlag.ClearAndSelect | sm.SelectionFlag.Rows)
            finally:
                self._syncing = False
        QTimer.singleShot(0, lambda: (self.view.verticalScrollBar().setValue(vs),
                                      self.view.horizontalScrollBar().setValue(hs)))

    def _on_resized(self, index: int, _old: int, new: int) -> None:
        cols = self.model.shown_columns()
        if 0 <= index < len(cols):
            self._widths[cols[index].key] = new

    def _apply_widths(self) -> None:
        """Each column as wide as the user made it, else as its title (at least its default, at most 220 px)."""
        hh = self.view.horizontalHeader()
        metrics = hh.fontMetrics()
        hh.blockSignals(True)
        for i, c in enumerate(self.model.shown_columns()):
            width = self._widths.get(c.key)
            if width is None:
                width = c.width if c.source == "row" else \
                    min(220, max(c.width, metrics.horizontalAdvance(self.model.title(c.key)) + 22))
            self.view.setColumnWidth(i, width)
        hh.blockSignals(False)

    def _update_count(self, *_args) -> None:
        n, shown = self.model.total_rows(), self.model.rowCount()
        text = f"{n:,} molecule{'s' if n != 1 else ''}" if shown == n else f"{shown:,} of {n:,} molecules"
        if self._reading is not None:
            text += f"; reading the values of {self._reading.name}..."
        self.count.setText(text)

    # ------------------------------------------------------------------ menus and dialogs
    def _cell_menu(self, pos) -> None:
        m = QMenu(self)
        m.addAction("Copy\tCtrl+C", lambda: self.view.copy(False))
        m.addAction("Copy with column titles\tCtrl+Shift+C", lambda: self.view.copy(True))
        m.addAction("Select all\tCtrl+A", self.view.selectAll)
        m.exec(self.view.viewport().mapToGlobal(pos))

    def _header_menu(self, pos) -> None:
        hh = self.view.horizontalHeader()
        i = hh.logicalIndexAt(pos)
        cols = self.model.shown_columns()
        m = QMenu(self)
        if 0 <= i < len(cols):
            col = cols[i]
            m.addAction(f"Hide {self.model.title(col.key)}", lambda: self.model.set_shown(col.key, False))
            if col.source == "joined":
                label = col.key[len("join:"):].split("::", 1)[0]
                m.addAction(f"Remove the columns joined from {label}", lambda: self.remove_join(label))
            if col.source == "formula":
                m.addAction("Edit formula...", lambda: self.edit_formula(col.alias))
                m.addAction("Remove formula column", lambda: self.remove_formula(col.alias))
            m.addSeparator()
        m.addAction("Columns...", lambda: self.choose_columns())
        m.addAction("New column...", lambda: self.edit_formula())
        m.exec(hh.mapToGlobal(pos))

    def choose_columns(self, modal: bool = True):
        from .table_dialogs import ColumnChooser

        dlg = ColumnChooser(self.model, self)
        if modal:
            dlg.exec()
        return dlg

    def edit_formula(self, name: str | None = None, modal: bool = True):
        from .table_dialogs import FormulaDialog

        dlg = FormulaDialog(self.model, name, self)
        if modal:
            dlg.exec()
        return dlg

    def remove_formula(self, name: str) -> None:
        if QMessageBox.question(self, "Remove formula column", f"Remove the formula column {name}?") == \
                QMessageBox.StandardButton.Yes:
            self.model.remove_formula(name)

    def open_fit(self, modal: bool = True):
        from .table_dialogs import FitDialog

        dlg = FitDialog(self.model, self)
        if modal:
            dlg.exec()
        return dlg

    # ------------------------------------------------------------------ feature builder (analysis.features)
    def _fill_features_menu(self) -> None:
        m = self.features_menu
        m.clear()
        for group in (feat.GROUPS, feat.RATIOS):
            m.addSection(group)
            names = [p.name for p in feat.PRESETS if p.group == group]
            for p in (feat.BY_NAME[n] for n in names):
                why = self.model.feature_status(p.name)
                a = m.addAction(f"{p.label}    [{p.name}]", lambda n=p.name: self.add_features([n]))
                a.setEnabled(not why)
                a.setToolTip(why or p.note)
            if group == feat.GROUPS:
                todo = [n for n in names if not self.model.feature_status(n)]
                a = m.addAction("All of these", lambda t=todo: self.add_features(t))
                a.setEnabled(bool(todo))
        m.addSection("Target")
        a = m.addAction("Reduced target σ* = σ·v_m^⅔/kT of a column...", lambda: self.reduced_target_dialog())
        a.setToolTip("A formula column σ* = <column> / (kT/v_m^⅔): fit or model σ* instead of σ, then multiply back")

    def add_features(self, names) -> list:
        """The presets ``names`` as formula columns (a message when an input is missing)."""
        try:
            cols = self.model.add_features(names)
        except (feat.FeatureError, fm.FormulaError) as exc:
            QMessageBox.information(self, "Features", str(exc))
            return []
        if cols:
            self.win.statusBar().showMessage("Added " + ", ".join(c.alias for c in cols), 8000)
        return cols

    def reduced_target_dialog(self, target: str | None = None) -> list:
        """Ask for the target column (unless ``target``, an alias, is given) and add its σ*."""
        from PySide6.QtWidgets import QInputDialog

        if target is None:
            cols = [c for c in self.model.numeric_columns() if c.source in ("data", "field", "joined", "formula")]
            if not cols:
                QMessageBox.information(self, "Reduced target", "The table has no numeric column yet.")
                return []
            names = [c.alias for c in cols]
            guess = next((i for i, a in enumerate(names) if any(w in a.lower() for w in
                                                               ("sigma", "surface_tension", "st25", "gamma"))), 0)
            target, ok = QInputDialog.getItem(self, "Reduced target", "Surface tension column (mN/m):", names,
                                              guess, False)
            if not ok:
                return []
        try:
            cols = self.model.add_reduced_target(target)
        except (feat.FeatureError, fm.FormulaError) as exc:
            QMessageBox.information(self, "Reduced target", str(exc))
            return []
        self.win.statusBar().showMessage("Added " + ", ".join(c.alias for c in cols), 8000)
        return cols

    # ------------------------------------------------------------------ joined files
    def join_dialog(self, path: str | None = None, modal: bool = True):
        from .table_dialogs import JoinDialog

        if path is None:
            base = Path(getattr(self.win, "run_folder", None) or self.win.s.output_path())
            path, _ = QFileDialog.getOpenFileName(self, "Join columns from a CSV file", str(base),
                                                  "CSV (*.csv *.tsv *.txt);;All files (*)")
            if not path:
                return None
        dlg = JoinDialog(self, path)
        if modal:
            dlg.exec()
        return dlg

    def join_csv(self, path, id_column: str, columns=None, label: str = "") -> int:
        """Join ``columns`` (None: all) of the CSV at ``path`` matched on ``id_column``; the rows matched."""
        spec = JoinSpec(str(Path(path).expanduser()), id_column, list(columns or ()), label)
        got = sheet.read_join(spec)
        n = self.model.add_join(spec, got)
        self.win.statusBar().showMessage(f"Joined {len(got.block.names)} column(s) of {spec.label}: {n:,} of "
                                         f"{self.model.total_rows():,} rows matched", 10000)
        return n

    def remove_join(self, label: str) -> None:
        self.model.remove_join(label)

    # ------------------------------------------------------------------ export
    def export_dialog(self, kind: str) -> None:
        if kind == "xlsx" and not sheet.xlsx_available():
            QMessageBox.information(self, "Export", XLSX_MISSING)
            return
        ext = ".xlsx" if kind == "xlsx" else ".csv"
        flt = "Excel workbook (*.xlsx)" if kind == "xlsx" else "CSV (*.csv)"
        base = Path(getattr(self.win, "run_folder", None) or self.win.s.output_path())
        path, _ = QFileDialog.getSaveFileName(self, "Export table", str(base / f"table{ext}"), flt)
        if path:
            if not path.lower().endswith(ext):
                path += ext
            self.export(path, kind)

    def export(self, path, kind: str | None = None, background: bool = True) -> None:
        """Write the rows and columns shown to ``path`` (CSV or XLSX by ``kind`` or the extension), on a worker
        thread (``background``); ``last_export`` is set when the file is written."""
        from . import workers

        path = Path(path)
        kind = kind or ("xlsx" if path.suffix.lower() == ".xlsx" else "csv")
        headers, rows = self.model.export_rows()
        self.last_export = None
        if kind == "xlsx":
            job = (sheet.write_xlsx, path, headers, rows)
        else:
            job = (sheet.write_csv, path, headers, rows)

        def done(out) -> None:
            self.last_export = Path(out)
            self.win.statusBar().showMessage(f"Saved {len(rows):,} rows to {out}", 10000)

        def failed(msg) -> None:
            QMessageBox.warning(self, "Export", f"Could not write {path}: {msg}")

        if not background:
            done(job[0](*job[1:]))
            return
        workers.start(workers.FunctionWorker(*job), self.win, on_done=done, on_failed=failed)

    # ------------------------------------------------------------------ state
    def state(self) -> TableState:
        st = self.model.state()
        st.plot = dict(self.plot_settings)
        return st

    def restore(self, state: TableState) -> None:
        self.plot_settings = dict(state.plot)
        joins = []
        for spec in state.joins:              # small files: read here, so formula columns that use them resolve
            try:
                joins.append((spec, sheet.read_join(spec)))
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                joins.append((spec, f"{type(exc).__name__}: {exc}"))
        self.model.restore(state, joins)

    def save_state(self) -> None:
        store_state(self.state())
        self.stateSaved.emit()


def load_state() -> TableState:
    """The table's state from the window settings (``STATE_KEY``)."""
    from .main_window import _window_settings

    return TableState.from_json(_window_settings().value(STATE_KEY))


def store_state(state: TableState) -> None:
    from .main_window import _window_settings

    qs = _window_settings()
    qs.setValue(STATE_KEY, state.to_json())
    qs.sync()


def _menu(win, title: str):
    for action in win.menuBar().actions():
        if action.menu() is not None and action.text().replace("&", "") == title:
            return action.menu()
    return None


def add_view_toggle(win, dock: QDockWidget, text: str, tip: str) -> None:
    """The dock's toggle in the View menu, after the Spectra (else Results) toggle."""
    toggle = dock.toggleViewAction()
    toggle.setText(text)
    toggle.setToolTip(tip)
    view = _menu(win, "View")
    if view is None:
        return
    actions = view.actions()
    after = None
    for d in (getattr(win, "plot_dock", None), getattr(win, "table_dock", None), getattr(win, "spectra_dock", None),
              getattr(win, "results_dock", None)):
        if isinstance(d, QDockWidget) and d is not dock and d.toggleViewAction() in actions:
            after = d.toggleViewAction()
            break
    idx = actions.index(after) if after is not None else -1
    if 0 <= idx < len(actions) - 1:
        view.insertAction(actions[idx + 1], toggle)
    else:
        view.addAction(toggle)


def install(win) -> None:
    panel = TablePanel(win)
    dock = QDockWidget("Table", win)
    dock.setObjectName("table")
    dock.setWidget(panel)
    win.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
    dock.hide()                                   # the workspaces show it (Analyze)
    # the table follows the molecule list only while it is in view (a batch in Calculate costs it nothing)
    dock.visibilityChanged.connect(lambda on: panel.model.set_active("table", on))
    add_view_toggle(win, dock, "Table", "Molecules x properties: descriptors, formula columns, export")
    win.table_dock = dock
    win.table_panel = panel
