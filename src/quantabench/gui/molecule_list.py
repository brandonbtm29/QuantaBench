"""The molecule list at 14,000 molecules (SCALE-1).

Spartan kept every molecule of a document rendered in memory although only one is viewed at a time, which
cut the PFAS QSAR set into 200-molecule documents. Here the list holds one light ``MoleculeRow`` per molecule
(id, name, status, method, conformer count, best energy, where its files are: well under 1 KB), shown by a
virtualized ``QTableView`` over ``MoleculeTableModel``; only the selected molecule and the last few viewed
keep their full result (RDKit molecule with every conformer, descriptors, vibrations) in memory, read back
from their folder when selected (``ResultStore``, an LRU cache).

    MoleculeTableModel   rows + batched inserts (one beginInsertRows per batch); sort() in Python by a key; data
                         fields posted on import (File > Open's Import Options) as extra columns after the fixed ones
    MoleculeFilter       search (id, name, SMILES, formula) and status filter; sorting goes to the source
    MoleculeListView     the table, with the few QListWidget-style helpers the window and tests use
    ResultStore          ``win.results``: results by id (in memory only for imports; the others from disk)
    read_run_folder      rows of a run folder from its job table (or its summary.csv)

Nothing here creates a QObject off the GUI thread: ``read_run_folder`` and ``load_result`` run on worker
threads (``gui.workers.FunctionWorker``) and return plain Python objects.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView

from .. import units

STATUS_ICON = {"ok": "✔", "failed": "✖", "cancelled": "○", "pending": "…", "running": "▶"}
FINISHED = ("ok", "failed", "cancelled")
#: (attribute, header, default width): the "#" column is the input order (sorting by it restores it); the energy
#: header gets the display unit (Settings > Display): "E (kJ/mol)"
COLUMNS = (("seq", "#", 46), ("status", "", 24), ("id", "ID", 120), ("conformers", "Conf.", 44),
           ("energy", "E", 96), ("name", "Name", 150), ("method", "Method", 90), ("formula", "Formula", 90))
COL = {key: i for i, (key, _h, _w) in enumerate(COLUMNS)}
SORT_ROLE = Qt.ItemDataRole.UserRole + 1
STATUS_FILTERS = (("", "All"), ("ok", "Built"), ("failed", "Failed"), ("todo", "Not built"))


@dataclass(slots=True)
class MoleculeRow:
    """What the list knows about one molecule without loading it."""

    id: str
    name: str = ""
    smiles: str = ""
    source: str = ""
    status: str = "pending"        # pending | ok | failed | cancelled (the job table's statuses)
    stage: str = ""                # while it runs: "queued", "conformers", ... ("" = not running)
    method: str = ""
    n_conformers: int | None = None
    energy_ev: float | None = None
    formula: str = ""
    error: str = ""
    folder: str = ""               # the molecule's result folder ("" = nothing on disk)
    run_dir: str = ""              # run folder whose job table holds the input (rows of an opened folder)
    seq: int = 0                   # position in the list as added (input order)
    input: object = None           # MoleculeInput, in memory for molecules added here (None: in run_dir)
    fields: dict | None = None     # data fields posted on import (SD properties, table columns): extra columns

    def text(self) -> str:
        """One line, as the old list showed it: icon, id, method and conformer count (stage while running)."""
        if self.stage:
            return f"{STATUS_ICON['running']} {self.id}  ({self.stage})"
        extra = f"  {self.method}, {self.n_conformers or 0} conf." if self.status == "ok" else ""
        return f"{STATUS_ICON.get(self.status, STATUS_ICON['pending'])} {self.id}{extra}"

    def matches(self, needle: str) -> bool:
        return (needle in self.id.lower() or needle in self.name.lower() or needle in self.smiles.lower()
                or needle in self.formula.lower())


def posted_fields(inp) -> dict | None:
    """The data fields an input brought from its file (``import_options.FIELDS``), or None."""
    from ..chem.import_options import FIELDS

    d = (getattr(inp, "meta", None) or {}).get(FIELDS)
    return dict(d) if isinstance(d, dict) and d else None


def _field_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _field_key(name: str):
    """Sort key of a data-field column: numbers in order, then text, then rows without the field."""
    def key(r):
        v = (r.fields or {}).get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return (0, float(v), "", r.seq)
        if v is None or v == "":
            return (2, 0.0, "", r.seq)
        return (1, 0.0, str(v).lower(), r.seq)
    return key


def result_folder(res) -> str:
    """Folder that holds ``res``'s files (where ``export.write_molecule`` put its summary), or ""."""
    for p in res.files or ():
        p = Path(p)
        if p.name == f"{res.input.id}_summary.json" and p.is_file():
            return str(p.parent)
    return ""


def row_updates(res) -> dict:
    """Row fields from a finished ``MoleculeResult`` (what the list keeps of it)."""
    best = res.conformers[0] if res.conformers else None
    return dict(status=res.status if res.status in FINISHED else "pending", stage="", method=res.method or "",
                n_conformers=len(res.conformers) if res.status == "ok" else None,
                energy_ev=best.energy_ev if best is not None else None,
                formula=str((res.descriptors or {}).get("formula") or ""), error=res.error or "",
                folder=result_folder(res), smiles=res.input.smiles or "", name=res.input.name or "")


def _num(value, kind=float):
    try:
        return kind(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def read_run_folder(run_dir: str | Path):
    """(rows, ``jobs.RunStatus`` or None) of a run folder: from its job table, or, for a folder written by a
    plain (non-resumable) batch, from its summary.csv. Plain Python only: safe on a worker thread."""
    from ..chem import jobs

    run_dir = Path(run_dir).expanduser()
    rows: list[MoleculeRow] = []
    if (run_dir / jobs.DB_NAME).is_file():
        status = jobs.open_run(run_dir)
        for d in jobs.molecule_rows(run_dir):
            st = d["status"] if d["status"] in FINISHED else "pending"
            rows.append(MoleculeRow(
                id=d["id"], name=d["name"] or "", smiles=d["smiles"] or "", source=d["source"] or "", status=st,
                method=d["method"] or "", n_conformers=_num(d["n_conformers"], int) if st == "ok" else None,
                energy_ev=_num(d["best_energy_eV"]), formula=d["formula"] or "", error=d["error"] or "",
                folder=str(run_dir / d["result_dir"]) if d["result_dir"] else "", run_dir=str(run_dir),
                seq=d["seq"]))
        return rows, status
    path = run_dir / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} is not a QuantaBench run folder (no {jobs.DB_NAME} or summary.csv)")
    with open(path, newline="", encoding="utf-8") as f:
        for k, d in enumerate(csv.DictReader(f)):
            st = d.get("status") if d.get("status") in FINISHED else "pending"
            ident = d.get("id") or f"row{k + 1}"
            rows.append(MoleculeRow(
                id=ident, name=d.get("name") or "", smiles=d.get("input_smiles") or d.get("smiles") or "",
                status=st, method=d.get("method") or "", n_conformers=_num(d.get("n_conformers"), int),
                energy_ev=_num(d.get("best_energy_eV")), error=d.get("error") or "",
                folder=str(run_dir / ident) if st == "ok" else "", seq=k))
    return rows, None


def gather_inputs(specs: list[tuple]) -> list:
    """Inputs for a build of list rows (worker thread). ``specs``: (id, run_dir, input, smiles, name, source);
    rows of an opened run folder have no input in memory and are read from its job table, one query per folder."""
    import sqlite3

    from ..chem import jobs
    from ..chem.inputs import MoleculeInput

    by_dir: dict[str, list[str]] = {}
    for ident, run_dir, inp, *_rest in specs:
        if inp is None and run_dir:
            by_dir.setdefault(run_dir, []).append(ident)
    stored = {}
    for run_dir, ids in by_dir.items():
        try:
            stored.update({x.id: x for x in jobs.load_inputs(run_dir, ids)})
        except (OSError, sqlite3.Error):
            pass
    return [inp or stored.get(ident) or MoleculeInput(id=ident, smiles=smiles, name=name, source=source)
            for ident, _run_dir, inp, smiles, name, source in specs]


def load_result(ident: str, folder: str, run_dir: str, inp=None):
    """Full ``MoleculeResult`` of one row (worker thread): from its folder, the input from memory or the job
    table; a row without a folder but in a job table comes back from the table (failed: status and error)."""
    from ..chem import export, jobs

    if run_dir and (Path(run_dir) / jobs.DB_NAME).is_file() and (
            not folder or Path(folder).parent == Path(run_dir)):
        return jobs.load_result(run_dir, ident)
    return export.read_molecule(Path(folder), ident, inp)


# ============================================================================================ model
class MoleculeTableModel(QAbstractTableModel):
    """Light rows for any number of molecules. Rows are plain dataclasses; ``_pos`` finds a row by id. Energies are
    kept in eV and shown in the display unit (``set_display``)."""

    def __init__(self, parent=None, display=None):
        super().__init__(parent)
        self._rows: list[MoleculeRow] = []
        self._pos: dict[str, int] = {}
        self._next_seq = 0
        self._energy = units.EnergyDisplay.of(display)
        self._extra: list[str] = []          # data-field columns (posted on import), after COLUMNS

    def extra_columns(self) -> list[str]:
        return list(self._extra)

    @staticmethod
    def _new_fields(rows, known) -> list[str]:
        seen = set(known)
        out = []
        for r in rows:
            for k in r.fields or ():
                if k not in seen:
                    seen.add(k)
                    out.append(k)
        return out

    def set_display(self, display) -> None:
        """Show the energy column in the units of ``display`` (``Settings``, its ``display`` or an EnergyDisplay)."""
        d = units.EnergyDisplay.of(display)
        if d == self._energy:
            return
        self._energy = d
        col = COL["energy"]
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, col, col)
        if self._rows:
            self.dataChanged.emit(self.index(0, col), self.index(len(self._rows) - 1, col))

    # -- Qt
    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: B008, N802 - Qt's signature
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else len(COLUMNS) + len(self._extra)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation == Qt.Orientation.Horizontal and 0 <= section - len(COLUMNS) < len(self._extra):
            if role == Qt.ItemDataRole.DisplayRole:
                return self._extra[section - len(COLUMNS)]
            if role == Qt.ItemDataRole.ToolTipRole:
                return f"{self._extra[section - len(COLUMNS)]}: a data field of the imported file"
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(COLUMNS):
            if role == Qt.ItemDataRole.DisplayRole:
                if COLUMNS[section][0] == "energy":
                    return self._energy.header("E", "absolute")
                return COLUMNS[section][1]
            if role == Qt.ItemDataRole.ToolTipRole:
                return {"seq": "Input order", "status": "Status", "energy": "Energy of the best conformer"}.get(
                    COLUMNS[section][0])
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r = self._rows[index.row()]
        if index.column() >= len(COLUMNS):
            name = self._extra[index.column() - len(COLUMNS)]
            value = (r.fields or {}).get(name)
            if role == Qt.ItemDataRole.DisplayRole:
                return _field_text(value)
            if role == SORT_ROLE:
                return _field_key(name)(r)
            if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float)):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return None
        key = COLUMNS[index.column()][0]
        if role == Qt.ItemDataRole.DisplayRole:
            if key == "seq":
                return str(r.seq + 1)
            if key == "status":
                return STATUS_ICON["running"] if r.stage else STATUS_ICON.get(r.status, STATUS_ICON["pending"])
            if key == "id":
                return r.id
            if key == "conformers":
                return "" if r.n_conformers is None else str(r.n_conformers)
            if key == "energy":
                return self._energy.text(r.energy_ev, units.EV, "absolute", 4)    # "" when None
            if key == "method":
                return r.stage or r.method
            return getattr(r, key)
        if role == SORT_ROLE:
            return self.sort_key(key)(r)
        if role == Qt.ItemDataRole.ToolTipRole:
            bits = [f"{r.id}  {r.name}".strip(), r.smiles]
            if r.stage:
                bits.append(f"running: {r.stage}")
            if r.error:
                bits.append(r.error)
            return "\n".join(b for b in bits if b)
        if role == Qt.ItemDataRole.TextAlignmentRole and key in ("seq", "conformers", "energy"):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.TextAlignmentRole and key == "status":
            return int(Qt.AlignmentFlag.AlignCenter)
        return None

    @staticmethod
    def sort_key(key: str):
        """Key for ``sort``: missing numbers sort last in ascending order."""
        big = float("inf")
        order = {"failed": 0, "cancelled": 1, "pending": 2, "ok": 3}
        return {
            "seq": lambda r: r.seq,
            "status": lambda r: (-1 if r.stage else order.get(r.status, 2), r.seq),
            "id": lambda r: r.id.lower(),
            "name": lambda r: (r.name.lower(), r.seq),
            "conformers": lambda r: (big if r.n_conformers is None else r.n_conformers, r.seq),
            "energy": lambda r: (big if r.energy_ev is None else r.energy_ev, r.seq),
            "method": lambda r: (r.method, r.seq),
            "formula": lambda r: (r.formula, r.seq),
        }[key]

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        """Sort the rows (Python's sort with a key: ~20 ms for 14,000, instead of Qt calling ``data`` for every
        comparison). Column -1 restores the input order. Persistent indexes (the view's selection) follow."""
        if 0 <= column - len(COLUMNS) < len(self._extra):
            sort_key = _field_key(self._extra[column - len(COLUMNS)])
        else:
            sort_key = self.sort_key(COLUMNS[column][0] if 0 <= column < len(COLUMNS) else "seq")
        self.layoutAboutToBeChanged.emit()
        old = self.persistentIndexList()
        ids = [self._rows[i.row()].id for i in old]
        self._rows.sort(key=sort_key, reverse=order == Qt.SortOrder.DescendingOrder and column >= 0)
        self._reindex()
        self.changePersistentIndexList(old, [self.index(self._pos[i], ix.column()) for i, ix in zip(ids, old)])
        self.layoutChanged.emit()

    # -- rows
    def _reindex(self) -> None:
        self._pos = {r.id: i for i, r in enumerate(self._rows)}

    def set_rows(self, rows: Iterable[MoleculeRow]) -> None:
        """Replace every row (one model reset)."""
        self.beginResetModel()
        self._rows = list(rows)
        self._extra = self._new_fields(self._rows, ())
        self._reindex()
        self._next_seq = max((r.seq for r in self._rows), default=-1) + 1
        self.endResetModel()

    def add_rows(self, rows: list[MoleculeRow]) -> None:
        """Append ``rows`` in one batch (one insert signal, however many rows); they get the next input order."""
        if not rows:
            return
        for r in rows:
            r.seq = self._next_seq
            self._next_seq += 1
        new = self._new_fields(rows, self._extra)
        if new:                              # data fields of an imported file: columns after the fixed ones
            first = len(COLUMNS) + len(self._extra)
            self.beginInsertColumns(QModelIndex(), first, first + len(new) - 1)
            self._extra += new
            self.endInsertColumns()
        n = len(self._rows)
        self.beginInsertRows(QModelIndex(), n, n + len(rows) - 1)
        self._rows.extend(rows)
        for i, r in enumerate(rows, n):
            self._pos[r.id] = i
        self.endInsertRows()

    def remove_ids(self, ids: Iterable[str]) -> None:
        positions = sorted({self._pos[i] for i in ids if i in self._pos}, reverse=True)
        while positions:                    # contiguous runs, last first, so earlier positions stay valid
            last = first = positions.pop(0)
            while positions and positions[0] == first - 1:
                first = positions.pop(0)
            self.beginRemoveRows(QModelIndex(), first, last)
            del self._rows[first:last + 1]
            self.endRemoveRows()
        self._reindex()

    def clear(self) -> None:
        self.set_rows([])
        self._next_seq = 0

    def row(self, ident: str) -> MoleculeRow | None:
        i = self._pos.get(ident)
        return self._rows[i] if i is not None else None

    def row_at(self, i: int) -> MoleculeRow:
        return self._rows[i]

    def position(self, ident: str) -> int:
        return self._pos.get(ident, -1)

    def ids(self) -> list[str]:
        return [r.id for r in self._rows]

    def rows(self) -> list[MoleculeRow]:
        return list(self._rows)

    def update(self, ident: str, **fields) -> None:
        i = self._pos.get(ident)
        if i is None:
            return
        r = self._rows[i]
        for k, v in fields.items():
            setattr(r, k, v)
        self.dataChanged.emit(self.index(i, 0), self.index(i, len(COLUMNS) - 1))

    def update_many(self, ids: Iterable[str], **fields) -> None:
        """Same fields on many rows, one ``dataChanged`` for the span they cover."""
        pos = [self._pos[i] for i in ids if i in self._pos]
        if not pos:
            return
        for i in pos:
            for k, v in fields.items():
                setattr(self._rows[i], k, v)
        self.dataChanged.emit(self.index(min(pos), 0), self.index(max(pos), len(COLUMNS) - 1))

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(("ok", "failed", "cancelled", "pending", "running"), 0)
        for r in self._rows:
            out["running" if r.stage else r.status] = out.get("running" if r.stage else r.status, 0) + 1
        return out


class MoleculeFilter(QSortFilterProxyModel):
    """Search and status filter. Sorting is the source model's (``MoleculeTableModel.sort``)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._needle = ""
        self._status = ""

    def _refilter(self) -> None:
        if hasattr(self, "beginFilterChange"):     # Qt >= 6.9: invalidateFilter is deprecated
            self.beginFilterChange()
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        else:
            self.invalidateFilter()

    def set_text(self, text: str) -> None:
        needle = text.strip().lower()
        if needle != self._needle:
            self._needle = needle
            self._refilter()

    def set_status(self, status: str) -> None:
        if status != self._status:
            self._status = status
            self._refilter()

    def filterAcceptsRow(self, source_row: int, _parent) -> bool:  # noqa: N802 - Qt's name
        if not self._needle and not self._status:
            return True
        r = self.sourceModel().row_at(source_row)
        if self._status == "todo":
            if r.status == "ok" or r.status == "failed":
                return False
        elif self._status and r.status != self._status:
            return False
        return not self._needle or r.matches(self._needle)

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        self.sourceModel().sort(column, order)


class MoleculeListView(QTableView):
    """The molecule table. ``currentIdChanged`` fires when the current molecule (not just its row) changes."""

    currentIdChanged = Signal(str)

    def __init__(self, model: MoleculeTableModel, parent=None):
        super().__init__(parent)
        self.source = model
        self.proxy = MoleculeFilter(self)
        self.proxy.setSourceModel(model)
        self.setModel(self.proxy)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setAlternatingRowColors(True)
        vh = self.verticalHeader()
        vh.setVisible(False)
        vh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)     # uniform rows: scrolling is O(visible rows)
        vh.setDefaultSectionSize(self.fontMetrics().height() + 6)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # never ResizeToContents: O(all rows)
        hh.setStretchLastSection(True)
        hh.setHighlightSections(False)
        for i, (_key, _head, width) in enumerate(COLUMNS):
            self.setColumnWidth(i, width)
        hh.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)       # input order until a header is clicked
        self.setSortingEnabled(True)
        self._current = ""
        self.selectionModel().currentRowChanged.connect(self._on_current)
        self.proxy.modelReset.connect(self._on_current)
        self.proxy.layoutChanged.connect(self._on_current)

    def _on_current(self, *_args) -> None:
        ident = self.current_id()
        if ident and ident != self._current:
            self._current = ident
            self.currentIdChanged.emit(ident)
        elif not ident:
            self._current = ""

    # -- QListWidget-style helpers (rows count in the view's order, after search and sort)
    def count(self) -> int:
        return self.proxy.rowCount()

    def currentRow(self) -> int:  # noqa: N802
        ix = self.currentIndex()
        return ix.row() if ix.isValid() else -1

    def setCurrentRow(self, row: int) -> None:  # noqa: N802
        if 0 <= row < self.proxy.rowCount():
            ix = self.proxy.index(row, COL["id"])
            self.selectionModel().setCurrentIndex(
                ix, self.selectionModel().SelectionFlag.ClearAndSelect | self.selectionModel().SelectionFlag.Rows)
            self.scrollTo(ix)

    def row_at(self, row: int) -> MoleculeRow | None:
        if not 0 <= row < self.proxy.rowCount():
            return None
        return self.source.row_at(self.proxy.mapToSource(self.proxy.index(row, 0)).row())

    def row_text(self, row: int) -> str:
        r = self.row_at(row)
        return r.text() if r is not None else ""

    def current_id(self) -> str:
        r = self.row_at(self.currentRow())
        return r.id if r is not None else ""

    def select_id(self, ident: str) -> bool:
        pos = self.source.position(ident)
        if pos < 0:
            return False
        ix = self.proxy.mapFromSource(self.source.index(pos, COL["id"]))
        if not ix.isValid():                         # hidden by the search
            return False
        self.setCurrentRow(ix.row())
        return True

    def selected_ids(self) -> list[str]:
        """Selected molecules in list order."""
        rows = sorted({ix.row() for ix in self.selectionModel().selectedRows()})
        return [r.id for r in (self.row_at(i) for i in rows) if r is not None]


# ====================================================================================== results
class ResultStore(Mapping):
    """``win.results``: finished molecules by id, without thousands of them in memory.

    ``pinned`` results exist only in memory (an imported output file, a run with file writing off); every
    other finished molecule is read from its folder when asked for (``store[id]``, synchronous) or by the
    window's loader (off the GUI thread), and the last ``size`` are kept (LRU). Iterating and ``len`` cover the
    finished molecules of the list, loaded or not."""

    def __init__(self, model: MoleculeTableModel, size: int = 12):
        self.model = model
        self.size = size
        self.pinned: dict[str, object] = {}
        self.cache: OrderedDict[str, object] = OrderedDict()

    def cached(self, ident: str):
        res = self.pinned.get(ident)
        if res is None and ident in self.cache:
            self.cache.move_to_end(ident)
            res = self.cache[ident]
        return res

    def put(self, res) -> None:
        ident = res.input.id
        if ident in self.pinned:
            self.pinned[ident] = res
            return
        self.cache[ident] = res
        self.cache.move_to_end(ident)
        while len(self.cache) > self.size:
            self.cache.popitem(last=False)

    def pin(self, res) -> None:
        self.cache.pop(res.input.id, None)
        self.pinned[res.input.id] = res

    def discard(self, ident: str) -> None:
        self.pinned.pop(ident, None)
        self.cache.pop(ident, None)

    def clear(self) -> None:
        self.pinned.clear()
        self.cache.clear()

    def loadable(self, ident: str) -> bool:
        r = self.model.row(ident)
        return r is not None and r.status in FINISHED and bool(r.folder or r.run_dir) and ident not in self.pinned

    def stand_in(self, r: MoleculeRow):
        """A result made from the row alone (failed molecules: status and error)."""
        from ..chem.inputs import MoleculeInput
        from ..chem.pipeline import MoleculeResult

        inp = r.input or MoleculeInput(id=r.id, smiles=r.smiles, name=r.name, source=r.source)
        return MoleculeResult(input=inp, status=r.status, error=r.error, method=r.method)

    def __getitem__(self, ident: str):
        res = self.cached(ident)
        if res is not None:
            return res
        r = self.model.row(ident)
        if r is None or r.status not in FINISHED:
            raise KeyError(ident)
        if r.folder or r.run_dir:
            try:
                res = load_result(r.id, r.folder, r.run_dir, r.input)
            except (OSError, KeyError, ValueError):
                res = None
            if res is not None:
                self.put(res)
                return res
        return self.stand_in(r)

    def _finished(self) -> list[str]:
        return [r.id for r in self.model.rows() if r.status in FINISHED or r.id in self.pinned]

    def __iter__(self) -> Iterator[str]:
        return iter(self._finished())

    def __len__(self) -> int:
        return len(self._finished())

    def __contains__(self, ident) -> bool:
        r = self.model.row(ident)
        return ident in self.pinned or (r is not None and r.status in FINISHED)
