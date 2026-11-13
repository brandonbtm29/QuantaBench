"""The Analyze table's data, without Qt (UI-3).

The table (``gui.table_panel``) shows one row per molecule of the Molecules dock: the light row fields the list
already has (id, name, status, formula, conformers, best energy, method), the data fields an imported file posted,
the molecule's summary and descriptor values, and formula columns (``analysis.formula``). The values beyond the
light row live here, by column, so 14,000 molecules with a few hundred columns stay a few tens of MB:

    ColumnStore      values per molecule id: a float64 array per numeric column, an object array per text column
    Block            a table read off the GUI thread (ids + one array per column), merged into a store in one go
    result_values    what the table keeps of a built ``MoleculeResult``: its summary row and descriptors
    read_run_tables  a run folder's summary.csv and descriptors.csv, and the job table for built molecules those
                     two files miss (an interrupted or running batch writes them only at its end)
    write_csv, write_xlsx, tsv   exports (UTF-8 with a byte-order mark, so Excel reads µ and Å) and the clipboard
    JoinSpec, read_join, csv_columns
                     columns joined from a CSV file by id (measured targets that are not fields of the input, e.g.
                     ``pubchem targets`` output): which file, which column holds the ids, which columns to bring in
    TableState       the user's column choices, formula columns, joined files and plot settings as JSON: in the
                     window settings until the project document (UI-2) exists, then in the project

A value in a summary row or a descriptors dict is a number (int or float), text, or missing (None, "", NaN). A
column whose values are all numbers is numeric; one text value makes the column text (CSV cells are parsed as
numbers where they can be). Names that duplicate a row field (``ROW_NAMES``) are left out: the row has them.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: summary/descriptor names the table's own row fields already show (or long text that does not belong in a cell)
ROW_NAMES = frozenset(("id", "name", "status", "error", "input_smiles", "method", "n_conformers", "best_energy_eV",
                       "formula", "notes", "phase_notes"))
#: descriptor and summary columns shown before the user picks any (the others are in the column chooser)
DEFAULT_SHOWN = ("mol_weight", "formal_charge", "n_fluorine", "tpsa", "logp_crippen", "dipole_debye",
                 "sasa_A2_best", "vdw_volume_A3_best", "polarizability_bosque_A3", "xtb_polarizability_iso_A3",
                 "qm_polarizability_iso_A3", "molar_volume_packing_cm3", "density_packing_g_cm3", "parachor_sugden")


def classify(v) -> tuple[str | None, object]:
    """("num", float) | ("txt", str) | (None, None) for a missing value or one that is not a scalar."""
    if v is None:
        return None, None
    if isinstance(v, (bool, np.bool_)):
        return "txt", str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return (None, None) if math.isnan(x) else ("num", x)
    if isinstance(v, str):
        return (None, None) if not v.strip() else ("txt", v)
    return None, None


def parse_cell(text):
    """A CSV cell: a float where it reads as one ("1.5", "-2e-3", "inf"), None when empty or "nan", else the text."""
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    t = text.strip()
    if not t or t.lower() == "nan":
        return None
    if "_" not in t:                       # float() reads "1_000"; a cell with an underscore is text
        try:
            return float(t)
        except ValueError:
            pass
    return text


# ============================================================================================ blocks
@dataclass
class Block:
    """``ids`` and one array per column (float64 with NaN for missing, or object with None), in ``names`` order."""

    ids: list[str]
    columns: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.columns)

    @classmethod
    def from_rows(cls, ids: Sequence[str], rows: Sequence[Mapping], parse: bool = True) -> Block:
        """A block from one dict per id (CSV cells are parsed as numbers where they can be with ``parse``)."""
        names: dict[str, None] = {}
        for r in rows:
            for k in r:
                if k and k not in ROW_NAMES:
                    names.setdefault(k, None)
        n = len(ids)
        cols: dict[str, np.ndarray] = {}
        for name in names:
            vals = []
            numeric = True
            for r in rows:
                v = r.get(name)
                if parse and isinstance(v, str):
                    v = parse_cell(v)
                kind, v = classify(v)
                if kind == "txt":
                    numeric = False
                vals.append(v)
            if numeric:
                arr = np.full(n, np.nan)
                for i, v in enumerate(vals):
                    if v is not None:
                        arr[i] = v
            else:
                arr = np.empty(n, dtype=object)
                arr[:] = vals
            cols[name] = arr
        return cls(ids=list(ids), columns=cols)


def result_values(res) -> dict:
    """The table's values of a ``MoleculeResult``: its summary row and descriptors (``ROW_NAMES`` left out;
    a descriptor wins over a summary column of the same name)."""
    try:
        row = res.summary_row()
    except Exception:  # noqa: BLE001 - a stand-in result without the fields summary_row needs
        row = {}
    out = {k: v for k, v in row.items() if k not in ROW_NAMES}
    out.update({k: v for k, v in (getattr(res, "descriptors", None) or {}).items() if k not in ROW_NAMES})
    return out


def read_run_tables(run_dir: str | Path) -> Block:
    """A run folder's values (worker thread): summary.csv, then descriptors.csv on top, then, for built molecules
    descriptors.csv does not list, their summary and descriptors from the job table. Plain Python and NumPy."""
    from ..chem import jobs

    run_dir = Path(run_dir).expanduser()
    by_id: dict[str, dict] = {}
    have: set[str] = set()
    for name in ("summary.csv", "descriptors.csv"):
        path = run_dir / name
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                ident = (row.get("id") or "").strip()
                if not ident:
                    continue
                by_id.setdefault(ident, {}).update({k: v for k, v in row.items() if k and k not in ROW_NAMES})
                if name == "descriptors.csv":
                    have.add(ident)
    if (run_dir / jobs.DB_NAME).is_file():
        try:
            stored = jobs.stored_tables(run_dir, skip=have)
        except Exception:  # noqa: BLE001 - a locked or damaged job table: the CSVs are what there is
            stored = []
        for ident, row, desc in stored:
            d = by_id.setdefault(ident, {})
            for src in (row, desc):
                d.update({k: v for k, v in src.items() if k not in ROW_NAMES})
    ids = list(by_id)
    return Block.from_rows(ids, [by_id[i] for i in ids])


# ============================================================================================ store
class ColumnStore:
    """Values by column for any number of molecule ids (module docstring). ``version`` grows with every change,
    so a view can tell whether what it cached is still current."""

    def __init__(self):
        self._slot: dict[str, int] = {}
        self._used = 0                        # slots handed out (pruned ones are not reused until ``compact``)
        self._cap = 0
        self._num: dict[str, np.ndarray] = {}
        self._txt: dict[str, np.ndarray] = {}
        self._names: dict[str, None] = {}     # column order: first seen first
        self.version = 0

    # -- what is in it
    def __len__(self) -> int:
        return len(self._slot)

    def __contains__(self, ident) -> bool:
        return ident in self._slot

    def names(self) -> list[str]:
        return list(self._names)

    def is_numeric(self, name: str) -> bool:
        return name in self._num

    def slots(self, ids: Iterable[str]) -> np.ndarray:
        get = self._slot.get
        return np.fromiter((get(i, -1) for i in ids), dtype=np.int64)

    def column(self, name: str, slots: np.ndarray) -> np.ndarray:
        """Values of ``name`` at ``slots`` (``slots()``; -1: NaN or None), as a new array."""
        slots = np.asarray(slots, dtype=np.int64)
        ok = slots >= 0
        if name in self._num:
            out = np.full(len(slots), np.nan)
            out[ok] = self._num[name][slots[ok]]
            return out
        out = np.empty(len(slots), dtype=object)
        if name in self._txt:
            out[ok] = self._txt[name][slots[ok]]
        return out

    def get(self, ident: str, name: str):
        s = self._slot.get(ident)
        if s is None:
            return None
        if name in self._num:
            v = float(self._num[name][s])
            return None if math.isnan(v) else v
        if name in self._txt:
            return self._txt[name][s]
        return None

    # -- changes
    def _grow(self, need: int) -> None:
        if need <= self._cap:
            return
        cap = max(64, self._cap)
        while cap < need:
            cap *= 2
        for k, a in self._num.items():
            new = np.full(cap, np.nan)
            new[:self._cap] = a
            self._num[k] = new
        for k, a in self._txt.items():
            new = np.empty(cap, dtype=object)
            new[:self._cap] = a
            self._txt[k] = new
        self._cap = cap

    def _slot_of(self, ident: str) -> int:
        s = self._slot.get(ident)
        if s is None:
            self._grow(self._used + 1)
            s = self._slot[ident] = self._used
            self._used += 1
        return s

    def _ensure(self, name: str, numeric: bool) -> None:
        if name in self._num:
            if not numeric:
                self._to_text(name)
            return
        if name in self._txt:
            return
        self._names.setdefault(name, None)
        if numeric:
            self._num[name] = np.full(self._cap, np.nan)
        else:
            self._txt[name] = np.empty(self._cap, dtype=object)

    def _to_text(self, name: str) -> None:
        arr = self._num.pop(name)
        t = np.empty(self._cap, dtype=object)
        for i in np.flatnonzero(~np.isnan(arr)):
            t[i] = float(arr[i])
        self._txt[name] = t

    def set_row(self, ident: str, values: Mapping, replace: bool = True) -> None:
        """Store ``values`` (name -> number, text or missing) for ``ident``; ``replace`` clears its other values
        first (a rebuilt molecule)."""
        s = self._slot_of(ident)
        if replace:
            for a in self._num.values():
                a[s] = np.nan
            for a in self._txt.values():
                a[s] = None
        for k, v in values.items():
            if not k or k in ROW_NAMES:
                continue
            kind, v = classify(v)
            if kind is None:
                continue
            self._ensure(k, kind == "num")
            (self._num if k in self._num else self._txt)[k][s] = v
        self.version += 1

    def clear_row(self, ident: str) -> None:
        if ident in self._slot:
            self.set_row(ident, {}, replace=True)

    def merge(self, block: Block) -> None:
        """Store every value of ``block`` (its ids' values for its columns; missing cells clear older values)."""
        if not block.ids:
            return
        slots = np.fromiter((self._slot_of(i) for i in block.ids), dtype=np.int64, count=len(block.ids))
        for name, arr in block.columns.items():
            numeric = arr.dtype != object
            self._ensure(name, numeric)
            if name in self._num:
                self._num[name][slots] = arr
            else:
                t = self._txt[name]
                if numeric:
                    for s, v in zip(slots, arr):
                        t[s] = None if math.isnan(v) else float(v)
                else:
                    t[slots] = arr
        self.version += 1

    def prune(self, keep: Iterable[str]) -> int:
        """Forget every id not in ``keep`` (molecules removed from the list); returns how many."""
        keep = set(keep)
        gone = [i for i in self._slot if i not in keep]
        if not gone:
            return 0
        for i in gone:
            s = self._slot.pop(i)
            for a in self._num.values():
                a[s] = np.nan
            for a in self._txt.values():
                a[s] = None
        if len(self._slot) < self._used // 2:
            self.compact()
        self._drop_empty_columns()
        self.version += 1
        return len(gone)

    def compact(self) -> None:
        ids = list(self._slot)
        old = np.array([self._slot[i] for i in ids], dtype=np.int64)
        cap = max(64, 1 << max(0, len(ids) - 1).bit_length())
        for k, a in self._num.items():
            new = np.full(cap, np.nan)
            new[:len(ids)] = a[old]
            self._num[k] = new
        for k, a in self._txt.items():
            new = np.empty(cap, dtype=object)
            new[:len(ids)] = a[old]
            self._txt[k] = new
        self._slot = {i: n for n, i in enumerate(ids)}
        self._used = len(ids)
        self._cap = cap

    def _drop_empty_columns(self) -> None:
        live = np.fromiter(self._slot.values(), dtype=np.int64, count=len(self._slot))
        for name in list(self._names):
            if name in self._num:
                empty = bool(np.isnan(self._num[name][live]).all())
            else:
                empty = all(v is None for v in self._txt[name][live])
            if empty:
                self._names.pop(name)
                self._num.pop(name, None)
                self._txt.pop(name, None)

    def clear(self) -> None:
        self.__init__()


# ============================================================================================ export
def cell_text(v) -> str:
    """A value as export and clipboard text: numbers with 15 significant digits (what Excel keeps), "" for missing."""
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if math.isnan(v) else format(float(v), ".15g")
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    return str(v)


def tsv(rows: Sequence[Sequence], headers: Sequence[str] | None = None) -> str:
    """Tab-separated text for the clipboard (Excel, LibreOffice, Origin paste it into cells)."""
    def clean(s: str) -> str:
        return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")

    lines = []
    if headers is not None:
        lines.append("\t".join(clean(str(h)) for h in headers))
    lines += ["\t".join(clean(cell_text(v)) for v in r) for r in rows]
    return "\n".join(lines) + ("\n" if lines else "")


def write_csv(path: str | Path, headers: Sequence[str], rows: Iterable[Sequence]) -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(list(headers))
        for r in rows:
            w.writerow([cell_text(v) for v in r])
    return path


def xlsx_available() -> bool:
    from importlib.util import find_spec

    return find_spec("openpyxl") is not None


def write_xlsx(path: str | Path, headers: Sequence[str], rows: Iterable[Sequence], sheet: str = "QuantaBench") -> Path:
    """An Excel workbook (openpyxl, imported here: optional). Numbers stay numbers; text that starts with "=" is
    written as text, not as an Excel formula."""
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell

    path = Path(path)
    wb = Workbook(write_only=True)
    ws = wb.create_sheet(sheet[:31] or "Sheet1")
    ws.freeze_panes = "A2"

    def cell(v):
        if v is None or (isinstance(v, (float, np.floating)) and not math.isfinite(v)):
            return None
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, str) and v.startswith("="):
            c = WriteOnlyCell(ws, value=v)
            c.data_type = "s"
            return c
        return v

    ws.append([cell(str(h)) for h in headers])
    for r in rows:
        ws.append([cell(v) for v in r])
    wb.save(path)
    return path


# ============================================================================================ joins
def norm_id(ident) -> str:
    """How joined ids are matched to the table's: stripped, case-folded ("  LiTFSI " matches "litfsi")."""
    return str(ident).strip().casefold()


def _dialect(sample: str):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def _open_csv(path: str | Path):
    path = Path(path).expanduser()
    f = open(path, newline="", encoding="utf-8-sig")      # noqa: SIM115 - closed by the caller's ``with``
    sample = f.read(16384)
    f.seek(0)
    return f, _dialect(sample)


def csv_columns(path: str | Path) -> list[str]:
    """The header of a CSV file (comma, semicolon or tab separated; UTF-8 with or without a byte-order mark)."""
    f, dialect = _open_csv(path)
    with f:
        header = next(csv.reader(f, dialect), [])
    return [h.strip() for h in header if h is not None]


@dataclass
class JoinSpec:
    """Columns joined from a CSV file: ``columns`` of ``path`` (empty: all but the id column), matched on
    ``id_column`` against the molecules' ids (``norm_id``). ``label`` names the file in the table (its stem)."""

    path: str
    id_column: str
    columns: list[str] = field(default_factory=list)
    label: str = ""

    def __post_init__(self):
        self.label = self.label or Path(self.path).stem


@dataclass
class JoinRead:
    block: Block                       # ids as norm_id, one array per joined column
    rows: int                          # data rows read
    duplicates: int                    # rows whose id came again (the last one wins)
    no_id: int                         # rows without an id


def read_join(spec: JoinSpec) -> JoinRead:
    """The joined columns of ``spec`` (any thread): numbers where the cells read as numbers (``parse_cell``)."""
    f, dialect = _open_csv(spec.path)
    by_id: dict[str, dict] = {}
    rows = dup = no_id = 0
    with f:
        reader = csv.DictReader(f, dialect=dialect)
        fields = [h.strip() if h else h for h in (reader.fieldnames or [])]
        reader.fieldnames = fields
        if spec.id_column not in fields:
            raise ValueError(f"{spec.path}: no column {spec.id_column!r} (columns: {', '.join(map(str, fields))})")
        want = [c for c in (spec.columns or fields) if c and c != spec.id_column]
        missing = [c for c in want if c not in fields]
        if missing:
            raise ValueError(f"{spec.path}: no column {', '.join(map(repr, missing))}")
        for row in reader:
            rows += 1
            ident = norm_id(row.get(spec.id_column) or "")
            if not ident:
                no_id += 1
                continue
            if ident in by_id:
                dup += 1
            by_id[ident] = {c: row.get(c) for c in want}
    ids = list(by_id)
    block = Block.from_rows(ids, [by_id[i] for i in ids])
    for c in want:                       # a column with no value at all is still a (numeric, empty) column
        if c not in block.columns:
            block.columns[c] = np.full(len(ids), np.nan)
    return JoinRead(block, rows, dup, no_id)


# ============================================================================================ state
@dataclass
class FormulaSpec:
    name: str
    expression: str
    note: str = ""                     # e.g. what a fit reported when it made this column


@dataclass
class TableState:
    """What the user set up in the table, as plain data (``to_json`` / ``from_json``)."""

    VERSION = 1
    columns: dict[str, bool] = field(default_factory=dict)       # column key -> shown (the user's choices only)
    formulas: list[FormulaSpec] = field(default_factory=list)
    plot: dict = field(default_factory=dict)                     # x, y, color (column keys), fit, parity (bool)
    joins: list[JoinSpec] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"version": self.VERSION, "columns": self.columns,
                           "formulas": [asdict(f) for f in self.formulas], "plot": self.plot,
                           "joins": [asdict(j) for j in self.joins]}, sort_keys=True)

    @classmethod
    def from_json(cls, text) -> TableState:
        """The state in ``text``; anything unreadable (another version, a hand edit) gives the defaults for that
        part, never an exception."""
        try:
            d = json.loads(text) if text else {}
        except (TypeError, ValueError):
            d = {}
        if not isinstance(d, dict) or d.get("version") not in (None, cls.VERSION):
            d = {}
        cols = d.get("columns") if isinstance(d.get("columns"), dict) else {}
        formulas = []
        for f in d.get("formulas") or ():
            if isinstance(f, dict) and isinstance(f.get("name"), str) and isinstance(f.get("expression"), str):
                formulas.append(FormulaSpec(f["name"], f["expression"], str(f.get("note") or "")))
        plot = d.get("plot") if isinstance(d.get("plot"), dict) else {}
        joins = []
        for j in d.get("joins") or ():
            if isinstance(j, dict) and isinstance(j.get("path"), str) and isinstance(j.get("id_column"), str):
                cols_j = [str(c) for c in j.get("columns") or () if isinstance(c, str)]
                joins.append(JoinSpec(j["path"], j["id_column"], cols_j, str(j.get("label") or "")))
        return cls(columns={str(k): bool(v) for k, v in cols.items()}, formulas=formulas, plot=dict(plot),
                   joins=joins)
