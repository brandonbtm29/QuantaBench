"""Turn what the user gives us (SMILES, a name, files, a CSV/Excel list) into ``MoleculeInput`` records."""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds

_log = logging.getLogger(__name__)

# Column names are matched case-insensitively. They include the EPA CompTox Chemicals Dashboard export
# headers (DTXSID, PREFERRED_NAME, CASRN, SMILES, QSAR_READY_SMILES, MS_READY_SMILES).
SMILES_COLUMNS = ("smiles", "qsar_ready_smiles", "ms_ready_smiles", "canonical_smiles", "isomeric_smiles")
ID_COLUMNS = ("dtxsid", "id", "name", "compound", "preferred_name", "cas", "casrn")
NAME_COLUMNS = ("preferred_name", "name", "chemical_name", "compound")
TABLE_TYPES = {".csv", ".xlsx", ".xlsm", ".xls"}
#: read here, or in ``chem.formats`` (MOL2, Marvin, ChemDraw, CML, CIF, Gaussian and ORCA inputs); any other extension
#: goes to Open Babel when the user has it (``formats.openbabel``)
STRUCTURE_TYPES = {".sdf", ".sd", ".mol", ".xyz", ".pdb", ".mol2", ".mrv", ".cdx", ".cdxml", ".cml", ".cif", ".gjf",
                   ".com", ".inp"}
LIST_TYPES = {".smi", ".smiles", ".txt", ".inchi"}
FILE_TYPES = STRUCTURE_TYPES | LIST_TYPES | TABLE_TYPES
#: a standard InChIKey: 14 letters, 8 letters + flag + version, protonation letter
INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")

# Which SMILES column to use when a table has several, in order of preference. A row whose preferred
# cell is blank (or does not parse in RDKit) falls back to the next column that has a structure.
#   "smiles": the registered structure as drawn, with salts, stereo and charges intact. The pipeline
#             does its own desalting, neutralization and pH protonation, so this keeps the most
#             information; stereocenters matter for 3D (e.g. the C2 stereocenter of HFPO-DA).
#   "qsar":   QSAR-ready SMILES: desalted, neutralized, stereo and isotopes removed, tautomers
#             normalized (Mansouri et al., J. Cheminform. 2024, 16, 19). Use it to match descriptor
#             tables built on QSAR-ready forms. Without stereo, RDKit embeds an arbitrary stereoisomer.
#   "ms":     MS-ready SMILES: QSAR-ready-like forms with mixtures split into components, made for
#             mass-spectrometry screening (McEachran et al., J. Cheminform. 2018, 10, 45).
SMILES_PREFERENCES: dict[str, tuple[str, ...]] = {
    "smiles": SMILES_COLUMNS,
    "qsar": ("qsar_ready_smiles", "smiles", "ms_ready_smiles", "canonical_smiles", "isomeric_smiles"),
    "ms": ("ms_ready_smiles", "qsar_ready_smiles", "smiles", "canonical_smiles", "isomeric_smiles"),
}
# Placeholders that CompTox exports and spreadsheets use for "no value".
MISSING_VALUES = {"", "n/a", "na", "nan", "none", "null", "-", "#n/a"}


@dataclass
class MoleculeInput:
    id: str
    smiles: str = ""
    name: str = ""
    source: str = ""
    mol: Chem.Mol | None = None      # set when read from a structure file
    charge: int | None = None        # None = take from the structure after protonation
    multiplicity: int = 1
    meta: dict = field(default_factory=dict)

    def to_mol(self) -> Chem.Mol:
        if self.mol is not None:
            return Chem.Mol(self.mol)
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES for {self.id}: {self.smiles!r}")
        return mol


def _ascii_char(ch: str) -> str:
    """One character in ASCII: Greek letters spelled out (beta-Tocopherol), accents dropped (e from é)."""
    if ord(ch) < 128:
        return ch
    m = re.match(r"GREEK (?:SMALL|CAPITAL) LETTER (\w+)$", unicodedata.name(ch, ""))
    if m:
        return m.group(1).lower()
    return unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii") or "_"


def safe_id(text: str, fallback: str = "mol") -> str:
    """Folder- and file-safe ASCII identifier. Never starts with "-" (a folder named "-x" reads as an option to
    command-line tools) or "."."""
    text = "".join(_ascii_char(ch) for ch in unicodedata.normalize("NFKC", text.strip()))
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", text).strip("._-")
    return (s or fallback)[:80]


def looks_like_smiles(text: str) -> bool:
    text = text.strip()
    if not text or " " in text:
        return False
    with rdBase.BlockLogs():         # names are tried as SMILES first; keep RDKit's parse errors quiet
        if any(ch in text for ch in "=#()[]@+\\/") or text.isupper() and len(text) <= 3:
            return Chem.MolFromSmiles(text, sanitize=False) is not None
        # All-letter SMILES (CCCO, OCCO, NCCN, ClCCl, CCCCCCCCCCCCO) parse and sanitize; plain words such as
        # "benzene" or "ethanol" do not (lowercase letters outside b/c/n/o/p/s, or aromatic atoms
        # without a ring), so a sanitizing parse tells them apart.
        return Chem.MolFromSmiles(text) is not None


def from_smiles(smiles: str, id: str | None = None, name: str = "") -> MoleculeInput:
    smiles = smiles.strip()
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"Not a valid SMILES string: {smiles!r}")
    return MoleculeInput(id=safe_id(id or name or smiles), smiles=smiles, name=name or id or "", source="smiles")


def lookup_name(name: str, timeout: float = 15.0) -> MoleculeInput:
    """Resolve a chemical name, CAS number or DTXSID-free identifier through PubChem (``chem.pubchem``: the shared,
    rate-limited and cached client)."""
    from . import pubchem

    try:
        hit = pubchem.lookup(name, timeout=timeout)
    except pubchem.PubChemError as exc:
        raise ValueError(f"PubChem lookup failed for {name!r}: {exc}") from exc
    return MoleculeInput(id=safe_id(name), smiles=hit["smiles"], name=hit["title"], source="pubchem",
                         meta={"pubchem_cid": hit["cid"]})


def from_inchi(text: str, id: str | None = None, name: str = "") -> MoleculeInput:
    """An InChI (``InChI=1S/...``), parsed locally by RDKit. The id is ``id``, else the InChIKey."""
    from rdkit.Chem import inchi

    text = text.strip()
    if not inchi.INCHI_AVAILABLE:
        raise ValueError("RDKit was built without InChI support")
    with rdBase.BlockLogs():
        mol = Chem.MolFromInchi(text, sanitize=True, removeHs=True, logLevel=None, treatWarningAsError=False)
    if mol is None:
        raise ValueError(f"Not a valid InChI: {text[:80]!r}")
    smiles = Chem.MolToSmiles(mol)
    with rdBase.BlockLogs():
        key = inchi.InchiToInchiKey(text) or ""
    return MoleculeInput(id=safe_id(id or name or key or smiles), smiles=smiles, name=name or id or "", source="inchi")


def lookup_inchikey(key: str, timeout: float = 15.0) -> MoleculeInput:
    """Resolve an InChIKey through PubChem (``chem.pubchem``, like names)."""
    from . import pubchem

    key = key.strip()
    try:
        hit = pubchem.lookup(key, timeout=timeout, namespace="inchikey")
    except pubchem.PubChemError as exc:
        raise ValueError(f"PubChem lookup failed for InChIKey {key}: {exc}") from exc
    return MoleculeInput(id=safe_id(key), smiles=hit["smiles"], name=hit["title"], source="pubchem",
                         meta={"pubchem_cid": hit["cid"]})


def parse_text(text: str, skip_errors: bool = False, log: Callable[[str], None] | None = None,
               source: str = "") -> list[MoleculeInput]:
    """Each non-empty line is a SMILES or an InChI (either optionally followed by an id), an InChIKey or a name
    to look up (InChIKeys and names through PubChem).

    By default the first line that cannot be resolved raises. With ``skip_errors`` (list files), such
    lines are reported through ``log`` and skipped, as the CSV/Excel readers skip bad rows; it still
    raises when no line could be read at all."""
    out = []
    errors = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first, _, rest = line.partition(" ")
        word, after = (line.split(None, 1) + [""])[:2]      # InChI / InChIKey, then an optional id (any whitespace)
        after = after.strip()
        try:
            if word.startswith("InChI="):
                out.append(from_inchi(word, id=after or None, name=after))
            elif INCHIKEY_RE.match(word):
                hit = lookup_inchikey(word)
                if after:
                    hit = replace(hit, id=safe_id(after), name=after)
                out.append(hit)
            elif looks_like_smiles(first):
                out.append(from_smiles(first, id=rest.strip() or None, name=rest.strip()))
            elif any(ch in first for ch in "=#()[]@\\/") and not rest:
                raise ValueError(f"Not a valid SMILES string: {first!r}")
            else:
                out.append(lookup_name(line))
        except ValueError as exc:
            if not skip_errors:
                raise
            errors.append(f"{source + ' ' if source else ''}line {lineno}: {exc}; skipped")
    for e in errors:
        (log or _log.warning)(e)
    if errors and not out:
        raise ValueError(f"No molecule could be read{' from ' + source if source else ''}: " + "; ".join(errors[:5])
                         + (f" (and {len(errors) - 5} more)" if len(errors) > 5 else ""))
    return out


def _pick(columns: list[str], wanted: tuple[str, ...], explicit: str | None) -> str | None:
    if explicit:
        if explicit in columns:
            return explicit
        lower = {c.lower().strip(): c for c in columns}
        if explicit.lower().strip() in lower:  # forgiving about case: headers get retyped in Excel
            return lower[explicit.lower().strip()]
        raise ValueError(f"Column {explicit!r} not in file (columns: {', '.join(columns)})")
    lower = {c.lower().strip(): c for c in columns}
    for w in wanted:
        if w in lower:
            return lower[w]
    return None


def _pick_all(columns: list[str], wanted: Sequence[str]) -> list[str]:
    lower = {c.lower().strip(): c for c in columns}
    return [lower[w] for w in wanted if w in lower]


def _cell(value) -> str:
    """Spreadsheet or CSV cell -> clean string ('' for blanks and N/A placeholders)."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # Excel stores every number as a float
    text = str(value).strip()
    return "" if text.lower() in MISSING_VALUES else text


def _parses(smiles: str) -> bool:
    with rdBase.BlockLogs():  # thousands of rows: keep RDKit's parse errors out of the console
        return Chem.MolFromSmiles(smiles) is not None


def smiles_preference_columns(preference: str | Sequence[str] | None) -> tuple[str, ...]:
    """Resolve a preset ("smiles", "qsar", "ms"), a single column name (tried first, then the
    defaults) or an explicit ordered list of column names. Matching is case-insensitive."""
    if preference is None:
        return SMILES_PREFERENCES["smiles"]
    if isinstance(preference, str):
        key = preference.lower().strip()
        if key in SMILES_PREFERENCES:
            return SMILES_PREFERENCES[key]
        return (key, *(c for c in SMILES_COLUMNS if c != key))
    return tuple(c.lower().strip() for c in preference)


@dataclass
class ReadReport:
    """What a table reader found, so callers can tell the user what was skipped and why."""

    inputs: list[MoleculeInput] = field(default_factory=list)
    n_rows: int = 0                                                  # non-empty data rows
    skipped: list[tuple[str, str]] = field(default_factory=list)    # (row label, reason)
    smiles_columns: Counter = field(default_factory=Counter)        # column -> rows that used it
    warnings: list[str] = field(default_factory=list)

    def emit(self, log: Callable[[str], None] | None = None) -> None:
        for w in self.warnings:
            (log or _log.warning)(w)


def dedupe_ids(items: Iterable[MoleculeInput], warnings: list[str] | None = None) -> list[MoleculeInput]:
    """Make ids unique, because the id names the output folder and the job record.

    Ids are compared ignoring letter case: Windows and macOS file systems are case-insensitive, so
    ``c1ccccc1`` (benzene) and ``C1CCCCC1`` (cyclohexane), ids made from SMILES, would share one folder.
    An exact repeat (same id and structure) is dropped; an id already taken (in any letter case) by a
    different structure gets a ``_2``, ``_3``... suffix. The result depends only on input order, so
    re-reading the same list gives the same ids, which resumable runs rely on.
    """
    out: list[MoleculeInput] = []
    seen: dict[str, tuple[str, str]] = {}      # casefolded id -> (id as given, structure)
    dropped = 0
    for it in items:
        key = it.smiles or (Chem.MolToSmiles(it.mol) if it.mol is not None else "")
        taken = seen.get(it.id.casefold())
        if taken is not None:
            if taken == (it.id, key):
                dropped += 1
                continue
            n = 2
            while f"{it.id}_{n}".casefold() in seen:
                n += 1
            if warnings is not None:
                how = ("with a different structure" if taken[0] == it.id
                       else f"(same as {taken[0]!r} except for letter case, which names one folder on Windows "
                            "and macOS)")
                warnings.append(f"Duplicate id {it.id!r} {how} renamed to {it.id}_{n}")
            it = replace(it, id=f"{it.id}_{n}")
        seen[it.id.casefold()] = (it.id, key)
        out.append(it)
    if dropped and warnings is not None:
        warnings.append(f"Dropped {dropped} exact duplicate row(s) (same id and structure)")
    return out


def _rows_to_inputs(columns: list[str], rows: Iterable[tuple[int, int, dict]], file_name: str, stem: str,
                    smiles_col: str | None, id_col: str | None,
                    smiles_preference: str | Sequence[str] | None) -> ReadReport:
    """Shared CSV/Excel logic. ``rows`` yields (spreadsheet row number, data row index, {column: value})."""
    if smiles_col:
        scols = [_pick(columns, (), smiles_col)]
    else:
        scols = _pick_all(columns, smiles_preference_columns(smiles_preference))
    if not scols:
        raise ValueError(f"No SMILES column found in {file_name} (columns: {', '.join(columns)})")
    icol = _pick(columns, ID_COLUMNS, id_col)
    ncol = _pick(columns, NAME_COLUMNS, None)
    rep = ReadReport()
    fallbacks = 0
    for rownum, index, row in rows:
        cells = {k: _cell(v) for k, v in row.items() if k}
        if not any(cells.values()):
            continue  # fully empty line (Excel sheets often end with some)
        rep.n_rows += 1
        ident = cells.get(icol, "") if icol else ""
        candidates = [(c, cells.get(c, "")) for c in scols if cells.get(c, "")]
        if not candidates:
            rep.skipped.append((ident or f"row {rownum}", "no structure"))
            continue
        # the first column whose SMILES parses; if none parses keep the first, so the run reports it
        used, smi = next(((c, v) for c, v in candidates if _parses(v)), candidates[0])
        if used != scols[0]:
            fallbacks += 1
        rep.smiles_columns[used] += 1
        name = (cells.get(ncol, "") if ncol else "") or ident
        rep.inputs.append(MoleculeInput(id=safe_id(ident or f"{stem}_{index}"), smiles=smi, name=name,
                                        source=f"{file_name}:{rownum}",
                                        meta={**cells, "smiles_column": used, "row": rownum}))
    if rep.skipped:
        shown = ", ".join(label for label, _ in rep.skipped[:5]) + (", ..." if len(rep.skipped) > 5 else "")
        rep.warnings.append(f"{file_name}: skipped {len(rep.skipped)} row(s) without a structure ({shown})")
    if fallbacks:
        used = ", ".join(f"{c} for {n}" for c, n in rep.smiles_columns.items() if c != scols[0])
        rep.warnings.append(f"{file_name}: {fallbacks} row(s) had no usable {scols[0]}; used {used}")
    rep.inputs = dedupe_ids(rep.inputs, rep.warnings)
    return rep


def read_csv_report(path: Path, smiles_col: str | None = None, id_col: str | None = None,
                    smiles_preference: str | Sequence[str] | None = None) -> ReadReport:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = ((i + 1, i, row) for i, row in enumerate(reader, 1))  # header is line 1
        return _rows_to_inputs(columns, rows, path.name, path.stem, smiles_col, id_col, smiles_preference)


def read_csv(path: Path, smiles_col: str | None = None, id_col: str | None = None,
             smiles_preference: str | Sequence[str] | None = None,
             log: Callable[[str], None] | None = None) -> list[MoleculeInput]:
    rep = read_csv_report(path, smiles_col, id_col, smiles_preference)
    rep.emit(log)
    return rep.inputs


def _select_sheet(names: list[str], sheet: str | int | None, path: Path) -> str:
    if sheet is None or sheet == "":
        return names[0]
    if isinstance(sheet, str) and sheet.strip().isdigit() and sheet.strip().lower() not in {n.lower() for n in names}:
        sheet = int(sheet)           # "1" from the command line: a 0-based index unless a sheet is named "1"
    if isinstance(sheet, int):
        if 0 <= sheet < len(names):
            return names[sheet]
    else:
        lower = {n.lower(): n for n in names}
        if sheet.lower() in lower:
            return lower[sheet.lower()]
    raise ValueError(f"Sheet {sheet!r} not in {path.name} (sheets: {', '.join(names)})")


def _header_and_rows(values_iter: Iterable[Sequence]) -> tuple[list[str], list[tuple[int, int, dict]]]:
    """The header is the first non-empty row; returns (columns, [(row number, data index, row)])."""
    header: list[str] = []
    rows = []
    header_row = 0
    for rownum, values in enumerate(values_iter, 1):
        if not header:
            if any(_cell(v) for v in values):
                header, header_row = [_cell(v) for v in values], rownum
            continue
        rows.append((rownum, rownum - header_row, {h: v for h, v in zip(header, values) if h}))
    return [h for h in header if h], rows


def _excel_rows(path: Path, sheet: str | int | None) -> tuple[list[str], list[tuple[int, int, dict]]]:
    if path.suffix.lower() == ".xls":
        try:
            import xlrd
        except ImportError:
            raise ValueError(f"{path.name} is a legacy Excel 97-2003 .xls file. Open it in Excel and "
                             "'Save As' .xlsx or .csv, then load that file (or `pip install xlrd`).") from None
        book = xlrd.open_workbook(str(path))
        ws = book.sheet_by_name(_select_sheet(book.sheet_names(), sheet, path))
        return _header_and_rows(ws.row_values(r) for r in range(ws.nrows))
    try:
        import openpyxl
    except ImportError:
        raise ValueError("Reading .xlsx needs openpyxl (`pip install openpyxl`), or save the sheet as .csv") from None
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)  # data_only: cached formula values
    try:
        ws = wb[_select_sheet(wb.sheetnames, sheet, path)]
        return _header_and_rows(ws.iter_rows(values_only=True))
    finally:
        wb.close()  # read-only workbooks keep the file open (and locked on Windows) until closed


def read_excel_report(path: Path, smiles_col: str | None = None, id_col: str | None = None,
                      sheet: str | int | None = None,
                      smiles_preference: str | Sequence[str] | None = None) -> ReadReport:
    """Read .xlsx/.xlsm (or .xls when xlrd is installed) with the same column logic as CSV.

    The first sheet is used unless ``sheet`` (name or 0-based index) says otherwise; the header is the
    first non-empty row. CompTox Dashboard exports work as downloaded: DTXSID becomes the id,
    PREFERRED_NAME the name, and every column is kept in ``meta`` (plus ``smiles_column``, the column
    the structure came from, and ``row``).
    """
    columns, rows = _excel_rows(path, sheet)
    return _rows_to_inputs(columns, rows, path.name, path.stem, smiles_col, id_col, smiles_preference)


def read_excel(path: Path, smiles_col: str | None = None, id_col: str | None = None,
               sheet: str | int | None = None, smiles_preference: str | Sequence[str] | None = None,
               log: Callable[[str], None] | None = None) -> list[MoleculeInput]:
    rep = read_excel_report(path, smiles_col, id_col, sheet, smiles_preference)
    rep.emit(log)
    return rep.inputs


def read_table(path: str | Path, smiles_col: str | None = None, id_col: str | None = None,
               sheet: str | int | None = None,
               smiles_preference: str | Sequence[str] | None = None) -> ReadReport:
    """CSV or spreadsheet -> ``ReadReport`` (inputs, skipped rows, which SMILES columns were used)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return read_csv_report(path, smiles_col, id_col, smiles_preference)
    if ext in TABLE_TYPES:
        return read_excel_report(path, smiles_col, id_col, sheet, smiles_preference)
    raise ValueError(f"Not a table file: {path.name} (expected {', '.join(sorted(TABLE_TYPES))})")


SDF_ID_PROPERTIES = ("DTXSID", "ID")
#: ``MoleculeInput.meta`` key of a note about where the id came from (``read_sdf`` with ``id_col``)
ID_NOTE = "qb_id_note"


def _sdf_failure(supplier: Chem.SDMolSupplier, index: int) -> tuple[str, str]:
    """(title, reason) for SD record ``index`` (0-based) that RDKit could not read."""
    try:
        text = supplier.GetItemText(index)
    except Exception:  # noqa: BLE001
        return "", "RDKit could not read the record"
    title = text.split("\n", 1)[0].strip()
    with rdBase.BlockLogs():
        raw = Chem.MolFromMolBlock(text, sanitize=False, removeHs=False)
        if raw is None:
            return title, "unreadable connection table"
        try:
            Chem.SanitizeMol(raw)
        except Exception as exc:  # noqa: BLE001 - RDKit's message is the reason
            return title, str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return title, "RDKit could not read the record"


def read_sdf(path: str | Path, id_col: str | None = None,
             log: Callable[[str], None] | None = None, dedupe: bool = True) -> list[MoleculeInput]:
    """Read every record of an SD file as a ``MoleculeInput`` (3D coordinates, explicit H and charges kept;
    V2000 and V3000 records alike).

    The id is the SD property ``id_col`` when given (matched ignoring case, e.g. ``CAS_RN``), else DTXSID or
    ID, else the title line, else ``<stem>_<record>``; ``name`` is the title. A record without the ``id_col``
    property takes its title as the id (then DTXSID or ID, then ``<stem>_<record>``), with a note in its
    ``meta[ID_NOTE]`` that the pipeline copies into the molecule's notes. Records RDKit cannot read are
    skipped and reported through ``log`` with their record number, title and RDKit's reason, and so are
    records without the ``id_col`` property. Ids are made unique (``dedupe_ids``), as for tables, unless
    ``dedupe`` is False (``read_raw``: the import options decide about repeated structures)."""
    from .formats import COORDS, READER, coord_kind

    path = Path(path)
    out: list[MoleculeInput] = []
    skipped: list[str] = []
    no_id: list[str] = []
    with rdBase.BlockLogs():   # the reasons are collected per record below, not printed by RDKit
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mols = list(supplier)
    for i, mol in enumerate(mols, 1):
        if mol is None:
            title, why = _sdf_failure(supplier, i - 1)
            skipped.append(f"record {i}{' (' + title + ')' if title else ''}: {why}")
            continue
        title = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        # the text as written: GetPropsAsDict would turn "007" into 7 and "1e3" into 1000.0
        props = {k: mol.GetProp(k) for k in mol.GetPropNames()}
        ident = ""
        meta = dict(props)
        fallback = next((str(props[k]).strip() for k in SDF_ID_PROPERTIES if str(props.get(k, "")).strip()), "")
        if id_col:
            lower = {k.lower(): k for k in props}
            key = lower.get(id_col.strip().lower())
            ident = str(props[key]).strip() if key is not None else ""
            if not ident:
                no_id.append(f"record {i}{' (' + title + ')' if title else ''}")
                ident = title or fallback or f"{path.stem}_{i}"
                source = "the SD title" if title else "its DTXSID/ID property" if fallback else "the record number"
                meta[ID_NOTE] = f"no {id_col} property in the SD record: id {ident!r} from {source}"
        ident = ident or fallback or title or f"{path.stem}_{i}"
        meta[READER] = "RDKit (SD file)"
        meta[COORDS] = coord_kind(mol)
        with rdBase.BlockLogs():   # "not removing hydrogen atom without neighbors" for H- fragments
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        out.append(MoleculeInput(id=safe_id(ident, fallback=f"{path.stem}_{i}"), name=title or ident,
                                 source=f"{path.name}#{i}", mol=mol, smiles=smiles, meta=meta))
    warnings: list[str] = []
    if skipped:
        warnings.append(f"{path.name}: skipped {len(skipped)} record(s) RDKit cannot read: " + "; ".join(skipped[:10])
                        + (f" (and {len(skipped) - 10} more)" if len(skipped) > 10 else ""))
    if no_id:
        warnings.append(f"{path.name}: {len(no_id)} record(s) have no {id_col} property; used the title: "
                        + ", ".join(no_id[:10]) + (", ..." if len(no_id) > 10 else ""))
    if dedupe:
        out = dedupe_ids(out, warnings)
    for w in warnings:
        (log or _log.warning)(w)
    if skipped and not out:
        raise ValueError(f"No molecule could be read from {path.name}: " + "; ".join(skipped[:5]))
    return out


def _single(path: Path, mol: Chem.Mol, reader: str) -> list[MoleculeInput]:
    """One structure from a .mol, .pdb or .xyz file: the file name is its id."""
    from .formats import COORDS, READER, coord_kind

    with rdBase.BlockLogs():
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    return [MoleculeInput(id=safe_id(path.stem), name=path.stem, source=path.name, mol=mol, smiles=smiles,
                          meta={READER: reader, COORDS: coord_kind(mol, path.suffix.lower())})]


def _from_records(path: Path, read, id_col: str | None, log: Callable[[str], None] | None) -> list[MoleculeInput]:
    """``formats.FileRead`` -> inputs. The id is the ``id_col`` property, else DTXSID or ID, else (several records)
    the title or ``<stem>_<record>``; a file with one structure takes the file name, as a .mol file does."""
    from .formats import COORDS, READER, coord_kind

    ext = path.suffix.lower()
    single = len(read.records) == 1
    out = []
    for i, rec in enumerate(read.records, 1):
        props = dict(rec.props)
        lower = {k.lower(): k for k in props}
        ident = ""
        if id_col and lower.get(id_col.strip().lower()):
            ident = str(props[lower[id_col.strip().lower()]]).strip()
        ident = ident or next((str(props[k]).strip() for k in SDF_ID_PROPERTIES if str(props.get(k, "")).strip()), "")
        ident = ident or (path.stem if single else rec.title or f"{path.stem}_{i}")
        with rdBase.BlockLogs():
            smiles = Chem.MolToSmiles(Chem.RemoveHs(rec.mol))
        out.append(MoleculeInput(id=safe_id(ident, fallback=f"{path.stem}_{i}"), name=rec.title or ident,
                                 source=path.name if single else f"{path.name}#{i}", mol=rec.mol, smiles=smiles,
                                 multiplicity=rec.multiplicity or 1,
                                 meta={**props, READER: read.reader, COORDS: coord_kind(rec.mol, ext)}))
    emit = log or _log.warning
    for w in read.warnings:
        emit(w)
    if read.skipped:
        emit(f"{path.name}: skipped {len(read.skipped)} record(s): " + "; ".join(read.skipped[:10])
             + (f" (and {len(read.skipped) - 10} more)" if len(read.skipped) > 10 else ""))
    if not out:
        raise ValueError(f"No molecule could be read from {path.name}"
                         + (": " + "; ".join(read.skipped[:5]) if read.skipped else ""))
    (log or _log.info)(f"{path.name}: {len(out)} structure(s) read with {read.reader}")
    return out


def read_raw(path: str | Path, smiles_col: str | None = None, id_col: str | None = None,
             charge: int = 0, sheet: str | int | None = None,
             smiles_preference: str | Sequence[str] | None = None,
             log: Callable[[str], None] | None = None) -> list[MoleculeInput]:
    """Read any supported file as it is, before import options (``chem.import_options.apply``): structure files
    keep repeated structures (the options decide about them), tables and lists are read as ``read_file`` reads
    them. ``meta[formats.READER]`` names the reader, ``meta[formats.COORDS]`` says whether the coordinates are 2D or
    3D. An extension QuantaBench does not read goes to Open Babel when the user has it."""
    from . import formats

    path = Path(path)
    ext = path.suffix.lower()
    if ext in TABLE_TYPES:
        rep = read_table(path, smiles_col, id_col, sheet, smiles_preference)
        rep.emit(log)
        return rep.inputs
    if ext in LIST_TYPES:
        return parse_text(path.read_text(encoding="utf-8"), skip_errors=True, log=log, source=path.name)
    if ext in (".sdf", ".sd"):
        return read_sdf(path, id_col=id_col, log=log, dedupe=False)
    if ext in formats.FILE_READERS:
        return _from_records(path, formats.read(path), id_col, log)
    if ext == ".mol":
        with rdBase.BlockLogs():
            mol = Chem.MolFromMolFile(str(path), removeHs=False)
        reader = "RDKit (MOL file)"
    elif ext == ".pdb":
        with rdBase.BlockLogs():
            mol = Chem.MolFromPDBFile(str(path), removeHs=False)
        reader = "RDKit (PDB)"
    elif ext == ".xyz":
        mol = Chem.MolFromXYZFile(str(path))
        if mol is not None:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        reader = "RDKit rdDetermineBonds (XYZ)"
    elif formats.openbabel() is not None:
        items, reader = formats.with_openbabel(path, lambda sdf: read_sdf(sdf, id_col=id_col, log=log, dedupe=False))
        for i, it in enumerate(items, 1):
            it.source = path.name if len(items) == 1 else f"{path.name}#{i}"
            it.meta[formats.READER] = reader
        (log or _log.info)(f"{path.name}: {len(items)} structure(s) read with {reader}")
        return items
    else:
        raise ValueError(f"Unsupported file type {ext} (supported: {', '.join(sorted(FILE_TYPES))}; "
                         f"{formats.OPENBABEL_HINT})")
    if mol is None:
        raise ValueError(f"Could not read a molecule from {path.name}")
    return _single(path, mol, reader)


def read_file(path: str | Path, smiles_col: str | None = None, id_col: str | None = None,
              charge: int = 0, sheet: str | int | None = None,
              smiles_preference: str | Sequence[str] | None = None,
              log: Callable[[str], None] | None = None, options=None) -> list[MoleculeInput]:
    """Read any supported file. For tables, skipped rows and SMILES-column fallbacks are reported
    through ``log`` (or this module's logger when no callback is given).

    ``options`` (``chem.import_options.ImportOptions``): convert 2D to 3D, grow hydrogens, keep coordinates, post
    fields, filters, atom order, repeated structures as conformers; ``options.id_field`` is the SD property for
    the id when ``id_col`` is not given. Without options the structures are read as they are (ids made unique)."""
    from . import import_options

    path = Path(path)
    if options is not None:
        items = read_raw(path, smiles_col, id_col or options.id_field or None, charge, sheet, smiles_preference, log)
        return import_options.apply(items, options, log=log)
    ext = path.suffix.lower()
    if ext in (".sdf", ".sd"):
        return read_sdf(path, id_col=id_col, log=log)
    items = read_raw(path, smiles_col, id_col, charge, sheet, smiles_preference, log)
    if ext in TABLE_TYPES or ext in LIST_TYPES:
        return items
    warnings: list[str] = []
    items = dedupe_ids(items, warnings)
    for w in warnings:
        (log or _log.warning)(w)
    return items
