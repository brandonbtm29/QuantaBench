"""Edit > Paste and Copy as, the molecule list's right-click menu, and drag and drop onto the window (UI-4).

CONTRACT:

    install(win: MainWindow) -> None
        Adds an Edit menu (after File) with Paste (Ctrl+V), Copy molecules (Ctrl+Shift+C) and a Copy as submenu, puts
        the same actions in the molecule list's right-click menu (``MoleculeListView.menu_hooks``), accepts drops on
        the window and on the molecule list, and lets Ctrl+V in the Molecules box paste a structure (see below).

    ACTIONS / default_shortcuts(name)
        The actions are QActions named by ``objectName`` ("edit.paste", "edit.copy_molecules",
        "edit.copy_as.<kind>"), kept in ``win.actions_by_name``; their keys come from the window's keymap
        (``gui.keymap``: defaults in ``SPECS``, rebinding on Settings > Controls). No ``QShortcut``.

Paste reads the clipboard's formats in this order and uses the first one that gives molecules (the log says
which): ChemDraw CDX, CDXML, an MDL molfile, file URLs, text. Format names are compared ignoring case:

- ChemDraw CDX: Windows' registered format "ChemDraw Interchange Format" (Qt names it
  ``application/x-qt-windows-mime;value="ChemDraw Interchange Format"``), ``chemical/x-cdx``, and on macOS
  the pasteboard type ``com.cambridgesoft.chemdraw.cdx``. Qt shows a Mac application only the pasteboard
  types it has a converter for (``QUtiMimeConverter``, which PySide6 does not offer), so on macOS the
  pasteboard is also read directly through the Objective-C runtime (``ctypes``; types with "chemdraw" or
  "cdx" in their name).
- CDXML: ``chemical/x-cdxml``, ``com.cambridgesoft.chemdraw.cdxml``, a Windows format named "CDXML".
- MDL molfile: ``chemical/x-mdl-molfile``, ``chemical/x-mdl-sdfile``, Windows' "MDLCT" (length-prefixed
  lines, ``chem.interchange.mdlct_decode``).
- Text (``chem.interchange.parse_structure_text``): MOL/SD blocks, CDXML and MRV text, XYZ (one or several frames,
  or bare ``element x y z`` lines), a table with a SMILES column (CSV, or cells copied from a spreadsheet, which are
  tab-separated; the Pipeline dock's SMILES/ID column fields and SMILES preference apply, as for a CSV file), or
  lines of SMILES, InChI, InChIKeys, names and CAS numbers (InChIKeys, names and CAS numbers go to PubChem, as in the
  Molecules box).

Pasted molecules take File > Open's path: they are read on a worker thread (``gui.workers.FunctionWorker``), then the
Import Options apply (``gui.import_options.choose``: the dialog when the paste has choices, such as a 2D drawing,
data fields or fragments, and Settings > Import says to ask), on a worker thread too (``MainWindow._import_inputs``),
so a table's columns can be posted to the molecule list as for a file. Paste with the focus in a text field pastes
into the field; Ctrl+V in the Molecules box adds molecules only when the clipboard holds a structure format or
several lines of text (a single line goes into the box as usual). A clipboard with nothing readable gives a message
that names its formats and what QuantaBench reads.

Copy molecules (Ctrl+Shift+C with the focus in the Molecules dock, so the Analyze table keeps its own Ctrl+Shift+C;
Ctrl+C stays Qt's copy of the current cell) puts the selected rows on the clipboard as a tab-separated table with a
header: ID, SMILES, Name and the posted data-field columns that are shown. It pastes into a spreadsheet, and back
into QuantaBench (``MOLECULES_MIME`` keeps the ID column the id when it comes straight back).

Copy as puts the selected molecules (or the current one) on the clipboard: SMILES as entered (the list's own text,
so any number of rows copies at once), canonical SMILES, InChI, InChIKey (one per line, with the id after it when
there are several), MOL (2D, an SD file for several; also ``chemical/x-mdl-molfile`` / ``-sdfile`` and on Windows
"MDLCT"), XYZ (the conformer shown for the current molecule, the best conformer of other built ones, the input's 3D
coordinates or the 3D view's preview geometry for molecules not built; several molecules are several frames) or
CDXML (also ``chemical/x-cdxml``). The text is always there, so ChemDraw's Edit > Paste Special (SMILES, MOL Text,
CDXML Text) reads it. More than ``SYNC_LIMIT`` molecules are written on a worker thread.

Drops: files go through ``win.open_files`` (structure files, lists and tables, the File > Open path with its Import
Options, any number at once, read on a worker thread; a type QuantaBench does not read goes to Open Babel when it is
installed, else File > Open's message names it), calculation outputs and Spartan documents through File > Import
calculation output, a run folder through Open run folder, a project folder (``.qbproj``) through File > Open
project, any other folder as the files directly in it that QuantaBench reads; anything else dropped (text,
ChemDraw drags) is read like a paste.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QMimeData, QObject
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QTextEdit,
)

from . import keymap

_log = logging.getLogger(__name__)

WINDOWS_MIME = 'application/x-qt-windows-mime;value="{}"'
CDX_NAMES = ("chemdraw interchange format", "com.cambridgesoft.chemdraw.cdx", "chemical/x-cdx")
CDXML_NAMES = ("chemical/x-cdxml", "com.cambridgesoft.chemdraw.cdxml", "cdxml")
MOL_NAMES = ("chemical/x-mdl-molfile", "chemical/x-mdl-sdfile", "mdl molfile")
MDLCT_NAMES = ("mdlct",)
#: calculation outputs that File > Import calculation output reads (when File > Open does not)
OUTPUT_TYPES = {".out", ".log", ".dat", ".output", ".lis", ".spartan"}
TITLE = "Paste"
#: what a paste or a drop can hold, for the messages
READS = ("QuantaBench reads SMILES, InChI, InChIKeys, names and CAS numbers (one per line), MOL and SD blocks, XYZ, "
         "tables with a SMILES column (CSV, or cells copied from a spreadsheet), ChemDraw (CDX, CDXML), Marvin, and "
         "structure files.")
#: Copy as of more molecules than this runs on a worker thread (canonical SMILES of 200 take ~20 ms, InChI ~50 ms)
SYNC_LIMIT = 200
#: Copy molecules' table, besides the text: Paste reads it with the ID and SMILES columns it was written with
MOLECULES_MIME = "application/x-quantabench-molecules"
MOLECULES_COLUMNS = ("ID", "SMILES", "Name")

#: objectName -> (text, tooltip) of the actions this module adds; the Copy as actions are "edit.copy_as.<kind>"
ACTIONS = {
    "edit.paste": ("Paste", "Add the structures on the clipboard to the Molecules list: ChemDraw, MOL or SD, XYZ, "
                   "CDXML, Marvin, a table with a SMILES column, SMILES, InChI, InChIKeys, names or CAS numbers "
                   "(in a text field: paste the text)"),
    "edit.copy_molecules": ("Copy molecules", "Copy the selected rows as a table (ID, SMILES, name and the data-field "
                            "columns shown) for a spreadsheet, or to paste back here"),
}


def default_shortcuts(name: str) -> list[QKeySequence]:
    """The default keys of action ``name``: the keymap's (``gui.keymap.SPECS``, the one place keys are set)."""
    return keymap.default_keys(name)


def triggers(action, event) -> bool:
    """Whether key ``event`` is one of ``action``'s keys as they are now (after any rebinding)."""
    try:
        pressed = QKeySequence(event.keyCombination())
    except (AttributeError, TypeError):
        return False
    return any(seq == pressed for seq in action.shortcuts())


def _clipboard():
    """The system clipboard (one place, so tests can stand in for it where the session has none)."""
    return QGuiApplication.clipboard()


def format_name(fmt: str) -> str:
    """The clipboard format's own name, lowercase: Qt's Windows wrapper removed."""
    m = re.match(r'application/x-qt-windows-mime;value="(.*)"$', fmt.strip(), re.I)
    return (m.group(1) if m else fmt).strip().lower()


def _kind(fmt: str) -> str | None:
    name = format_name(fmt)
    if name == MOLECULES_MIME:
        return "molecules"
    if name in CDX_NAMES or "chemdraw interchange" in name:
        return "cdx"
    if name in CDXML_NAMES:
        return "cdxml"
    if name in MDLCT_NAMES:
        return "mdlct"
    if name in MOL_NAMES:
        return "mol"
    return None


_ORDER = ("molecules", "cdx", "cdxml", "mdlct", "mol")


def candidates(mime: QMimeData) -> list[tuple[str, str, object]]:
    """(format as named, kind, data) of the structures in ``mime``, in the order they are tried: QuantaBench's own
    molecule table (Copy molecules), ChemDraw CDX, CDXML, MDLCT, MOL, then text. Plain Python data (bytes, str), so a worker thread can parse them."""
    found: list[tuple[str, str, object]] = []
    for fmt in mime.formats():
        kind = _kind(fmt)
        if kind is not None:
            data = bytes(mime.data(fmt).data())
            if data:
                found.append((fmt, kind, data))
    found.sort(key=lambda c: _ORDER.index(c[1]))
    if mime.hasText() and mime.text().strip():
        found.append(("text/plain", "text", mime.text()))
    return found


def local_files(mime: QMimeData) -> list[str]:
    # Path(): native separators (QUrl gives C:/... on Windows)
    return [str(Path(u.toLocalFile())) for u in mime.urls() if u.isLocalFile() and u.toLocalFile()] \
        if mime.hasUrls() else []


def readable(mime: QMimeData | None) -> bool:
    """Files, a structure format or text: what a drop can take (checked without reading the data)."""
    if mime is None:
        return False
    return bool(local_files(mime)) or any(_kind(f) for f in mime.formats()) or mime.hasText()


def _decode(data) -> str:
    if isinstance(data, str):
        return data
    try:
        if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return data.decode("utf-16").rstrip("\0")
        return data.decode("utf-8-sig").rstrip("\0")
    except UnicodeDecodeError:
        return data.decode("latin-1").rstrip("\0")


#: what ``interchange.detect_format`` found in pasted text, for the log line ("Paste: 3 molecule(s) from text/plain
#: (XYZ text)")
_TEXT_KINDS = {"molecules": "QuantaBench molecules", "mol": "MOL text", "sdf": "SD text", "xyz": "XYZ text", "table": "table", "cdxml": "CDXML text",
               "mrv": "Marvin text", "text": "text"}


def read_candidates(cands: list[tuple[str, str, object]], table: dict | None = None, source: str = "clipboard"):
    """Worker thread: the molecules of the first candidate that gives any, as ``inputs.read_raw`` gives a file's
    (before the import options). ``table``: ``smiles_col``, ``id_col`` and ``smiles_preference`` for a pasted table.
    Returns (format used, MoleculeInputs, messages); raises ValueError with every format's reason when none does."""
    from ..chem import interchange

    table = {k: v for k, v in (table or {}).items() if v}
    errors = []
    for fmt, kind, data in cands:
        label = f"{format_name(fmt)} ({ {'text': 'text', 'molecules': 'QuantaBench molecules'}.get(kind, kind.upper()) })"
        messages: list[str] = []
        try:
            if kind == "molecules":                 # Copy molecules: its ID column is the id
                items = interchange.read_table_text(_decode(data), source, smiles_col="SMILES", id_col="ID",
                                                    log=messages.append)
                for it in items:                    # the id and the name, not data fields to post
                    for k in ("ID", "Name"):
                        it.meta.pop(k, None)
            elif kind == "cdx":
                items = interchange.parse_cdx_bytes(data, source=source)
            elif kind == "mdlct":
                items = interchange.parse_structure_text(interchange.mdlct_decode(data), source, messages.append)
            else:
                items = interchange.parse_structure_text(_decode(data), source, messages.append, **table)
        except Exception as exc:  # noqa: BLE001 - try the next format; RDKit raises RuntimeError too
            errors.append(f"{label}: {exc}")
            continue
        if items:
            if kind == "text":
                label = f"{format_name(fmt)} ({_TEXT_KINDS.get(interchange.detect_format(_decode(data)), 'text')})"
            return label, items, messages
        errors.append(f"{label}: no molecule")
    raise ValueError(("No molecule could be read: " + "; ".join(errors)) if errors else "Nothing to read")


# ================================================================== macOS pasteboard


def _objc():
    import ctypes
    import ctypes.util

    objc = ctypes.CDLL(ctypes.util.find_library("objc"))
    ctypes.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")    # NSPasteboard
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def send(obj, selector: bytes, restype=ctypes.c_void_p, argtypes=(), *args):
        # objc_msgSend with this call's exact prototype (needed on Apple silicon, where variadic calls differ)
        fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(("objc_msgSend", objc))
        return fn(obj, objc.sel_registerName(selector), *args)

    return ctypes, objc, send


#: the macOS pasteboard's types at the last ``mac_pasteboard`` call (Paste writes them to the log)
mac_types: list[str] = []


def mac_pasteboard() -> list[tuple[str, str, bytes]]:
    """ChemDraw data on the macOS general pasteboard that Qt does not show: (type, kind, bytes) for types with
    "chemdraw" or "cdx" in their name. [] on other systems or on any failure (logged)."""
    if sys.platform != "darwin":
        return []
    mac_types.clear()
    try:
        ctypes, objc, send = _objc()
        pb = send(objc.objc_getClass(b"NSPasteboard"), b"generalPasteboard")
        types = send(pb, b"types")
        names = []
        for i in range(send(types, b"count", ctypes.c_ulong) if types else 0):
            s = send(types, b"objectAtIndex:", ctypes.c_void_p, [ctypes.c_ulong], i)
            raw = send(s, b"UTF8String", ctypes.c_char_p) if s else None
            if raw:
                names.append(raw.decode("utf-8", "replace"))
        mac_types[:] = names
        _log.info("macOS pasteboard types: %s", ", ".join(names))
        out = []
        for name in names:
            low = name.lower()
            if "chemdraw" not in low and "cdx" not in low:
                continue
            ns = send(objc.objc_getClass(b"NSString"), b"stringWithUTF8String:", ctypes.c_void_p,
                      [ctypes.c_char_p], name.encode("utf-8"))
            data = send(pb, b"dataForType:", ctypes.c_void_p, [ctypes.c_void_p], ns)
            if not data:
                continue
            n = send(data, b"length", ctypes.c_ulong)
            ptr = send(data, b"bytes", ctypes.c_void_p)
            if n and ptr:
                blob = ctypes.string_at(ptr, n)
                out.append((name, "cdxml" if b"<CDXML" in blob[:4096] else "cdx", blob))
        return out
    except Exception as exc:  # noqa: BLE001 - never let the pasteboard bridge break Paste
        _log.warning("Could not read the macOS pasteboard directly: %s: %s", type(exc).__name__, exc)
        return []


# ================================================================== paste and drop


def _where(what: str) -> str:
    return "clipboard" if what == TITLE else "dropped data"


def _report_nothing(win, mime: QMimeData, what: str) -> None:
    fmts = ", ".join(mime.formats()) or "none"
    win.log(f"{what}: no structure or text found (formats: {fmts})")
    win.statusBar().showMessage(f"{what}: nothing QuantaBench can read", 5000)
    QMessageBox.information(win, what, f"The {_where(what)} holds no structure or text (formats: {fmts}).\n\n{READS}")


def _table_options(win) -> dict:
    """The Pipeline dock's CSV fields (SMILES column, ID column) and SMILES preference, as File > Open uses them."""
    def text(name: str) -> str | None:
        w = getattr(win, name, None)
        return (w.text().strip() or None) if w is not None else None

    pref = getattr(win, "smiles_pref", None)
    return dict(smiles_col=text("csv_smiles_col"), id_col=text("csv_id_col"),
                smiles_preference=pref.currentData() if pref is not None else None)


def asks_import_options(win, items: list) -> bool:
    """Whether ``import_items`` will show the Import Options dialog for ``items``."""
    from ..chem.import_options import survey

    return bool(win.s.import_.ask) and survey(items).has_choices()


def import_note(asks: bool) -> str:
    """How the Import Options were chosen and where to change them (the paste log line ends with it)."""
    from .import_options import SETTINGS_PATH

    if asks:
        return f"Import Options asked; you can change them later in {SETTINGS_PATH}"
    return f"stored Import Options applied; you can change them later in {SETTINGS_PATH}"


def import_items(win, items: list, what: str = TITLE) -> bool:
    """Molecules read from the clipboard or a drop into the list through File > Open's Import Options (the dialog
    when they have choices and Tools > Settings > Import says to ask, until its "Don't ask again" is ticked). False
    when the user cancels the dialog."""
    from . import import_options as import_gui

    options = import_gui.choose(win, win.s, items, [f"the {_where(what)}"])
    if options is None:
        win.log(f"{what}: cancelled in Import Options; nothing added")
        return False
    win._import_inputs(items, options)
    return True


def add_from(win, cands, what: str = TITLE) -> None:
    """Parse ``cands`` on a worker thread, then add the molecules through the Import Options (``import_items``)."""
    from . import workers

    win.statusBar().showMessage(f"{what}: reading structures...")

    def done(value):
        label, items, messages = value
        win.statusBar().clearMessage()
        for m in messages:
            win.log(m)
        _log.info("%s: %d molecule(s) from %s", what, len(items), label)
        win.log(f"{what}: {len(items)} molecule(s) from {label}; {import_note(asks_import_options(win, items))}")
        import_items(win, items, what)

    def failed(msg):
        win.statusBar().clearMessage()
        msg = msg.removeprefix("ValueError: ")
        win.log(f"{what}: {msg}")
        QMessageBox.warning(win, what, f"Nothing in the {_where(what)} could be read as a molecule.\n\n{msg}\n\n{READS}")

    workers.start(workers.FunctionWorker(read_candidates, list(cands), _table_options(win),
                                         "clipboard" if what == TITLE else "drop"),
                  win, on_done=done, on_failed=failed)


def _is_run_folder(folder: Path) -> bool:
    from ..chem import jobs

    return (folder / jobs.DB_NAME).is_file() or (folder / "summary.csv").is_file()


def _readable_suffixes() -> set[str]:
    from ..chem import inputs
    from .import_options import CALCULATION_OUTPUTS

    return set(inputs.FILE_TYPES) | OUTPUT_TYPES | set(CALCULATION_OUTPUTS)


def open_paths(win, paths: list[str], what: str = "Drop") -> None:
    """Dropped (or pasted) files and folders: each goes to the reader File would use for it. A run folder opens as
    one, a project folder as a project; any other folder stands for the files directly in it that QuantaBench
    reads."""
    from .. import project as proj
    from ..chem import inputs

    files: list[str] = []
    run_folders: list[str] = []
    projects: list[str] = []
    empty: list[str] = []
    suffixes = _readable_suffixes()
    for p in paths:
        path = Path(p)
        if path.is_dir():
            if proj.is_project(path) and getattr(win, "project", None) is not None:
                projects.append(p)
            elif _is_run_folder(path):
                run_folders.append(p)
            else:
                inside = sorted(str(f) for f in path.iterdir() if f.is_file() and f.suffix.lower() in suffixes)
                files += inside
                if not inside:
                    empty.append(path.name)
        elif path.is_file():
            files.append(p)
    outputs = [p for p in files if Path(p).suffix.lower() not in inputs.FILE_TYPES
               and Path(p).suffix.lower() in OUTPUT_TYPES]
    structures = [p for p in files if p not in outputs]
    if empty:
        win.log(f"{what}: no file QuantaBench reads in {', '.join(empty)}")
        if not (structures or outputs or run_folders or projects):
            QMessageBox.information(win, what, f"No file QuantaBench reads in {', '.join(empty)}.\n\n{READS}")
    if structures:
        win.open_files(structures)
    if outputs and getattr(win, "a_import_output", None) is not None:
        from . import import_action

        import_action.import_files(win, outputs)
    if projects:
        win.project.open_project(path=projects[0])
    elif run_folders:
        win.open_run_folder(folder=run_folders[0])
    if len(projects) + len(run_folders) > 1:
        win.log(f"Opened {(projects or run_folders)[0]}; only one run folder or project can be open at a time")


def paste(win, mime: QMimeData | None = None, what: str = TITLE) -> bool:
    """Add the molecules on the clipboard (or in ``mime``). False when there was nothing to read."""
    from_clipboard = mime is None
    if mime is None:
        mime = _clipboard().mimeData()
    if mime is None:
        return False
    files = local_files(mime)
    if files:
        open_paths(win, files, what)
        return True
    cands = candidates(mime)
    if from_clipboard:
        extra = [c for c in mac_pasteboard() if c[0] not in {f for f, _, _ in cands}]
        cands = sorted(extra + cands, key=lambda c: _ORDER.index(c[1]) if c[1] in _ORDER else len(_ORDER))
        if mac_types:
            win.log(f"{what}: macOS pasteboard types: {', '.join(mac_types)}")
    if not cands:
        _report_nothing(win, mime, what)
        return False
    add_from(win, cands, what)
    return True


def drop(win, mime: QMimeData | None) -> bool:
    """A drop on the window or the molecule list: files and folders open, anything else is read like a paste."""
    if mime is None:
        return False
    files = local_files(mime)
    if files:
        open_paths(win, files, "Drop")
        return True
    if candidates(mime):
        return paste(win, mime, "Drop")
    return False


def _text_target():
    """The focused text field Paste should go to, if any."""
    w = QApplication.focusWidget()
    if isinstance(w, QComboBox) and w.isEditable():
        w = w.lineEdit()
    if isinstance(w, QLineEdit) and not w.isReadOnly():
        return w
    if isinstance(w, (QPlainTextEdit, QTextEdit)) and not w.isReadOnly():
        return w
    return None


def edit_paste(win) -> None:
    """Edit > Paste: into the focused text field, else as molecules (in the Molecules box, as Ctrl+V there)."""
    target = _text_target()
    if target is not None and not (target is getattr(win, "entry", None)
                                   and structure_on_clipboard(_clipboard().mimeData())):
        target.paste()
        return
    paste(win)


def structure_on_clipboard(mime: QMimeData | None) -> bool:
    """A structure format, or text of several lines (a MOL block, a list): what the Molecules box's single line
    cannot take."""
    if mime is None:
        return False
    if any(_kind(f) for f in mime.formats()) or mac_pasteboard():
        return True
    return mime.hasText() and len([s for s in mime.text().splitlines() if s.strip()]) > 1


class _EventFilter(QObject):
    """Drops on the window and the molecule list; Ctrl+V in the Molecules box when the clipboard holds a
    structure."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.targets: set = {win}

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt's name
        t = event.type()
        if obj in self.targets and t in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if readable(event.mimeData()):
                event.acceptProposedAction()
                return True
            return False
        if obj in self.targets and t == QEvent.Type.Drop:
            mime = event.mimeData()
            if mime is None:
                return False
            event.acceptProposedAction()
            drop(self.win, mime)
            return True
        entry = getattr(self.win, "entry", None)
        if obj is entry and t in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            a_paste = getattr(self.win, "a_paste", None)
            if a_paste is not None and triggers(a_paste, event) and \
                    structure_on_clipboard(_clipboard().mimeData()):
                if t == QEvent.Type.KeyPress:
                    paste(self.win)
                event.accept()
                return True
        return False


# ================================================================== copy

COPY_LABELS = {"smiles": "SMILES", "canonical": "Canonical SMILES", "inchi": "InChI", "inchikey": "InChIKey",
               "mol": "MOL / SDF", "xyz": "XYZ (current conformer)", "cdxml": "CDXML"}
COPY_TIPS = {
    "smiles": "The SMILES as entered (or read from the file), one per line with the id after it when there are several",
    "canonical": "RDKit canonical isomeric SMILES of the input structures, one per line with the id after it",
    "inchi": "Standard InChI of the input structures, one per line with the id after it",
    "inchikey": "Standard InChIKey of the input structures, one per line with the id after it",
    "mol": "MOL block with 2D coordinates (an SD file for several molecules), for ChemDraw and other editors",
    "xyz": "Cartesian coordinates (Å): the conformer shown for the current molecule, the best conformer of other "
           "built molecules, else the input's 3D coordinates or the 3D view's preview; several molecules are several "
           "frames",
    "cdxml": "ChemDraw CDXML of the input structures (several side by side in one drawing)",
}
#: extra clipboard formats per kind, besides text (the text is always there)
_MIME = {"smiles": "chemical/x-daylight-smiles", "canonical": "chemical/x-daylight-smiles",
         "inchi": "chemical/x-inchi", "inchikey": "chemical/x-inchikey", "xyz": "chemical/x-xyz",
         "cdxml": "chemical/x-cdxml"}


def _ident(obj) -> str:
    inp = getattr(obj, "input", None)
    return getattr(obj, "id", "") or getattr(inp, "id", "") or ""


def _lines_with_ids(pairs: list[tuple[str, str]]) -> str:
    if len(pairs) == 1:
        return pairs[0][0] + "\n"
    return "".join(f"{value} {ident}".rstrip() + "\n" for value, ident in pairs)


def _xyz_frame(obj) -> str:
    """One XYZ frame for ``obj``: (id, molecule, conformer id, comment); a built ``MoleculeResult`` (its best
    conformer); else a structure's own 3D coordinates, or a preview geometry (ETKDG + MMFF, as the 3D view shows a
    molecule that is not built)."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    from ..chem import interchange

    if isinstance(obj, tuple):
        ident, mol, conf_id, comment = obj
        return interchange.to_xyz(mol, conf_id, comment)
    res_mol, confs = getattr(obj, "mol", None), getattr(obj, "conformers", None)
    if getattr(obj, "status", "") == "ok" and res_mol is not None and confs:
        c = confs[0]
        return interchange.to_xyz(res_mol, c.conf_id, _xyz_comment(_ident(obj), c, len(confs), obj.method))
    ident = _ident(obj)
    inp = getattr(obj, "input", None)
    if hasattr(obj, "to_mol"):
        mol = obj.to_mol()
    elif isinstance(obj, Chem.Mol):
        mol = Chem.Mol(obj)
    elif hasattr(inp, "to_mol"):
        mol = inp.to_mol()
    else:
        raise TypeError(f"Not a molecule: {type(obj).__name__}")
    if mol.GetNumConformers() and mol.GetConformer().Is3D():
        if any(a.GetTotalNumHs() for a in mol.GetAtoms()):
            with rdBase.BlockLogs():
                mol = Chem.AddHs(mol, addCoords=True)
        return interchange.to_xyz(mol, -1, f"{ident} (input coordinates, not built)".strip())
    mol = Chem.AddHs(Chem.Mol(mol))
    with rdBase.BlockLogs():
        if AllChem.EmbedMolecule(mol, randomSeed=1) != 0:
            raise ValueError("no 3D coordinates (not built, and RDKit could not embed a preview)")
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
    return interchange.to_xyz(mol, -1, f"{ident} (preview geometry, not built)".strip())


def _xyz_comment(ident: str, c, n: int, method: str = "") -> str:
    energy = f", E = {c.energy_ev:.6f} eV" if isinstance(c.energy_ev, (int, float)) else ""
    return f"{ident} conformer {c.rank + 1} of {n}{energy}" + (f" ({method})" if method else "")


def copy_text(kind: str, items: list) -> tuple[str, list[str], int]:
    """(text, problems, molecules written) for ``items`` as ``kind`` (``COPY_LABELS``). No Qt: runs on a worker
    thread for large selections. ``items``: ``MoleculeInput``, RDKit molecules, ``MoleculeResult``; for "xyz" also
    (id, molecule, conformer id, comment). Molecules that cannot be written are left out and named in the problems."""
    from ..chem import interchange

    if kind not in COPY_LABELS:
        raise ValueError(f"Unknown format {kind!r}")
    good, problems = [], []
    for it in items:
        ident = it[0] if isinstance(it, tuple) else _ident(it)
        try:
            if kind == "smiles":            # as entered: a result's input, not its built SMILES
                value = getattr(getattr(it, "input", it), "smiles", "") or interchange.to_smiles(it)
            elif kind == "canonical":
                value = interchange.to_smiles(it)
            elif kind == "inchi":
                value = interchange.to_inchi(it)
            elif kind == "inchikey":
                value = interchange.to_inchikey(it)
            elif kind == "mol":
                value = interchange.to_molblock(it)
            elif kind == "xyz":
                value = _xyz_frame(it)
            else:                                   # cdxml: checked here, written below as one drawing
                interchange.to_smiles(it)
                value = ""
        except Exception as exc:  # noqa: BLE001 - one bad structure must not stop the others
            problems.append(f"{ident or '?'}: {exc}")
            continue
        good.append((value, ident, it))
    if not good:
        return "", problems, 0
    if kind in ("smiles", "canonical", "inchi", "inchikey"):
        text = _lines_with_ids([(v, i) for v, i, _ in good])
    elif kind == "mol":
        text = good[0][0] if len(good) == 1 else "".join(v + "$$$$\n" for v, _, _ in good)
    elif kind == "xyz":
        text = "".join(v for v, _, _ in good)
    else:
        text = interchange.to_cdxml([it for _, _, it in good])
    return text, problems, len(good)


def mime_for(kind: str, text: str, n: int, platform: str = sys.platform) -> QMimeData:
    """Clipboard data for ``copy_text``'s text (GUI thread): the text, and the chemical format's own MIME type."""
    from ..chem import interchange

    mime = QMimeData()
    if not text:
        return mime
    if kind in ("smiles", "canonical", "inchi", "inchikey") and n == 1:
        text = text.rstrip("\n")                    # one value: no line end, for pasting into a field
    mime.setText(text)
    if kind == "mol":
        mime.setData("chemical/x-mdl-molfile" if n == 1 else "chemical/x-mdl-sdfile", text.encode())
        if platform == "win32" and n == 1:
            mime.setData(WINDOWS_MIME.format("MDLCT"), interchange.mdlct_encode(text))
    elif kind in _MIME:
        mime.setData(_MIME[kind], text.encode("utf-8"))
    return mime


def copy_mime(kind: str, items: list, platform: str = sys.platform) -> tuple[QMimeData, list[str]]:
    """(clipboard data, problems) for ``items`` as ``kind`` (``COPY_LABELS``): ``copy_text`` then ``mime_for``."""
    text, problems, n = copy_text(kind, items)
    return mime_for(kind, text, n, platform), problems


def selected_rows(win) -> list:
    """The selected rows of the molecule list (in list order), else the current one."""
    ids = win.mol_list.selected_ids()
    if not ids and win.mol_list.current_id():
        ids = [win.mol_list.current_id()]
    rows = (win.model.row(i) for i in ids)
    return [r for r in rows if r is not None]


def selected_inputs(win) -> list:
    """The inputs of the selected rows (an opened run folder's from its job table, one query per folder)."""
    from .molecule_list import gather_inputs

    return gather_inputs([(r.id, r.run_dir, r.input, r.smiles, r.name, r.source) for r in selected_rows(win)])


def _is_built(res) -> bool:
    return res is not None and res.status == "ok" and res.mol is not None and bool(res.conformers)


def _xyz_specs(win, rows) -> list[dict]:
    """What the worker needs to write XYZ for ``rows`` (GUI thread: the results already in memory and the
    conformer the views show for the current molecule)."""
    current = win.mol_list.current_id()
    shown = getattr(win, "_shown_res", None)
    conf_table = getattr(win, "conf_table", None)
    specs = []
    for r in rows:
        res = win.results.cached(r.id)
        conf = 0
        if r.id == current and shown is not None and res is shown and conf_table is not None:
            conf = max(conf_table.currentRow(), 0)
        specs.append(dict(id=r.id, folder=r.folder, run_dir=r.run_dir, status=r.status, input=r.input,
                          smiles=r.smiles, name=r.name, source=r.source, res=res if _is_built(res) else None, conf=conf))
    return specs


def _xyz_items(specs: list[dict]) -> tuple[list, list[str]]:
    """Worker thread: (XYZ items for ``copy_text``, problems). Built molecules not in memory are read from their
    folders; inputs of an opened run folder from its job table."""
    from .molecule_list import gather_inputs, load_result

    items, problems, need = [], [], []
    for sp in specs:
        res = sp["res"]
        if res is None and sp["status"] == "ok" and (sp["folder"] or sp["run_dir"]):
            try:
                res = load_result(sp["id"], sp["folder"], sp["run_dir"], sp["input"])
            except Exception as exc:  # noqa: BLE001 - a moved folder: say so, write the others
                problems.append(f"{sp['id']}: could not read its result ({exc})")
                continue
        if _is_built(res):
            k = min(sp["conf"], len(res.conformers) - 1)
            c = res.conformers[k]
            items.append((sp["id"], res.mol, c.conf_id, _xyz_comment(sp["id"], c, len(res.conformers), res.method)))
        else:
            need.append(sp)
            items.append(None)
    if need:
        found = iter(gather_inputs([(s["id"], s["run_dir"], s["input"], s["smiles"], s["name"], s["source"])
                                    for s in need]))
        items = [it if it is not None else next(found) for it in items]
    return items, problems


def _copy_job(kind: str, payload) -> tuple[str, list[str], int]:
    """Worker thread (or GUI thread for small selections): the text of Copy as."""
    from .molecule_list import gather_inputs

    if kind == "xyz":
        items, problems = _xyz_items(payload)
        text, more, n = copy_text(kind, items)
        return text, problems + more, n
    return copy_text(kind, gather_inputs(payload))


def _finish_copy(win, kind: str, value) -> bool:
    text, problems, n = value
    label = COPY_LABELS[kind]
    for p in problems[:20]:
        win.log(f"Copy as {label}: {p}")
    if len(problems) > 20:
        win.log(f"Copy as {label}: ... and {len(problems) - 20} more molecule(s) left out")
    if not n:
        QMessageBox.warning(win, f"Copy as {label}", "Nothing could be copied.\n\n" + "\n".join(problems[:10]))
        return False
    _clipboard().setMimeData(mime_for(kind, text, n))
    msg = f"Copied {n:,} molecule(s) as {label}"
    if problems:
        msg += f" ({len(problems):,} left out: see the log)"
    win.statusBar().showMessage(msg, 5000)
    return True


def copy_as(win, kind: str) -> bool:
    """Edit > Copy as / the molecule list's Copy as: the selected molecules (or the current one) as ``kind``. SMILES
    as entered copies at once, however many rows; the other formats run on a worker thread above ``SYNC_LIMIT``
    molecules (True then means started)."""
    from . import workers

    rows = selected_rows(win)
    label = COPY_LABELS[kind]
    if not rows:
        win.statusBar().showMessage("Select a molecule to copy", 4000)
        return False
    if kind == "smiles":                    # the list's own text: no structure to read, any number of rows
        pairs = [(r.smiles or (r.input.smiles if r.input is not None else ""), r.id) for r in rows]
        missing = [i for s, i in pairs if not s]
        pairs = [(s, i) for s, i in pairs if s]
        problems = [f"{i}: no SMILES" for i in missing]
        return _finish_copy(win, kind, (_lines_with_ids(pairs) if pairs else "", problems, len(pairs)))
    if kind == "xyz":
        payload = _xyz_specs(win, rows)
    else:
        payload = [(r.id, r.run_dir, r.input, r.smiles, r.name, r.source) for r in rows]
    if len(rows) <= SYNC_LIMIT:
        return _finish_copy(win, kind, _copy_job(kind, payload))
    win.statusBar().showMessage(f"Copying {len(rows):,} molecules as {label}...")

    def failed(msg):
        win.statusBar().clearMessage()
        QMessageBox.warning(win, f"Copy as {label}", msg)

    workers.start(workers.FunctionWorker(_copy_job, kind, payload), win,
                  on_done=lambda value: _finish_copy(win, kind, value), on_failed=failed)
    return True


def _cell(value) -> str:
    from .molecule_list import _field_text

    return " ".join(_field_text(value).split()) if isinstance(value, str) else _field_text(value)


def molecules_table(win, rows) -> str:
    """``rows`` as a tab-separated table with a header: ID, SMILES (as entered), Name, then the posted data-field
    columns the list shows (hidden ones left out). Tabs and line breaks inside a cell become spaces."""
    import csv
    import io

    from .molecule_list import COLUMNS

    taken = {c.lower() for c in MOLECULES_COLUMNS}
    fields = [name for k, name in enumerate(win.model.extra_columns())
              if not win.mol_list.isColumnHidden(len(COLUMNS) + k) and name.strip().lower() not in taken]
    out = io.StringIO()
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    w.writerow(MOLECULES_COLUMNS + tuple(fields))
    for r in rows:
        smiles = r.smiles or (getattr(r.input, "smiles", "") if r.input is not None else "")
        w.writerow([_cell(r.id), _cell(smiles), _cell(r.name)] + [_cell((r.fields or {}).get(f)) for f in fields])
    return out.getvalue()


def copy_molecules(win) -> bool:
    """Edit > Copy molecules (Ctrl+Shift+C in the Molecules dock): the selected rows as a table (``molecules_table``),
    as text for spreadsheets and as ``MOLECULES_MIME`` for a paste back into QuantaBench."""
    rows = selected_rows(win)
    if not rows:
        win.statusBar().showMessage("Select a molecule to copy", 4000)
        return False
    text = molecules_table(win, rows)
    mime = QMimeData()
    mime.setText(text)
    mime.setData(MOLECULES_MIME, text.encode("utf-8"))
    _clipboard().setMimeData(mime)
    win.statusBar().showMessage(f"Copied {len(rows):,} molecule(s) as a table (ID, SMILES, name, fields)", 5000)
    return True


# ================================================================== install


def _edit_menu(win):
    bar = win.menuBar()
    actions = bar.actions()
    for a in actions:
        if a.menu() is not None and a.text().replace("&", "") == "Edit":
            return a.menu()
    menu = QMenu("&Edit", win)
    after_file = next((i + 1 for i, a in enumerate(actions)
                       if a.menu() is not None and a.text().replace("&", "") == "File"), None)
    if after_file is not None and after_file < len(actions):
        bar.insertMenu(actions[after_file], menu)
    else:
        bar.addMenu(menu)
    return menu


def _update_copy_actions(win) -> None:
    on = bool(win.mol_list.selected_ids() or win.mol_list.current_id())
    win.copy_as_menu.setEnabled(on)
    win.a_copy_molecules.setEnabled(on)


def _list_menu(win, menu: QMenu) -> None:
    """The molecule list's right-click menu: Copy molecules, Copy as, Paste."""
    _update_copy_actions(win)
    menu.addAction(win.a_copy_molecules)
    menu.addMenu(win.copy_as_menu)
    menu.addAction(win.a_paste)


def _named_action(win, name: str, slot, text: str = "", tip: str = "", widget=None) -> QAction:
    """A QAction named ``name`` (its objectName) with its keys from the window's keymap (``gui.keymap``: the defaults
    are in ``keymap.SPECS``, Settings > Controls rebinds them), also in ``win.actions_by_name``. ``widget``: where a
    dock-local action lives (Copy molecules: the Molecules dock)."""
    a = QAction(text or ACTIONS[name][0], win)
    keymap.bind(win, a, name, tip or ACTIONS.get(name, ("", ""))[1], widget)
    a.triggered.connect(lambda _checked=False: slot())
    if not hasattr(win, "actions_by_name"):
        win.actions_by_name = {}
    win.actions_by_name[name] = a
    return a


def install(win) -> None:
    menu = _edit_menu(win)
    a_paste = _named_action(win, "edit.paste", lambda: edit_paste(win))
    menu.addAction(a_paste)
    menu.addSeparator()
    # Ctrl+Shift+C in the Molecules dock only (the keymap's "molecules" scope): the Analyze table's own Ctrl+Shift+C
    # (copy with column titles) stays
    a_copy = _named_action(win, "edit.copy_molecules", lambda: copy_molecules(win), widget=win.mol_dock)
    menu.addAction(a_copy)
    copy_menu = QMenu("Copy as", win)
    copy_menu.setObjectName("edit.copy_as")
    copy_menu.setToolTipsVisible(True)
    copy_menu.setToolTip("Put the selected molecules on the clipboard")
    win.a_copy_as = {}
    for kind, label in COPY_LABELS.items():
        a = _named_action(win, f"edit.copy_as.{kind}", lambda k=kind: copy_as(win, k), label, COPY_TIPS[kind])
        copy_menu.addAction(a)
        win.a_copy_as[kind] = a
    menu.addMenu(copy_menu)
    menu.aboutToShow.connect(lambda: _update_copy_actions(win))
    win.copy_as_menu = copy_menu
    win.a_paste = a_paste
    win.a_copy_molecules = a_copy
    win.mol_list.menu_hooks.append(lambda m: _list_menu(win, m))
    win.setAcceptDrops(True)
    filt = _EventFilter(win)
    win.installEventFilter(filt)
    viewport = win.mol_list.viewport()          # drops on the list itself (a QTableView takes no drops by default)
    viewport.setAcceptDrops(True)
    viewport.installEventFilter(filt)
    filt.targets.add(viewport)
    entry = getattr(win, "entry", None)
    if entry is not None:
        entry.installEventFilter(filt)
    win._clipboard_filter = filt
