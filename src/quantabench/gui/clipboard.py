"""Edit > Paste and Copy as SMILES / MOL / CDXML, and drag and drop onto the main window.

CONTRACT:

    install(win: MainWindow) -> None
        Adds an Edit menu (after File) with Paste (Ctrl+V) and Copy as SMILES / MOL / CDXML, accepts drops
        on the window, and lets Ctrl+V in the Molecules box paste a structure (see below).

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
- Text: MOL/SD blocks, CDXML and MRV text, or lines of SMILES, names and CAS numbers
  (``chem.interchange.parse_structure_text``; names go to PubChem).

Parsing runs on a worker thread (``gui.workers.FunctionWorker``), as the Molecules box does, and the molecules
are added with ``win.add_inputs`` like typed ones. Paste with the focus in a text field pastes into the field;
Ctrl+V in the Molecules box adds molecules only when the clipboard holds a structure format or several lines
of text (a single line goes into the box as usual).

Copy as puts the selected molecules' input structures (as entered, before standardizing) on the clipboard:
SMILES (one per line, with the id after it when there are several), MOL (2D, an SD file for several; also
``chemical/x-mdl-molfile`` and on Windows "MDLCT") or CDXML (also ``chemical/x-cdxml``). The text is
always there, so ChemDraw's Edit > Paste Special (SMILES, MOL Text, CDXML Text) reads it.

Drops: files go through ``win.open_files`` (structure files and tables, the File > Open path), calculation
outputs and Spartan documents through File > Import calculation output, a folder through Open run folder;
anything else dropped (text, ChemDraw drags) is read like a paste.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QMimeData, QObject, Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QMessageBox, QPlainTextEdit, QTextEdit

_log = logging.getLogger(__name__)

WINDOWS_MIME = 'application/x-qt-windows-mime;value="{}"'
CDX_NAMES = ("chemdraw interchange format", "com.cambridgesoft.chemdraw.cdx", "chemical/x-cdx")
CDXML_NAMES = ("chemical/x-cdxml", "com.cambridgesoft.chemdraw.cdxml", "cdxml")
MOL_NAMES = ("chemical/x-mdl-molfile", "chemical/x-mdl-sdfile", "mdl molfile")
MDLCT_NAMES = ("mdlct",)
#: calculation outputs that File > Import calculation output reads (when File > Open does not)
OUTPUT_TYPES = {".out", ".log", ".dat", ".output", ".lis", ".spartan"}
TITLE = "Paste"


def format_name(fmt: str) -> str:
    """The clipboard format's own name, lowercase: Qt's Windows wrapper removed."""
    m = re.match(r'application/x-qt-windows-mime;value="(.*)"$', fmt.strip(), re.I)
    return (m.group(1) if m else fmt).strip().lower()


def _kind(fmt: str) -> str | None:
    name = format_name(fmt)
    if name in CDX_NAMES or "chemdraw interchange" in name:
        return "cdx"
    if name in CDXML_NAMES:
        return "cdxml"
    if name in MDLCT_NAMES:
        return "mdlct"
    if name in MOL_NAMES:
        return "mol"
    return None


_ORDER = ("cdx", "cdxml", "mdlct", "mol")


def candidates(mime: QMimeData) -> list[tuple[str, str, object]]:
    """(format as named, kind, data) of the structures in ``mime``, in the order they are tried: ChemDraw CDX,
    CDXML, MDLCT, MOL, then text. Plain Python data (bytes, str), so a worker thread can parse them."""
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
    return [u.toLocalFile() for u in mime.urls() if u.isLocalFile() and u.toLocalFile()] if mime.hasUrls() else []


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


def read_candidates(cands: list[tuple[str, str, object]]):
    """Worker thread: the molecules of the first candidate that gives any.
    Returns (format used, MoleculeInputs, messages); raises ValueError with every format's reason when none does."""
    from ..chem import interchange

    errors = []
    for fmt, kind, data in cands:
        label = f"{format_name(fmt)} ({kind.upper() if kind != 'text' else 'text'})"
        messages: list[str] = []
        try:
            if kind == "cdx":
                items = interchange.parse_cdx_bytes(data, source="clipboard")
            elif kind == "mdlct":
                items = interchange.parse_structure_text(interchange.mdlct_decode(data), "clipboard", messages.append)
            else:
                items = interchange.parse_structure_text(_decode(data), "clipboard", messages.append)
        except Exception as exc:  # noqa: BLE001 - try the next format; RDKit raises RuntimeError too
            errors.append(f"{label}: {exc}")
            continue
        if items:
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


def _report_nothing(win, mime: QMimeData, what: str) -> None:
    fmts = ", ".join(mime.formats()) or "none"
    win.log(f"{what}: no structure or text found (formats: {fmts})")
    win.statusBar().showMessage(f"{what}: nothing QuantaBench can read", 5000)


def add_from(win, cands, what: str = TITLE) -> None:
    """Parse ``cands`` on a worker thread and add the molecules to the list."""
    from . import workers

    win.statusBar().showMessage(f"{what}: reading structures...")

    def done(value):
        label, items, messages = value
        win.statusBar().clearMessage()
        for m in messages:
            win.log(m)
        _log.info("%s: %d molecule(s) from %s", what, len(items), label)
        win.log(f"{what}: {len(items)} molecule(s) from {label}")
        win.add_inputs(items)

    def failed(msg):
        win.statusBar().clearMessage()
        win.log(f"{what}: {msg}")
        QMessageBox.warning(win, what, msg)

    workers.start(workers.FunctionWorker(read_candidates, list(cands)), win, on_done=done, on_failed=failed)


def open_paths(win, paths: list[str]) -> None:
    """Dropped (or pasted) files and folders: each goes to the reader File would use for it."""
    from ..chem import inputs

    folders = [p for p in paths if Path(p).is_dir()]
    files = [p for p in paths if Path(p).is_file()]
    outputs = [p for p in files if Path(p).suffix.lower() not in inputs.FILE_TYPES
               and Path(p).suffix.lower() in OUTPUT_TYPES]
    structures = [p for p in files if p not in outputs]
    if structures:
        win.open_files(structures)
    if outputs and getattr(win, "a_import_output", None) is not None:
        from . import import_action

        import_action.import_files(win, outputs)
    if folders:
        win.open_run_folder(folder=folders[0])
        if len(folders) > 1:
            win.log(f"Opened {folders[0]}; only one run folder can be open at a time")


def paste(win, mime: QMimeData | None = None, what: str = TITLE) -> bool:
    """Add the molecules on the clipboard (or in ``mime``). False when there was nothing to read."""
    from_clipboard = mime is None
    if mime is None:
        mime = QGuiApplication.clipboard().mimeData()
    if mime is None:
        return False
    files = local_files(mime)
    if files:
        open_paths(win, files)
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
                                   and structure_on_clipboard(QGuiApplication.clipboard().mimeData())):
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
    """Drops on the window; Ctrl+V in the Molecules box when the clipboard holds a structure."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win

    def eventFilter(self, obj, event) -> bool:
        t = event.type()
        if obj is self.win and t in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if readable(event.mimeData()):
                event.acceptProposedAction()
                return True
            return False
        if obj is self.win and t == QEvent.Type.Drop:
            mime = event.mimeData()
            if mime is None:
                return False
            event.acceptProposedAction()
            files = local_files(mime)
            if files:
                open_paths(self.win, files)
            else:
                cands = candidates(mime)
                if cands:
                    add_from(self.win, cands, "Drop")
            return True
        entry = getattr(self.win, "entry", None)
        if obj is entry and t in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            if event.matches(QKeySequence.StandardKey.Paste) and \
                    structure_on_clipboard(QGuiApplication.clipboard().mimeData()):
                if t == QEvent.Type.KeyPress:
                    paste(self.win)
                event.accept()
                return True
        return False


# ================================================================== copy


def selected_inputs(win) -> list:
    ids = win.mol_list.selected_ids()
    if not ids and win.mol_list.current_id():
        ids = [win.mol_list.current_id()]
    out = []
    for ident in ids:
        row = win.model.row(ident)
        if row is not None:
            out.append(win._input_of(row))
    return out


def copy_mime(kind: str, items: list, platform: str = sys.platform) -> tuple[QMimeData, list[str]]:
    """(clipboard data, problems) for ``items`` (MoleculeInputs, molecules or results) as ``kind``: "smiles",
    "mol" or "cdxml". Molecules that cannot be written are left out and named in the problems."""
    from ..chem import interchange

    good, problems = [], []
    for it in items:
        try:
            interchange.to_smiles(it)
            good.append(it)
        except Exception as exc:  # noqa: BLE001 - one bad structure must not stop the others
            problems.append(f"{getattr(it, 'id', '?')}: {exc}")
    mime = QMimeData()
    if not good:
        return mime, problems
    if kind == "smiles":
        if len(good) == 1:
            text = interchange.to_smiles(good[0])
        else:
            text = "".join(f"{interchange.to_smiles(it)} {getattr(it, 'id', '')}".rstrip() + "\n" for it in good)
        mime.setText(text)
    elif kind == "mol":
        text = interchange.to_molblock(good[0]) if len(good) == 1 else interchange.to_sdf(good)
        mime.setText(text)
        mime.setData("chemical/x-mdl-molfile" if len(good) == 1 else "chemical/x-mdl-sdfile", text.encode())
        if platform == "win32" and len(good) == 1:
            mime.setData(WINDOWS_MIME.format("MDLCT"), interchange.mdlct_encode(text))
    elif kind == "cdxml":
        text = interchange.to_cdxml(good)
        mime.setText(text)
        mime.setData("chemical/x-cdxml", text.encode("utf-8"))
    else:
        raise ValueError(f"Unknown format {kind!r}")
    return mime, problems


COPY_LABELS = {"smiles": "SMILES", "mol": "MOL", "cdxml": "CDXML"}


def copy_as(win, kind: str) -> bool:
    items = selected_inputs(win)
    if not items:
        win.statusBar().showMessage("Select a molecule to copy", 4000)
        return False
    mime, problems = copy_mime(kind, items)
    for p in problems:
        win.log(f"Copy as {COPY_LABELS[kind]}: {p}")
    if not mime.formats():
        QMessageBox.warning(win, f"Copy as {COPY_LABELS[kind]}", "\n".join(problems) or "Nothing to copy")
        return False
    QGuiApplication.clipboard().setMimeData(mime)
    n = len(items) - len(problems)
    win.statusBar().showMessage(f"Copied {n} molecule(s) as {COPY_LABELS[kind]}", 4000)
    return True


# ================================================================== install


def _edit_menu(win):
    bar = win.menuBar()
    actions = bar.actions()
    for a in actions:
        if a.menu() is not None and a.text().replace("&", "") == "Edit":
            return a.menu()
    from PySide6.QtWidgets import QMenu

    menu = QMenu("&Edit", win)
    after_file = next((i + 1 for i, a in enumerate(actions)
                       if a.menu() is not None and a.text().replace("&", "") == "File"), None)
    if after_file is not None and after_file < len(actions):
        bar.insertMenu(actions[after_file], menu)
    else:
        bar.addMenu(menu)
    return menu


def install(win) -> None:
    menu = _edit_menu(win)
    a_paste = QAction("Paste", win)
    a_paste.setShortcut(QKeySequence(QKeySequence.StandardKey.Paste))
    a_paste.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
    a_paste.setToolTip("Add the structures on the clipboard to the Molecules list: ChemDraw, MOL or SD, "
                       "CDXML, Marvin, SMILES, names or CAS numbers (in a text field: paste the text)")
    a_paste.triggered.connect(lambda _checked=False: edit_paste(win))
    menu.addAction(a_paste)
    menu.addSeparator()
    win.a_copy_as = {}
    for kind, label in COPY_LABELS.items():
        a = QAction(f"Copy as {label}", win)
        a.setToolTip(f"Put the selected molecules on the clipboard as {label} (the structure as entered)")
        a.triggered.connect(lambda _checked=False, k=kind: copy_as(win, k))
        menu.addAction(a)
        win.a_copy_as[kind] = a
    win.a_paste = a_paste
    win.setAcceptDrops(True)
    filt = _EventFilter(win)
    win.installEventFilter(filt)
    entry = getattr(win, "entry", None)
    if entry is not None:
        entry.installEventFilter(filt)
    win._clipboard_filter = filt
