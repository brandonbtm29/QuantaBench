"""gui.clipboard: Edit > Paste from each clipboard source (fake QMimeData), Copy as round trips, drag and drop."""

import pytest
from conftest import HAS_DISPLAY, PFOA
from rdkit import Chem
from rdkit.Chem import rdDepictor
from test_interchange import GENX, _cdx_of

pytest.importorskip("PySide6")
needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")

WIN_CDX = 'application/x-qt-windows-mime;value="ChemDraw Interchange Format"'
WIN_MDLCT = 'application/x-qt-windows-mime;value="MDLCT"'


def canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _block(smiles: str, name: str = "") -> str:
    m = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(m)
    if name:
        m.SetProp("_Name", name)
    return Chem.MolToMolBlock(m)


def _mime(**formats):
    from PySide6.QtCore import QMimeData

    m = QMimeData()
    text = formats.pop("text", None)
    for fmt, data in formats.items():
        m.setData(fmt, data)
    if text is not None:
        m.setText(text)
    return m


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    import time

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


def _skip_without_system_clipboard(app):
    """Some sessions have no usable system clipboard (a Windows process started outside the desktop's window
    station reads back nothing); the QMimeData tests above cover the logic, these need the real one."""
    from PySide6.QtGui import QGuiApplication

    clip = QGuiApplication.clipboard()
    clip.setText("qb-clipboard-probe")
    _wait(app, lambda: clip.text() == "qb-clipboard-probe", timeout=2.0)
    if clip.text() != "qb-clipboard-probe":
        pytest.skip("the system clipboard does not keep what this process puts on it")


def _window(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    return MainWindow(sm.Settings(output_dir=str(tmp_path)))


def _smiles_in(w) -> list[str]:
    return [canon(w.model.row(i).smiles) for i in w.model.ids()]


def _log(w) -> str:
    return w.log_view.toPlainText()


# ------------------------------------------------------------------ reading mime data (no window)


def test_format_names_and_order():
    from quantabench.gui import clipboard as cb

    assert cb.format_name(WIN_CDX) == "chemdraw interchange format"
    assert cb._kind(WIN_CDX) == "cdx"
    assert cb._kind("com.cambridgesoft.chemdraw.cdx") == "cdx"
    assert cb._kind("COM.CambridgeSoft.ChemDraw.CDXML") == "cdxml"        # case does not matter
    assert cb._kind(WIN_MDLCT) == cb._kind('application/x-qt-windows-mime;value="mdlct"') == "mdlct"
    assert cb._kind("chemical/x-mdl-molfile") == "mol" and cb._kind("text/html") is None
    mime = _mime(**{"chemical/x-mdl-molfile": _block("CCO").encode(), WIN_CDX: _cdx_of(PFOA),
                    "chemical/x-cdxml": b"<CDXML/>", "text": "CCO"})
    assert [k for _, k, _ in cb.candidates(mime)] == ["cdx", "cdxml", "mol", "text"]


@pytest.mark.parametrize("source", ["cdx", "cdx-mac", "cdxml", "mdlct", "molfile", "sdfile", "text-mol",
                                    "text-smiles"])
def test_each_clipboard_source(source):
    from quantabench.chem import interchange
    from quantabench.gui import clipboard as cb

    block = _block(GENX, "GenX")
    mime = {
        "cdx": lambda: _mime(**{WIN_CDX: _cdx_of(GENX)}),
        "cdx-mac": lambda: _mime(**{"com.cambridgesoft.chemdraw.cdx": _cdx_of(GENX)}),
        "cdxml": lambda: _mime(**{"chemical/x-cdxml": interchange.to_cdxml(Chem.MolFromSmiles(GENX)).encode()}),
        "mdlct": lambda: _mime(**{WIN_MDLCT: interchange.mdlct_encode(block)}),
        "molfile": lambda: _mime(**{"chemical/x-mdl-molfile": block.encode()}),
        "sdfile": lambda: _mime(**{"chemical/x-mdl-sdfile": (block + "$$$$\n").encode()}),
        "text-mol": lambda: _mime(text=block),
        "text-smiles": lambda: _mime(text=GENX + " GenX"),
    }[source]()
    label, items, _ = cb.read_candidates(cb.candidates(mime))
    assert [canon(i.smiles) for i in items] == [canon(GENX)]
    assert label.split(" (")[0] == cb.format_name(mime.formats()[0])


def test_a_format_that_fails_falls_back_to_the_next():
    from quantabench.gui import clipboard as cb

    broken = _cdx_of(PFOA)[:60]
    label, items, _ = cb.read_candidates(cb.candidates(_mime(**{WIN_CDX: broken}, text=PFOA)))
    assert label.startswith("text/plain") and [canon(i.smiles) for i in items] == [canon(PFOA)]
    with pytest.raises(ValueError, match="chemdraw interchange format"):
        cb.read_candidates(cb.candidates(_mime(**{WIN_CDX: broken})))


def test_copy_mime_formats():
    from quantabench.chem import interchange
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui import clipboard as cb

    one, two = from_smiles(GENX, id="GenX"), from_smiles(PFOA, id="PFOA")
    mime, problems = cb.copy_mime("smiles", [one])
    assert not problems and canon(mime.text()) == canon(GENX)
    mime, _ = cb.copy_mime("smiles", [one, two])
    assert mime.text().splitlines()[1].endswith(" PFOA")
    mime, _ = cb.copy_mime("mol", [one], platform="win32")
    assert "M  END" in mime.text() and bytes(mime.data("chemical/x-mdl-molfile").data()).decode() == mime.text()
    assert interchange.mdlct_decode(bytes(mime.data(WIN_MDLCT).data())).splitlines() == mime.text().splitlines()
    mime, _ = cb.copy_mime("mol", [one], platform="linux")
    assert WIN_MDLCT not in mime.formats()
    mime, _ = cb.copy_mime("mol", [one, two])
    assert mime.text().count("$$$$") == 2 and "chemical/x-mdl-sdfile" in mime.formats()
    mime, _ = cb.copy_mime("cdxml", [one])
    assert "<CDXML" in mime.text() and "chemical/x-cdxml" in mime.formats()
    bad = from_smiles("CCO", id="bad")
    bad.smiles = "C1CC("
    mime, problems = cb.copy_mime("smiles", [bad, two])
    assert len(problems) == 1 and problems[0].startswith("bad:") and canon(mime.text()) == canon(PFOA)


# ------------------------------------------------------------------ the window


@needs_display
def test_edit_menu_paste_and_copy_round_trips(tmp_path):
    from PySide6.QtGui import QGuiApplication

    app = _app()
    _skip_without_system_clipboard(app)
    w = _window(tmp_path)
    titles = [a.text().replace("&", "") for a in w.menuBar().actions()]
    assert titles[:3] == ["File", "Edit", "Build"]
    assert w.a_paste.shortcut().toString() == "Ctrl+V"
    clip = QGuiApplication.clipboard()

    clip.setMimeData(_mime(**{WIN_CDX: _cdx_of(GENX)}))     # what ChemDraw puts there on Windows
    w.a_paste.trigger()
    assert _wait(app, lambda: w.model.rowCount() == 1)
    assert _smiles_in(w) == [canon(GENX)]
    assert "chemdraw interchange format (CDX)" in _log(w)

    w.mol_list.select_id(w.model.ids()[0])
    for kind in ("mol", "cdxml", "smiles"):
        n = w.model.rowCount()
        w.a_copy_as[kind].trigger()
        assert clip.text()
        w.a_paste.trigger()
        assert _wait(app, lambda n=n: w.model.rowCount() == n + 1)
        assert _smiles_in(w)[-1] == canon(GENX), kind
        w.mol_list.select_id(w.model.ids()[0])
    _close(app, w)


@needs_display
def test_paste_into_a_text_field_and_the_molecules_box(tmp_path, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QGuiApplication, QKeyEvent
    from PySide6.QtWidgets import QApplication

    from quantabench.gui import clipboard as cb

    app = _app()
    _skip_without_system_clipboard(app)
    w = _window(tmp_path)
    clip = QGuiApplication.clipboard()

    # Edit > Paste with the focus in a text field pastes the text there
    clip.setText("CCO")
    with pytest.MonkeyPatch.context() as mp:     # not monkeypatch.undo(): that would also undo isolated_settings
        mp.setattr(cb.QApplication, "focusWidget", staticmethod(lambda: w.search))
        w.a_paste.trigger()
        assert w.search.text() == "CCO" and w.model.rowCount() == 0

    def ctrl_v(kind):
        ev = QKeyEvent(kind, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier, "v")
        QApplication.sendEvent(w.entry, ev)
        return ev

    # Ctrl+V in the Molecules box: one line of text goes into the box as usual
    clip.setText(PFOA)
    ctrl_v(QEvent.Type.KeyPress)
    assert w.entry.text() == PFOA and w.model.rowCount() == 0
    w.entry.clear()
    # ... a MOL block (several lines) or a ChemDraw format becomes molecules instead
    clip.setText(_block(GENX, "GenX"))
    assert ctrl_v(QEvent.Type.ShortcutOverride).isAccepted()
    ctrl_v(QEvent.Type.KeyPress)
    assert _wait(app, lambda: w.model.rowCount() == 1)
    assert w.entry.text() == "" and w.model.ids() == ["GenX"]
    _close(app, w)


@needs_display
def test_paste_with_nothing_readable_says_so(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from quantabench.gui import clipboard as cb

    app = _app()
    w = _window(tmp_path)
    assert not cb.paste(w, _mime(**{"image/png": b"\x89PNG"}))
    assert "no structure or text found" in _log(w)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2])))
    assert cb.paste(w, _mime(text="C1CC("))
    assert _wait(app, lambda: bool(warned))
    assert "No molecule could be read" in warned[0] and w.model.rowCount() == 0
    _close(app, w)


@needs_display
def test_drop_files_folders_and_text(tmp_path, monkeypatch):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import QApplication

    from quantabench.gui import import_action

    app = _app()
    w = _window(tmp_path)
    assert w.acceptDrops()
    opened, imported, folders = [], [], []
    monkeypatch.setattr(w, "open_files", lambda paths: opened.append(paths))
    monkeypatch.setattr(import_action, "import_files", lambda win, paths: imported.append(paths))
    monkeypatch.setattr(w, "open_run_folder", lambda _checked=False, folder=None: folders.append(folder))
    smi, out, run = tmp_path / "list.smi", tmp_path / "water.out", tmp_path / "run1"
    smi.write_text("CCO ethanol\n")
    out.write_text("not really ORCA\n")
    run.mkdir()

    def drop(mime):
        enter = QDragEnterEvent(QPoint(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                                Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(w, enter)
        assert enter.isAccepted()
        QApplication.sendEvent(w, QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime,
                                             Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))

    files = QMimeData()
    files.setUrls([QUrl.fromLocalFile(str(p)) for p in (smi, out, run)])
    drop(files)
    assert opened == [[str(smi)]] and imported == [[str(out)]] and folders == [str(run)]

    drop(_mime(text=_block(PFOA, "PFOA")))                    # text dragged from an editor
    assert _wait(app, lambda: w.model.rowCount() == 1)
    assert w.model.ids() == ["PFOA"] and "Drop: 1 molecule(s)" in _log(w)

    png = _mime(**{"image/png": b"x"})                      # the event does not keep its mime data alive
    ignored = QDragEnterEvent(QPoint(20, 20), Qt.DropAction.CopyAction, png,
                              Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(w, ignored)
    assert not ignored.isAccepted()
    _close(app, w)
